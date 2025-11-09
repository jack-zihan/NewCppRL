import os
import time
import uuid
import numpy as np
from datetime import datetime
import gymnasium as gym

import sys
import torch
import warnings
import wandb

from torchrl._utils import compile_with_warmup, logger as torchrl_logger
from tensordict.nn import CudaGraphModule
from tensordict import TensorDict
from tqdm import tqdm

from torchrl_utils_new.local_video_recorder import LocalVideoRecorder
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record import VideoRecorder
from rl_new.sac_cont_sy.env_utils import make_single_environment, make_drop_pixels_eval_environment
from functools import partial
from pathlib import Path
from torchrl.record.loggers import CSVLogger
import shutil

from torchrl_utils.model.resnet_fpn_dual import HIFReconstructionLoss

# ================= Layer 1: Unified Loss & Modes =================
from enum import Enum
from typing import Optional


class LossMode(Enum):
    """Training loss composition mode.

    PRETRAIN: only HIF reconstruction (actor forward only)
    JOINT:    SAC (actor+q+alpha) + HIF with a weight
    SAC_ONLY: SAC only (used when HIF is disabled)
    """

    PRETRAIN = "pretrain"
    JOINT = "joint"
    SAC_ONLY = "sac_only"


class HIFAssistedSACLoss(torch.nn.Module):
    """Unified loss for SAC with optional HIF assistance.

    This module centralizes loss composition per phase to simplify training code:
    - PRETRAIN: run actor only and optimize HIF reconstruction; write td_error for PRB
    - JOINT:    run full SAC + HIF (weighted)
    - SAC_ONLY: run standard SAC only

    It always returns a TensorDict containing a scalar tensor under key "total_loss"
    so the update function can consistently backprop on it.

    Args:
        actor:        the actor module (ProbabilisticActor) used for PRETRAIN forward
        sac_loss:     TorchRL SACLoss module (produces loss_actor/loss_qvalue/loss_alpha)
        hif_loss:     HIF reconstruction loss module (optional, required for PRETRAIN/JOINT)
        mode:         initial LossMode
        hif_weight:   initial HIF weight (only meaningful for JOINT)
    """

    def __init__(self,
                 actor: torch.nn.Module,
                 sac_loss: torch.nn.Module,
                 hif_loss: Optional[torch.nn.Module] = None,
                 mode: LossMode = LossMode.SAC_ONLY,
                 hif_weight: float = 0.0):
        super().__init__()
        self.actor = actor
        self.sac_loss = sac_loss
        self.hif_loss = hif_loss
        self._mode = mode
        self.hif_weight = float(hif_weight)

        if self._mode in (LossMode.PRETRAIN, LossMode.JOINT) and self.hif_loss is None:
            raise ValueError(f"LossMode {self._mode.value} requires a valid hif_loss module")

    @property
    def mode(self) -> LossMode:
        return self._mode

    def set_mode(self, mode: LossMode, hif_weight: Optional[float] = None):
        if mode in (LossMode.PRETRAIN, LossMode.JOINT) and self.hif_loss is None:
            raise ValueError(f"Switching to {mode.value} requires hif_loss")
        old = self._mode
        self._mode = mode
        if hif_weight is not None:
            self.hif_weight = float(hif_weight)
        msg = f"[Loss] mode: {old.value} -> {self._mode.value}"
        if self._mode == LossMode.JOINT:
            msg += f", hif_weight={self.hif_weight:.4f}"
        torchrl_logger.info(msg)

    def forward(self, td: TensorDict) -> TensorDict:
        # PRETRAIN: only run actor and HIF
        if self._mode == LossMode.PRETRAIN:
            # run actor to produce predictions (including pred_ego_hif)
            td = self.actor(td)
            hif_val, hif_metrics = self.hif_loss(td)

            # write td_error so PRB priority update works during pretrain
            if "td_error" in hif_metrics.keys():
                td.set("td_error", hif_metrics["td_error"])  # keep same behavior as HIFPretrainLoss

            out = TensorDict({}, [])
            out["loss_hif"] = hif_val
            out["total_loss"] = hif_val
            # 填充SAC三项loss为0，便于统一日志与均值统计
            zero = hif_val.new_zeros(())
            out["loss_actor"] = zero
            out["loss_qvalue"] = zero
            out["loss_alpha"] = zero
            out.update(hif_metrics)
            return out

        # SAC (and possibly HIF)
        sac_out = self.sac_loss(td)  # contains loss_actor/loss_qvalue/loss_alpha and alpha/entropy, also writes td_error

        loss_actor = sac_out["loss_actor"]
        loss_qvalue = sac_out["loss_qvalue"]
        loss_alpha = sac_out["loss_alpha"]

        out = TensorDict({}, [])
        out["loss_actor"] = loss_actor
        out["loss_qvalue"] = loss_qvalue
        out["loss_alpha"] = loss_alpha
        if "alpha" in sac_out.keys():
            out["alpha"] = sac_out["alpha"]
        if "entropy" in sac_out.keys():
            out["entropy"] = sac_out["entropy"]

        total = loss_actor + loss_qvalue + loss_alpha

        if self._mode == LossMode.JOINT:
            hif_val, hif_metrics = self.hif_loss(td)
            out["loss_hif"] = hif_val
            out.update(hif_metrics)
            total = total + self.hif_weight * hif_val

        out["total_loss"] = total
        return out


def dump_video(module, step):
    """Helper function to dump video from VideoRecorder."""
    if isinstance(module, VideoRecorder):
        module.iter = step  # 用训练迭代 i 作为视频 step
        module.dump()


def log_metrics(logger, metrics, step):
    for metric_name, metric_value in metrics.items():
        logger.log_scalar(metric_name, metric_value, step)

def setup_torch_cache():
    # 设置缓存目录（确保有写权限）
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/root/.cache/torchinductor'  # 主缓存目录
    os.environ['TORCHINDUCTOR_FX_GRAPH_CACHE'] = '1'  # 启用 FX 图缓存
    os.environ['TORCHINDUCTOR_AUTOGRAD_CACHE'] = '1'  # 启用 AOTAutograd 缓存
    os.environ['TRITON_CACHE_DIR'] = '/root/.cache/triton'  # Triton 缓存目录

    # 创建缓存目录
    os.makedirs('/root/.cache/torchinductor', exist_ok=True)
    os.makedirs('/root/.cache/torchinductor/fxgraph', exist_ok=True)
    os.makedirs('/root/.cache/torchinductor/aotautograd', exist_ok=True)
    os.makedirs('/root/.cache/triton', exist_ok=True)

    # 启用缓存
    torch._inductor.config.fx_graph_cache = True  # 这个属性仍然有效
    torch._inductor.config.force_disable_caches = False  # 确保不禁用缓存
    torch._dynamo.config.cache_size_limit = 256  # 缓存条目数限制
    torch._dynamo.config.accumulated_cache_size_limit = 256  # 累积缓存限制


def generate_exp_name(model_name: str, experiment_name: str) -> str:
    """Generates an ID (str) for the described experiment using UUID and current date."""
    exp_name = "_".join(
        (
            model_name,
            experiment_name,
            str(uuid.uuid4())[:8],
            datetime.now().strftime("%y_%m_%d-%H_%M_%S"),
        )
    )
    return exp_name


# ============ 设备配置============
def _resolve_gpu_devices(gpu_config, train_gpu_id):
    """解析GPU设备列表（处理-1自动检测和列表验证）"""
    if gpu_config is None:
        return []

    if gpu_config == -1:  # 自动检测：使用所有GPU（排除训练GPU）
        all_gpus = list(range(torch.cuda.device_count()))
        if train_gpu_id in all_gpus and len(all_gpus) > 1:
            all_gpus.remove(train_gpu_id)

        if not all_gpus:
            torchrl_logger.warning("没有可用的GPU用于收集，将使用CPU")
        elif len(all_gpus) == 1 and all_gpus[0] == train_gpu_id:
            torchrl_logger.warning(f"只有一个GPU，训练和收集将共享GPU {train_gpu_id}")
        return all_gpus

    # 验证GPU列表
    return [gpu_id for gpu_id in gpu_config if gpu_id < torch.cuda.device_count()
            or torchrl_logger.warning(f"GPU {gpu_id} 不存在，跳过") is None]


def _resolve_cpu_workers(cpu_config):
    """解析CPU工作进程数（处理-1自动检测）"""
    if cpu_config is None:
        return 0
    return max(1, os.cpu_count() - 2) if cpu_config == -1 else cpu_config


def setup_devices(cfg):
    """设置训练和收集设备"""
    # 训练设备
    train_device = torch.device(cfg.training.device if cfg.training.device
                                else "cuda:0" if torch.cuda.is_available() else "cpu")
    train_gpu_id = int(str(train_device).split(':')[1]) if 'cuda' in str(train_device) else None

    # 收集设备配置
    collector_devices = []

    # GPU收集器
    gpu_devices = _resolve_gpu_devices(cfg.collector.gpu_devices, train_gpu_id)
    for gpu_id in gpu_devices:
        collector_devices.extend([f'cuda:{gpu_id}'] * cfg.collector.processes_per_gpu)

    # CPU收集器
    cpu_workers = _resolve_cpu_workers(cfg.collector.cpu_workers)
    if cpu_workers > 0:
        collector_devices.extend(['cpu'] * cpu_workers)

    # 默认配置
    if not collector_devices:
        collector_devices = ['cpu'] * cfg.collector.num_envs

    return train_device, collector_devices


# ============ 优化的更新函数============
def create_update_fn(loss_module, optimizer, target_net_updater=None, cfg=None,
                     compile_mode=None, scaler=None):
    """统一的更新函数 - 极简版

    支持任何符合 TensorDict → TensorDict 接口的 loss 模块:
    - SACLoss: 标准SAC
    - HIFAssistedSACLoss: 统一的SAC+HIF（或仅HIF预训练/仅SAC）

    自动组合所有 loss_* 键进行反向传播。
    """

    def update(sampled_tensordict):
        if cfg.training.use_amp and scaler is not None:
            # 混合精度训练
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss_out = loss_module(sampled_tensordict)
                total_loss = loss_out["total_loss"] # Research-grade: explicit contract requires total_loss
            scaler.scale(total_loss.sum()).backward()

            if cfg.optim.max_grad_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg.optim.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        else:
            # 标准训练
            loss_out = loss_module(sampled_tensordict)
            total_loss = loss_out["total_loss"]
            total_loss.sum().backward()

            if cfg.optim.max_grad_norm:torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg.optim.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # 更新目标网络（如果有）
        if target_net_updater is not None:
            target_net_updater.step()

        return loss_out.detach()

    # 编译优化
    if cfg.compile.enable:
        mode = compile_mode if compile_mode is not None else cfg.compile.mode
        warmup = cfg.compile.warmup
        update = compile_with_warmup(update, mode=mode, warmup=warmup)
        torchrl_logger.info(f"启用编译加速，模式: {mode}, warmup: {warmup}")

    # CudaGraph优化
    if cfg.compile.cudagraphs and torch.cuda.is_available():
        try:
            update = CudaGraphModule(update, in_keys=[], out_keys=[], warmup=10)
            torchrl_logger.info("启用CudaGraph优化")
            warnings.warn("CudaGraphModule is experimental and may lead to silently wrong results. Use with caution.",
                          category=UserWarning)
        except Exception as e:
            torchrl_logger.warning(f"CudaGraph初始化失败: {e}")
    return update


# ============ Loss组合类（TorchRL风格）============


def set_optimizer_group_lrs(optimizer, all_groups_lr=None,
                           actor_lr=None, critic_lr=None, alpha_lr=None):
    """设置SAC optimizer学习率（确定性访问，符合研究代码原则）

    直接访问TorchRL CombinedOptimizers的确定结构：[actor, critic, alpha]。
    配置错误时立即crash，暴露问题而非掩盖（fail-fast原则）。

    Args:
        optimizer: TorchRL的CombinedOptimizers
        all_groups_lr: 统一设置所有组的学习率（优先级最高）
        actor_lr: actor optimizer的学习率
        critic_lr: critic optimizer的学习率
        alpha_lr: alpha optimizer的学习率
    """
    if all_groups_lr is not None:
        # 统一设置所有组
        for g in optimizer.param_groups:
            g["lr"] = all_groups_lr
    else:
        # 分别设置（直接索引访问，配置错误时立即crash）
        if actor_lr is not None:
            optimizer.param_groups[0]["lr"] = actor_lr
        if critic_lr is not None:
            optimizer.param_groups[1]["lr"] = critic_lr
        if alpha_lr is not None:
            optimizer.param_groups[2]["lr"] = alpha_lr


# ============ 辅助函数 ============
def flatten(td):
    """将TensorDict展平为一维"""
    return td.reshape(-1)


# =========== 从actor并行提取动作（评估用） ============
def get_actor_actions(actor, obss, device):
    """
    从actor提取多个观测的动作（用于评估）。

    Args:
        actor: SAC actor模型
        obss: 观测列表
        device: 计算设备

    Returns:
        actions: numpy数组的动作列表
    """
    observations = []
    vectors = []

    for obs in obss:
        if isinstance(obs, dict):
            observations.append(obs['observation'])
            vectors.append(obs['vector'])  # 不需要额外包装成列表

    observations = torch.from_numpy(np.stack(observations, axis=0)).float().to(device)
    vectors = torch.tensor(np.array(vectors)).float().to(device)

    # 确保vector是正确的shape
    if vectors.ndim == 1:
        vectors = vectors.unsqueeze(-1)

    # 创建TensorDict并获取确定性动作
    td = TensorDict({"observation": observations, "vector": vectors}, batch_size=observations.shape[0])
    with torch.no_grad():
        td = actor(td)

    # 返回动作的numpy数组
    actions = td["action"].cpu().numpy()
    return actions


# =========== 新版并行评估策略 ============
def evaluate_policy_parallel(actor_critic, cfg, logger, step, position: int = 1):
    """
    并行优化版评估函数 - 利用TorchRL的rollout机制
    
    优势：
    - 复用make_environment，无需创建新函数； 利用RewardSum和StepCounter自动统计； 代码量减少60%+，更优雅简洁；与TorchRL官方实现风格一致
    Args:
        actor_critic: SAC actor-critic模型
        cfg: 完整配置
        train_device: 训练设备
        logger: 日志记录器
        step: 当前训练步数
        position: 进度条显示位置
        
    Returns:
        eval_metrics: 评估指标字典
    """
    torchrl_logger.info(f"开始并行评估 - {cfg.logger.eval_episodes} episodes")

    # 1. 复用make_environment获取eval_env（已配置正确的eval_episodes）
    # 如果要保存视频到本地，就不传logger，避免自动上传
    _, eval_env = make_drop_pixels_eval_environment(cfg, logger, eval_device=torch.device('cpu'))

    # 2. 执行rollout - 使用break_when_all_done确保所有环境完成
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        # 创建进度条（使用传入的position，评估完成后自动清除,+2使得StepCounter有效）
        pbar = tqdm(total=cfg.logger.eval_max_steps+2, desc=f"Eval step={step}", disable=not cfg.logger.show_progress,
                    position=position, leave=False, dynamic_ncols=True)  # 使用传入的position, 评估完成后自动清除, 适应终端宽度

        eval_rollout = eval_env.rollout(max_steps=cfg.logger.eval_max_steps+2, policy=actor_critic[0],  # 使用actor
                                        auto_cast_to_device=True, break_when_all_done=True,  # 确保所有环境完成完整episode
                                        callback=lambda env, td: (pbar.update(1),
                                                                  pbar.set_postfix(done=int(td["done"].sum()),
                                                                                   step=step,
                                                                                   reward=td["episode_reward"].mean(),
                                                                                   completion_ratio=td[
                                                                                       "completion_ratio"].mean())))
        # pbar.close()
    # 3. 视频上传（如果配置了）
    if cfg.logger.eval_video: eval_env.apply(partial(dump_video, step=step))

    # 4. 从"next"字典的最后一帧提取所有数据
    episode_end = (eval_rollout["next", "done"] if eval_rollout["next", "done"].any()
                   else eval_rollout["next", "truncated"])
    episode_rewards = eval_rollout["next", "episode_reward"][episode_end].cpu().numpy()  # RewardSum和StepCounter的输出在"next"中
    episode_lengths = eval_rollout["next", "step_count"][episode_end].cpu().numpy()
    steps_95_to_done = None
    if "steps_95_to_done" in eval_rollout["next"].keys():
        steps_95_to_done = eval_rollout["next", "steps_95_to_done"][episode_end].cpu().numpy()
    overlap_counts = None
    if "overlap_count" in eval_rollout["next"].keys(): # overlap_counts 是在wrapper = wrapper.auto_register_info_dict(info_dict_reader=default_info_dict_reader(keys=['overlap_count']))中计算，返回的索引维度不一致，因此需要-1加一维度
        overlap_counts = eval_rollout["next", "overlap_count"].unsqueeze(-1)[episode_end].cpu().numpy()

    # 5. completion_ratio也在"next"的observation中
    completion_ratios = None
    if "completion_ratio" in eval_rollout["next"].keys():  # 使用[-1]是安全的，因为rollout保存的是done时的数据（reset发生在保存之后）
        completion_ratios = eval_rollout["next", "completion_ratio"][episode_end].cpu().numpy()

    # 6. 关闭环境
    eval_env.close()

    # 7. 计算统计指标
    eval_metrics = {"eval/reward_mean": np.mean(episode_rewards), "eval/reward_min": np.min(episode_rewards),
                    "eval/reward_max": np.max(episode_rewards), "eval/episode_length": np.mean(episode_lengths), }

    # 添加completion_ratio统计（如果有）
    if completion_ratios is not None:
        eval_metrics["eval/completion_ratio"] = np.mean(completion_ratios)
        eval_metrics["eval/completion_ratio_max"] = np.max(completion_ratios)
    if steps_95_to_done is not None:
        eval_metrics["eval/steps_95_to_done_mean"] = float(np.mean(steps_95_to_done))
        # 比例：防御性处理 0 长度
        ratios_95_to_done = steps_95_to_done / np.clip(episode_lengths, 1, None)
        eval_metrics["eval/ratio_95_to_done_mean"] = float(np.mean(ratios_95_to_done))
    if overlap_counts is not None:
        eval_metrics["eval/overlap_count_mean"] = float(np.mean(overlap_counts))

    torchrl_logger.info(f"并行评估完成 - 平均奖励: {eval_metrics['eval/reward_mean']:.2f}")
    return eval_metrics

def log_evaluate_results(results, checkpoint_dir, logger=None):
    """
    处理评估结果列表，包括日志记录、视频上传和模型重命名
    
    Args:
        results: 评估结果列表
        checkpoint_dir: checkpoint保存目录
        logger: 日志记录器（可选）
    """
    for result in results:
        # 记录metrics到wandb
        torchrl_logger.info(f"上传评估指标: collected_frames_step {result['step']}， {result['metrics']}")

        if logger is not None:
            log_data = {**result['metrics'], 'eval_step': result['step']} # 构建所有数据的字典
            if Path(result['video_path']).exists(): # 如果有视频，添加到同一个log中
                log_data['eval/video'] = wandb.Video(result['video_path'], fps=1, format="mp4")
            logger.experiment.log(log_data)
        else:
            torchrl_logger.info(f"视频上传失败，Path(result['video_path']).exists() and logger is not None判断不满足")

        # 重命名模型文件（加入评估结果）
        old_model_path = checkpoint_dir / f"model_s{result['step']:08d}_eval_pending.pt"
        # 从metrics中提取附加指标用于文件名（可选）
        metrics_map = result['metrics'] or {}
        steps_95_to_done_mean = metrics_map.get('eval/steps_95_to_done_mean', None)
        overlap_count_mean = metrics_map.get('eval/overlap_count_mean', None)
        filename_suffix = f"_reward{result['reward_mean']:.3f}_completion{result['completion_rate']:.3f}"
        if steps_95_to_done_mean is not None:
            filename_suffix += f"_steps95{float(steps_95_to_done_mean):.1f}"
        if overlap_count_mean is not None:
            filename_suffix += f"_overlap{float(overlap_count_mean):.0f}"
        new_model_filename = f"model_step{result['step']:08d}{filename_suffix}.pt"

        if old_model_path.exists():
            old_model_path.rename(checkpoint_dir / new_model_filename)
            torchrl_logger.info(f"模型已保存: {new_model_filename}")
        time.sleep(3)  # 确保文件系统稳定


def evaluate_policy_standalone(model_path: str, cfg, step: int, position: int = 1):
    """
    独立评估函数 - 在子进程中运行，使用CSVLogger保存视频到本地
    
    Args:
        model_path: 模型文件路径
        cfg: 完整配置
        step: 当前训练步数
        position: 进度条显示位置
        
    Returns:
        dict: 包含metrics、视频路径和关键指标的字典
    """
    # 从模型路径推断工作目录
    model_path = Path(model_path)
    working_dir = model_path.parent.parent  # checkpoints -> 工作目录

    # 加载模型到评估设备（weights_only=False以兼容TorchRL模型）
    actor_critic = torch.load(model_path, map_location=torch.device(cfg.logger['eval_device']), weights_only=False)

    # 创建CSVLogger - 使用MP4格式
    csv_logger = CSVLogger(exp_name=f"eval_{step}", log_dir=str(working_dir / "eval_videos_temp"),  # 使用推断的工作目录
                           video_format="mp4", video_fps=1)

    # 执行评估
    eval_metrics = evaluate_policy_parallel(actor_critic=actor_critic, cfg=cfg, logger=csv_logger, step=step,
                                            position=position)
    # 提取关键指标
    reward_mean, completion_rate = float(eval_metrics['eval/reward_mean']), float(eval_metrics['eval/completion_ratio'])

    # 生成最终文件名并移动
    video_dir = working_dir / "eval_videos";
    video_dir.mkdir(exist_ok=True)
    video_path = video_dir / f"video_s{step:08d}_reward{reward_mean:.3f}_completion_rate{completion_rate:.3f}.mp4"
    temp_video_path = working_dir / "eval_videos_temp" / f"eval_{step}" / "videos" / "eval" / f"video_{step}.mp4"
    shutil.move(str(temp_video_path), str(video_path))

    # 清理临时目录
    tmp_root = working_dir / "eval_videos_temp" / f"eval_{step}"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)

    torchrl_logger.info(f"评估视频已保存: {video_path}")
    torch.cuda.memory.empty_cache() # 清理显存
    return {'metrics': eval_metrics, 'reward_mean': reward_mean, 'completion_rate': completion_rate, 'step': step,
            'video_path': str(video_path) if video_path else None}


def is_time_to_evaluate(current_frames, collected_frames, cfg):
    prev_frames = collected_frames - current_frames
    crossed = (prev_frames // cfg.logger.eval_interval) < (collected_frames // cfg.logger.eval_interval)
    final = collected_frames >= cfg.collector.total_frames

    return True if ((crossed and collected_frames >= cfg.collector.init_random_frames) or final) else False

# ============ Checkpoint管理 ============
from typing import Dict, List, Optional, Tuple

class CheckpointManager:
    """
    管理Top-N checkpoint保存，基于评估奖励
    """

    def __init__(self, save_dir, max_checkpoints: int = 5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints: List[Tuple[float, Path]] = []  # [(reward, path), ...]

    def save_if_best(self, model, reward: float, step: int) -> bool:
        """
        仅保存Top-N个最佳模型

        Args:
            model: 要保存的模型
            reward: 评估奖励
            step: 当前训练步数

        Returns:
            bool: 是否保存了模型
        """
        filename = f"step_{step:08d}_reward_{reward:.2f}.pt"
        filepath = self.save_dir / filename

        # 如果还没有足够的checkpoints，直接保存
        if len(self.checkpoints) < self.max_checkpoints:
            torch.save(model, filepath)
            self.checkpoints.append((reward, filepath))
            self.checkpoints.sort(key=lambda x: x[0], reverse=True)
            torchrl_logger.info(f"保存checkpoint: {filename}")
            return True

        # 检查是否比最差的checkpoint更好
        worst_reward = self.checkpoints[-1][0]
        if reward > worst_reward:
            # 删除最差的checkpoint
            worst_path = self.checkpoints[-1][1]
            if worst_path.exists():
                worst_path.unlink()
                torchrl_logger.info(f"删除旧checkpoint: {worst_path.name}")

            # 保存新的checkpoint
            torch.save(model, filepath)
            self.checkpoints[-1] = (reward, filepath)
            self.checkpoints.sort(key=lambda x: x[0], reverse=True)
            torchrl_logger.info(f"保存新checkpoint: {filename}")
            return True

        return False

    def get_best_checkpoint(self) -> Optional[Path]:
        """返回最佳checkpoint的路径"""
        if self.checkpoints:
            return self.checkpoints[0][1]
        return None


# =========== 评估策略 ============
def evaluate_policy(actor_critic, cfg, train_device, logger, step):
    """
    统一的评估函数，使用TorchRL环境确保与训练环境完全一致
    可用于同步和异步训练

    Args:
        actor_critic: SAC actor-critic模型
        cfg: 完整配置（包含env子配置）
        train_device: 训练设备
        logger: 日志记录器
        step: 当前训练步数
        eval_cfg: 评估配置

    Returns:
        eval_metrics: 评估指标字典
    """
    eval_cfg = cfg.logger
    eval_episodes = eval_cfg.eval_episodes
    torchrl_logger.info(f"开始评估 - {eval_episodes} episodes")

    eval_envs = []
    seeds = [cfg.seed + i for i in range(eval_episodes)]  # 固定种子确保可重复性

    # 1. 使用make_single_environment创建环境（在CPU上评估以节省GPU内存）
    for seed in seeds:
        env = make_single_environment(cfg, device="cpu", seed=seed, from_pixels=eval_cfg['eval_video'])
        eval_envs.append(env)

    # 2. 设置多episode网格视频录制器（使用memmap避免内存溢出）
    recorder = None
    if eval_cfg.eval_video and logger is not None:
        max_frames = min(4, eval_episodes)  # 最多录制4个环境的视频
        recorder = LocalVideoRecorder(
            device="cpu", max_len=(eval_cfg.eval_max_steps * max_frames) // eval_cfg.eval_video_skip + 2,  # 明确指定CPU设备
            use_memmap=True, make_grid=True, nrow=2, skip=1,
            fps=6)  # 关键：启用内存映射，避免显存/内存溢出, 制作2x2网格视频 # 2列网格, 不跳帧（评估时已经通过eval_video_skip控制）, # 视频帧率

    # 3. 进行各种数据和组件初始化
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        # 初始化环境和收集所有transitions
        tds = []
        all_transitions = []

        for env in eval_envs:
            td = env.reset()
            tds.append(td)

        # 初始化进度条
        max_steps = eval_cfg['eval_max_steps']
        use_progress_bar = eval_cfg.show_progress

        if use_progress_bar:
            pbar = tqdm(range(max_steps), desc="Evaluating", file=sys.stderr, leave=False)
            step_iterator = pbar
        else:
            step_iterator = range(max_steps)

        # 4. rollout进行数据收集
        for t in step_iterator:
            # 批量获取动作
            batch_td = torch.stack(tds).to(train_device)
            with torch.no_grad():
                batch_td = actor_critic[0](batch_td).to("cpu")

            # 环境执行动作
            new_tds = []
            for i, (td, env) in enumerate(zip(batch_td.unbind(0), eval_envs)):
                # 步进环境，获取完整的transition
                transition = env.step(td)
                all_transitions.append(transition)

                # 提取下一状态用于下一轮
                next_td = env.step_mdp(transition)
                new_tds.append(next_td)

            tds = new_tds

            # 检查是否所有环境都已完成
            current_dones = [t["next", "done"].item() if hasattr(t["next", "done"], 'item')
                             else t["next", "done"] for t in all_transitions[-len(eval_envs):]]

            if all(current_dones):
                break

            # 更新进度条信息（每10步更新一次）
            if use_progress_bar and t % 10 == 0:
                completed_count = sum(1 for t in all_transitions
                                      if (t["next", "done"].item() if hasattr(t["next", "done"], 'item') else t[
                    "next", "done"]))
                # 计算当前完成的episodes的平均奖励
                done_rewards = [t["next", "episode_reward"].item() for t in all_transitions
                                if
                                (t["next", "done"].item() if hasattr(t["next", "done"], 'item') else t["next", "done"])]
                current_mean = np.mean(done_rewards) if done_rewards else 0.0

                pbar.set_postfix({'done': completed_count, 'reward': f'{current_mean:.2f}',
                                  'best': f'{max(done_rewards):.2f}' if done_rewards else '0.00'})

            # 录制视频帧（如果启用）
            if recorder and (t + 1) % eval_cfg.eval_video_skip == 0:
                pixels = [tds[i]["pixels"] for i in range(min(4, eval_episodes))]
                stacked = torch.stack(pixels, 0)
                recorder.apply(stacked)

        if use_progress_bar:
            pbar.close()

        # 上传录制的视频
        vid_tensor = None
        if recorder:
            vid_tensor = recorder.dump()
        if vid_tensor is not None and logger is not None:
            logger.log_video('eval/video', vid_tensor, step=step)

    # 关闭所有环境
    for env in eval_envs:
        env.close()

    # 从transitions中提取完成的episodes的统计信息
    done_transitions = [t for t in all_transitions
                        if (t["next", "done"].item() if hasattr(t["next", "done"], 'item') else t["next", "done"])]

    # 提取episode统计信息
    episode_rewards = [t["next", "episode_reward"].item() for t in done_transitions]
    episode_lengths = [t["next", "step_count"].item() for t in done_transitions]

    # 提取completion_ratio（如果存在）
    completion_ratios = []
    for t in done_transitions:
        if "completion_ratio" in t["next"]:
            completion_ratios.append(t["next", "completion_ratio"].item())

    # 计算统计指标
    eval_metrics = {
        "eval/reward_mean": np.mean(episode_rewards) if episode_rewards else 0.0,
        "eval/reward_std": np.std(episode_rewards) if episode_rewards else 0.0,
        "eval/reward_min": np.min(episode_rewards) if episode_rewards else 0.0,
        "eval/reward_max": np.max(episode_rewards) if episode_rewards else 0.0,
        "eval/episode_length": np.mean(episode_lengths) if episode_lengths else 0.0,
        "eval/episodes_completed": len(done_transitions)
    }

    # 添加completion_ratio统计（如果有）
    if completion_ratios:
        eval_metrics["eval/completion_ratio"] = np.mean(completion_ratios)
        eval_metrics["eval/completion_ratio_max"] = np.max(completion_ratios)

    torchrl_logger.info(f"评估完成 - 平均奖励: {eval_metrics['eval/reward_mean']:.2f}")
    return eval_metrics
