import os
import sys
import uuid
import shutil
import warnings
import numpy as np
import torch
import wandb
from datetime import datetime
from pathlib import Path
from functools import partial
from typing import List, Optional, Tuple
from tqdm import tqdm
from tensordict import TensorDict
from torchrl.record.loggers import CSVLogger
from tensordict.nn import CudaGraphModule
from torchrl._utils import compile_with_warmup, logger as torchrl_logger
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl_utils_new.local_video_recorder import LocalVideoRecorder
from torchrl.record import VideoRecorder
from rl_new.sac_cont_sy.env_utils import make_single_environment, make_drop_pixels_eval_environment
from rl_new.sac_cont_sy.model_utils import make_sac_resnet_dual_models, make_sac_models

class HIFAssistedSACLoss(torch.nn.Module):
    """自适应Loss：phase回调驱动，PRETRAIN仅HIF，训练阶段SAC+可选HIF（fail-fast设计）"""

    def __init__(self, actor, sac_loss, hif_loss, cfg,
                 phase_provider=None, weight_provider=None):
        super().__init__()
        self.actor, self.sac, self.hif, self.cfg = actor, sac_loss, hif_loss, cfg
        self.phase_provider, self.weight_provider = phase_provider or (lambda: 0), weight_provider

        # 预计算HIF权重映射（避免forward时查询配置）
        self.hif_weights = {}
        if cfg.hif.enabled:
            for stage in ['S1', 'S2', 'S3']:
                self.hif_weights[stage] = float(getattr(cfg.hif.weights, stage))

    @property
    def current_hif_weight(self) -> float:
        """当前HIF权重（日志用）：provider优先 → PRETRAIN=1.0 → 映射/0.0"""
        phase = self.phase_provider()
        if phase == 0: return 1.0
        if self.weight_provider: return float(self.weight_provider())
        return self.hif_weights.get({1: 'S1', 2: 'S2', 3: 'S3'}.get(phase))

    def forward(self, td: TensorDict) -> TensorDict:
        """统一Loss计算：phase=0时仅HIF，phase>0时SAC+可选HIF"""
        phase = self.phase_provider()

        if phase == 0: # PRETRAIN阶段：HIF预训练损失
            td = self.actor(td)
            hif_val, hif_metrics = self.hif(td)
            td.set("td_error", hif_metrics["td_error"])

            out = TensorDict({"total_loss": hif_val, "loss_hif": hif_val,
                             "loss_actor": torch.zeros_like(hif_val), "loss_qvalue": torch.zeros_like(hif_val),
                             "loss_alpha": torch.zeros_like(hif_val), "alpha": torch.zeros_like(hif_val),
                             "entropy": torch.zeros_like(hif_val)}, [])
            out.update(hif_metrics)
            return out
        else: # 训练阶段：SAC + 可选HIF
            sac_out = self.sac(td)
            total = sac_out["loss_actor"] + sac_out["loss_qvalue"] + sac_out["loss_alpha"]
            out = TensorDict({"loss_actor": sac_out["loss_actor"], "loss_qvalue": sac_out["loss_qvalue"],
                             "loss_alpha": sac_out["loss_alpha"], "alpha": sac_out["alpha"],
                             "entropy": sac_out["entropy"]}, [])

            # HIF辅助损失
            if self.cfg.hif.enabled:
                hif_weight = (float(self.weight_provider()) if self.weight_provider
                             else self.hif_weights.get({1: 'S1', 2: 'S2', 3: 'S3'}.get(phase, 'S3')))
                if hif_weight > 0:
                    hif_value, hif_metrics = self.hif(td)
                    out["loss_hif"], total = hif_value, total + hif_weight * hif_value
                    out.update(hif_metrics)
            out["total_loss"] = total
            return out

def create_grad_accum_update_fn(loss_module, optimizer, target_net_updater=None, cfg=None,
                     compile_mode=None, scaler=None):
    """创建更新函数（支持compile/cudagraph/AMP/梯度累积，适配SACLoss和HIFAssistedSACLoss）"""
    torch._dynamo.config.capture_scalar_outputs = True
    # 梯度累积配置
    grad_accum_steps = cfg.training.get('grad_accum_steps', 1)
    acc_counter = [0]  # 使用列表作为闭包可变变量

    def update(sampled_tensordict):
        step_taken = False

        # AMP训练路径
        if cfg.training.use_amp and scaler:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss_out = loss_module(sampled_tensordict)
            scaler.scale(loss_out["total_loss"].sum() / grad_accum_steps).backward()

            acc_counter[0] += 1
            if acc_counter[0] % grad_accum_steps == 0:
                if cfg.optim.max_grad_norm:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg.optim.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if target_net_updater: target_net_updater.step()
                step_taken = True

        else:  # 标准训练路径
            loss_out = loss_module(sampled_tensordict)
            (loss_out["total_loss"].sum() / grad_accum_steps).backward()

            acc_counter[0] += 1
            if acc_counter[0] % grad_accum_steps == 0:
                if cfg.optim.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg.optim.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if target_net_updater: target_net_updater.step()
                step_taken = True

        return loss_out.detach(), step_taken

    # Compile优化
    if cfg.compile.enable:
        mode = compile_mode or cfg.compile.mode
        update = compile_with_warmup(update, mode=mode, warmup=cfg.compile.warmup)
        torchrl_logger.info(f"[Compile] 模式: {mode}, warmup: {cfg.compile.warmup}")

    # CudaGraph优化
    if cfg.compile.cudagraphs and torch.cuda.is_available():
        try:
            update = CudaGraphModule(update, in_keys=[], out_keys=[], warmup=10)
            torchrl_logger.info("[CudaGraph] 已启用")
            warnings.warn("CudaGraphModule is experimental, use with caution.", UserWarning)
        except Exception as e:
            torchrl_logger.warning(f"[CudaGraph] 初始化失败: {e}")

    return update

def create_update_fn(loss_module, optimizer, target_net_updater=None, cfg=None,
                     compile_mode=None, scaler=None):
    """创建更新函数（支持compile/cudagraph/AMP，适配SACLoss和HIFAssistedSACLoss）"""

    def update(sampled_tensordict):
        # AMP训练路径
        if cfg.training.use_amp and scaler:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss_out = loss_module(sampled_tensordict)
            scaler.scale(loss_out["total_loss"].sum()).backward()
            if cfg.optim.max_grad_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg.optim.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        else:  # 标准训练路径
            loss_out = loss_module(sampled_tensordict)
            loss_out["total_loss"].sum().backward()
            if cfg.optim.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg.optim.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if target_net_updater: target_net_updater.step() # # 更新目标网络
        return loss_out.detach()

    # Compile优化
    if cfg.compile.enable:
        mode = compile_mode or cfg.compile.mode
        update = compile_with_warmup(update, mode=mode, warmup=cfg.compile.warmup)
        torchrl_logger.info(f"[Compile] 模式: {mode}, warmup: {cfg.compile.warmup}")

    # CudaGraph优化
    if cfg.compile.cudagraphs and torch.cuda.is_available():
        try:
            update = CudaGraphModule(update, in_keys=[], out_keys=[], warmup=10)
            torchrl_logger.info("[CudaGraph] 已启用")
            warnings.warn("CudaGraphModule is experimental, use with caution.", UserWarning)
        except Exception as e:
            torchrl_logger.warning(f"[CudaGraph] 初始化失败: {e}")

    return update

# def create_update_fn(loss_module, optimizer, target_net_updater=None, cfg=None,
#                      compile_mode=None, scaler=None):
#     """ 目前该函数有问题，只转换了数据的BF16, 但遇到ResNet的BatchNorm2D的时候会因为模型中的FP32出问题，暂时弃用
#     创建更新函数（支持 compile / CudaGraph / BF16 AMP，适配 SACLoss 和 HIFAssistedSACLoss）"""
#
#     def update(sampled_tensordict):
#         if cfg.training.use_amp: # ====== BF16 AMP 路径 ======
#             with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
#                 loss_out = loss_module(sampled_tensordict)
#         else:
#             loss_out = loss_module(sampled_tensordict) # # 回退到标准 FP32 路径
#
#         loss_out["total_loss"].sum().backward()
#         if cfg.optim.max_grad_norm:
#             torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg.optim.max_grad_norm)
#         optimizer.step()
#         optimizer.zero_grad(set_to_none=True)
#         if target_net_updater: target_net_updater.step()  # 更新目标网络
#         return loss_out.detach()
#
#     # ====== Compile 优化 ======
#     if cfg.compile.enable:
#         mode = compile_mode or cfg.compile.mode
#         update = compile_with_warmup(update, mode=mode, warmup=cfg.compile.warmup)
#         torchrl_logger.info(f"[Compile] 模式: {mode}, warmup: {cfg.compile.warmup}")
#
#     # ====== CudaGraph 优化（如果你之后想保留）======
#     if cfg.compile.cudagraphs and torch.cuda.is_available():
#         try:
#             update = CudaGraphModule(update, in_keys=[], out_keys=[], warmup=10)
#             torchrl_logger.info("[CudaGraph] 已启用")
#             warnings.warn("CudaGraphModule is experimental, use with caution.", UserWarning)
#         except Exception as e:
#             torchrl_logger.warning(f"[CudaGraph] 初始化失败: {e}")
#
#     return update


def set_optimizer_group_lrs(optimizer, all_groups_lr=None,
                           actor_lr=None, critic_lr=None, alpha_lr=None):
    """设置SAC优化器学习率（确定性访问param_groups[0/1/2]，fail-fast）"""
    if all_groups_lr is not None:
        for g in optimizer.param_groups: g["lr"] = all_groups_lr
        torchrl_logger.info(f"[LR] 统一: {all_groups_lr:.2e}")
    else:
        if actor_lr: optimizer.param_groups[0]["lr"] = actor_lr; torchrl_logger.info(f"[LR] Actor: {actor_lr:.2e}")
        if critic_lr: optimizer.param_groups[1]["lr"] = critic_lr; torchrl_logger.info(f"[LR] Critic: {critic_lr:.2e}")
        if alpha_lr: optimizer.param_groups[2]["lr"] = alpha_lr; torchrl_logger.info(f"[LR] Alpha: {alpha_lr:.2e}")


def evaluate_policy(actor_critic, cfg, logger, step, position: int = 1):
    """评估策略性能（确定性策略rollout + 指标计算）"""
    torchrl_logger.info(f"[Eval] 开始 - {cfg.logger.eval_episodes} episodes")

    actor = actor_critic[0] if isinstance(actor_critic, (tuple, list)) else actor_critic
    _, eval_env = make_drop_pixels_eval_environment(cfg, logger, eval_device=torch.device('cpu'))

    # 确定性rollout
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        pbar = tqdm(total=cfg.logger.eval_max_steps + 2, desc=f"Eval step={step}",
                   disable=not cfg.logger.show_progress, position=position, leave=False, dynamic_ncols=True)
        eval_rollout = eval_env.rollout(
            max_steps=cfg.logger.eval_max_steps + 2, policy=actor, auto_cast_to_device=True, break_when_all_done=True,
            callback=lambda env, td: (pbar.update(1), pbar.set_postfix(
                done=int(td["done"].sum()), step=step,
                reward=td["episode_reward"].mean().item(), completion=td["completion_ratio"].mean().item())))
        pbar.close()

    if cfg.logger.eval_video: eval_env.apply(partial(dump_video, step=step)) # 导出视频

    # 提取episode结束时刻
    episode_end = eval_rollout["next", "done"] if eval_rollout["next", "done"].any() else eval_rollout["next", "truncated"]
    episode_rewards = eval_rollout["next", "episode_reward"][episode_end].cpu().numpy()
    episode_lengths = eval_rollout["next", "step_count"][episode_end].cpu().numpy()
    completion_ratios = eval_rollout["next", "completion_ratio"][episode_end].cpu().numpy()

    eval_metrics = {
        "eval/reward_mean": float(np.mean(episode_rewards)), "eval/reward_min": float(np.min(episode_rewards)),
        "eval/reward_max": float(np.max(episode_rewards)), "eval/episode_length": float(np.mean(episode_lengths)),
        "eval/completion_ratio": float(np.mean(completion_ratios)),
        "eval/completion_ratio_max": float(np.max(completion_ratios)),}

    # 可选指标
    if "steps_95_to_done" in eval_rollout["next"].keys():
        steps_95 = eval_rollout["next", "steps_95_to_done"][episode_end].cpu().numpy()
        eval_metrics["eval/steps_95_to_done_mean"] = float(np.mean(steps_95))
        eval_metrics["eval/ratio_95_to_done_mean"] = float(np.mean(steps_95 / np.clip(episode_lengths, 1, None)))
    if "overlap_count" in eval_rollout["next"].keys():
        overlap = eval_rollout["next", "overlap_count"].unsqueeze(-1)[episode_end].cpu().numpy()
        eval_metrics["eval/overlap_count_mean"] = float(np.mean(overlap))

    eval_env.close()
    torchrl_logger.info(f"[Eval] 完成 - 奖励: {eval_metrics['eval/reward_mean']:.2f}, "
                       f"完成率: {eval_metrics['eval/completion_ratio']:.2%}")
    return eval_metrics


def evaluate_policy_standalone(model_path, cfg, step, position=1, phase_name=None):
    """独立进程评估函数（AsyncEvaluator用，CSVLogger保存视频，phase_name用于标记提交时的阶段）"""
    model_path, working_dir = Path(model_path), Path(model_path).parent.parent

    # 加载模型
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if cfg.hif.enabled:
        backbone_type = getattr(cfg.hif, "backbone", "resnet34")
        actor, _ = make_sac_resnet_dual_models(
            env=make_single_environment(cfg, device="cpu"),
            device="cpu",
            hif_decoder_type=cfg.hif.decoder_type,
            backbone_type=backbone_type,
        )
    else:
        actor, _ = make_sac_models(env=make_single_environment(cfg, device="cpu"), device="cpu")
    actor.load_state_dict(checkpoint['actor'])
    actor.eval()

    # 创建csv_logger并执行评估
    csv_logger = (CSVLogger(exp_name=f"eval_{step}", log_dir=str(working_dir / "eval_videos_temp"),
                           video_format="mp4", video_fps=1) if cfg.logger.eval_video else None)
    metrics = evaluate_policy(actor, cfg, logger=csv_logger, step=step, position=position)
    reward_mean, completion_rate = float(metrics['eval/reward_mean']), float(metrics['eval/completion_ratio'])

    # 移动视频文件
    video_path = None
    if csv_logger:
        video_dir = working_dir / "eval_videos"
        video_dir.mkdir(exist_ok=True)
        video_path = video_dir / f"video_s{step:08d}_reward{reward_mean:.3f}_completion{completion_rate:.3f}.mp4"
        temp_video_path = working_dir / "eval_videos_temp" / f"eval_{step}" / "videos" / "eval" / f"video_{step}.mp4"
        shutil.move(str(temp_video_path), str(video_path))
        torchrl_logger.info(f"[Eval] 视频已保存: {video_path.name}")

        # 清理临时目录和显存
        tmp_root = working_dir / "eval_videos_temp" / f"eval_{step}"
        if tmp_root.exists(): shutil.rmtree(tmp_root)
        torch.cuda.memory.empty_cache()  # 清理显存

    return {'step': step, 'phase_name': phase_name, 'metrics': metrics, 'reward_mean': reward_mean,
            'completion_rate': completion_rate, 'video_path': str(video_path) if video_path else None}


def log_evaluate_results(eval_results, checkpoint_dir, logger):
    """记录评估结果 + 上传视频 + 重命名checkpoint"""
    for result in eval_results:
        step, metrics = result['step'], result['metrics']
        reward_mean, completion_rate, video_path = result['reward_mean'], result['completion_rate'], result.get('video_path')

        # 上传metrics和视频到wandb
        torchrl_logger.info(f"[Eval] 上传评估指标: step={step}")
        if logger:
            log_data = {**metrics, 'eval_step': step}
            if video_path and Path(video_path).exists():
                log_data['eval/video'] = wandb.Video(video_path, fps=1, format="mp4")
            logger.experiment.log(log_data)

        # 重命名checkpoint文件
        old_model_path = checkpoint_dir / f"model_step{step:08d}_eval_pending.pt"
        if old_model_path.exists():
            suffix = f"_reward{reward_mean:.3f}_completion{completion_rate:.3f}"
            if (steps95 := metrics.get('eval/steps_95_to_done_mean')) is not None:
                suffix += f"_steps95{float(steps95):.1f}"
            if (overlap := metrics.get('eval/overlap_count_mean')) is not None:
                suffix += f"_overlap{float(overlap):.1f}"
            new_model_path = checkpoint_dir / f"model_step{step:08d}{suffix}.pt"
            old_model_path.rename(new_model_path)
            torchrl_logger.info(f"[Checkpoint] Renamed: {new_model_path.name}")


def is_time_to_evaluate(current_frames, collected_frames, cfg):
    """判断是否需要评估（越界触发 + init_random_frames门槛）"""
    prev_frames = collected_frames - current_frames
    crossed = (prev_frames // cfg.logger.eval_interval) < (collected_frames // cfg.logger.eval_interval)
    final = collected_frames >= cfg.collector.total_frames
    return (crossed and collected_frames >= cfg.collector.init_random_frames) or final


def dump_video(module, step):
    """从VideoRecorder导出视频"""
    if isinstance(module, VideoRecorder):
        module.iter = step  # 用训练迭代 i 作为视频 step
        module.dump()
        torchrl_logger.info(f"[Video] Dumped at step {step}")


def log_metrics(logger, metrics, step):
    """记录训练指标（控制台 + CSV + WandB）"""
    for metric_name, metric_value in metrics.items():
        logger.log_scalar(metric_name, metric_value, step)

def flatten(td):
    """将TensorDict展平为一维"""
    return td.reshape(-1)


def generate_exp_name(model_name: str, experiment_name: str) -> str:
    """Generates an ID (str) for the described experiment using UUID and current date."""
    exp_name = "_".join((model_name,experiment_name,str(uuid.uuid4())[:8],
                         datetime.now().strftime("%y_%m_%d-%H_%M_%S"),))
    return exp_name


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


def setup_devices(cfg):
    """设置训练和收集设备 Returns: (train_device, collector_devices列表)"""
    # 训练设备
    train_device = torch.device(cfg.training.device if cfg.training.device
                               else "cuda:0" if torch.cuda.is_available() else "cpu")
    train_gpu_id = int(str(train_device).split(':')[1]) if 'cuda' in str(train_device) else None

    # 收集设备配置
    collector_devices = []

    # GPU收集器
    if hasattr(cfg.collector, 'gpu_devices') and cfg.collector.gpu_devices:
        gpu_devices = cfg.collector.gpu_devices if isinstance(cfg.collector.gpu_devices, list) else [cfg.collector.gpu_devices]
        available_gpus = [gpu_id for gpu_id in gpu_devices
                         if gpu_id != train_gpu_id and gpu_id < torch.cuda.device_count()]
        for gpu_id in available_gpus:
            collector_devices.extend([f'cuda:{gpu_id}'] * cfg.collector.processes_per_gpu)

    # CPU收集器
    if hasattr(cfg.collector, 'cpu_workers') and cfg.collector.cpu_workers:
        cpu_workers = max(1, os.cpu_count() - 2) if cfg.collector.cpu_workers == -1 else cfg.collector.cpu_workers
        collector_devices.extend(['cpu'] * cpu_workers)

    # 默认配置
    if not collector_devices:
        collector_devices = ['cpu'] * cfg.env.num_collectors

    return train_device, collector_devices

# =========== 评估策略 ============
def evaluate_policy_local_esitimate(actor_critic, cfg, train_device, logger, step):
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
