import os
import uuid
import numpy as np
from datetime import datetime
import gymnasium as gym

import sys
import torch
import warnings
from torchrl._utils import compile_with_warmup, logger as torchrl_logger
from tensordict.nn import CudaGraphModule
from tensordict import TensorDict
from tqdm import tqdm

from torchrl_utils_new.local_video_recorder import LocalVideoRecorder
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record import VideoRecorder
from rl_new.sac_cont_sy.env_utils import make_environment, make_single_environment, make_drop_pixels_eval_environment
from functools import partial
from torchrl.record import VideoRecorder


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
def setup_devices(cfg):
    """设置训练和收集设备"""
    # 训练设备
    if cfg.training.device:
        train_device = torch.device(cfg.training.device)
    else:
        train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 提取训练GPU ID（用于排除）
    train_gpu_id = int(str(train_device).split(':')[1])

    # 收集设备配置
    collector_devices = []
    gpu_devices = cfg.collector.gpu_devices

    if gpu_devices is not None:
        # GPU收集器
        if gpu_devices == -1:  # 使用所有GPU（除了训练GPU）cudagraph
            all_gpus = list(range(torch.cuda.device_count()))
            if train_gpu_id is not None and train_gpu_id in all_gpus and len(all_gpus) > 1:
                all_gpus.remove(train_gpu_id)  # 排除训练GPU（如果有多个GPU）
            gpu_devices = all_gpus
            if not gpu_devices:  # 如果没有可用GPU
                torchrl_logger.warning("没有可用的GPU用于收集，将使用CPU")
                gpu_devices = []
            elif len(gpu_devices) == 1 and gpu_devices[0] == train_gpu_id:
                torchrl_logger.warning(f"只有一个GPU，训练和收集将共享GPU {train_gpu_id}")
        elif isinstance(gpu_devices, list):
            # 验证GPU ID是否有效
            valid_gpus = []
            for gpu_id in gpu_devices:
                if gpu_id < torch.cuda.device_count():
                    valid_gpus.append(gpu_id)
                else:
                    torchrl_logger.warning(f"GPU {gpu_id} 不存在，跳过")
            gpu_devices = valid_gpus

        processes_per_gpu = cfg.collector.processes_per_gpu
        for gpu_id in gpu_devices:
            collector_devices.extend([f'cuda:{gpu_id}'] * processes_per_gpu)

    # CPU收集器
    cpu_workers = cfg.collector.cpu_workers
    if cpu_workers is not None:
        if cpu_workers == -1:  # 最大化CPU使用
            cpu_workers = max(1, os.cpu_count() - 2)
        if cpu_workers > 0:
            collector_devices.extend(['cpu'] * cpu_workers)

    # 如果没有配置任何设备，使用默认配置
    if not collector_devices:
        collector_devices = ['cpu'] * cfg.collector.num_envs

    return train_device, collector_devices


# ============ 优化的更新函数============
def create_update_fn(loss_module, optimizer, target_net_updater, cfg, compile_mode=None, scaler=None):
    """创建优化的更新函数，支持编译和cudagraph"""

    def update(sampled_tensordict):
        optimizer.zero_grad(set_to_none=True)

        if cfg.training.use_amp and scaler is not None: # 混合精度训练 - 使用autocast
            # 计算损失
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss_out = loss_module(sampled_tensordict)
                actor_loss, q_loss, alpha_loss = loss_out["loss_actor"], loss_out["loss_qvalue"], loss_out["loss_alpha"]
                total_loss = (actor_loss + q_loss + alpha_loss).sum()

            # 使用GradScaler进行反向传播
            scaler.scale(total_loss).backward()

            # 梯度裁剪
            if cfg.optim.max_grad_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg.optim.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
        else: # 标准训练
            # 计算损失
            loss_out = loss_module(sampled_tensordict)
            actor_loss, q_loss, alpha_loss = loss_out["loss_actor"], loss_out["loss_qvalue"], loss_out["loss_alpha"]
            (actor_loss + q_loss + alpha_loss).sum().backward() # 反向传播

            if cfg.optim.max_grad_norm: # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg.optim.max_grad_norm)

            optimizer.step()

        # 更新目标网络
        target_net_updater.step()
        return loss_out.detach()

    # 编译优化（使用compile_with_warmup）
    if cfg.compile.enable:
        mode = compile_mode if compile_mode is not None else cfg.compile.mode
        warmup = cfg.compile.warmup
        update = compile_with_warmup(update, mode=mode, warmup=warmup)
        torchrl_logger.info(f"启用编译加速，模式: {mode}, warmup: {warmup}")

    # CudaGraph优化（需要PyTorch 2.0+）
    if cfg.compile.cudagraphs and torch.cuda.is_available():
        try:
            update = CudaGraphModule(update, in_keys=[], out_keys=[], warmup=10)
            torchrl_logger.info("启用CudaGraph优化")
            warnings.warn("CudaGraphModule is experimental and may lead to silently wrong results. Use with caution.",
                          category=UserWarning, )
        except Exception as e:
            torchrl_logger.warning(f"CudaGraph初始化失败: {e}")

    return update


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
def evaluate_policy_parallel(actor_critic, cfg, logger, step):
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
        
    Returns:
        eval_metrics: 评估指标字典
    """
    torchrl_logger.info(f"开始并行评估 - {cfg.logger.eval_episodes} episodes")

    # 1. 复用make_environment获取eval_env（已配置正确的eval_episodes）
    # 如果要保存视频到本地，就不传logger，避免自动上传
    _, eval_env = make_drop_pixels_eval_environment(cfg, logger, eval_device=torch.device('cpu'))

    # 2. 执行rollout - 使用break_when_all_done确保所有环境完成
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        # 创建进度条（disable参数让它在关闭时变成no-op，无需if判断）
        pbar = tqdm(total=cfg.logger.eval_max_steps, desc="Evaluating", disable=not cfg.logger.show_progress)

        eval_rollout = eval_env.rollout(max_steps=cfg.logger.eval_max_steps, policy=actor_critic[0],  # 使用actor
                                        auto_cast_to_device=True, break_when_all_done=True,  # 确保所有环境完成完整episode
                                        callback=lambda env, td: (pbar.update(1),
                                                                  pbar.set_postfix(done=int(td["done"].sum()),
                                                                                   step=step, reward=td[
                                                                          "episode_reward"].mean(), )))
        pbar.close()
    # 3. 视频上传（如果配置了）
    if cfg.logger.eval_video: eval_env.apply(partial(dump_video, step=step))

    # 4. 从"next"字典的最后一帧提取所有数据
    episode_rewards = eval_rollout["next", "episode_reward"][
        eval_rollout["next", "done"]].cpu().numpy()  # RewardSum和StepCounter的输出在"next"中
    episode_lengths = eval_rollout["next", "step_count"][eval_rollout["next", "done"]].cpu().numpy()

    # 5. completion_ratio也在"next"的observation中
    completion_ratios = None
    if "completion_ratio" in eval_rollout["next"].keys():  # 使用[-1]是安全的，因为rollout保存的是done时的数据（reset发生在保存之后）
        completion_ratios = eval_rollout["next", "completion_ratio"][eval_rollout["next", "done"]].cpu().numpy()

    # 6. 关闭环境
    eval_env.close()

    # 7. 计算统计指标
    eval_metrics = {"eval/reward_mean": np.mean(episode_rewards), "eval/reward_min": np.min(episode_rewards),
                    "eval/reward_max": np.max(episode_rewards), "eval/episode_length": np.mean(episode_lengths), }

    # 添加completion_ratio统计（如果有）
    if completion_ratios is not None:
        eval_metrics["eval/completion_ratio"] = np.mean(completion_ratios)
        eval_metrics["eval/completion_ratio_max"] = np.max(completion_ratios)

    torchrl_logger.info(f"并行评估完成 - 平均奖励: {eval_metrics['eval/reward_mean']:.2f}")
    return eval_metrics


# ============ Checkpoint管理 ============
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torchrl._utils import logger as torchrl_logger


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
        if logger is not None:
            log_metrics(logger, result['metrics'], result['step'])

        # 上传视频到wandb（保持本地文件）
        if result['video_path'] and Path(result['video_path']).exists() and logger is not None:
            logger.log_video('eval/video', result['video_path'], step=result['step'])

        # 重命名模型文件（加入评估结果）
        old_model_path = checkpoint_dir / f"model_s{result['step']:06d}_eval_pending.pt"
        new_model_filename = f"model_s{result['step']:06d}_r{result['reward_mean']:.2f}_c{result['completion_rate']:.1f}.pt"
        new_model_path = checkpoint_dir / new_model_filename

        if old_model_path.exists():
            old_model_path.rename(new_model_path)
            torchrl_logger.info(f"模型已保存: {new_model_filename}")


def evaluate_policy_standalone(model_path: str, cfg, step: int):
    """
    独立评估函数 - 在子进程中运行，使用CSVLogger保存视频到本地
    
    Args:
        model_path: 模型文件路径
        cfg: 完整配置
        step: 当前训练步数
        
    Returns:
        dict: 包含metrics、视频路径和关键指标的字典
    """
    import torch
    from pathlib import Path
    from torchrl.record.loggers import CSVLogger
    import shutil

    # 根据配置选择评估设备
    device = torch.device(cfg.logger['eval_device'])

    # 加载模型到评估设备（weights_only=False以兼容TorchRL模型）
    actor_critic = torch.load(model_path, map_location=device, weights_only=False)

    # 创建CSVLogger - 使用MP4格式
    csv_logger = CSVLogger(
        exp_name=f"eval_{step}",
        log_dir=str(Path.cwd() / "eval_videos_temp"),  # 临时目录
        video_format="mp4",  # MP4格式便于直接查看
        video_fps=cfg.logger.get('eval_video_fps', 6)
    )

    # 执行评估
    eval_metrics = evaluate_policy_parallel(actor_critic=actor_critic, cfg=cfg, logger=csv_logger, step=step)

    # 提取关键指标（completion_ratio 可能不存在，做降级）
    reward_mean = float(eval_metrics['eval/reward_mean'])
    completion_rate = float(eval_metrics['eval/completion_ratio'])

    # 查找 CSVLogger 生成的视频文件并移动到最终目录（严格模式：仅接受 eval_video_{step}.mp4）
    tmp_dir = Path.cwd() / "eval_videos_temp" / f"eval_{step}" / "videos"
    final_dir = Path.cwd() / "eval_videos"
    final_dir.mkdir(exist_ok=True)

    temp_video_path = tmp_dir / "eval" / f"video_{step}.mp4"
    if not temp_video_path.exists():
        # 严格模式：不做回退，直接报错，便于及时发现评估录制路径问题
        raise FileNotFoundError(f"未找到严格匹配的视频文件: {temp_video_path}")

    # 生成最终文件名并移动
    video_filename = f"video_s{step:06d}_r{reward_mean:.2f}_c{completion_rate:.1f}.mp4"
    final_video_path = final_dir / video_filename
    try:
        shutil.move(str(temp_video_path), str(final_video_path))
        # 清理临时目录（容错）
        try:
            tmp_root = Path.cwd() / "eval_videos_temp" / f"eval_{step}"
            if tmp_root.exists():
                shutil.rmtree(tmp_root)
        except Exception:
            pass
        torchrl_logger.info(f"评估视频已保存: {video_filename}")
    except Exception as e:
        torchrl_logger.warning(f"移动评估视频失败: {e}")
        final_video_path = None

    return {
        'metrics': eval_metrics,
        'video_path': str(final_video_path) if final_video_path else None,
        'step': step,
        'reward_mean': reward_mean,
        'completion_rate': completion_rate
    }

def is_time_to_evaluate(current_frames, collected_frames, cfg):
    prev_frames = collected_frames - current_frames
    crossed = (prev_frames // cfg.logger.eval_interval) < (collected_frames // cfg.logger.eval_interval)
    final = collected_frames >= cfg.collector.total_frames

    return True if ((crossed and collected_frames >= cfg.collector.init_random_frames) or final) else False