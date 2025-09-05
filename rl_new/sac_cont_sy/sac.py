"""
SAC同步训练脚本 (sync version)
基于数据收集批次的同步训练模式
"""
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.cuda
import tqdm
from omegaconf import DictConfig
from tensordict import TensorDict
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import MultiaSyncDataCollector
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import SoftUpdate, SACLoss, group_optimizers
from torchrl.data import LazyMemmapStorage, TensorDictPrioritizedReplayBuffer
from torchrl.record.loggers import get_logger
import gymnasium as gym

# 添加项目路径
base_dir = Path(__file__).parent.parent.parent
sys.path.append(str(base_dir))

from rl_new.sac_cont_sy.model_utils import make_sac_models
from torchrl_utils_new.utils_env import make_sac_env
from rl_new.sac_cont_sy.sac_utils import setup_devices, create_update_fn
from torchrl_utils.local_video_recorder import LocalVideoRecorder

torch.set_float32_matmul_precision("high")  # 提升矩阵乘法性能

algo_name = 'sac_cont_sy'

# ============ 辅助函数 ============
def flatten(td):
    """将TensorDict展平为一维"""
    return td.reshape(-1)

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

def evaluate_policy(actor, env_id, env_kwargs, device, logger, step, cfg):
    """
    评估策略在多个episode上的表现，并录制视频。
    
    Args:
        actor: SAC actor模型
        env_id: 环境ID
        env_kwargs: 环境参数
        device: 计算设备
        logger: 日志记录器
        step: 当前训练步数
        cfg: 配置对象
    
    Returns:
        eval_metrics: 评估指标字典
    """
    # 创建评估环境（确保使用连续动作空间）
    eval_envs = []
    for _ in range(cfg.logger.eval_episodes):
        # 创建连续动作空间的环境
        env = gym.make(
            env_id, 
            render_mode=None,
            action_type='continuous',  # 关键：使用连续动作空间
            **env_kwargs
        )
        eval_envs.append(env)
    
    # 设置视频录制器（如果启用）
    recorder = None
    if cfg.logger.eval_video:
        max_frames = min(4, cfg.logger.eval_episodes)  # 最多录制4个环境的视频
        recorder = LocalVideoRecorder(
            max_len=(cfg.logger.eval_max_steps * max_frames) // cfg.logger.eval_video_skip + 2,
            skip=1,
            use_memmap=True,
            make_grid=True,
            nrow=2,
            fps=6,
        )
    
    # 使用确定性策略进行评估
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        # 初始化episode
        obss = []
        for env in eval_envs:
            obs, _ = env.reset()
            obss.append(obs)
        
        rewards = [0.0] * cfg.logger.eval_episodes
        dones = [False] * cfg.logger.eval_episodes
        completion_ratios = [0.0] * cfg.logger.eval_episodes  # 场地覆盖率（如果存在）
        
        # 录制初始帧
        if recorder:
            pixels = []
            for idx in range(min(4, cfg.logger.eval_episodes)):
                env = eval_envs[idx]
                # 使用render()获取图像
                pixel = env.render()
                if pixel is not None:
                    pixels.append(pixel)
            if pixels:
                recorder.apply(torch.from_numpy(np.stack(pixels, 0)))
        
        # 评估循环
        for t in range(cfg.logger.eval_max_steps):
            # 获取动作
            actions = get_actor_actions(actor, obss, device)
            
            # 执行环境步进
            new_obss = []
            act_idx = 0
            for idx, env in enumerate(eval_envs):
                if not dones[idx]:
                    obs, reward, terminated, truncated, info = env.step(actions[act_idx])
                    new_obss.append(obs)
                    rewards[idx] += reward
                    dones[idx] = terminated or truncated
                    # 获取completion_ratio（如果存在）
                    if isinstance(obs, dict) and 'completion_ratio' in obs:
                        completion_ratios[idx] = obs['completion_ratio']
                    act_idx += 1
            
            obss = new_obss
            
            # 录制视频帧
            if recorder and (t + 1) % cfg.logger.eval_video_skip == 0:
                pixels = []
                for idx in range(min(4, cfg.logger.eval_episodes)):
                    if not dones[idx]:
                        pixel = eval_envs[idx].render()
                        if pixel is not None:
                            pixels.append(pixel)
                if pixels:
                    recorder.apply(torch.from_numpy(np.stack(pixels, 0)))
            
            # 检查是否所有episode都结束
            if all(dones):
                break
    
    # 计算评估指标
    eval_metrics = {
        "eval/reward": np.mean(rewards),
        "eval/reward_std": np.std(rewards),
        "eval/reward_min": np.min(rewards),
        "eval/reward_max": np.max(rewards),
        "eval/episodes": len(rewards),
    }
    
    # 如果有completion_ratio，添加到指标中
    if any(cr > 0 for cr in completion_ratios):
        eval_metrics["eval/completion_ratio"] = np.mean(completion_ratios)
    
    # 记录指标和视频
    if logger:
        for key, value in eval_metrics.items():
            logger.log_scalar(key, value, step=step)
        
        if recorder:
            video_tensor = recorder.dump()
            if video_tensor is not None:
                logger.log_video('eval/video', video_tensor, step=step)
    
    # 清理环境
    for env in eval_envs:
        env.close()
    
    return eval_metrics

# ============ 主训练函数 ============
@hydra.main(version_base="1.1", config_path=".", config_name="config")
def main(cfg: DictConfig):
    # 处理临时目录路径
    temp_dir = cfg.buffer.temp_dir
    if temp_dir and temp_dir.startswith('~'):
        temp_dir = os.path.expanduser(temp_dir)
    
    with tempfile.TemporaryDirectory(dir=temp_dir) as tmpdir:
        # ============ 创建实验目录和基础设置 ============
        exp_name = cfg.ckpt_name
        ckpt_path = base_dir / 'ckpt' / algo_name / exp_name / time.strftime('%Y-%m-%d_%H-%M-%S')
        ckpt_path.mkdir(parents=True, exist_ok=True)

        # 设备配置
        train_device, collector_devices = setup_devices(cfg)
        torchrl_logger.info(f"训练设备: {train_device}")
        torchrl_logger.info(f"收集设备: {collector_devices[:5]}... (共{len(collector_devices)}个)")

        # 设置随机种子
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        # ============ 创建日志记录器 ============
        logger = None
        if cfg.logger.backend:
            logger = get_logger(
                logger_type=cfg.logger.backend,
                logger_name=str(ckpt_path),
                experiment_name=exp_name,
                wandb_kwargs={
                    "config": dict(cfg),
                    "project": cfg.logger.project_name,
                    "name": exp_name,
                },
            )

        # ============ 创建模型 ============
        # 使用配置中的环境ID创建模型
        if cfg.pretrained_model:
            actor_critic = torch.load(base_dir / cfg.pretrained_model)
            torchrl_logger.info(f"加载预训练模型: {cfg.pretrained_model}")
        else:
            # 创建一个样本环境用于模型创建
            torchrl_logger.info(f"创建环境: {cfg.env.env_id}")
            env_kwargs = dict(cfg.env.env_kwargs) if hasattr(cfg.env, 'env_kwargs') and cfg.env.env_kwargs else {}
            proof_env = make_sac_env(
                env_id=cfg.env.env_id, 
                num_envs=1,
                **env_kwargs
            )
            actor_critic = make_sac_models(env=proof_env)
            proof_env.close()
            
        actor_critic = actor_critic.to(train_device)
        actor = actor_critic[0]
        q_critic = actor_critic[1]

        # ============ 创建回放缓冲区 ============
        replay_buffer = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.5,
            pin_memory=cfg.buffer.pin_memory,
            prefetch=cfg.buffer.prefetch,
            storage=LazyMemmapStorage(
                max_size=cfg.buffer.buffer_size,
                scratch_dir=tmpdir,
            ),
            batch_size=cfg.buffer.batch_size,
        )
        replay_buffer.append_transform(lambda td: td.to(train_device))
        replay_buffer.empty()

        # ============ 创建收集器（同步模式，不传递replay_buffer） ============
        env_kwargs = dict(cfg.env.env_kwargs) if hasattr(cfg.env, 'env_kwargs') and cfg.env.env_kwargs else {}
        collector = MultiaSyncDataCollector(
            create_env_fn=[lambda d=dev: make_sac_env(
                env_id=cfg.env.env_id,
                num_envs=cfg.collector.processes_per_gpu if 'cuda' in str(dev) else 1,
                device=str(d),
                **env_kwargs
            ) for dev in collector_devices],
            policy=actor,
            policy_device='cpu',
            frames_per_batch=cfg.collector.frames_per_batch,
            total_frames=cfg.collector.total_frames,
            device=collector_devices,
            storing_device='cpu',
            max_frames_per_traj=-1,
            # 不传递 replay_buffer，使用同步收集模式
            postproc=flatten,
        )
        collector.set_seed(cfg.seed)
        torchrl_logger.info(f"创建{len(collector_devices)}个收集进程 (同步模式)")

        # ============ 创建损失和优化器 ============
        loss_module = SACLoss(
            actor_network=actor,
            qvalue_network=q_critic,
            num_qvalue_nets=2,
            loss_function=cfg.loss.loss_function,
            delay_actor=False,
            delay_qvalue=True,
        )
        loss_module.make_value_estimator(gamma=cfg.loss.gamma)

        # 目标网络更新器
        target_net_updater = SoftUpdate(loss_module, eps=cfg.loss.target_update_polyak)

        # 创建优化器
        critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
        actor_params = list(loss_module.actor_network_params.flatten_keys().values())

        optimizer_actor = torch.optim.AdamW(
            actor_params,
            lr=cfg.optim.lr_actor,
            weight_decay=cfg.optim.weight_decay_actor,
        )
        optimizer_critic = torch.optim.AdamW(
            critic_params,
            lr=cfg.optim.lr_critic,
            weight_decay=cfg.optim.weight_decay_critic,
        )
        optimizer_alpha = torch.optim.AdamW(
            [loss_module.log_alpha],
            lr=cfg.optim.lr_alpha,
            weight_decay=cfg.optim.weight_decay_alpha,
        )

        # 使用group_optimizers合并优化器
        optimizer = group_optimizers(optimizer_actor, optimizer_critic, optimizer_alpha)
        del optimizer_actor, optimizer_critic, optimizer_alpha

        # 创建GradScaler（如果使用混合精度）
        scaler = None
        if cfg.training.use_amp and torch.cuda.is_available():
            scaler = torch.amp.GradScaler('cuda')
            torchrl_logger.info("启用混合精度训练 (AMP)")

        # 创建优化函数
        update_fn = create_update_fn(
            loss_module, optimizer, target_net_updater, cfg, scaler
        )

        # ============ 主训练循环（同步模式） ============
        start_time = time.time()
        
        # 核心参数
        init_random_frames = cfg.collector.init_random_frames
        batch_size = cfg.buffer.batch_size
        frames_per_batch = cfg.collector.frames_per_batch
        num_updates = math.ceil(frames_per_batch / batch_size * cfg.loss.utd_ratio)
        test_interval = cfg.logger.test_interval
        log_freq = cfg.logger.log_freq
        
        # 初始化统计
        collected_frames = 0
        pbar = tqdm.tqdm(total=cfg.collector.total_frames, desc="收集数据")
        
        # 同步收集循环
        for i, data in enumerate(collector):
            log_info = {}
            
            # 处理收集到的数据
            pbar.update(data.numel())
            data = data.reshape(-1)  # 展平数据
            current_frames = data.numel()
            collected_frames += current_frames
            
            # 提取episode统计信息
            if ("next", "done") in data.keys(include_nested=True):
                done_mask = data["next", "done"]
                if done_mask.any():
                    # 收集episode奖励
                    if ("next", "episode_reward") in data.keys(include_nested=True):
                        episode_rewards = data["next", "episode_reward"][done_mask]
                        if len(episode_rewards) > 0:
                            log_info["train/episode_reward"] = episode_rewards.mean().item()
                            log_info["train/episode_reward_max"] = episode_rewards.max().item()
                            log_info["train/episode_reward_min"] = episode_rewards.min().item()
                    
                    # 收集episode长度
                    if ("next", "step_count") in data.keys(include_nested=True):
                        episode_lengths = data["next", "step_count"][done_mask]
                        if len(episode_lengths) > 0:
                            log_info["train/episode_length"] = episode_lengths.float().mean().item()
                    
                    # 收集weed_ratio（如果存在）
                    if ("next", "completion_ratio") in data.keys(include_nested=True):
                        episode_completion_ratios = data["next", "completion_ratio"][done_mask]
                        if len(episode_completion_ratios) > 0:
                            log_info["train/episode_completion_ratio"] = episode_completion_ratios.mean().item()
                            # 删除额外信息，避免replay buffer存储
                            data.pop('completion_ratio', None)
                            data.pop(('next', 'completion_ratio'), None)
            
            # 手动添加到replay_buffer
            replay_buffer.extend(data)
            
            # 如果还在收集初始随机帧，跳过训练
            if collected_frames < init_random_frames:
                if logger and log_info:
                    for key, value in log_info.items():
                        logger.log_scalar(key, value, step=collected_frames)
                continue
            
            # ============ 训练更新 ============
            losses = []
            for j in range(num_updates):
                # 从回放缓冲区采样
                sampled_tensordict = replay_buffer.sample()
                if sampled_tensordict.device != train_device:
                    sampled_tensordict = sampled_tensordict.to(train_device, non_blocking=True)
                
                # 执行更新
                loss_out = update_fn(sampled_tensordict)
                losses.append(loss_out.select("loss_actor", "loss_qvalue", "loss_alpha"))
                
                # 更新优先级
                td_error = (loss_out["loss_qvalue"] + loss_out["loss_actor"]).abs()
                priority = td_error.expand(batch_size).detach()
                replay_buffer.update_priority(sampled_tensordict["index"], priority)
            
            # ============ 日志记录 ============
            if i % 10 == 0 and len(losses) > 0:  # 每10个批次记录一次
                # 计算平均损失
                losses_tensor = torch.stack(losses)
                log_info.update({
                    "train/q_loss": losses_tensor["loss_qvalue"].mean().item(),
                    "train/a_loss": losses_tensor["loss_actor"].mean().item(),
                    "train/alpha_loss": losses_tensor["loss_alpha"].mean().item(),
                    "train/alpha": loss_out["alpha"],
                    "train/entropy": loss_out["entropy"],
                })
                
                # 记录收集信息
                elapsed_time = time.time() - start_time
                log_info.update({
                    "train/collected_frames": collected_frames,
                    "train/frames_per_sec": collected_frames / elapsed_time,
                    "train/batches": i + 1,
                })
                
                # 写入日志
                if logger:
                    for key, value in log_info.items():
                        if isinstance(value, torch.Tensor):
                            value = value.item()
                        logger.log_scalar(key, value, step=collected_frames)
            
            # ============ 保存模型 ============
            if collected_frames % test_interval == 0:
                reward_str = f"{log_info.get('train/episode_reward', 0):.2f}" if 'train/episode_reward' in log_info else "0"
                model_name = f"f[{collected_frames//1000:05d}]_r[{reward_str}].pt"
                torch.save(
                    actor_critic,
                    ckpt_path / model_name
                )
                torchrl_logger.info(f"保存模型: {model_name}")
            
            # ============ 评估 ============
            eval_interval = cfg.logger.eval_interval
            if collected_frames % eval_interval == 0 and collected_frames > 0:
                torchrl_logger.info(f"开始评估 (frames: {collected_frames})")
                
                # 执行评估
                eval_metrics = evaluate_policy(
                    actor=actor,
                    env_id=cfg.env.env_id,
                    env_kwargs=env_kwargs,
                    device=train_device,
                    logger=logger,
                    step=collected_frames,
                    cfg=cfg
                )
                
                # 记录评估结果
                torchrl_logger.info(
                    f"评估完成 - 奖励: {eval_metrics['eval/reward']:.2f} ± {eval_metrics['eval/reward_std']:.2f} "
                    f"[{eval_metrics['eval/reward_min']:.2f}, {eval_metrics['eval/reward_max']:.2f}]"
                )
            
            # 检查是否达到总帧数
            if collected_frames >= cfg.collector.total_frames:
                break
        
        # 训练结束
        collector.shutdown()
        end_time = time.time()
        execution_time = end_time - start_time
        torchrl_logger.info(f"训练完成，耗时: {execution_time:.2f}秒")
        torchrl_logger.info(f"总帧数: {collected_frames}")
        torchrl_logger.info(f"平均FPS: {collected_frames / execution_time:.2f}")

if __name__ == "__main__":
    main()