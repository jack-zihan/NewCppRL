"""
全面优化的SAC训练脚本
"""
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from functools import partial

import hydra
import numpy as np
import tensordict
import torch
import torch.cuda
import tqdm
from omegaconf import DictConfig
from tensordict import TensorDict
from tensordict.nn import CudaGraphModule
from torchrl._utils import compile_with_warmup, logger as torchrl_logger, timeit
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.collectors import MultiaSyncDataCollector
from torchrl.objectives import SoftUpdate, SACLoss, group_optimizers
from torchrl.data import LazyMemmapStorage, TensorDictPrioritizedReplayBuffer
from torchrl.record.loggers import generate_exp_name, get_logger

# 添加项目路径
base_dir = Path(__file__).parent.parent.parent
sys.path.append(str(base_dir))

from rl_new.sac_cont_new.sac_cont_model import make_sac_models
from torchrl_utils_new.utils_env import make_sac_env
from rl_new.sac_cont_new.sac_utils import setup_devices, create_update_fn

torch.set_float32_matmul_precision("high")  # 提升矩阵乘法性能
tensordict.nn.functional_modules._exclude_td_from_pytree().set()  # TensorDict优化

algo_name = 'sac_cont_new'

# ============ 辅助函数 ============
def flatten(td):
    """将TensorDict展平为一维"""
    return td.reshape(-1)

class EpisodeStatsTracker:
    """异步训练中的episode统计跟踪器"""
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_weed_ratios = []
        self.last_check_frames = 0
    
    def update_from_buffer(self, replay_buffer, current_frames):
        """从replay_buffer中提取最新的episode统计信息"""
        # 只检查新增的数据
        if current_frames <= self.last_check_frames:
            return {}
        
        # 采样一批数据（优先采样最新的）
        sample_size = min(5000, len(replay_buffer))
        if sample_size == 0:
            return {}
        
        sample = replay_buffer.sample(batch_size=sample_size)
        stats = {}
        
        if ("next", "done") in sample.keys():
            done_mask = sample["next", "done"]
            
            # 收集episode奖励
            if ("next", "episode_reward") in sample.keys():
                episode_rewards = sample["next", "episode_reward"][done_mask]
                if len(episode_rewards) > 0:
                    self.episode_rewards.extend(episode_rewards.tolist())
                    # 保持窗口大小
                    self.episode_rewards = self.episode_rewards[-self.window_size:]
                    
                    if self.episode_rewards:
                        rewards_array = torch.tensor(self.episode_rewards)
                        stats["train/episode_reward"] = rewards_array.mean().item()
                        stats["train/episode_reward_std"] = rewards_array.std().item()
                        stats["train/episode_reward_max"] = rewards_array.max().item()
                        stats["train/episode_reward_min"] = rewards_array.min().item()
                        stats["train/episodes_in_window"] = len(self.episode_rewards)
            
            # 收集episode长度
            if ("next", "step_count") in sample.keys():
                episode_lengths = sample["next", "step_count"][done_mask]
                if len(episode_lengths) > 0:
                    self.episode_lengths.extend(episode_lengths.tolist())
                    self.episode_lengths = self.episode_lengths[-self.window_size:]
                    
                    if self.episode_lengths:
                        lengths_array = torch.tensor(self.episode_lengths, dtype=torch.float32)
                        stats["train/episode_length"] = lengths_array.mean().item()
                        stats["train/episode_length_std"] = lengths_array.std().item()
            
            # 收集weed_ratio（如果存在）
            if ("next", "weed_ratio") in sample.keys():
                episode_weed_ratios = sample["next", "weed_ratio"][done_mask]
                if len(episode_weed_ratios) > 0:
                    self.episode_weed_ratios.extend(episode_weed_ratios.tolist())
                    self.episode_weed_ratios = self.episode_weed_ratios[-self.window_size:]
                    
                    if self.episode_weed_ratios:
                        weed_array = torch.tensor(self.episode_weed_ratios)
                        stats["train/episode_weed_ratio"] = weed_array.mean().item()
        
        self.last_check_frames = current_frames
        return stats

# ============ 主训练函数 ============
@hydra.main(version_base="1.1", config_path=".", config_name="config")
def main(cfg: DictConfig):
    with tempfile.TemporaryDirectory(cfg.buffer.temp_dir) as tmpdir: # 使用临时目录存储内存映射文件
        # ============ 创建实验目录和基础设置 ============
        exp_name = cfg.ckpt_name # 创建实验名称
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

        # ============ 创建模型 ============ TODO：这里要小心，上面说的内容
        if cfg.pretrained_model:
            actor_critic = torch.load(base_dir / cfg.pretrained_model)
            torchrl_logger.info(f"加载预训练模型: {cfg.pretrained_model}")
        else:
            actor_critic = make_sac_models()
        actor_critic = actor_critic.to(train_device)# 没有使用分布式训练，模型就在一个设备上
        actor = actor_critic[0]
        q_critic = actor_critic[1]

        # ============ 创建回放缓冲区（优化版） ============
        replay_buffer = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.5,
            pin_memory=cfg.buffer.get('pin_memory', True),  # 锁定内存
            prefetch=cfg.buffer.get('prefetch', 10),  # 预取优化
            storage=LazyMemmapStorage(
                max_size=cfg.buffer.buffer_size,
                scratch_dir=tmpdir,
            ),
            batch_size=cfg.buffer.batch_size,
            shared=True,
        )
        replay_buffer.append_transform(lambda td: td.to(train_device))
        replay_buffer.empty()

        # ============ 创建收集器（保持MultiaSyncDataCollector） ============
        collector = MultiaSyncDataCollector(
            create_env_fn=[lambda d=dev: make_env(num_envs=cfg.collector.processes_per_gpu, device=str(d))
                          for dev in collector_devices],
            policy=actor,
            frames_per_batch=cfg.collector.frames_per_batch,
            total_frames=cfg.collector.total_frames,
            device=collector_devices,  # 直接传递设备列表
            storing_device='cpu',
            max_frames_per_traj=-1,
            replay_buffer=replay_buffer,
            extend_buffer=True,
            no_cuda_sync=True,
            postproc=flatten, # [num_envs, frames_per_batch, ...]  -> [num_envs * frames_per_batch, ...]
            compile_policy={"mode": cfg.compile.mode, "warmup": 5} if cfg.compile.enable else False,
            cudagraph_policy={"warmup": 20} if cfg.compile.cudagraph else False,
            # 注意：MultiaSyncDataCollector在TorchRL 0.9.2中不支持pin_memory参数
        )
        collector.set_seed(cfg.seed)
        # collector.start()
        torchrl_logger.info(f"创建{len(collector_devices)}个收集进程 (MultiaSyncDataCollector)")

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

        # 创建优化器（使用group_optimizers）
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

        # 使用group_optimizers合并优化器（官方方式）
        optimizer = group_optimizers(optimizer_actor, optimizer_critic, optimizer_alpha)
        del optimizer_actor, optimizer_critic, optimizer_alpha

        # 创建GradScaler（如果使用混合精度）
        scaler = None
        if cfg.training.use_amp and torch.cuda.is_available():
            scaler = torch.amp.GradScaler('cuda')
            torchrl_logger.info("启用混合精度训练 (AMP)")

        # ============ 创建优化函数 ============
        update_fn = create_update_fn(
            loss_module, optimizer, target_net_updater, cfg, scaler
        )

        # ============ 主训练循环（异步架构） ============
        start_time = time.time()

        # 核心参数
        init_random_frames = cfg.collector.init_random_frames
        batch_size = cfg.buffer.batch_size
        total_updates = cfg.training.total_updates
        update_freq = cfg.collector.get('update_freq', 1000)  # 多少次更新后同步权重
        test_interval = cfg.logger.test_interval
        log_freq = cfg.logger.get('log_freq', 10000)
        
        # 性能监控初始化
        timeit.printevery(
            num_prints=total_updates // (log_freq // 10),
            total_count=total_updates,
            erase=True,
        )
        
        # 等待初始随机数据收集
        torchrl_logger.info(f"等待收集初始随机帧: {init_random_frames}")
        while replay_buffer._writer._write_count < init_random_frames:
            time.sleep(0.1)
            current_frames = replay_buffer._writer._write_count
            if current_frames > 0 and current_frames % 10000 == 0:
                torchrl_logger.info(f"已收集: {current_frames}/{init_random_frames} 帧")
        
        torchrl_logger.info(f"初始数据收集完成，开始训练 {total_updates} 次更新")
        pbar = tqdm.tqdm(total=total_updates, desc="Training Updates")
        
        # 初始化episode统计跟踪器
        episode_tracker = EpisodeStatsTracker(window_size=100)
        
        # 训练循环：基于更新次数，而非数据批次
        losses = []
        training_start = time.time()
        
        for update_i in range(total_updates):
            pbar.update(1)
            log_info = {}
            
            # ============ 采样和训练 ============
            with timeit("train"):
                # 从回放缓冲区采样
                with timeit("rb_sample"):
                    sampled_tensordict = replay_buffer.sample()
                    if sampled_tensordict.device != train_device:
                        sampled_tensordict = sampled_tensordict.to(train_device, non_blocking=True)
                
                # 执行更新
                with timeit("update"):
                    if cfg.compile.enable and cfg.compile.cudagraph:
                        torch.compiler.cudagraph_mark_step_begin()
                    
                    loss_out = update_fn(sampled_tensordict)
                
                # 记录损失
                losses.append(loss_out.select("loss_actor", "loss_qvalue", "loss_alpha"))
                
                # 更新优先级
                with timeit("rb_update_priority"):
                    td_error = (loss_out["loss_qvalue"] + loss_out["loss_actor"]).abs()
                    priority = td_error.expand(cfg.buffer.batch_size).detach()
                    replay_buffer.update_priority(sampled_tensordict["index"], priority)
            
            # ============ 定期同步权重到收集器 ============
            if (update_i + 1) % update_freq == 0:
                with timeit("sync_weights"):
                    collector.update_policy_weights_()
                    torchrl_logger.info(f"同步权重到收集器 (update {update_i + 1})")
            
            # ============ 日志记录（不阻塞训练） ============
            if (update_i + 1) % (log_freq // 100) == 0:  # 每log_freq/100次更新记录一次
                # 计算平均损失
                if len(losses) > 0:
                    losses_tensor = torch.stack(losses[-100:])  # 最近100次的平均
                    log_info.update({
                        "train/q_loss": losses_tensor.get("loss_qvalue").mean().item(),
                        "train/a_loss": losses_tensor.get("loss_actor").mean().item(),
                        "train/alpha_loss": losses_tensor.get("loss_alpha").mean().item(),
                        "train/alpha": loss_out.get("alpha", 0),
                        "train/entropy": loss_out.get("entropy", 0),
                    })
                
                # 使用episode统计跟踪器获取episode信息
                with timeit("log_episode"):
                    collected_frames = replay_buffer._writer._write_count
                    episode_stats = episode_tracker.update_from_buffer(replay_buffer, collected_frames)
                    log_info.update(episode_stats)
                
                # 记录收集信息
                collected_frames = replay_buffer._writer._write_count
                elapsed_time = time.time() - training_start
                log_info.update({
                    "train/updates": update_i + 1,
                    "train/collected_frames": collected_frames,
                    "train/updates_per_sec": (update_i + 1) / elapsed_time,
                    "train/frames_per_sec": collected_frames / elapsed_time,
                })
                
                # 添加timeit性能指标
                if update_i % 1000 == 0:
                    log_info.update(timeit.todict(prefix="time"))
                
                # 写入日志
                if logger:
                    for key, value in log_info.items():
                        if isinstance(value, torch.Tensor):
                            value = value.item()
                        logger.log_scalar(key, value, step=update_i)
            
            # ============ 保存模型 ============
            if (update_i + 1) % (test_interval // 10) == 0:  # 基于更新次数保存
                model_name = str(update_i // 1000).rjust(5, '0')
                torch.save(
                    actor_critic,
                    ckpt_path / f'u[{model_name}].pt'  # u表示updates
                )
                torchrl_logger.info(f"保存模型: u[{model_name}].pt (update {update_i + 1})")

        # 训练结束
        collector.shutdown()
        end_time = time.time()
        execution_time = end_time - start_time
        torchrl_logger.info(f"训练完成，耗时: {execution_time:.2f}秒")
        torchrl_logger.info(f"平均FPS: {collected_frames / execution_time:.2f}")

        # 记录最终性能统计
        if logger:
            final_metrics = {
                "final/total_time": execution_time,
                "final/avg_fps": collected_frames / execution_time,
                "final/total_frames": collected_frames,
            }
            final_metrics.update(timeit.todict(prefix="final/time"))
            for key, value in final_metrics.items():
                logger.log_scalar(key, value, step=collected_frames)


if __name__ == "__main__":
    main()