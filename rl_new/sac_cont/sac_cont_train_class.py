"""
优化的SAC训练实现 - 模块化、高效、可扩展
支持6x RTX 3090单机多GPU数据收集和训练
"""

import math
import multiprocessing as mp
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import yaml
from omegaconf import DictConfig
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import MultiaSyncDataCollector, SyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictPrioritizedReplayBuffer
from torchrl.objectives import SoftUpdate, SACLoss
from torchrl.record.loggers import get_logger
from torchrl.envs import ExplorationType

from rl.sac_cont.sac_cont_utils import make_sac_models
from torchrl_utils import make_env


# ==================== 配置管理 ====================
@dataclass
class TrainingConfig:
    """统一的训练配置管理"""
    # 设备配置
    gpu_devices: Optional[List[int]] = None  # None=使用所有GPU, []仅CPU
    processes_per_gpu: int = 2  # 每个GPU的收集进程数
    cpu_workers: Optional[int] = None  # CPU工作进程数
    training_device: str = "cuda:0"  # 训练使用的主设备
    
    # 训练参数
    batch_size: int = 512
    learning_rate_actor: float = 1e-5
    learning_rate_critic: float = 3e-4
    learning_rate_alpha: float = 3e-4
    weight_decay_actor: float = 1e-4
    weight_decay_critic: float = 0.0
    weight_decay_alpha: float = 0.0
    
    # 优化配置
    use_amp: bool = True  # 混合精度训练
    gradient_accumulation: int = 1  # 梯度累积步数
    gradient_clip: Optional[float] = 1.0  # 梯度裁剪
    
    # 收集器配置
    frames_per_batch: int = 2000
    total_frames: int = 4_000_000
    init_random_frames: int = 50_000
    max_frames_per_traj: int = -1
    
    # 回放缓冲区配置
    buffer_size: int = 500_000
    buffer_alpha: float = 0.7
    buffer_beta: float = 0.5
    prefetch: int = 10
    
    # 损失函数配置
    gamma: float = 0.99
    target_update_polyak: float = 0.9997
    utd_ratio: float = 1.0
    loss_function: str = "l2"
    
    # 日志配置
    logger_backend: Optional[str] = "wandb"
    test_interval: int = 50_000
    checkpoint_interval: int = 50_000
    
    # 学习率调度
    use_lr_scheduler: bool = True
    scheduler_type: str = "cosine"  # "cosine", "step", "exponential"
    
    # 早停配置
    use_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    
    # 其他
    seed: int = 42
    ckpt_name: Optional[str] = None
    pretrained_model: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, cfg: DictConfig) -> 'TrainingConfig':
        """从YAML配置创建"""
        return cls(
            # 设备配置
            gpu_devices=cfg.collector.get('gpu_devices'),
            processes_per_gpu=cfg.collector.get('processes_per_gpu', 2),
            cpu_workers=cfg.collector.get('cpu_workers'),
            training_device=cfg.get('device', 'cuda:0'),
            
            # 训练参数
            batch_size=cfg.buffer.batch_size,
            learning_rate_actor=cfg.optim.lr_actor,
            learning_rate_critic=cfg.optim.lr_critic,
            learning_rate_alpha=cfg.optim.lr_alpha,
            weight_decay_actor=cfg.optim.weight_decay_actor,
            weight_decay_critic=cfg.optim.weight_decay_critic,
            weight_decay_alpha=cfg.optim.weight_decay_alpha,
            
            # 收集器配置
            frames_per_batch=cfg.collector.frames_per_batch,
            total_frames=cfg.collector.total_frames,
            init_random_frames=cfg.collector.init_random_frames,
            
            # 缓冲区配置
            buffer_size=cfg.buffer.buffer_size,
            
            # 损失配置
            gamma=cfg.loss.gamma,
            target_update_polyak=cfg.loss.target_update_polyak,
            utd_ratio=cfg.loss.utd_ratio,
            loss_function=cfg.loss.loss_function,
            gradient_clip=cfg.optim.get('max_grad_norm'),
            
            # 日志配置
            logger_backend=cfg.logger.backend,
            test_interval=cfg.logger.test_interval,
            
            # 其他
            seed=cfg.seed,
            ckpt_name=cfg.get('ckpt_name'),
            pretrained_model=cfg.get('pretrained_model'),
        )


# ==================== 组件工厂 ====================
class ComponentFactory:
    """统一的组件创建工厂"""
    
    @staticmethod
    def create_collector(config: TrainingConfig, actor: nn.Module) -> Any:
        """创建优化的数据收集器"""
        devices = ComponentFactory._get_devices(config)
        
        if len(devices) == 1:
            # 单进程收集器
            return SyncDataCollector(
                create_env_fn=lambda: make_env(num_envs=1, device='cpu'),
                policy=actor,
                frames_per_batch=config.frames_per_batch,
                total_frames=config.total_frames,
                device=devices[0],
                storing_device='cpu',
                max_frames_per_traj=config.max_frames_per_traj,
            )
        else:
            # 多进程异步收集器（推荐用于多GPU）
            return MultiaSyncDataCollector(
                create_env_fn=[lambda: make_env(num_envs=1, device='cpu')] * len(devices),
                policy=actor,
                frames_per_batch=config.frames_per_batch,
                total_frames=config.total_frames,
                device=devices,
                storing_device='cpu',
                max_frames_per_traj=config.max_frames_per_traj,
                postproc=None,
                split_trajs=False,
                exploration_type=ExplorationType.RANDOM,
                reset_when_done=True,
            )
    
    @staticmethod
    def _get_devices(config: TrainingConfig) -> List[str]:
        """获取设备列表"""
        devices = []
        
        # GPU设备
        if config.gpu_devices is not None:
            gpu_list = config.gpu_devices
            if gpu_list == -1:  # 使用所有GPU
                gpu_list = list(range(torch.cuda.device_count()))
            elif isinstance(gpu_list, int):  # 单个GPU ID
                gpu_list = [gpu_list]
            
            for gpu_id in gpu_list:
                devices.extend([f'cuda:{gpu_id}'] * config.processes_per_gpu)
        
        # CPU设备
        if config.cpu_workers is not None:
            cpu_count = config.cpu_workers
            if cpu_count == -1:  # 最大化CPU使用
                cpu_count = mp.cpu_count() - 2
            devices.extend(['cpu'] * cpu_count)
        
        # 默认配置
        if not devices:
            devices = ['cpu'] * 32  # 默认32个CPU进程
        
        return devices
    
    @staticmethod
    def create_replay_buffer(config: TrainingConfig) -> Any:
        """创建优化的回放缓冲区"""
        tempdir = tempfile.TemporaryDirectory()
        return TensorDictPrioritizedReplayBuffer(
            alpha=config.buffer_alpha,
            beta=config.buffer_beta,
            pin_memory=True,  # 固定内存以加速传输
            prefetch=config.prefetch,  # 预取批次
            storage=LazyMemmapStorage(
                max_size=config.buffer_size,
                scratch_dir=tempdir.name,
            ),
            batch_size=config.batch_size,
        )
    
    @staticmethod
    def create_optimizers(
        config: TrainingConfig,
        loss_module: SACLoss
    ) -> Tuple[optim.Optimizer, optim.Optimizer, optim.Optimizer]:
        """创建优化器"""
        critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
        actor_params = list(loss_module.actor_network_params.flatten_keys().values())
        
        optimizer_actor = optim.AdamW(
            actor_params,
            lr=config.learning_rate_actor,
            weight_decay=config.weight_decay_actor,
        )
        
        optimizer_critic = optim.AdamW(
            critic_params,
            lr=config.learning_rate_critic,
            weight_decay=config.weight_decay_critic,
        )
        
        optimizer_alpha = optim.AdamW(
            [loss_module.log_alpha],
            lr=config.learning_rate_alpha,
            weight_decay=config.weight_decay_alpha,
        )
        
        return optimizer_actor, optimizer_critic, optimizer_alpha
    
    @staticmethod
    def create_lr_schedulers(
        config: TrainingConfig,
        optimizer_actor: optim.Optimizer,
        optimizer_critic: optim.Optimizer,
        optimizer_alpha: optim.Optimizer,
    ) -> Optional[Tuple[Any, Any, Any]]:
        """创建学习率调度器"""
        if not config.use_lr_scheduler:
            return None, None, None
        
        total_steps = config.total_frames // config.frames_per_batch
        
        if config.scheduler_type == "cosine":
            scheduler_actor = optim.lr_scheduler.CosineAnnealingLR(
                optimizer_actor, T_max=total_steps
            )
            scheduler_critic = optim.lr_scheduler.CosineAnnealingLR(
                optimizer_critic, T_max=total_steps
            )
            scheduler_alpha = optim.lr_scheduler.CosineAnnealingLR(
                optimizer_alpha, T_max=total_steps
            )
        elif config.scheduler_type == "step":
            scheduler_actor = optim.lr_scheduler.StepLR(
                optimizer_actor, step_size=total_steps // 4, gamma=0.5
            )
            scheduler_critic = optim.lr_scheduler.StepLR(
                optimizer_critic, step_size=total_steps // 4, gamma=0.5
            )
            scheduler_alpha = optim.lr_scheduler.StepLR(
                optimizer_alpha, step_size=total_steps // 4, gamma=0.5
            )
        else:
            scheduler_actor = optim.lr_scheduler.ExponentialLR(
                optimizer_actor, gamma=0.999
            )
            scheduler_critic = optim.lr_scheduler.ExponentialLR(
                optimizer_critic, gamma=0.999
            )
            scheduler_alpha = optim.lr_scheduler.ExponentialLR(
                optimizer_alpha, gamma=0.999
            )
        
        return scheduler_actor, scheduler_critic, scheduler_alpha


# ==================== 辅助功能类 ====================
class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_reward = -float('inf')
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, reward: float) -> bool:
        if reward > self.best_reward + self.min_delta:
            self.best_reward = reward
            self.counter = 0
        else:
            self.counter += 1
            
        self.should_stop = self.counter >= self.patience
        return self.should_stop


class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save(
        self,
        actor_critic: Any,
        optimizers: Dict[str, optim.Optimizer],
        schedulers: Dict[str, Any],
        collected_frames: int,
        metrics: Dict[str, float],
    ):
        """保存检查点"""
        checkpoint = {
            'actor_critic': actor_critic.state_dict() if hasattr(actor_critic, 'state_dict') else actor_critic,
            'optimizers': {k: v.state_dict() for k, v in optimizers.items()},
            'schedulers': {k: v.state_dict() if v else None for k, v in schedulers.items()},
            'collected_frames': collected_frames,
            'metrics': metrics,
            'timestamp': time.time(),
        }
        
        model_name = str(collected_frames // 1000).rjust(5, '0')
        path = self.checkpoint_dir / f't[{model_name}].pt'
        torch.save(checkpoint, path)
        return path
    
    def load_latest(self):
        """加载最新的检查点"""
        checkpoints = list(self.checkpoint_dir.glob('t[*.pt'))
        if not checkpoints:
            return None
        
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return torch.load(latest)


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.timings = defaultdict(list)
    
    def track(self, name: str, value: float):
        """跟踪指标"""
        self.metrics[name].append(value)
    
    def track_time(self, name: str, duration: float):
        """跟踪时间"""
        self.timings[name].append(duration)
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """获取统计信息"""
        stats = {}
        
        # 指标统计
        for name, values in self.metrics.items():
            if values:
                recent = values[-100:]  # 最近100个值
                stats[name] = {
                    'mean': np.mean(recent),
                    'std': np.std(recent),
                    'max': np.max(recent),
                    'min': np.min(recent),
                }
        
        # 时间统计
        for name, durations in self.timings.items():
            if durations:
                recent = durations[-100:]
                stats[f'time_{name}'] = {
                    'mean': np.mean(recent),
                    'total': sum(durations),
                }
        
        return stats


# ==================== 主训练类 ====================
class OptimizedSACTrainer:
    """优化的SAC训练器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.training_device)
        
        # 设置随机种子
        self._set_seeds(config.seed)
        
        # 创建目录
        self.base_dir = Path(__file__).parent.parent.parent
        self.algo_name = 'sac_cont'
        self.ckpt_dir = self._create_checkpoint_dir()
        
        # 初始化组件
        self._setup_models()
        self._setup_training_components()
        self._setup_monitoring()
        
        # 混合精度训练 - 修复新版PyTorch的GradScaler API
        if torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler('cuda', enabled=config.use_amp)
        else:
            self.scaler = torch.amp.GradScaler('cpu', enabled=False)
        
        # 训练状态
        self.collected_frames = 0
        self.training_step = 0
    
    def _set_seeds(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _create_checkpoint_dir(self) -> Path:
        """创建检查点目录"""
        ckpt_dir = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        if self.config.ckpt_name:
            ckpt_dir += f'_{self.config.ckpt_name}'
        
        ckpt_path = self.base_dir / 'ckpt' / self.algo_name / ckpt_dir
        ckpt_path.mkdir(parents=True, exist_ok=True)
        return ckpt_path
    
    def _setup_models(self):
        """设置模型"""
        if self.config.pretrained_model:
            self.actor_critic = torch.load(
                self.base_dir / self.config.pretrained_model
            )
        else:
            self.actor_critic = make_sac_models()
        
        self.actor_critic = self.actor_critic.to(self.device)
        self.actor = self.actor_critic[0]
        self.q_critic = self.actor_critic[1]
    
    def _setup_training_components(self):
        """设置训练组件"""
        # 数据收集器
        self.collector = ComponentFactory.create_collector(self.config, self.actor)
        
        # 回放缓冲区
        self.replay_buffer = ComponentFactory.create_replay_buffer(self.config)
        
        # 损失模块
        self.loss_module = SACLoss(
            actor_network=self.actor,
            qvalue_network=self.q_critic,
            num_qvalue_nets=2,
            loss_function=self.config.loss_function,
            delay_actor=False,
            delay_qvalue=True,
        )
        self.loss_module.make_value_estimator(gamma=self.config.gamma)
        
        # 目标网络更新器
        self.target_net_updater = SoftUpdate(
            self.loss_module,
            eps=self.config.target_update_polyak
        )
        
        # 优化器
        self.optimizer_actor, self.optimizer_critic, self.optimizer_alpha = \
            ComponentFactory.create_optimizers(self.config, self.loss_module)
        
        # 学习率调度器
        self.scheduler_actor, self.scheduler_critic, self.scheduler_alpha = \
            ComponentFactory.create_lr_schedulers(
                self.config,
                self.optimizer_actor,
                self.optimizer_critic,
                self.optimizer_alpha
            )
        
        # 早停机制
        if self.config.use_early_stopping:
            self.early_stopping = EarlyStopping(
                patience=self.config.early_stopping_patience,
                min_delta=self.config.early_stopping_min_delta
            )
        else:
            self.early_stopping = None
    
    def _setup_monitoring(self):
        """设置监控组件"""
        # 检查点管理
        self.checkpoint_manager = CheckpointManager(self.ckpt_dir)
        
        # 性能监控
        self.performance_monitor = PerformanceMonitor()
        
        # 日志记录器
        self.logger = None
        if self.config.logger_backend:
            if self.config.logger_backend == 'wandb':
                self.logger = get_logger(
                    self.config.logger_backend,
                    logger_name=str(self.base_dir / 'ckpt'),
                    experiment_name=self.ckpt_dir.name,
                    wandb_kwargs={
                        "config": vars(self.config),
                    },
                )
            else:
                self.logger = get_logger(
                    self.config.logger_backend,
                    logger_name=str(self.base_dir / 'ckpt' / self.algo_name),
                    experiment_name=self.ckpt_dir.name,
                )
    
    def train_step(self, data):
        """单步训练"""
        # 计算需要的更新次数
        num_updates = math.ceil(
            self.config.frames_per_batch / self.config.batch_size * self.config.utd_ratio
        )
        
        actor_losses = []
        q_losses = []
        alpha_losses = []
        
        for _ in range(num_updates):
            # 从回放缓冲区采样
            sampled_tensordict = self.replay_buffer.sample()
            if sampled_tensordict.device != self.device:
                sampled_tensordict = sampled_tensordict.to(self.device, non_blocking=True)
            else:
                sampled_tensordict = sampled_tensordict.clone()
            
            # 使用混合精度计算损失
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                loss_out = self.loss_module(sampled_tensordict)
            
            actor_loss = loss_out["loss_actor"]
            q_loss = loss_out["loss_qvalue"]
            alpha_loss = loss_out["loss_alpha"]
            
            # 更新Actor
            self.optimizer_actor.zero_grad()
            self.scaler.scale(actor_loss).backward()
            if self.config.gradient_clip:
                self.scaler.unscale_(self.optimizer_actor)
                torch.nn.utils.clip_grad_norm_(
                    self.loss_module.actor_network_params.flatten_keys().values(),
                    self.config.gradient_clip
                )
            self.scaler.step(self.optimizer_actor)
            
            # 更新Critic
            self.optimizer_critic.zero_grad()
            self.scaler.scale(q_loss).backward()
            if self.config.gradient_clip:
                self.scaler.unscale_(self.optimizer_critic)
                torch.nn.utils.clip_grad_norm_(
                    self.loss_module.qvalue_network_params.flatten_keys().values(),
                    self.config.gradient_clip
                )
            self.scaler.step(self.optimizer_critic)
            
            # 更新Alpha
            self.optimizer_alpha.zero_grad()
            self.scaler.scale(alpha_loss).backward()
            self.scaler.step(self.optimizer_alpha)
            
            # 更新scaler
            self.scaler.update()
            
            # 更新目标网络
            self.target_net_updater.step()
            
            # 更新优先级
            self.replay_buffer.update_tensordict_priority(sampled_tensordict)
            
            # 记录损失
            actor_losses.append(actor_loss.detach().cpu().item())
            q_losses.append(q_loss.detach().cpu().item())
            alpha_losses.append(alpha_loss.detach().cpu().item())
        
        # 更新学习率
        if self.scheduler_actor:
            self.scheduler_actor.step()
            self.scheduler_critic.step()
            self.scheduler_alpha.step()
        
        return {
            "train/q_loss": np.mean(q_losses),
            "train/a_loss": np.mean(actor_losses),
            "train/alpha_loss": np.mean(alpha_losses),
        }
    
    def train(self):
        """主训练循环"""
        start_time = time.time()
        pbar = tqdm.tqdm(total=self.config.total_frames)
        
        # 尝试恢复训练
        checkpoint = self.checkpoint_manager.load_latest()
        if checkpoint:
            self._restore_from_checkpoint(checkpoint)
            torchrl_logger.info(f"Resumed from checkpoint at frame {self.collected_frames}")
        
        sampling_start = time.time()
        
        for i, data in enumerate(self.collector):
            # 性能监控
            sampling_time = time.time() - sampling_start
            self.performance_monitor.track_time('sampling', sampling_time)
            
            # 更新进度条
            pbar.update(data.numel())
            data = data.reshape(-1)
            current_frames = data.numel()
            self.collected_frames += current_frames
            
            # 处理episode数据
            log_info = self._process_episode_data(data)
            
            # 清理数据并加入缓冲区
            if 'weed_ratio' in data.keys():
                data.pop('weed_ratio')
                data.pop(('next', 'weed_ratio'))
            self.replay_buffer.extend(data)
            
            # 跳过随机初始化阶段
            if self.collected_frames < self.config.init_random_frames:
                if self.logger:
                    self._log_metrics(log_info)
                sampling_start = time.time()
                continue
            
            # 训练步骤
            training_start = time.time()
            train_metrics = self.train_step(data)
            training_time = time.time() - training_start
            self.performance_monitor.track_time('training', training_time)
            
            # 合并指标
            log_info.update(train_metrics)
            log_info.update({
                "train/sampling_time": sampling_time,
                "train/training_time": training_time,
                "train/fps": current_frames / (sampling_time + training_time),
            })
            
            # 记录性能指标
            for key, value in train_metrics.items():
                self.performance_monitor.track(key, value)
            
            # 保存检查点
            if self._should_save_checkpoint():
                self._save_checkpoint(log_info)
            
            # 日志记录
            if self.logger:
                self._log_metrics(log_info)
            
            # 早停检查
            if self.early_stopping:
                if 'train/episode_reward' in log_info:
                    if self.early_stopping(log_info['train/episode_reward']):
                        torchrl_logger.info(f"Early stopping triggered at frame {self.collected_frames}")
                        break
            
            # 更新策略权重
            self.collector.update_policy_weights_()
            
            sampling_start = time.time()
            self.training_step += 1
        
        # 训练结束
        self.collector.shutdown()
        pbar.close()
        
        end_time = time.time()
        execution_time = end_time - start_time
        torchrl_logger.info(f"Training completed in {execution_time:.2f} seconds")
        
        # 打印最终统计
        self._print_final_stats()
    
    def _process_episode_data(self, data) -> Dict[str, float]:
        """处理episode数据"""
        log_info = {}
        
        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        if len(episode_rewards) > 0:
            episode_reward_mean = episode_rewards.mean().item()
            episode_length = data["next", "step_count"][data["next", "done"]]
            episode_length_mean = episode_length.sum().item() / len(episode_length)
            
            log_info.update({
                "train/episode_reward": episode_reward_mean,
                "train/episode_length": episode_length_mean,
            })
            
            # 处理额外的环境特定指标
            if ("next", "weed_ratio") in data.keys():
                episode_weed_ratio = data["next", "weed_ratio"][data["next", "done"]]
                episode_weed_ratio_mean = episode_weed_ratio.sum().item() / len(episode_length)
                log_info["train/episode_weed_ratio"] = episode_weed_ratio_mean
        
        return log_info
    
    def _should_save_checkpoint(self) -> bool:
        """判断是否应该保存检查点"""
        return (self.collected_frames % self.config.checkpoint_interval) < self.config.frames_per_batch
    
    def _save_checkpoint(self, metrics: Dict[str, float]):
        """保存检查点"""
        optimizers = {
            'actor': self.optimizer_actor,
            'critic': self.optimizer_critic,
            'alpha': self.optimizer_alpha,
        }
        
        schedulers = {
            'actor': self.scheduler_actor,
            'critic': self.scheduler_critic,
            'alpha': self.scheduler_alpha,
        }
        
        path = self.checkpoint_manager.save(
            self.actor_critic,
            optimizers,
            schedulers,
            self.collected_frames,
            metrics
        )
        
        torchrl_logger.info(f"Checkpoint saved to {path}")
    
    def _restore_from_checkpoint(self, checkpoint: Dict):
        """从检查点恢复"""
        if isinstance(checkpoint['actor_critic'], dict):
            self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        else:
            self.actor_critic = checkpoint['actor_critic']
        
        self.optimizer_actor.load_state_dict(checkpoint['optimizers']['actor'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizers']['critic'])
        self.optimizer_alpha.load_state_dict(checkpoint['optimizers']['alpha'])
        
        if self.scheduler_actor and checkpoint['schedulers']['actor']:
            self.scheduler_actor.load_state_dict(checkpoint['schedulers']['actor'])
            self.scheduler_critic.load_state_dict(checkpoint['schedulers']['critic'])
            self.scheduler_alpha.load_state_dict(checkpoint['schedulers']['alpha'])
        
        self.collected_frames = checkpoint['collected_frames']
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """记录指标到日志"""
        if self.logger:
            for key, value in metrics.items():
                self.logger.log_scalar(key, value, step=self.collected_frames)
    
    def _print_final_stats(self):
        """打印最终统计信息"""
        stats = self.performance_monitor.get_stats()
        
        print("\n" + "="*50)
        print("Training Summary")
        print("="*50)
        
        for name, values in stats.items():
            if 'mean' in values:
                print(f"{name}: mean={values['mean']:.4f}, std={values['std']:.4f}")
            elif 'total' in values:
                print(f"{name}: total={values['total']:.2f}s, mean={values['mean']:.4f}s")
        
        print("="*50)


# ==================== 主函数 ====================
def main(cfg: Optional[DictConfig] = None):
    """主函数"""
    if cfg is None:
        # 从配置文件加载
        base_dir = Path(__file__).parent.parent.parent
        cfg = yaml.load(
            open(base_dir / 'configs' / 'train_sac_cont_config.yaml'),
            Loader=yaml.FullLoader
        )
        cfg = DictConfig(cfg)
    
    # 创建训练配置
    config = TrainingConfig.from_yaml(cfg)
    
    # 创建训练器并开始训练
    trainer = OptimizedSACTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()