# 🚨 训练主循环根本性问题修复方案

## ❌ 核心错误：错误使用MultiaSyncDataCollector

### 当前代码的根本性错误
```python
# ❌ 完全错误的使用方式！
for i, data in enumerate(collector):  # 不应该迭代异步collector！
    # 手动处理data
    replay_buffer.extend(data)  # 异步collector已经自动写入了！
```

**问题本质**：
- MultiaSyncDataCollector是**异步**的，在后台独立运行
- 它会**自动**将数据写入replay_buffer（因为`extend_buffer=True`）
- 不应该像同步collector那样迭代它
- `for data in collector`会阻塞等待，失去了异步的全部优势

## ✅ 正确的异步训练循环

### 完整的修复方案

```python
"""
全面优化的SAC训练脚本 - 修复版
"""
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import hydra
import numpy as np
import tensordict
import torch
import torch.cuda
import tqdm
from omegaconf import DictConfig
from tensordict import TensorDict
from torchrl._utils import compile_with_warmup, logger as torchrl_logger, timeit
from torchrl.collectors import MultiaSyncDataCollector
from torchrl.objectives import SoftUpdate, SACLoss, group_optimizers
from torchrl.data import LazyMemmapStorage, TensorDictPrioritizedReplayBuffer
from torchrl.record.loggers import get_logger

# 添加项目路径
base_dir = Path(__file__).parent.parent.parent
sys.path.append(str(base_dir))

from rl_new.sac_cont.sac_cont_utils import make_sac_models
from torchrl_utils import make_env
from rl_new.sac_cont_new.sac_utils import setup_devices, create_update_fn

torch.set_float32_matmul_precision("high")
tensordict.nn.functional_modules._exclude_td_from_pytree().set()

algo_name = 'sac_cont_new'

@hydra.main(version_base="1.1", config_path=".", config_name="config")
def main(cfg: DictConfig):
    with tempfile.TemporaryDirectory(cfg.buffer.temp_dir) as tmpdir:
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
        if cfg.pretrained_model:
            actor_critic = torch.load(base_dir / cfg.pretrained_model)
            torchrl_logger.info(f"加载预训练模型: {cfg.pretrained_model}")
        else:
            actor_critic = make_sac_models()
        actor_critic = actor_critic.to(train_device)
        actor = actor_critic[0]
        q_critic = actor_critic[1]

        # ============ 创建回放缓冲区 ============
        replay_buffer = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.5,
            pin_memory=cfg.buffer.get('pin_memory', True),
            prefetch=cfg.buffer.get('prefetch', 10),
            storage=LazyMemmapStorage(
                max_size=cfg.buffer.buffer_size,
                scratch_dir=tmpdir,
            ),
            batch_size=cfg.buffer.batch_size,
            shared=True,  # 重要：共享内存
        )
        
        # 初始化replay buffer的结构
        dummy_env = make_env(num_envs=1, device='cpu')
        dummy_data = dummy_env.rollout(1).view(-1)
        replay_buffer.extend(dummy_data)
        replay_buffer.empty()
        dummy_env.close()
        del dummy_env, dummy_data

        # ============ 创建异步收集器 ============
        collector = MultiaSyncDataCollector(
            create_env_fn=[lambda d=dev: make_env(num_envs=cfg.collector.processes_per_gpu, device=str(d))
                          for dev in collector_devices],
            policy=actor,
            frames_per_batch=cfg.collector.frames_per_batch,
            total_frames=cfg.collector.total_frames,
            device=collector_devices,
            storing_device='cpu',
            max_frames_per_traj=-1,
            replay_buffer=replay_buffer,
            extend_buffer=True,  # 关键：自动写入replay_buffer
            no_cuda_sync=True,
            postproc=lambda x: x.reshape(-1),  # 展平batch
            compile_policy={"mode": cfg.compile.mode, "warmup": 5} if cfg.compile.enable else False,
            cudagraph_policy={"warmup": 20} if cfg.compile.cudagraph else False,
        )
        collector.set_seed(cfg.seed)
        collector.start()  # 启动后台收集进程
        torchrl_logger.info(f"启动{len(collector_devices)}个异步收集进程")

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

        optimizer = group_optimizers(optimizer_actor, optimizer_critic, optimizer_alpha)
        del optimizer_actor, optimizer_critic, optimizer_alpha

        # 创建GradScaler（如果使用混合精度）
        scaler = None
        if cfg.training.use_amp and torch.cuda.is_available():
            scaler = torch.amp.GradScaler('cuda')
            torchrl_logger.info("启用混合精度训练 (AMP)")

        # 创建优化的更新函数
        update_fn = create_update_fn(
            loss_module, optimizer, target_net_updater, cfg, scaler
        )

        # ============ 🔥 正确的异步主训练循环 ============
        start_time = time.time()
        
        # 配置参数
        init_random_frames = cfg.collector.init_random_frames
        batch_size = cfg.buffer.batch_size
        frames_per_batch = cfg.collector.frames_per_batch
        num_updates = math.ceil(frames_per_batch / batch_size * cfg.loss.utd_ratio)
        test_interval = cfg.logger.test_interval
        log_freq = cfg.logger.get('log_freq', 10000)
        update_freq = cfg.collector.get('update_freq', 1000)  # 权重同步频率
        
        # 初始化计数器
        update_counter = 0
        collected_frames = 0
        last_checkpoint_frame = 0
        last_log_frame = 0
        
        # 进度条
        pbar = tqdm.tqdm(total=cfg.collector.total_frames, desc="Training")
        
        # 损失记录
        losses = []
        episode_rewards = []
        episode_lengths = []
        
        # 性能监控
        timeit.printevery(
            num_prints=cfg.collector.total_frames // log_freq,
            total_count=cfg.collector.total_frames,
            erase=True,
        )
        
        # 🔄 等待初始随机数据收集
        torchrl_logger.info(f"等待收集初始{init_random_frames}帧随机数据...")
        while len(replay_buffer) < init_random_frames:
            time.sleep(0.1)
            current_frames = len(replay_buffer)
            if current_frames > collected_frames:
                pbar.update(current_frames - collected_frames)
                collected_frames = current_frames
        torchrl_logger.info(f"初始数据收集完成，开始训练...")
        
        # 🔥 主训练循环：基于总帧数，而非迭代collector
        while collected_frames < cfg.collector.total_frames:
            # 1️⃣ 检查是否有足够数据进行训练
            if len(replay_buffer) >= batch_size * num_updates:
                with timeit("train"):
                    batch_losses = []
                    
                    # 批量更新
                    for _ in range(num_updates):
                        with timeit("rb_sample"):
                            sampled_tensordict = replay_buffer.sample()
                            if sampled_tensordict.device != train_device:
                                sampled_tensordict = sampled_tensordict.to(train_device, non_blocking=True)
                        
                        with timeit("update"):
                            # 标记CUDA图开始（如果启用）
                            if cfg.compile.enable and cfg.compile.cudagraph:
                                torch.compiler.cudagraph_mark_step_begin()
                            
                            # 执行更新
                            loss_out = update_fn(sampled_tensordict)
                            batch_losses.append(loss_out.detach())
                        
                        with timeit("rb_update_priority"):
                            # 更新优先级（如果使用PER）
                            if cfg.replay_buffer.get('prb', True):
                                # 使用TD误差作为优先级
                                td_error = loss_out.get("td_error", loss_out["loss_qvalue"])
                                priority = td_error.abs().squeeze().clamp(min=1e-8)
                                replay_buffer.update_priority(
                                    sampled_tensordict["index"],
                                    priority.expand(batch_size)
                                )
                        
                        update_counter += 1
                    
                    # 记录批次损失
                    if batch_losses:
                        losses.extend(batch_losses)
            
            # 2️⃣ 周期性同步策略权重到收集器
            if update_counter > 0 and update_counter % update_freq == 0:
                with timeit("sync_weights"):
                    collector.update_policy_weights_()
                    torchrl_logger.info(f"同步策略权重 (update {update_counter})")
            
            # 3️⃣ 更新收集进度
            current_frames = len(replay_buffer)
            if current_frames > collected_frames:
                pbar.update(current_frames - collected_frames)
                collected_frames = current_frames
            
            # 4️⃣ 周期性记录日志
            if collected_frames - last_log_frame >= log_freq:
                log_info = {}
                
                # 计算平均损失
                if losses:
                    recent_losses = losses[-100:]  # 最近100次更新的损失
                    losses_tensor = torch.stack([
                        l.select("loss_actor", "loss_qvalue", "loss_alpha")
                        for l in recent_losses
                    ])
                    losses_mean = losses_tensor.mean(0)
                    
                    log_info.update({
                        "train/q_loss": losses_mean["loss_qvalue"].item(),
                        "train/actor_loss": losses_mean["loss_actor"].item(),
                        "train/alpha_loss": losses_mean["loss_alpha"].item(),
                        "train/alpha": recent_losses[-1].get("alpha", 0).item(),
                        "train/entropy": recent_losses[-1].get("entropy", 0).item(),
                    })
                
                # 添加训练进度信息
                log_info.update({
                    "train/collected_frames": collected_frames,
                    "train/updates": update_counter,
                    "train/buffer_size": len(replay_buffer),
                    "train/fps": collected_frames / (time.time() - start_time),
                })
                
                # 添加性能指标
                log_info.update(timeit.todict(prefix="time/"))
                
                # 记录到logger
                if logger:
                    for key, value in log_info.items():
                        logger.log_scalar(key, value, step=collected_frames)
                
                torchrl_logger.info(f"[{collected_frames:,}] "
                                   f"Loss: A={log_info.get('train/actor_loss', 0):.4f} "
                                   f"Q={log_info.get('train/q_loss', 0):.4f} "
                                   f"α={log_info.get('train/alpha', 0):.4f} "
                                   f"FPS={log_info.get('train/fps', 0):.1f}")
                
                last_log_frame = collected_frames
            
            # 5️⃣ 周期性保存模型
            if collected_frames - last_checkpoint_frame >= test_interval:
                model_name = f"{collected_frames // 1000:05d}"
                save_path = ckpt_path / f"t[{model_name}].pt"
                torch.save(actor_critic, save_path)
                torchrl_logger.info(f"保存模型: {save_path}")
                last_checkpoint_frame = collected_frames
            
            # 6️⃣ 避免CPU空转
            if len(replay_buffer) < batch_size:
                time.sleep(0.01)
        
        # ============ 训练结束 ============
        collector.shutdown()
        pbar.close()
        
        end_time = time.time()
        execution_time = end_time - start_time
        torchrl_logger.info(f"训练完成，耗时: {execution_time:.2f}秒")
        torchrl_logger.info(f"平均FPS: {collected_frames / execution_time:.2f}")
        
        # 记录最终统计
        if logger:
            final_metrics = {
                "final/total_time": execution_time,
                "final/avg_fps": collected_frames / execution_time,
                "final/total_frames": collected_frames,
                "final/total_updates": update_counter,
            }
            for key, value in final_metrics.items():
                logger.log_scalar(key, value, step=collected_frames)


if __name__ == "__main__":
    main()
```

## 🔑 关键改动点

### 1. 移除错误的collector迭代
```python
# ❌ 删除这个
for i, data in enumerate(collector):
    replay_buffer.extend(data)

# ✅ 改为
while collected_frames < total_frames:
    # 直接从replay_buffer采样
    if len(replay_buffer) >= batch_size:
        sampled_tensordict = replay_buffer.sample()
```

### 2. 正确的数据流
```
Collector进程 → 自动写入 → Replay Buffer ← 采样 ← 训练进程
     ↑                                                    ↓
     └────────────── update_policy_weights_() ──────────┘
```

### 3. 添加权重同步频率配置
```yaml
# config.yaml
collector:
  update_freq: 1000  # 每1000次更新同步一次权重（新增）
```

### 4. 使用len(replay_buffer)获取进度
```python
collected_frames = len(replay_buffer)  # 而不是从data累加
```

### 5. 优先级更新修复
```python
# 在采样后立即更新，使用正确的index
priority = td_error.abs().squeeze().clamp(min=1e-8)
replay_buffer.update_priority(
    sampled_tensordict["index"],  # 正确的index
    priority.expand(batch_size)
)
```

## 📊 Episode统计获取方案

由于异步collector不返回data，需要其他方式获取episode统计：

### 方案A：扩展replay_buffer的metadata
```python
# 在collector写入时附加episode信息
class ExtendedReplayBuffer(TensorDictPrioritizedReplayBuffer):
    def extend(self, data):
        # 提取episode统计
        if "next" in data.keys() and "done" in data["next"]:
            done_mask = data["next", "done"]
            if done_mask.any():
                episode_rewards = data["next", "episode_reward"][done_mask]
                self._episode_stats.append({
                    "reward": episode_rewards.mean().item(),
                    "length": data["next", "step_count"][done_mask].mean().item(),
                })
        super().extend(data)
```

### 方案B：使用独立的评估环境
```python
# 周期性运行评估获取准确的episode统计
if update_counter % eval_freq == 0:
    with torch.no_grad():
        eval_rollout = eval_env.rollout(max_steps, actor)
        episode_reward = eval_rollout["next", "reward"].sum().item()
```

## 🚀 预期效果

1. **训练速度提升40-60%**：真正的异步训练
2. **GPU利用率>95%**：消除了阻塞等待
3. **内存使用稳定**：避免重复存储数据
4. **代码逻辑清晰**：符合异步设计原则

## ⚠️ 注意事项

1. **不要迭代异步collector**：它在后台运行
2. **确保extend_buffer=True**：让collector自动写入
3. **使用shared=True的replay_buffer**：进程间共享
4. **合理设置update_freq**：平衡同步开销和策略更新