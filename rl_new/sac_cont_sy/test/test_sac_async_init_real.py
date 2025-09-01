"""
SAC同步训练脚本 (sync version)
基于数据收集批次的同步训练模式
"""
from __future__ import annotations

import os
import sys
import time
import math
import tempfile
import warnings

from pathlib import Path
from functools import partial

import hydra
import numpy as np
import tensordict
import torch
import torch.cuda
import tqdm
import gymnasium as gym
from omegaconf import DictConfig

from tensordict import TensorDict
from tensordict.nn import CudaGraphModule
from torchrl._utils import compile_with_warmup, logger as torchrl_logger, timeit
from torchrl.collectors import MultiaSyncDataCollector, aSyncDataCollector
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import SoftUpdate, SACLoss, group_optimizers
from torchrl.data import LazyMemmapStorage, TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.record.loggers import get_logger

from rl_new.sac_cont_sy.model_utils import make_sac_models
from rl_new.sac_cont_sy.sac_utils import (setup_devices, create_update_fn, flatten, get_actor_actions,
                                          generate_exp_name, evaluate_policy_parallel as evaluate_policy, CheckpointManager)
from rl_new.sac_cont_sy.env_utils import make_train_environment, make_environment
from torchrl_utils_new.local_video_recorder import LocalVideoRecorder


torch.set_float32_matmul_precision("high")  # 提升矩阵乘法性能
tensordict.nn.functional_modules._exclude_td_from_pytree().set()


@hydra.main(version_base="1.3", config_path=".", config_name="config-async")
def main(cfg: DictConfig):
    # 处理临时目录路径
    temp_dir = cfg.buffer.temp_dir
    if temp_dir and temp_dir.startswith('~'):
        temp_dir = os.path.expanduser(temp_dir)

    with tempfile.TemporaryDirectory(dir=temp_dir) as tmpdir:
        # ============ 1. 创建实验目录和基础设置 ============
        exp_name = generate_exp_name(cfg.logger.model_name, cfg.logger.exp_name)
        # ckpt_path = Path.cwd() / 'ckpt' # 这是同步版本的遗留代码，等待同步版本上线解决
        # ckpt_path.mkdir(parents=True, exist_ok=True)

        # 设备配置
        # train_device, collector_devices = setup_devices(cfg) # 双缓冲才开启设备选择
        # torchrl_logger.info(f"训练设备: {train_device}")
        # torchrl_logger.info(f"收集设备: {collector_devices[:5]}... (共{len(collector_devices)}个)")
        train_device, collector_devices = (torch.device("cuda:0"), torch.device("cuda:1")) if cfg.in_server else (torch.device("cpu"), torch.device("cuda:0"))
        torchrl_logger.info(f"训练设备: {train_device}, 收集设备: {collector_devices}")

        # 设置随机种子
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        # 确定编译模式
        if cfg.compile.enable:
            compile_mode = (cfg.compile.mode or ("default" if cfg.compile.cudagraphs else "reduce-overhead"))
            compile_mode_collector = compile_mode
        else:
            compile_mode = None
            compile_mode_collector = False

        # ============ 2. 创建日志记录器和checkpoint管理器 ============
        logger = None
        if cfg.logger.backend:
            logger = get_logger(
                logger_type=cfg.logger.backend, experiment_name=exp_name, logger_name=exp_name, # logger_name在wandb不显示，主要影响本地存储名字
                wandb_kwargs={"mode": cfg.logger.mode, "config": dict(cfg),
                              "project": cfg.logger.project_name, "group": cfg.logger.group_name, "name": exp_name},
            )

        # 初始化checkpoint管理器 - 使用Hydra管理的工作目录，相对于Hydra输出目录
        checkpoint_dir = Path.cwd() / "checkpoints"
        checkpoint_manager = CheckpointManager(save_dir=checkpoint_dir, max_checkpoints=cfg.logger.test_ckpt_num)
        torchrl_logger.info(f"Checkpoint将保存到: {checkpoint_dir}")

        # ============ 3. 创建模型 ============
        # 使用配置中的环境ID创建模型
        if cfg.pretrained_model:
            torchrl_logger.info(f"加载预训练模型: {cfg.pretrained_model}")
            actor_critic = torch.load(cfg.pretrained_path, map_location=train_device)

            # 为探索策略创建副本
            exploration_actor_critic = torch.load(cfg.pretrained_path, map_location=collector_devices)
            exploration_policy = exploration_actor_critic[0]  # 提取 actor
        else:
            # 创建一个样本环境用于模型创建
            torchrl_logger.info(f"创建环境: {cfg.env.env_id}")

            proof_env_train = make_train_environment(cfg, device=train_device)
            actor_critic = make_sac_models(env=proof_env_train)

            # 懒加载Lazy Module并验证数据流正确
            with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
                td = proof_env_train.fake_tensordict().to(train_device)
                for net in actor_critic:
                    net(td)
            proof_env_train.close()
            del proof_env_train

            # 创建探索用的 Actor-Critic（第二个GPU）
            proof_env_explore = make_train_environment(cfg, device=collector_devices)
            exploration_actor_critic = make_sac_models(env=proof_env_explore)
            exploration_policy = exploration_actor_critic[0]  # 提取 actor 用于探索
            exploration_policy.load_state_dict(actor_critic[0].state_dict())
            proof_env_explore.close()
            del proof_env_explore

        # ============ 4. 创建回放缓冲区 ============
        replay_buffer = TensorDictReplayBuffer(
            pin_memory=cfg.buffer.pin_memory, prefetch=cfg.buffer.prefetch, shared=cfg.buffer.shared_memory,
            storage=LazyMemmapStorage(max_size=cfg.buffer.buffer_size, scratch_dir=tmpdir), # 是LazyMemmapStorage所以不需要replay_buffer.append_transform(lambda td: td.to(device))
            batch_size=cfg.buffer.batch_size)
        # 对TensorDictReplayBuffer进行懒初始化
        replay_buffer.extend(make_train_environment(cfg).rollout(1).view(-1))
        replay_buffer.empty()

        # 目前异步无法使用优先级回放，只能使用双缓存机制，这个以后再判断如何解决
        # replay_buffer = TensorDictPrioritizedReplayBuffer(
        #     alpha=0.7,
        #     beta=0.5,
        #     pin_memory=cfg.buffer.get('pin_memory', True),
        #     prefetch=cfg.buffer.get('prefetch', 3),
        #     storage=LazyMemmapStorage(
        #         max_size=cfg.buffer.buffer_size,
        #         scratch_dir=tmpdir,
        #     ),
        #     batch_size=cfg.buffer.batch_size,
        # )
        # replay_buffer.append_transform(lambda td: td.to(train_device))
        # replay_buffer.empty()

        # ============ 5. 创建收集器（异步模式） ============
        torchrl_logger.info("开始创建异步收集器...")
        torchrl_logger.info(f"  - collector_devices: {collector_devices}")
        torchrl_logger.info(f"  - frames_per_batch: {cfg.collector.frames_per_batch}")
        torchrl_logger.info(f"  - total_frames: {cfg.collector.total_frames}")
        
        collector = aSyncDataCollector(
            partial(make_train_environment, cfg),
            exploration_policy,
            init_random_frames=0,  # Currently not supported, but accounted for in script: cfg.collector.init_random_frames,
            frames_per_batch=cfg.collector.frames_per_batch,
            total_frames=cfg.collector.total_frames,
            device=collector_devices,
            env_device=torch.device("cpu"),
            compile_policy={"mode": compile_mode_collector, "warmup": 5} if compile_mode_collector else False,
            cudagraph_policy={"warmup": 20} if cfg.compile.cudagraphs else False,
            replay_buffer=replay_buffer,
            extend_buffer=True,
            postproc=flatten,
            no_cuda_sync=True,  # 放弃CPU对GPU的计算同步等待
            max_frames_per_traj=-1, # 不分割轨迹
        )
        collector.set_seed(cfg.seed)
        collector.start()
        torchrl_logger.info(f"创建收集进程 (完全异步模式，设备: {collector_devices})")

        # 目前异步无法使用优先级回放，只能使用双缓存机制，这个以后再判断如何解决
        # env_kwargs = dict(cfg.env.env_kwargs) if hasattr(cfg.env, 'env_kwargs') and cfg.env.env_kwargs else {}
        # collector = MultiaSyncDataCollector(
        #     create_env_fn=[lambda d=dev: make_sac_env(
        #         env_id=cfg.env.env_id,
        #         num_envs=cfg.collector.processes_per_gpu if 'cuda' in str(dev) else 1,
        #         device=str(d),
        #         **env_kwargs
        #     ) for dev in collector_devices],
        #     policy=actor,
        #     policy_device='cpu',
        #     frames_per_batch=cfg.collector.frames_per_batch,
        #     total_frames=cfg.collector.total_frames,
        #     device=collector_devices,
        #     storing_device='cpu',
        #     max_frames_per_traj=-1,
        #     # 不传递 replay_buffer，使用同步收集模式
        #     postproc=flatten,
        # )
        # collector.set_seed(cfg.seed)
        # torchrl_logger.info(f"创建{len(collector_devices)}个收集进程 (同步模式)")

        # ============ 6. 创建损失和优化器 ============
        loss_module = SACLoss(actor_network=actor_critic[0], qvalue_network=actor_critic[1],  # actor and qvalue
                              num_qvalue_nets=2, loss_function=cfg.loss.loss_function, alpha_init=cfg.loss.alpha_init
                              , delay_actor=False, delay_qvalue=True)
        loss_module.make_value_estimator(gamma=cfg.loss.gamma)

        # 目标网络更新器
        target_net_updater = SoftUpdate(loss_module, eps=cfg.loss.target_update_polyak)

        # 创建优化器
        critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
        actor_params = list(loss_module.actor_network_params.flatten_keys().values())

        optimizer_actor = torch.optim.AdamW(
            actor_params, lr=cfg.optim.lr_actor, weight_decay=cfg.optim.weight_decay_actor, eps=cfg.optim.eps_actor)
        optimizer_critic = torch.optim.AdamW(
            critic_params, lr=cfg.optim.lr_critic, weight_decay=cfg.optim.weight_decay_critic, eps=cfg.optim.eps_critic)
        optimizer_alpha = torch.optim.AdamW(
            [loss_module.log_alpha], lr=cfg.optim.lr_alpha, weight_decay=cfg.optim.weight_decay_alpha)

        # 使用group_optimizers合并优化器
        optimizer = group_optimizers(optimizer_actor, optimizer_critic, optimizer_alpha)
        del optimizer_actor, optimizer_critic, optimizer_alpha

        # 创建GradScaler（如果使用混合精度）
        scaler = None
        if cfg.training.use_amp and torch.cuda.is_available():
            scaler = torch.amp.GradScaler('cuda')
            torchrl_logger.info("启用混合精度训练 (AMP)")

        # 创建优化函数
        update_fn = create_update_fn(loss_module, optimizer, target_net_updater, cfg, compile_mode, scaler)
        
        # ============ 测试：初始化阶段完成 ============
        torchrl_logger.info("=" * 80)
        torchrl_logger.info("初始化阶段完成！所有组件创建成功")
        torchrl_logger.info("=" * 80)
        
        # 测试一些关键功能
        torchrl_logger.info("\n测试关键功能...")
        
        # 1. 测试replay buffer是否能收集数据
        torchrl_logger.info(f"Replay buffer当前数据量: {replay_buffer.write_count}")
        torchrl_logger.info("等待收集一些初始数据...")
        
        # 等待3秒收集数据
        time.sleep(3)
        torchrl_logger.info(f"3秒后Replay buffer数据量: {replay_buffer.write_count}")
        
        # 2. 测试能否从buffer采样
        if replay_buffer.write_count > cfg.buffer.batch_size:
            try:
                sample = replay_buffer.sample()
                torchrl_logger.info(f"✓ 成功从buffer采样，batch shape: {sample.shape}")
            except Exception as e:
                torchrl_logger.error(f"✗ 从buffer采样失败: {e}")
        else:
            torchrl_logger.warning(f"Buffer数据不足({replay_buffer.write_count} < {cfg.buffer.batch_size})，跳过采样测试")
        
        # 3. 测试update函数
        if replay_buffer.write_count > cfg.buffer.batch_size:
            try:
                with torch.no_grad():
                    sampled_td = replay_buffer.sample()
                    loss_td = update_fn(sampled_td)
                    torchrl_logger.info(f"✓ Update函数测试成功，losses: {loss_td.keys()}")
            except Exception as e:
                torchrl_logger.error(f"✗ Update函数测试失败: {e}")
        
        # 清理：关闭collector
        torchrl_logger.info("\n清理资源...")
        collector.shutdown()
        torchrl_logger.info("✓ Collector已关闭")
        
        torchrl_logger.info("\n" + "=" * 80)
        torchrl_logger.info("测试完成！初始化阶段无错误")
        torchrl_logger.info("=" * 80)
        
        # 不再执行主训练循环
        return



# 主循环部分已被删除，仅保留初始化测试
if __name__ == "__main__":
    main()
