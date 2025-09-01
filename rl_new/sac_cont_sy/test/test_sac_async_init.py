#!/usr/bin/env python3
"""
测试sac-async.py的初始化阶段
逐步检查每个组件的初始化是否正确
"""

from __future__ import annotations

import os
import sys
import time
import tempfile
import warnings
from pathlib import Path
from functools import partial

import hydra
import numpy as np
import tensordict
import torch
import torch.cuda
from omegaconf import DictConfig, OmegaConf

from tensordict import TensorDict
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import aSyncDataCollector
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import SoftUpdate, SACLoss, group_optimizers
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.record.loggers import get_logger

# 导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from rl_new.sac_cont_sy.model_utils import make_sac_models
from rl_new.sac_cont_sy.sac_utils import (setup_devices, create_update_fn, flatten, get_actor_actions,
                                          generate_exp_name, evaluate_policy_parallel as evaluate_policy, CheckpointManager)
from rl_new.sac_cont_sy.env_utils import make_train_environment, make_environment


torch.set_float32_matmul_precision("high")
tensordict.nn.functional_modules._exclude_td_from_pytree().set()


def test_initialization():
    """测试初始化阶段"""
    
    print("=" * 80)
    print("SAC-Async 初始化阶段测试")
    print("=" * 80)
    
    # 创建测试配置
    cfg = OmegaConf.create({
        "seed": 42,
        "in_server": False,
        "pretrained_model": None,
        
        "env": {
            "env_id": "NewPasture-v2",
            "env_kwargs": None,  # 测试None值处理
            "seed": 42
        },
        
        "collector": {
            "total_frames": 1000000,
            "frames_per_batch": 20,
            "init_random_frames": 10000,
            "update_freq": 10000,
            "env_per_collector": 4
        },
        
        "buffer": {
            "buffer_size": 1000000,
            "batch_size": 256,
            "temp_dir": None
        },
        
        "compile": {
            "enable": False,
            "mode": None,
            "cudagraphs": False
        },
        
        "logger": {
            "backend": None,  # 不使用logger避免复杂性
            "model_name": "sac",
            "exp_name": "test",
            "test_ckpt_num": 3,
            "log_freq": 10000,
            "video": False,
            "eval_episodes": 4,
            "eval_max_steps": 100,
            "eval_video_skip": 10
        },
        
        "optim": {
            "gamma": 0.99,
            "alpha_init": 0.01,
            "lr_actor": 3e-4,
            "lr_critic": 3e-4,
            "lr_alpha": 3e-4,
            "weight_decay_actor": 0.0,
            "weight_decay_critic": 0.0,
            "weight_decay_alpha": 0.0,
            "eps_actor": 1e-8,
            "eps_critic": 1e-8,
            "polyak": 0.995
        },
        
        "training": {
            "use_amp": False
        }
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        print("\n1. 基础设置...")
        print("-" * 40)
        
        # 实验名称
        exp_name = generate_exp_name(cfg.logger.model_name, cfg.logger.exp_name)
        print(f"✓ 实验名称: {exp_name}")
        
        # 设备配置
        train_device = torch.device("cpu")
        collector_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        print(f"✓ 训练设备: {train_device}, 收集设备: {collector_device}")
        
        # 设置随机种子
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        print(f"✓ 随机种子: {cfg.seed}")
        
        # 编译模式
        compile_mode = None
        compile_mode_collector = False
        print(f"✓ 编译模式: {compile_mode}")
        
        print("\n2. 创建日志记录器和checkpoint管理器...")
        print("-" * 40)
        
        # 不使用logger避免复杂性
        logger = None
        print("✓ Logger: None (测试模式)")
        
        # checkpoint管理器
        checkpoint_dir = Path.cwd() / "test_checkpoints"
        checkpoint_manager = CheckpointManager(save_dir=checkpoint_dir, max_checkpoints=cfg.logger.test_ckpt_num)
        print(f"✓ Checkpoint目录: {checkpoint_dir}")
        
        print("\n3. 创建模型...")
        print("-" * 40)
        
        # 创建dummy环境用于模型初始化（与官方一致）
        print(f"创建环境: {cfg.env.env_id}")
        dummy_env = make_train_environment(cfg, device="cpu")
        print(f"✓ Dummy环境创建成功")
        
        # 在正确设备上创建模型（使用新的device参数）
        actor_critic = make_sac_models(dummy_env, device=train_device)
        print(f"✓ 训练模型创建成功 (设备: {train_device})")
        
        exploration_actor_critic = make_sac_models(dummy_env, device=collector_device)
        print(f"✓ 探索模型创建成功 (设备: {collector_device})")
        
        # 同步权重
        exploration_actor_critic[0].load_state_dict(actor_critic[0].state_dict())
        exploration_policy = exploration_actor_critic[0]
        print(f"✓ 权重同步成功")
        
        # 验证前向传播
        with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
            test_td = dummy_env.reset().to(train_device)
            test_td = actor_critic[0](test_td)  # actor
            print(f"✓ Actor前向传播成功，动作形状: {test_td['action'].shape}")
            test_td = actor_critic[1](test_td)  # critic
            print(f"✓ Critic前向传播成功，Q值形状: {test_td['state_action_value'].shape}")
        
        # 清理
        dummy_env.close()
        del dummy_env
        print("✓ Dummy环境清理完成")
        
        print("\n4. 创建回放缓冲区...")
        print("-" * 40)
        
        replay_buffer = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(
                max_size=cfg.buffer.buffer_size,
                scratch_dir=tmpdir,
            ),
            batch_size=cfg.buffer.batch_size,
        )
        replay_buffer.append_transform(lambda td: td.to(train_device))
        replay_buffer.empty()
        print(f"✓ 回放缓冲区创建成功，容量: {cfg.buffer.buffer_size}")
        
        print("\n5. 创建收集器（异步模式）...")
        print("-" * 40)
        
        try:
            collector = aSyncDataCollector(
                partial(make_train_environment, cfg),
                exploration_policy,
                init_random_frames=0,  # 异步模式不支持
                frames_per_batch=cfg.collector.frames_per_batch,
                total_frames=cfg.collector.total_frames,
                device=collector_device,
                env_device=torch.device("cpu"),
                compile_policy=False,
                cudagraph_policy=False,
                replay_buffer=replay_buffer,
                extend_buffer=True,
                postproc=flatten,
                no_cuda_sync=True,
                max_frames_per_traj=-1,
            )
            collector.set_seed(cfg.seed)
            print(f"✓ 异步收集器创建成功")
            
            # 启动收集器
            print("启动收集器...")
            collector.start()
            print(f"✓ 收集器启动成功")
            
            # 等待一些数据
            print("等待收集初始数据...")
            time.sleep(2)
            print(f"✓ 已收集 {replay_buffer.write_count} 帧数据")
            
            # 停止收集器
            print("停止收集器...")
            collector.shutdown()
            print(f"✓ 收集器停止成功")
            
        except Exception as e:
            print(f"✗ 收集器创建/启动失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("\n6. 创建损失函数和优化器...")
        print("-" * 40)
        
        # 创建损失函数
        loss_module = SACLoss(
            actor_network=actor_critic[0],
            qvalue_network=actor_critic[1],
            num_qvalue_nets=2,
            loss_function="l2",
            delay_actor=False,
            delay_qvalue=True,
            alpha_init=cfg.optim.alpha_init,
        )
        loss_module.make_value_estimator(gamma=cfg.optim.gamma)
        print(f"✓ 损失函数创建成功")
        
        # 创建目标网络更新器
        target_net_updater = SoftUpdate(loss_module, eps=cfg.optim.polyak)
        print(f"✓ 目标网络更新器创建成功")
        
        # 创建优化器
        critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
        actor_params = list(loss_module.actor_network_params.flatten_keys().values())
        
        optimizer_actor = torch.optim.AdamW(
            actor_params, lr=cfg.optim.lr_actor, weight_decay=cfg.optim.weight_decay_actor, eps=cfg.optim.eps_actor)
        optimizer_critic = torch.optim.AdamW(
            critic_params, lr=cfg.optim.lr_critic, weight_decay=cfg.optim.weight_decay_critic, eps=cfg.optim.eps_critic)
        optimizer_alpha = torch.optim.AdamW(
            [loss_module.log_alpha], lr=cfg.optim.lr_alpha, weight_decay=cfg.optim.weight_decay_alpha)
        
        optimizer = group_optimizers(optimizer_actor, optimizer_critic, optimizer_alpha)
        print(f"✓ 优化器创建成功")
        
        # 创建优化函数
        update_fn = create_update_fn(loss_module, optimizer, target_net_updater, cfg, compile_mode, None)
        print(f"✓ 优化函数创建成功")
        
        print("\n" + "=" * 80)
        print("初始化阶段测试完成！")
        print("=" * 80)
        
        return True


if __name__ == "__main__":
    success = test_initialization()
    sys.exit(0 if success else 1)