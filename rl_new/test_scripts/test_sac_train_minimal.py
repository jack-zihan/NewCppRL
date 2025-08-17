#!/usr/bin/env python
"""简化的SAC训练测试脚本，用于检测训练流程的兼容性问题"""

import sys
import traceback
import torch
import torch.nn as nn
from omegaconf import OmegaConf

print("=== 开始测试SAC训练流程 ===")

try:
    from torchrl.collectors import MultiaSyncDataCollector
    from torchrl.data import LazyMemmapStorage, TensorDictPrioritizedReplayBuffer
    from torchrl.objectives import SoftUpdate, SACLoss
    from torchrl.record.loggers import get_logger
    
    from rl.sac_cont.area_coverage_utils import (
        make_area_coverage_sac_models,
        make_area_coverage_env
    )
    
    print("✓ 所有模块导入成功")
    
    # 创建最小配置
    cfg = OmegaConf.create({
        'device': 'cpu',
        'seed': 42,
        'num_envs': 2,  # 测试用少量环境
        'collector': {
            'frames_per_batch': 100,  # 每批收集少量帧用于测试
            'total_frames': 500,       # 总共运行少量帧
        },
        'replay_buffer': {
            'size': 1000,
            'prb': True,
            'alpha': 0.6,
            'beta': 0.4,
        },
        'optim': {
            'lr': 3e-4,
            'weight_decay': 0.0,
            'batch_size': 32,  # 小批量用于测试
            'utd_ratio': 1.0,
        },
        'loss': {
            'gamma': 0.99,
            'tau': 0.005,
            'alpha_init': 1.0,
            'learnable_alpha': True,
            'target_entropy': 'auto',
        },
        'gradient_steps': 5,  # 每次收集后的梯度步数
    })
    
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    
    print("\n=== 创建SAC模型 ===")
    # 新版本不需要传入环境，函数内部会创建
    actor_critic = make_area_coverage_sac_models()
    actor = actor_critic[0]  # 第一个是policy
    qvalue = actor_critic[1]  # 第二个是qvalue
    print(f"✓ Actor模型: {actor}")
    print(f"✓ Q-value模型: {qvalue}")
    
    print("\n=== 创建损失函数 ===")
    # 创建SAC损失函数
    loss_kwargs = {
        "actor_network": actor,
        "qvalue_network": qvalue,
        "num_qvalue_nets": 2,
        "loss_function": "smooth_l1",
        "alpha_init": cfg.loss.alpha_init,
    }
    
    # 处理可学习的alpha
    if cfg.loss.learnable_alpha:
        loss_kwargs["alpha_init"] = torch.tensor(cfg.loss.alpha_init, dtype=torch.float32, device=device)
        
    # 处理目标熵 (SAC的action空间是2维)
    if cfg.loss.target_entropy == "auto":
        loss_kwargs["target_entropy"] = -2  # area_coverage环境有2个动作维度
    else:
        loss_kwargs["target_entropy"] = cfg.loss.target_entropy
        
    loss_module = SACLoss(**loss_kwargs)
    
    # 创建目标网络更新器
    target_net_updater = SoftUpdate(loss_module, tau=cfg.loss.tau)
    
    print(f"✓ SAC损失函数创建成功")
    
    print("\n=== 创建优化器 ===")
    optimizer_actor = torch.optim.Adam(
        loss_module.actor_network_params.values(True, True),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )
    optimizer_qvalue = torch.optim.Adam(
        loss_module.qvalue_network_params.values(True, True),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )
    optimizer_alpha = torch.optim.Adam(
        [loss_module.log_alpha],
        lr=3e-4,
    )
    print("✓ 优化器创建成功")
    
    print("\n=== 创建数据收集器 ===")
    collector = MultiaSyncDataCollector(
        create_env_fn=[lambda: make_area_coverage_env(
            num_envs=1,
            device='cpu',
        ) for _ in range(cfg.num_envs)],
        policy=actor,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device='cpu',
        storing_device='cpu',
        max_frames_per_traj=-1,
        reset_at_each_iter=False,
    )
    print(f"✓ 数据收集器创建成功")
    
    print("\n=== 创建回放缓冲区 ===")
    replay_buffer = TensorDictPrioritizedReplayBuffer(
        cfg.replay_buffer.alpha,
        cfg.replay_buffer.beta,
        storage=LazyMemmapStorage(
            cfg.replay_buffer.size,
            scratch_dir="/tmp",
        ),
        batch_size=cfg.optim.batch_size,
        pin_memory=False,
    )
    print(f"✓ 回放缓冲区创建成功")
    
    print("\n=== 开始测试训练循环 ===")
    collected_frames = 0
    for i, batch in enumerate(collector):
        print(f"批次 {i}: 收集了 {batch.numel()} 帧")
        
        # 将数据加入回放缓冲区
        replay_buffer.extend(batch.cpu())
        collected_frames += batch.numel()
        
        # 执行几个梯度步骤
        if len(replay_buffer) >= cfg.optim.batch_size:
            for _ in range(min(cfg.gradient_steps, 2)):  # 最多2步用于测试
                # 从回放缓冲区采样
                sampled_tensordict = replay_buffer.sample()
                
                # 计算损失
                loss_td = loss_module(sampled_tensordict.to(device))
                
                # Actor损失
                loss_actor = loss_td["loss_actor"]
                optimizer_actor.zero_grad()
                loss_actor.backward()
                optimizer_actor.step()
                
                # Q-value损失
                loss_qvalue = loss_td["loss_qvalue"]
                optimizer_qvalue.zero_grad()
                loss_qvalue.backward()
                optimizer_qvalue.step()
                
                # Alpha损失
                loss_alpha = loss_td["loss_alpha"]
                optimizer_alpha.zero_grad()
                loss_alpha.backward()
                optimizer_alpha.step()
                
                # 更新目标网络
                target_net_updater.step()
                
                # 更新优先级
                if cfg.replay_buffer.prb:
                    replay_buffer.update_priority(sampled_tensordict)
            
            print(f"  执行梯度更新: actor_loss={loss_actor.item():.4f}, "
                  f"q_loss={loss_qvalue.item():.4f}, alpha_loss={loss_alpha.item():.4f}")
        
        # 测试运行3批次
        if i >= 2:
            print(f"\n✅ 训练循环测试成功！共收集 {collected_frames} 帧")
            break
    
    collector.shutdown()
    print("\n🎉 所有测试通过！SAC训练流程兼容TorchRL 0.9.2")
    
except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    traceback.print_exc()