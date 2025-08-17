#!/usr/bin/env python
"""测试使用SyncDataCollector避免多进程问题"""

import torch
import sys

print("=== 测试SyncDataCollector ===")

try:
    from torchrl.collectors import SyncDataCollector
    from torchrl.data import LazyMemmapStorage, TensorDictPrioritizedReplayBuffer
    from torchrl.objectives import SACLoss, SoftUpdate
    from rl.sac_cont.area_coverage_utils import (
        make_area_coverage_sac_models,
        make_area_coverage_env
    )
    
    print("1. 创建环境和模型...")
    env = make_area_coverage_env(num_envs=1, device='cpu')
    actor_critic = make_area_coverage_sac_models()
    actor = actor_critic[0]
    qvalue = actor_critic[1]
    print("✓ 环境和模型创建成功")
    
    print("\n2. 创建损失函数...")
    loss_module = SACLoss(
        actor_network=actor,
        qvalue_network=qvalue,
        num_qvalue_nets=2,
        loss_function="smooth_l1",
        alpha_init=1.0,
        target_entropy=-2,
    )
    target_net_updater = SoftUpdate(loss_module, tau=0.005)
    print("✓ 损失函数创建成功")
    
    print("\n3. 创建优化器...")
    optimizer_actor = torch.optim.Adam(
        loss_module.actor_network_params.values(True, True),
        lr=3e-4,
    )
    optimizer_qvalue = torch.optim.Adam(
        loss_module.qvalue_network_params.values(True, True),
        lr=3e-4,
    )
    optimizer_alpha = torch.optim.Adam(
        [loss_module.log_alpha],
        lr=3e-4,
    )
    print("✓ 优化器创建成功")
    
    print("\n4. 创建SyncDataCollector...")
    collector = SyncDataCollector(
        env,
        actor,
        frames_per_batch=100,
        total_frames=300,
        device='cpu',
        storing_device='cpu',
        max_frames_per_traj=-1,
    )
    print("✓ SyncDataCollector创建成功")
    
    print("\n5. 创建回放缓冲区...")
    import tempfile
    import os
    # 创建唯一的临时目录
    temp_dir = tempfile.mkdtemp(prefix="torchrl_test_")
    replay_buffer = TensorDictPrioritizedReplayBuffer(
        alpha=0.6,
        beta=0.4,
        storage=LazyMemmapStorage(
            max_size=1000,
            scratch_dir=temp_dir,
        ),
        batch_size=32,
        pin_memory=False,
    )
    print(f"✓ 回放缓冲区创建成功 (临时目录: {temp_dir})")
    
    print("\n6. 测试训练循环...")
    collected_frames = 0
    for i, batch in enumerate(collector):
        print(f"批次 {i}: 收集了 {batch.numel()} 帧")
        
        # 将数据加入回放缓冲区
        replay_buffer.extend(batch.cpu())
        collected_frames += batch.numel()
        
        # 执行几个梯度步骤
        if len(replay_buffer) >= 32:
            for _ in range(2):  # 执行2个梯度步骤
                # 从回放缓冲区采样
                sampled_tensordict = replay_buffer.sample()
                
                # 计算损失
                loss_td = loss_module(sampled_tensordict)
                
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
                
                # 更新优先级 - TorchRL 0.9.2需要显式传入priority
                # 暂时注释掉，后续需要从loss_td中获取priority
                # replay_buffer.update_priority(sampled_tensordict, priority)
            
            print(f"  梯度更新: actor_loss={loss_actor.item():.4f}, "
                  f"q_loss={loss_qvalue.item():.4f}, alpha_loss={loss_alpha.item():.4f}")
        
        # 测试2批次
        if i >= 1:
            print(f"\n✅ 训练循环测试成功！共收集 {collected_frames} 帧")
            break
    
    collector.shutdown()
    print("\n🎉 所有测试通过！SyncDataCollector工作正常")
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)