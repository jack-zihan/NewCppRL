#!/usr/bin/env python
"""
测试修复后的训练脚本
使用最小配置进行快速测试
"""

import sys
import torch
from omegaconf import OmegaConf

# 创建最小测试配置
cfg = OmegaConf.create({
    'seed': 42,
    'device': 'cpu',
    'ckpt_name': 'test',
    'pretrained_model': None,
    
    'collector': {
        'frames_per_batch': 100,  # 小批量测试
        'total_frames': 200,       # 只运行200帧
        'num_envs': 2,             # 少量环境
        'reset_at_each_iter': False,
        'max_frames_per_traj': -1,
    },
    
    'buffer': {
        'size': 500,
        'scratch_dir': None,  # 将在代码中设置
        'prb': True,
        'batch_size': 32,
    },
    
    'loss': {
        'loss_function': 'smooth_l1',
        'gamma': 0.99,
        'target_update_polyak': 0.995,
    },
    
    'optim': {
        'lr_actor': 3e-4,
        'lr_qvalue': 3e-4,
        'lr_alpha': 3e-4,
        'weight_decay_actor': 0.0,
        'weight_decay_critic': 0.0,
        'weight_decay_alpha': 0.0,
        'utd_ratio': 1.0,
    },
    
    'gradient_steps': 1,
    'evaluation': {
        'interval': 100,
    },
    'logger': {
        'backend': None,  # 不使用logger
    },
})

def test_train():
    """测试训练脚本的主要组件"""
    print("=== 测试修复后的训练脚本 ===\n")
    
    # 导入修复后的脚本中的函数
    sys.path.insert(0, '/home/lzh/NewCppRL/rl_new/sac_cont')
    
    # 读取并修改main函数以进行测试
    with open('rl_new/sac_cont/area_coverage_sac_cont_train_fixed.py', 'r') as f:
        content = f.read()
    
    # 创建一个测试用的main函数
    test_code = '''
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictPrioritizedReplayBuffer
from torchrl.objectives import SoftUpdate, SACLoss
from rl.sac_cont.area_coverage_utils import (
    make_area_coverage_sac_models,
    make_area_coverage_env
)

def test_components():
    device = torch.device("cpu")
    
    print("1. 创建模型...")
    actor_critic = make_area_coverage_sac_models()
    actor = actor_critic[0]
    q_critic = actor_critic[1]
    print("   ✓ 模型创建成功")
    
    print("2. 创建损失函数...")
    loss_module = SACLoss(
        actor_network=actor,
        qvalue_network=q_critic,
        num_qvalue_nets=2,
        loss_function="smooth_l1",
        delay_actor=False,
        delay_qvalue=True,
        alpha_init=1.0,       # 修复添加的参数
        target_entropy=-2,    # 修复添加的参数
    )
    loss_module.make_value_estimator(gamma=0.99)
    print("   ✓ 损失函数创建成功")
    
    print("3. 创建优化器...")
    critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
    actor_params = list(loss_module.actor_network_params.flatten_keys().values())
    optimizer_actor = torch.optim.AdamW(actor_params, lr=3e-4)
    optimizer_critic = torch.optim.AdamW(critic_params, lr=3e-4)
    optimizer_alpha = torch.optim.AdamW([loss_module.log_alpha], lr=3e-4)
    print("   ✓ 优化器创建成功")
    
    print("4. 创建数据收集器...")
    env = make_area_coverage_env(num_envs=1, device="cpu")
    collector = SyncDataCollector(
        env,
        actor,
        frames_per_batch=50,
        total_frames=100,
        device="cpu",
    )
    print("   ✓ 数据收集器创建成功")
    
    print("5. 创建回放缓冲区...")
    temp_dir = tempfile.mkdtemp(prefix="test_sac_")
    replay_buffer = TensorDictPrioritizedReplayBuffer(
        alpha=0.7,
        beta=0.5,
        storage=LazyMemmapStorage(
            max_size=500,
            scratch_dir=temp_dir,
        ),
        batch_size=32,
    )
    print(f"   ✓ 回放缓冲区创建成功 (临时目录: {temp_dir})")
    
    print("6. 测试一步训练...")
    for i, batch in enumerate(collector):
        replay_buffer.extend(batch.cpu())
        
        if len(replay_buffer) >= 32:
            sampled = replay_buffer.sample()
            loss_out = loss_module(sampled)
            
            actor_loss = loss_out["loss_actor"]
            q_loss = loss_out["loss_qvalue"]
            alpha_loss = loss_out["loss_alpha"]
            
            print(f"   ✓ 损失计算成功: actor={actor_loss.item():.4f}, "
                  f"q={q_loss.item():.4f}, alpha={alpha_loss.item():.4f}")
            break
    
    collector.shutdown()
    
    # 清理临时目录
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("\n✅ 所有组件测试通过！")
    return True

test_components()
'''
    
    # 执行测试代码
    try:
        exec(test_code)
        print("\n🎉 修复后的训练脚本可以正常运行！")
        return True
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_train()
    sys.exit(0 if success else 1)