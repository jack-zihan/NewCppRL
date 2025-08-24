#!/usr/bin/env python
"""简化的异步训练测试"""

import sys
from pathlib import Path
import time

base_dir = Path(__file__).parent.parent
sys.path.append(str(base_dir))

import torch
from torchrl.collectors import SyncDataCollector
from torchrl_utils import make_env
from rl_new.sac_cont.sac_cont_utils import make_sac_models


def test_sync_collector():
    """先测试同步收集器是否正常工作"""
    print("=" * 50)
    print("测试同步收集器")
    print("=" * 50)
    
    # 创建模型
    print("\n1. 创建模型...")
    actor_critic = make_sac_models()
    actor = actor_critic[0]
    print("✅ 模型创建成功")
    
    # 创建环境
    print("\n2. 创建环境...")
    env = make_env(num_envs=1, device='cpu')
    print(f"✅ 环境创建成功")
    print(f"   Action space: {env.action_spec}")
    print(f"   Observation space: {env.observation_spec}")
    
    # 创建同步收集器
    print("\n3. 创建同步收集器...")
    collector = SyncDataCollector(
        env,
        actor,
        frames_per_batch=50,
        total_frames=200,
        device='cpu',
    )
    print("✅ 同步收集器创建成功")
    
    # 收集数据
    print("\n4. 开始收集数据...")
    collected_frames = 0
    for i, data in enumerate(collector):
        print(f"   批次 {i+1}: shape={data.shape}, keys={list(data.keys())}")
        collected_frames += data.numel()
        
        # 检查关键字段
        if "next" in data.keys() and "done" in data["next"].keys():
            done_count = data["next", "done"].sum().item()
            print(f"   Episodes completed: {done_count}")
        
        if collected_frames >= 200:
            break
    
    print(f"\n✅ 数据收集完成，总帧数: {collected_frames}")
    collector.shutdown()


def test_async_without_buffer():
    """测试异步收集器（不使用replay_buffer）"""
    print("\n" + "=" * 50)
    print("测试异步收集器（无Buffer）")
    print("=" * 50)
    
    from torchrl.collectors import MultiaSyncDataCollector
    
    # 创建模型
    print("\n1. 创建模型...")
    actor_critic = make_sac_models()
    actor = actor_critic[0]
    print("✅ 模型创建成功")
    
    # 创建环境函数
    def make_single_env():
        return make_env(num_envs=1, device='cpu')
    
    # 创建异步收集器（不传递replay_buffer）
    print("\n2. 创建异步收集器...")
    try:
        collector = MultiaSyncDataCollector(
            create_env_fn=[make_single_env],
            policy=actor,
            frames_per_batch=50,
            total_frames=200,
            device=['cpu'],
            storing_device='cpu',
            # 不传递replay_buffer，只测试基本收集功能
        )
        collector.set_seed(42)
        print("✅ 异步收集器创建成功")
        
        # 收集数据
        print("\n3. 开始收集数据...")
        collected_frames = 0
        for i, data in enumerate(collector):
            print(f"   批次 {i+1}: shape={data.shape}")
            collected_frames += data.numel()
            
            if collected_frames >= 200:
                break
        
        print(f"\n✅ 数据收集完成，总帧数: {collected_frames}")
        collector.shutdown()
        
    except Exception as e:
        print(f"❌ 异步收集器测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 先测试同步收集器
    test_sync_collector()
    
    # 再测试异步收集器
    test_async_without_buffer()
    
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)