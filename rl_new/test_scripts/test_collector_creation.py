#!/usr/bin/env python
"""测试数据收集器创建"""

import torch
import sys

print("=== 测试数据收集器创建 ===")

try:
    from torchrl.collectors import MultiaSyncDataCollector
    from rl.sac_cont.area_coverage_utils import (
        make_area_coverage_sac_models,
        make_area_coverage_env
    )
    
    print("1. 创建模型...")
    actor_critic = make_area_coverage_sac_models()
    actor = actor_critic[0]
    print("✓ 模型创建成功")
    
    print("\n2. 创建数据收集器...")
    
    # 简化的收集器配置
    num_envs = 2
    frames_per_batch = 100
    total_frames = 200
    
    print(f"配置: num_envs={num_envs}, frames_per_batch={frames_per_batch}")
    
    # 尝试创建收集器
    collector = MultiaSyncDataCollector(
        create_env_fn=[lambda: make_area_coverage_env(
            num_envs=1,
            device='cpu',
        ) for _ in range(num_envs)],
        policy=actor,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device='cpu',
        storing_device='cpu',
        max_frames_per_traj=-1,
        reset_at_each_iter=False,
    )
    
    print("✓ 收集器创建成功")
    
    print("\n3. 测试收集一批数据...")
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("数据收集超时")
    
    # 设置30秒超时
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)
    
    # 收集一批数据
    for i, batch in enumerate(collector):
        print(f"✓ 成功收集批次 {i}: {batch.numel()} 帧")
        print(f"  批次键: {list(batch.keys())}")
        
        # 只测试一批
        break
    
    # 取消超时
    signal.alarm(0)
    
    print("\n4. 关闭收集器...")
    collector.shutdown()
    print("✓ 收集器关闭成功")
    
    print("\n🎉 数据收集器测试通过！")
    
except TimeoutError as e:
    print(f"\n⏱️ {e}")
    print("数据收集可能卡在环境步进中")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)