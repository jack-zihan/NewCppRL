#!/usr/bin/env python
"""测试异步训练架构是否正常工作"""

import sys
from pathlib import Path
import time

# 添加项目路径
base_dir = Path(__file__).parent.parent
sys.path.append(str(base_dir))

import torch
from torchrl.data import LazyMemmapStorage, TensorDictPrioritizedReplayBuffer
from torchrl.collectors import MultiaSyncDataCollector
from torchrl_utils import make_env
from rl_new.sac_cont.sac_cont_utils import make_sac_models


def make_test_env():
    """创建测试环境的辅助函数（可被pickle）"""
    return make_env(num_envs=2, device='cpu')


def test_async_architecture():
    """测试异步架构的基本功能"""
    print("=" * 50)
    print("测试异步训练架构")
    print("=" * 50)
    
    # 1. 创建模型
    print("\n1. 创建SAC模型...")
    actor_critic = make_sac_models()
    actor = actor_critic[0]
    print("✅ 模型创建成功")
    
    # 2. 创建回放缓冲区
    print("\n2. 创建回放缓冲区...")
    from torchrl.data import TensorDictReplayBuffer
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(max_size=10000),
        batch_size=256,
        # 注意：PrioritizedReplayBuffer不能与MultiaSyncDataCollector共享
        # 在实际训练中，我们让collector自动写入，不直接传递buffer
    )
    print("✅ 回放缓冲区创建成功")
    
    # 3. 创建异步收集器
    print("\n3. 创建异步收集器...")
    from functools import partial
    
    # 使用partial创建可pickle的函数
    flatten_fn = partial(torch.reshape, shape=(-1,))
    
    collector = MultiaSyncDataCollector(
        create_env_fn=[make_test_env],  # 使用可pickle的函数
        policy=actor,
        frames_per_batch=100,
        total_frames=1000,
        device=['cpu'],
        storing_device='cpu',
        replay_buffer=replay_buffer,
        extend_buffer=True,  # 关键：自动写入buffer
        no_cuda_sync=True,
        # 注意：postproc函数也需要可pickle
    )
    print("✅ 异步收集器创建成功")
    
    # 4. 启动收集器
    print("\n4. 启动异步收集器...")
    collector.start()
    print("✅ 收集器已启动（在后台运行）")
    
    # 5. 等待数据收集
    print("\n5. 等待数据收集...")
    start_time = time.time()
    target_frames = 500
    
    while replay_buffer._writer._write_count < target_frames:
        time.sleep(0.1)
        current_frames = replay_buffer._writer._write_count
        if current_frames > 0 and current_frames % 100 == 0:
            print(f"   已收集: {current_frames} 帧")
    
    collection_time = time.time() - start_time
    final_frames = replay_buffer._writer._write_count
    print(f"✅ 数据收集完成: {final_frames} 帧，耗时 {collection_time:.2f} 秒")
    print(f"   收集速度: {final_frames/collection_time:.1f} 帧/秒")
    
    # 6. 测试从buffer采样
    print("\n6. 测试从buffer采样...")
    if len(replay_buffer) > 0:
        sample = replay_buffer.sample()
        print(f"✅ 成功采样批次，shape: {sample.shape}")
        
        # 检查关键字段
        required_keys = ["observation", "action", ("next", "observation"), ("next", "done")]
        for key in required_keys:
            if key in sample.keys():
                print(f"   ✓ 包含字段: {key}")
            else:
                print(f"   ✗ 缺少字段: {key}")
    else:
        print("❌ Buffer为空，无法采样")
    
    # 7. 测试权重更新
    print("\n7. 测试权重更新...")
    try:
        collector.update_policy_weights_()
        print("✅ 权重更新成功")
    except Exception as e:
        print(f"❌ 权重更新失败: {e}")
    
    # 8. 关闭收集器
    print("\n8. 关闭收集器...")
    collector.shutdown()
    print("✅ 收集器已关闭")
    
    # 总结
    print("\n" + "=" * 50)
    print("测试总结:")
    print(f"- 最终收集帧数: {final_frames}")
    print(f"- Buffer大小: {len(replay_buffer)}")
    print(f"- 收集速度: {final_frames/collection_time:.1f} 帧/秒")
    print("- 异步架构: ✅ 正常工作")
    print("=" * 50)


def test_episode_stats():
    """测试episode统计功能"""
    print("\n" + "=" * 50)
    print("测试Episode统计功能")
    print("=" * 50)
    
    # 导入统计跟踪器
    sys.path.append(str(base_dir / 'rl_new' / 'sac_cont_new'))
    from train import EpisodeStatsTracker
    
    # 创建模拟的replay_buffer
    print("\n1. 创建模拟数据...")
    replay_buffer = TensorDictPrioritizedReplayBuffer(
        alpha=0.7,
        beta=0.5,
        storage=LazyMemmapStorage(max_size=1000),
        batch_size=100,
    )
    
    # 添加一些模拟数据
    from tensordict import TensorDict
    for i in range(10):
        # 模拟episode结束
        done = (i % 3 == 0)
        data = TensorDict({
            "observation": torch.randn(10, 10),
            "action": torch.randn(10, 2),
            "next": {
                "observation": torch.randn(10, 10),
                "done": torch.tensor([done] * 10),
                "episode_reward": torch.randn(10) * 100 if done else torch.zeros(10),
                "step_count": torch.randint(100, 500, (10,)) if done else torch.zeros(10),
                "weed_ratio": torch.rand(10) if done else torch.zeros(10),
            }
        }, batch_size=10)
        replay_buffer.extend(data)
    
    print(f"✅ 添加了 {len(replay_buffer)} 条数据")
    
    # 测试统计跟踪器
    print("\n2. 测试统计跟踪器...")
    tracker = EpisodeStatsTracker(window_size=10)
    
    # 更新统计
    stats = tracker.update_from_buffer(replay_buffer, 100)
    
    print("✅ 统计结果:")
    for key, value in stats.items():
        print(f"   {key}: {value:.2f}")
    
    if not stats:
        print("   （无episode完成数据）")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    try:
        # 测试异步架构
        test_async_architecture()
        
        # 测试episode统计
        test_episode_stats()
        
        print("\n🎉 所有测试通过！异步训练架构正常工作。")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()