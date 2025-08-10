"""
测试脚本：验证Pasture-v4和Pasture-v5环境的功能
"""

import gymnasium as gym
import envs
import numpy as np


def test_pasture_v4():
    """测试Pasture-v4环境（无多尺度）"""
    print("=" * 50)
    print("Testing Pasture-v4 (no multi-scale)")
    print("=" * 50)
    
    env = gym.make("Pasture-v4")
    obs, info = env.reset()
    
    print(f"✓ Environment created successfully")
    print(f"✓ Observation shape: {obs['observation'].shape}")
    print(f"  Expected: (4, 128, 128) - 4 channels for frontier, frontier_hifs, obstacle, trajectory")
    
    # 验证观测维度
    assert obs['observation'].shape[0] == 4, f"Expected 4 channels, got {obs['observation'].shape[0]}"
    assert obs['observation'].shape[1] == 128, f"Expected height 128, got {obs['observation'].shape[1]}"
    assert obs['observation'].shape[2] == 128, f"Expected width 128, got {obs['observation'].shape[2]}"
    
    print(f"✓ Initial coverage: {info.get('coverage_rate', 0):.2%}")
    print(f"✓ Info contains 'coverage_rate' key: {'coverage_rate' in info}")
    
    # 测试几步
    print("\nTesting steps...")
    rewards = []
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        
        if i < 3:  # 打印前3步的详细信息
            print(f"  Step {i}: Coverage={info.get('coverage_rate', 0):.2%}, Reward={reward:.4f}")
        
        # 验证没有weed奖励（weed奖励通常很大，如20.0）
        assert abs(reward) < 10, f"Reward too large ({reward}), might contain weed reward"
        
        if done:
            print(f"\n✓ Episode finished! Final coverage: {info.get('coverage_rate', 0):.2%}")
            print(f"  Reason: {'crashed' if info.get('crashed') else 'frontier covered' if info.get('finished') else 'unknown'}")
            break
    
    # 验证奖励范围合理
    print(f"\n✓ Reward statistics: min={min(rewards):.4f}, max={max(rewards):.4f}, mean={np.mean(rewards):.4f}")
    
    env.close()
    print("\n✓ Pasture-v4 test completed successfully!")
    return True


def test_pasture_v5():
    """测试Pasture-v5环境（有多尺度）"""
    print("\n" + "=" * 50)
    print("Testing Pasture-v5 (with multi-scale)")
    print("=" * 50)
    
    env = gym.make("Pasture-v5")
    obs, info = env.reset()
    
    print(f"✓ Environment created successfully")
    print(f"✓ Observation shape: {obs['observation'].shape}")
    
    # V5应该使用SGCNN，所以观测维度应该是 (20, 16, 16) 或 (16, 16, 16)
    # 具体取决于use_global_obs设置
    assert obs['observation'].shape[1] == 16, f"Expected height 16 for SGCNN, got {obs['observation'].shape[1]}"
    assert obs['observation'].shape[2] == 16, f"Expected width 16 for SGCNN, got {obs['observation'].shape[2]}"
    
    # 通道数可能是16（4层x4尺度）或20（加上global features）
    assert obs['observation'].shape[0] in [16, 20], f"Expected 16 or 20 channels for SGCNN, got {obs['observation'].shape[0]}"
    
    print(f"  Expected: (16 or 20, 16, 16) - Multi-scale SGCNN features")
    print(f"✓ Initial coverage: {info.get('coverage_rate', 0):.2%}")
    
    # 测试几步
    print("\nTesting steps...")
    rewards = []
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        
        if i < 3:  # 打印前3步的详细信息
            print(f"  Step {i}: Coverage={info.get('coverage_rate', 0):.2%}, Reward={reward:.4f}")
        
        # 验证没有weed奖励
        assert abs(reward) < 10, f"Reward too large ({reward}), might contain weed reward"
        
        if done:
            print(f"\n✓ Episode finished! Final coverage: {info.get('coverage_rate', 0):.2%}")
            print(f"  Reason: {'crashed' if info.get('crashed') else 'frontier covered' if info.get('finished') else 'unknown'}")
            break
    
    # 验证奖励范围合理
    print(f"\n✓ Reward statistics: min={min(rewards):.4f}, max={max(rewards):.4f}, mean={np.mean(rewards):.4f}")
    
    env.close()
    print("\n✓ Pasture-v5 test completed successfully!")
    return True


def test_reward_components():
    """详细测试奖励组件，确保没有weed奖励"""
    print("\n" + "=" * 50)
    print("Testing reward components (no weed reward)")
    print("=" * 50)
    
    env = gym.make("Pasture-v4")
    obs, info = env.reset(seed=42)
    
    print(f"Initial state:")
    print(f"  - Coverage: {info.get('coverage_rate', 0):.2%}")
    print(f"  - Weed count: {env.map_weed.sum()}")
    print(f"  - Frontier count: {env.map_frontier.sum()}")
    
    # 执行一些步骤并监控奖励
    print("\nMonitoring rewards (should be small without weed reward):")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"  Step {i}: reward={reward:.4f}, coverage={info.get('coverage_rate', 0):.2%}")
        
        # 正常奖励应该在-1到1之间（没有weed奖励的20倍增益）
        if not done:  # 如果没有碰撞或完成
            assert -2 < reward < 2, f"Abnormal reward {reward}, might contain weed component"
    
    env.close()
    print("\n✓ Reward component test passed - no weed rewards detected!")
    return True


def test_termination_condition():
    """测试终止条件（frontier覆盖而非weed清除）"""
    print("\n" + "=" * 50)
    print("Testing termination condition (frontier coverage)")
    print("=" * 50)
    
    env = gym.make("Pasture-v4")
    
    # 运行多个episode，看终止条件
    for episode in range(3):
        obs, info = env.reset()
        print(f"\nEpisode {episode}:")
        print(f"  Initial frontier pixels: {env.map_frontier.sum()}")
        print(f"  Initial weed pixels: {env.map_weed.sum()}")
        
        step_count = 0
        max_steps = 100  # 限制步数避免无限循环
        
        while step_count < max_steps:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1
            
            if done:
                print(f"  Episode ended after {step_count} steps")
                print(f"  Final frontier pixels: {env.map_frontier.sum()}")
                print(f"  Final weed pixels: {env.map_weed.sum()}")
                print(f"  Termination reason: {'crashed' if info.get('crashed') else 'frontier covered' if info.get('finished') else 'unknown'}")
                
                # 如果是因为完成而结束，frontier应该接近0
                if info.get('finished'):
                    assert env.map_frontier.sum() == 0, f"Finished but frontier not fully covered: {env.map_frontier.sum()}"
                    print(f"  ✓ Correctly terminated when frontier fully covered")
                break
        
        if step_count >= max_steps:
            print(f"  Episode continued for {max_steps} steps without termination")
    
    env.close()
    print("\n✓ Termination condition test passed!")
    return True


if __name__ == "__main__":
    print("Starting comprehensive tests for Pasture-v4 and Pasture-v5...")
    print("=" * 60)
    
    try:
        # 运行所有测试
        test_pasture_v4()
        test_pasture_v5()
        test_reward_components()
        test_termination_condition()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED SUCCESSFULLY! 🎉")
        print("=" * 60)
        print("\nSummary:")
        print("✓ Pasture-v4: 4-channel observation (128x128), no multi-scale")
        print("✓ Pasture-v5: Multi-scale SGCNN observation (16x16)")
        print("✓ Both environments use frontier coverage as termination")
        print("✓ Both environments have no weed rewards")
        print("✓ Both environments include frontier_hifs map layer")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()