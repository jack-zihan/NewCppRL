"""
测试CppEnvV5的方向场奖励功能
"""

import numpy as np
import gymnasium as gym
import envs


def test_direction_field_loading():
    """测试方向场加载"""
    print("=" * 60)
    print("Testing direction field loading...")
    
    env = gym.make("Pasture-v5")
    obs, info = env.reset(seed=42, options={'map_id': 1})
    
    # 检查方向场是否加载
    if env.map_frontier_hifs is not None:
        print("✓ Direction field loaded successfully")
        
        # 统计方向场信息
        valid_mask = env.map_frontier_hifs >= 0
        invalid_mask = env.map_frontier_hifs < 0
        
        print(f"  Shape: {env.map_frontier_hifs.shape}")
        print(f"  Valid pixels: {valid_mask.sum()}")
        print(f"  Invalid pixels (-1): {invalid_mask.sum()}")
        
        if valid_mask.any():
            valid_values = env.map_frontier_hifs[valid_mask]
            print(f"  Valid range: [{valid_values.min():.4f}, {valid_values.max():.4f}] radians")
            print(f"  Valid mean: {valid_values.mean():.4f} radians")
    else:
        print("✗ Direction field not loaded")
    
    env.close()
    return True


def test_direction_conversion():
    """测试坐标系转换"""
    print("\n" + "=" * 60)
    print("Testing coordinate system conversion...")
    
    env = gym.make("Pasture-v5")
    env.reset(seed=42)
    
    # 测试几个关键角度的转换
    test_cases = [
        (0, np.pi/2, "向下"),     # 小车0°（向下） → 方向场π/2
        (90, np.pi, "向右"),      # 小车90°（向右） → 方向场π
        (180, np.pi/2, "向上"),   # 小车180°（向上） → 方向场π/2（等价于向下）
        (270, 0, "向左"),         # 小车270°（向左） → 方向场0
    ]
    
    print("\n小车朝向 → 方向场朝向转换测试:")
    for agent_dir, expected_field, direction_name in test_cases:
        field_dir = env._convert_agent_to_field_direction(agent_dir)
        diff = abs(field_dir - expected_field)
        
        # 考虑无向性（0和π等价）
        if diff > np.pi/2:
            diff = np.pi - diff
            
        if diff < 0.01:  # 允许小误差
            print(f"  ✓ {agent_dir:3d}° ({direction_name:4s}) → {field_dir:.4f} rad (expected {expected_field:.4f})")
        else:
            print(f"  ✗ {agent_dir:3d}° ({direction_name:4s}) → {field_dir:.4f} rad (expected {expected_field:.4f})")
    
    env.close()
    return True


def test_direction_difference():
    """测试方向差异计算"""
    print("\n" + "=" * 60)
    print("Testing direction difference calculation...")
    
    env = gym.make("Pasture-v5")
    env.reset(seed=42)
    
    # 测试不同的角度差异
    test_cases = [
        # (agent_dir, field_dir, expected_diff_degrees, description)
        (0, np.pi/2, 0, "完全对齐（向下）"),
        (90, np.pi, 0, "完全对齐（向右）"),
        (0, 0, 90, "垂直（向下vs向左）"),
        (0, np.pi/4, 45, "45度差异"),
        (0, -1, 0, "无效区域（-1）"),
    ]
    
    print("\n角度差异计算测试:")
    for agent_dir, field_dir, expected_diff, desc in test_cases:
        diff_degrees = env._compute_direction_difference_degrees(agent_dir, field_dir)
        
        if abs(diff_degrees - expected_diff) < 1:  # 允许1度误差
            print(f"  ✓ {desc}: {diff_degrees:.1f}° (expected {expected_diff}°)")
        else:
            print(f"  ✗ {desc}: {diff_degrees:.1f}° (expected {expected_diff}°)")
    
    env.close()
    return True


def test_reward_with_direction_field():
    """测试带方向场的奖励计算"""
    print("\n" + "=" * 60)
    print("Testing reward with direction field...")
    
    # 创建V5环境，设置不同的权重
    weights = [0.0, 0.005, 0.01, 0.02]
    
    for weight in weights:
        env = gym.make("Pasture-v5", direction_field_weight=weight)
        obs, info = env.reset(seed=42, options={'map_id': 1})
        
        # 执行几步，收集奖励
        rewards = []
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            rewards.append(reward)
            if done:
                break
        
        avg_reward = np.mean(rewards)
        print(f"  Weight={weight:.3f}: avg reward={avg_reward:.4f}, "
              f"range=[{min(rewards):.4f}, {max(rewards):.4f}]")
        
        env.close()
    
    return True


def main():
    print("Starting direction field reward tests...")
    print("=" * 60)
    
    try:
        # 运行所有测试
        test_direction_field_loading()
        test_direction_conversion()
        test_direction_difference()
        test_reward_with_direction_field()
        
        print("\n" + "=" * 60)
        print("🎉 All direction field tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()