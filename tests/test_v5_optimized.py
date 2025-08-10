"""
测试优化后的CppEnvV5实现（只重写get_extra_reward）
"""

import numpy as np
import gymnasium as gym
import envs


def test_v5_optimized():
    """测试V5优化实现"""
    print("=" * 60)
    print("Testing optimized CppEnvV5 implementation")
    print("=" * 60)
    
    try:
        # 创建V5环境
        env = gym.make("Pasture-v5", direction_field_weight=0.01)
        
        # 检查是否有direction_field_weight属性
        if hasattr(env, 'direction_field_weight'):
            print(f"✓ Direction field weight: {env.direction_field_weight}")
        
        # 检查是否有get_extra_reward方法
        if hasattr(env, 'get_extra_reward'):
            print("✓ get_extra_reward method exists")
        
        # 重置环境
        obs, info = env.reset(seed=42)
        print(f"✓ Environment reset successful")
        print(f"  Observation shape: {obs['observation'].shape}")
        
        # 执行几步
        rewards = []
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            rewards.append(reward)
            
            if i < 3:
                print(f"  Step {i}: reward={reward:.4f}")
            
            if done:
                break
        
        print(f"✓ Steps executed successfully")
        print(f"  Average reward: {np.mean(rewards):.4f}")
        print(f"  Reward range: [{min(rewards):.4f}, {max(rewards):.4f}]")
        
        env.close()
        print("\n✅ Optimized V5 implementation works correctly!")
        
        # 比较有无方向场奖励的差异
        print("\n" + "=" * 60)
        print("Comparing with and without direction field reward:")
        
        # 无方向场奖励
        env1 = gym.make("Pasture-v5", direction_field_weight=0.0)
        obs, _ = env1.reset(seed=42)
        rewards_without = []
        for _ in range(20):
            action = env1.action_space.sample()
            obs, reward, done, truncated, _ = env1.step(action)
            rewards_without.append(reward)
            if done:
                break
        env1.close()
        
        # 有方向场奖励
        env2 = gym.make("Pasture-v5", direction_field_weight=0.01)
        obs, _ = env2.reset(seed=42)
        rewards_with = []
        for _ in range(20):
            action = env2.action_space.sample()
            obs, reward, done, truncated, _ = env2.step(action)
            rewards_with.append(reward)
            if done:
                break
        env2.close()
        
        print(f"  Without direction field (weight=0.0): avg={np.mean(rewards_without):.4f}")
        print(f"  With direction field (weight=0.01): avg={np.mean(rewards_with):.4f}")
        print(f"  Difference: {np.mean(rewards_with) - np.mean(rewards_without):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_v5_optimized()