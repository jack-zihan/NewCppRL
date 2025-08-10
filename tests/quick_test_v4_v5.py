"""
快速测试Pasture-v4和Pasture-v5环境
"""

import gymnasium as gym
import envs

# 测试Pasture-v4（无多尺度）
print("Testing Pasture-v4:")
env = gym.make("Pasture-v4")
obs, info = env.reset()
print(f"  Observation shape: {obs['observation'].shape}")  # (4, 128, 128)
print(f"  Initial coverage: {info['coverage_rate']:.2%}")

for i in range(5):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f"  Step {i}: reward={reward:.4f}, coverage={info['coverage_rate']:.2%}")
    if done:
        break
env.close()

print("\nTesting Pasture-v5:")
# 测试Pasture-v5（有多尺度）  
env = gym.make("Pasture-v5")
obs, info = env.reset()
print(f"  Observation shape: {obs['observation'].shape}")  # (20, 16, 16)
print(f"  Initial coverage: {info['coverage_rate']:.2%}")

for i in range(5):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f"  Step {i}: reward={reward:.4f}, coverage={info['coverage_rate']:.2%}")
    if done:
        break
env.close()

print("\n✅ Both environments work correctly!")