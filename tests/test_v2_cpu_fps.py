"""测试v2环境使用CPU APF的帧率"""
import time
from envs_new.cpp_env_v2 import CppEnv

print("测试v2环境 - CPU APF版本")

env = CppEnv()
obs, info = env.reset(seed=42)

# 测试1000步
num_steps = 1000
print(f"运行 {num_steps} 步...")

start = time.time()
for i in range(num_steps):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
        
    if (i + 1) % 100 == 0:
        print(f"  步数: {i + 1}/{num_steps}")

elapsed = time.time() - start
fps = num_steps / elapsed

print(f"\n结果:")
print(f"总时间: {elapsed:.2f}s")
print(f"FPS: {fps:.1f}")
print(f"每步: {elapsed/num_steps*1000:.2f}ms")

env.close()