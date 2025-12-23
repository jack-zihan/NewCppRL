"""REACT算法调试脚本 - 诊断agent初始化位置问题"""
import sys
sys.path.insert(0, "/home/lzh/NewCppRL")

import gymnasium as gym
import numpy as np
import envs_new

env = gym.make("NewPasture-v2", render_mode="rgb_array")
obs, _ = env.reset(seed=42)

base = getattr(env, "unwrapped", env)
field = base.maps_dict.get("field")
agent = base.agent

# 找田地内的所有点
ys, xs = np.nonzero(field > 0)
print(f"Field non-zero pixels: {len(xs)}")
print(f"Field bbox: x=[{xs.min()}, {xs.max()}], y=[{ys.min()}, {ys.max()}]")

# Agent位置
ax, ay = agent.x, agent.y
print(f"\nAgent position: ({ax:.1f}, {ay:.1f})")

# 检查agent是否在field内
in_field = field[int(ay), int(ax)] > 0 if (0 <= int(ax) < field.shape[1] and 0 <= int(ay) < field.shape[0]) else False
print(f"Agent in field: {in_field}")

# 如果不在field内，找最近的field点
if not in_field:
    distances = np.sqrt((xs - ax)**2 + (ys - ay)**2)
    nearest_idx = np.argmin(distances)
    nearest_x, nearest_y = xs[nearest_idx], ys[nearest_idx]
    print(f"Nearest field point: ({nearest_x}, {nearest_y}), distance={distances[nearest_idx]:.1f}")

# 打印field mask局部
print(f"\n--- Field mask (rows {int(ay)-5}-{int(ay)+5}, cols {int(ax)-5}-{int(ax)+10}) ---")
for y in range(max(0, int(ay)-5), min(field.shape[0], int(ay)+6)):
    row = []
    for x in range(max(0, int(ax)-5), min(field.shape[1], int(ax)+11)):
        if x == int(ax) and y == int(ay):
            row.append('A')  # Agent position
        elif field[y, x] > 0:
            row.append('#')
        else:
            row.append('.')
    print(f"y={y:3d}: {''.join(row)}")

env.close()
