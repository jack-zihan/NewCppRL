#!/usr/bin/env python
"""验证env坐标系和atan2坐标系的关系"""
import sys
sys.path.insert(0, '/home/lzh/NewCppRL')

import gymnasium as gym
import numpy as np
import math
import envs_new

# 创建环境
env = gym.make('NewPasture-v2',
    state_size=(256, 256),
    state_downsize=(256, 256),
    action_type="continuous",
    use_trajectory=True,
    boundary_source="box",
    use_apf=True,
    use_mist=True,
    num_obstacles_range=(0, 0),
    render_mode='rgb_array'
)

obs, info = env.reset(seed=42)
base = env.unwrapped

print("=" * 60)
print("坐标系验证")
print("=" * 60)

# 获取初始位置和方向
x, y = base.agent.position
direction = base.agent.direction
print(f"初始位置: ({x:.2f}, {y:.2f})")
print(f"初始方向 (agent.direction): {direction:.2f}°")

# 执行纯前进动作，观察位移
action = (3.0, 0.0)  # v=3, w=0 (纯前进)
print(f"\n执行动作: v={action[0]}, w={action[1]} (纯前进)")
obs, _, _, _, _ = env.step(action)

new_x, new_y = base.agent.position
new_direction = base.agent.direction
dx, dy = new_x - x, new_y - y
print(f"新位置: ({new_x:.2f}, {new_y:.2f})")
print(f"新方向: {new_direction:.2f}°")
print(f"位移: dx={dx:.2f}, dy={dy:.2f}")

# 用atan2计算移动方向
atan2_dir = math.degrees(math.atan2(dy, dx))
print(f"atan2计算的移动方向: {atan2_dir:.2f}°")

print(f"\n比较:")
print(f"  agent.direction = {direction:.2f}°")
print(f"  atan2(dy, dx) = {atan2_dir:.2f}°")
print(f"  差值 = {direction - atan2_dir:.2f}°")

if abs(direction - atan2_dir) < 5:
    print("\n结论: env使用标准数学坐标系 (逆时针为正, 0°沿+x)")
    print("       _dubins_to_env_deg 不需要取负!")
elif abs(direction + atan2_dir) < 5:
    print("\n结论: env使用顺时针坐标系")
    print("       _dubins_to_env_deg 需要取负")
else:
    print("\n结论: 需要进一步分析坐标系关系")

print("\n" + "=" * 60)
print("测试不同初始方向")
print("=" * 60)

for test_dir in [0, 45, 90, 135, 180, -45, -90]:
    obs, info = env.reset(seed=42)
    # 设置方向
    base.agent.direction = float(test_dir)
    x, y = base.agent.position

    # 前进一步
    obs, _, _, _, _ = env.step((3.0, 0.0))
    new_x, new_y = base.agent.position
    dx, dy = new_x - x, new_y - y

    atan2_dir = math.degrees(math.atan2(dy, dx))
    print(f"方向={test_dir:>4}° → 移动方向(atan2)={atan2_dir:>7.2f}° → 差值={test_dir - atan2_dir:>7.2f}°")

env.close()
