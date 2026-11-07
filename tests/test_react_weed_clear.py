#!/usr/bin/env python
"""诊断REACT杂草清除问题"""
import sys
sys.path.insert(0, '/home/lzh/NewCppRL')

import gymnasium as gym
import numpy as np
import envs_new

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

from rules_new.policies_rules import ReactController
controller = ReactController()

obs, info = env.reset(seed=42, options={
    'weed_dist': 'gaussian',
    'map_id': 2,
    'weed_num': 50
})
controller.reset(env)

base = env.unwrapped
maps = base.maps_dict

print("诊断REACT杂草清除...")
print(f"初始weed map sum: {np.sum(maps['weed'])}")

for step in range(100):
    # 获取杂草位置
    weeds = controller._get_discovered_weeds_world(env)
    x, y = controller._pose(env)

    if weeds.shape[0] > 0:
        # 计算到最近杂草的距离
        rel = weeds - np.array([x, y])
        dists = np.linalg.norm(rel, axis=1)
        min_dist = dists.min()
        nearest_weed = weeds[np.argmin(dists)]

        # 检查机器人是否在杂草位置
        robot_pixel = (int(y), int(x))
        weed_at_robot = maps['weed'][robot_pixel[0], robot_pixel[1]] if 0 <= robot_pixel[0] < 256 and 0 <= robot_pixel[1] < 256 else 0

        if step < 40 or min_dist < 5:
            print(f"Step {step}: pos=({x:.1f},{y:.1f}), nearest_weed=({nearest_weed[0]:.1f},{nearest_weed[1]:.1f}), "
                  f"dist={min_dist:.1f}, weed_at_robot={weed_at_robot}, total_weeds_pixels={np.sum(maps['weed'] > 0)}")

    action = controller.act(obs, env)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

print(f"\n最终weed map sum: {np.sum(maps['weed'])}")
print(f"杂草是否被清除: {np.sum(maps['weed']) < np.sum(maps['weed'])}")  # 这行有bug，先看结果

# 检查机器人FOV和weed mask的关系
print(f"\nFOV相关参数:")
print(f"  - agent_fov_radius (如果存在): {getattr(base.config, 'agent_fov_radius', 'N/A')}")
print(f"  - pass_width: {getattr(base.config, 'pass_width', 'N/A')}")

env.close()
