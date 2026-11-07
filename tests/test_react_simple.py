#!/usr/bin/env python
"""REACT算法简化诊断"""
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

print("REACT 诊断开始...")

for step in range(300):
    action = controller.act(obs, env)
    obs, reward, terminated, truncated, info = env.step(action)

    if step < 30 or step % 50 == 0:
        x, y = controller._pose(env)
        weeds = controller._get_discovered_weeds_world(env)
        print(f"Step {step}: pos=({x:.1f},{y:.1f}), mode={controller.mode}, "
              f"path_len={len(controller.path)}, weeds={weeds.shape[0]}")

    if terminated or truncated:
        break

# 打印最终状态
if isinstance(obs, dict) and 'completion_ratio' in obs:
    completion = obs['completion_ratio'][0] * 100
else:
    completion = info.get('completion_ratio', 0) * 100

print(f"\n最终完成率: {completion:.1f}%")

import cv2
frame = env.render()
if frame is not None:
    cv2.imwrite('/tmp/react_simple.png', frame[:, :, ::-1])
    print("图片保存到 /tmp/react_simple.png")

env.close()
print("完成")
