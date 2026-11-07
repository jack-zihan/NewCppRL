#!/usr/bin/env python
"""测试修复后的REACT算法"""
import sys
sys.path.insert(0, '/home/lzh/NewCppRL')

import gymnasium as gym
import numpy as np
import cv2
import envs_new
from rules_new.policies_rules import ReactController

MAX_STEPS = 1500

env = gym.make('NewPasture-v2',
    state_size=(256, 256),
    state_downsize=(256, 256),
    action_type='continuous',
    use_trajectory=True,
    boundary_source='box',
    use_apf=True,
    use_mist=True,
    num_obstacles_range=(0, 0),
    render_mode='rgb_array'
)

obs, info = env.reset(seed=42, options={
    'weed_dist': 'gaussian',
    'map_id': 2,
    'weed_num': 50
})

ctrl = ReactController()
ctrl.reset(env)

print(f"测试 REACT, 步数={MAX_STEPS}")

explore_count = 0
weed_count = 0
first_explore = None

for step in range(MAX_STEPS):
    try:
        action = ctrl.act(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)

        if ctrl.mode == 'explore':
            explore_count += 1
            if first_explore is None:
                first_explore = step
        else:
            weed_count += 1

        if step % 300 == 0:
            if isinstance(obs, dict) and 'completion_ratio' in obs:
                completion = obs['completion_ratio'][0] * 100
            else:
                completion = 0
            weeds = ctrl._get_discovered_weeds_world(env)
            print(f"  Step {step}: completion={completion:.1f}%, mode={ctrl.mode}, weeds={weeds.shape[0]}")

        if terminated or truncated:
            break
    except Exception as e:
        print(f"Error at step {step}: {e}")
        break

# 最终状态
if isinstance(obs, dict) and 'completion_ratio' in obs:
    final_completion = obs['completion_ratio'][0] * 100
else:
    final_completion = 0

print(f"\n结果: completion={final_completion:.1f}%, steps={step+1}")
print(f"Mode分布: explore={explore_count}, weed={weed_count}")
if first_explore:
    print(f"First explore at step: {first_explore}")

frame = env.render()
if frame is not None:
    cv2.imwrite('/tmp/test_react_fixed.png', frame[:, :, ::-1])
    print(f"图片: /tmp/test_react_fixed.png")

env.close()
