#!/usr/bin/env python
"""单个算法测试"""
import sys
sys.path.insert(0, '/home/lzh/NewCppRL')

import gymnasium as gym
import numpy as np
import cv2
import envs_new

ALG = sys.argv[1] if len(sys.argv) > 1 else "BCP"
MAX_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

print(f"测试 {ALG}, 步数={MAX_STEPS}")

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

from rules_new.policies_rules import (
    BCPController, JumpController, SnakeController,
    RSnakeController, ReactController
)

controllers = {
    'BCP': BCPController,
    'JUMP': JumpController,
    'SNAKE': SnakeController,
    'R_SNAKE': RSnakeController,
    'REACT': ReactController
}

controller = controllers[ALG]()

obs, info = env.reset(seed=42, options={
    'weed_dist': 'gaussian',
    'map_id': 2,
    'weed_num': 50
})
controller.reset(env)

collision_count = 0
for step in range(MAX_STEPS):
    action = controller.act(obs, env)
    obs, reward, terminated, truncated, info = env.step(action)

    if info.get('collision', False):
        collision_count += 1

    if step % 200 == 0:
        if isinstance(obs, dict) and 'completion_ratio' in obs:
            completion = obs['completion_ratio'][0] * 100
        else:
            completion = 0
        print(f"  Step {step}: completion={completion:.1f}%")

    if terminated or truncated:
        break

# 最终状态
if isinstance(obs, dict) and 'completion_ratio' in obs:
    final_completion = obs['completion_ratio'][0] * 100
else:
    final_completion = 0

print(f"\n结果: completion={final_completion:.1f}%, collisions={collision_count}, steps={step+1}")

frame = env.render()
if frame is not None:
    cv2.imwrite(f'/tmp/test_{ALG.lower()}.png', frame[:, :, ::-1])
    print(f"图片: /tmp/test_{ALG.lower()}.png")

env.close()
