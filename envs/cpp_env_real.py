from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import gymnasium as gym

from envs.cpp_env_v2 import CppEnv
from envs.utils import RealAgent
from gymnasium.wrappers import HumanRendering

class CppEnvReal(CppEnv):
    def __init__(self, *args, **kwargs):
        """
        Using this env in real robot test, could not directly traing in torchrl
        """
        super().__init__(*args, **kwargs)
        self.agent = RealAgent(
            position=self.agent.position,
            direction=self.agent.direction,
        )

        self.action_type = 'real'
        self.action_space = gym.spaces.Dict({
            'position': gym.spaces.Box(
                low=np.array([0, 0]),
                high=np.array(self.dimensions),
                dtype=np.float32
            ),
            'direction': gym.spaces.Box(
                low=0.0,
                high=360.0,
                dtype=np.float32
            ),
        })

    def get_action(self, action: dict) -> tuple[np.ndarray, float]:
        new_position = np.array(action['position'], dtype=np.float32)
        new_direction = float(action['direction'])
        return new_position, new_direction
if __name__ == "__main__":
    if_render = True
    episodes = 3
    real_map_dir = '/home/lzh/NewCppRL/envs/maps/real_true'
    env = CppEnvReal(
        render_mode='rgb_array'if if_render else None,
        # state_pixels=True,
        state_pixels=False,
        # use_sgcnn=False,
        # use_global_obs=False,
        # num_obstacles_range = [0, 0]
    )

    env: CppEnv = HumanRendering(env)  # 封装后，只接收render_mode="rgb_array"的env，使得step和reset的时候展示渲染图像
    for _ in range(episodes):
        # env.set_obstacle_range([0,0])
        obs, info = env.reset(seed=120, options={
            'weed_dist': 'gaussian',
            # 'map_id': 80,
            "weed_num": 100,
            # "specific_scenario_dir": real_map_dir,
            'initial_position': (150.0, 200.0),  # X 和 Y 坐标
            'initial_direction': 360.0,
        })
        env.action_space.seed(66)
        done = False
        while not done:
            # action = env.action_space.sample()
            action = {
                'position': [150.0, 200.0],
                'direction': 360.0,
            }
            obs, reward, done, _, info = env.step(action)
            # obs, reward, done, _, info = env.step((0, 4))
            print(reward,action)
            if if_render:
                img = env.render()

    env.close()

