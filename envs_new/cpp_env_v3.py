"""
CppEnv v3 - Environment with mist exploration mechanics.
Based on the new modular architecture.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Optional, Any
from gymnasium.wrappers import HumanRendering

from envs_new.cpp_env_base import CppEnvBase
from envs_new.components.config.environment_config import EnvironmentConfig


class CppEnv(CppEnvBase):
    """
    Environment with mist that the agent must explore to discover what's inside.
    Features visibility limitations and exploration rewards.
    """

    def __init__(self, render_mode="rgb_array", **kwargs):
        """Initialize v3 environment with mist exploration configuration."""
        # v3特定默认值：有mist，无apf
        v3_defaults = {'use_mist': True, 'use_apf': False, 'use_trajectory': True, 'render_mist': True,
                       # 'position_noise': 0.1,  # v3特有的噪声设置
                       }
        final_kwargs = {**v3_defaults, **kwargs}
        super().__init__(render_mode=render_mode, **final_kwargs)

    def _get_observation_channels(self) -> int:
        """v3环境的观察通道数：field, mist_inv, obstacle, weed, (trajectory)"""
        return 4 + int(self.config.use_trajectory)

    def _get_observation_maps(self) -> Dict[str, Dict[str, Any]]:
        """
        v3使用带mist的地图，具有探索机制, 但不使用apf和reward_apf, 包含4个地图的字典：field, mist_inv, obstacle, weed
        """
        # 提取所需地图
        map_field, map_weed = self.maps_dict['field'], self.maps_dict['weed']
        map_obstacle, map_mist = self.maps_dict['obstacle'], self.maps_dict.get('mist', np.ones_like(map_field))  # 默认全部可见

        if self.config.weed_noise and self.np_random.uniform() < self.config.weed_noise:
            map_weed = self.maps_dict.get('weed_noisy', map_weed)

        # 创建观察地图字典，注意obstacle的pad值为1.0
        obs_maps = {
            'field': {'map': np.logical_and((map_field), map_mist), 'pad': 0.0},  # 田地区域
            'mist_inv': {'map': np.logical_not(map_mist), 'pad': 0.0},  # 反转的mist（未探索区域）
            'obstacle': {'map': np.logical_and(map_obstacle, map_mist), 'pad': 1.0},  # 障碍物（边界视为障碍物）
            'weed': {'map': np.logical_and(map_weed, map_mist),'pad': 0.0} # 非田地区域, 已经发现的杂草
        }
        if 'trajectory' in self.maps_dict:
            obs_maps['trajectory'] = {'map': self.maps_dict['trajectory'], 'pad': 0.0}
        return obs_maps


if __name__ == "__main__":
    if_render = True
    episodes = 3
    env = CppEnv(
        render_first_person=True,  # 控制渲染第一人称视角
        action_type = "continuous",
    )
    env: CppEnv = HumanRendering(env)

    for _ in range(episodes):
        obs, info = env.reset(seed=120, options={
            'weed_dist': 'gaussian',
            "weed_num": 100
        })
        env.action_space.seed(66)
        done = False
        while not done:
            # action = env.action_space.sample()
            action = (0,0)
            obs, reward, done, _, info = env.step(action)
            print(reward)
            if if_render:
                env.render()
