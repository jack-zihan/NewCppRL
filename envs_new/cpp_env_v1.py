"""
CppEnv v1 - Environment without mist, simple map observation.
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
    Simple environment without mist.
    Provides basic map observation for frontier, obstacles, weeds, and trajectory.
    """

    def __init__(self, render_mode="rgb_array", **kwargs):
        """Initialize v1 environment with specific configuration."""
        # v1特定默认值：无mist，无APF
        v1_configs = {'use_mist': False, 'use_apf': False, 'use_trajectory': True,}

        # 合并用户参数，用户参数优先
        final_kwargs = {**v1_configs, **kwargs}
        super().__init__(render_mode=render_mode, **final_kwargs)

    def _get_observation_maps(self) -> Dict[str, Dict[str, Any]]:
        """
        v1使用简单的地图观察，不包含mist于探索机制（frontier, obstacle, weed, trajectory）。
        """
        maps = {
            'field': {'map': self.maps_dict['field'], 'pad': 0.0},
            'obstacle': {'map': self.maps_dict['obstacle'], 'pad': 1.0},  # obstacle的pad值为1.0
            'weed': {'map': self.maps_dict['weed_noisy']
                    if (self.config.weed_noise and self.np_random.uniform() < self.config.weed_noise)
                    else self.maps_dict['weed'], 'pad': 0.0}
        }
        # 添加轨迹地图
        if 'trajectory' in self.maps_dict:
            maps['trajectory'] = {'map': self.maps_dict['trajectory'], 'pad': 0.0}

        return maps


if __name__ == "__main__":
    if_render = True
    episodes = 3
    env = CppEnv(
        render_mode='rgb_array' if if_render else None,  # v1默认使用全局视图
    )
    env: CppEnv = HumanRendering(env)

    for _ in range(episodes):
        reset_state = {}
        obs, info = env.reset(**reset_state)
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step(action)
            print(reward)
            if if_render:
                env.render()
