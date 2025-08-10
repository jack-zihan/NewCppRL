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
    
    def __init__(self, render_mode=None, **kwargs):
        """Initialize v1 environment with specific configuration."""
        # v1特定默认值：无mist，无APF
        v1_defaults = {
            'use_mist': False,
            'use_apf': False,
            'obs_use_mist': False,
            'use_traj': True,
            # v1特定的奖励系数
            'reward_frontier_coverage_coef': 0.5,
            'reward_frontier_total_coef': 0.125,
        }
        
        # 合并用户参数，用户参数优先
        final_kwargs = {**v1_defaults, **kwargs}
        super().__init__(render_mode=render_mode, **final_kwargs)
    
    def _get_observation_maps(self) -> Dict[str, Dict[str, Any]]:
        """
        获取v1环境的观察地图。
        v1使用简单的地图观察，不包含mist。
        
        Returns:
            包含4个地图的字典：frontier, obstacle, weed, trajectory
        """
        # 基础地图，注意obstacle的pad值为1.0
        maps = {
            'frontier': {'map': self.maps_dict['field_frontier'], 'pad': 0.0},
            'obstacle': {'map': self.maps_dict['obstacle'], 'pad': 1.0},
            'weed': {
                'map': np.logical_and(
                    self.maps_dict['weed'], 
                    np.logical_not(self.maps_dict['field_frontier'])
                ),  # 只包含非frontier区域的杂草
                'pad': 0.0
            }
        }
        
        # 添加轨迹地图
        if 'trajectory' in self.maps_dict:
            maps['trajectory'] = {'map': self.maps_dict['trajectory'], 'pad': 0.0}
        else:
            maps['trajectory'] = {'map': np.zeros_like(self.maps_dict['field_frontier']), 'pad': 0.0}
        
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