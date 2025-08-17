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
    
    def __init__(self, render_mode=None, **kwargs):
        """Initialize v3 environment with mist exploration configuration."""
        # v3特定默认值：有mist，无APF
        v3_defaults = {
            'use_mist': True,
            'use_apf': False,
            'obs_use_mist': True,
            'use_traj': True,
            'render_mist': True,
            'position_noise': 0.1,  # v3特有的噪声设置
            # v3特定的奖励系数（与v1/v2相同的值）
            'reward_frontier_coverage_coef': 0.5,
            'reward_frontier_total_coef': 0.125,
        }
        
        # 合并用户参数，用户参数优先
        final_kwargs = {**v3_defaults, **kwargs}
        super().__init__(render_mode=render_mode, **final_kwargs)
        
        self.use_traj = self.config.use_traj
        self.noise_weed = self.config.weed_noise
        
        self.obs_mask = None
    
    def _get_observation_maps(self) -> Dict[str, Dict[str, Any]]:
        """
        获取v3环境的观察地图。
        v3使用带mist的地图，具有探索机制。
        
        Returns:
            包含4个地图的字典：frontier, mist_inv, obstacle, weed
        """        
        # 提取所需地图
        map_frontier = self.maps_dict['field_frontier']
        map_obstacle = self.maps_dict['obstacle'] 
        map_weed = self.maps_dict['weed']
        map_mist = self.maps_dict.get('mist', np.ones_like(map_frontier))  # 默认全部可见

        if self.config.weed_noise and self.np_random.uniform() < self.noise_weed:
            map_weed = self.maps_dict.get('weed_noisy', map_weed)

        # 创建观察地图字典，注意obstacle的pad值为1.0
        obs_maps = {
            'frontier': {'map': map_frontier, 'pad': 0.0},                    # 前沿区域
            'mist_inv': {'map': np.logical_not(map_mist), 'pad': 0.0},       # 反转的mist（未探索区域）
            'obstacle': {'map': map_obstacle, 'pad': 1.0},                    # 障碍物（边界视为障碍物）
            'weed': {                                                          # 非前沿区域的杂草
                'map': np.logical_and(map_weed, np.logical_not(map_frontier)),
                'pad': 0.0
            }
        }
        
        # 存储mask列表
        self.obs_mask = [0., 0., 1., 0.]  # 只有obstacle获得权重1.0
        
        return obs_maps




if __name__ == "__main__":
    if_render = True
    episodes = 3
    env = CppEnv(
        render_mode='rgb_array' if if_render else None,  # HumanRendering需要rgb_array
        render_first_person=True,  # 控制渲染第一人称视角
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
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step(action)
            print(reward)
            if if_render:
                env.render()