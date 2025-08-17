"""
CppEnv v2 - Environment using APF (Artificial Potential Field) for observations and rewards.
Based on the new modular architecture.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Optional, Union, Any
from gymnasium.wrappers import HumanRendering

from envs_new.cpp_env_base import CppEnvBase
from envs_new.components.config.environment_config import EnvironmentConfig
from envs_new.components.reward.reward_system import RewardSystem
from envs_new.components.state.environment_state import EnvironmentState

try:
    from cpu_apf import cpu_apf_bool
except ImportError:
    print("Warning: cpu_apf module not available, APF features will be disabled")
    cpu_apf_bool = None


class APFCalculator:
    """基于APF的奖励计算器"""
    coefficient = 1.0
    
    @staticmethod
    def calculate(env_state: EnvironmentState, **kwargs) -> float:
        """使用标准接口计算基于APF的奖励"""
        agent_pos_info = env_state.get_info('agent_position')
        if not agent_pos_info or agent_pos_info.last is None:
            return 0.0
        
        if 'apf_maps' not in kwargs:
            return 0.0
        
        apf_maps = kwargs['apf_maps']
        
        current_position = agent_pos_info.current
        previous_position = agent_pos_info.last
        
        # 使用int()进行网格索引查询（floor操作），获取当前所在的网格
        x_curr, y_curr = int(current_position[0]), int(current_position[1])
        x_prev, y_prev = int(previous_position[0]), int(previous_position[1])
        
        if (apf_maps.shape[1] <= max(y_prev, y_curr) or 
            apf_maps.shape[2] <= max(x_prev, x_curr) or
            min(x_prev, y_prev, x_curr, y_curr) < 0):
            return 0.0
        
        reward_apf_frontier = 0.0 * (apf_maps[0][y_curr, x_curr] - apf_maps[0][y_prev, x_prev])
        reward_apf_obstacle = 0.3 * (apf_maps[2][y_curr, x_curr] - apf_maps[2][y_prev, x_prev])
        reward_apf_obstacle = min(0., reward_apf_obstacle)
        reward_apf_weed = 5.0 * (apf_maps[3][y_curr, x_curr] - apf_maps[3][y_prev, x_prev])
        
        reward_apf_traj = 0.
        if len(apf_maps) > 4:
            reward_apf_traj = 0.0 * (apf_maps[4][y_curr, x_curr] - apf_maps[4][y_prev, x_prev])
            reward_apf_traj = min(0., reward_apf_traj)
        
        return APFCalculator.coefficient * (reward_apf_frontier + reward_apf_obstacle + reward_apf_weed + reward_apf_traj)


class CppEnv(CppEnvBase):
    """
    Environment using APF (Artificial Potential Field) for frontier edges observation.
    Features sophisticated potential field calculations and APF-based rewards.
    """
    
    def __init__(self, render_mode=None, **kwargs):
        """Initialize v2 environment with APF configuration."""
        # v2特定默认值：有mist，有APF
        v2_defaults = {
            'use_mist': True,
            'use_apf': True,
            'obs_use_mist': True,
            'use_traj': True,
            'render_mist': True,
            # v2特定的奖励系数（与v1相同的值）
            'reward_frontier_coverage_coef': 1.0,
            'reward_frontier_total_coef': 0.125,
        }
        
        # 合并用户参数，用户参数优先
        final_kwargs = {**v2_defaults, **kwargs}
        super().__init__(render_mode=render_mode, **final_kwargs)
        
        self.use_apf = self.config.use_apf
        self.use_traj = self.config.use_traj
        self.noise_weed = self.config.weed_noise
        
        self.reward_system.add_calculator("apf_reward", APFCalculator)
        
        self.obs_apf = None
        self.obs_mask = None
    
    @staticmethod
    def get_discounted_apf(map_apf: np.ndarray,
                          max_step: int,
                          eps: Optional[float] = None,
                          pad: bool = False) -> np.ndarray:
        """Get discounted APF value."""
        if pad:
            map_apf = np.pad(map_apf, pad_width=[[1, 1], [1, 1]],mode='constant',
                             constant_values=(1, 1))
        
        if cpu_apf_bool is not None:
            # APF距离传播：将二值地图转换为距离场，gamma^步数计算势能衰减
            map_apf, is_empty = cpu_apf_bool(map_apf)
            if not is_empty:
                gamma = (max_step - 1) / max_step
                map_apf = gamma ** map_apf
                if eps is None:
                    eps = gamma ** max_step
                map_apf = np.where(map_apf < eps, 0., map_apf)
        else:
            # Fallback when cpu_apf is not available
            print("Warning: cpu_apf not available, using fallback APF implementation")
        
        if pad:
            map_apf = map_apf[1:-1, 1:-1]
        return map_apf
    
    def _get_observation_maps(self) -> Dict[str, Dict[str, Any]]:
        """
        获取v2环境的观察地图。
        v2使用APF处理后的地图，包含mist信息。
        
        Returns:
            包含5个地图的字典：frontier_apf, mist_inv, obstacle_apf, weed_apf, trajectory_apf
        """
        map_frontier = self.maps_dict['field_frontier']
        map_obstacle = self.maps_dict['obstacle'] 
        map_weed = self.maps_dict['weed']
        map_mist = self.maps_dict.get('mist', np.ones_like(map_frontier))
        map_trajectory = self.maps_dict.get('trajectory', np.zeros_like(map_frontier))
        
        if self.config.weed_noise and self.np_random.uniform() < self.noise_weed:
            map_weed = self.maps_dict.get('weed_noisy', map_weed)
        
        from envs_new.utils.math_utils import total_variation_mat
        
        apf_frontier = np.logical_and(total_variation_mat(map_frontier), map_mist)
        apf_obstacle = np.logical_and(total_variation_mat(map_obstacle), map_mist)
        apf_weed = np.logical_and(map_weed, np.logical_not(map_frontier))
        apf_trajectory = map_trajectory
        
        if hasattr(self, 'use_apf') and self.use_apf:
            apf_frontier = self.get_discounted_apf(apf_frontier, 30)
            apf_obstacle = self.get_discounted_apf(apf_obstacle, 10, pad=True)
            # Critical fix: 确保障碍物本身在APF场中的值为1
            apf_obstacle = np.maximum(apf_obstacle, np.logical_and(map_obstacle, map_mist))
            apf_weed = self.get_discounted_apf(apf_weed, 40, 1e-2)
            apf_trajectory = self.get_discounted_apf(apf_trajectory, 4)
        
        # Create maps list
        maps_list = [
            apf_frontier,
            np.logical_not(map_mist),  # Inverted mist - critical for 5-channel output
            apf_obstacle,
            apf_weed,
        ]

        mask_list = [0., 0., 1., 0.]
        
        # 添加轨迹地图
        if hasattr(self, 'use_traj') and self.use_traj:
            maps_list.append(apf_trajectory)
            mask_list.append(0.)
        
        # 存储APF地图用于奖励计算
        self.obs_apf = np.stack(maps_list, axis=0)  # 存储为 (C, H, W) 格式
        self.obs_mask = mask_list
        
        # 创建观察地图字典，注意obstacle_apf的pad值为1.0
        obs_maps = {
            'frontier_apf': {'map': apf_frontier, 'pad': 0.0},
            'mist_inv': {'map': np.logical_not(map_mist), 'pad': 0.0},
            'obstacle_apf': {'map': apf_obstacle, 'pad': 1.0},  # 边界视为障碍物
            'weed_apf': {'map': apf_weed, 'pad': 0.0},
        }
        
        if hasattr(self, 'use_traj') and self.use_traj:
            obs_maps['trajectory_apf'] = {'map': apf_trajectory, 'pad': 0.0}
        
        return obs_maps
    
    def step(self, action):
        """Override step to provide APF maps for reward calculation."""
        # Apply dynamics (this will update agent position and maps)
        self.agent, self.maps_dict, self.env_state = self.env_dynamics.step(
            self.agent, self.maps_dict, self.env_state, action, self.config.action_type
        )
        
        # 获取观察地图（这会更新self.obs_apf）
        observation_maps = self._get_observation_maps()
        
        # Calculate reward with APF maps
        reward = self.reward_system.calculate_reward(
            self.env_state, 
            apf_maps=self.obs_apf
        )
        
        # Check episode termination
        terminated = self.env_state.crashed or self.env_state.finished
        truncated = self.env_state.timeout
        
        # Generate observation
        observation = self._generate_observation()
        
        # Additional info
        info = {
            'crashed': self.env_state.crashed,
            'finished': self.env_state.finished,
            'timeout': self.env_state.timeout,
            'weed_count': self.env_state.weed_count,
            'weed_ratio': self.env_state.weed_completion_ratio
        }
        
        return observation, np.array(reward), np.bool_(terminated), truncated, info


if __name__ == "__main__":
    if_render = True
    episodes = 3
    real_map_dir = '/home/lzh/NewCppRL/envs/maps/real'
    env = CppEnv(
        render_mode='rgb_array' if if_render else None,  # HumanRendering需要rgb_array
        render_first_person=True,  # 控制渲染第一人称视角
        # use_multiscale=False, # 是否使用多尺度观察
        # use_global_obs=False,
        # num_obstacles_range = [0, 0]
    )
    # Note: HumanRendering wrapper would be added here if available
    env: CppEnv = HumanRendering(env)  # 封装后，只接收render_mode="rgb_array"的env，使得step和reset的时候展示渲染图像

    for _ in range(episodes):
        # env.set_obstacle_range([0,0])
        obs, info = env.reset(seed=120, options={
            'weed_distribution': 'gaussian',  # Updated parameter name
            # 'map_id': 80,
            "weed_count": 100,  # Updated parameter name
            # "specific_scenario_dir": real_map_dir
        })
        env.action_space.seed(66)
        done = False
        while not done:
            action = env.action_space.sample()
            # action = 1 * 21 + 10
            obs, reward, done, truncated, info = env.step(action)
            # obs, reward, done, _, info = env.step((0, 4))
            print(reward)
            if if_render:
                img = env.render()

    env.close()