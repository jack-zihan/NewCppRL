"""
CppEnv v2 - Environment using APF (Artificial Potential Field) for observations and rewards.
Based on the new modular architecture.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Optional
from gymnasium.wrappers import HumanRendering

from envs_new.cpp_env_base import CppEnvBase
from envs_new.components.config.environment_config import EnvironmentConfig
from envs_new.components.reward.reward_system import RewardComponent

try:
    from cpu_apf import cpu_apf_bool
except ImportError:
    print("Warning: cpu_apf module not available, APF features will be disabled")
    cpu_apf_bool = None


class APFRewardComponent(RewardComponent):
    """APF-based reward component for v2 environment."""
    
    def __init__(self, weight: float = 1.0):
        super().__init__(weight)
        self.previous_apf_maps = None
    
    def calculate(self, 
                 agent_prev_pos: Tuple[float, float, float],
                 agent_current_pos: Tuple[float, float, float],
                 maps_dict: Dict[str, np.ndarray],
                 **kwargs) -> float:
        """Calculate APF-based reward."""
        if 'apf_maps' not in kwargs:
            return 0.0
        
        apf_maps = kwargs['apf_maps']
        
        # Get positions in map coordinates
        x_prev, y_prev = int(agent_prev_pos[0]), int(agent_prev_pos[1])
        x_curr, y_curr = int(agent_current_pos[0]), int(agent_current_pos[1])
        
        # Calculate APF potential differences
        reward_apf_frontier = 0.0 * (apf_maps[0][y_curr, x_curr] - apf_maps[0][y_prev, x_prev])
        reward_apf_obstacle = 0.3 * (apf_maps[2][y_curr, x_curr] - apf_maps[2][y_prev, x_prev])
        reward_apf_obstacle = min(0., reward_apf_obstacle)
        reward_apf_weed = 5.0 * (apf_maps[3][y_curr, x_curr] - apf_maps[3][y_prev, x_prev])
        
        reward_apf_traj = 0.
        if len(apf_maps) > 4:
            reward_apf_traj = 0.0 * (apf_maps[4][y_curr, x_curr] - apf_maps[4][y_prev, x_prev])
            reward_apf_traj = min(0., reward_apf_traj)
        
        return self.weight * (reward_apf_frontier + reward_apf_obstacle + reward_apf_weed + reward_apf_traj)


class CppEnv(CppEnvBase):
    """
    Environment using APF (Artificial Potential Field) for frontier edges observation.
    Features sophisticated potential field calculations and APF-based rewards.
    """
    
    def __init__(self, render_mode=None, **kwargs):
        """Initialize v2 environment with APF configuration."""
        # Create configuration for v2 (with APF)
        config_overrides = {
            'map_config': {
                'use_mist': True,
                'use_traj': True
            },
            'observation_config': {
                'use_multiscale': False,
                'state_size': (128, 128),
                'state_downsize': (128, 128),
                'use_global_features': False
            },
            'reward_config': {
                'coefficients': {
                    'base_penalty': -0.1,
                    'weed_removal_coef': 20.0,
                    'frontier_coverage_coef': 0.5,
                    'turn_total_coef': 0.0,
                    'turn_gap_coef': -0.5,
                    'turn_direction_coef': -0.30,
                    'turn_self_coef': 0.25,
                    'frontier_total_coef': 0.125,
                    'frontier_tv_coef': 0.5,
                    'collision_penalty': -399.0,
                    'completion_bonus': 500.0
                }
            },
            'render_config': {
                'render_mist': True,
                'render_covered_weed': True,
                'render_covered_farmland': True
            }
        }
        
        # Merge with user provided kwargs (excluding render_mode)
        for key, value in kwargs.items():
            if key != 'render_mode':
                if key in config_overrides:
                    if isinstance(value, dict):
                        config_overrides[key].update(value)
                    else:
                        config_overrides[key] = value
                else:
                    config_overrides[key] = value
        
        config = EnvironmentConfig.from_dict(config_overrides)
        super().__init__(config=config, render_mode=render_mode)
        
        # Add APF reward component
        self.apf_reward_component = APFRewardComponent(weight=1.0)
        self.reward_manager.add_component("apf_reward", self.apf_reward_component)
        
        # Store APF maps for reward calculation
        self.obs_apf = None
    
    @staticmethod
    def get_discounted_apf(map_apf: np.ndarray,
                          max_step: int,
                          eps: Optional[float] = None) -> float:
        """Get discounted APF value."""
        if eps is None:
            eps = 1e-6
        
        # Simple discounted sum implementation
        total = 0.0
        discount = 0.99
        for step in range(min(max_step, 100)):
            step_value = np.sum(map_apf * (discount ** step))
            total += step_value
            if step_value < eps:
                break
        
        return total
    
    def _create_apf_observation_maps(self, maps_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Create APF-based observation maps (v2 style).
        
        Args:
            maps_dict: Dictionary containing all map types
            
        Returns:
            Observation array with APF-processed maps
        """
        # Get basic maps
        map_frontier = maps_dict['map_frontier']
        map_obstacle = maps_dict['map_obstacle'] 
        map_weed = maps_dict['map_weed']
        map_traj = maps_dict.get('map_trajectory', np.zeros_like(map_frontier))
        
        # Get noisy weed map if enabled
        if (hasattr(self.config.map_config, 'weed_noise') and
            self.config.map_config.weed_noise > 0 and
            hasattr(self, 'np_random')):
            if self.np_random.uniform() < self.weed_noise and 'map_weed_noisy' in maps_dict:
                map_weed = maps_dict['map_weed_noisy']
        
        # Apply total variation to get edges
        from envs_new.components.utils import total_variation_mat
        
        apf_frontier = np.logical_and(total_variation_mat(maps_dict['map_frontier']), 
                                     maps_dict.get('map_mist', np.ones_like(maps_dict['map_frontier'])))
        apf_obstacle = np.logical_and(total_variation_mat(maps_dict['map_obstacle']), 
                                     maps_dict.get('map_mist', np.ones_like(maps_dict['map_obstacle'])))
        apf_weed = map_weed.copy()
        
        # Mask with mist if available
        if 'map_mist' in maps_dict:
            apf_frontier = apf_frontier * maps_dict['map_mist']
            apf_obstacle = apf_obstacle * maps_dict['map_mist'] 
            apf_weed = apf_weed * maps_dict['map_mist']
        
        # Stack observation maps
        obs_maps = np.stack([
            apf_frontier.astype(np.float32),
            apf_obstacle.astype(np.float32), 
            apf_weed.astype(np.float32),
            map_traj.astype(np.float32)
        ], axis=0)
        
        # Store APF maps for reward calculation
        self.obs_apf = [apf_frontier, map_obstacle, apf_obstacle, apf_weed, map_traj]
        
        return obs_maps


if __name__ == "__main__":
    # Test direct execution
    import sys
    sys.path.append('/home/lzh/NewCppRL')
    
    print("Testing CppEnv v2 direct execution...")
    env = CppEnv()
    print("✓ Environment created successfully")
    
    obs, info = env.reset()
    print(f"✓ Environment reset successful, obs shape: {obs['observation'].shape}")
    
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.3f}, done={done}")
        if done or truncated:
            obs, info = env.reset()
    
    print("✓ CppEnv v2 direct execution test completed successfully!")