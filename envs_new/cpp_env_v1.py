"""
CppEnv v1 - Environment without mist, simple map observation.
Based on the new modular architecture.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Optional
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
        # Create configuration for v1 (no mist)
        config_overrides = {
            'map_config': {
                'use_mist': False,  # Key difference: no mist
                'use_traj': True
            },
            'observation_config': {
                'use_multiscale': False,  # Simple first-person observation
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
    
    def _create_simple_observation_maps(self, maps_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Create simple observation maps (v1 style).
        
        Args:
            maps_dict: Dictionary containing all map types
            
        Returns:
            Stacked maps array with shape (H, W, C)
        """
        maps_list = [
            maps_dict['map_frontier'],
            maps_dict['map_obstacle'],
            np.logical_and(maps_dict['map_weed'], 
                          np.logical_not(maps_dict['map_frontier'])),  # Only weeds in non-frontier areas
            maps_dict['map_trajectory'] if 'map_trajectory' in maps_dict else np.zeros_like(maps_dict['map_frontier'])
        ]
        
        return np.stack(maps_list, axis=-1)


def create_cpp_env_v1(**kwargs) -> CppEnv:
    """Create CppEnv v1 with default parameters."""
    return CppEnv(**kwargs)


if __name__ == "__main__":
    if_render = True
    episodes = 3
    
    env = CppEnv(
        render_mode='rgb_array' if if_render else None,
        state_pixels=False,
    )
    
    if if_render:
        env = HumanRendering(env)

    for episode in range(episodes):
        print(f"Episode {episode + 1}")
        obs, info = env.reset(seed=42 + episode)
        done = False
        step_count = 0
        
        while not done and step_count < 1000:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1
            
            if step_count % 100 == 0:
                print(f"  Step {step_count}, Reward: {reward:.3f}")
        
        print(f"  Episode finished in {step_count} steps")