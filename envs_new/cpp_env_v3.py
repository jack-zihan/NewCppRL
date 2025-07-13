"""
CppEnv v3 - Environment with mist exploration mechanics.
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
    Environment with mist that the agent must explore to discover what's inside.
    Features visibility limitations and exploration rewards.
    """
    
    def __init__(self, render_mode=None, **kwargs):
        """Initialize v3 environment with mist exploration configuration."""
        # Create configuration for v3 (with mist exploration)
        config_overrides = {
            'map_config': {
                'use_mist': True,  # Key feature: mist system
                'use_traj': True
            },
            'observation_config': {
                'use_multiscale': False,
                'state_size': (128, 128),
                'state_downsize': (128, 128),
                'use_global_features': False,
                'position_noise': 0.1  # Add some noise for realism
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
        
        # Track explored areas for exploration rewards
        self.explored_mist_areas = None
    
    def _create_mist_observation_maps(self, maps_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Create mist-aware observation maps (v3 style).
        
        Args:
            maps_dict: Dictionary containing all map types
            
        Returns:
            Stacked maps array with shape (H, W, C) with mist visibility applied
        """
        # Get noise-adjusted weed map if noise is enabled
        map_weed = maps_dict['map_weed']
        if (hasattr(self.config.observation_config, 'position_noise') and 
            self.config.observation_config.position_noise > 0 and 
            hasattr(self, 'np_random')):
            if (self.np_random.uniform() < self.config.observation_config.position_noise and 
                'map_weed_noisy' in maps_dict):
                map_weed = maps_dict['map_weed_noisy']
        
        # Get mist map - only visible areas can be observed
        map_mist = maps_dict.get('map_mist', np.ones_like(maps_dict['map_frontier']))
        
        maps_list = [
            maps_dict['map_trajectory'],  # Trajectory first in v3
            np.logical_not(map_mist),     # Mist visibility mask
            maps_dict['map_obstacle'],    # Obstacles (only visible in explored areas)
            np.logical_and(map_weed, np.logical_not(maps_dict['map_frontier'])),  # Weeds outside frontier
        ]
        
        if self.config.map_config.use_traj:
            maps_list.append(maps_dict['map_trajectory'])
        
        return np.stack(maps_list, axis=-1)
    
    def _calculate_exploration_reward(self, agent_pos: Tuple[float, float, float]) -> float:
        """
        Calculate reward for exploring new misted areas.
        
        Args:
            agent_pos: Current agent position (x, y, direction)
            
        Returns:
            Exploration reward
        """
        if 'map_mist' not in self.maps_dict:
            return 0.0
        
        x, y = int(agent_pos[0]), int(agent_pos[1])
        map_mist = self.maps_dict['map_mist']
        
        # Initialize explored areas map if needed
        if self.explored_mist_areas is None:
            self.explored_mist_areas = np.zeros_like(map_mist)
        
        # Check if current position was previously unexplored mist
        if (0 <= y < map_mist.shape[0] and 
            0 <= x < map_mist.shape[1] and
            not map_mist[y, x] and  # Was misted
            not self.explored_mist_areas[y, x]):  # Not previously explored
            
            # Mark as explored
            self.explored_mist_areas[y, x] = 1
            
            # Give exploration reward (hardcoded weight for v3)
            exploration_weight = 0.1
            return exploration_weight
        
        return 0.0
    
    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """Override step to include exploration rewards."""
        # Call parent step
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Add exploration reward
        if self.agent is not None:
            exploration_reward = self._calculate_exploration_reward(
                (self.agent.x, self.agent.y, self.agent.direction)
            )
            reward += exploration_reward
            info['exploration_reward'] = exploration_reward
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Override reset to initialize exploration tracking."""
        obs, info = super().reset(seed=seed, options=options)
        
        # Reset exploration tracking
        self.explored_mist_areas = None
        
        return obs, info
    
    def render_map_with_mist_effect(self) -> Optional[np.ndarray]:
        """
        Render map with mist effect (darkened unexplored areas).
        
        Returns:
            Rendered map with mist effect applied
        """
        if not hasattr(self, 'render_manager'):
            return None
        
        # Get base rendered map
        rendered_map = self.render_manager.render_complete_map(
            self.maps_dict, self.agent, self.env_state
        )
        
        if rendered_map is None or 'map_mist' not in self.maps_dict:
            return rendered_map
        
        # Apply mist effect (darken misted areas)
        map_mist = self.maps_dict['map_mist']
        mist_mask = np.expand_dims(map_mist, axis=-1)  # Add channel dimension
        
        # Darken misted areas by 50%
        rendered_map = np.where(
            mist_mask,
            rendered_map,
            (rendered_map * 0.5).astype(np.uint8)
        )
        
        return rendered_map


def create_cpp_env_v3(**kwargs) -> CppEnv:
    """Create CppEnv v3 with default parameters."""
    return CppEnv(**kwargs)


if __name__ == "__main__":
    if_render = True
    episodes = 3
    
    env = CppEnv(
        render_mode='rgb_array' if if_render else None,
        state_pixels=True,  # v3 typically uses pixel rendering
    )
    
    if if_render:
        env = HumanRendering(env)

    for episode in range(episodes):
        print(f"Episode {episode + 1}")
        obs, info = env.reset(seed=42 + episode)
        done = False
        step_count = 0
        total_reward = 0
        total_exploration_reward = 0
        
        while not done and step_count < 1000:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1
            total_reward += reward
            
            exploration_reward = info.get('exploration_reward', 0)
            total_exploration_reward += exploration_reward
            
            if step_count % 100 == 0:
                print(f"  Step {step_count}, Reward: {reward:.3f}, Exploration: {exploration_reward:.3f}")
        
        print(f"  Episode finished in {step_count} steps")
        print(f"  Total reward: {total_reward:.3f}, Total exploration: {total_exploration_reward:.3f}")