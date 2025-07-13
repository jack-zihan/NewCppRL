"""
Improved CppEnvBase environment using modular component architecture.
Provides better maintainability, extensibility, and testability.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Union, Optional, Any

try:
    import pygame
except ImportError:
    pygame = None

from envs_new.components.config.environment_config import EnvironmentConfig
from envs_new.components.entity.agent import AgentFactory
from envs_new.components.state.environment_state import EnvironmentState
from envs_new.components.map.map_generator import MapGenerator
from envs_new.components.observation.observation_strategy import ObservationManager
from envs_new.components.dynamics.environment_dynamics import EnvironmentDynamics
from envs_new.components.dynamics.action_processor import ActionProcessor
from envs_new.components.reward.reward_system import RewardManager
from envs_new.components.render.render_manager import RenderManager


class CppEnvBase(gym.Env):
    """
    Modular mowing robot environment with improved architecture.
    
    This environment provides:
    - Configurable component system
    - Clear separation of concerns
    - Enhanced maintainability and extensibility
    - Full compatibility with original functionality
    """
    
    metadata = {
        "render_modes": ["rgb_array", "state_pixels"],
        "render_fps": 50,
    }
    
    def __init__(self, 
                 config: Optional[EnvironmentConfig] = None,
                 render_mode: Optional[str] = None,
                 **kwargs):
        """
        Initialize environment with component-based architecture.
        
        Args:
            config: Environment configuration (uses defaults if None)
            render_mode: Rendering mode
            **kwargs: Additional configuration overrides
        """
        super().__init__()
        
        # Initialize configuration
        if config is None:
            # Create default config with any overrides
            config_dict = kwargs
            self.config = EnvironmentConfig.from_dict(config_dict)
        else:
            self.config = config
        
        self.render_mode = render_mode
        
        # Initialize components
        self._initialize_components()
        
        # Initialize spaces
        self._initialize_spaces()
        
        # Environment state
        self.agent = None
        self.maps_dict = {}
        self.env_state = EnvironmentState()
        
        # Rendering
        self.screen = None
        self.clock = None
        self.is_open = True
    
    def _initialize_components(self) -> None:
        """Initialize all environment components."""
        # Map generation
        self.map_generator = MapGenerator(
            self.config.map_config,
            self.config.agent_config
        )
        
        # Action processing
        self.action_processor = ActionProcessor(self.config.action_config)
        
        # Environment dynamics
        self.env_dynamics = EnvironmentDynamics(
            self.config.agent_config,
            self.action_processor
        )
        
        # Observation generation
        self.observation_manager = ObservationManager(self.config.observation_config)
        
        # Reward calculation
        self.reward_manager = RewardManager(
            self.config.reward_config,
            self.config.action_config,
            self.config.agent_config
        )
        
        # Rendering
        self.render_manager = RenderManager(self.config.render_config)
    
    def _initialize_spaces(self) -> None:
        """Initialize action and observation spaces."""
        # Action space
        if self.config.action_type == "discrete":
            self.action_space = gym.spaces.Discrete(
                self.action_processor.get_action_space_size()
            )
        elif self.config.action_type == "multi_discrete":
            self.action_space = gym.spaces.MultiDiscrete(self.config.action_config.nvec)
        elif self.config.action_type == "continuous":
            bounds = self.action_processor.get_action_bounds()
            self.action_space = gym.spaces.Box(
                low=np.array([bounds[0][0], bounds[1][0]], dtype=np.float32),
                high=np.array([bounds[0][1], bounds[1][1]], dtype=np.float32),
                shape=(2,),
                dtype=np.float32
            )
        else:
            raise ValueError(f"Unsupported action type: {self.config.action_type}")
        
        # Observation space
        obs_config = self.config.observation_config
        
        # Calculate observation shape
        if obs_config.use_multiscale:
            # Multi-scale observation
            base_channels = 4 + obs_config.use_trajectory  # Basic map channels
            multiscale_channels = base_channels * obs_config.n_scales
            
            if obs_config.use_global_features:
                total_channels = multiscale_channels + base_channels
            else:
                total_channels = multiscale_channels
            
            obs_shape = (total_channels, obs_config.multiscale_feature_size, obs_config.multiscale_feature_size)
        else:
            # Standard first-person observation
            base_channels = 4 + obs_config.use_trajectory
            obs_shape = (base_channels, obs_config.state_downsize[0], obs_config.state_downsize[1])
        
        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(
                low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
            ),
            "vector": gym.spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            ),
            "weed_ratio": gym.spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            )
        })
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset environment for new episode.
        
        Args:
            seed: Random seed
            options: Episode options
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Set random generators
        self.map_generator.set_random_generator(self.np_random)
        self.observation_manager.set_random_generator(self.np_random)
        
        # Parse options
        options = options or {}
        
        # Generate scenario
        self.maps_dict, scenario_info = self.map_generator.generate_scenario(
            map_id=options.get('map_id'),
            weed_distribution=options.get('weed_distribution', 'uniform'),
            weed_count=options.get('weed_count', 100),
            scenario_directory=options.get('specific_scenario_dir'),
            initial_position=options.get('initial_position'),
            initial_direction=options.get('initial_direction')
        )
        
        # Initialize agent
        self.agent = scenario_info['agent_info']['agent']
        
        # Reset environment state
        dimensions = scenario_info['dimensions']
        total_weed_count = scenario_info['total_weed_count']
        self.env_state.reset(dimensions, total_weed_count, self.config.max_episode_steps)
        
        # Initialize dynamics
        self.env_dynamics.reset(self.agent, self.maps_dict, self.env_state)
        
        # Generate initial observation
        observation = self._generate_observation()
        
        return observation, {}
    
    def step(self, action: Union[int, Tuple]) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Apply dynamics
        self.agent, self.maps_dict, self.env_state = self.env_dynamics.step(
            self.agent, self.maps_dict, self.env_state, action, self.config.action_type
        )
        
        # Calculate reward
        reward = self.reward_manager.calculate_step_reward(self.env_state)
        
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
        
        return observation, float(reward), terminated, truncated, info
    
    def _generate_observation(self) -> Dict[str, np.ndarray]:
        """Generate observation from current state."""
        # Prepare maps for observation
        maps_for_obs = {
            'field_frontier': {
                'map': self.maps_dict['field_frontier'], 
                'pad': 0.0
            },
            'obstacle': {
                'map': self.maps_dict['obstacle'], 
                'pad': 1.0
            },
            'weed': {
                'map': self.maps_dict['weed'], 
                'pad': 0.0
            }
        }
        
        # Add optional maps
        if self.config.observation_config.use_trajectory and 'trajectory' in self.maps_dict:
            maps_for_obs['trajectory'] = {
                'map': self.maps_dict['trajectory'],
                'pad': 0.0
            }
        
        # Generate visual observation
        visual_obs = self.observation_manager.generate_observation(self.agent, maps_for_obs)
        
        # Generate vector observation (normalized steering)
        vector_obs = np.array([self.agent.last_steer / self.config.action_config.w_range.max], 
                            dtype=np.float32)
        
        # Generate weed ratio
        weed_ratio = np.array([self.env_state.weed_completion_ratio], dtype=np.float32)
        
        return {
            'observation': visual_obs,
            'vector': vector_obs,
            'weed_ratio': weed_ratio
        }
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render environment.
        
        Returns:
            Rendered image if render_mode is set, None otherwise
        """
        if self.render_mode is None:
            return None
        
        if self.render_mode not in self.metadata["render_modes"]:
            gym.logger.warn(f"Render mode {self.render_mode} not supported")
            return None
        
        if pygame is None:
            raise gym.error.DependencyNotInstalled(
                "pygame is not installed, run `pip install pygame`"
            )
        
        # Determine render mode
        if self.render_mode == "state_pixels":
            render_mode = "first_person"
            dimensions = self.config.observation_config.state_size
        else:
            render_mode = "map"
            dimensions = self.env_state.dimensions
        
        # Render image
        rendered = self.render_manager.render(
            self.maps_dict, 
            self.agent, 
            self.env_state.dimensions,
            mode=render_mode,
            observation_size=dimensions if render_mode == "first_person" else None
        )
        
        # Handle pygame surface creation
        if self.screen is None:
            pygame.init()
            h, w = rendered.shape[:2]
            self.screen = pygame.Surface((w, h))
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # Convert to pygame surface
        surf = pygame.surfarray.make_surface(rendered.swapaxes(0, 1))
        self.screen.blit(surf, (0, 0))
        
        # Return as numpy array
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), 
            axes=(1, 0, 2)
        )
    
    def close(self) -> None:
        """Close environment and cleanup resources."""
        if self.screen is not None and pygame is not None:
            pygame.display.quit()
            pygame.quit()
            self.is_open = False
    
    def get_reward_breakdown(self) -> Dict[str, float]:
        """Get detailed reward breakdown for analysis."""
        return self.reward_manager.get_reward_breakdown(self.env_state)
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get current environment state information."""
        return self.env_state.to_dict()
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update environment configuration.
        
        Args:
            new_config: New configuration values
        """
        # Update reward coefficients if provided
        if 'reward_coefficients' in new_config:
            self.reward_manager.update_config(new_config['reward_coefficients'])
        
        # Additional config updates can be added here
    
    def set_action_type(self, action_type: str) -> None:
        """Change action type and reinitialize action space."""
        if action_type not in ["discrete", "continuous", "multi_discrete"]:
            raise ValueError(f"Unsupported action type: {action_type}")
        
        self.config.action_type = action_type
        self._initialize_spaces()
    
    def get_collision_info(self) -> Dict[str, bool]:
        """Get detailed collision information."""
        return self.env_dynamics.get_collision_info(self.agent, self.maps_dict)


# Factory function for backward compatibility
def create_cpp_env_base(**kwargs) -> CppEnvBase:
    """
    Create CppEnvBase with backward-compatible parameter interface.
    
    Args:
        **kwargs: Environment parameters
        
    Returns:
        Configured environment instance
    """
    # Map old parameter names to new config structure
    config_mapping = {
        # Map parameters
        'map_dir': ('map_config', 'map_dir'),
        'num_obstacles_range': ('map_config', 'num_obstacles_range'),
        'use_box_boundary': ('map_config', 'use_box_boundary'),
        'weed_noise': ('map_config', 'weed_noise'),
        
        # Observation parameters
        'state_size': ('observation_config', 'state_size'),
        'state_downsize': ('observation_config', 'state_downsize'),
        'use_multiscale': ('observation_config', 'use_multiscale'),
        'use_global_features': ('observation_config', 'use_global_features'),
        'position_noise': ('observation_config', 'position_noise'),
        'direction_noise': ('observation_config', 'direction_noise'),
        
        # Environment parameters
        'max_episode_steps': ('max_episode_steps',),
        'action_type': ('action_type',),
    }
    
    # Build nested config dictionary
    config_dict = {}
    render_mode = kwargs.pop('render_mode', None)
    
    for old_key, value in kwargs.items():
        if old_key in config_mapping:
            path = config_mapping[old_key]
            if len(path) == 1:
                config_dict[path[0]] = value
            else:
                if path[0] not in config_dict:
                    config_dict[path[0]] = {}
                config_dict[path[0]][path[1]] = value
    
    # Create config and environment
    config = EnvironmentConfig.from_dict(config_dict)
    return CppEnvBase(config=config, render_mode=render_mode)


if __name__ == "__main__":
    # Test the new environment
    env = CppEnvBase(render_mode='rgb_array')
    
    obs, info = env.reset()
    print(f"Observation shape: {obs['observation'].shape}")
    print(f"Vector shape: {obs['vector'].shape}")
    print(f"Weed ratio: {obs['weed_ratio']}")
    
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step reward: {reward:.3f}, Done: {terminated or truncated}")
        
        if terminated or truncated:
            break
    
    env.close()
    print("Environment test completed successfully!")