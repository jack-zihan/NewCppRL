"""
Environment configuration management for the mowing robot simulation.
Provides centralized, validated configuration for all environment components.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
from pathlib import Path


@dataclass
class NumericalRange:
    """Represents a numerical range with min, max, and computed mode."""
    min: float
    max: float
    
    @property
    def mode(self) -> float:
        """Returns the range span (max - min)."""
        return self.max - self.min
    
    def __post_init__(self):
        if self.min >= self.max:
            raise ValueError(f"min ({self.min}) must be less than max ({self.max})")


@dataclass
class MapConfig:
    """Configuration for map-related parameters."""
    map_dir: str = "envs/maps/1-400"
    num_obstacles_range: Tuple[int, int] = (5, 8)
    obstacle_size_range: Tuple[int, int] = (10, 25)
    use_box_boundary: bool = True
    weed_noise: float = 0.0
    use_traj: bool = True
    use_mist: bool = True
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        if not Path(self.map_dir).exists():
            raise ValueError(f"Map directory does not exist: {self.map_dir}")
        if self.num_obstacles_range[0] < 0 or self.num_obstacles_range[1] < self.num_obstacles_range[0]:
            raise ValueError(f"Invalid obstacle range: {self.num_obstacles_range}")
        if self.obstacle_size_range[0] <= 0 or self.obstacle_size_range[1] < self.obstacle_size_range[0]:
            raise ValueError(f"Invalid obstacle size range: {self.obstacle_size_range}")
        if not 0 <= self.weed_noise <= 1:
            raise ValueError(f"Weed noise must be in [0, 1], got {self.weed_noise}")


@dataclass  
class AgentConfig:
    """Configuration for agent parameters."""
    width: float = 4.0
    length: float = 6.0
    vision_length: float = 28.0
    vision_angle: float = 75.0
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        if self.width <= 0 or self.length <= 0:
            raise ValueError("Agent dimensions must be positive")
        if self.vision_length <= 0:
            raise ValueError("Vision length must be positive")
        if not 0 < self.vision_angle <= 360:
            raise ValueError("Vision angle must be in (0, 360]")


@dataclass
class ActionConfig:
    """Configuration for action space parameters."""
    v_range: NumericalRange = field(default_factory=lambda: NumericalRange(0.0, 3.5))
    w_range: NumericalRange = field(default_factory=lambda: NumericalRange(-28.6, 28.6))
    nvec: Tuple[int, int] = (7, 21)
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        if self.nvec[0] <= 0 or self.nvec[1] <= 0:
            raise ValueError("Action discretization must be positive")


@dataclass
class ObservationConfig:
    """Configuration for observation generation."""
    state_size: Tuple[int, int] = (128, 128)
    state_downsize: Tuple[int, int] = (128, 128)
    use_multiscale: bool = True
    n_scales: int = 4
    multiscale_feature_size: int = 16
    use_global_features: bool = True
    use_trajectory: bool = True
    use_mist: bool = True
    position_noise: float = 0.0
    direction_noise: float = 0.0
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        if any(dim <= 0 for dim in self.state_size):
            raise ValueError("State size dimensions must be positive")
        if any(dim <= 0 for dim in self.state_downsize):
            raise ValueError("State downsize dimensions must be positive")
        if self.n_scales <= 0:
            raise ValueError("Number of scales must be positive")
        if self.multiscale_feature_size <= 0:
            raise ValueError("Multiscale feature size must be positive")
        if self.position_noise < 0 or self.direction_noise < 0:
            raise ValueError("Noise parameters must be non-negative")


@dataclass
class RewardConfig:
    """Configuration for reward calculation."""
    coefficients: Dict[str, float] = field(default_factory=lambda: {
        'turn_total_coef': 0.0,
        'turn_gap_coef': -0.5,
        'turn_direction_coef': -0.30,
        'turn_self_coef': 0.25,
        'frontier_total_coef': 0.125,
        'frontier_coverage_coef': 1.0,
        'frontier_tv_coef': 0.5,
        'base_penalty': -0.1,
        'weed_removal_coef': 20.0,
        'collision_penalty': -399.0,
        'completion_bonus': 500.0
    })
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        required_keys = {
            'turn_total_coef', 'turn_gap_coef', 'turn_direction_coef', 'turn_self_coef',
            'frontier_total_coef', 'frontier_coverage_coef', 'frontier_tv_coef',
            'base_penalty', 'weed_removal_coef', 'collision_penalty', 'completion_bonus'
        }
        missing_keys = required_keys - set(self.coefficients.keys())
        if missing_keys:
            raise ValueError(f"Missing reward coefficients: {missing_keys}")


@dataclass
class RenderConfig:
    """Configuration for rendering parameters."""
    render_modes: Tuple[str, ...] = ("rgb_array", "state_pixels")
    render_fps: int = 50
    render_repeat_times: int = 2
    render_tv: bool = False
    render_mist: bool = False
    render_covered_weed: bool = True
    render_covered_farmland: bool = True
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        if self.render_fps <= 0:
            raise ValueError("Render FPS must be positive")
        if self.render_repeat_times <= 0:
            raise ValueError("Render repeat times must be positive")


@dataclass
class EnvironmentConfig:
    """Main configuration class that aggregates all component configurations."""
    
    # Core environment parameters
    action_type: str = "discrete"
    max_episode_steps: int = 3000
    
    # Component configurations
    map_config: MapConfig = field(default_factory=MapConfig)
    agent_config: AgentConfig = field(default_factory=AgentConfig)
    action_config: ActionConfig = field(default_factory=ActionConfig)
    observation_config: ObservationConfig = field(default_factory=ObservationConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    render_config: RenderConfig = field(default_factory=RenderConfig)
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        if self.action_type not in ["discrete", "continuous", "multi_discrete"]:
            raise ValueError(f"Unsupported action type: {self.action_type}")
        if self.max_episode_steps <= 0:
            raise ValueError("Max episode steps must be positive")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EnvironmentConfig':
        """Create configuration from dictionary."""
        # Extract component configurations
        map_config = MapConfig(**config_dict.get('map_config', {}))
        agent_config = AgentConfig(**config_dict.get('agent_config', {}))
        action_config = ActionConfig(**config_dict.get('action_config', {}))
        observation_config = ObservationConfig(**config_dict.get('observation_config', {}))
        reward_config = RewardConfig(**config_dict.get('reward_config', {}))
        render_config = RenderConfig(**config_dict.get('render_config', {}))
        
        # Create main config
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ['map_config', 'agent_config', 'action_config', 
                                   'observation_config', 'reward_config', 'render_config']}
        
        return cls(
            map_config=map_config,
            agent_config=agent_config,
            action_config=action_config,
            observation_config=observation_config,
            reward_config=reward_config,
            render_config=render_config,
            **main_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'action_type': self.action_type,
            'max_episode_steps': self.max_episode_steps,
            'map_config': self.map_config.__dict__,
            'agent_config': self.agent_config.__dict__,
            'action_config': {
                'v_range': {'min': self.action_config.v_range.min, 'max': self.action_config.v_range.max},
                'w_range': {'min': self.action_config.w_range.min, 'max': self.action_config.w_range.max},
                'nvec': self.action_config.nvec
            },
            'observation_config': self.observation_config.__dict__,
            'reward_config': self.reward_config.__dict__,
            'render_config': self.render_config.__dict__
        }