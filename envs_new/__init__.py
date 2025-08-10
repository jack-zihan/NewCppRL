"""
Modern modular CppEnv package with improved architecture.

This package provides a clean, maintainable implementation of the mowing robot
environment with the following key features:

- Modular component system for easy extension and testing
- Type-safe configuration management  
- Composable reward and observation systems
- Multiple environment variants (v1, v2, v3)
- Performance monitoring and debugging tools
- Environment factory for simplified creation

Usage:
    # Basic usage
    from envs_new import make_env
    env = make_env('v1', render_mode='human')
    
    # Factory usage
    from envs_new import EnvironmentFactory
    env = EnvironmentFactory.create_simple_env()
    
    # Direct class usage
    from envs_new.cpp_env_v2 import CppEnv
    env = CppEnv(render_mode='rgb_array')
"""

# Core environment classes
from envs_new.cpp_env_base import CppEnvBase
from envs_new.cpp_env_v1 import CppEnv as CppEnvV1
from envs_new.cpp_env_v2 import CppEnv as CppEnvV2  
from envs_new.cpp_env_v3 import CppEnv as CppEnvV3

# Factory and convenience functions
from envs_new.environment_factory import (
    EnvironmentFactory,
    make_env
)

# Configuration system
from envs_new.components.config.environment_config import EnvironmentConfig

# Performance monitoring
from envs_new.components.performance_monitor import (
    performance_monitor,
    measure_time,
    monitor_performance,
    get_performance_summary,
    print_performance_summary,
    reset_performance_metrics
)

# Component classes for extension
from envs_new.components.reward.reward_system import RewardSystem
from envs_new.components.observation.observation_generator import ObservationGenerator
from envs_new.components.map.map_generator import ScenarioGenerator

__version__ = "2.0.0"

__all__ = [
    # Core environments
    'CppEnvBase',
    'CppEnvV1', 
    'CppEnvV2',
    'CppEnvV3',
    
    # Factory system
    'EnvironmentFactory',
    'make_env',
    
    # Configuration
    'EnvironmentConfig',
    
    # Performance monitoring
    'performance_monitor',
    'measure_time',
    'monitor_performance', 
    'get_performance_summary',
    'print_performance_summary',
    'reset_performance_metrics',
    
    # Extension points
    'RewardSystem',
    'ObservationGenerator',
    'ScenarioGenerator',
]