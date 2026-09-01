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

# ====================================================================
# Gymnasium Registration for new environments
# --------------------------------------------------------------------
import gymnasium.envs.registration

# Register v1-v5 environments with new IDs to avoid conflicts
gymnasium.envs.registration.register(
    id="NewPasture-v1",
    entry_point="envs_new.cpp_env_v1:CppEnv",
)

gymnasium.envs.registration.register(
    id="NewPasture-v2",
    entry_point="envs_new.cpp_env_v2:CppEnv",
)

gymnasium.envs.registration.register(
    id="NewPasture-v3",
    entry_point="envs_new.cpp_env_v3:CppEnv",
)

# Import v4/v5/v6 from their respective modules (optional in some configurations).
import importlib


def _register_optional(version: str):
    """Register NewPasture-v{version} if its module exists; surface real import bugs.

    只有目标模块**确实不存在**（ModuleNotFoundError 且 name 正是该模块）才视为可选、静默跳过；
    其余 ImportError —— 即一个**已存在**的 cpp_env_v{version} 内部真正的缺依赖/导入 bug ——
    一律重新抛出，而不是被吞成下游莫名其妙的 "env id not found"（fail-fast：准确报错优于优雅掩盖）。
    """
    module = f"envs_new.cpp_env_v{version}"
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as e:
        if e.name == module:
            return None  # 该版本模块在当前配置下不存在 —— 真·可选
        raise  # 模块内部的真实缺依赖/导入错误 —— 暴露出来
    gymnasium.envs.registration.register(
        id=f"NewPasture-v{version}",
        entry_point=f"{module}:CppEnv",
    )
    return mod.CppEnv


CppEnvV4 = _register_optional("4")
CppEnvV5 = _register_optional("5")
CppEnvV6 = _register_optional("6")