"""
Environment factory for creating different versions of CppEnv.
Simplifies environment instantiation and configuration.
"""
from __future__ import annotations

from typing import Dict, Any, Optional, Type, Union
from envs_new.cpp_env_base import CppEnvBase
from envs_new.cpp_env_v1 import CppEnv as CppEnvV1
from envs_new.cpp_env_v2 import CppEnv as CppEnvV2
from envs_new.cpp_env_v3 import CppEnv as CppEnvV3


class EnvironmentFactory:
    """Factory for creating CppEnv instances with predefined configurations."""
    
    # Registry of available environment versions
    _env_registry: Dict[str, Type[CppEnvBase]] = {
        'base': CppEnvBase,
        'v1': CppEnvV1,
        'v2': CppEnvV2, 
        'v3': CppEnvV3,
        'simple': CppEnvV1,        # Alias for v1
        'apf': CppEnvV2,           # Alias for v2
        'exploration': CppEnvV3,   # Alias for v3
    }
    
    @classmethod
    def create(cls, 
              version: str = 'base',
              render_mode: Optional[str] = None,
              **kwargs) -> CppEnvBase:
        """
        Create environment instance.
        
        Args:
            version: Environment version ('base', 'v1', 'v2', 'v3', 'simple', 'apf', 'exploration')
            render_mode: Rendering mode for the environment
            **kwargs: Additional configuration overrides
            
        Returns:
            Configured environment instance
            
        Raises:
            ValueError: If version is not supported
        """
        if version not in cls._env_registry:
            available = ', '.join(cls._env_registry.keys())
            raise ValueError(f"Unsupported environment version '{version}'. Available: {available}")
        
        env_class = cls._env_registry[version]
        return env_class(render_mode=render_mode, **kwargs)
    
    @classmethod
    def create_simple_env(cls, render_mode: Optional[str] = None, **kwargs) -> CppEnvV1:
        """Create simple environment without mist (v1)."""
        return cls.create('v1', render_mode=render_mode, **kwargs)
    
    @classmethod 
    def create_apf_env(cls, render_mode: Optional[str] = None, **kwargs) -> CppEnvV2:
        """Create APF-based environment (v2)."""
        return cls.create('v2', render_mode=render_mode, **kwargs)
    
    @classmethod
    def create_exploration_env(cls, render_mode: Optional[str] = None, **kwargs) -> CppEnvV3:
        """Create exploration environment with mist (v3)."""
        return cls.create('v3', render_mode=render_mode, **kwargs)
    
    @classmethod
    def list_available_versions(cls) -> list[str]:
        """Get list of available environment versions."""
        return list(cls._env_registry.keys())
    
    @classmethod
    def register_version(cls, name: str, env_class: Type[CppEnvBase]) -> None:
        """
        Register a new environment version.
        
        Args:
            name: Version name
            env_class: Environment class
        """
        cls._env_registry[name] = env_class


# Convenience functions for direct usage
def make_env(version: str = 'base', render_mode: Optional[str] = None, **kwargs) -> CppEnvBase:
    """
    Create environment using factory.
    
    Args:
        version: Environment version
        render_mode: Rendering mode  
        **kwargs: Additional configuration
        
    Returns:
        Environment instance
    """
    return EnvironmentFactory.create(version, render_mode, **kwargs)


def make_simple_env(render_mode: Optional[str] = None, **kwargs) -> CppEnvV1:
    """Create simple environment (v1)."""
    return EnvironmentFactory.create_simple_env(render_mode, **kwargs)


def make_apf_env(render_mode: Optional[str] = None, **kwargs) -> CppEnvV2:
    """Create APF environment (v2)."""
    return EnvironmentFactory.create_apf_env(render_mode, **kwargs)


def make_exploration_env(render_mode: Optional[str] = None, **kwargs) -> CppEnvV3:
    """Create exploration environment (v3)."""
    return EnvironmentFactory.create_exploration_env(render_mode, **kwargs)