"""
Rules_new 核心模块

包含系统核心功能：
- 坐标系统
- 异常体系
- 状态验证
- 性能监控
"""

from .coordinate_system import CoordinateSystem, CS
from .exceptions import (
    RulesNewError,
    AlgorithmError,
    ExperimentError,
    CoordinateError,
    StateError,
    ConfigurationError,
    EnvironmentError
)

__all__ = [
    'CoordinateSystem',
    'CS',
    'RulesNewError',
    'AlgorithmError', 
    'ExperimentError',
    'CoordinateError',
    'StateError',
    'ConfigurationError',
    'EnvironmentError'
]