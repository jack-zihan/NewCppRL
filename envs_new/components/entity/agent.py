"""
Agent entities for the mowing robot simulation.
Provides abstract base classes and concrete implementations for different agent types.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np

from envs_new.components.config.environment_config import EnvironmentConfig


class Agent(ABC):
    """Abstract base class for all agents in the environment."""

    def __init__(self, config: EnvironmentConfig, initial_position: Tuple[float, float] = (0.0, 0.0),
                 initial_direction: float = 0.0):
        self.config = config
        self._x, self._y = initial_position
        self._direction = initial_direction % 360.0 # agent的角度确实是东为0°顺时针增大
        self._speed, self._steer = 0.0, 0.0
        self._backup_state()

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @property
    def direction(self) -> float:
        return self._direction

    @property
    def position(self) -> Tuple[float, float]:
        return self._x, self._y

    @property
    def position_discrete(self) -> Tuple[int, int]:
        return int(self._x), int(self._y) # 这里离散网格应该落在floor中而不是round

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def steer(self) -> float:
        return self._steer

    @property
    def last_speed(self) -> float:
        return self._last_speed

    @property
    def last_steer(self) -> float:
        return self._last_steer

    @property
    def width(self) -> float:
        return self.config.agent_width

    @property
    def length(self) -> float:
        return self.config.agent_length

    @property
    def vision_length(self) -> float:
        return self.config.agent_vision_length

    @property
    def vision_angle(self) -> float:
        return self.config.agent_vision_angle

    @property
    def occupancy(self) -> float:
        """Agent's occupancy radius (diagonal distance from center to corner)."""
        return math.hypot(self.width, self.length)

    @property
    def lw_ratio(self) -> float:
        """Length-width ratio angle used for convex hull corner calculations."""
        return math.degrees(math.atan2(self.width / self.occupancy, self.length / self.occupancy))

    @property
    def convex_hull(self) -> np.ndarray:
        """Agent's current convex hull without extra padding."""
        offsets = np.array([self.lw_ratio, 180 - self.lw_ratio, 180 + self.lw_ratio, -self.lw_ratio]) # 计算偏移
        angles = np.radians(self._direction + offsets)  # 转换为弧度
        return np.column_stack([self._x + self.width * np.cos(angles),self._y + self.width * np.sin(angles)])  # 计算顶点坐标 shape (4,2)

    @property
    def extended_convex_hull(self) -> np.ndarray:
        """Return convex hull optionally expanded by ``padding_px`` pixels."""
        if self.config.coverage_extended_px <= 0.0: # 没有padding，直接返回当前凸包
            return self.convex_hull
        # 否则计算padding凸包
        center = np.array([self._x, self._y]) # 计算中心点
        offsets = self.convex_hull - center # 计算每个顶点相对于中心的偏移
        norms = np.linalg.norm(offsets, axis=1, keepdims=True) # 计算每个偏移的范数
        safe_norms = np.maximum(norms, 1e-6) # 避免除零
        scale = (norms + self.config.coverage_extended_px) / safe_norms # 计算缩放比例
        return center + offsets * scale # 计算扩展后的顶点坐标

    def reset(self, position: Tuple[float, float], direction: float) -> None:
        self._x, self._y = position
        self._direction = direction % 360.0
        self._backup_state()

    def set_position(self, x: float, y: float) -> None:
        self._x, self._y = x, y

    def set_direction(self, direction: float) -> None:
        self._direction = direction % 360.0

    def rollback_position(self) -> None:
        self._x, self._y, self._direction = self._last_x, self._last_y, self._last_direction

    def _backup_state(self, **kwargs):
        self._last_x, self._last_y, self._last_direction = self._x, self._y, self._direction
        self._last_speed, self._last_steer = self._speed, self._steer

    @abstractmethod
    def control(self, *args, **kwargs) -> None:
        """Apply control input to update agent state."""
        pass


class MowerAgent(Agent):
    """Concrete implementation of a mowing robot agent with differential drive dynamics."""

    def control(self, speed: float, steer: float) -> None:
        """
        Apply differential drive control to update agent pose.
        Updates direction first, then calculates position change based on new heading.
        """
        self._backup_state()

        # Update direction first
        self._direction = (self._direction + steer) % 360

        # Calculate position change using updated direction
        rad = math.radians(self._direction)
        dx, dy = speed * math.cos(rad), speed * math.sin(rad)

        self._x += dx
        self._y += dy
        self._speed, self._steer = speed, steer

    def clip_to_bounds(self, width: float, height: float) -> None:
        self._x = float(np.clip(self._x, 0, width - 1))  # 确保0->width-1开区间
        self._y = float(np.clip(self._y, 0, height - 1))


class RealAgent(Agent):
    """Agent implementation for real robot data, where pose is set directly."""

    def control(self, new_position: Tuple[float, float], new_direction: float) -> None:
        """Set agent pose directly from real robot telemetry data."""
        self._backup_state()
        self._x, self._y = new_position
        self._direction = new_direction % 360


class AgentFactory:
    """Factory for creating different types of agents."""

    @staticmethod
    def create_mower_agent(config: EnvironmentConfig,
                           position: Tuple[float, float] = (0.0, 0.0), direction: float = 0.0) -> MowerAgent:
        return MowerAgent(config, position, direction)

    @staticmethod
    def create_real_agent(config: EnvironmentConfig,
                          position: Tuple[float, float] = (0.0, 0.0), direction: float = 0.0) -> RealAgent:
        return RealAgent(config, position, direction)
