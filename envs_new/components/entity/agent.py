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
        self._direction = initial_direction % 360.0
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
        return round(self._x), round(self._y)

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
        """
        Agent's convex hull as array of corner points.
        
        Calculates 4 corners of rectangular agent rotated by current direction.
        Uses trigonometric projection with lw_ratio for precise corner positioning.
        """
        return np.array([
            (self._x + 1.0 * self.width * math.cos(math.radians(self._direction + 0 + self.lw_ratio)),
             self._y + 1.0 * self.width * math.sin(math.radians(self._direction + 0 + self.lw_ratio))),
            (self._x + 1.0 * self.width * math.cos(math.radians(self._direction + 180 - self.lw_ratio)),
             self._y + 1.0 * self.width * math.sin(math.radians(self._direction + 180 - self.lw_ratio))),
            (self._x + 1.0 * self.width * math.cos(math.radians(self._direction + 180 + self.lw_ratio)),
             self._y + 1.0 * self.width * math.sin(math.radians(self._direction + 180 + self.lw_ratio))),
            (self._x + 1.0 * self.width * math.cos(math.radians(self._direction + 0 - self.lw_ratio)),
             self._y + 1.0 * self.width * math.sin(math.radians(self._direction + 0 - self.lw_ratio))),
        ])

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

    def _backup_state(self, speed: float = 0.0, steer: float = 0.0, **kwargs):
        self._last_x, self._last_y, self._last_direction = self._x, self._y, self._direction
        self._last_speed, self._last_steer = speed, steer

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
        self._backup_state(speed, steer)

        # Update direction first
        self._direction = (self._direction + steer) % 360

        # Calculate position change using updated direction
        rad = math.radians(self._direction)
        dx = speed * math.cos(rad)
        dy = speed * math.sin(rad)

        self._x += dx
        self._y += dy

    def clip_to_bounds(self, width: float, height: float) -> None:
        self._x = float(np.clip(self._x, 0, width))
        self._y = float(np.clip(self._y, 0, height))


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
                           position: Tuple[float, float] = (0.0, 0.0),
                           direction: float = 0.0) -> MowerAgent:
        return MowerAgent(config, position, direction)

    @staticmethod
    def create_real_agent(config: EnvironmentConfig,
                          position: Tuple[float, float] = (0.0, 0.0),
                          direction: float = 0.0) -> RealAgent:
        return RealAgent(config, position, direction)
