"""
Agent entities for the mowing robot simulation.
Provides abstract base classes and concrete implementations for different agent types.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np

from envs_new.components.config.environment_config import AgentConfig


class Agent(ABC):
    """Abstract base class for all agents in the environment."""
    
    def __init__(self, config: AgentConfig, initial_position: Tuple[float, float] = (0.0, 0.0), 
                 initial_direction: float = 0.0):
        """
        Initialize agent with configuration.
        
        Args:
            config: Agent configuration
            initial_position: Initial (x, y) position
            initial_direction: Initial direction in degrees
        """
        self.config = config
        self._x, self._y = initial_position
        self._direction = initial_direction % 360.0
        self._last_speed = 0.0
        self._last_steer = 0.0
    
    @property
    def x(self) -> float:
        """X coordinate."""
        return self._x
    
    @property
    def y(self) -> float:
        """Y coordinate."""
        return self._y
    
    @property
    def direction(self) -> float:
        """Direction in degrees."""
        return self._direction
    
    @property
    def position(self) -> Tuple[float, float]:
        """Current position as (x, y) tuple."""
        return self._x, self._y
    
    @property
    def position_discrete(self) -> Tuple[int, int]:
        """Current position as discrete (rounded) coordinates."""
        return round(self._x), round(self._y)
    
    @property
    def last_speed(self) -> float:
        """Last applied speed."""
        return self._last_speed
    
    @property
    def last_steer(self) -> float:
        """Last applied steering angle."""
        return self._last_steer
    
    @property
    def width(self) -> float:
        """Agent width."""
        return self.config.width
    
    @property
    def length(self) -> float:
        """Agent length."""
        return self.config.length
    
    @property
    def vision_length(self) -> float:
        """Vision range length."""
        return self.config.vision_length
    
    @property
    def vision_angle(self) -> float:
        """Vision field of view angle in degrees."""
        return self.config.vision_angle
    
    @property
    def occupancy(self) -> float:
        """Agent's occupancy radius (diagonal distance)."""
        return math.hypot(self.width, self.length)
    
    @property
    def lw_ratio(self) -> float:
        """Length-width ratio angle in degrees."""
        return math.degrees(math.atan2(self.width / self.occupancy, self.length / self.occupancy))
    
    @property
    def convex_hull(self) -> np.ndarray:
        """Agent's convex hull as array of corner points."""
        angles = [
            self._direction + self.lw_ratio,
            self._direction + 180 - self.lw_ratio,
            self._direction + 180 + self.lw_ratio,
            self._direction - self.lw_ratio
        ]
        
        hull_points = []
        for angle in angles:
            rad = math.radians(angle)
            x = self._x + self.width * math.cos(rad)
            y = self._y + self.width * math.sin(rad)
            hull_points.append((x, y))
        
        return np.array(hull_points)
    
    def reset(self, position: Tuple[float, float], direction: float) -> None:
        """
        Reset agent to initial state.
        
        Args:
            position: New position (x, y)
            direction: New direction in degrees
        """
        self._x, self._y = position
        self._direction = direction % 360.0
        self._last_speed = 0.0
        self._last_steer = 0.0
    
    def set_position(self, x: float, y: float) -> None:
        """Set agent position directly."""
        self._x, self._y = x, y
    
    def set_direction(self, direction: float) -> None:
        """Set agent direction directly."""
        self._direction = direction % 360.0
    
    @abstractmethod
    def control(self, *args, **kwargs) -> None:
        """Apply control input to update agent state."""
        pass


class MowerAgent(Agent):
    """Concrete implementation of a mowing robot agent with differential drive dynamics."""
    
    def control(self, speed: float, steer: float) -> None:
        """
        Apply differential drive control to update agent pose.
        
        Args:
            speed: Linear velocity
            steer: Angular velocity (steering rate)
        """
        self._last_speed = speed
        self._last_steer = steer
        
        # Update direction
        self._direction = (self._direction + steer) % 360
        
        # Update position based on direction and speed
        rad = math.radians(self._direction)
        dx = speed * math.cos(rad)
        dy = speed * math.sin(rad)
        
        self._x += dx
        self._y += dy
    
    def clip_to_bounds(self, width: float, height: float) -> None:
        """
        Clip agent position to stay within specified bounds.
        
        Args:
            width: Maximum x coordinate
            height: Maximum y coordinate
        """
        self._x = float(np.clip(self._x, 0, width))
        self._y = float(np.clip(self._y, 0, height))


class RealAgent(Agent):
    """Agent implementation for real robot data, where pose is set directly."""
    
    def control(self, new_position: Tuple[float, float], new_direction: float) -> None:
        """
        Set agent pose directly (for real robot telemetry).
        
        Args:
            new_position: New (x, y) position
            new_direction: New direction in degrees
        """
        self._x, self._y = new_position
        self._direction = new_direction % 360
        self._last_speed = 0.0  # Unknown for real data
        self._last_steer = 0.0  # Unknown for real data


class AgentFactory:
    """Factory for creating different types of agents."""
    
    @staticmethod
    def create_mower_agent(config: AgentConfig, 
                          position: Tuple[float, float] = (0.0, 0.0),
                          direction: float = 0.0) -> MowerAgent:
        """Create a mowing robot agent."""
        return MowerAgent(config, position, direction)
    
    @staticmethod
    def create_real_agent(config: AgentConfig,
                         position: Tuple[float, float] = (0.0, 0.0),
                         direction: float = 0.0) -> RealAgent:
        """Create a real robot agent."""
        return RealAgent(config, position, direction)