"""
Action processing for the mowing robot environment.
Handles conversion between different action representations.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, Union
from envs_new.components.config.environment_config import ActionConfig


class ActionProcessor:
    """Processes and converts between different action representations."""
    
    def __init__(self, config: ActionConfig):
        """
        Initialize action processor.
        
        Args:
            config: Action configuration containing ranges and discretization
        """
        self.config = config
        self.v_range = config.v_range
        self.w_range = config.w_range
        self.nvec = config.nvec
    
    def parse_discrete_action(self, action: int) -> Tuple[float, float]:
        """
        Convert discrete action to continuous velocities.
        
        Args:
            action: Discrete action index
            
        Returns:
            Tuple of (linear_velocity, angular_velocity)
            
        Raises:
            ValueError: If action index is out of range
        """
        max_actions = self.nvec[0] * self.nvec[1]
        if not (0 <= action < max_actions):
            raise ValueError(f"Action {action} out of range [0, {max_actions-1}]")
        
        # Decode action
        acc_index = action // self.nvec[1]  # [0, nvec[0]-1]
        steer_index = action % self.nvec[1]  # [0, nvec[1]-1]
        
        # Convert to velocities
        linear_velocity = (self.v_range.min + 
                          (acc_index + 1) / self.nvec[0] * self.v_range.mode)
        angular_velocity = (self.w_range.min + 
                           steer_index / (self.nvec[1] - 1) * self.w_range.mode)
        
        return linear_velocity, angular_velocity
    
    def parse_multi_discrete_action(self, action: Tuple[int, int]) -> Tuple[float, float]:
        """
        Convert multi-discrete action to continuous velocities.
        
        Args:
            action: Tuple of (acceleration_index, steering_index)
            
        Returns:
            Tuple of (linear_velocity, angular_velocity)
            
        Raises:
            ValueError: If action indices are out of range
        """
        acc_index, steer_index = action
        
        if not (0 <= acc_index < self.nvec[0]):
            raise ValueError(f"Acceleration index {acc_index} out of range [0, {self.nvec[0]-1}]")
        if not (0 <= steer_index < self.nvec[1]):
            raise ValueError(f"Steering index {steer_index} out of range [0, {self.nvec[1]-1}]")
        
        # Convert to velocities
        linear_velocity = (self.v_range.min + 
                          (acc_index + 1) / self.nvec[0] * self.v_range.mode)
        angular_velocity = (self.w_range.min + 
                           steer_index / (self.nvec[1] - 1) * self.w_range.mode)
        
        return linear_velocity, angular_velocity
    
    def parse_continuous_action(self, action: Tuple[float, float]) -> Tuple[float, float]:
        """
        Validate and return continuous action.
        
        Args:
            action: Tuple of (linear_velocity, angular_velocity)
            
        Returns:
            Validated tuple of (linear_velocity, angular_velocity)
            
        Raises:
            ValueError: If velocities are out of valid range
        """
        linear_velocity, angular_velocity = action
        
        if not (self.v_range.min <= linear_velocity <= self.v_range.max):
            raise ValueError(f"Linear velocity {linear_velocity} out of range "
                           f"[{self.v_range.min}, {self.v_range.max}]")
        
        if not (self.w_range.min <= angular_velocity <= self.w_range.max):
            raise ValueError(f"Angular velocity {angular_velocity} out of range "
                           f"[{self.w_range.min}, {self.w_range.max}]")
        
        return linear_velocity, angular_velocity
    
    def parse_action(self, action: Union[int, Tuple[int, int], Tuple[float, float]], 
                    action_type: str) -> Tuple[float, float]:
        """
        Parse action based on type.
        
        Args:
            action: Action in any supported format
            action_type: Type of action ("discrete", "multi_discrete", "continuous")
            
        Returns:
            Tuple of (linear_velocity, angular_velocity)
            
        Raises:
            ValueError: If action type is unsupported or action is invalid
        """
        if action_type == "discrete":
            if not isinstance(action, (int, np.integer)):
                raise ValueError(f"Expected int for discrete action, got {type(action)}")
            return self.parse_discrete_action(int(action))
        
        elif action_type == "multi_discrete":
            if not (isinstance(action, (tuple, list)) and len(action) == 2):
                raise ValueError(f"Expected tuple of length 2 for multi_discrete action, got {action}")
            return self.parse_multi_discrete_action(tuple(action))
        
        elif action_type == "continuous":
            if not (isinstance(action, (tuple, list)) and len(action) == 2):
                raise ValueError(f"Expected tuple of length 2 for continuous action, got {action}")
            return self.parse_continuous_action(tuple(action))
        
        else:
            raise ValueError(f"Unsupported action type: {action_type}")
    
    def clip_action(self, linear_velocity: float, angular_velocity: float) -> Tuple[float, float]:
        """
        Clip velocities to valid ranges.
        
        Args:
            linear_velocity: Linear velocity to clip
            angular_velocity: Angular velocity to clip
            
        Returns:
            Clipped velocities
        """
        clipped_linear = max(self.v_range.min, min(linear_velocity, self.v_range.max))
        clipped_angular = max(self.w_range.min, min(angular_velocity, self.w_range.max))
        
        return clipped_linear, clipped_angular
    
    def action_to_discrete(self, linear_velocity: float, angular_velocity: float) -> int:
        """
        Convert continuous velocities to discrete action index.
        
        Args:
            linear_velocity: Linear velocity
            angular_velocity: Angular velocity
            
        Returns:
            Discrete action index
        """
        # Clip to valid ranges
        linear_velocity = max(self.v_range.min, min(linear_velocity, self.v_range.max))
        angular_velocity = max(self.w_range.min, min(angular_velocity, self.w_range.max))
        
        # Convert to indices
        acc_ratio = (linear_velocity - self.v_range.min) / self.v_range.mode
        acc_index = max(0, min(int(acc_ratio * self.nvec[0]) - 1, self.nvec[0] - 1))
        
        steer_ratio = (angular_velocity - self.w_range.min) / self.w_range.mode
        steer_index = max(0, min(int(steer_ratio * (self.nvec[1] - 1)), self.nvec[1] - 1))
        
        return acc_index * self.nvec[1] + steer_index
    
    def get_action_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Get action bounds for continuous actions.
        
        Returns:
            Tuple of ((v_min, v_max), (w_min, w_max))
        """
        return ((self.v_range.min, self.v_range.max), 
                (self.w_range.min, self.w_range.max))
    
    def get_action_space_size(self) -> int:
        """Get total number of discrete actions."""
        return self.nvec[0] * self.nvec[1]