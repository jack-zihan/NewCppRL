"""
Action processing for the mowing robot environment.
Handles conversion between different action representations.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, Union
from envs_new.components.config.environment_config import EnvironmentConfig


class ActionProcessor:
    """
    Processes and converts between different action representations.
    
    Key concepts:
    - Discrete actions: Single integer index [0, nvec[0]*nvec[1]-1]
    - Multi-discrete actions: Tuple of (acceleration_index, steering_index)
    - Continuous actions: Tuple of (linear_velocity, angular_velocity)
    - All conversions preserve exact mathematical relationships
    """
    
    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.v_range = config.v_range
        self.w_range = config.w_range
        self.nvec = config.action_nvec
        
        # 预计算查找表以避免运行时除法运算
        self._linear_velocity_table = [
            self.v_range.min + (i + 1) / self.nvec[0] * self.v_range.mode
            for i in range(self.nvec[0])
        ]
        self._angular_velocity_table = [
            self.w_range.min + i / (self.nvec[1] - 1) * self.w_range.mode
            for i in range(self.nvec[1])
        ]
    
    def parse_discrete_action(self, action: int) -> Tuple[float, float]:
        max_actions = self.nvec[0] * self.nvec[1]
        if not (0 <= action < max_actions):
            raise ValueError(f"Action {action} out of range [0, {max_actions-1}]")
        
        # Decode single index into 2D grid coordinates
        acc_index = action // self.nvec[1]  # [0, nvec[0]-1]
        steer_index = action % self.nvec[1]  # [0, nvec[1]-1]
        
        return self._indices_to_velocities(acc_index, steer_index)
    
    def _indices_to_velocities(self, acc_index: int, steer_index: int) -> Tuple[float, float]:
        """
        Core index-to-velocity conversion using precomputed lookup tables.
        
        Optimized version that avoids runtime division operations.
        """
        return self._linear_velocity_table[acc_index], self._angular_velocity_table[steer_index]
    
    def _velocities_to_indices(self, linear_velocity: float, angular_velocity: float) -> Tuple[int, int]:
        """
        Core velocity-to-index conversion (inverse of _indices_to_velocities).
        Preserves exact mathematical relationship and handles boundary conditions.
        """
        # Convert velocities to normalized ratios [0, 1]
        acc_ratio = (linear_velocity - self.v_range.min) / self.v_range.mode
        steer_ratio = (angular_velocity - self.w_range.min) / self.w_range.mode
        
        # Convert ratios to indices with boundary clamping
        acc_index = max(0, min(int(acc_ratio * self.nvec[0]) - 1, self.nvec[0] - 1))
        steer_index = max(0, min(int(steer_ratio * (self.nvec[1] - 1)), self.nvec[1] - 1))
        
        return acc_index, steer_index
    
    def _validate_and_convert_action(self, action, action_type: str):
        """Validate action type and format, convert to standard representation."""
        if action_type == "discrete":
            if not isinstance(action, (int, np.integer)):
                raise ValueError(f"Expected int for discrete action, got {type(action)}")
            return int(action)
        
        elif action_type in ["multi_discrete", "continuous"]:
            if not (isinstance(action, (tuple, list, np.ndarray)) and len(action) == 2):
                raise ValueError(f"Expected tuple of length 2 for {action_type} action, got {action}")
            return tuple(action)
        
        else:
            raise ValueError(f"Unsupported action type: {action_type}")
    
    def parse_multi_discrete_action(self, action: Tuple[int, int]) -> Tuple[float, float]:
        acc_index, steer_index = action
        
        if not (0 <= acc_index < self.nvec[0]):
            raise ValueError(f"Acceleration index {acc_index} out of range [0, {self.nvec[0]-1}]")
        if not (0 <= steer_index < self.nvec[1]):
            raise ValueError(f"Steering index {steer_index} out of range [0, {self.nvec[1]-1}]")
        
        return self._indices_to_velocities(acc_index, steer_index)
    
    def parse_continuous_action(self, action: Tuple[float, float]) -> Tuple[float, float]:
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
        """Unified action parsing interface for all action types."""
        validated_action = self._validate_and_convert_action(action, action_type)
        
        if action_type == "discrete":
            return self.parse_discrete_action(validated_action)
        elif action_type == "multi_discrete":
            return self.parse_multi_discrete_action(validated_action)
        else:  # continuous
            return self.parse_continuous_action(validated_action)
    
    def clip_action(self, linear_velocity: float, angular_velocity: float) -> Tuple[float, float]:
        clipped_linear = max(self.v_range.min, min(linear_velocity, self.v_range.max))
        clipped_angular = max(self.w_range.min, min(angular_velocity, self.w_range.max))
        return clipped_linear, clipped_angular
    
    def action_to_discrete(self, linear_velocity: float, angular_velocity: float) -> int:
        """Convert continuous velocities to discrete action index."""
        # Ensure velocities are within valid bounds
        linear_velocity, angular_velocity = self.clip_action(linear_velocity, angular_velocity)
        
        # Convert to grid indices and encode as single integer
        acc_index, steer_index = self._velocities_to_indices(linear_velocity, angular_velocity)
        return acc_index * self.nvec[1] + steer_index
    
    def get_action_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return ((self.v_range.min, self.v_range.max), 
                (self.w_range.min, self.w_range.max))
    
    def get_action_space_size(self) -> int:
        return self.nvec[0] * self.nvec[1]