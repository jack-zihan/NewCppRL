"""
Collision detection for the mowing robot environment.
Provides efficient collision checking between agents and obstacles.
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, Tuple

from envs_new.components.entity.agent import Agent


class CollisionDetector:
    """Handles collision detection between agents and environment obstacles."""
    
    def __init__(self):
        """Initialize collision detector."""
        pass
    
    def check_collision(self, agent: Agent, maps_dict: Dict[str, np.ndarray]) -> bool:
        """
        Check if agent collides with obstacles or boundaries.
        
        Args:
            agent: Agent to check collision for
            maps_dict: Dictionary containing environment maps
            
        Returns:
            True if collision detected, False otherwise
            
        Raises:
            ValueError: If obstacle map not found
        """
        if 'obstacle' not in maps_dict:
            raise ValueError("Obstacle map required for collision detection")
        
        obstacle_map = maps_dict['obstacle']
        dimensions = obstacle_map.shape  # (height, width)
        
        # Check boundary collision
        boundary_collision = self._check_boundary_collision(agent, dimensions)
        
        # Check obstacle collision
        obstacle_collision = self._check_obstacle_collision(agent, obstacle_map)
        
        return boundary_collision or obstacle_collision
    
    def _check_boundary_collision(self, agent: Agent, dimensions: Tuple[int, int]) -> bool:
        """
        Check if agent collides with map boundaries.
        
        Args:
            agent: Agent to check
            dimensions: Map dimensions (height, width)
            
        Returns:
            True if boundary collision detected
        """
        height, width = dimensions
        convex_hull = agent.convex_hull
        
        # Check if any point of the agent's convex hull is outside bounds
        x_coords = convex_hull[:, 0]
        y_coords = convex_hull[:, 1]
        
        # Check bounds: x in [0, width), y in [0, height)
        x_in_bounds = np.all((x_coords >= 0) & (x_coords < width))
        y_in_bounds = np.all((y_coords >= 0) & (y_coords < height))
        
        return not (x_in_bounds and y_in_bounds)
    
    def _check_obstacle_collision(self, agent: Agent, obstacle_map: np.ndarray) -> bool:
        """
        Check if agent collides with obstacles.
        
        Args:
            agent: Agent to check
            obstacle_map: Binary obstacle map
            
        Returns:
            True if obstacle collision detected
        """
        # Create agent occupancy map
        agent_map = self._create_agent_occupancy_map(agent, obstacle_map.shape)
        
        # Check for overlap with obstacles
        collision = np.any(np.logical_and(agent_map, obstacle_map))
        
        return collision
    
    def _create_agent_occupancy_map(self, agent: Agent, map_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create binary map showing agent's occupied area.
        
        Args:
            agent: Agent to create map for
            map_shape: Shape of the output map (height, width)
            
        Returns:
            Binary occupancy map
        """
        agent_map = np.zeros(map_shape, dtype=np.uint8)
        
        # Get agent's convex hull and round to integers
        convex_hull = agent.convex_hull.round().astype(np.int32)
        
        # Fill agent's occupied area
        cv2.fillPoly(agent_map, [convex_hull], color=1)
        
        return agent_map
    
    def get_collision_details(self, agent: Agent, maps_dict: Dict[str, np.ndarray]) -> Dict[str, bool]:
        """
        Get detailed collision information.
        
        Args:
            agent: Agent to check
            maps_dict: Environment maps
            
        Returns:
            Dictionary with collision details
        """
        if 'obstacle' not in maps_dict:
            raise ValueError("Obstacle map required for collision detection")
        
        obstacle_map = maps_dict['obstacle']
        dimensions = obstacle_map.shape
        
        boundary_collision = self._check_boundary_collision(agent, dimensions)
        obstacle_collision = self._check_obstacle_collision(agent, obstacle_map)
        
        return {
            'any_collision': boundary_collision or obstacle_collision,
            'boundary_collision': boundary_collision,
            'obstacle_collision': obstacle_collision
        }
    
    def get_safe_position(self, agent: Agent, maps_dict: Dict[str, np.ndarray]) -> Tuple[float, float]:
        """
        Get nearest safe position for agent if currently in collision.
        
        Args:
            agent: Agent in collision
            maps_dict: Environment maps
            
        Returns:
            Safe position (x, y)
        """
        if 'obstacle' not in maps_dict:
            raise ValueError("Obstacle map required")
        
        obstacle_map = maps_dict['obstacle']
        height, width = obstacle_map.shape
        
        # Current position
        current_x, current_y = agent.position
        
        # If not in collision, return current position
        if not self.check_collision(agent, maps_dict):
            return current_x, current_y
        
        # Clip to map bounds first
        safe_x = np.clip(current_x, 0, width - 1)
        safe_y = np.clip(current_y, 0, height - 1)
        
        # If still in obstacle, find nearest free space
        # This is a simple implementation - could be improved with distance transform
        for radius in range(1, min(width, height) // 2):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    test_x = safe_x + dx
                    test_y = safe_y + dy
                    
                    # Check bounds
                    if not (0 <= test_x < width and 0 <= test_y < height):
                        continue
                    
                    # Check if position is free
                    if obstacle_map[int(test_y), int(test_x)] == 0:
                        # Test agent at this position
                        test_agent = Agent.__new__(type(agent))
                        test_agent.__dict__ = agent.__dict__.copy()
                        test_agent.set_position(test_x, test_y)
                        
                        if not self.check_collision(test_agent, maps_dict):
                            return test_x, test_y
        
        # Fallback: return center of map
        return width / 2, height / 2