"""
Obstacle generation for the mowing robot environment.
Handles random obstacle placement and boundary generation.
"""
from __future__ import annotations

import math
import cv2
import numpy as np
from typing import Tuple, List, Optional

from envs_new.components.config.environment_config import MapConfig, AgentConfig
from envs_new.components.entity.agent import Agent


class ObstacleGenerator:
    """Generates random obstacles and boundaries for the environment."""
    
    def __init__(self, map_config: MapConfig, agent_config: AgentConfig):
        """
        Initialize obstacle generator.
        
        Args:
            map_config: Map configuration
            agent_config: Agent configuration for safety distance calculations
        """
        self.map_config = map_config
        self.agent_config = agent_config
        self.rng: Optional[np.random.Generator] = None
    
    def set_random_generator(self, rng: np.random.Generator) -> None:
        """Set random number generator."""
        self.rng = rng
    
    def generate_obstacles(self, dimensions: Tuple[int, int], frontier_map: np.ndarray,
                          agent_position: Tuple[float, float]) -> np.ndarray:
        """
        Generate random obstacles in the environment.
        
        Args:
            dimensions: Map dimensions (width, height)
            frontier_map: Frontier map to modify (obstacles remove frontier areas)
            agent_position: Agent position to maintain safe distance
            
        Returns:
            Binary obstacle map
            
        Raises:
            ValueError: If random generator not set
        """
        if self.rng is None:
            raise ValueError("Random generator not set. Call set_random_generator() first.")
        
        width, height = dimensions
        obstacle_map = np.zeros((height, width), dtype=np.uint8)
        
        # Determine number of obstacles to generate
        num_obstacles = self._get_obstacle_count()
        
        if num_obstacles == 0:
            return obstacle_map
        
        # Generate obstacles with safety constraints
        obstacles_placed = 0
        max_attempts = num_obstacles * 10  # Prevent infinite loops
        
        for _ in range(max_attempts):
            if obstacles_placed >= num_obstacles:
                break
            
            # Generate random obstacle
            obstacle = self._generate_random_obstacle(dimensions)
            
            # Check if obstacle placement is valid
            if self._is_valid_obstacle_placement(obstacle, agent_position, obstacle_map):
                # Place obstacle
                cv2.fillPoly(obstacle_map, [obstacle], color=(1,))
                
                # Remove frontier area around obstacle
                expanded_obstacle = self._expand_obstacle(obstacle, 15)
                cv2.fillPoly(frontier_map, [expanded_obstacle], color=(0,))
                
                obstacles_placed += 1
        
        return obstacle_map
    
    def generate_boundary(self, dimensions: Tuple[int, int], 
                         bounding_box: List[np.ndarray]) -> np.ndarray:
        """
        Generate boundary obstacles around the environment.
        
        Args:
            dimensions: Map dimensions (width, height)
            bounding_box: List containing bounding box points
            
        Returns:
            Binary boundary obstacle map
        """
        if not self.map_config.use_box_boundary or not bounding_box:
            return np.zeros(dimensions[::-1], dtype=np.uint8)  # (height, width)
        
        width, height = dimensions
        boundary_map = np.ones((height, width), dtype=np.uint8)
        
        # Extract bounding box
        box = bounding_box[0]
        
        # Calculate expanded bounding box
        expanded_box = self._calculate_expanded_box(box)
        
        # Create boundary by filling everything as obstacle, then clearing the expanded box
        cv2.fillPoly(boundary_map, [expanded_box], color=(0,))
        
        return boundary_map
    
    def _get_obstacle_count(self) -> int:
        """Get number of obstacles to generate."""
        min_obstacles, max_obstacles = self.map_config.num_obstacles_range
        if max_obstacles <= 0:
            return 0
        return self.rng.integers(min_obstacles, max_obstacles + 1)
    
    def _generate_random_obstacle(self, dimensions: Tuple[int, int]) -> np.ndarray:
        """
        Generate a random rotated rectangular obstacle.
        
        Args:
            dimensions: Map dimensions (width, height)
            
        Returns:
            Array of obstacle corner points
        """
        width, height = dimensions
        
        # Random center position (with margin from edges)
        margin = 100
        center_x = self.rng.uniform(margin, width - margin)
        center_y = self.rng.uniform(margin, height - margin)
        
        # Random obstacle dimensions
        min_size, max_size = self.map_config.obstacle_size_range
        obs_length = self.rng.uniform(min_size, max_size)
        obs_width = self.rng.uniform(min_size, max_size)
        
        # Random rotation angle
        angle = self.rng.uniform(0., 360.)
        
        # Create rotated rectangle
        rotated_rect = cv2.RotatedRect(
            center=(center_x, center_y),
            size=(obs_length, obs_width),
            angle=angle
        )
        
        points = np.array(rotated_rect.points(), dtype=np.int32).reshape((-1, 1, 2))
        return points
    
    def _is_valid_obstacle_placement(self, obstacle: np.ndarray, 
                                   agent_position: Tuple[float, float],
                                   existing_obstacles: np.ndarray) -> bool:
        """
        Check if obstacle placement is valid.
        
        Args:
            obstacle: Obstacle points array
            agent_position: Current agent position
            existing_obstacles: Map of existing obstacles
            
        Returns:
            True if placement is valid, False otherwise
        """
        # Check if center is already occupied
        center = obstacle.mean(axis=0)[0]  # Get center point
        center_int = (int(center[1]), int(center[0]))  # (y, x) for array indexing
        
        if (0 <= center_int[0] < existing_obstacles.shape[0] and 
            0 <= center_int[1] < existing_obstacles.shape[1] and
            existing_obstacles[center_int]):
            return False
        
        # Check distance from agent
        agent_x, agent_y = agent_position
        distance_to_agent = cv2.pointPolygonTest(obstacle, (agent_x, agent_y), True)
        min_distance = -2.0 * self.agent_config.length  # Negative because point is outside
        
        return distance_to_agent < min_distance
    
    def _expand_obstacle(self, obstacle: np.ndarray, expansion: float) -> np.ndarray:
        """
        Expand obstacle by given amount.
        
        Args:
            obstacle: Original obstacle points
            expansion: Amount to expand in pixels
            
        Returns:
            Expanded obstacle points
        """
        # Get bounding rectangle and expand it
        rect = cv2.minAreaRect(obstacle)
        center, (width, height), angle = rect
        
        expanded_rect = cv2.RotatedRect(
            center=center,
            size=(width + expansion, height + expansion),
            angle=angle
        )
        
        return np.array(expanded_rect.points(), dtype=np.int32).reshape((-1, 1, 2))
    
    def _calculate_expanded_box(self, box: np.ndarray) -> np.ndarray:
        """
        Calculate expanded bounding box for boundary generation.
        
        Args:
            box: Original bounding box points
            
        Returns:
            Expanded bounding box points
        """
        # Calculate center and orientation
        center = 0.5 * (box[0, 0] + box[2, 0])
        
        # Calculate edge vectors
        edge1 = box[0, 0] - box[1, 0]
        edge2 = box[1, 0] - box[2, 0]
        
        # Determine which edge is width vs height
        if abs(edge2[1]) < abs(edge2[0]):
            width_vec, height_vec = edge1, edge2
        else:
            width_vec, height_vec = edge2, edge1
        
        # Calculate angle and dimensions
        angle = math.atan2(width_vec[1], width_vec[0]) * 180.0 / math.pi
        width = math.hypot(*width_vec)
        height = math.hypot(*height_vec)
        
        # Expand dimensions
        expanded_width = max(width * 1.2, width + 60)
        expanded_height = max(height * 1.2, height + 60)
        
        # Create expanded rectangle
        expanded_rect = cv2.RotatedRect(
            center=(center[0], center[1]),
            size=(expanded_width, expanded_height),
            angle=angle
        )
        
        box_points = cv2.boxPoints(expanded_rect)
        start_idx = box_points.sum(axis=1).argmin()
        box_points = np.roll(box_points, 4 - start_idx, 0)
        
        return box_points.reshape((-1, 1, 2)).astype(np.int32)