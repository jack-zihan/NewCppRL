"""
Map generation coordinator for the mowing robot environment.
Orchestrates loading, obstacle generation, and weed placement.
"""
from __future__ import annotations

import math
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

from envs_new.components.config.environment_config import MapConfig, AgentConfig
from envs_new.components.entity.agent import AgentFactory
from envs_new.components.map.map_loader import MapLoader
from envs_new.components.map.obstacle_generator import ObstacleGenerator
from envs_new.components.map.weed_manager import WeedManager


class MapGenerator:
    """Coordinates map generation using specialized components."""
    
    def __init__(self, map_config: MapConfig, agent_config: AgentConfig):
        """
        Initialize map generator with required configurations.
        
        Args:
            map_config: Map configuration
            agent_config: Agent configuration
        """
        self.map_config = map_config
        self.agent_config = agent_config
        
        # Initialize specialized components
        self.map_loader = MapLoader(map_config)
        self.obstacle_generator = ObstacleGenerator(map_config, agent_config)
        self.weed_manager = WeedManager(map_config)
        
        # State tracking
        self.current_dimensions: Tuple[int, int] = (0, 0)
        self.bounding_box: List[np.ndarray] = []
        self.rng: Optional[np.random.Generator] = None
    
    def set_random_generator(self, rng: np.random.Generator) -> None:
        """Set random number generator for all components."""
        self.rng = rng
        self.obstacle_generator.set_random_generator(rng)
        self.weed_manager.set_random_generator(rng)
    
    def generate_scenario(self, 
                         map_id: Optional[int] = None,
                         weed_distribution: str = "uniform",
                         weed_count: int = 100,
                         scenario_directory: Optional[Union[str, Path]] = None,
                         initial_position: Optional[Tuple[float, float]] = None,
                         initial_direction: Optional[float] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, any]]:
        """
        Generate complete scenario with maps and initial agent.
        
        Args:
            map_id: Map ID to load (ignored if scenario_directory provided)
            weed_distribution: Weed distribution type ("uniform" or "gaussian")
            weed_count: Number of weeds to place
            scenario_directory: Directory containing pre-made scenario
            initial_position: Override agent initial position
            initial_direction: Override agent initial direction
            
        Returns:
            Tuple of (maps_dict, scenario_info)
        """
        if self.rng is None:
            raise ValueError("Random generator not set. Call set_random_generator() first.")
        
        if scenario_directory is not None:
            # Load from directory
            maps_dict, agent_info = self._load_scenario_from_directory(
                scenario_directory, initial_position, initial_direction
            )
        else:
            # Generate random scenario
            maps_dict, agent_info = self._generate_random_scenario(
                map_id, weed_distribution, weed_count, initial_position, initial_direction
            )
        
        # Add metadata
        scenario_info = {
            'dimensions': self.current_dimensions,
            'bounding_box': self.bounding_box,
            'total_weed_count': self.weed_manager.get_total_weed_count(),
            'agent_info': agent_info
        }
        
        return maps_dict, scenario_info
    
    def _load_scenario_from_directory(self, 
                                    directory: Union[str, Path],
                                    initial_position: Optional[Tuple[float, float]],
                                    initial_direction: Optional[float]) -> Tuple[Dict[str, np.ndarray], Dict[str, any]]:
        """Load scenario from pre-made directory."""
        # Load maps
        base_maps = self.map_loader.load_scenario_from_directory(directory)
        
        # Update dimensions and extract bounding box
        frontier_map = base_maps['frontier']
        self.current_dimensions = frontier_map.shape[::-1]  # (width, height)
        self.bounding_box = self._extract_bounding_box(frontier_map)
        
        # Initialize agent
        agent_pos, agent_dir = self._get_agent_initial_pose(initial_position, initial_direction)
        agent = AgentFactory.create_mower_agent(self.agent_config, agent_pos, agent_dir)
        
        # Build complete maps dictionary
        maps_dict = self._build_maps_dict(base_maps)
        
        agent_info = {
            'position': agent_pos,
            'direction': agent_dir,
            'agent': agent
        }
        
        return maps_dict, agent_info
    
    def _generate_random_scenario(self,
                                map_id: Optional[int],
                                weed_distribution: str,
                                weed_count: int,
                                initial_position: Optional[Tuple[float, float]],
                                initial_direction: Optional[float]) -> Tuple[Dict[str, np.ndarray], Dict[str, any]]:
        """Generate random scenario."""
        # Load base frontier map
        if map_id is None:
            map_id = self.rng.integers(0, self.map_loader.get_map_count())
        
        frontier_map, self.current_dimensions = self.map_loader.load_frontier_map(map_id)
        self.bounding_box = self._extract_bounding_box(frontier_map)
        
        # Initialize agent
        agent_pos, agent_dir = self._get_agent_initial_pose(initial_position, initial_direction)
        agent = AgentFactory.create_mower_agent(self.agent_config, agent_pos, agent_dir)
        
        # Generate obstacles
        obstacle_map = self.obstacle_generator.generate_obstacles(
            self.current_dimensions, frontier_map, agent_pos
        )
        
        # Add boundary obstacles if enabled
        if self.map_config.use_box_boundary:
            boundary_map = self.obstacle_generator.generate_boundary(
                self.current_dimensions, self.bounding_box
            )
            obstacle_map = np.logical_or(obstacle_map, boundary_map).astype(np.uint8)
        
        # Generate weed distribution
        weed_map, noisy_weed_map = self.weed_manager.generate_weed_distribution(
            frontier_map, weed_distribution, weed_count
        )
        
        # Apply obstacle exclusion to weeds
        weed_map = self.weed_manager.apply_obstacle_exclusion(weed_map, obstacle_map)
        noisy_weed_map = self.weed_manager.apply_obstacle_exclusion(noisy_weed_map, obstacle_map)
        
        # Build maps dictionary
        base_maps = {
            'frontier': frontier_map,
            'obstacle': obstacle_map,
            'weed': weed_map,
            'weed_noisy': noisy_weed_map
        }
        
        maps_dict = self._build_maps_dict(base_maps)
        
        agent_info = {
            'position': agent_pos,
            'direction': agent_dir,
            'agent': agent
        }
        
        return maps_dict, agent_info
    
    def _extract_bounding_box(self, frontier_map: np.ndarray) -> List[np.ndarray]:
        """Extract bounding box from frontier map."""
        contours, _ = cv2.findContours(frontier_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("No contours found in frontier map")
        
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        
        # Normalize box orientation
        start_idx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - start_idx, 0).astype(int)
        box = box.reshape((-1, 1, 2))
        
        return [box]
    
    def _get_agent_initial_pose(self, 
                              override_position: Optional[Tuple[float, float]],
                              override_direction: Optional[float]) -> Tuple[Tuple[float, float], float]:
        """Get agent initial position and direction."""
        if override_position is not None and override_direction is not None:
            return override_position, override_direction
        
        if not self.bounding_box:
            # Fallback to center of map
            width, height = self.current_dimensions
            return (width / 2, height / 2), 0.0
        
        # Calculate from bounding box
        box = self.bounding_box[0]
        
        # Determine initial position (corner of bounding box)
        edge1 = box[1, 0] - box[0, 0]
        edge2 = box[2, 0] - box[1, 0]
        
        # Choose longer edge for direction
        if math.hypot(*edge1) > math.hypot(*edge2):
            pos_index = 0
            direction_vector = edge1
        else:
            pos_index = 1
            direction_vector = edge2
        
        position = (float(box[pos_index, 0, 0]), float(box[pos_index, 0, 1]))
        
        # Calculate direction
        direction_vector = direction_vector / math.hypot(*direction_vector)
        direction = math.degrees(math.atan2(direction_vector[1], direction_vector[0]))
        
        # Apply overrides if provided
        if override_position is not None:
            position = override_position
        if override_direction is not None:
            direction = override_direction
        
        return position, direction
    
    def _build_maps_dict(self, base_maps: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Build complete maps dictionary with additional layers."""
        height, width = self.current_dimensions[1], self.current_dimensions[0]
        
        maps_dict = {
            'field_frontier': base_maps['frontier'],
            'original_field_frontier': base_maps['frontier'].copy(),
            'obstacle': base_maps['obstacle'],
            'weed': base_maps['weed']
        }
        
        # Add optional maps
        if 'weed_noisy' in base_maps:
            maps_dict['weed_noisy'] = base_maps['weed_noisy']
        
        # Add original weed map if available
        original_weed = self.weed_manager.get_original_weed_map()
        if original_weed is not None:
            maps_dict['original_weed'] = original_weed
        
        # Add dynamic maps
        if self.map_config.use_traj:
            maps_dict['trajectory'] = np.zeros((height, width), dtype=np.uint8)
        
        if self.map_config.use_mist:
            maps_dict['mist'] = np.ones((height, width), dtype=np.uint8)
        
        return maps_dict
    
    def get_current_dimensions(self) -> Tuple[int, int]:
        """Get current map dimensions."""
        return self.current_dimensions
    
    def get_bounding_box(self) -> List[np.ndarray]:
        """Get current bounding box."""
        return self.bounding_box.copy()
    
    def validate_scenario(self, maps_dict: Dict[str, np.ndarray]) -> bool:
        """Validate that generated scenario is consistent."""
        return self.map_loader.validate_map_consistency(maps_dict)