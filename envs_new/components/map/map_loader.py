"""
Map loading functionality for the mowing robot environment.
Handles loading maps from files and directories.
"""
from __future__ import annotations

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

from envs_new.components.config.environment_config import MapConfig


class MapLoader:
    """Handles loading and validation of map files."""
    
    def __init__(self, config: MapConfig):
        """
        Initialize map loader with configuration.
        
        Args:
            config: Map configuration containing directory and validation settings
        """
        self.config = config
        self.map_dir = Path(config.map_dir)
        self._validate_map_directory()
        self.map_names = sorted(os.listdir(self.map_dir))
        
    def _validate_map_directory(self) -> None:
        """Validate that map directory exists and is accessible."""
        if not self.map_dir.exists():
            raise FileNotFoundError(f"Map directory does not exist: {self.map_dir}")
        if not self.map_dir.is_dir():
            raise ValueError(f"Map path is not a directory: {self.map_dir}")
        if not any(self.map_dir.iterdir()):
            raise ValueError(f"Map directory is empty: {self.map_dir}")
    
    def get_map_count(self) -> int:
        """Get total number of available maps."""
        return len(self.map_names)
    
    def get_map_names(self) -> List[str]:
        """Get list of available map names."""
        return self.map_names.copy()
    
    def load_frontier_map(self, map_id: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Load frontier map by ID.
        
        Args:
            map_id: Index of map to load
            
        Returns:
            Tuple of (frontier_map, dimensions) where dimensions is (width, height)
            
        Raises:
            ValueError: If map_id is out of range
            FileNotFoundError: If map file doesn't exist
        """
        if not (0 <= map_id < len(self.map_names)):
            raise ValueError(f"Map ID {map_id} out of range [0, {len(self.map_names)-1}]")
        
        map_name = self.map_names[map_id]
        map_path = self.map_dir / map_name
        
        if not map_path.exists():
            raise FileNotFoundError(f"Map file not found: {map_path}")
        
        # Load image and convert to binary mask
        image = cv2.imread(str(map_path))
        if image is None:
            raise ValueError(f"Failed to load image from: {map_path}")
        
        # Convert to binary frontier map (any non-black pixel becomes farmland)
        frontier_map = (image.sum(axis=-1) > 0).astype(np.uint8)
        dimensions = frontier_map.shape[::-1]  # (width, height)
        
        return frontier_map, dimensions
    
    def load_scenario_from_directory(self, directory: Union[str, Path]) -> Dict[str, np.ndarray]:
        """
        Load complete scenario from directory containing frontier, obstacle, and weed maps.
        
        Args:
            directory: Path to directory containing map files
            
        Returns:
            Dictionary containing loaded maps
            
        Raises:
            FileNotFoundError: If required files are missing
            ValueError: If maps have inconsistent dimensions
        """
        directory = Path(directory)
        
        # Define required files
        required_files = {
            'frontier': 'map_frontier.png',
            'obstacle': 'map_obstacle.png', 
            'weed': 'map_weed.png'
        }
        
        # Check all required files exist
        missing_files = []
        for map_type, filename in required_files.items():
            if not (directory / filename).exists():
                missing_files.append(filename)
        
        if missing_files:
            raise FileNotFoundError(f"Missing required files in {directory}: {missing_files}")
        
        # Load all maps
        maps = {}
        base_dimensions = None
        
        for map_type, filename in required_files.items():
            file_path = directory / filename
            image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                raise ValueError(f"Failed to load image: {file_path}")
            
            # Convert to binary map
            binary_map = (image > 0).astype(np.uint8)
            
            # Check dimensions consistency
            if base_dimensions is None:
                base_dimensions = binary_map.shape
            elif binary_map.shape != base_dimensions:
                raise ValueError(f"Map {map_type} has inconsistent dimensions: "
                               f"{binary_map.shape} vs {base_dimensions}")
            
            maps[map_type] = binary_map
        
        # Post-process maps for consistency
        self._post_process_loaded_maps(maps)
        
        return maps
    
    def _post_process_loaded_maps(self, maps: Dict[str, np.ndarray]) -> None:
        """
        Post-process loaded maps to ensure consistency.
        
        Args:
            maps: Dictionary of loaded maps to process in-place
        """
        frontier_map = maps['frontier']
        obstacle_map = maps['obstacle']
        weed_map = maps['weed']
        
        # Ensure weeds only exist in frontier areas
        maps['weed'] = np.logical_and(weed_map, frontier_map).astype(np.uint8)
        
        # Apply obstacle dilation to prevent weeds near obstacles
        if self.config.weed_noise > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29))
            dilated_obstacles = cv2.dilate(obstacle_map, kernel, iterations=1)
            maps['weed'][dilated_obstacles > 0] = 0
    
    def validate_map_consistency(self, maps: Dict[str, np.ndarray]) -> bool:
        """
        Validate that maps have consistent dimensions and logical relationships.
        
        Args:
            maps: Dictionary of maps to validate
            
        Returns:
            True if maps are consistent, False otherwise
        """
        if not maps:
            return False
        
        # Check all maps have same dimensions
        base_shape = None
        for map_name, map_array in maps.items():
            if base_shape is None:
                base_shape = map_array.shape
            elif map_array.shape != base_shape:
                return False
        
        # Check logical relationships if all required maps present
        required_maps = {'frontier', 'obstacle', 'weed'}
        if required_maps.issubset(set(maps.keys())):
            frontier = maps['frontier']
            obstacle = maps['obstacle']
            weed = maps['weed']
            
            # Weeds should only exist where there's frontier
            if np.any(np.logical_and(weed, np.logical_not(frontier))):
                return False
            
            # Obstacles and frontier shouldn't overlap significantly
            overlap = np.logical_and(obstacle, frontier)
            if np.sum(overlap) > 0.1 * np.sum(frontier):  # Allow 10% overlap
                return False
        
        return True