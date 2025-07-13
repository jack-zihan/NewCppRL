"""
Weed management for the mowing robot environment.
Handles weed distribution, noise application, and tracking.
"""
from __future__ import annotations

import math
import cv2
import numpy as np
from typing import Tuple, Optional

from envs_new.components.config.environment_config import MapConfig


class WeedManager:
    """Manages weed distribution and state in the environment."""
    
    def __init__(self, map_config: MapConfig):
        """
        Initialize weed manager.
        
        Args:
            map_config: Map configuration containing weed parameters
        """
        self.map_config = map_config
        self.rng: Optional[np.random.Generator] = None
        self.original_weed_map: Optional[np.ndarray] = None
        self.total_weed_count: int = 0
    
    def set_random_generator(self, rng: np.random.Generator) -> None:
        """Set random number generator."""
        self.rng = rng
    
    def generate_weed_distribution(self, frontier_map: np.ndarray, 
                                 distribution: str = "uniform",
                                 weed_count: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate weed distribution in the frontier area.
        
        Args:
            frontier_map: Binary map of farmland frontier
            distribution: Distribution type ("uniform" or "gaussian")
            weed_count: Number of weeds to place
            
        Returns:
            Tuple of (weed_map, noisy_weed_map)
            
        Raises:
            ValueError: If random generator not set or invalid distribution
        """
        if self.rng is None:
            raise ValueError("Random generator not set. Call set_random_generator() first.")
        
        if distribution not in ["uniform", "gaussian"]:
            raise ValueError(f"Unsupported distribution: {distribution}")
        
        # Convert float weed count to actual count if needed
        if isinstance(weed_count, float):
            weed_count = math.ceil(frontier_map.sum() * weed_count)
        
        self.total_weed_count = weed_count
        
        # Generate base weed distribution
        if distribution == "uniform":
            weed_map = self._generate_uniform_distribution(frontier_map, weed_count)
        else:  # gaussian
            weed_map = self._generate_gaussian_distribution(frontier_map, weed_count)
        
        # Store original for tracking
        self.original_weed_map = weed_map.copy()
        
        # Apply noise if configured
        if self.map_config.weed_noise > 0:
            noisy_weed_map = self._apply_weed_noise(weed_map)
        else:
            noisy_weed_map = weed_map.copy()
        
        return weed_map, noisy_weed_map
    
    def apply_obstacle_exclusion(self, weed_map: np.ndarray, 
                               obstacle_map: np.ndarray) -> np.ndarray:
        """
        Remove weeds from areas near obstacles.
        
        Args:
            weed_map: Current weed map
            obstacle_map: Binary obstacle map
            
        Returns:
            Updated weed map with obstacles excluded
        """
        # Create dilated obstacle mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29))
        dilated_obstacles = cv2.dilate(obstacle_map, kernel, iterations=1)
        
        # Remove weeds from dilated obstacle areas
        cleaned_weed_map = weed_map.copy()
        cleaned_weed_map[dilated_obstacles > 0] = 0
        
        return cleaned_weed_map
    
    def get_original_weed_map(self) -> Optional[np.ndarray]:
        """Get the original weed map for rendering covered areas."""
        return self.original_weed_map.copy() if self.original_weed_map is not None else None
    
    def get_total_weed_count(self) -> int:
        """Get total number of weeds initially placed."""
        return self.total_weed_count
    
    def _generate_uniform_distribution(self, frontier_map: np.ndarray, 
                                     weed_count: int) -> np.ndarray:
        """
        Generate uniform weed distribution.
        
        Args:
            frontier_map: Binary frontier map
            weed_count: Number of weeds to place
            
        Returns:
            Binary weed map
        """
        weed_map = np.zeros_like(frontier_map, dtype=np.uint8)
        
        # Get all possible positions
        possible_positions = np.argwhere(frontier_map)
        
        if len(possible_positions) == 0:
            return weed_map
        
        # Randomly select positions
        actual_count = min(weed_count, len(possible_positions))
        self.rng.shuffle(possible_positions)
        selected_positions = possible_positions[:actual_count]
        
        # Place weeds
        weed_map[selected_positions[:, 0], selected_positions[:, 1]] = 1
        
        return weed_map
    
    def _generate_gaussian_distribution(self, frontier_map: np.ndarray,
                                      weed_count: int) -> np.ndarray:
        """
        Generate Gaussian weed distribution centered on the map.
        
        Args:
            frontier_map: Binary frontier map
            weed_count: Number of weeds to place
            
        Returns:
            Binary weed map
        """
        weed_map = np.zeros_like(frontier_map, dtype=np.uint8)
        height, width = frontier_map.shape
        
        # Gaussian parameters (centered on map)
        center_y, center_x = height / 2, width / 2
        scale_y, scale_x = height * 0.35, width * 0.35
        
        # Generate more candidates than needed to account for filtering
        candidates = self.rng.normal(
            loc=[center_y, center_x],
            scale=[scale_y, scale_x],
            size=(weed_count * 5, 2)
        )
        
        # Clip to map bounds and convert to integers
        candidates = np.round(candidates).astype(int)
        candidates = np.clip(candidates, [0, 0], [height - 1, width - 1])
        
        # Remove duplicates
        unique_candidates = np.unique(candidates, axis=0)
        
        # Filter to only valid frontier positions
        valid_mask = frontier_map[unique_candidates[:, 0], unique_candidates[:, 1]] == 1
        valid_candidates = unique_candidates[valid_mask]
        
        # Select up to desired count
        actual_count = min(weed_count, len(valid_candidates))
        if actual_count > 0:
            selected_positions = valid_candidates[:actual_count]
            weed_map[selected_positions[:, 0], selected_positions[:, 1]] = 1
        
        return weed_map
    
    def _apply_weed_noise(self, weed_map: np.ndarray) -> np.ndarray:
        """
        Apply positional noise to weed locations.
        
        Args:
            weed_map: Original weed map
            
        Returns:
            Noisy weed map
        """
        if self.rng is None:
            return weed_map.copy()
        
        height, width = weed_map.shape
        
        # Get existing weed positions
        weed_positions = np.argwhere(weed_map)
        num_weeds = len(weed_positions)
        
        if num_weeds == 0:
            return np.zeros_like(weed_map)
        
        # Generate random shifts (-1, 0, 1) for each weed
        shifts = self.rng.integers(-1, 2, size=(num_weeds, 2))
        shifted_positions = weed_positions + shifts
        
        # Clip to map bounds
        shifted_positions = np.clip(shifted_positions, [0, 0], [height - 1, width - 1])
        
        # Remove duplicates
        unique_positions = np.unique(shifted_positions, axis=0)
        
        # Create noisy weed map
        noisy_weed_map = np.zeros_like(weed_map)
        noisy_weed_map[unique_positions[:, 0], unique_positions[:, 1]] = 1
        
        return noisy_weed_map