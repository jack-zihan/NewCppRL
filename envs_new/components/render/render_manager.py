"""
Basic rendering system for the mowing robot environment.
Provides map visualization and first-person view rendering.
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
import math

from envs_new.components.config.environment_config import RenderConfig
from envs_new.components.entity.agent import Agent


def get_map_pasture_larger(map_pasture: np.ndarray) -> np.ndarray:
    """Enlarge map features by one pixel in all directions."""
    map_pasture_larger = map_pasture.copy()
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for shift in shifts:
        map_pasture_larger = np.logical_or(
            map_pasture_larger,
            np.roll(map_pasture, shift, axis=(0, 1))
        )
    return map_pasture_larger


class MapRenderer:
    """Renders environment maps with different visual elements."""
    
    def __init__(self, config: RenderConfig):
        """
        Initialize map renderer.
        
        Args:
            config: Render configuration
        """
        self.config = config
        
        # Color scheme
        self.colors = {
            'background': (255, 255, 255),
            'field_frontier': (76, 187, 23),
            'covered_farmland': (112, 173, 7),
            'obstacle': (30, 75, 130),
            'obstacle_edge': (47, 82, 143),
            'weed_undiscovered': (0, 0, 0),
            'weed_discovered': (255, 0, 0),
            'weed_covered': (0, 0, 0),
            'trajectory': (255, 38, 255),
            'agent': (255, 0, 0),
            'agent_vision': (192, 192, 192),
            'mist': (128, 128, 128)
        }
    
    def render_map(self, maps_dict: Dict[str, np.ndarray], agent: Agent,
                  dimensions: Tuple[int, int]) -> np.ndarray:
        """
        Render complete environment map.
        
        Args:
            maps_dict: Dictionary of environment maps
            agent: Current agent
            dimensions: Map dimensions (width, height)
            
        Returns:
            Rendered map as RGB image
        """
        width, height = dimensions
        rendered_map = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Render field frontier
        if 'field_frontier' in maps_dict:
            mask = maps_dict['field_frontier'].astype(bool)
            rendered_map[mask] = self.colors['field_frontier']
        
        # Render covered farmland
        if (self.config.render_covered_farmland and 
            'original_field_frontier' in maps_dict and 
            'field_frontier' in maps_dict):
            
            covered_mask = np.logical_and(
                maps_dict['original_field_frontier'],
                np.logical_not(maps_dict['field_frontier'])
            )
            if covered_mask.any():
                rendered_map[covered_mask] = (
                    0.25 * np.array(self.colors['covered_farmland']) +
                    0.75 * rendered_map[covered_mask]
                ).astype(np.uint8)
        
        # Render obstacles
        if 'obstacle' in maps_dict:
            obstacle_mask = maps_dict['obstacle'].astype(bool)
            rendered_map[obstacle_mask] = self.colors['obstacle']
        
        # Render agent vision
        self._render_agent_vision(rendered_map, agent)
        
        # Render weeds
        self._render_weeds(rendered_map, maps_dict)
        
        # Render trajectory
        if 'trajectory' in maps_dict:
            traj_mask = maps_dict['trajectory'].astype(bool)
            rendered_map[traj_mask] = self.colors['trajectory']
        
        # Render agent
        self._render_agent(rendered_map, agent)
        
        # Apply mist effect
        if self.config.render_mist and 'mist' in maps_dict:
            mist_mask = maps_dict['mist'].astype(bool)
            rendered_map[mist_mask] = (rendered_map[mist_mask] * 0.7).astype(np.uint8)
        
        return rendered_map
    
    def _render_agent_vision(self, rendered_map: np.ndarray, agent: Agent) -> None:
        """Render agent's vision cone."""
        cv2.ellipse(
            img=rendered_map,
            center=agent.position_discrete,
            axes=(int(agent.vision_length), int(agent.vision_length)),
            angle=agent.direction,
            startAngle=-int(agent.vision_angle / 2),
            endAngle=int(agent.vision_angle / 2),
            color=self.colors['agent_vision'],
            thickness=-1
        )
    
    def _render_weeds(self, rendered_map: np.ndarray, maps_dict: Dict[str, np.ndarray]) -> None:
        """Render weeds in different states."""
        if 'weed' not in maps_dict:
            return
        
        weed_map = maps_dict['weed']
        
        if 'field_frontier' in maps_dict:
            # Undiscovered weeds (in frontier areas)
            weed_undiscovered = get_map_pasture_larger(
                np.logical_and(weed_map, maps_dict['field_frontier'])
            )
            rendered_map[weed_undiscovered] = self.colors['weed_undiscovered']
            
            # Discovered weeds (not in frontier areas)
            weed_discovered = get_map_pasture_larger(
                np.logical_and(weed_map, np.logical_not(maps_dict['field_frontier']))
            )
            rendered_map[weed_discovered] = self.colors['weed_discovered']
        else:
            # All weeds as discovered
            weed_mask = get_map_pasture_larger(weed_map.astype(bool))
            rendered_map[weed_mask] = self.colors['weed_discovered']
        
        # Render covered weeds
        if (self.config.render_covered_weed and 
            'original_weed' in maps_dict):
            weed_covered = get_map_pasture_larger(
                np.logical_and(
                    maps_dict['original_weed'],
                    np.logical_not(weed_map)
                )
            )
            if weed_covered.any():
                rendered_map[weed_covered] = (
                    0.9 * np.array(self.colors['weed_covered']) +
                    0.1 * rendered_map[weed_covered]
                ).astype(np.uint8)
    
    def _render_agent(self, rendered_map: np.ndarray, agent: Agent) -> None:
        """Render agent as filled polygon."""
        convex_hull = agent.convex_hull.round().astype(np.int32)
        cv2.fillPoly(rendered_map, [convex_hull], color=self.colors['agent'])


class FirstPersonRenderer:
    """Renders first-person view from agent perspective."""
    
    def __init__(self, config: RenderConfig, map_renderer: MapRenderer):
        """
        Initialize first-person renderer.
        
        Args:
            config: Render configuration
            map_renderer: Map renderer for base image
        """
        self.config = config
        self.map_renderer = map_renderer
    
    def render_first_person(self, maps_dict: Dict[str, np.ndarray], agent: Agent,
                           dimensions: Tuple[int, int], 
                           observation_size: Tuple[int, int]) -> np.ndarray:
        """
        Render first-person view from agent.
        
        Args:
            maps_dict: Environment maps
            agent: Current agent
            dimensions: Map dimensions
            observation_size: Size of observation patch
            
        Returns:
            First-person view image
        """
        # Get base rendered map
        base_map = self.map_renderer.render_map(maps_dict, agent, dimensions)
        
        # Extract first-person patch
        patch = self._extract_ego_patch(base_map, agent, observation_size)
        
        return patch
    
    def _extract_ego_patch(self, rendered_map: np.ndarray, agent: Agent,
                          patch_size: Tuple[int, int]) -> np.ndarray:
        """Extract and rotate patch centered on agent."""
        patch_height, patch_width = patch_size
        
        # Calculate padding needed for rotation
        diagonal_length = math.ceil(max(patch_height, patch_width) / 2 * math.sqrt(2))
        
        # Pad map
        padded_map = cv2.copyMakeBorder(
            rendered_map,
            diagonal_length, diagonal_length,
            diagonal_length, diagonal_length,
            cv2.BORDER_CONSTANT,
            value=(128, 128, 128)
        )
        
        # Adjust agent position for padding
        agent_y = agent.y + diagonal_length
        agent_x = agent.x + diagonal_length
        
        # Crop square region
        top = int(round(agent_y - diagonal_length))
        bottom = int(round(agent_y + diagonal_length))
        left = int(round(agent_x - diagonal_length))
        right = int(round(agent_x + diagonal_length))
        
        cropped = padded_map[top:bottom, left:right, :]
        
        # Rotate to align agent direction upward
        rotation_angle = 180 + agent.direction
        rotation_center = (diagonal_length, diagonal_length)
        rotation_matrix = cv2.getRotationMatrix2D(rotation_center, rotation_angle, 1.0)
        
        rotated = cv2.warpAffine(
            cropped.astype(np.float32),
            rotation_matrix,
            (cropped.shape[1], cropped.shape[0])
        )
        
        # Final crop to desired size
        rotated_height, rotated_width = rotated.shape[:2]
        start_y = max(0, (rotated_height - patch_height) // 2)
        start_x = max(0, (rotated_width - patch_width) // 2)
        
        final_patch = rotated[start_y:start_y + patch_height,
                            start_x:start_x + patch_width, :]
        
        return final_patch.astype(np.uint8)


class RenderManager:
    """Main rendering manager that coordinates different rendering modes."""
    
    def __init__(self, config: RenderConfig):
        """
        Initialize render manager.
        
        Args:
            config: Render configuration
        """
        self.config = config
        self.map_renderer = MapRenderer(config)
        self.first_person_renderer = FirstPersonRenderer(config, self.map_renderer)
    
    def render(self, maps_dict: Dict[str, np.ndarray], agent: Agent,
              dimensions: Tuple[int, int], mode: str = "map",
              observation_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Render environment in specified mode.
        
        Args:
            maps_dict: Environment maps
            agent: Current agent
            dimensions: Map dimensions
            mode: Render mode ("map" or "first_person")
            observation_size: Size for first-person view
            
        Returns:
            Rendered image
        """
        if mode == "map":
            rendered = self.map_renderer.render_map(maps_dict, agent, dimensions)
        elif mode == "first_person":
            if observation_size is None:
                observation_size = (128, 128)
            rendered = self.first_person_renderer.render_first_person(
                maps_dict, agent, dimensions, observation_size
            )
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
        
        # Apply scaling if configured
        if self.config.render_repeat_times > 1:
            rendered = rendered.repeat(self.config.render_repeat_times, axis=0).repeat(
                self.config.render_repeat_times, axis=1
            )
        
        return rendered