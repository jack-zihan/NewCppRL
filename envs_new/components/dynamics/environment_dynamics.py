"""
Environment dynamics for the mowing robot environment.
Manages environment state updates and dynamics simulation.
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, Tuple, Union, Optional, Callable

from envs_new.components.config.environment_config import AgentConfig
from envs_new.components.entity.agent import Agent
from envs_new.components.state.environment_state import EnvironmentState
from envs_new.components.dynamics.collision_detector import CollisionDetector
from envs_new.components.dynamics.action_processor import ActionProcessor


def total_variation(image: np.ndarray) -> int:
    """
    Calculate total variation of an image.
    
    Args:
        image: Input image array
        
    Returns:
        Total variation value
    """
    pixel_dif1 = image[1:, :] - image[:-1, :]
    pixel_dif2 = image[:, 1:] - image[:, :-1]
    return int(np.abs(pixel_dif1).sum() + np.abs(pixel_dif2).sum())


class EnvironmentDynamics:
    """Manages environment dynamics and state updates."""
    
    def __init__(self, agent_config: AgentConfig, action_processor: ActionProcessor):
        """
        Initialize environment dynamics.
        
        Args:
            agent_config: Agent configuration
            action_processor: Action processor for handling different action types
        """
        self.agent_config = agent_config
        self.action_processor = action_processor
        self.collision_detector = CollisionDetector()
        
        # Map update handlers for different map types
        self._map_update_handlers = {
            'weed': self._update_weed_map,
            'field_frontier': self._update_field_frontier_map,
            'mist': self._update_mist_map,
            'trajectory': self._update_trajectory_map,
        }
    
    def step(self, agent: Agent, maps_dict: Dict[str, np.ndarray], 
             env_state: EnvironmentState, action: Union[int, Tuple],
             action_type: str) -> Tuple[Agent, Dict[str, np.ndarray], EnvironmentState]:
        """
        Perform one step of environment dynamics.
        
        Args:
            agent: Current agent
            maps_dict: Environment maps
            env_state: Current environment state
            action: Action to apply
            action_type: Type of action
            
        Returns:
            Tuple of (updated_agent, updated_maps, updated_state)
        """
        # Parse action to velocities
        linear_velocity, angular_velocity = self.action_processor.parse_action(action, action_type)
        
        # Store previous state
        prev_position = agent.position_discrete
        prev_steer = agent.last_steer
        
        # Apply control to agent
        agent.control(linear_velocity, angular_velocity)
        
        # Update maps based on agent's new state
        self._update_maps(maps_dict, agent, env_state)
        
        # Check collision
        crashed = self.collision_detector.check_collision(agent, maps_dict)
        
        # Handle collision by clipping agent position
        if crashed:
            safe_x, safe_y = self.collision_detector.get_safe_position(agent, maps_dict)
            agent.set_position(safe_x, safe_y)
        
        # Update environment state
        self._update_environment_state(env_state, agent, maps_dict, crashed)
        
        # Increment step counter
        env_state.step()
        
        return agent, maps_dict, env_state
    
    def reset(self, agent: Agent, maps_dict: Dict[str, np.ndarray], 
              env_state: EnvironmentState) -> None:
        """
        Reset environment dynamics for new episode.
        
        Args:
            agent: Agent to reset
            maps_dict: Environment maps
            env_state: Environment state to reset
        """
        # Apply initial map updates (agent's initial observation)
        self._update_maps(maps_dict, agent, env_state)
        
        # Initialize environment state
        self._update_environment_state(env_state, agent, maps_dict, crashed=False)
    
    def _update_maps(self, maps_dict: Dict[str, np.ndarray], agent: Agent,
                    env_state: EnvironmentState) -> None:
        """
        Update all maps based on agent's current state.
        
        Args:
            maps_dict: Dictionary of environment maps
            agent: Current agent
            env_state: Current environment state
        """
        for map_key in maps_dict:
            if map_key in self._map_update_handlers:
                self._map_update_handlers[map_key](maps_dict, agent, env_state)
    
    def _update_weed_map(self, maps_dict: Dict[str, np.ndarray], agent: Agent,
                        env_state: EnvironmentState) -> None:
        """Remove weeds in agent's occupied area."""
        if 'weed' in maps_dict:
            convex_hull = agent.convex_hull.round().astype(np.int32)
            cv2.fillPoly(maps_dict['weed'], [convex_hull], color=(0,))
    
    def _update_field_frontier_map(self, maps_dict: Dict[str, np.ndarray], agent: Agent,
                                  env_state: EnvironmentState) -> None:
        """Update frontier map by removing agent's vision area."""
        if 'field_frontier' in maps_dict:
            cv2.ellipse(
                img=maps_dict['field_frontier'],
                center=agent.position_discrete,
                axes=(int(agent.vision_length), int(agent.vision_length)),
                angle=agent.direction,
                startAngle=-int(agent.vision_angle / 2),
                endAngle=int(agent.vision_angle / 2),
                color=(0,),
                thickness=-1
            )
    
    def _update_mist_map(self, maps_dict: Dict[str, np.ndarray], agent: Agent,
                        env_state: EnvironmentState) -> None:
        """Update mist map by revealing agent's vision area."""
        if 'mist' in maps_dict:
            cv2.ellipse(
                img=maps_dict['mist'],
                center=agent.position_discrete,
                axes=(int(agent.vision_length + 1), int(agent.vision_length + 1)),
                angle=agent.direction,
                startAngle=-int(agent.vision_angle / 2),
                endAngle=int(agent.vision_angle / 2),
                color=(0,),
                thickness=-1
            )
    
    def _update_trajectory_map(self, maps_dict: Dict[str, np.ndarray], agent: Agent,
                              env_state: EnvironmentState) -> None:
        """Update trajectory map by drawing path from previous to current position."""
        if 'trajectory' in maps_dict:
            prev_pos = env_state.previous_agent_position
            curr_pos = agent.position_discrete
            cv2.line(maps_dict['trajectory'], prev_pos, curr_pos, color=(1,), thickness=1)
    
    def _update_environment_state(self, env_state: EnvironmentState, agent: Agent,
                                 maps_dict: Dict[str, np.ndarray], crashed: bool) -> None:
        """
        Update environment state with current information.
        
        Args:
            env_state: Environment state to update
            agent: Current agent
            maps_dict: Current maps
            crashed: Whether agent crashed this step
        """
        # Calculate current metrics
        frontier_area = self._calculate_frontier_area(maps_dict)
        frontier_variation = self._calculate_frontier_variation(maps_dict)
        weed_count = self._calculate_weed_count(maps_dict)
        
        # Update state
        env_state.update_frontier(frontier_area, frontier_variation)
        env_state.update_weed_count(weed_count)
        env_state.update_agent(agent.position_discrete, agent.last_steer)
        
        # Check if task is finished (all weeds removed)
        finished = weed_count == 0
        
        # Update flags
        env_state.update_flags(crashed, finished)
    
    def _calculate_frontier_area(self, maps_dict: Dict[str, np.ndarray]) -> int:
        """Calculate remaining frontier area."""
        if 'field_frontier' in maps_dict:
            return int(maps_dict['field_frontier'].sum())
        return 0
    
    def _calculate_frontier_variation(self, maps_dict: Dict[str, np.ndarray]) -> int:
        """Calculate frontier total variation."""
        if 'field_frontier' in maps_dict:
            return total_variation(maps_dict['field_frontier'].astype(np.int32))
        return 0
    
    def _calculate_weed_count(self, maps_dict: Dict[str, np.ndarray]) -> int:
        """Calculate remaining weed count."""
        if 'weed' in maps_dict:
            return int(maps_dict['weed'].sum())
        return 0
    
    def add_map_update_handler(self, map_key: str, 
                              handler: Callable[[Dict[str, np.ndarray], Agent, EnvironmentState], None]) -> None:
        """
        Add custom map update handler.
        
        Args:
            map_key: Map key to handle
            handler: Update function
        """
        self._map_update_handlers[map_key] = handler
    
    def remove_map_update_handler(self, map_key: str) -> None:
        """Remove map update handler."""
        if map_key in self._map_update_handlers:
            del self._map_update_handlers[map_key]
    
    def get_collision_info(self, agent: Agent, maps_dict: Dict[str, np.ndarray]) -> Dict[str, bool]:
        """Get detailed collision information."""
        return self.collision_detector.get_collision_details(agent, maps_dict)
    
    def is_valid_action(self, action: Union[int, Tuple], action_type: str) -> bool:
        """
        Check if action is valid.
        
        Args:
            action: Action to validate
            action_type: Type of action
            
        Returns:
            True if action is valid
        """
        try:
            self.action_processor.parse_action(action, action_type)
            return True
        except (ValueError, TypeError):
            return False