"""
State management for the mowing robot environment.
Provides encapsulated state tracking and change detection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional
import numpy as np


@dataclass
class EnvironmentState:
    """Encapsulates the complete state of the mowing environment."""
    
    # Map dimensions
    dimensions: Tuple[int, int] = (0, 0)  # (width, height)
    
    # Episode tracking
    current_step: int = 0
    max_steps: int = 3000
    
    # Frontier state
    frontier_area: int = 0
    frontier_variation: int = 0
    
    # Weed state  
    weed_count: int = 0
    total_weed_count: int = 0
    
    # Agent state
    agent_position: Tuple[int, int] = (0, 0)
    agent_steer: float = 0.0
    
    # Environment flags
    crashed: bool = False
    finished: bool = False
    timeout: bool = False
    
    # Previous state for change tracking
    _previous_frontier_area: int = 0
    _previous_frontier_variation: int = 0
    _previous_weed_count: int = 0
    _previous_agent_position: Tuple[int, int] = (0, 0)
    _previous_agent_steer: float = 0.0
    
    def __post_init__(self):
        """Initialize previous states to current values."""
        self._previous_frontier_area = self.frontier_area
        self._previous_frontier_variation = self.frontier_variation
        self._previous_weed_count = self.weed_count
        self._previous_agent_position = self.agent_position
        self._previous_agent_steer = self.agent_steer
    
    def update_frontier(self, area: int, variation: int) -> None:
        """Update frontier state and track changes."""
        self._previous_frontier_area = self.frontier_area
        self._previous_frontier_variation = self.frontier_variation
        self.frontier_area = area
        self.frontier_variation = variation
    
    def update_weed_count(self, count: int) -> None:
        """Update weed count and track changes."""
        self._previous_weed_count = self.weed_count
        self.weed_count = count
    
    def update_agent(self, position: Tuple[int, int], steer: float) -> None:
        """Update agent state and track changes."""
        self._previous_agent_position = self.agent_position
        self._previous_agent_steer = self.agent_steer
        self.agent_position = position
        self.agent_steer = steer
    
    def update_flags(self, crashed: bool, finished: bool, timeout: bool = None) -> None:
        """Update environment status flags."""
        self.crashed = crashed
        self.finished = finished
        if timeout is not None:
            self.timeout = timeout
    
    def step(self) -> None:
        """Increment step counter and update timeout status."""
        self.current_step += 1
        self.timeout = self.current_step >= self.max_steps
    
    def reset(self, dimensions: Tuple[int, int], total_weed_count: int, 
              max_steps: int = 3000) -> None:
        """Reset state for new episode."""
        self.dimensions = dimensions
        self.total_weed_count = total_weed_count
        self.max_steps = max_steps
        self.current_step = 0
        
        # Reset state values
        self.frontier_area = 0
        self.frontier_variation = 0
        self.weed_count = total_weed_count
        self.agent_position = (0, 0)
        self.agent_steer = 0.0
        
        # Reset flags
        self.crashed = False
        self.finished = False
        self.timeout = False
        
        # Reset previous values
        self._previous_frontier_area = 0
        self._previous_frontier_variation = 0
        self._previous_weed_count = total_weed_count
        self._previous_agent_position = (0, 0)
        self._previous_agent_steer = 0.0
    
    # Properties for change tracking
    @property
    def frontier_area_change(self) -> int:
        """Change in frontier area since last update."""
        return self._previous_frontier_area - self.frontier_area
    
    @property
    def frontier_variation_change(self) -> int:
        """Change in frontier variation since last update."""
        return self._previous_frontier_variation - self.frontier_variation
    
    @property
    def weed_count_change(self) -> int:
        """Change in weed count since last update."""
        return self._previous_weed_count - self.weed_count
    
    @property
    def agent_steer_change(self) -> float:
        """Change in agent steering since last update."""
        return self.agent_steer - self._previous_agent_steer
    
    @property
    def previous_agent_position(self) -> Tuple[int, int]:
        """Previous agent position."""
        return self._previous_agent_position
    
    @property
    def previous_agent_steer(self) -> float:
        """Previous agent steering."""
        return self._previous_agent_steer
    
    @property
    def is_done(self) -> bool:
        """Check if episode is done."""
        return self.crashed or self.finished or self.timeout
    
    @property
    def weed_completion_ratio(self) -> float:
        """Ratio of weeds removed (0.0 = none removed, 1.0 = all removed)."""
        if self.total_weed_count == 0:
            return 1.0
        return 1.0 - (self.weed_count / self.total_weed_count)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for backward compatibility."""
        return {
            'dimensions': self.dimensions,
            'current_step': self.current_step,
            'frontier_area': self.frontier_area,
            'prev_frontier_area': self._previous_frontier_area,
            'frontier_variation': self.frontier_variation,
            'prev_frontier_variation': self._previous_frontier_variation,
            'weed_count': self.weed_count,
            'prev_weed_count': self._previous_weed_count,
            'discrete_pos': self.agent_position,
            'prev_discrete_pos': self._previous_agent_position,
            'current_steer': self.agent_steer,
            'prev_steer': self._previous_agent_steer,
            'crashed': self.crashed,
            'finished': self.finished,
            'timeout': self.timeout
        }
    
    @classmethod
    def from_dict(cls, state_dict: Dict[str, Any]) -> 'EnvironmentState':
        """Create state from dictionary for backward compatibility."""
        state = cls()
        state.dimensions = state_dict.get('dimensions', (0, 0))
        state.current_step = state_dict.get('current_step', 0)
        state.frontier_area = state_dict.get('frontier_area', 0)
        state._previous_frontier_area = state_dict.get('prev_frontier_area', 0)
        state.frontier_variation = state_dict.get('frontier_variation', 0)
        state._previous_frontier_variation = state_dict.get('prev_frontier_variation', 0)
        state.weed_count = state_dict.get('weed_count', 0)
        state._previous_weed_count = state_dict.get('prev_weed_count', 0)
        state.agent_position = state_dict.get('discrete_pos', (0, 0))
        state._previous_agent_position = state_dict.get('prev_discrete_pos', (0, 0))
        state.agent_steer = state_dict.get('current_steer', 0.0)
        state._previous_agent_steer = state_dict.get('prev_steer', 0.0)
        state.crashed = state_dict.get('crashed', False)
        state.finished = state_dict.get('finished', False)
        state.timeout = state_dict.get('timeout', False)
        return state


@dataclass
class StateTracker:
    """Tracks state changes and provides analytics."""
    
    state_history: List[EnvironmentState] = field(default_factory=list)
    max_history_length: int = 1000
    
    def add_state(self, state: EnvironmentState) -> None:
        """Add a state to the history."""
        # Create a copy to avoid reference issues
        state_copy = EnvironmentState(**state.__dict__)
        self.state_history.append(state_copy)
        
        # Maintain maximum history length
        if len(self.state_history) > self.max_history_length:
            self.state_history.pop(0)
    
    def get_recent_states(self, n: int = 10) -> List[EnvironmentState]:
        """Get the most recent n states."""
        return self.state_history[-n:]
    
    def get_state_at_step(self, step: int) -> Optional[EnvironmentState]:
        """Get state at a specific step."""
        for state in self.state_history:
            if state.current_step == step:
                return state
        return None
    
    def clear_history(self) -> None:
        """Clear state history."""
        self.state_history.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about state changes."""
        if not self.state_history:
            return {}
        
        total_steps = len(self.state_history)
        final_state = self.state_history[-1]
        
        # Calculate total changes
        total_frontier_reduction = 0
        total_weed_removal = 0
        
        for state in self.state_history:
            total_frontier_reduction += state.frontier_area_change
            total_weed_removal += state.weed_count_change
        
        return {
            'total_steps': total_steps,
            'final_crashed': final_state.crashed,
            'final_finished': final_state.finished,
            'final_timeout': final_state.timeout,
            'total_frontier_reduction': total_frontier_reduction,
            'total_weed_removal': total_weed_removal,
            'weed_completion_ratio': final_state.weed_completion_ratio,
            'final_frontier_area': final_state.frontier_area,
            'final_weed_count': final_state.weed_count
        }