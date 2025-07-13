"""
Reward system for the mowing robot environment.
Provides composable reward components using strategy pattern.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Any
import numpy as np

from envs_new.components.config.environment_config import RewardConfig, ActionConfig, AgentConfig
from envs_new.components.state.environment_state import EnvironmentState


class RewardComponent(ABC):
    """Abstract base class for individual reward components."""
    
    def __init__(self, coefficient: float = 1.0):
        """
        Initialize reward component.
        
        Args:
            coefficient: Scaling coefficient for this reward component
        """
        self.coefficient = coefficient
    
    @abstractmethod
    def calculate(self, env_state: EnvironmentState, **kwargs) -> float:
        """
        Calculate reward component value.
        
        Args:
            env_state: Current environment state
            **kwargs: Additional context (action configs, etc.)
            
        Returns:
            Raw reward value (before coefficient scaling)
        """
        pass
    
    def get_reward(self, env_state: EnvironmentState, **kwargs) -> float:
        """
        Get scaled reward value.
        
        Args:
            env_state: Current environment state
            **kwargs: Additional context
            
        Returns:
            Scaled reward value
        """
        return self.coefficient * self.calculate(env_state, **kwargs)


class BaseReward(RewardComponent):
    """Constant base reward (usually negative)."""
    
    def __init__(self, coefficient: float = -0.1):
        super().__init__(coefficient)
    
    def calculate(self, env_state: EnvironmentState, **kwargs) -> float:
        """Return constant base reward."""
        return 1.0  # Coefficient will scale this to actual base penalty


class WeedRemovalReward(RewardComponent):
    """Reward for removing weeds."""
    
    def __init__(self, coefficient: float = 20.0):
        super().__init__(coefficient)
    
    def calculate(self, env_state: EnvironmentState, **kwargs) -> float:
        """Calculate reward based on weeds removed."""
        return float(env_state.weed_count_change)


class FrontierCoverageReward(RewardComponent):
    """Reward for exploring frontier areas."""
    
    def __init__(self, coefficient: float = 1.0):
        super().__init__(coefficient)
    
    def calculate(self, env_state: EnvironmentState, **kwargs) -> float:
        """Calculate reward based on frontier area covered."""
        action_config = kwargs.get('action_config')
        agent_config = kwargs.get('agent_config')
        
        if action_config is None or agent_config is None:
            return 0.0
        
        # Normalize by expected coverage per step
        normalization = 2 * agent_config.width * action_config.v_range.max
        
        return float(env_state.frontier_area_change) / normalization


class FrontierVariationReward(RewardComponent):
    """Reward for reducing frontier fragmentation."""
    
    def __init__(self, coefficient: float = 0.5):
        super().__init__(coefficient)
    
    def calculate(self, env_state: EnvironmentState, **kwargs) -> float:
        """Calculate reward based on frontier variation reduction."""
        action_config = kwargs.get('action_config')
        
        if action_config is None:
            return 0.0
        
        # Normalize by max velocity
        normalization = action_config.v_range.max
        
        return float(env_state.frontier_variation_change) / normalization


class TurningPenalty(RewardComponent):
    """Penalty for excessive turning."""
    
    def __init__(self, coefficient: float = -0.5):
        super().__init__(coefficient)
    
    def calculate(self, env_state: EnvironmentState, **kwargs) -> float:
        """Calculate penalty based on steering change."""
        action_config = kwargs.get('action_config')
        
        if action_config is None:
            return 0.0
        
        # Penalty for steering acceleration
        steer_change = abs(env_state.agent_steer_change)
        normalized_change = steer_change / action_config.w_range.max
        
        return normalized_change


class DirectionChangePenalty(RewardComponent):
    """Penalty for changing direction (left/right switch)."""
    
    def __init__(self, coefficient: float = -0.30):
        super().__init__(coefficient)
    
    def calculate(self, env_state: EnvironmentState, **kwargs) -> float:
        """Calculate penalty for direction changes."""
        current_steer = env_state.agent_steer
        previous_steer = env_state.previous_agent_steer
        
        # Penalty if steering direction changed (sign change)
        if current_steer * previous_steer < 0:  # Different signs
            return 1.0
        else:
            return 0.0


class SteeringSmoothnesReward(RewardComponent):
    """Reward for smooth steering (moderate steering is preferred)."""
    
    def __init__(self, coefficient: float = 0.25):
        super().__init__(coefficient)
    
    def calculate(self, env_state: EnvironmentState, **kwargs) -> float:
        """Calculate reward for smooth steering."""
        action_config = kwargs.get('action_config')
        
        if action_config is None:
            return 0.0
        
        # Reward function: 0.4 - |steer/max_steer|^0.5
        # This gives higher reward for moderate steering
        normalized_steer = abs(env_state.agent_steer / action_config.w_range.max)
        smoothness_reward = 0.4 - (normalized_steer ** 0.5)
        
        return smoothness_reward


class CollisionPenalty(RewardComponent):
    """Large penalty for collisions."""
    
    def __init__(self, coefficient: float = -399.0):
        super().__init__(coefficient)
    
    def calculate(self, env_state: EnvironmentState, **kwargs) -> float:
        """Calculate collision penalty."""
        return 1.0 if env_state.crashed else 0.0


class CompletionBonus(RewardComponent):
    """Large bonus for task completion."""
    
    def __init__(self, coefficient: float = 500.0):
        super().__init__(coefficient)
    
    def calculate(self, env_state: EnvironmentState, **kwargs) -> float:
        """Calculate completion bonus."""
        return 1.0 if env_state.finished else 0.0


class CompositeReward:
    """Composite reward system that combines multiple reward components."""
    
    def __init__(self, config: RewardConfig, action_config: ActionConfig, agent_config: AgentConfig):
        """
        Initialize composite reward system.
        
        Args:
            config: Reward configuration
            action_config: Action configuration for normalization
            agent_config: Agent configuration for normalization
        """
        self.config = config
        self.action_config = action_config
        self.agent_config = agent_config
        
        # Initialize reward components
        self.components = self._create_reward_components()
        
        # Group components for easier management
        self.turning_components = [
            self.components['turning_penalty'],
            self.components['direction_change_penalty'],
            self.components['steering_smoothness']
        ]
        
        self.frontier_components = [
            self.components['frontier_coverage'],
            self.components['frontier_variation']
        ]
    
    def _create_reward_components(self) -> Dict[str, RewardComponent]:
        """Create all reward components with configured coefficients."""
        coeffs = self.config.coefficients
        
        components = {
            'base': BaseReward(coeffs['base_penalty']),
            'weed_removal': WeedRemovalReward(coeffs['weed_removal_coef']),
            'frontier_coverage': FrontierCoverageReward(coeffs['frontier_coverage_coef']),
            'frontier_variation': FrontierVariationReward(coeffs['frontier_tv_coef']),
            'turning_penalty': TurningPenalty(coeffs['turn_gap_coef']),
            'direction_change_penalty': DirectionChangePenalty(coeffs['turn_direction_coef']),
            'steering_smoothness': SteeringSmoothnesReward(coeffs['turn_self_coef']),
            'collision_penalty': CollisionPenalty(coeffs['collision_penalty']),
            'completion_bonus': CompletionBonus(coeffs['completion_bonus'])
        }
        
        return components
    
    def calculate_reward(self, env_state: EnvironmentState) -> float:
        """
        Calculate total reward for current state.
        
        Args:
            env_state: Current environment state
            
        Returns:
            Total reward value
        """
        # Context for reward calculations
        context = {
            'action_config': self.action_config,
            'agent_config': self.agent_config
        }
        
        # Calculate individual reward components
        base_reward = self.components['base'].get_reward(env_state, **context)
        weed_reward = self.components['weed_removal'].get_reward(env_state, **context)
        
        # Frontier rewards
        frontier_coverage_reward = self.components['frontier_coverage'].get_reward(env_state, **context)
        frontier_variation_reward = self.components['frontier_variation'].get_reward(env_state, **context)
        frontier_reward = self.config.coefficients['frontier_total_coef'] * (
            frontier_coverage_reward + frontier_variation_reward
        )
        
        # Turning rewards
        turning_penalty = self.components['turning_penalty'].get_reward(env_state, **context)
        direction_penalty = self.components['direction_change_penalty'].get_reward(env_state, **context)
        smoothness_reward = self.components['steering_smoothness'].get_reward(env_state, **context)
        turning_reward = self.config.coefficients['turn_total_coef'] * (
            turning_penalty + direction_penalty + smoothness_reward
        )
        
        # Event-based rewards
        collision_penalty = self.components['collision_penalty'].get_reward(env_state, **context)
        completion_bonus = self.components['completion_bonus'].get_reward(env_state, **context)
        
        # Total reward
        total_reward = (base_reward + weed_reward + frontier_reward + 
                       turning_reward + collision_penalty + completion_bonus)
        
        # Apply small value clipping
        if abs(total_reward) < 1e-8:
            total_reward = 0.0
        
        return float(total_reward)
    
    def get_reward_breakdown(self, env_state: EnvironmentState) -> Dict[str, float]:
        """
        Get detailed breakdown of reward components.
        
        Args:
            env_state: Current environment state
            
        Returns:
            Dictionary with individual reward values
        """
        context = {
            'action_config': self.action_config,
            'agent_config': self.agent_config
        }
        
        breakdown = {}
        for name, component in self.components.items():
            breakdown[name] = component.get_reward(env_state, **context)
        
        # Add composite values
        breakdown['total_frontier'] = (breakdown['frontier_coverage'] + 
                                     breakdown['frontier_variation']) * self.config.coefficients['frontier_total_coef']
        breakdown['total_turning'] = (breakdown['turning_penalty'] + breakdown['direction_change_penalty'] + 
                                    breakdown['steering_smoothness']) * self.config.coefficients['turn_total_coef']
        breakdown['total'] = self.calculate_reward(env_state)
        
        return breakdown
    
    def add_component(self, name: str, component: RewardComponent) -> None:
        """Add custom reward component."""
        self.components[name] = component
    
    def remove_component(self, name: str) -> None:
        """Remove reward component."""
        if name in self.components:
            del self.components[name]
    
    def update_coefficients(self, new_coefficients: Dict[str, float]) -> None:
        """Update reward coefficients."""
        self.config.coefficients.update(new_coefficients)
        
        # Update component coefficients
        coeff_mapping = {
            'base': 'base_penalty',
            'weed_removal': 'weed_removal_coef',
            'frontier_coverage': 'frontier_coverage_coef',
            'frontier_variation': 'frontier_tv_coef',
            'turning_penalty': 'turn_gap_coef',
            'direction_change_penalty': 'turn_direction_coef',
            'steering_smoothness': 'turn_self_coef',
            'collision_penalty': 'collision_penalty',
            'completion_bonus': 'completion_bonus'
        }
        
        for comp_name, coeff_name in coeff_mapping.items():
            if comp_name in self.components and coeff_name in new_coefficients:
                self.components[comp_name].coefficient = new_coefficients[coeff_name]


class RewardManager:
    """High-level reward manager interface."""
    
    def __init__(self, reward_config: RewardConfig, action_config: ActionConfig, agent_config: AgentConfig):
        """
        Initialize reward manager.
        
        Args:
            reward_config: Reward configuration
            action_config: Action configuration
            agent_config: Agent configuration
        """
        self.reward_system = CompositeReward(reward_config, action_config, agent_config)
    
    def calculate_step_reward(self, env_state: EnvironmentState) -> float:
        """Calculate reward for current step."""
        return self.reward_system.calculate_reward(env_state)
    
    def get_reward_breakdown(self, env_state: EnvironmentState) -> Dict[str, float]:
        """Get detailed reward breakdown."""
        return self.reward_system.get_reward_breakdown(env_state)
    
    def update_config(self, new_coefficients: Dict[str, float]) -> None:
        """Update reward coefficients."""
        self.reward_system.update_coefficients(new_coefficients)
    
    def add_component(self, name: str, component: RewardComponent) -> None:
        """Add custom reward component."""
        self.reward_system.add_component(name, component)
    
    def remove_component(self, name: str) -> None:
        """Remove reward component."""
        self.reward_system.remove_component(name)