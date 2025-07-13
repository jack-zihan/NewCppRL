"""
Reward system module.
"""

from envs_new.components.reward.reward_system import (
    RewardComponent,
    BaseReward,
    WeedRemovalReward,
    FrontierCoverageReward,
    FrontierVariationReward,
    TurningPenalty,
    DirectionChangePenalty,
    SteeringSmoothnesReward,
    CollisionPenalty,
    CompletionBonus,
    CompositeReward,
    RewardManager
)

__all__ = [
    'RewardComponent',
    'BaseReward', 
    'WeedRemovalReward',
    'FrontierCoverageReward',
    'FrontierVariationReward',
    'TurningPenalty',
    'DirectionChangePenalty',
    'SteeringSmoothnesReward',
    'CollisionPenalty',
    'CompletionBonus',
    'CompositeReward',
    'RewardManager'
]