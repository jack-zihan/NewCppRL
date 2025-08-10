"""
Reward system module.
"""

from envs_new.components.reward.reward_system import (
    BaseCalculator,
    WeedRemovalCalculator,
    FrontierCoverageCalculator,
    FrontierVariationCalculator,
    TurningPenaltyCalculator,
    DirectionChangePenaltyCalculator,
    SteeringSmoothnessCalculator,
    CollisionPenaltyCalculator,
    CompletionBonusCalculator,
    RewardSystem,
    CompositeReward  # 向后兼容别名
)

__all__ = [
    'BaseCalculator', 
    'WeedRemovalCalculator',
    'FrontierCoverageCalculator',
    'FrontierVariationCalculator',
    'TurningPenaltyCalculator',
    'DirectionChangePenaltyCalculator',
    'SteeringSmoothnessCalculator',
    'CollisionPenaltyCalculator',
    'CompletionBonusCalculator',
    'RewardSystem',
    'CompositeReward'
]