"""
Observation generation module.
"""

from envs_new.components.observation.observation_strategy import (
    ObservationStrategy,
    FirstPersonObservation, 
    MultiScaleObservation,
    ObservationManager,
    NoiseInjector
)

__all__ = [
    'ObservationStrategy',
    'FirstPersonObservation',
    'MultiScaleObservation', 
    'ObservationManager',
    'NoiseInjector'
]