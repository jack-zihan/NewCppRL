"""
rules_new2
----------

Minimal, geometry-faithful reimplementations of the RAL 2022 coverage
algorithms on top of ``envs_new.cpp_env_v2:CppEnv``.

This package focuses on a very small set of core concepts:

- A rectified pasture frame aligned with the bounding-box long edge.
- Paths represented as sequences of (x, y, theta) poses in world space.
- A single path executor that turns poses into (v, w) actions.

All higher-level planners (BCP, JUMP, SNAKE, R-SNAKE) are built as thin
layers on top of these primitives.

The initial implementation only includes a cleaned-up BCP planner; other
algorithms can be added incrementally.
"""

from .coverage_planners_v2 import (
    BCPPlannerV2,
    JumpPlannerV2,
    SnakePlannerV2,
    RestrictedSnakePlannerV2,
    ReactPlannerV2,
)

__all__ = [
    "BCPPlannerV2",
    "JumpPlannerV2",
    "SnakePlannerV2",
    "RestrictedSnakePlannerV2",
    "ReactPlannerV2",
]
