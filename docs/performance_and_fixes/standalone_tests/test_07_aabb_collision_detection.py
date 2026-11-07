#!/usr/bin/env python3
"""
test_07_aabb_collision_detection.py: Standalone Verification of AABB Collision Detection Early Exit.

Verifies the mathematical and algorithmic equivalence of AABB early-exit collision detection (Vol 02 / Kernel 4):
Replaces full-canvas (512x512) zero allocation and rasterization with local Axis-Aligned Bounding Box (AABB) slicing.
If the local bounding slice contains zero obstacles, the detector early-exits with False immediately (zero rasterization).
If obstacles exist within the AABB, polygon rasterization is performed strictly on the local patch (~10x10 pixels).

Execution Time Target: < 0.2s
Dependencies: standard library, numpy, cv2
"""

from __future__ import annotations

import math
import time
import cv2
import numpy as np


def collision_legacy_full_canvas(obstacle_map: np.ndarray, poly: np.ndarray) -> bool:
    """Legacy approach: allocates full-canvas zeros_like and computes global intersection."""
    agent_map = np.zeros_like(obstacle_map, dtype=np.uint8)
    cv2.fillPoly(agent_map, [poly.astype(np.int32)], 1)
    return bool(np.logical_and(agent_map, obstacle_map).any())


def collision_optimized_aabb(obstacle_map: np.ndarray, poly: np.ndarray) -> bool:
    """Optimized approach: AABB bounding box early-exit + tiny local patch rasterization."""
    H, W = obstacle_map.shape
    min_x = max(0, int(math.floor(poly[:, 0].min())))
    max_x = min(W - 1, int(math.ceil(poly[:, 0].max())))
    min_y = max(0, int(math.floor(poly[:, 1].min())))
    max_y = min(H - 1, int(math.ceil(poly[:, 1].max())))

    if min_x > max_x or min_y > max_y:
        return False

    local_obs = obstacle_map[min_y : max_y + 1, min_x : max_x + 1]

    # Fast short-circuit: if local AABB has zero obstacles, collision is impossible
    if not local_obs.any():
        return False

    # Local tiny patch rasterization (~10x10 pixels)
    patch_poly = poly.copy()
    patch_poly[:, 0] -= min_x
    patch_poly[:, 1] -= min_y
    local_agent = np.zeros(local_obs.shape, dtype=np.uint8)
    cv2.fillPoly(local_agent, [patch_poly.astype(np.int32)], 1)
    return bool(np.logical_and(local_agent, local_obs).any())


def test_aabb_collision_detection() -> None:
    t0 = time.perf_counter()
    np.random.seed(42)

    H, W = 512, 512
    obstacle_map = np.zeros((H, W), dtype=np.uint8)

    # Place 30 rectangular obstacles
    for _ in range(30):
        ox, oy = np.random.randint(20, 480, size=2)
        ow, oh = np.random.randint(10, 40, size=2)
        obstacle_map[oy : oy + oh, ox : ox + ow] = 1

    # Generate 1,000 random vehicle poses
    n_poses = 1000
    polys: list[np.ndarray] = []
    for _ in range(n_poses):
        cx, cy = np.random.uniform(10, 500, size=2)
        deg = np.random.uniform(0, 360)
        rad = math.radians(deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        # Vehicle body: length 6, width 4
        corners = np.array([[-3.0, -2.0], [3.0, -2.0], [3.0, 2.0], [-3.0, 2.0]]) @ R.T + np.array([cx, cy])
        polys.append(corners)

    # Run legacy full-canvas detector
    t_start = time.perf_counter()
    results_legacy = [collision_legacy_full_canvas(obstacle_map, p) for p in polys]
    t_legacy = time.perf_counter() - t_start

    # Run optimized AABB detector
    t_start = time.perf_counter()
    results_aabb = [collision_optimized_aabb(obstacle_map, p) for p in polys]
    t_aabb = time.perf_counter() - t_start

    # Bit-level decision equivalence assertion
    assert results_legacy == results_aabb, "AABB collision decisions must be 100% identical to full-canvas baseline"

    n_hits = sum(results_aabb)
    speedup = t_legacy / max(t_aabb, 1e-9)
    print(f"  Tested {n_poses} random vehicle poses ({n_hits} collisions detected):")
    print(f"  Legacy time: {t_legacy*1000:.2f}ms | AABB time: {t_aabb*1000:.2f}ms | Speedup: {speedup:.2f}x")
    assert speedup > 1.3, f"Expected speedup > 1.3x, got {speedup:.2f}x"

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[PASS] test_07_aabb_collision_detection completed in {elapsed_ms:.2f}ms")


if __name__ == "__main__":
    test_aabb_collision_detection()
