#!/usr/bin/env python3
"""
test_01_b01_apf_boundary_wrap.py: Standalone Deterministic Verification of APF 2D Boundary Isolation.

Verifies the mathematical and algorithmic fix for Bug B01:
In src/cpu_apf.cpp, flattened 1D index boundary check (0 <= aim < H*W) permitted horizontal
wrapping between right edge (row, W-1) and left edge (row+1, 0).
The fix replaces 1D indexing with strict 2D coordinate bounds: 0 <= nx < H && 0 <= ny < W.

Execution Time Target: < 0.1s
Dependencies: standard library, numpy
"""

from __future__ import annotations

import time
import numpy as np


def simulate_apf_bfs_legacy(H: int, W: int, obstacle_coords: list[tuple[int, int]], max_steps: int = 5) -> np.ndarray:
    """Simulate legacy BFS APF with flawed 1D index bounds check."""
    grid = np.full((H, W), -1.0, dtype=np.float32)
    visited = np.zeros(H * W, dtype=bool)
    queue: list[tuple[int, int, int]] = []

    for r, c in obstacle_coords:
        aim = r * W + c
        visited[aim] = True
        grid[r, c] = 0.0
        queue.append((r, c, 0))

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while queue:
        cx, cy, step = queue.pop(0)
        if step >= max_steps:
            continue
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            # Flawed 1D check from src/cpu_apf.cpp:82-83
            aim = nx * W + ny
            if 0 <= aim < H * W and not visited[aim]:
                visited[aim] = True
                grid.flat[aim] = float(step + 1)
                queue.append((nx, ny, step + 1))

    return grid


def simulate_apf_bfs_fixed(H: int, W: int, obstacle_coords: list[tuple[int, int]], max_steps: int = 5) -> np.ndarray:
    """Simulate fixed BFS APF with explicit 2D coordinate bounds check."""
    grid = np.full((H, W), -1.0, dtype=np.float32)
    visited = np.zeros((H, W), dtype=bool)
    queue: list[tuple[int, int, int]] = []

    for r, c in obstacle_coords:
        visited[r, c] = True
        grid[r, c] = 0.0
        queue.append((r, c, 0))

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while queue:
        cx, cy, step = queue.pop(0)
        if step >= max_steps:
            continue
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            # Fixed 2D bounds check
            if 0 <= nx < H and 0 <= ny < W and not visited[nx, ny]:
                visited[nx, ny] = True
                grid[nx, ny] = float(step + 1)
                queue.append((nx, ny, step + 1))

    return grid


def test_b01_apf_boundary_wrap() -> None:
    t0 = time.perf_counter()
    H, W = 12, 12

    # Case 1: Obstacle at right border (4, W - 1)
    # Under 1D check: (4, W-1) + (0, 1) -> nx=4, ny=W -> aim = 4*W + W = 5*W + 0 -> wraps to (5, 0)
    obs_pos = [(4, W - 1)]
    legacy_grid = simulate_apf_bfs_legacy(H, W, obs_pos, max_steps=4)
    fixed_grid = simulate_apf_bfs_fixed(H, W, obs_pos, max_steps=4)

    # Legacy must falsely propagate to left edge of next row in 1 step
    assert legacy_grid[5, 0] == 1.0, f"Expected legacy leakage at (5, 0), got {legacy_grid[5, 0]}"
    # Fixed must strictly block wrap-around: (5, 0) is distance W-1 away, impossible in 4 steps
    assert fixed_grid[5, 0] == -1.0, f"Fixed check must block leakage at (5, 0), got {fixed_grid[5, 0]}"

    # Case 2: Obstacle at left border (3, 0)
    # Under 1D check: (3, 0) + (0, -1) -> nx=3, ny=-1 -> aim = 3*W - 1 = 2*W + (W-1) -> wraps to (2, W-1)
    obs_pos_left = [(3, 0)]
    legacy_grid_l = simulate_apf_bfs_legacy(H, W, obs_pos_left, max_steps=4)
    fixed_grid_l = simulate_apf_bfs_fixed(H, W, obs_pos_left, max_steps=4)

    assert legacy_grid_l[2, W - 1] == 1.0, "Legacy must leak leftward to (2, W-1)"
    assert fixed_grid_l[2, W - 1] == -1.0, "Fixed must strictly isolate left boundary"

    # Case 3: Exhaustive edge scan on a larger 40x40 grid
    H_big, W_big = 40, 40
    for row in [0, 10, 25, 38]:
        # Right boundary
        grid_fixed = simulate_apf_bfs_fixed(H_big, W_big, [(row, W_big - 1)], max_steps=3)
        if row + 1 < H_big:
            assert grid_fixed[row + 1, 0] == -1.0, f"Boundary wrap detected at ({row+1}, 0)"

        # Left boundary
        grid_fixed_l = simulate_apf_bfs_fixed(H_big, W_big, [(row, 0)], max_steps=3)
        if row - 1 >= 0:
            assert grid_fixed_l[row - 1, W_big - 1] == -1.0, f"Boundary wrap detected at ({row-1}, {W_big-1})"

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[PASS] test_01_b01_apf_boundary_wrap completed in {elapsed_ms:.2f}ms")


if __name__ == "__main__":
    test_b01_apf_boundary_wrap()
