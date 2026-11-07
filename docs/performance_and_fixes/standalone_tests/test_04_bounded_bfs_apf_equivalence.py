#!/usr/bin/env python3
"""
test_04_bounded_bfs_apf_equivalence.py: Standalone Verification of Bounded BFS APF Equivalence.

Verifies the mathematical theorem for bounded BFS potential field computation (Vol 03 / Kernel 3):
On a 4-connected grid, graph search depth represents Manhattan (L1) distance.
To ensure that all cells within Euclidean (L2) distance D_max are visited without directional distortion,
the Manhattan search depth must be bounded by:
    R_L1 = ceil(D_max * sqrt(2))
Stopping at step >= D_max causes diagonal truncation (error > 0.15).
Stopping at ceil(D_max * sqrt(2)) with a 2D LUT achieves strictly 0.00e+00 bitwise equivalence
compared to unconstrained full-grid BFS.

Execution Time Target: < 0.2s
Dependencies: standard library, collections, numpy
"""

from __future__ import annotations

from collections import deque
import math
import time
import numpy as np


def compute_apf_unconstrained_baseline(grid: np.ndarray, propagate_d: int, gamma: float) -> np.ndarray:
    """Ground truth: Unconstrained full-grid BFS followed by distance exponential attenuation."""
    H, W = grid.shape
    dist_map = np.full((H, W), np.inf, dtype=np.float32)
    visited = np.zeros((H, W), dtype=bool)
    q: deque[tuple[int, int, int, int]] = deque()

    for r, c in zip(*np.where(grid)):
        q.append((r, c, r, c))
        visited[r, c] = True
        dist_map[r, c] = 0.0

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while q:
        r, c, or_, oc = q.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc]:
                visited[nr, nc] = True
                d = math.sqrt((nr - or_) ** 2 + (nc - oc) ** 2)
                dist_map[nr, nc] = d
                q.append((nr, nc, or_, oc))

    apf = np.zeros((H, W), dtype=np.float32)
    mask = dist_map <= propagate_d
    apf[mask] = gamma ** dist_map[mask]
    return apf


def compute_apf_bounded_naive(grid: np.ndarray, propagate_d: int, gamma: float) -> np.ndarray:
    """Flawed naive bounded BFS: truncates search at step >= propagate_d."""
    H, W = grid.shape
    apf = np.zeros((H, W), dtype=np.float32)
    visited = np.zeros((H, W), dtype=bool)
    q: deque[tuple[int, int, int, int, int]] = deque()

    for r, c in zip(*np.where(grid)):
        q.append((r, c, r, c, 0))
        visited[r, c] = True
        apf[r, c] = 1.0

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while q:
        r, c, or_, oc, step = q.popleft()
        if step >= propagate_d:
            continue
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc]:
                visited[nr, nc] = True
                d = math.sqrt((nr - or_) ** 2 + (nc - oc) ** 2)
                if d <= propagate_d:
                    apf[nr, nc] = gamma ** d
                    q.append((nr, nc, or_, oc, step + 1))
    return apf


def compute_apf_bounded_sqrt2_lut(grid: np.ndarray, propagate_d: int, gamma: float) -> np.ndarray:
    """Fixed First-Principles Bounded BFS: Manhattan bound = ceil(propagate_d * sqrt(2)) with 2D LUT."""
    H, W = grid.shape
    max_d = propagate_d
    max_manhattan = int(math.ceil(max_d * math.sqrt(2.0)))
    lut_size = max_d + 2

    # 2D Euclidean lookup table (fits in L1 cache)
    lut = np.zeros((lut_size, lut_size), dtype=np.float32)
    for dx in range(lut_size):
        for dy in range(lut_size):
            d = math.sqrt(dx * dx + dy * dy)
            lut[dx, dy] = (gamma ** d) if d <= max_d else 0.0

    apf = np.zeros((H, W), dtype=np.float32)
    visited = np.zeros((H, W), dtype=bool)
    q: deque[tuple[int, int, int, int, int]] = deque()

    for r, c in zip(*np.where(grid)):
        q.append((r, c, r, c, 0))
        visited[r, c] = True
        apf[r, c] = 1.0

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while q:
        r, c, or_, oc, step = q.popleft()
        if step >= max_manhattan:
            continue
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc]:
                visited[nr, nc] = True
                dx = abs(nr - or_)
                dy = abs(nc - oc)
                if dx < lut_size and dy < lut_size:
                    val = lut[dx, dy]
                    if val > 0.0:
                        apf[nr, nc] = val
                        q.append((nr, nc, or_, oc, step + 1))
    return apf


def test_bounded_bfs_apf_equivalence() -> None:
    t0 = time.perf_counter()
    H, W = 150, 150
    grid = np.zeros((H, W), dtype=np.uint8)
    # Seed obstacle patches
    grid[40:45, 40:45] = 1
    grid[90:94, 110:115] = 1

    propagate_d = 5
    gamma = (propagate_d - 1) / float(propagate_d)  # 0.8

    # 1. Unconstrained Baseline
    apf_base = compute_apf_unconstrained_baseline(grid, propagate_d, gamma)

    # 2. Naive Bounded BFS: demonstrates diagonal truncation defect
    apf_naive = compute_apf_bounded_naive(grid, propagate_d, gamma)
    naive_diff = float(np.max(np.abs(apf_base - apf_naive)))
    print(f"  Naive bounded BFS maximum error: {naive_diff:.4f} (demonstrating diagonal distortion)")
    assert naive_diff > 0.10, f"Naive bounded BFS should exhibit truncation error, got {naive_diff}"

    # 3. Correct Bounded BFS with ceil(D_max * sqrt(2)) and 2D LUT
    apf_fixed = compute_apf_bounded_sqrt2_lut(grid, propagate_d, gamma)
    fixed_diff = float(np.max(np.abs(apf_base - apf_fixed)))
    print(f"  Fixed bounded BFS (sqrt(2) bound) maximum error: {fixed_diff:.2e}")

    # Must be bitwise equivalent within floating point tolerance
    assert fixed_diff < 1e-6, f"Fixed bounded BFS must match unconstrained ground truth, diff: {fixed_diff}"

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[PASS] test_04_bounded_bfs_apf_equivalence completed in {elapsed_ms:.2f}ms")


if __name__ == "__main__":
    test_bounded_bfs_apf_equivalence()
