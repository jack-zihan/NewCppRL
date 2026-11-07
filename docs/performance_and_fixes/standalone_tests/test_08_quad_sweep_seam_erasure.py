#!/usr/bin/env python3
"""
test_08_quad_sweep_seam_erasure.py: Standalone Verification of Quadrilateral Sweep Seam Erasure.

Verifies the geometric and algorithmic fix for Bug B07 (Coverage Overlap Seam Erasure):
When an agent moves between consecutive steps, swept quadrilaterals Q_t and Q_{t+1} share a boundary edge F_{t+1}.
Without seam line erasure, the shared boundary line is rasterized twice, falsely generating overlap
penalties on virgin farmland during straight-line driving.
With seam line erasure:
    cv2.line(footprint, tuple(F_{t+1}[0]), tuple(F_{t+1}[1]), color=(0,), thickness=1)
the shared seam is erased from the leading footprint, ensuring strictly 0 overlap on virgin field.
Also verifies stationary vehicle short-circuit (np.array_equal(curr_front, last_front)).

Execution Time Target: < 0.1s
Dependencies: standard library, numpy, cv2
"""

from __future__ import annotations

import time
import cv2
import numpy as np


def simulate_sweep_step(
    overlap_map: np.ndarray,
    last_front: np.ndarray,
    curr_front: np.ndarray,
    erase_seam: bool,
) -> int:
    """Simulate single step of swept quadrilateral rasterization."""
    # Stationary check
    if np.array_equal(curr_front, last_front):
        return 0

    quad = np.array([last_front[0], last_front[1], curr_front[1], curr_front[0]], dtype=np.int32)
    footprint = np.zeros_like(overlap_map, dtype=np.uint8)
    cv2.fillPoly(footprint, [quad], color=(1,))

    if erase_seam:
        # Erase trailing boundary line that was already covered by previous step
        cv2.line(footprint, tuple(last_front[0]), tuple(last_front[1]), color=(0,), thickness=1)

    # Accumulate onto overlap map (virgin farmland = -1, first visit -> 0, repeat visit >= 1)
    overlap_map += footprint.astype(np.int16)
    return int(np.maximum(overlap_map, 0).sum())


def test_quad_sweep_seam_erasure() -> None:
    t0 = time.perf_counter()
    H, W = 60, 60

    # ---------------------------------------------------------
    # Scenario 1: Straight drive across virgin farmland
    # ---------------------------------------------------------
    # Baseline WITHOUT seam erasure: demonstrates false overlap bug
    map_legacy = np.full((H, W), -1, dtype=np.int16)
    F0 = np.array([[10, 15], [10, 35]], dtype=np.int32)
    F1 = np.array([[20, 15], [20, 35]], dtype=np.int32)
    F2 = np.array([[30, 15], [30, 35]], dtype=np.int32)
    F3 = np.array([[40, 15], [40, 35]], dtype=np.int32)

    simulate_sweep_step(map_legacy, F0, F1, erase_seam=False)
    simulate_sweep_step(map_legacy, F1, F2, erase_seam=False)
    overlap_legacy = simulate_sweep_step(map_legacy, F2, F3, erase_seam=False)

    print(f"  Legacy straight drive overlap count: {overlap_legacy} (false overlap on virgin field!)")
    assert overlap_legacy > 0, "Legacy straight drive must demonstrate false seam overlap"

    # Fixed WITH seam erasure: virgin farmland must have strictly 0 overlap!
    map_fixed = np.full((H, W), -1, dtype=np.int16)
    simulate_sweep_step(map_fixed, F0, F1, erase_seam=True)
    simulate_sweep_step(map_fixed, F1, F2, erase_seam=True)
    overlap_fixed = simulate_sweep_step(map_fixed, F2, F3, erase_seam=True)

    print(f"  Fixed straight drive overlap count:  {overlap_fixed} (strictly zero overlap)")
    assert overlap_fixed == 0, f"Fixed straight drive must yield 0 overlap, got {overlap_fixed}"

    # ---------------------------------------------------------
    # Scenario 2: Actual overlapping turn (returning over covered area)
    # ---------------------------------------------------------
    # Agent turns 180 degrees and drives back across [F1, F2] area
    F_return1 = np.array([[25, 15], [25, 35]], dtype=np.int32)
    F_return2 = np.array([[15, 15], [15, 35]], dtype=np.int32)

    overlap_turn = simulate_sweep_step(map_fixed, F_return1, F_return2, erase_seam=True)
    print(f"  Overlapping turn overlap count:     {overlap_turn} (true overlap detected)")
    assert overlap_turn > 0, "Genuine overlap during U-turn must be detected and positive"

    # ---------------------------------------------------------
    # Scenario 3: Stationary vehicle short-circuit
    # ---------------------------------------------------------
    map_before = map_fixed.copy()
    overlap_stationary = simulate_sweep_step(map_fixed, F_return2, F_return2.copy(), erase_seam=True)
    assert np.array_equal(map_before, map_fixed), "Stationary step must not alter overlap map"
    assert overlap_stationary == 0, "Stationary step must return 0"

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[PASS] test_08_quad_sweep_seam_erasure completed in {elapsed_ms:.2f}ms")


if __name__ == "__main__":
    test_quad_sweep_seam_erasure()
