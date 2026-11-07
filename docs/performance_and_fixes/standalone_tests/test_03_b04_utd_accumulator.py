#!/usr/bin/env python3
"""
test_03_b04_utd_accumulator.py: Standalone Verification of UTD Frame Debt Accumulator Determinism.

Verifies the mathematical and algorithmic fix for Bug B04:
1. Falsifies baseline ceil deadzone: ceil(frames_per_batch / batch_size * utd) locked at 1.
2. Falsifies Doc 05 / batch_size flaw: (4M frames / 2048 * 1.0) yields only 1,953 updates (102.4x starvation).
3. Formulates true first-principles frame-based UTD accumulator:
   debt += current_frames * utd_target; updates = floor(debt); debt -= updates.
4. Validates bit-level update count determinism across standard and irregular frame arrivals.

Execution Time Target: < 0.1s
Dependencies: standard library, math
"""

from __future__ import annotations

import math
import time


def simulate_baseline_ceil(frames_per_batch: int, batch_size: int, utd_ratios: list[float]) -> list[int]:
    """Baseline ceil implementation."""
    return [math.ceil((frames_per_batch / batch_size) * utd) for utd in utd_ratios]


def simulate_doc05_division_flaw(total_frames: int, frames_per_batch: int, batch_size: int, utd: float) -> int:
    """Doc 05 flawed division: update_debt += (current_frames / batch_size) * utd."""
    debt = 0.0
    total_updates = 0
    n_steps = total_frames // frames_per_batch
    for _ in range(n_steps):
        debt += (frames_per_batch / batch_size) * utd
        n_up = int(math.floor(debt))
        if n_up > 0:
            debt -= n_up
            total_updates += n_up
    return total_updates


def simulate_true_utd_accumulator(total_frames: int, frames_per_batch: int, utd_target: float) -> tuple[int, float]:
    """True RL definition: UTD = updates per environment frame."""
    debt = 0.0
    total_updates = 0
    n_steps = total_frames // frames_per_batch
    for _ in range(n_steps):
        debt += frames_per_batch * utd_target
        n_up = int(math.floor(debt))
        if n_up > 0:
            debt -= n_up
            total_updates += n_up
    return total_updates, debt


def simulate_irregular_chunks_accumulator(chunk_sizes: list[int], utd_target: float) -> tuple[int, float]:
    """True accumulator under variable dynamic frame arrival."""
    debt = 0.0
    total_updates = 0
    for chunk in chunk_sizes:
        debt += chunk * utd_target
        n_up = int(math.floor(debt))
        if n_up > 0:
            debt -= n_up
            total_updates += n_up
    return total_updates, debt


def test_b04_utd_accumulator() -> None:
    t0 = time.perf_counter()

    frames_per_batch = 20
    batch_size = 2048

    # 1. Falsification of Baseline Ceil Deadzone
    test_utds = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    ceil_results = simulate_baseline_ceil(frames_per_batch, batch_size, test_utds)
    # 20 / 2048 = 0.009765625; for utd <= 100, product is <= 0.9765625, ceil is always 1
    assert all(r == 1 for r in ceil_results), f"Baseline ceil must be deadlocked at 1: {ceil_results}"

    # 2. Falsification of Doc 05 Division Starvation Flaw
    total_frames = 4_000_000
    updates_doc05 = simulate_doc05_division_flaw(total_frames, frames_per_batch, batch_size, utd=1.0)
    expected_doc05 = int(total_frames / batch_size * 1.0)  # 1953
    assert updates_doc05 == expected_doc05, f"Expected 1953 updates from doc05 flaw, got {updates_doc05}"
    starvation_ratio = total_frames / max(updates_doc05, 1)
    assert starvation_ratio > 2000.0, f"Doc 05 severely starves updates by {starvation_ratio:.1f}x"

    # 3. Bit-Level Verification of True Frame-Based UTD Accumulator
    # At UTD = 0.05: 1 update per 20 frames -> 200,000 updates in 4M frames
    updates_005, debt_005 = simulate_true_utd_accumulator(total_frames, frames_per_batch, utd_target=0.05)
    assert updates_005 == 200_000, f"Expected 200,000 updates, got {updates_005}"
    assert abs(debt_005) < 1e-9, f"Remaining debt should be zero, got {debt_005}"

    # At UTD = 0.025: 1 update per 40 frames -> 100,000 updates in 4M frames
    updates_0025, debt_0025 = simulate_true_utd_accumulator(total_frames, frames_per_batch, utd_target=0.025)
    assert updates_0025 == 100_000, f"Expected 100,000 updates, got {updates_0025}"
    assert abs(debt_0025) < 1e-9, f"Remaining debt should be zero, got {debt_0025}"

    # At UTD = 1.0: 1 update per frame -> 4,000,000 updates in 4M frames
    # Simulate on 400k frames for speed:
    updates_10, debt_10 = simulate_true_utd_accumulator(400_000, frames_per_batch, utd_target=1.0)
    assert updates_10 == 400_000, f"Expected 400,000 updates, got {updates_10}"
    assert abs(debt_10) < 1e-9, f"Remaining debt should be zero, got {debt_10}"

    # 4. Irregular Chunk Arrival Determinism
    # Test with varying frame chunks (e.g. asynchronous collector jitter)
    chunks = [17, 23, 19, 21, 15, 25, 20, 20] * 500  # Total 80,000 frames
    sum_frames = sum(chunks)
    utd_jitter = 0.05
    updates_jitter, debt_jitter = simulate_irregular_chunks_accumulator(chunks, utd_target=utd_jitter)
    expected_updates = int(math.floor(sum_frames * utd_jitter))
    assert updates_jitter == expected_updates, f"Irregular chunks updates {updates_jitter} != {expected_updates}"
    expected_debt = (sum_frames * utd_jitter) - expected_updates
    assert abs(debt_jitter - expected_debt) < 1e-9, f"Remaining debt mismatch: {debt_jitter} vs {expected_debt}"

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[PASS] test_03_b04_utd_accumulator completed in {elapsed_ms:.2f}ms")


if __name__ == "__main__":
    test_b04_utd_accumulator()
