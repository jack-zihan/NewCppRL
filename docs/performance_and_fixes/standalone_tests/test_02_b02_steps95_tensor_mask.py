#!/usr/bin/env python3
"""
test_02_b02_steps95_tensor_mask.py: Standalone Verification of Steps95 Vectorized Reset Isolation.

Verifies the mathematical and interface contract for Bug B02:
In multi-worker parallel environments, Steps95ToDoneCounter must isolate individual environment resets
using boolean tensor masks, ensuring that an environment finishing an episode does NOT reset or corrupt
the running step counters of concurrently active peer environments.
Also verifies the (*batch_size, 1) tensor shape contract required by TorchRL Transform modules.

Execution Time Target: < 0.1s
Dependencies: standard library, numpy
"""

from __future__ import annotations

import time
import numpy as np


class Steps95CounterLegacy:
    """Flawed legacy counter: resets all batch elements upon any single env reset."""

    def __init__(self, batch_size: int, threshold: float = 0.95):
        self.batch_size = batch_size
        self.threshold = threshold
        self.has_reached_95 = np.zeros(batch_size, dtype=bool)
        self.steps_since_95 = np.zeros(batch_size, dtype=np.int64)

    def step(self, completion_ratios: np.ndarray) -> np.ndarray:
        self.has_reached_95 |= (completion_ratios >= self.threshold)
        self.steps_since_95 += self.has_reached_95.astype(np.int64)
        return self.steps_since_95.copy()

    def reset(self, reset_mask: np.ndarray) -> None:
        # Flawed: ignores reset_mask and resets entire batch
        self.has_reached_95.fill(False)
        self.steps_since_95.fill(0)


class Steps95CounterFixed:
    """Fixed first-principles counter: strictly isolated per-environment tensor masking."""

    def __init__(self, batch_size: int, threshold: float = 0.95):
        self.batch_size = batch_size
        self.threshold = threshold
        self.has_reached_95 = np.zeros(batch_size, dtype=bool)
        self.steps_since_95 = np.zeros(batch_size, dtype=np.int64)

    def step(self, completion_ratios: np.ndarray) -> np.ndarray:
        self.has_reached_95 |= (completion_ratios >= self.threshold)
        self.steps_since_95 += self.has_reached_95.astype(np.int64)
        # Enforce (*batch_size, 1) shape contract
        return self.steps_since_95.reshape(-1, 1).copy()

    def reset(self, reset_mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(reset_mask, dtype=bool).reshape(-1)
        self.has_reached_95[mask] = False
        self.steps_since_95[mask] = 0
        return self.steps_since_95.reshape(-1, 1).copy()


def test_b02_steps95_tensor_mask() -> None:
    t0 = time.perf_counter()
    B = 4
    threshold = 0.95

    legacy = Steps95CounterLegacy(batch_size=B, threshold=threshold)
    fixed = Steps95CounterFixed(batch_size=B, threshold=threshold)

    # Step 1: Env 0 and Env 2 cross 95% threshold
    comp_step1 = np.array([0.96, 0.70, 0.98, 0.50])
    legacy.step(comp_step1)
    out_fixed1 = fixed.step(comp_step1)

    assert out_fixed1.shape == (B, 1), f"Shape must be (B, 1), got {out_fixed1.shape}"
    assert np.array_equal(fixed.steps_since_95, [1, 0, 1, 0])

    # Step 2: Env 0 advances, Env 2 advances, Env 1 crosses 95%
    comp_step2 = np.array([0.97, 0.96, 0.99, 0.60])
    legacy.step(comp_step2)
    fixed.step(comp_step2)
    assert np.array_equal(fixed.steps_since_95, [2, 1, 2, 0])
    assert np.array_equal(fixed.has_reached_95, [True, True, True, False])

    # Partial Reset Event: Only Env 2 completes episode (done=True) and triggers reset
    reset_mask = np.array([False, False, True, False], dtype=bool)

    legacy.reset(reset_mask)
    reset_out = fixed.reset(reset_mask)

    # Legacy defect: wipes out all envs
    assert np.all(legacy.steps_since_95 == 0), "Legacy must wipe all envs"
    assert np.all(~legacy.has_reached_95), "Legacy must clear all flags"

    # Fixed verification: Env 0 and Env 1 must be strictly isolated and intact!
    assert reset_out.shape == (B, 1), f"Reset output shape must be (B, 1), got {reset_out.shape}"
    assert fixed.steps_since_95[0] == 2, f"Env 0 steps corrupted! Expected 2, got {fixed.steps_since_95[0]}"
    assert fixed.steps_since_95[1] == 1, f"Env 1 steps corrupted! Expected 1, got {fixed.steps_since_95[1]}"
    assert fixed.steps_since_95[2] == 0, f"Env 2 must be reset to 0, got {fixed.steps_since_95[2]}"
    assert fixed.steps_since_95[3] == 0, f"Env 3 must remain 0, got {fixed.steps_since_95[3]}"

    assert fixed.has_reached_95[0] is True or fixed.has_reached_95[0] == True
    assert fixed.has_reached_95[1] is True or fixed.has_reached_95[1] == True
    assert fixed.has_reached_95[2] is False or fixed.has_reached_95[2] == False

    # Step 3: Subsequent step verification
    comp_step3 = np.array([0.98, 0.97, 0.80, 0.65])
    out_step3 = fixed.step(comp_step3)
    # Env 0 -> 3, Env 1 -> 2, Env 2 -> 0 (80% < 95%), Env 3 -> 0
    assert np.array_equal(out_step3.squeeze(), [3, 2, 0, 0]), f"Step 3 mismatch: {out_step3.squeeze()}"

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[PASS] test_02_b02_steps95_tensor_mask completed in {elapsed_ms:.2f}ms")


if __name__ == "__main__":
    test_b02_steps95_tensor_mask()
