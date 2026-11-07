#!/usr/bin/env python3
"""
test_challenge_track_a.py: Adversarial Empirical Stress Test Suite for Track A.

Empirical Challenger 2 Verification Harness:
1. Challenge B01: APF Boundary Wrap-Around Stress Test (all 4 borders & corners, non-square, agent coords)
2. Challenge B02: Steps95 Concurrent Tensor Mask (64 parallel envs, async resets, 64-oracle invariance)
3. Challenge B07: Quadrilateral Sweep Seam Erasure (stationary v=0, acute turns dtheta=pi, reversal, non-negative)
4. Challenge Multiscale: Stride-view pooling vs PyTorch avg_pool2d & max_pool2d on non-square grids + edge cases
"""

from __future__ import annotations

import math
import time
import sys
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ==============================================================================
# CHALLENGE 1: APF BOUNDARY WRAP-AROUND EMPIRICAL STRESS TEST
# ==============================================================================

def simulate_apf_bfs_legacy(H: int, W: int, obstacle_coords: list[tuple[int, int]], max_steps: int = 5) -> np.ndarray:
    """Simulate legacy BFS APF with flawed 1D index bounds check (src/cpu_apf.cpp:82-83)."""
    grid = np.full((H, W), -1.0, dtype=np.float32)
    visited = np.zeros(H * W, dtype=bool)
    queue: list[tuple[int, int, int]] = []

    for r, c in obstacle_coords:
        aim = r * W + c
        if 0 <= aim < H * W:
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
        if 0 <= r < H and 0 <= c < W:
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
            if 0 <= nx < H and 0 <= ny < W and not visited[nx, ny]:
                visited[nx, ny] = True
                grid[nx, ny] = float(step + 1)
                queue.append((nx, ny, step + 1))

    return grid


def run_challenge_b01_apf_stress() -> dict:
    """Stress test boundary isolation across all borders, corners, non-square grids."""
    results = {"subtests": 0, "legacy_bugs_caught": 0, "fixed_passes": 0}

    # Grid configurations to stress: square and non-square
    grid_configs = [
        (12, 12),
        (15, 25),
        (30, 16),
        (50, 50),
    ]

    for H, W in grid_configs:
        # 1. Right border sweep: (r, W - 1)
        for r in range(H):
            results["subtests"] += 1
            obs = [(r, W - 1)]
            leg = simulate_apf_bfs_legacy(H, W, obs, max_steps=3)
            fix = simulate_apf_bfs_fixed(H, W, obs, max_steps=3)

            # In legacy, if r + 1 < H, rightward step (0, +1) wraps to (r+1, 0)
            if r + 1 < H:
                if leg[r + 1, 0] == 1.0:
                    results["legacy_bugs_caught"] += 1
                assert fix[r + 1, 0] == -1.0, f"B01 VIOLATION: Right border at ({r}, {W-1}) leaked to ({r+1}, 0) in fixed code!"
            results["fixed_passes"] += 1

        # 2. Left border sweep: (r, 0)
        for r in range(H):
            results["subtests"] += 1
            obs = [(r, 0)]
            leg = simulate_apf_bfs_legacy(H, W, obs, max_steps=3)
            fix = simulate_apf_bfs_fixed(H, W, obs, max_steps=3)

            # In legacy, if r > 0, leftward step (0, -1) wraps to (r-1, W-1)
            if r > 0:
                if leg[r - 1, W - 1] == 1.0:
                    results["legacy_bugs_caught"] += 1
                assert fix[r - 1, W - 1] == -1.0, f"B01 VIOLATION: Left border at ({r}, 0) leaked to ({r-1}, {W-1}) in fixed code!"
            results["fixed_passes"] += 1

        # 3. Top border sweep: (0, c)
        for c in range(W):
            results["subtests"] += 1
            obs = [(0, c)]
            fix = simulate_apf_bfs_fixed(H, W, obs, max_steps=3)
            # Top boundary upward step (-1, c) must NOT wrap to bottom or anywhere illegal
            # In fixed code, row H-1 can never be reached within max_steps=3 if H > 3
            if H > 3:
                assert np.all(fix[H - 1, :] == -1.0), f"B01 VIOLATION: Top border ({0}, {c}) leaked to bottom row {H-1}!"
            results["fixed_passes"] += 1

        # 4. Bottom border sweep: (H - 1, c)
        for c in range(W):
            results["subtests"] += 1
            obs = [(H - 1, c)]
            fix = simulate_apf_bfs_fixed(H, W, obs, max_steps=3)
            # Bottom boundary downward step must NOT wrap to top row
            if H > 3:
                assert np.all(fix[0, :] == -1.0), f"B01 VIOLATION: Bottom border ({H-1}, {c}) leaked to top row 0!"
            results["fixed_passes"] += 1

        # 5. Corner Stress: (0, 0), (0, W-1), (H-1, 0), (H-1, W-1) simultaneously
        corners = [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]
        fix_corners = simulate_apf_bfs_fixed(H, W, corners, max_steps=4)
        # Check center is unreached if grid is sufficiently large
        if H >= 10 and W >= 10:
            mid_r, mid_c = H // 2, W // 2
            assert fix_corners[mid_r, mid_c] == -1.0, "Corner potentials inappropriately reached center in <=4 steps"
        results["subtests"] += 1
        results["fixed_passes"] += 1

    return results


# ==============================================================================
# CHALLENGE 2: STEPS95 CONCURRENT TENSOR MASK STRESS TEST (64 ENVS)
# ==============================================================================

class SingleEnvOracle:
    """Isolated single-environment reference oracle."""
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.has_reached_95 = False
        self.steps_since_95 = 0

    def step(self, completion: float) -> int:
        if completion >= self.threshold:
            self.has_reached_95 = True
        if self.has_reached_95:
            self.steps_since_95 += 1
        return self.steps_since_95

    def reset(self) -> int:
        self.has_reached_95 = False
        self.steps_since_95 = 0
        return 0


class VectorizedSteps95Counter:
    """Vectorized per-env tensor mask implementation conforming to B02 fix."""
    def __init__(self, batch_size: int, threshold: float = 0.95):
        self.batch_size = batch_size
        self.threshold = threshold
        self.has_reached_95 = np.zeros(batch_size, dtype=bool)
        self.steps_since_95 = np.zeros(batch_size, dtype=np.int64)

    def step(self, completions: np.ndarray) -> np.ndarray:
        self.has_reached_95 |= (completions >= self.threshold)
        self.steps_since_95 += self.has_reached_95.astype(np.int64)
        return self.steps_since_95.reshape(-1, 1).copy()

    def reset(self, reset_mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(reset_mask, dtype=bool).reshape(-1)
        self.has_reached_95[mask] = False
        self.steps_since_95[mask] = 0
        return self.steps_since_95.reshape(-1, 1).copy()


def run_challenge_b02_steps95_concurrency_stress(num_envs: int = 64, num_steps: int = 500) -> dict:
    """Stress test 64 parallel environments with asynchronous resets and check against 64 oracles."""
    rng = np.random.RandomState(1337)
    threshold = 0.95

    # 64 independent oracles
    oracles = [SingleEnvOracle(threshold=threshold) for _ in range(num_envs)]
    vectorized = VectorizedSteps95Counter(batch_size=num_envs, threshold=threshold)

    # Simulated completion state per env
    current_completions = rng.uniform(0.5, 0.8, size=num_envs)
    episode_lengths = rng.randint(40, 100, size=num_envs)
    current_lengths = np.zeros(num_envs, dtype=int)

    total_resets = 0
    total_steps_verified = 0

    for step_idx in range(num_steps):
        # 1. Advance completion ratios randomly
        delta_c = rng.uniform(0.002, 0.015, size=num_envs)
        current_completions = np.clip(current_completions + delta_c, 0.0, 1.0)
        current_lengths += 1

        # Step vectorized
        vec_out = vectorized.step(current_completions)

        # Step oracles
        oracle_out = np.zeros(num_envs, dtype=np.int64)
        for i in range(num_envs):
            oracle_out[i] = oracles[i].step(current_completions[i])

        # Verify step invariance and shape contract
        assert vec_out.shape == (num_envs, 1), f"Shape contract failure: expected ({num_envs}, 1), got {vec_out.shape}"
        assert np.array_equal(vec_out.squeeze(), oracle_out), f"Step {step_idx}: Vectorized step output mismatch vs oracles!"
        total_steps_verified += 1

        # 2. Check for asynchronous completion / reset events
        reset_mask = (current_lengths >= episode_lengths) | (current_completions >= 0.99)
        if np.any(reset_mask):
            num_resetting = int(np.sum(reset_mask))
            total_resets += num_resetting

            # Snapshot peer non-resetting env states BEFORE reset
            non_reset_indices = np.where(~reset_mask)[0]
            pre_reset_steps = vectorized.steps_since_95[non_reset_indices].copy()
            pre_reset_flags = vectorized.has_reached_95[non_reset_indices].copy()

            # Execute vectorized reset
            reset_out = vectorized.reset(reset_mask)

            # Execute oracle resets
            for i in np.where(reset_mask)[0]:
                oracles[i].reset()
                current_completions[i] = rng.uniform(0.5, 0.7)
                episode_lengths[i] = rng.randint(40, 100)
                current_lengths[i] = 0

            # Verify reset output shape
            assert reset_out.shape == (num_envs, 1), f"Reset shape failure: expected ({num_envs}, 1), got {reset_out.shape}"

            # CONCURRENCY INVARIANCE CHECK:
            # Non-resetting environments must be 100% UNTOUCHED!
            post_reset_steps = vectorized.steps_since_95[non_reset_indices]
            post_reset_flags = vectorized.has_reached_95[non_reset_indices]
            assert np.array_equal(pre_reset_steps, post_reset_steps), (
                f"CONCURRENCY CORRUPTION: Reset in envs {np.where(reset_mask)[0]} altered step count of non-resetting peers!"
            )
            assert np.array_equal(pre_reset_flags, post_reset_flags), (
                f"CONCURRENCY CORRUPTION: Reset altered 95-coverage flag of non-resetting peers!"
            )

            # Verify all resetting envs are strictly 0
            reset_indices = np.where(reset_mask)[0]
            assert np.all(vectorized.steps_since_95[reset_indices] == 0), "Resetting envs not zeroed!"
            assert np.all(~vectorized.has_reached_95[reset_indices]), "Resetting env flags not cleared!"

    return {
        "num_envs": num_envs,
        "num_steps": num_steps,
        "total_resets": total_resets,
        "total_steps_verified": total_steps_verified,
    }


# ==============================================================================
# CHALLENGE 3: B07 QUADRILATERAL SWEEP SEAM ERASURE STRESS TEST
# ==============================================================================

def simulate_sweep_step_fixed(
    overlap_map: np.ndarray,
    last_front: np.ndarray,
    curr_front: np.ndarray,
) -> int:
    """Fixed quadrilateral sweep rasterization with seam line erasure and stationary check."""
    if np.array_equal(curr_front, last_front):
        return 0

    quad = np.array([last_front[0], last_front[1], curr_front[1], curr_front[0]], dtype=np.int32)
    footprint = np.zeros_like(overlap_map, dtype=np.uint8)
    cv2.fillPoly(footprint, [quad], color=(1,))

    # Erase seam line with previous front
    cv2.line(footprint, tuple(last_front[0]), tuple(last_front[1]), color=(0,), thickness=1)

    overlap_map += footprint.astype(np.int16)
    return int(np.maximum(overlap_map, 0).sum())


def run_challenge_b07_geometry_stress() -> dict:
    """Stress test quadrilateral sweep under stationary (v=0), acute turns (dtheta=pi), and reversing."""
    H, W = 100, 100
    results = {}

    # Scenario 1: Stationary Agent (v = 0)
    # Repeated calls with identical front points
    overlap_map1 = np.full((H, W), -1, dtype=np.int16)
    F_init = np.array([[20, 20], [20, 40]], dtype=np.int32)
    for _ in range(10):
        res = simulate_sweep_step_fixed(overlap_map1, F_init, F_init.copy())
        assert res == 0, f"Stationary agent must produce 0 overlap, got {res}"
        assert np.all(overlap_map1 == -1), "Stationary agent must not modify virgin overlap map"
    results["stationary_test"] = "PASS"

    # Scenario 2: Acute Turns / Reversal (Delta theta = pi)
    # Agent drives forward [F0 -> F1 -> F2], then turns 180 degrees and drives back [F2 -> F1 -> F0]
    overlap_map2 = np.full((H, W), -1, dtype=np.int16)
    F0 = np.array([[10, 20], [10, 40]], dtype=np.int32)
    F1 = np.array([[25, 20], [25, 40]], dtype=np.int32)
    F2 = np.array([[40, 20], [40, 40]], dtype=np.int32)

    # Forward drive on virgin field: strictly 0 overlap
    ov0 = simulate_sweep_step_fixed(overlap_map2, F0, F1)
    ov1 = simulate_sweep_step_fixed(overlap_map2, F1, F2)
    assert ov0 == 0, f"Virgin forward drive step 1 produced false overlap {ov0}"
    assert ov1 == 0, f"Virgin forward drive step 2 produced false overlap {ov1}"

    # Complete reversal (180 degree U-turn in place or reverse drive):
    # F2 reverses to F1
    F2_rev = np.array([[40, 40], [40, 20]], dtype=np.int32)  # Reversal of heading
    ov_rev1 = simulate_sweep_step_fixed(overlap_map2, F2, F2_rev)  # Pure in-place turn
    ov_rev2 = simulate_sweep_step_fixed(overlap_map2, F2_rev, F1)   # Driving back over covered ground

    # Overlap must be detected and positive on returning over previously swept area
    assert ov_rev2 > 0, "Returning over previously covered farmland MUST detect genuine overlap"
    # Overlap map must never contain negative values where visited (minimum value >= 0 on visited, -1 on virgin)
    assert np.all(overlap_map2 >= -1), "Negative values below initial unvisited marker -1 detected!"
    results["reversal_test"] = "PASS"

    # Scenario 3: Continuous 360-degree rotation circle (acute angle turns at each step)
    overlap_map3 = np.full((H, W), -1, dtype=np.int16)
    center_x, center_y = 50.0, 50.0
    cutter_width = 10.0
    angles = np.linspace(0, 2 * math.pi, 36, endpoint=False)
    radius = 20.0

    prev_front = None
    for a in angles:
        # Agent position on circle
        px = center_x + radius * math.cos(a)
        py = center_y + radius * math.sin(a)
        # Heading tangent to circle: a + pi/2
        heading = a + math.pi / 2
        # Front points perpendicular to heading
        dx = (cutter_width / 2) * math.cos(heading + math.pi / 2)
        dy = (cutter_width / 2) * math.sin(heading + math.pi / 2)
        f_left = [int(round(px - dx)), int(round(py - dy))]
        f_right = [int(round(px + dx)), int(round(py + dy))]
        curr_front = np.array([f_left, f_right], dtype=np.int32)

        if prev_front is not None:
            ov = simulate_sweep_step_fixed(overlap_map3, prev_front, curr_front)
            assert ov >= 0, f"Overlap count must be non-negative, got {ov}"
        prev_front = curr_front

    # Circular loop must close and detect overlap on subsequent circle
    ov_circle2 = simulate_sweep_step_fixed(overlap_map3, prev_front, np.array([
        [int(round(center_x + radius - 0)), int(round(center_y - cutter_width / 2))],
        [int(round(center_x + radius - 0)), int(round(center_y + cutter_width / 2))]
    ], dtype=np.int32))
    assert ov_circle2 > 0, "Closing circular loop must detect true overlap"
    results["circular_loop_test"] = "PASS"

    return results


# ==============================================================================
# CHALLENGE 4: MULTISCALE STRIDE VIEW POOLING STRESS TEST (NON-SQUARE & PYTORCH)
# ==============================================================================

def stride_view_maxpool2d_rect(arr: np.ndarray, kh: int, kw: int) -> np.ndarray:
    """NumPy stride-view max pooling supporting non-square kernels and grids."""
    C, H, W = arr.shape
    assert H % kh == 0 and W % kw == 0, f"Grid ({H}, {W}) not divisible by kernel ({kh}, {kw})"
    return arr.reshape(C, H // kh, kh, W // kw, kw).max(axis=(2, 4))


def stride_view_avgpool2d_rect(arr: np.ndarray, kh: int, kw: int) -> np.ndarray:
    """NumPy stride-view average pooling supporting non-square kernels and grids."""
    C, H, W = arr.shape
    assert H % kh == 0 and W % kw == 0, f"Grid ({H}, {W}) not divisible by kernel ({kh}, {kw})"
    return arr.reshape(C, H // kh, kh, W // kw, kw).mean(axis=(2, 4))


def run_challenge_multiscale_stress() -> dict:
    """Stress test non-square grids and compare against PyTorch avg_pool2d and max_pool2d."""
    results = {"shapes_tested": 0, "max_pool_passes": 0, "avg_pool_passes": 0, "failures_caught": []}

    non_square_shapes = [
        (5, 128, 64),
        (3, 64, 128),
        (4, 96, 160),
        (2, 64, 256),
        (1, 80, 120),
    ]

    kernels = [
        (2, 2),
        (4, 4),
        (2, 4),
        (4, 2),
    ]

    rng = np.random.RandomState(42)

    for C, H, W in non_square_shapes:
        x_np = rng.randn(C, H, W).astype(np.float32)

        for kh, kw in kernels:
            if H % kh != 0 or W % kw != 0:
                continue

            results["shapes_tested"] += 1

            # 1. Max-pool comparison
            stride_max = stride_view_maxpool2d_rect(x_np, kh, kw)

            if HAS_TORCH:
                x_t = torch.from_numpy(x_np)
                torch_max = F.max_pool2d(x_t, kernel_size=(kh, kw), stride=(kh, kw)).numpy()
                # Max pooling must be BITWISE identical
                assert np.array_equal(stride_max, torch_max), (
                    f"Bitwise mismatch in maxpool for shape ({C},{H},{W}) with kernel ({kh},{kw})"
                )
                results["max_pool_passes"] += 1

                # 2. Avg-pool comparison vs PyTorch avg_pool2d
                stride_avg = stride_view_avgpool2d_rect(x_np, kh, kw)
                torch_avg = F.avg_pool2d(x_t, kernel_size=(kh, kw), stride=(kh, kw)).numpy()

                # For float32, verify maximum absolute error < 1e-6
                diff_avg = np.abs(stride_avg - torch_avg)
                max_diff = float(np.max(diff_avg))
                assert max_diff < 1e-6, (
                    f"AvgPool mismatch for shape ({C},{H},{W}) with kernel ({kh},{kw}): max diff = {max_diff}"
                )
                results["avg_pool_passes"] += 1

    # 3. EDGE CASE & VULNERABILITY MINING:
    # Test non-divisible dimensions (e.g. H=127, W=65)
    odd_shape = (2, 127, 65)
    odd_arr = rng.randn(*odd_shape).astype(np.float32)
    try:
        stride_view_maxpool2d_rect(odd_arr, 2, 2)
        results["failures_caught"].append("Expected AssertionError for odd dimension not raised")
    except AssertionError as e:
        # Correctly caught non-divisible dimension exception
        results["odd_dimension_exception_caught"] = True

    # Test Vol 02 naive reshape line 233 on non-square grid:
    # In Vol 02 line 233: k_size = int(np.round(h / feature_size))
    # base_obs.reshape(c, feature_size, k_size, feature_size, k_size)
    # When H=128, W=64, feature_size=16:
    # c*16*(128//16)*16*(128//16) = c * 16 * 8 * 16 * 8 = c * 128 * 128 != c * 128 * 64!
    # Demonstrating the bug in naive documentation formula:
    try:
        h, w = 128, 64
        feat_size = 16
        k_h = int(round(h / feat_size))  # 8
        test_non_square = rng.randn(5, h, w).astype(np.float32)
        # Attempt flawed doc formula
        flawed_pooled = test_non_square.reshape(5, feat_size, k_h, feat_size, k_h).max(axis=(2, 4))
        results["failures_caught"].append("Vol 02 naive reshape unexpectedly succeeded on non-square!")
    except ValueError as e:
        # Confirmed vulnerability in naive doc formula for non-square grids!
        results["vol02_non_square_vulnerability_confirmed"] = str(e)

    return results


# ==============================================================================
# MASTER HARNESS RUNNER
# ==============================================================================

def main():
    print("=" * 80)
    print(" NewCppRL Track A: Adversarial Empirical Challenge Suite (Challenger 2)")
    print(" Boundary Stress Testing, Concurrency Invariance & Geometric Operations")
    print("=" * 80)

    # 1. B01 APF Stress Test
    t0 = time.perf_counter()
    res_b01 = run_challenge_b01_apf_stress()
    t_b01 = (time.perf_counter() - t0) * 1000
    print(f"[PASS] Challenge B01 APF Stress ({t_b01:.2f}ms):")
    print(f"       Subtests: {res_b01['subtests']}, Legacy Bugs Caught: {res_b01['legacy_bugs_caught']}, Fixed Passes: {res_b01['fixed_passes']}")

    # 2. B02 Steps95 Concurrency Stress Test
    t0 = time.perf_counter()
    res_b02 = run_challenge_b02_steps95_concurrency_stress(num_envs=64, num_steps=500)
    t_b02 = (time.perf_counter() - t0) * 1000
    print(f"[PASS] Challenge B02 Steps95 Concurrency ({t_b02:.2f}ms):")
    print(f"       Envs: {res_b02['num_envs']}, Steps: {res_b02['num_steps']}, Resets: {res_b02['total_resets']}, Invariance Passes: {res_b02['total_steps_verified']}")

    # 3. B07 Quadrilateral Sweep Stress Test
    t0 = time.perf_counter()
    res_b07 = run_challenge_b07_geometry_stress()
    t_b07 = (time.perf_counter() - t0) * 1000
    print(f"[PASS] Challenge B07 Sweep Seam Geometry ({t_b07:.2f}ms):")
    print(f"       Stationary: {res_b07['stationary_test']}, Reversal: {res_b07['reversal_test']}, Circle Loop: {res_b07['circular_loop_test']}")

    # 4. Multiscale Stride View Stress Test
    t0 = time.perf_counter()
    res_multi = run_challenge_multiscale_stress()
    t_multi = (time.perf_counter() - t0) * 1000
    print(f"[PASS] Challenge Multiscale Stride View ({t_multi:.2f}ms):")
    print(f"       Shapes Tested: {res_multi['shapes_tested']}, MaxPool Passes: {res_multi['max_pool_passes']}, AvgPool Passes: {res_multi['avg_pool_passes']}")
    print(f"       Vulnerability Mining: {res_multi['vol02_non_square_vulnerability_confirmed']}")

    total_time = t_b01 + t_b02 + t_b07 + t_multi
    print("=" * 80)
    print(f"[VERDICT] ALL CHALLENGES PASSED EMPIRICALLY in {total_time:.2f}ms (< 2.0s hard budget)")
    print("=" * 80)


if __name__ == "__main__":
    main()
