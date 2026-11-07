#!/usr/bin/env python3
"""
run_all_tests.py: Master Test Runner for NewCppRL Standalone Numerical & Bit-Level Verification Suite.

Executes all 8 micro-kernel verification tests sequentially, enforcing:
1. 100% Pass Rate across all algorithmic and bit-level invariant checks.
2. Hard Execution Time Limit: Total suite runtime < 2.0s (measured in-process < 0.6s).

Usage:
    python run_all_tests.py                 # Fast in-process execution (default, <0.6s)
    python run_all_tests.py --subprocess    # Subprocess-isolated execution mode
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import subprocess
import sys
import time

TEST_REGISTRY: list[tuple[str, str, str, str]] = [
    ("test_01_b01_apf_boundary_wrap", "test_b01_apf_boundary_wrap", "B01 APF Boundary Wrap Prevention", "M1"),
    ("test_02_b02_steps95_tensor_mask", "test_b02_steps95_tensor_mask", "B02 Steps95 Concurrent Tensor Mask Isolation", "M1"),
    ("test_03_b04_utd_accumulator", "test_b04_utd_accumulator", "B04 UTD Frame Debt Accumulator Determinism", "M1"),
    ("test_04_bounded_bfs_apf_equivalence", "test_bounded_bfs_apf_equivalence", "Bounded BFS APF ceil(D*sqrt(2)) Equivalence", "M2"),
    ("test_05_closed_form_affine_ego_patch", "test_closed_form_affine_ego_patch", "Closed-Form 2x3 Affine Ego Patch (0MB buffer)", "M2"),
    ("test_06_multiscale_stride_pooling", "test_multiscale_stride_pooling", "Multiscale Stride Pooling View Equivalence", "M2"),
    ("test_07_aabb_collision_detection", "test_aabb_collision_detection", "AABB Collision Detection Early Exit", "M2"),
    ("test_08_quad_sweep_seam_erasure", "test_quad_sweep_seam_erasure", "B07 Quadrilateral Sweep Seam Erasure", "M1"),
]

TIME_BUDGET_SECONDS = 2.0


def run_tests_in_process(test_dir: Path) -> tuple[int, list[tuple[str, str, str, str, float]], float]:
    """Execute tests in-process for maximum throughput (<0.6s total)."""
    if str(test_dir) not in sys.path:
        sys.path.insert(0, str(test_dir))

    total_passed = 0
    results: list[tuple[str, str, str, str, float]] = []
    t_start = time.perf_counter()

    for mod_name, fn_name, desc, milestone in TEST_REGISTRY:
        t0 = time.perf_counter()
        try:
            module = importlib.import_module(mod_name)
            test_fn = getattr(module, fn_name)
            test_fn()
            elapsed = time.perf_counter() - t0
            status = "PASS"
            total_passed += 1
            print(f"  [{status}] {mod_name + '.py':<42} ({elapsed*1000:6.1f}ms) [{milestone}]")
            results.append((mod_name, desc, milestone, status, elapsed))
        except Exception as e:
            elapsed = time.perf_counter() - t0
            status = "FAIL"
            print(f"  [{status}] {mod_name + '.py':<42} ({elapsed*1000:6.1f}ms) [{milestone}]")
            print(f"    Error: {type(e).__name__}: {e}")
            results.append((mod_name, desc, milestone, status, elapsed))

    t_total = time.perf_counter() - t_start
    return total_passed, results, t_total


def run_tests_subprocess(test_dir: Path) -> tuple[int, list[tuple[str, str, str, str, float]], float]:
    """Execute tests via individual subprocesses."""
    total_passed = 0
    results: list[tuple[str, str, str, str, float]] = []
    t_start = time.perf_counter()

    for mod_name, _, desc, milestone in TEST_REGISTRY:
        script_file = test_dir / f"{mod_name}.py"
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, str(script_file)], capture_output=True, text=True)
        elapsed = time.perf_counter() - t0

        if proc.returncode == 0:
            status = "PASS"
            total_passed += 1
            print(f"  [{status}] {mod_name + '.py':<42} ({elapsed*1000:6.1f}ms) [{milestone}]")
            results.append((mod_name, desc, milestone, status, elapsed))
        else:
            status = "FAIL"
            print(f"  [{status}] {mod_name + '.py':<42} ({elapsed*1000:6.1f}ms) [{milestone}]")
            print("    --- Stderr / Stdout Output ---")
            err_msg = proc.stderr.strip() or proc.stdout.strip()
            for line in err_msg.splitlines()[:10]:
                print(f"      {line}")
            print("    ------------------------------")
            results.append((mod_name, desc, milestone, status, elapsed))

    t_total = time.perf_counter() - t_start
    return total_passed, results, t_total


def main() -> int:
    parser = argparse.ArgumentParser(description="Master Test Runner for NewCppRL Standalone Tests")
    parser.add_argument("--subprocess", action="store_true", help="Run tests in separate subprocesses")
    args = parser.parse_args()

    test_dir = Path(__file__).resolve().parent

    print("=" * 86)
    print(" NewCppRL Track A: Standalone Numerical & Bit-Level Verification Suite")
    print(f" Directory: {test_dir}")
    print(f" Mode: {'Subprocess' if args.subprocess else 'Fast In-Process (Zero Startup Overhead)'}")
    print(f" Hard Execution Budget: < {TIME_BUDGET_SECONDS:.1f}s | Pass Criterion: 100% (8/8)")
    print("=" * 86)

    if args.subprocess:
        total_passed, results, t_total = run_tests_subprocess(test_dir)
    else:
        total_passed, results, t_total = run_tests_in_process(test_dir)

    print("=" * 86)
    print(" Suite Execution Summary:")
    print(f"   Tests Executed:  {len(TEST_REGISTRY)}")
    print(f"   Tests Passed:    {total_passed}/{len(TEST_REGISTRY)}")
    print(f"   Total Duration:  {t_total:.3f}s (Budget: < {TIME_BUDGET_SECONDS:.1f}s)")
    print("=" * 86)

    pass_rate_ok = (total_passed == len(TEST_REGISTRY))
    time_budget_ok = (t_total < TIME_BUDGET_SECONDS)

    if pass_rate_ok and time_budget_ok:
        print("[VERDICT: SUCCESS] All 8 standalone verification tests passed within time budget.")
        return 0
    else:
        reasons = []
        if not pass_rate_ok:
            reasons.append(f"Pass rate failure ({total_passed}/{len(TEST_REGISTRY)})")
        if not time_budget_ok:
            reasons.append(f"Time budget exceeded ({t_total:.3f}s >= {TIME_BUDGET_SECONDS:.1f}s)")
        print(f"[VERDICT: FAILURE] {'; '.join(reasons)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
