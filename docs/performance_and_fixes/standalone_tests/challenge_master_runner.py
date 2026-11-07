#!/usr/bin/env python3
"""
challenge_master_runner.py: Master Adversarial Challenge Runner for Track A.

Executes all 3 empirical numerical stress tests and oracles:
1. challenge_01_utd_accumulator_stress.py
2. challenge_02_bounded_bfs_apf_oracle.py
3. challenge_03_closed_form_affine_adversarial.py
Followed by the baseline test suite:
4. run_all_tests.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


def run_challenge_script(script_path: Path) -> tuple[bool, float, str]:
    t0 = time.perf_counter()
    res = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - t0
    output = res.stdout if res.returncode == 0 else (res.stdout + "\n" + res.stderr)
    return res.returncode == 0, elapsed, output


def main() -> int:
    current_dir = Path(__file__).resolve().parent

    challenges = [
        ("Challenge 1: B04 UTD Accumulator Stress Test", current_dir / "challenge_01_utd_accumulator_stress.py"),
        ("Challenge 2: Bounded BFS APF Empirical Oracle", current_dir / "challenge_02_bounded_bfs_apf_oracle.py"),
        ("Challenge 3: Closed-Form 2x3 Affine Matrix Adversarial Test", current_dir / "challenge_03_closed_form_affine_adversarial.py"),
        ("Baseline Suite: Standalone Tests (8/8)", current_dir / "run_all_tests.py"),
    ]

    print("=" * 80)
    print(" NewCppRL Track A: Adversarial Challenge & Numerical Oracle Verification")
    print("=" * 80)

    all_passed = True
    total_start = time.perf_counter()

    for name, path in challenges:
        print(f"\n>>> Running {name}...")
        passed, elapsed, output = run_challenge_script(path)
        status_str = "[PASS]" if passed else "[FAIL]"
        print(f"    Status: {status_str} ({elapsed:.3f}s)")
        if not passed:
            all_passed = False
            print(output)
            break
        else:
            # Print last 3 lines of output as summary
            lines = [l for l in output.strip().splitlines() if l.strip()]
            for l in lines[-2:]:
                print(f"    | {l}")

    total_elapsed = time.perf_counter() - total_start
    print("\n" + "=" * 80)
    if all_passed:
        print(f" ALL ADVERSARIAL CHALLENGES & ORACLES PASSED ({total_elapsed:.3f}s)")
        print(" VERDICT: APPROVE")
    else:
        print(f" ADVERSARIAL CHALLENGE FAILED ({total_elapsed:.3f}s)")
        print(" VERDICT: REJECT")
    print("=" * 80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
