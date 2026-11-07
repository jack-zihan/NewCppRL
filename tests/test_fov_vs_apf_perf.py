import time

import numpy as np

from cpu_apf import cpu_apf_bool
from cpu_fov import cpu_fov_bool
from envs_new.components.config.environment_config import EnvironmentConfig
from envs_new.components.entity.agent import AgentFactory


def _python_fov_update(mist: np.ndarray, obstacle: np.ndarray, agent, num_rays: int = 180) -> None:
    """Simple LiDAR-style FOV in pure Python for perf comparison."""
    height, width = mist.shape
    vision_length = float(agent.vision_length)
    half_angle = float(agent.vision_angle) / 2.0

    # angles in degrees around agent.direction
    start_angle = agent.direction - half_angle
    end_angle = agent.direction + half_angle
    if num_rays <= 1:
        angles = [agent.direction]
    else:
        angles = np.linspace(start_angle, end_angle, num_rays)

    rad_angles = np.deg2rad(angles)
    cos_vals = np.cos(rad_angles)
    sin_vals = np.sin(rad_angles)

    x0, y0 = agent.position

    max_steps = int(vision_length)
    for cos_a, sin_a in zip(cos_vals, sin_vals):
        for step in range(1, max_steps + 1):
            x = x0 + step * cos_a
            y = y0 + step * sin_a
            xi = int(round(x))
            yi = int(round(y))

            if xi < 0 or yi < 0 or xi >= width or yi >= height:
                break

            mist[yi, xi] = 1
            if obstacle[yi, xi]:
                break


def _make_dummy_maps(size: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Create a simple obstacle map and empty mist for benchmarking."""
    obstacle = np.zeros((size, size), dtype=np.uint8)

    # Add a thin vertical wall in the middle
    wall_x = size // 2
    obstacle[:, wall_x : wall_x + 2] = 1

    mist = np.zeros_like(obstacle, dtype=np.uint8)
    return obstacle, mist


def _make_dummy_agent(size: int = 400):
    cfg = EnvironmentConfig()
    # Place agent somewhere on the left side, facing right (0 degrees)
    position = (size * 0.25, size * 0.5)
    direction = 0.0
    return AgentFactory.create_mower_agent(cfg, position=position, direction=direction)


def _run_single_benchmark(num_rays: int = 72):
    """Run one benchmark configuration with given num_rays."""
    size = 400
    obstacle, mist = _make_dummy_maps(size=size)
    agent = _make_dummy_agent(size=size)

    # Warm up FOV (python)
    _python_fov_update(mist, obstacle, agent, num_rays=num_rays)

    # Benchmark Python FOV
    num_runs = 100
    start = time.perf_counter()
    for _ in range(num_runs):
        mist_tmp = np.zeros_like(mist)
        _python_fov_update(mist_tmp, obstacle, agent, num_rays=num_rays)
    fov_elapsed = time.perf_counter() - start

    # Prepare APF input
    binary_map = obstacle.copy()

    # Warm up APF
    cpu_apf_bool(binary_map)

    # Benchmark C++ APF
    start = time.perf_counter()
    for _ in range(num_runs):
        cpu_apf_bool(binary_map)
    apf_elapsed = time.perf_counter() - start

    # Benchmark C++ FOV
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = cpu_fov_bool(
            obstacle,
            agent.x,
            agent.y,
            agent.direction,
            agent.vision_length,
            agent.vision_angle,
            num_rays,
        )
    fov_cpp_elapsed = time.perf_counter() - start

    # Print helpful info for manual inspection
    print(f"\n=== Benchmark with num_rays={num_rays} ===")
    print(f"FOV 100 runs (python): {fov_elapsed:.6f}s")
    print(f"APF 100 runs (C++):    {apf_elapsed:.6f}s")
    print(f"FOV 100 runs (C++):    {fov_cpp_elapsed:.6f}s")
    if apf_elapsed > 0:
        print(f"FOV/APF time ratio (python): {fov_elapsed / apf_elapsed:.3f}")
        print(f"FOV_CPP/APF time ratio:      {fov_cpp_elapsed / apf_elapsed:.3f}")

    # Sanity assertions
    if apf_elapsed > 0:
        assert fov_elapsed / apf_elapsed < 10.0
    if fov_cpp_elapsed > 0:
        assert fov_cpp_elapsed < fov_elapsed


def test_fov_vs_apf_performance():
    """Compare Python/C++ FOV vs C++ APF on a 400x400 grid.

    Runs benchmarks for num_rays=72 and num_rays=180.
    """
    _run_single_benchmark(num_rays=72)
    _run_single_benchmark(num_rays=180)


if __name__ == "__main__":
    # Allow running as a standalone script without pytest
    test_fov_vs_apf_performance()
