"""
n_scales 性能对比测试

测试配置:
1. n_scales=6, state_size=512, multiscale_feature_size=16
2. n_scales=7, state_size=1024, multiscale_feature_size=16

在 v2 环境中运行 100 steps 并比较耗时
"""
import time
import numpy as np
import sys
sys.path.insert(0, '.')

def run_performance_test(config_name: str, env_kwargs: dict, num_steps: int = 100, num_runs: int = 3):
    """运行性能测试"""
    from envs_new.cpp_env_v2 import CppEnv

    print(f"\n{'='*60}")
    print(f"配置: {config_name}")
    print(f"参数: n_scales={env_kwargs.get('n_scales')}, "
          f"state_size={env_kwargs.get('state_size')}, "
          f"state_downsize={env_kwargs.get('state_downsize')}, "
          f"multiscale_feature_size={env_kwargs.get('multiscale_feature_size')}")
    print(f"{'='*60}")

    run_times = []
    obs_shapes = []

    for run in range(num_runs):
        # 创建环境
        env = CppEnv(**env_kwargs)
        obs, info = env.reset(seed=42 + run)
        obs_shapes.append(obs['observation'].shape)

        # 预热
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                obs, info = env.reset()

        # 正式计时
        env.action_space.seed(123 + run)
        obs, info = env.reset(seed=42 + run)

        start_time = time.perf_counter()
        for step in range(num_steps):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                obs, info = env.reset()
        end_time = time.perf_counter()

        elapsed = end_time - start_time
        run_times.append(elapsed)
        print(f"  Run {run+1}: {elapsed:.4f}s ({num_steps/elapsed:.1f} steps/s)")

        env.close()

    avg_time = np.mean(run_times)
    std_time = np.std(run_times)
    steps_per_sec = num_steps / avg_time

    print(f"\n结果汇总:")
    print(f"  观测形状: {obs_shapes[0]}")
    print(f"  平均耗时: {avg_time:.4f}s ± {std_time:.4f}s")
    print(f"  平均速度: {steps_per_sec:.1f} steps/s")

    return {
        'config_name': config_name,
        'avg_time': avg_time,
        'std_time': std_time,
        'steps_per_sec': steps_per_sec,
        'obs_shape': obs_shapes[0]
    }


def main():
    print("="*60)
    print("n_scales 性能对比测试 - v2 环境")
    print("="*60)

    num_steps = 100
    num_runs = 3

    # 配置1: n_scales=6, state_size=512
    config1 = {
        'n_scales': 6,
        'state_size': (512, 512),
        'state_downsize': (512, 512),
        'multiscale_feature_size': 16,
        'use_global_features': True,
        'use_multiscale': True,
    }

    # 配置2: n_scales=7, state_size=1024
    config2 = {
        'n_scales': 7,
        'state_size': (1024, 1024),
        'state_downsize': (1024, 1024),
        'multiscale_feature_size': 16,
        'use_global_features': True,
        'use_multiscale': True,
    }

    results = []

    # 测试配置1
    result1 = run_performance_test("n_scales=6, state=512", config1, num_steps, num_runs)
    results.append(result1)

    # 测试配置2
    result2 = run_performance_test("n_scales=7, state=1024", config2, num_steps, num_runs)
    results.append(result2)

    # 对比结果
    print("\n" + "="*60)
    print("性能对比总结")
    print("="*60)

    print(f"\n{'配置':<25} {'观测形状':<20} {'速度 (steps/s)':<15} {'耗时比':<10}")
    print("-"*70)

    baseline_time = results[0]['avg_time']
    for r in results:
        ratio = r['avg_time'] / baseline_time
        print(f"{r['config_name']:<25} {str(r['obs_shape']):<20} {r['steps_per_sec']:<15.1f} {ratio:<10.2f}x")

    # 计算效率比
    speedup = results[0]['steps_per_sec'] / results[1]['steps_per_sec']
    print(f"\n结论: n_scales=6 (512) 比 n_scales=7 (1024) 快 {speedup:.2f}x")
    print(f"      n_scales=7 (1024) 比 n_scales=6 (512) 慢 {(1/speedup - 1)*100:.1f}%")


if __name__ == "__main__":
    main()
