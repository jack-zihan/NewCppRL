"""详细的端到端APF性能测试"""
import time
import numpy as np
import gc
import torch

def clear_gpu_cache():
    """清理GPU缓存"""
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except:
        pass
    
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except:
        pass
    
    gc.collect()

def test_env_fps(use_gpu=False, num_steps=1000, warmup_steps=100):
    """测试环境帧率"""
    # 动态导入和替换
    import envs_new.cpp_env_v2 as v2_module
    
    if use_gpu:
        from envs_new.utils.gpu_apf import gpu_apf_bool
        original = v2_module.cpu_apf_bool
        v2_module.cpu_apf_bool = gpu_apf_bool
    
    from envs_new.cpp_env_v2 import CppEnv
    
    # 创建环境
    env = CppEnv()
    obs, info = env.reset(seed=42)
    
    # 预热
    print(f"  预热 {warmup_steps} 步...")
    for _ in range(warmup_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    
    # 清理缓存
    clear_gpu_cache()
    time.sleep(0.5)  # 等待系统稳定
    
    # 正式测试
    print(f"  正式测试 {num_steps} 步...")
    step_times = []
    start_total = time.perf_counter()
    
    for i in range(num_steps):
        step_start = time.perf_counter()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        step_time = time.perf_counter() - step_start
        step_times.append(step_time)
        
        if terminated or truncated:
            obs, info = env.reset()
        
        if (i + 1) % 200 == 0:
            print(f"    进度: {i+1}/{num_steps} 步")
    
    total_time = time.perf_counter() - start_total
    fps = num_steps / total_time
    
    env.close()
    
    # 恢复原函数
    if use_gpu:
        v2_module.cpu_apf_bool = original
    
    # 统计信息
    step_times = np.array(step_times) * 1000  # 转换为毫秒
    stats = {
        'fps': fps,
        'total_time': total_time,
        'mean_step_ms': np.mean(step_times),
        'median_step_ms': np.median(step_times),
        'std_step_ms': np.std(step_times),
        'min_step_ms': np.min(step_times),
        'max_step_ms': np.max(step_times),
        'p95_step_ms': np.percentile(step_times, 95),
        'p99_step_ms': np.percentile(step_times, 99)
    }
    
    return stats

print("=" * 60)
print("详细端到端APF性能测试")
print("=" * 60)

# 测试参数
num_steps = 1000
num_runs = 3

# 清理系统
print("\n清理系统资源...")
clear_gpu_cache()
time.sleep(2)  # 等待系统完全稳定

# CPU测试
print(f"\n📊 CPU APF测试 ({num_runs}轮，每轮{num_steps}步)")
print("-" * 40)
cpu_stats_list = []
for i in range(num_runs):
    print(f"\n第{i+1}轮:")
    stats = test_env_fps(use_gpu=False, num_steps=num_steps)
    cpu_stats_list.append(stats)
    print(f"  FPS: {stats['fps']:.1f}")
    print(f"  平均步时间: {stats['mean_step_ms']:.2f}ms")
    print(f"  中位数步时间: {stats['median_step_ms']:.2f}ms")
    clear_gpu_cache()
    time.sleep(1)

# GPU测试
print(f"\n📊 GPU APF测试 ({num_runs}轮，每轮{num_steps}步)")
print("-" * 40)
gpu_stats_list = []
for i in range(num_runs):
    print(f"\n第{i+1}轮:")
    stats = test_env_fps(use_gpu=True, num_steps=num_steps)
    gpu_stats_list.append(stats)
    print(f"  FPS: {stats['fps']:.1f}")
    print(f"  平均步时间: {stats['mean_step_ms']:.2f}ms")
    print(f"  中位数步时间: {stats['median_step_ms']:.2f}ms")
    clear_gpu_cache()
    time.sleep(1)

# 计算平均值
def average_stats(stats_list):
    avg = {}
    for key in stats_list[0].keys():
        avg[key] = np.mean([s[key] for s in stats_list])
    return avg

cpu_avg = average_stats(cpu_stats_list)
gpu_avg = average_stats(gpu_stats_list)

# 结果汇总
print("\n" + "=" * 60)
print("📈 性能对比结果")
print("=" * 60)

print("\n🖥️ CPU APF性能:")
print(f"  平均FPS: {cpu_avg['fps']:.1f}")
print(f"  平均步时间: {cpu_avg['mean_step_ms']:.2f}ms ± {cpu_avg['std_step_ms']:.2f}ms")
print(f"  中位数步时间: {cpu_avg['median_step_ms']:.2f}ms")
print(f"  P95步时间: {cpu_avg['p95_step_ms']:.2f}ms")
print(f"  P99步时间: {cpu_avg['p99_step_ms']:.2f}ms")

print("\n🎮 GPU APF性能:")
print(f"  平均FPS: {gpu_avg['fps']:.1f}")
print(f"  平均步时间: {gpu_avg['mean_step_ms']:.2f}ms ± {gpu_avg['std_step_ms']:.2f}ms")
print(f"  中位数步时间: {gpu_avg['median_step_ms']:.2f}ms")
print(f"  P95步时间: {gpu_avg['p95_step_ms']:.2f}ms")
print(f"  P99步时间: {gpu_avg['p99_step_ms']:.2f}ms")

print("\n🚀 性能提升:")
fps_improvement = (gpu_avg['fps'] - cpu_avg['fps']) / cpu_avg['fps'] * 100
time_saved = cpu_avg['mean_step_ms'] - gpu_avg['mean_step_ms']
speedup = gpu_avg['fps'] / cpu_avg['fps']

print(f"  FPS提升: {fps_improvement:+.1f}% ({speedup:.2f}x)")
print(f"  每步节省: {time_saved:.2f}ms")
print(f"  中位数提升: {(cpu_avg['median_step_ms'] - gpu_avg['median_step_ms']):.2f}ms")
print(f"  稳定性对比: CPU σ={cpu_avg['std_step_ms']:.2f}ms, GPU σ={gpu_avg['std_step_ms']:.2f}ms")

# 判断结果
print("\n📝 结论:")
if fps_improvement > 10:
    print(f"  ✅ GPU APF显著提升性能 ({fps_improvement:.1f}%)")
elif fps_improvement > 0:
    print(f"  ✅ GPU APF略微提升性能 ({fps_improvement:.1f}%)")
elif fps_improvement > -5:
    print(f"  ⚠️ GPU APF性能相当 ({fps_improvement:.1f}%)")
else:
    print(f"  ❌ GPU APF性能下降 ({fps_improvement:.1f}%)")

print("\n" + "=" * 60)