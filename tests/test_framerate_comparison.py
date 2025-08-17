#!/usr/bin/env python3
"""
测试新旧环境的运行帧率对比
比较两个版本的性能差异
"""

import time
import numpy as np
import sys
sys.path.append('/home/lzh/NewCppRL')

# 导入新旧版本的环境
from envs.cpp_env_v2 import CppEnv as OldCppEnv
from envs_new.cpp_env_v2 import CppEnv as NewCppEnv

def test_environment_framerate(env_class, env_name, num_steps=1000, num_episodes=5):
    """
    测试单个环境的帧率
    
    Args:
        env_class: 环境类
        env_name: 环境名称（用于显示）
        num_steps: 每个episode运行的步数
        num_episodes: 测试的episode数量
    
    Returns:
        dict: 包含帧率统计信息
    """
    print(f"\n{'='*60}")
    print(f"测试 {env_name}")
    print(f"{'='*60}")
    
    # 创建环境
    env = env_class(render_mode=None)  # 不渲染以获得纯计算性能
    
    step_times = []
    reset_times = []
    episode_fps = []
    
    for episode in range(num_episodes):
        # 测试reset时间
        reset_start = time.perf_counter()
        obs, info = env.reset(seed=42 + episode)
        reset_end = time.perf_counter()
        reset_times.append(reset_end - reset_start)
        
        # 测试step时间
        episode_step_times = []
        episode_start = time.perf_counter()
        
        for step in range(num_steps):
            # 随机动作
            action = env.action_space.sample()
            
            # 测量step时间
            step_start = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(action)
            step_end = time.perf_counter()
            
            episode_step_times.append(step_end - step_start)
            
            if terminated or truncated:
                obs, info = env.reset(seed=42 + episode * 1000 + step)
        
        episode_end = time.perf_counter()
        episode_time = episode_end - episode_start
        episode_fps.append(num_steps / episode_time)
        step_times.extend(episode_step_times)
        
        print(f"  Episode {episode+1}/{num_episodes}: {num_steps/episode_time:.1f} FPS")
    
    env.close()
    
    # 计算统计信息
    stats = {
        'mean_fps': np.mean(episode_fps),
        'std_fps': np.std(episode_fps),
        'min_fps': np.min(episode_fps),
        'max_fps': np.max(episode_fps),
        'mean_step_time': np.mean(step_times) * 1000,  # 转换为毫秒
        'std_step_time': np.std(step_times) * 1000,
        'mean_reset_time': np.mean(reset_times) * 1000,
        'std_reset_time': np.std(reset_times) * 1000,
    }
    
    return stats

def test_environment_with_rendering(env_class, env_name, num_steps=500):
    """
    测试带渲染的环境帧率
    """
    print(f"\n测试带渲染的 {env_name}")
    
    env = env_class(render_mode='rgb_array')
    obs, info = env.reset(seed=42)
    
    render_times = []
    total_start = time.perf_counter()
    
    for step in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 测试渲染时间
        render_start = time.perf_counter()
        frame = env.render()
        render_end = time.perf_counter()
        render_times.append(render_end - render_start)
        
        if terminated or truncated:
            obs, info = env.reset(seed=42 + step)
    
    total_end = time.perf_counter()
    total_time = total_end - total_start
    
    env.close()
    
    stats = {
        'fps_with_render': num_steps / total_time,
        'mean_render_time': np.mean(render_times) * 1000,
        'std_render_time': np.std(render_times) * 1000,
    }
    
    return stats

def main():
    print("🚀 环境帧率对比测试")
    print("=" * 80)
    
    # 测试参数
    num_steps = 1000
    num_episodes = 5
    
    # 测试旧版本（无渲染）
    print("\n📊 测试无渲染性能...")
    old_stats = test_environment_framerate(OldCppEnv, "旧版本环境", num_steps, num_episodes)
    new_stats = test_environment_framerate(NewCppEnv, "新版本环境", num_steps, num_episodes)
    
    # 测试带渲染性能
    print("\n🎨 测试渲染性能...")
    old_render_stats = test_environment_with_rendering(OldCppEnv, "旧版本环境", 500)
    new_render_stats = test_environment_with_rendering(NewCppEnv, "新版本环境", 500)
    
    # 显示对比结果
    print("\n" + "=" * 80)
    print("📈 性能对比总结")
    print("=" * 80)
    
    print("\n【无渲染性能】")
    print(f"{'指标':<20} {'旧版本':>15} {'新版本':>15} {'差异':>15}")
    print("-" * 65)
    
    metrics = [
        ('平均帧率 (FPS)', old_stats['mean_fps'], new_stats['mean_fps']),
        ('帧率标准差', old_stats['std_fps'], new_stats['std_fps']),
        ('最小帧率', old_stats['min_fps'], new_stats['min_fps']),
        ('最大帧率', old_stats['max_fps'], new_stats['max_fps']),
        ('平均步时间 (ms)', old_stats['mean_step_time'], new_stats['mean_step_time']),
        ('步时间标准差 (ms)', old_stats['std_step_time'], new_stats['std_step_time']),
        ('平均重置时间 (ms)', old_stats['mean_reset_time'], new_stats['mean_reset_time']),
    ]
    
    for name, old_val, new_val in metrics:
        if 'FPS' in name or '帧率' in name:
            diff = ((new_val - old_val) / old_val) * 100
            print(f"{name:<20} {old_val:>14.1f} {new_val:>14.1f} {diff:>13.1f}%")
        else:
            diff = ((new_val - old_val) / old_val) * 100
            print(f"{name:<20} {old_val:>14.2f} {new_val:>14.2f} {diff:>13.1f}%")
    
    print("\n【渲染性能】")
    print(f"{'指标':<20} {'旧版本':>15} {'新版本':>15} {'差异':>15}")
    print("-" * 65)
    
    render_metrics = [
        ('带渲染帧率 (FPS)', old_render_stats['fps_with_render'], new_render_stats['fps_with_render']),
        ('平均渲染时间 (ms)', old_render_stats['mean_render_time'], new_render_stats['mean_render_time']),
        ('渲染时间标准差 (ms)', old_render_stats['std_render_time'], new_render_stats['std_render_time']),
    ]
    
    for name, old_val, new_val in render_metrics:
        diff = ((new_val - old_val) / old_val) * 100
        if 'FPS' in name:
            print(f"{name:<20} {old_val:>14.1f} {new_val:>14.1f} {diff:>13.1f}%")
        else:
            print(f"{name:<20} {old_val:>14.2f} {new_val:>14.2f} {diff:>13.1f}%")
    
    # 性能总结
    print("\n" + "=" * 80)
    print("📊 性能分析")
    print("-" * 80)
    
    fps_improvement = ((new_stats['mean_fps'] - old_stats['mean_fps']) / old_stats['mean_fps']) * 100
    step_time_change = ((new_stats['mean_step_time'] - old_stats['mean_step_time']) / old_stats['mean_step_time']) * 100
    
    if fps_improvement > 0:
        print(f"✅ 新版本性能提升: 平均帧率提高 {fps_improvement:.1f}%")
    elif fps_improvement < 0:
        print(f"⚠️ 新版本性能下降: 平均帧率降低 {abs(fps_improvement):.1f}%")
    else:
        print(f"➡️ 两版本性能相当")
    
    print(f"\n详细对比:")
    print(f"  • 无渲染帧率: 旧版 {old_stats['mean_fps']:.1f} FPS → 新版 {new_stats['mean_fps']:.1f} FPS")
    print(f"  • 带渲染帧率: 旧版 {old_render_stats['fps_with_render']:.1f} FPS → 新版 {new_render_stats['fps_with_render']:.1f} FPS")
    print(f"  • 单步耗时: 旧版 {old_stats['mean_step_time']:.2f}ms → 新版 {new_stats['mean_step_time']:.2f}ms")
    print(f"  • 渲染耗时: 旧版 {old_render_stats['mean_render_time']:.2f}ms → 新版 {new_render_stats['mean_render_time']:.2f}ms")
    
    # 瓶颈分析
    print(f"\n瓶颈分析:")
    old_render_ratio = old_render_stats['mean_render_time'] / (old_stats['mean_step_time'] + old_render_stats['mean_render_time']) * 100
    new_render_ratio = new_render_stats['mean_render_time'] / (new_stats['mean_step_time'] + new_render_stats['mean_render_time']) * 100
    
    print(f"  • 旧版本渲染占比: {old_render_ratio:.1f}%")
    print(f"  • 新版本渲染占比: {new_render_ratio:.1f}%")
    
    if new_stats['std_fps'] < old_stats['std_fps']:
        print(f"  ✅ 新版本帧率更稳定（标准差更小）")
    elif new_stats['std_fps'] > old_stats['std_fps']:
        print(f"  ⚠️ 新版本帧率波动更大（标准差更大）")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()