#!/usr/bin/env python3
"""
快速测试新旧环境的运行帧率对比
"""

import time
import numpy as np
import sys
sys.path.append('/home/lzh/NewCppRL')

# 导入新旧版本的环境
from envs.cpp_env_v2 import CppEnv as OldCppEnv
from envs_new.cpp_env_v2 import CppEnv as NewCppEnv

def quick_fps_test(env_class, env_name, num_steps=200):
    """快速测试环境帧率"""
    print(f"\n测试 {env_name}...")
    
    # 创建环境（无渲染）
    env = env_class(render_mode=None)
    
    # 预热
    obs, info = env.reset(seed=42)
    for _ in range(10):
        action = env.action_space.sample()
        env.step(action)
    
    # 重置并开始正式测试
    obs, info = env.reset(seed=42)
    
    # 测试纯step性能
    start_time = time.perf_counter()
    for step in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset(seed=42 + step)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    fps = num_steps / total_time
    avg_step_time = (total_time / num_steps) * 1000  # 毫秒
    
    env.close()
    
    return fps, avg_step_time

def quick_render_test(env_class, env_name, num_steps=100):
    """快速测试带渲染的帧率"""
    print(f"测试带渲染的 {env_name}...")
    
    env = env_class(render_mode='rgb_array')
    obs, info = env.reset(seed=42)
    
    # 预热
    for _ in range(5):
        env.step(env.action_space.sample())
        env.render()
    
    # 重置并开始正式测试
    obs, info = env.reset(seed=42)
    
    start_time = time.perf_counter()
    for step in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        if terminated or truncated:
            obs, info = env.reset(seed=42 + step)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    fps_with_render = num_steps / total_time
    
    env.close()
    
    return fps_with_render

def main():
    print("🚀 环境帧率快速对比测试")
    print("=" * 60)
    
    # 测试无渲染性能
    print("\n📊 无渲染性能测试:")
    old_fps, old_step_time = quick_fps_test(OldCppEnv, "旧版本", 200)
    new_fps, new_step_time = quick_fps_test(NewCppEnv, "新版本", 200)
    
    # 测试渲染性能
    print("\n🎨 渲染性能测试:")
    old_render_fps = quick_render_test(OldCppEnv, "旧版本", 100)
    new_render_fps = quick_render_test(NewCppEnv, "新版本", 100)
    
    # 显示结果
    print("\n" + "=" * 60)
    print("📈 测试结果")
    print("=" * 60)
    
    print(f"\n{'环境版本':<10} {'无渲染FPS':>12} {'单步耗时(ms)':>12} {'带渲染FPS':>12}")
    print("-" * 50)
    print(f"{'旧版本':<10} {old_fps:>11.1f} {old_step_time:>11.2f} {old_render_fps:>11.1f}")
    print(f"{'新版本':<10} {new_fps:>11.1f} {new_step_time:>11.2f} {new_render_fps:>11.1f}")
    
    # 计算性能差异
    fps_diff = ((new_fps - old_fps) / old_fps) * 100
    step_diff = ((new_step_time - old_step_time) / old_step_time) * 100
    render_diff = ((new_render_fps - old_render_fps) / old_render_fps) * 100
    
    print("\n" + "=" * 60)
    print("📊 性能对比")
    print("=" * 60)
    
    print(f"\n无渲染性能:")
    print(f"  • 帧率变化: {fps_diff:+.1f}% ({old_fps:.1f} → {new_fps:.1f} FPS)")
    print(f"  • 步时变化: {step_diff:+.1f}% ({old_step_time:.2f} → {new_step_time:.2f} ms)")
    
    print(f"\n渲染性能:")
    print(f"  • 帧率变化: {render_diff:+.1f}% ({old_render_fps:.1f} → {new_render_fps:.1f} FPS)")
    
    # 总结
    print("\n" + "=" * 60)
    if fps_diff > 5:
        print("✅ 结论: 新版本性能明显提升")
    elif fps_diff < -5:
        print("⚠️ 结论: 新版本性能有所下降")
    else:
        print("➡️ 结论: 两版本性能相当")
    
    # 性能指标参考
    print(f"\n参考标准:")
    print(f"  • 训练推荐: >1000 FPS (无渲染)")
    print(f"  • 实时交互: >30 FPS (带渲染)")
    print(f"  • 当前新版本: {new_fps:.0f} FPS (无渲染), {new_render_fps:.0f} FPS (带渲染)")

if __name__ == "__main__":
    main()