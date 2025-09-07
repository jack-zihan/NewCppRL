#!/usr/bin/env python3
"""
可视化不同步结束时的视频录制机制
展示已done环境如何保持最后一帧
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def visualize_async_done():
    """模拟并可视化不同步结束的情况"""
    
    print("\n" + "="*80)
    print("模拟4个环境不同步结束的视频录制")
    print("="*80)
    
    # 模拟4个环境的运行情况
    env_done_steps = [20, 47, 71, 508]  # 各环境完成的步数
    max_steps = max(env_done_steps)
    video_skip = 10  # 每10步录制一帧
    
    print(f"\n环境完成步数: {env_done_steps}")
    print(f"最大步数: {max_steps}")
    print(f"录制间隔: 每{video_skip}步")
    
    # 模拟tensordict列表
    tds = [f"Env{i}_初始帧" for i in range(4)]
    dones = [False] * 4
    
    # 记录每个录制时刻的状态
    recording_frames = []
    recording_steps = []
    
    print("\n模拟运行过程:")
    print("-" * 60)
    
    for t in range(max_steps):
        # 更新每个环境的状态
        for idx in range(4):
            if not dones[idx]:
                # 环境还在运行，更新tensordict
                tds[idx] = f"Env{idx}_第{t}步"
                
                # 检查是否完成
                if t >= env_done_steps[idx] - 1:
                    dones[idx] = True
                    tds[idx] = f"Env{idx}_最终帧(步{t})"
                    print(f"  步{t:3d}: 环境{idx}完成！最终状态: {tds[idx]}")
        
        # 录制视频帧
        if (t + 1) % video_skip == 0:
            # 记录当前所有环境的状态（包括已done的）
            frame_data = {
                'step': t + 1,
                'states': tds.copy(),  # 复制当前状态
                'dones': dones.copy(),
                'active_count': sum([not d for d in dones])
            }
            recording_frames.append(frame_data)
            recording_steps.append(t + 1)
            
            # 打印录制信息
            print(f"\n  📹 第{t+1}步录制视频:")
            for i, (state, done) in enumerate(zip(tds, dones)):
                status = "✅已完成" if done else "🔄运行中"
                print(f"     环境{i} [{status}]: {state}")
            print(f"     活跃环境数: {frame_data['active_count']}/4")
        
        # 如果所有环境都完成，结束
        if all(dones):
            print(f"\n  步{t:3d}: 所有环境都完成了！结束评估")
            break
    
    # 可视化结果
    print("\n" + "="*80)
    print("视频录制总结:")
    print("="*80)
    print(f"总共录制了 {len(recording_frames)} 帧")
    print(f"录制时刻: {recording_steps}")
    
    # 创建可视化图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('不同步环境完成时的视频录制机制', fontsize=16)
    
    for env_idx in range(4):
        ax = axes[env_idx // 2, env_idx % 2]
        ax.set_title(f'环境{env_idx} (完成于步{env_done_steps[env_idx]})')
        ax.set_xlabel('时间步')
        ax.set_ylabel('状态')
        ax.set_xlim(0, max_steps)
        ax.set_ylim(-0.5, 1.5)
        
        # 绘制环境运行时段
        running_rect = Rectangle((0, 0.4), env_done_steps[env_idx], 0.2, 
                                 facecolor='green', alpha=0.3, label='运行中')
        ax.add_patch(running_rect)
        
        # 绘制环境完成后时段
        if env_done_steps[env_idx] < max_steps:
            done_rect = Rectangle((env_done_steps[env_idx], 0.4), 
                                 max_steps - env_done_steps[env_idx], 0.2,
                                 facecolor='red', alpha=0.3, label='已完成(保持最后帧)')
            ax.add_patch(done_rect)
        
        # 标记录制时刻
        for step in recording_steps:
            if step <= max_steps:
                ax.axvline(x=step, color='blue', linestyle='--', alpha=0.5)
                if step <= env_done_steps[env_idx]:
                    ax.plot(step, 0.5, 'bo', markersize=8)  # 运行中录制
                else:
                    ax.plot(step, 0.5, 'ro', markersize=8)  # 完成后录制
        
        # 标记完成时刻
        ax.axvline(x=env_done_steps[env_idx], color='red', linewidth=2, label='完成时刻')
        
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/lzh/NewCppRL/tests/async_done_visualization.png', dpi=100)
    print(f"\n可视化图表已保存到: async_done_visualization.png")
    
    # 详细展示每帧的内容
    print("\n" + "="*80)
    print("详细帧内容分析:")
    print("="*80)
    
    for i, frame in enumerate(recording_frames):
        print(f"\n第{i+1}帧 (步{frame['step']}):")
        print(f"  活跃环境: {frame['active_count']}/4")
        for env_idx, (state, done) in enumerate(zip(frame['states'], frame['dones'])):
            if done:
                print(f"  环境{env_idx}: {state} [保持不变]")
            else:
                print(f"  环境{env_idx}: {state} [持续更新]")
    
    print("\n" + "="*80)
    print("关键洞察:")
    print("1. 已完成的环境保持最后一帧状态不变")
    print("2. 未完成的环境继续更新新的帧")
    print("3. 视频录制始终包含所有4个环境的画面")
    print("4. 2x2网格视频中，已完成环境显示静止画面")
    print("5. 直到最后一个环境完成，评估才结束")
    print("="*80)


if __name__ == "__main__":
    visualize_async_done()