#!/usr/bin/env python3
"""
检查转置/旋转问题的根源
"""

import numpy as np
import cv2
import sys
sys.path.append('/home/lzh/NewCppRL')

from envs_new.cpp_env_v2 import CppEnv as NewCppEnv
from envs.cpp_env_v2 import CppEnv as OldCppEnv

def check_transpose():
    print("🔍 检查转置/旋转问题...")
    print("=" * 60)
    
    # 创建环境
    new_env = NewCppEnv(render_mode='rgb_array')
    old_env = OldCppEnv(render_mode='rgb_array')
    
    # 重置
    new_obs, _ = new_env.reset(seed=42)
    old_obs, _ = old_env.reset(seed=42)
    
    print("📊 环境信息:")
    print(f"  新版agent位置: {new_env.agent.position}")
    print(f"  旧版agent位置: {old_env.agent.position}")
    
    # 渲染
    new_img = new_env.render()
    
    # 旧版使用render_map
    if hasattr(old_env, 'render_map'):
        old_img = old_env.render_map()
    else:
        old_img = old_env.render()
    
    # 确保数据类型正确
    old_img = old_img.astype(np.uint8)
    
    print(f"\n🖼️ 渲染图像:")
    print(f"  新版形状: {new_img.shape}")
    print(f"  旧版形状: {old_img.shape}")
    
    # 缩小新版图像到原始大小以便比较
    if new_img.shape[0] == 2 * old_img.shape[0]:
        print("\n📐 缩放分析:")
        print(f"  新版是2倍缩放 (render_repeat_times=2)")
        
        # 缩小新版到原始大小
        new_img_downscaled = new_img[::2, ::2]
        print(f"  缩小后新版形状: {new_img_downscaled.shape}")
        
        # 比较缩小后的图像
        if new_img_downscaled.shape == old_img.shape:
            diff = np.mean(np.abs(new_img_downscaled.astype(float) - old_img.astype(float)))
            similarity = 100 * (1 - diff / 255)
            print(f"  直接对比相似度: {similarity:.1f}%")
            
            # 保存对比图像
            cv2.imwrite('/tmp/new_downscaled.png', cv2.cvtColor(new_img_downscaled, cv2.COLOR_RGB2BGR))
            cv2.imwrite('/tmp/old_original.png', cv2.cvtColor(old_img, cv2.COLOR_RGB2BGR))
        
        # 测试转置
        new_img_transposed = np.transpose(new_img_downscaled, (1, 0, 2))
        if new_img_transposed.shape == old_img.shape:
            diff = np.mean(np.abs(new_img_transposed.astype(float) - old_img.astype(float)))
            similarity = 100 * (1 - diff / 255)
            print(f"  转置后相似度: {similarity:.1f}%")
            
            if similarity > 90:
                print(f"  ⚠️ 发现问题：新版图像是旧版的转置！")
                cv2.imwrite('/tmp/new_transposed.png', cv2.cvtColor(new_img_transposed, cv2.COLOR_RGB2BGR))
    
    # 执行一些步骤来观察agent移动
    print("\n🚶 执行步骤观察移动方向:")
    
    # 向右移动（增加x）
    action_right = 7  # 假设这是向右的动作
    
    for i in range(3):
        new_obs, _, _, _, _ = new_env.step(action_right)
        old_obs, _, _, _, _ = old_env.step(action_right)
        
        print(f"  Step {i+1}:")
        print(f"    新版位置: {new_env.agent.position}")
        print(f"    旧版位置: {old_env.agent.position}")
    
    # 渲染移动后的图像
    new_img_moved = new_env.render()
    old_img_moved = old_env.render_map() if hasattr(old_env, 'render_map') else old_env.render()
    old_img_moved = old_img_moved.astype(np.uint8)
    
    # 保存移动后的图像
    cv2.imwrite('/tmp/new_moved.png', cv2.cvtColor(new_img_moved, cv2.COLOR_RGB2BGR))
    cv2.imwrite('/tmp/old_moved.png', cv2.cvtColor(old_img_moved, cv2.COLOR_RGB2BGR))
    
    # 分析地图坐标系
    print("\n🗺️ 地图坐标系分析:")
    
    # 在特定位置放置标记
    if hasattr(new_env, 'maps_dict') and 'trajectory' in new_env.maps_dict:
        # 清空轨迹
        new_env.maps_dict['trajectory'].fill(0)
        
        # 在特定位置画点 (100, 50) - x=100, y=50
        test_x, test_y = 100, 50
        # OpenCV使用 (x, y)
        cv2.circle(new_env.maps_dict['trajectory'], (test_x, test_y), 3, 1, -1)
        print(f"  新版：在(x={test_x}, y={test_y})画圆")
        
    if hasattr(old_env, 'map_trajectory'):
        old_env.map_trajectory.fill(0)
        cv2.circle(old_env.map_trajectory, (test_x, test_y), 3, 1., -1)
        print(f"  旧版：在(x={test_x}, y={test_y})画圆")
    
    # 渲染标记后的图像
    new_img_marked = new_env.render()
    old_img_marked = old_env.render_map() if hasattr(old_env, 'render_map') else old_env.render()
    old_img_marked = old_img_marked.astype(np.uint8)
    
    cv2.imwrite('/tmp/new_marked.png', cv2.cvtColor(new_img_marked, cv2.COLOR_RGB2BGR))
    cv2.imwrite('/tmp/old_marked.png', cv2.cvtColor(old_img_marked, cv2.COLOR_RGB2BGR))
    
    print("\n📷 图像已保存到/tmp/")
    print("  对比 new_marked.png 和 old_marked.png 查看坐标系差异")
    
    # 检查观察生成
    print("\n👁️ 观察生成检查:")
    print(f"  新版观察形状: {new_obs.shape}")
    print(f"  旧版观察形状: {old_obs.shape}")
    
    if new_obs.shape != old_obs.shape:
        print(f"  ⚠️ 观察形状不同！训练会受影响！")
    
    new_env.close()
    old_env.close()
    
    print("\n" + "=" * 60)
    print("诊断完成！")

if __name__ == "__main__":
    check_transpose()