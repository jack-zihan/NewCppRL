#!/usr/bin/env python3
"""
综合诊断渲染旋转和轨迹粗细问题
深入分析坐标系统差异
"""

import numpy as np
import cv2
import sys
import os

# 添加路径
sys.path.append('/home/lzh/NewCppRL')

from envs_new.cpp_env_v2 import CppEnv as NewCppEnv
from envs.cpp_env_v2 import CppEnv as OldCppEnv

def analyze_coordinate_systems():
    """深入分析坐标系统差异"""
    print("🔍 综合诊断渲染差异问题...")
    print("=" * 60)
    
    # 创建环境
    print("创建环境...")
    new_env = NewCppEnv(render_mode='rgb_array')
    old_env = OldCppEnv(render_mode='rgb_array')
    
    # 使用相同的种子重置
    print("重置环境...")
    new_obs, _ = new_env.reset(seed=42)
    old_obs, _ = old_env.reset(seed=42)
    
    print("\n📊 基础配置对比:")
    print(f"  新版dimensions: {new_env.dimensions if hasattr(new_env, 'dimensions') else 'N/A'}")
    print(f"  旧版dimensions: {old_env.dimensions if hasattr(old_env, 'dimensions') else 'N/A'}")
    
    # 检查地图属性
    print("\n🗺️ 地图属性对比:")
    
    # 新版使用maps_dict
    if hasattr(new_env, 'maps_dict'):
        for map_name in ['field', 'trajectory', 'weed', 'obstacle']:
            if map_name in new_env.maps_dict:
                print(f"  新版 {map_name}: shape={new_env.maps_dict[map_name].shape}")
    
    # 旧版使用map_*属性
    for map_name in ['field', 'trajectory', 'weed', 'obstacle']:
        attr_name = f'map_{map_name}'
        if hasattr(old_env, attr_name):
            map_data = getattr(old_env, attr_name)
            print(f"  旧版 {attr_name}: shape={map_data.shape}")
    
    print("\n📍 Agent位置对比:")
    print(f"  新版: position={new_env.agent.position}, direction={new_env.agent.direction}")
    print(f"  旧版: position={old_env.agent.position}, direction={old_env.agent.direction}")
    
    # 渲染图像
    print("\n🎨 渲染图像...")
    new_img = new_env.render()
    old_img = old_env.render_map() if hasattr(old_env, 'render_map') else old_env.render()
    
    print(f"  新版渲染形状: {new_img.shape}")
    print(f"  旧版渲染形状: {old_img.shape}")
    
    # 测试转置
    print("\n🔄 转置测试:")
    if new_img.shape[:2] == old_img.shape[:2][::-1]:
        print(f"  ⚠️ 形状是转置关系: 新版{new_img.shape[:2]} vs 旧版{old_img.shape[:2]}")
        
        # 转置新版图像
        new_img_transposed = np.transpose(new_img, (1, 0, 2))
        print(f"  转置后新版形状: {new_img_transposed.shape}")
        
        if new_img_transposed.shape == old_img.shape:
            # 计算相似度
            diff = np.mean(np.abs(new_img_transposed.astype(float) - old_img.astype(float)))
            similarity = 100 * (1 - diff / 255)
            print(f"  转置后相似度: {similarity:.1f}%")
            
            # 保存转置版本
            cv2.imwrite('/tmp/new_transposed.png', cv2.cvtColor(new_img_transposed, cv2.COLOR_RGB2BGR))
    
    # 检查轨迹线属性
    print("\n✏️ 轨迹线渲染对比:")
    
    # 执行几步来生成轨迹
    for i in range(5):
        action = new_env.action_space.sample()
        new_env.step(action)
        old_env.step(action)
    
    # 再次渲染
    new_img_with_traj = new_env.render()
    old_img_with_traj = old_env.render_map() if hasattr(old_env, 'render_map') else old_env.render()
    
    # 分析轨迹地图
    if hasattr(new_env, 'maps_dict') and 'trajectory' in new_env.maps_dict:
        new_traj = new_env.maps_dict['trajectory']
        print(f"  新版trajectory地图: 非零像素={np.sum(new_traj > 0)}")
        
        # 检查轨迹线的形态学特征
        kernel = np.ones((3,3), np.uint8)
        new_traj_dilated = cv2.dilate(new_traj.astype(np.uint8), kernel, iterations=1)
        thickness_estimate = np.sum(new_traj_dilated > 0) / max(np.sum(new_traj > 0), 1)
        print(f"  新版轨迹线粗细估计: {thickness_estimate:.2f}")
    
    if hasattr(old_env, 'map_trajectory'):
        old_traj = old_env.map_trajectory
        print(f"  旧版trajectory地图: 非零像素={np.sum(old_traj > 0)}")
        
        kernel = np.ones((3,3), np.uint8)
        old_traj_dilated = cv2.dilate(old_traj.astype(np.uint8), kernel, iterations=1)
        thickness_estimate = np.sum(old_traj_dilated > 0) / max(np.sum(old_traj > 0), 1)
        print(f"  旧版轨迹线粗细估计: {thickness_estimate:.2f}")
    
    # 检查渲染中的轨迹颜色
    print("\n🎨 轨迹渲染颜色对比:")
    
    # 新版轨迹颜色（紫色）
    trajectory_color_new = (255, 38, 255)  # RENDER_COLORS['trajectory']
    print(f"  新版轨迹颜色: RGB{trajectory_color_new}")
    
    # 旧版轨迹颜色
    trajectory_color_old = (255, 38, 255)  # 从代码中看到的颜色
    print(f"  旧版轨迹颜色: RGB{trajectory_color_old}")
    
    # 坐标系统深入分析
    print("\n🔬 坐标系统深入分析:")
    
    # 测试特定坐标点
    test_points = [(50, 100), (100, 50), (150, 150)]
    
    for x, y in test_points:
        print(f"\n  测试点 (x={x}, y={y}):")
        
        # 新版
        if hasattr(new_env, 'maps_dict') and 'field' in new_env.maps_dict:
            field_shape = new_env.maps_dict['field'].shape
            if y < field_shape[0] and x < field_shape[1]:
                # NumPy索引: [y, x]
                value = new_env.maps_dict['field'][y, x]
                print(f"    新版 field[{y}, {x}] = {value}")
        
        # 旧版
        if hasattr(old_env, 'map_field'):
            field_shape = old_env.map_field.shape
            if y < field_shape[0] and x < field_shape[1]:
                value = old_env.map_field[y, x]
                print(f"    旧版 map_field[{y}, {x}] = {value}")
    
    # 检查OpenCV操作的坐标系
    print("\n🔧 OpenCV操作分析:")
    print("  cv2.ellipse期望: center=(x, y) - 列在前，行在后")
    print("  cv2.line期望: pt1=(x, y), pt2=(x, y) - 列在前，行在后")
    print("  NumPy数组索引: array[y, x] - 行在前，列在后")
    print("  agent.position_discrete应该返回: (x, y) 格式")
    
    # 保存所有图像
    cv2.imwrite('/tmp/diag_new_original.png', cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite('/tmp/diag_old_original.png', cv2.cvtColor(old_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite('/tmp/diag_new_with_traj.png', cv2.cvtColor(new_img_with_traj, cv2.COLOR_RGB2BGR))
    cv2.imwrite('/tmp/diag_old_with_traj.png', cv2.cvtColor(old_img_with_traj, cv2.COLOR_RGB2BGR))
    
    print("\n📷 诊断图像已保存:")
    print("  - /tmp/diag_new_original.png")
    print("  - /tmp/diag_old_original.png")
    print("  - /tmp/diag_new_with_traj.png")
    print("  - /tmp/diag_old_with_traj.png")
    if new_img.shape[:2] == old_img.shape[:2][::-1]:
        print("  - /tmp/new_transposed.png")
    
    # 检查观察生成
    print("\n👁️ 观察生成影响分析:")
    print(f"  新版观察形状: {new_obs.shape}")
    print(f"  旧版观察形状: {old_obs.shape}")
    
    if new_obs.shape != old_obs.shape:
        print(f"  ⚠️ 观察形状不一致！这会影响训练！")
    else:
        # 比较观察内容
        obs_diff = np.mean(np.abs(new_obs - old_obs))
        print(f"  观察差异度: {obs_diff:.4f}")
        if obs_diff > 0.1:
            print(f"  ⚠️ 观察内容差异较大，可能影响训练效果！")
    
    new_env.close()
    old_env.close()
    
    print("\n" + "=" * 60)
    print("🎯 诊断结论:")
    print("  1. 检查新旧版本的坐标系统是否一致")
    print("  2. 验证渲染时的坐标转换是否正确")
    print("  3. 确认轨迹线渲染参数是否相同")
    print("  4. 评估对观察和训练的潜在影响")

if __name__ == "__main__":
    analyze_coordinate_systems()