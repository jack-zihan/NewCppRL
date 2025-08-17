#!/usr/bin/env python3
"""
深入诊断90度旋转问题
比较新旧版本的坐标系统、渲染参数等
"""

import numpy as np
import cv2
from envs_new.cpp_env_v2 import CppEnv as NewCppEnv
from envs.cpp_env_v2 import CppEnv as OldCppEnv

def diagnose_rotation():
    print("🔍 深入诊断90度旋转问题...")
    print("=" * 60)
    
    # 创建环境
    new_env = NewCppEnv(render_mode='rgb_array')
    old_env = OldCppEnv(render_mode='rgb_array')
    
    # 使用相同的种子重置
    new_obs, _ = new_env.reset(seed=42)
    old_obs, _ = old_env.reset(seed=42)
    
    print("📊 环境配置对比:")
    print(f"  新版地图尺寸: {new_env.maps_dict['field'].shape if 'field' in new_env.maps_dict else 'N/A'}")
    print(f"  旧版地图尺寸: {old_env.maps_dict['field'].shape if 'field' in old_env.maps_dict else 'N/A'}")
    
    print("\n📍 Agent初始位置对比:")
    print(f"  新版: position={new_env.agent.position}, discrete={new_env.agent.position_discrete}")
    print(f"  旧版: position={old_env.agent.position}, discrete={old_env.agent.position_discrete}")
    
    print("\n🎨 渲染器配置对比:")
    # 检查渲染器
    if hasattr(new_env, 'renderer') and hasattr(old_env, 'renderer'):
        new_renderer = new_env.renderer
        old_renderer = old_env.renderer
        
        # 检查渲染参数
        print("  新版渲染器:")
        if hasattr(new_renderer, 'map_size'):
            print(f"    map_size: {new_renderer.map_size}")
        if hasattr(new_renderer, 'render_size'):
            print(f"    render_size: {new_renderer.render_size}")
        if hasattr(new_renderer, 'scale'):
            print(f"    scale: {new_renderer.scale}")
            
        print("  旧版渲染器:")
        if hasattr(old_renderer, 'map_size'):
            print(f"    map_size: {old_renderer.map_size}")
        if hasattr(old_renderer, 'render_size'):
            print(f"    render_size: {old_renderer.render_size}")
        if hasattr(old_renderer, 'scale'):
            print(f"    scale: {old_renderer.scale}")
    
    # 渲染图像
    new_img = new_env.render()
    old_img = old_env.render()
    
    print(f"\n🖼️ 渲染图像分析:")
    print(f"  新版形状: {new_img.shape}")
    print(f"  旧版形状: {old_img.shape}")
    
    # 测试是否是转置关系
    if new_img.shape[:2] == old_img.shape[:2]:
        # 尝试转置
        new_img_transposed = np.transpose(new_img, (1, 0, 2))
        
        # 计算相似度
        if new_img_transposed.shape == old_img.shape:
            diff = np.mean(np.abs(new_img_transposed.astype(float) - old_img.astype(float)))
            similarity = 100 * (1 - diff / 255)
            print(f"\n🔄 转置测试:")
            print(f"  转置后形状匹配: ✓")
            print(f"  图像相似度: {similarity:.1f}%")
            
            if similarity > 90:
                print(f"  ⚠️ 高相似度表明新版图像是旧版的转置!")
    
    # 检查坐标系统
    print("\n🗺️ 坐标系统分析:")
    
    # 在地图上标记一个特定点测试
    test_x, test_y = 50, 100
    
    # 检查新版
    if 'field' in new_env.maps_dict:
        new_field = new_env.maps_dict['field'].copy()
        # 测试点标记
        if test_y < new_field.shape[0] and test_x < new_field.shape[1]:
            new_field[test_y, test_x] = 255  # NumPy索引 [y, x]
            print(f"  新版: 在NumPy坐标[{test_y}, {test_x}]标记点")
            
    # 检查旧版
    if 'field' in old_env.maps_dict:
        old_field = old_env.maps_dict['field'].copy()
        if test_y < old_field.shape[0] and test_x < old_field.shape[1]:
            old_field[test_y, test_x] = 255
            print(f"  旧版: 在NumPy坐标[{test_y}, {test_x}]标记点")
    
    # 检查cv2操作
    print("\n🔧 OpenCV操作检查:")
    print("  cv2.ellipse期望: center=(x, y) - OpenCV约定")
    print("  cv2.line期望: pt1=(x, y), pt2=(x, y) - OpenCV约定")
    print("  NumPy数组索引: array[y, x] - 行列索引")
    
    # 保存对比图像
    cv2.imwrite('/tmp/rotation_new.png', cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite('/tmp/rotation_old.png', cv2.cvtColor(old_img, cv2.COLOR_RGB2BGR))
    
    # 保存转置版本
    if new_img.shape[:2] == old_img.shape[:2]:
        new_img_transposed = np.transpose(new_img, (1, 0, 2))
        cv2.imwrite('/tmp/rotation_new_transposed.png', cv2.cvtColor(new_img_transposed, cv2.COLOR_RGB2BGR))
    
    print("\n📷 图像已保存:")
    print("  - /tmp/rotation_new.png (新版原始)")
    print("  - /tmp/rotation_old.png (旧版原始)")
    print("  - /tmp/rotation_new_transposed.png (新版转置)")
    
    new_env.close()
    old_env.close()
    
    print("\n" + "=" * 60)
    print("诊断完成!")

if __name__ == "__main__":
    diagnose_rotation()