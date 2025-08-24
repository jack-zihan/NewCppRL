"""测试障碍物生成是否正常工作"""
import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs_new.cpp_env_v2 import CppEnv


def test_obstacle_generation():
    """测试障碍物生成功能"""
    print("=" * 50)
    print("测试障碍物生成（修复后）")
    print("=" * 50)
    
    env = CppEnv()
    
    # 测试多个种子
    for seed in [42, 100, 200]:
        print(f"\n种子 {seed}:")
        obs, info = env.reset(seed=seed)
        
        # 检查障碍物地图
        obstacle_map = env.maps_dict.get('obstacle')
        field_map = env.maps_dict.get('field')
        
        if obstacle_map is None:
            print("  ❌ 错误：obstacle_map 为 None！")
            continue
            
        # 分析障碍物
        height, width = obstacle_map.shape
        total_pixels = height * width
        obstacle_pixels = np.sum(obstacle_map > 0)
        
        # 分离边界和内部障碍物
        # 创建内部区域掩码（去掉边缘100像素）
        inner_mask = np.zeros((height, width), dtype=bool)
        inner_mask[100:-100, 100:-100] = True
        
        inner_obstacles = np.sum(obstacle_map[inner_mask] > 0)
        boundary_obstacles = obstacle_pixels - inner_obstacles
        
        print(f"  地图尺寸: {width}x{height}")
        print(f"  总障碍物像素: {obstacle_pixels} ({obstacle_pixels/total_pixels*100:.1f}%)")
        print(f"  边界障碍物: {boundary_obstacles}")
        print(f"  内部障碍物: {inner_obstacles}")
        
        # 检查随机障碍物数量
        if inner_obstacles < 50:
            print("  ⚠️ 警告：内部障碍物太少！")
        elif inner_obstacles > 5000:
            print("  ✅ 内部有足够的障碍物")
        else:
            print("  ✅ 内部障碍物数量正常")
        
        # 检查配置
        if hasattr(env, 'config'):
            print(f"  配置: num_obstacles_range={env.config.num_obstacles_range}, "
                  f"obstacle_size_range={env.config.obstacle_size_range}")
    
    env.close()
    
    print("\n" + "=" * 50)
    print("测试边界框对障碍物的影响")
    print("=" * 50)
    
    # 测试关闭边界框
    env = CppEnv()
    
    # 保存原始设置
    original_use_box = env.config.use_box_boundary
    
    # 测试无边界框
    env.config.use_box_boundary = False
    obs, info = env.reset(seed=42)
    no_boundary_obstacles = np.sum(env.maps_dict['obstacle'] > 0)
    
    # 测试有边界框
    env.config.use_box_boundary = True
    obs, info = env.reset(seed=42)
    with_boundary_obstacles = np.sum(env.maps_dict['obstacle'] > 0)
    
    print(f"无边界框时障碍物像素: {no_boundary_obstacles}")
    print(f"有边界框时障碍物像素: {with_boundary_obstacles}")
    print(f"边界框增加的像素: {with_boundary_obstacles - no_boundary_obstacles}")
    
    if no_boundary_obstacles == 0:
        print("❌ 错误：没有边界框时障碍物为0，随机障碍物生成失败！")
    elif no_boundary_obstacles < 100:
        print("⚠️ 警告：随机障碍物数量偏少")
    else:
        print("✅ 随机障碍物生成正常")
    
    env.close()


if __name__ == "__main__":
    test_obstacle_generation()