"""详细调试障碍物生成问题"""
import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs_new.cpp_env_v2 import CppEnv
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt


def analyze_obstacle_generation():
    """分析障碍物生成的详细过程"""
    print("=" * 50)
    print("详细分析障碍物生成")
    print("=" * 50)
    
    # 创建环境
    env = CppEnv()
    
    # 测试多个种子
    for seed in [42, 100, 200]:
        print(f"\n测试种子 {seed}:")
        obs, info = env.reset(seed=seed)
        
        obstacle_map = env.maps_dict.get('obstacle')
        field_map = env.maps_dict.get('field')
        
        # 分析障碍物分布
        total_obstacles = np.sum(obstacle_map > 0)
        
        # 分离边界和内部障碍物
        # 边界障碍物通常在地图边缘形成一个框
        height, width = obstacle_map.shape
        
        # 创建一个边缘掩码（只保留边缘50像素）
        edge_mask = np.ones((height, width), dtype=bool)
        edge_mask[50:-50, 50:-50] = False
        
        edge_obstacles = np.sum(obstacle_map[edge_mask] > 0)
        inner_obstacles = np.sum(obstacle_map[~edge_mask] > 0)
        
        print(f"  总障碍物像素: {total_obstacles}")
        print(f"  边缘区域障碍物: {edge_obstacles}")
        print(f"  内部区域障碍物: {inner_obstacles}")
        print(f"  障碍物占比: {total_obstacles/(height*width)*100:.1f}%")
        
        # 检查是否有随机障碍物
        if inner_obstacles < 100:
            print("  ⚠️ 警告：内部几乎没有障碍物！")
        
        # 可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 显示field地图
        axes[0].imshow(field_map, cmap='gray')
        axes[0].set_title(f'Field Map (seed={seed})')
        axes[0].axis('off')
        
        # 显示obstacle地图
        axes[1].imshow(obstacle_map, cmap='gray')
        axes[1].set_title(f'Obstacle Map (total={total_obstacles})')
        axes[1].axis('off')
        
        # 显示合成图
        combined = np.zeros((height, width, 3), dtype=np.uint8)
        combined[:,:,0] = field_map * 100  # 红色通道：field
        combined[:,:,1] = obstacle_map * 255  # 绿色通道：obstacle
        axes[2].imshow(combined)
        axes[2].set_title('Combined (R=field, G=obstacle)')
        axes[2].axis('off')
        
        plt.savefig(f'debug_maps_seed_{seed}.png')
        print(f"  已保存可视化到 debug_maps_seed_{seed}.png")
    
    plt.close('all')
    env.close()
    
    print("\n" + "=" * 50)
    print("检查边界框配置")
    print("=" * 50)
    
    # 专门测试边界框
    env = CppEnv()
    
    print(f"use_box_boundary: {env.config.use_box_boundary}")
    print(f"boundary_expand_ratio: {env.config.boundary_expand_ratio}")
    print(f"boundary_min_expand_pixels: {env.config.boundary_min_expand_pixels}")
    
    # 测试禁用边界框
    original_use_box = env.config.use_box_boundary
    env.config.use_box_boundary = False
    
    obs, info = env.reset(seed=42)
    obstacle_map_no_boundary = env.maps_dict.get('obstacle')
    obstacles_no_boundary = np.sum(obstacle_map_no_boundary > 0)
    
    print(f"\n无边界框时的障碍物像素: {obstacles_no_boundary}")
    
    # 恢复边界框
    env.config.use_box_boundary = original_use_box
    obs, info = env.reset(seed=42)
    obstacle_map_with_boundary = env.maps_dict.get('obstacle')
    obstacles_with_boundary = np.sum(obstacle_map_with_boundary > 0)
    
    print(f"有边界框时的障碍物像素: {obstacles_with_boundary}")
    print(f"边界框贡献的像素: {obstacles_with_boundary - obstacles_no_boundary}")
    
    env.close()


def test_obstacle_creation_directly():
    """直接测试障碍物创建函数"""
    print("\n" + "=" * 50)
    print("直接测试障碍物创建函数")
    print("=" * 50)
    
    from envs_new.components.map.map_components import ObstacleCreator
    
    # 创建测试环境
    creator = ObstacleCreator()
    
    # 模拟state
    state = {
        'options': {},
        'config': type('Config', (), {
            'num_obstacles_range': (5, 8),
            'obstacle_size_range': (10, 25),
            'obstacle_min_distance_to_edge': 30,
            'obstacle_min_distance_to_agent': 2.0,
            'obstacle_expand_pixels': 2,
            'use_box_boundary': False,  # 禁用边界框
        })(),
        'maps_dict': {
            'field': np.ones((200, 200), dtype=np.uint8)
        },
        'agent': type('Agent', (), {
            'position': (100, 100),
            'length': 10
        })(),
        'env_state': type('EnvState', (), {
            'get_static_info': lambda x: (200, 200) if x == 'dimensions' else None
        })()
    }
    
    rng = np.random.default_rng(42)
    
    # 测试生成障碍物
    creator.generate(state, rng)
    
    obstacle_map = state['maps_dict'].get('obstacle')
    if obstacle_map is not None:
        obstacles_count = np.sum(obstacle_map > 0)
        print(f"生成的障碍物像素数: {obstacles_count}")
        
        # 显示并保存
        plt.figure(figsize=(6, 6))
        plt.imshow(obstacle_map, cmap='gray')
        plt.title(f'Direct Test Obstacles (pixels={obstacles_count})')
        plt.colorbar()
        plt.savefig('debug_direct_obstacles.png')
        print("已保存到 debug_direct_obstacles.png")
        plt.close()
        
        # 分析障碍物数量
        if obstacles_count < 100:
            print("⚠️ 障碍物太少！")
        elif obstacles_count > 10000:
            print("⚠️ 障碍物太多！")
        else:
            print("✅ 障碍物数量正常")


if __name__ == "__main__":
    analyze_obstacle_generation()
    test_obstacle_creation_directly()