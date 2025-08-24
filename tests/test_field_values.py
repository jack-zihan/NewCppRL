#!/usr/bin/env python3
"""
Check field map values to understand coverage logic
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_field_values():
    """检查field地图的值来理解覆盖逻辑"""
    print("\n=== 检查Field地图值 ===")
    
    from envs_new.cpp_env_v4 import CppEnv as CppEnvV4
    env = CppEnvV4()
    obs, _ = env.reset(seed=42)
    
    field_map = env.maps_dict['field']
    
    print(f"Field地图形状: {field_map.shape}")
    print(f"Field地图数据类型: {field_map.dtype}")
    print(f"Field地图唯一值: {np.unique(field_map)}")
    print(f"Field地图最小值: {field_map.min()}")
    print(f"Field地图最大值: {field_map.max()}")
    print(f"Field地图总和: {field_map.sum()}")
    
    # 统计每个值的数量
    unique, counts = np.unique(field_map, return_counts=True)
    print("\n值分布:")
    for val, count in zip(unique, counts):
        print(f"  值 {val}: {count} 像素 ({count/field_map.size*100:.2f}%)")
    
    # 检查agent初始位置的field值
    agent_pos = env.agent.position_discrete
    print(f"\nAgent初始位置: {agent_pos}")
    print(f"Agent位置的field值: {field_map[agent_pos[1], agent_pos[0]]}")
    
    # 检查agent周围的field值
    x, y = agent_pos
    window_size = 5
    x_start = max(0, x - window_size)
    x_end = min(field_map.shape[1], x + window_size + 1)
    y_start = max(0, y - window_size)
    y_end = min(field_map.shape[0], y + window_size + 1)
    
    print(f"\nAgent周围{window_size}x{window_size}区域的field值:")
    window = field_map[y_start:y_end, x_start:x_end]
    print(window)
    
    # 测试覆盖逻辑
    print("\n=== 测试覆盖逻辑 ===")
    
    # 获取原始凸包区域的值
    convex_hull = env.agent.convex_hull.round().astype(np.int32)
    
    # 创建一个mask来查看凸包覆盖的区域
    import cv2
    mask = np.zeros_like(field_map)
    cv2.fillPoly(mask, [convex_hull], color=(1,))
    
    covered_area = mask.sum()
    covered_field_before = (field_map * mask).sum()
    
    print(f"凸包覆盖区域: {covered_area} 像素")
    print(f"覆盖区域内的field值总和（覆盖前）: {covered_field_before}")
    
    # 执行一步看看field是否变化
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    new_field_map = env.maps_dict['field']
    print(f"\n执行一步后:")
    print(f"Field地图总和: {new_field_map.sum()}")
    print(f"Field覆盖率: {info.get('field_ratio', 'N/A')}")
    
    # 检查是否有任何变化
    diff = field_map - new_field_map
    changed_pixels = np.sum(diff != 0)
    print(f"改变的像素数: {changed_pixels}")
    
    if changed_pixels > 0:
        print("改变的像素值:")
        unique_changes = np.unique(diff[diff != 0])
        for val in unique_changes:
            count = np.sum(diff == val)
            print(f"  变化 {val}: {count} 像素")
    
    env.close()


def test_initial_field_state():
    """检查初始field状态"""
    print("\n=== 检查初始Field状态 ===")
    
    from envs_new.cpp_env_v4 import CppEnv as CppEnvV4
    env = CppEnvV4()
    
    # 重置几次看看初始状态
    for i in range(3):
        obs, _ = env.reset(seed=42+i)
        field_sum = env.maps_dict['field'].sum()
        total_field_area = env.env_state.total_field_area if hasattr(env.env_state, 'total_field_area') else 'N/A'
        field_area = env.env_state.field_area if hasattr(env.env_state, 'field_area') else 'N/A'
        field_ratio = env.env_state.field_coverage_ratio if hasattr(env.env_state, 'field_coverage_ratio') else 'N/A'
        
        print(f"\n种子 {42+i}:")
        print(f"  Field地图总和: {field_sum}")
        print(f"  total_field_area: {total_field_area}")
        print(f"  field_area: {field_area}")
        print(f"  field_coverage_ratio: {field_ratio}")
    
    env.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Field地图值分析")
    print("=" * 60)
    
    test_field_values()
    test_initial_field_state()