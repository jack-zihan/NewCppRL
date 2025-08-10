#!/usr/bin/env python3
"""
简单测试第一个waypoint的生成
"""
import sys
import math
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'rules'))
sys.path.insert(0, str(project_root / 'rules_new'))

# 设置固定的测试参数
farm_vertices = np.array([[195.75228941, 145.24337962],
                          [411.45569833, 215.58821163],
                          [363.13018695, 390.85177843],
                          [147.42677802, 320.50694642]])

def test_rules_new():
    """测试rules_new的waypoint生成"""
    print("="*60)
    print("Rules_new JUMP算法测试")
    print("="*60)
    
    from rules.jump_path import find_longest_edge, H, W, sight_width, agent_width
    from matplotlib.path import Path
    from shapely.geometry import LineString
    
    # 计算参数
    max_edge_points = find_longest_edge(farm_vertices)
    dx = max_edge_points[1][0] - max_edge_points[0][0]
    dy = max_edge_points[1][1] - max_edge_points[0][1]
    real_radians = np.arctan2(dy, dx)
    real_radians = real_radians % (2 * np.pi) if real_radians >= 0 else (real_radians + 2 * np.pi) % (2 * np.pi)
    
    min_x, min_y = farm_vertices.min(axis=0)
    max_x, max_y = farm_vertices.max(axis=0)
    diag_length = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)
    
    print(f"Real radians: {real_radians}")
    print(f"Diagonal length: {diag_length}")
    
    # 创建mask
    poly_path = Path(farm_vertices)
    y, x = np.mgrid[:H, :W]
    coor = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
    mask = np.zeros((H, W))
    mask[poly_path.contains_points(coor).reshape(H, W)] = 1
    
    # 初始参数
    y_offset = -diag_length + agent_width / 2
    turn = False
    
    # 生成路径线
    start = [0, 0]
    end = np.array([100 * np.cos(real_radians), 100 * np.sin(real_radians)])
    
    new_start = [start[0] + y_offset * np.cos(real_radians + np.pi / 2) - diag_length * np.cos(real_radians),
                 start[1] + y_offset * np.sin(real_radians + np.pi / 2) - diag_length * np.sin(real_radians)]
    new_end = [end[0] + y_offset * np.cos(real_radians + np.pi / 2) + diag_length * np.cos(real_radians),
               end[1] + y_offset * np.sin(real_radians + np.pi / 2) + diag_length * np.sin(real_radians)]
    
    line = LineString([new_start, new_end])
    
    x_points = []
    for i in np.arange(0, line.length, 1):
        interpolated_point = np.array(line.interpolate(i).coords[0])
        x_points.append(interpolated_point)
    
    valid_points = [point for point in x_points if
                    0 <= int(point[1]) < H and 0 <= int(point[0]) < W and mask[int(point[1]), int(point[0])] == 1]
    
    if turn:
        valid_points = valid_points[::-1]
    
    if valid_points:
        first_point = valid_points[0]
        print(f"\nFirst waypoint (raw): {first_point}")
        print(f"Returned as: [{first_point[1]}, {first_point[0]}]")
        return [first_point[1], first_point[0]]
    
    return None

def test_rules_new1():
    """测试rules_new1的waypoint生成"""
    print("\n" + "="*60)
    print("Rules_new1 JUMP算法测试")
    print("="*60)
    
    from rules_new.algorithms.jump_planner import JumpPlanner
    from rules_new.utils.config_manager import ConfigManager
    
    config_manager = ConfigManager()
    algo_config = config_manager.get_algorithm_config('JUMP')
    env_config = config_manager.get_environment_config()
    
    planner = JumpPlanner(algo_config, env_config)
    
    # 准备初始状态
    initial_state = {
        'agent_position': [300, 300],
        'agent_direction': 0,
        'discovered_weeds': [],
        'farm_vertices': farm_vertices,
        'turning_radius': 7.011721269083501  # 从实际环境计算得出
    }
    
    planner.reset(initial_state)
    
    print(f"Real radians: {planner.real_radians}")
    print(f"Diagonal length: {planner.diagonal_length}")
    print(f"Turn direction: {planner.turn_direction}")
    print(f"Y offset: {planner.y_offset}")
    
    # 获取第一个waypoint
    waypoint = planner.plan_next_waypoint(initial_state)
    
    if waypoint:
        print(f"\nFirst waypoint: {waypoint}")
        return waypoint
    
    return None

if __name__ == "__main__":
    waypoint_old = test_rules_new()
    waypoint_new = test_rules_new1()
    
    print("\n" + "="*60)
    print("对比结果")
    print("="*60)
    print(f"Rules_new:  {waypoint_old}")
    print(f"Rules_new1: {waypoint_new}")
    
    if waypoint_old and waypoint_new:
        diff = np.linalg.norm(np.array(waypoint_old) - np.array(waypoint_new))
        print(f"距离差异: {diff:.2f}")