#!/usr/bin/env python3
"""
深入分析坐标系统差异
"""
import sys
import math
import numpy as np
from pathlib import Path
import yaml
from omegaconf import DictConfig
import gymnasium as gym

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'rules'))
sys.path.insert(0, str(project_root / 'rules_new'))

def test_coordinate_systems():
    """测试两个系统的坐标系统"""
    print("="*80)
    print("坐标系统分析")
    print("="*80)
    
    # 创建环境
    env_config_path = project_root / 'configs' / 'env_config.yaml'
    cfg = DictConfig(yaml.load(open(env_config_path), Loader=yaml.FullLoader))
    
    # 使用老环境
    sys.path.insert(0, str(project_root / 'envs'))
    from cpp_env_v2 import CppEnv
    
    # 移除gym特定参数
    env_params = dict(cfg.env.params)
    env_params.pop('id', None)
    env_params.pop('entry_point', None)
    env_params.pop('max_episode_steps', None)
    
    env = CppEnv(**env_params)
    obs, info = env.reset(seed=42)
    
    # 获取环境参数
    farm_vertices = env.min_area_rect[0][:, 0, ::-1]  # [y, x] -> [x, y]
    agent_position = [float(env.agent.x), float(env.agent.y)]
    
    print(f"\n环境信息:")
    print(f"  Agent position: {agent_position}")
    print(f"  Farm vertices shape: {farm_vertices.shape}")
    print(f"  Farm vertices: \n{farm_vertices}")
    
    # 测试rules_new的JUMP算法初始化
    print("\n" + "="*40)
    print("Rules_new JUMP算法初始化:")
    
    from rules.jump_path import (
        find_longest_edge, H, W, farm_vertices as farm_vertices_old,
        sight_width, agent_width
    )
    from matplotlib.path import Path
    from shapely.geometry import LineString
    
    # 使用rules_new的逻辑
    max_edge_points = find_longest_edge(farm_vertices_old)
    dx = max_edge_points[1][0] - max_edge_points[0][0]
    dy = max_edge_points[1][1] - max_edge_points[0][1]
    real_radians = np.arctan2(dy, dx)
    real_radians = real_radians % (2 * np.pi) if real_radians >= 0 else (real_radians + 2 * np.pi) % (2 * np.pi)
    
    min_x, min_y = farm_vertices_old.min(axis=0)
    max_x, max_y = farm_vertices_old.max(axis=0)
    diag_length = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)
    
    # 创建mask
    poly_path = Path(farm_vertices_old)
    y, x = np.mgrid[:H, :W]
    coor = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
    mask = np.zeros((H, W))
    mask[poly_path.contains_points(coor).reshape(H, W)] = 1
    
    # 生成第一条路径线
    y_offset = -diag_length + agent_width / 2
    turn = False
    
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
    
    if valid_points:
        first_point = valid_points[0]
        print(f"  First valid point (raw): {first_point}")
        print(f"  Rules_new would set:")
        print(f"    env.agent.x = {first_point[1]}")
        print(f"    env.agent.y = {first_point[0]}")
        print(f"  Returned waypoint: [{first_point[1]}, {first_point[0]}]")
    
    # 测试rules_new1的JUMP算法初始化
    print("\n" + "="*40)
    print("Rules_new1 JUMP算法初始化:")
    
    from rules_new.algorithms.jump_planner import JumpPlanner
    from rules_new.utils.config_manager import ConfigManager
    
    config_manager = ConfigManager()
    algo_config = config_manager.get_algorithm_config('JUMP')
    env_config = config_manager.get_environment_config()
    
    planner = JumpPlanner(algo_config, env_config)
    
    # 准备初始状态
    turning_radius = env.v_range.max / (abs(env.w_range.max) * (math.pi / 180))
    
    initial_state = {
        'agent_position': agent_position,
        'agent_direction': float(env.agent.direction),
        'discovered_weeds': [],
        'farm_vertices': farm_vertices,
        'turning_radius': turning_radius
    }
    
    planner.reset(initial_state)
    
    # 生成第一条路径线
    path_points = planner._generate_path_line()
    
    if path_points:
        first_point = path_points[0]
        print(f"  First path point: {first_point}")
        print(f"  Returned as waypoint: {first_point}")
    
    env.close()
    
    print("\n" + "="*80)
    print("分析结论:")
    print("  - rules_new使用[x,y]格式生成点，但返回时交换为[y,x]")
    print("  - rules_new1直接使用[x,y]格式")
    print("  - 这导致了y坐标的巨大差异")

if __name__ == "__main__":
    test_coordinate_systems()