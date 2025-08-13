#!/usr/bin/env python3
"""
场景构建器 - 生成测试场景

负责生成确定性的测试场景，包括地图、障碍物、起点终点等。
简化版本，去除过度复杂的场景管理。
"""

import numpy as np
from typing import Dict, List, Any, Tuple
import random


class ScenarioBuilder:
    """
    场景构建器
    
    生成路径规划测试所需的各种场景。
    保证相同seed生成相同场景（确定性）。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化场景构建器
        
        Args:
            config: 场景配置字典
        """
        self.config = config
        self.seeds = config.get('seeds', [42])
        self.difficulties = config.get('difficulties', ['easy'])
        self.map_sizes = config.get('map_sizes', [(100, 100)])
        
    def build_all(self) -> List[Dict[str, Any]]:
        """
        构建所有测试场景
        
        Returns:
            场景列表，每个场景是一个配置字典
        """
        scenarios = []
        
        for seed in self.seeds:
            for difficulty in self.difficulties:
                for map_size in self.map_sizes:
                    scenario = self.build_scenario(seed, difficulty, map_size)
                    scenarios.append(scenario)
        
        return scenarios
    
    def build_scenario(self, seed: int, difficulty: str, map_size: Tuple[int, int]) -> Dict[str, Any]:
        """
        构建单个场景
        
        Args:
            seed: 随机种子
            difficulty: 难度等级 (easy/medium/hard)
            map_size: 地图尺寸
            
        Returns:
            场景配置字典
        """
        # 设置随机种子，确保确定性
        np.random.seed(seed)
        random.seed(seed)
        
        # 根据难度设置参数
        params = self._get_difficulty_params(difficulty)
        
        # 生成场景元素
        scenario = {
            'id': f"s{seed}_{difficulty}_{map_size[0]}x{map_size[1]}",
            'seed': seed,
            'difficulty': difficulty,
            'map_size': map_size,
            'start_position': self._generate_start_position(map_size),
            'goal_position': self._generate_goal_position(map_size),
            'obstacles': self._generate_obstacles(map_size, params['num_obstacles']),
            'boundaries': self._generate_boundaries(map_size),
            'coverage_target': params['coverage_target']
        }
        
        return scenario
    
    def _get_difficulty_params(self, difficulty: str) -> Dict:
        """获取难度参数"""
        params_map = {
            'easy': {
                'num_obstacles': 3,
                'coverage_target': 0.8,
                'obstacle_size_range': (5, 10)
            },
            'medium': {
                'num_obstacles': 5,
                'coverage_target': 0.85,
                'obstacle_size_range': (10, 20)
            },
            'hard': {
                'num_obstacles': 8,
                'coverage_target': 0.9,
                'obstacle_size_range': (15, 30)
            }
        }
        return params_map.get(difficulty, params_map['easy'])
    
    def _generate_start_position(self, map_size: Tuple[int, int]) -> List[float]:
        """生成起始位置"""
        # 简单起见，从地图左下角区域开始
        x = np.random.uniform(0, map_size[0] * 0.2)
        y = np.random.uniform(0, map_size[1] * 0.2)
        return [y, x]  # 保持[y, x]格式
    
    def _generate_goal_position(self, map_size: Tuple[int, int]) -> List[float]:
        """生成目标位置"""
        # 目标在地图右上角区域
        x = np.random.uniform(map_size[0] * 0.8, map_size[0])
        y = np.random.uniform(map_size[1] * 0.8, map_size[1])
        return [y, x]  # 保持[y, x]格式
    
    def _generate_obstacles(self, map_size: Tuple[int, int], num_obstacles: int) -> List[Dict]:
        """
        生成障碍物
        
        Args:
            map_size: 地图尺寸
            num_obstacles: 障碍物数量
            
        Returns:
            障碍物列表，每个障碍物包含位置和尺寸
        """
        obstacles = []
        
        for i in range(num_obstacles):
            # 随机位置（避免边缘）
            x = np.random.uniform(map_size[0] * 0.1, map_size[0] * 0.9)
            y = np.random.uniform(map_size[1] * 0.1, map_size[1] * 0.9)
            
            # 随机尺寸
            width = np.random.uniform(5, 15)
            height = np.random.uniform(5, 15)
            
            obstacle = {
                'position': [y, x],  # [y, x]格式
                'size': [height, width],
                'type': 'rectangle'  # 可扩展为其他形状
            }
            obstacles.append(obstacle)
        
        return obstacles
    
    def _generate_boundaries(self, map_size: Tuple[int, int]) -> List[List[float]]:
        """
        生成地图边界
        
        Args:
            map_size: 地图尺寸
            
        Returns:
            边界点列表，形成闭合多边形
        """
        # 简单矩形边界
        boundaries = [
            [0, 0],
            [0, map_size[0]],
            [map_size[1], map_size[0]],
            [map_size[1], 0],
            [0, 0]  # 闭合
        ]
        return boundaries
    
    def generate_farm_vertices(self, map_size: Tuple[int, int], shape: str = 'rectangle') -> List[List[float]]:
        """
        生成农场顶点（用于更复杂的场景）
        
        Args:
            map_size: 地图尺寸
            shape: 形状类型
            
        Returns:
            顶点列表
        """
        if shape == 'rectangle':
            return self._generate_boundaries(map_size)
        elif shape == 'polygon':
            # 生成不规则多边形
            num_vertices = np.random.randint(5, 8)
            angles = np.sort(np.random.uniform(0, 2*np.pi, num_vertices))
            radius_variation = np.random.uniform(0.7, 1.0, num_vertices)
            
            center_x = map_size[0] / 2
            center_y = map_size[1] / 2
            base_radius = min(map_size) * 0.4
            
            vertices = []
            for angle, r_var in zip(angles, radius_variation):
                x = center_x + base_radius * r_var * np.cos(angle)
                y = center_y + base_radius * r_var * np.sin(angle)
                vertices.append([y, x])
            
            vertices.append(vertices[0])  # 闭合
            return vertices
        else:
            return self._generate_boundaries(map_size)
    
    def add_dynamic_obstacles(self, scenario: Dict, num_dynamic: int = 2) -> Dict:
        """
        添加动态障碍物（可选功能）
        
        Args:
            scenario: 基础场景
            num_dynamic: 动态障碍物数量
            
        Returns:
            包含动态障碍物的场景
        """
        dynamic_obstacles = []
        map_size = scenario['map_size']
        
        for i in range(num_dynamic):
            obstacle = {
                'initial_position': [
                    np.random.uniform(0, map_size[1]),
                    np.random.uniform(0, map_size[0])
                ],
                'velocity': [
                    np.random.uniform(-1, 1),
                    np.random.uniform(-1, 1)
                ],
                'size': [5, 5],
                'type': 'dynamic'
            }
            dynamic_obstacles.append(obstacle)
        
        scenario['dynamic_obstacles'] = dynamic_obstacles
        return scenario