#!/usr/bin/env python3
"""
指标计算器 - 评估算法性能

计算路径规划算法的各种性能指标：
- 覆盖率
- 路径长度
- 碰撞检测
- 效率指标
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union


class MetricsCalculator:
    """
    指标计算器
    
    负责计算路径规划算法的性能指标。
    设计原则：方法职责单一，计算逻辑清晰。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化指标计算器
        
        Args:
            config: 指标配置
        """
        self.config = config
        self.coverage_thresholds = config.get('coverage_thresholds', [0.90, 0.95, 0.98])
        self.collision_threshold = config.get('collision_threshold', 2.0)
        
    def calculate(self, trajectory: List[Tuple[float, float]], 
                  env: Any, scenario: Dict) -> Dict[str, Any]:
        """
        计算所有指标
        
        Args:
            trajectory: 轨迹点列表 [(y, x), ...]
            env: 环境实例
            scenario: 场景配置
            
        Returns:
            指标字典
        """
        metrics = {}
        
        # 基础指标
        metrics['path_length'] = self.calculate_path_length(trajectory)
        metrics['num_steps'] = len(trajectory)
        
        # 覆盖率相关
        coverage_info = self.calculate_coverage(trajectory, scenario)
        metrics.update(coverage_info)
        
        # 碰撞检测
        collision_info = self.check_collisions(trajectory, scenario)
        metrics.update(collision_info)
        
        # 效率指标
        efficiency_info = self.calculate_efficiency(metrics)
        metrics.update(efficiency_info)
        
        # 平滑度
        metrics['smoothness'] = self.calculate_smoothness(trajectory)
        
        return metrics
    
    def calculate_path_length(self, trajectory: List[Tuple[float, float]]) -> float:
        """
        计算路径总长度
        
        Args:
            trajectory: 轨迹点列表
            
        Returns:
            路径总长度
        """
        if len(trajectory) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(trajectory)):
            p1 = np.array(trajectory[i-1])
            p2 = np.array(trajectory[i])
            total_length += np.linalg.norm(p2 - p1)
        
        return total_length
    
    def calculate_coverage(self, trajectory: List[Tuple[float, float]], 
                           scenario: Dict) -> Dict[str, Any]:
        """
        计算覆盖率相关指标
        
        Args:
            trajectory: 轨迹点列表
            scenario: 场景配置
            
        Returns:
            覆盖率指标字典
        """
        map_size = scenario.get('map_size', (100, 100))
        boundaries = scenario.get('boundaries', [])
        
        # 创建地图多边形
        if boundaries:
            map_polygon = Polygon(boundaries)
        else:
            map_polygon = Polygon([
                (0, 0), (0, map_size[0]), 
                (map_size[1], map_size[0]), (map_size[1], 0)
            ])
        
        # 计算覆盖区域（使用缓冲区模拟机器人覆盖范围）
        coverage_radius = self.config.get('coverage_radius', 2.0)
        covered_areas = []
        
        for point in trajectory:
            # 创建覆盖圆
            coverage_circle = Point(point).buffer(coverage_radius)
            covered_areas.append(coverage_circle)
        
        # 合并所有覆盖区域
        if covered_areas:
            total_covered = unary_union(covered_areas)
            covered_area = total_covered.intersection(map_polygon).area
            total_area = map_polygon.area
            coverage_rate = covered_area / total_area if total_area > 0 else 0
        else:
            coverage_rate = 0
            covered_area = 0
        
        # 计算达到不同覆盖率阈值时的路径长度
        threshold_lengths = self._calculate_threshold_lengths(
            trajectory, covered_areas, map_polygon, self.coverage_thresholds
        )
        
        return {
            'coverage_rate': coverage_rate,
            'covered_area': covered_area,
            'threshold_lengths': threshold_lengths,
            'coverage_90_length': threshold_lengths.get(0.90, np.inf),
            'coverage_95_length': threshold_lengths.get(0.95, np.inf),
            'coverage_98_length': threshold_lengths.get(0.98, np.inf)
        }
    
    def _calculate_threshold_lengths(self, trajectory: List[Tuple[float, float]],
                                    covered_areas: List, map_polygon: Polygon,
                                    thresholds: List[float]) -> Dict[float, float]:
        """
        计算达到不同覆盖率阈值时的路径长度
        
        Args:
            trajectory: 轨迹点列表
            covered_areas: 覆盖区域列表
            map_polygon: 地图多边形
            thresholds: 覆盖率阈值列表
            
        Returns:
            阈值到路径长度的映射
        """
        threshold_lengths = {}
        total_area = map_polygon.area
        
        if total_area == 0:
            return {t: np.inf for t in thresholds}
        
        # 逐步累积覆盖区域
        accumulated_coverage = None
        path_length = 0.0
        
        for i, coverage_circle in enumerate(covered_areas):
            if i > 0:
                # 计算当前步的路径长度
                p1 = np.array(trajectory[i-1])
                p2 = np.array(trajectory[i])
                path_length += np.linalg.norm(p2 - p1)
            
            # 更新累积覆盖区域
            if accumulated_coverage is None:
                accumulated_coverage = coverage_circle
            else:
                accumulated_coverage = unary_union([accumulated_coverage, coverage_circle])
            
            # 计算当前覆盖率
            current_covered = accumulated_coverage.intersection(map_polygon).area
            current_rate = current_covered / total_area
            
            # 检查是否达到阈值
            for threshold in thresholds:
                if threshold not in threshold_lengths and current_rate >= threshold:
                    threshold_lengths[threshold] = path_length
        
        # 未达到的阈值设为无穷大
        for threshold in thresholds:
            if threshold not in threshold_lengths:
                threshold_lengths[threshold] = np.inf
        
        return threshold_lengths
    
    def check_collisions(self, trajectory: List[Tuple[float, float]], 
                        scenario: Dict) -> Dict[str, Any]:
        """
        检查碰撞
        
        Args:
            trajectory: 轨迹点列表
            scenario: 场景配置
            
        Returns:
            碰撞信息字典
        """
        obstacles = scenario.get('obstacles', [])
        
        collision_count = 0
        collision_points = []
        min_obstacle_distance = np.inf
        
        for point in trajectory:
            point_obj = Point(point)
            
            for obstacle in obstacles:
                obs_pos = obstacle['position']
                obs_size = obstacle['size']
                
                # 创建障碍物矩形
                obs_rect = Polygon([
                    (obs_pos[0] - obs_size[0]/2, obs_pos[1] - obs_size[1]/2),
                    (obs_pos[0] + obs_size[0]/2, obs_pos[1] - obs_size[1]/2),
                    (obs_pos[0] + obs_size[0]/2, obs_pos[1] + obs_size[1]/2),
                    (obs_pos[0] - obs_size[0]/2, obs_pos[1] + obs_size[1]/2)
                ])
                
                # 检查碰撞
                distance = point_obj.distance(obs_rect)
                min_obstacle_distance = min(min_obstacle_distance, distance)
                
                if distance < self.collision_threshold:
                    collision_count += 1
                    collision_points.append(point)
                    break  # 一个点只计一次碰撞
        
        return {
            'has_collision': collision_count > 0,
            'collision_count': collision_count,
            'collision_rate': collision_count / len(trajectory) if trajectory else 0,
            'min_obstacle_distance': min_obstacle_distance,
            'collision_points': collision_points
        }
    
    def calculate_efficiency(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算效率指标
        
        Args:
            metrics: 已计算的指标
            
        Returns:
            效率指标字典
        """
        efficiency = {}
        
        # 覆盖效率：覆盖率 / 路径长度
        coverage_rate = metrics.get('coverage_rate', 0)
        path_length = metrics.get('path_length', 1)
        efficiency['coverage_efficiency'] = coverage_rate / path_length if path_length > 0 else 0
        
        # 步数效率：覆盖率 / 步数
        num_steps = metrics.get('num_steps', 1)
        efficiency['step_efficiency'] = coverage_rate / num_steps if num_steps > 0 else 0
        
        # 综合效率评分（可自定义权重）
        weights = self.config.get('efficiency_weights', {
            'coverage': 0.4,
            'path_length': 0.3,
            'collision': 0.3
        })
        
        # 归一化各项指标
        coverage_score = coverage_rate  # 已经是0-1
        length_score = 1.0 / (1.0 + path_length / 1000.0)  # 路径越短越好
        collision_score = 1.0 - metrics.get('collision_rate', 0)  # 碰撞越少越好
        
        efficiency['overall_score'] = (
            weights['coverage'] * coverage_score +
            weights['path_length'] * length_score +
            weights['collision'] * collision_score
        )
        
        return efficiency
    
    def calculate_smoothness(self, trajectory: List[Tuple[float, float]]) -> float:
        """
        计算路径平滑度
        
        通过计算转向角度的标准差来评估路径平滑度。
        值越小表示路径越平滑。
        
        Args:
            trajectory: 轨迹点列表
            
        Returns:
            平滑度指标
        """
        if len(trajectory) < 3:
            return 0.0
        
        angles = []
        for i in range(1, len(trajectory) - 1):
            p1 = np.array(trajectory[i-1])
            p2 = np.array(trajectory[i])
            p3 = np.array(trajectory[i+1])
            
            v1 = p2 - p1
            v2 = p3 - p2
            
            # 计算转向角度
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                angles.append(angle)
        
        if angles:
            # 返回角度变化的标准差（越小越平滑）
            return float(np.std(angles))
        else:
            return 0.0
    
    def calculate_completion_time(self, trajectory: List[Tuple[float, float]], 
                                 velocity: float = 1.0) -> float:
        """
        估算完成时间
        
        Args:
            trajectory: 轨迹点列表
            velocity: 机器人速度（单位/秒）
            
        Returns:
            估算的完成时间（秒）
        """
        path_length = self.calculate_path_length(trajectory)
        return path_length / velocity if velocity > 0 else np.inf