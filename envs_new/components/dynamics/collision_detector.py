"""
割草机器人环境的碰撞检测系统。

核心算法：
1. 双重碰撞检测：边界碰撞 + 障碍物碰撞
2. 凸包几何检测：使用OpenCV多边形填充算法生成机器人占用图
3. 距离变换优化：从O(n²)暴力搜索优化到O(n)欧氏距离变换
4. 渐进式安全退避：多级安全余量递减策略
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import Dict, Tuple

from envs_new.components.entity.agent import Agent


class CollisionDetector:
    """
    碰撞检测器 - 处理机器人与环境的碰撞检测。
    
    坐标系统约定：
    - 地图坐标：numpy数组使用(y, x)顺序，y为行索引，x为列索引
    - 机器人坐标：位置和凸包使用(x, y)顺序，符合笛卡尔坐标系
    - 障碍物地图：二值数组，1=障碍物，0=可通行区域
    """
    
    def __init__(self):
        pass
    
    def check_collision(self, agent: Agent, maps_dict: Dict[str, np.ndarray]) -> bool:
        """主碰撞检测接口 - 检查机器人是否与任何障碍物或边界发生碰撞。"""
        if 'obstacle' not in maps_dict:
            raise ValueError("Obstacle map required for collision detection")
        
        obstacle_map = maps_dict['obstacle']
        dimensions = obstacle_map.shape  # (height, width)
        
        # 双重碰撞检测策略：分别检测边界和障碍物碰撞
        boundary_collision = self._check_boundary_collision(agent, dimensions)
        obstacle_collision = self._check_obstacle_collision(agent, obstacle_map)
        
        return boundary_collision or obstacle_collision
    
    def _check_boundary_collision(self, agent: Agent, dimensions: Tuple[int, int]) -> bool:
        """
        边界碰撞检测 - 检查机器人凸包是否超出地图边界。
        
        算法：检查凸包所有顶点是否在合法范围内 [0, width) x [0, height)
        """
        height, width = dimensions
        convex_hull = agent.convex_hull
        
        x_coords = convex_hull[:, 0]
        y_coords = convex_hull[:, 1]
        
        x_in_bounds = np.all((x_coords >= 0) & (x_coords < width))
        y_in_bounds = np.all((y_coords >= 0) & (y_coords < height))
        
        return not (x_in_bounds and y_in_bounds)
    
    def _check_obstacle_collision(self, agent: Agent, obstacle_map: np.ndarray) -> bool:
        """
        障碍物碰撞检测 - 使用像素级重叠检测。
        
        算法：生成机器人占用图，与障碍物图进行逻辑与操作
        """
        agent_map = self._create_agent_occupancy_map(agent, obstacle_map.shape)
        collision = np.any(np.logical_and(agent_map, obstacle_map))
        return collision
    
    def _create_agent_occupancy_map(self, agent: Agent, map_shape: Tuple[int, int]) -> np.ndarray:
        """
        创建机器人占用图 - 将机器人凸包转换为二值占用图。
        
        算法：使用OpenCV的fillPoly栅格化填充算法，高效将凸包多边形
        转换为像素级占用表示。
        """
        agent_map = np.zeros(map_shape, dtype=np.uint8)
        
        convex_hull = agent.convex_hull.round().astype(np.int32)
        cv2.fillPoly(agent_map, [convex_hull], color=1)
        
        return agent_map
    
    def _estimate_agent_radius(self, agent: Agent) -> float:
        """
        估算机器人有效碰撞半径 - 用于安全边界计算。
        
        算法：使用凸包边界框的最大维度的一半作为基础半径，
        并添加1像素的安全余量以应对栅格化误差。
        """
        convex_hull = agent.convex_hull
        
        # Calculate bounding box dimensions
        x_min, x_max = convex_hull[:, 0].min(), convex_hull[:, 0].max()
        y_min, y_max = convex_hull[:, 1].min(), convex_hull[:, 1].max()
        
        # Use half of max dimension as radius, plus small safety margin
        max_dimension = max(x_max - x_min, y_max - y_min)
        safety_margin = 1.0  # Add 1 pixel safety margin
        
        return (max_dimension / 2.0) + safety_margin
    
    def get_collision_details(self, agent: Agent, maps_dict: Dict[str, np.ndarray]) -> Dict[str, bool]:
        if 'obstacle' not in maps_dict:
            raise ValueError("Obstacle map required for collision detection")
        
        obstacle_map = maps_dict['obstacle']
        dimensions = obstacle_map.shape
        
        boundary_collision = self._check_boundary_collision(agent, dimensions)
        obstacle_collision = self._check_obstacle_collision(agent, obstacle_map)
        
        return {
            'any_collision': boundary_collision or obstacle_collision,
            'boundary_collision': boundary_collision,
            'obstacle_collision': obstacle_collision
        }
    
    def get_safe_position(self, agent: Agent, maps_dict: Dict[str, np.ndarray]) -> Tuple[float, float]:
        """
        查找最近安全位置 - 使用高效的距离变换算法。
        
        算法优化：
        - 传统方法：O(n²)暴力遍历所有可能位置
        - 优化方法：O(n)欧氏距离变换（EDT），一次计算得到所有点到障碍物的距离
        
        退避策略：
        1. 优先保持完整安全半径
        2. 逐级降低安全余量（80%→60%→40%→20%）
        3. 最后选择任意可通行点
        4. 极端情况返回地图中心
        """
        if 'obstacle' not in maps_dict:
            raise ValueError("Obstacle map required")
        
        obstacle_map = maps_dict['obstacle']
        height, width = obstacle_map.shape
        current_x, current_y = agent.position
        
        # Early return if agent is not in collision
        if not self.check_collision(agent, maps_dict):
            return current_x, current_y
        
        # 欧氏距离变换：计算每个自由空间像素到最近障碍物的距离
        free_space_map = (obstacle_map == 0).astype(float)
        distance_map = distance_transform_edt(free_space_map)
        
        # 应用机器人安全半径，确保机器人整体不会碰撞
        agent_radius = self._estimate_agent_radius(agent)
        safe_positions_mask = distance_map >= agent_radius
        
        # Progressive fallback strategy if no safe positions found
        if not np.any(safe_positions_mask):
            # Try with reduced safety margins: 80%, 60%, 40%, 20%
            for margin_factor in [0.8, 0.6, 0.4, 0.2]:
                reduced_radius = agent_radius * margin_factor
                safe_positions_mask = distance_map >= reduced_radius
                if np.any(safe_positions_mask):
                    break
            
            # Last resort: use any free space
            if not np.any(safe_positions_mask):
                safe_positions_mask = free_space_map > 0
                
            # Final fallback: center of map
            if not np.any(safe_positions_mask):
                return width / 2.0, height / 2.0
        
        # 寻找离当前位置最近的安全点
        safe_coords = np.column_stack(np.where(safe_positions_mask))
        current_pos = np.array([current_y, current_x])  # 注意：numpy数组使用(y, x)格式
        
        distances = np.linalg.norm(safe_coords - current_pos, axis=1)
        nearest_idx = np.argmin(distances)
        safe_y, safe_x = safe_coords[nearest_idx]
        
        return float(safe_x), float(safe_y)