"""
JUMP算法实现 - 基于原始jump_path.py的JUMP逻辑重构
"""
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from matplotlib.path import Path

from .base import BasePathPlanner
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from helpers import to_yx, to_xy, calculate_distance

# 算法常量
class PathConstants:
    JUMP_THRESHOLD = 10.0
    SAFETY_MARGIN = 2.0
    DEFAULT_TURNING_RADIUS = 5.0

class AlgorithmDefaults:
    INITIAL_TURN_DIRECTION = True


class JumpPlanner(BasePathPlanner):
    """
    JUMP算法实现
    
    特点：
    - 牛耕式覆盖模式
    - 前向跳跃到发现的杂草点
    - 支持障碍物避让
    """
    
    def __init__(self, config: Dict[str, Any], env_config: Dict[str, Any]):
        super().__init__(config, env_config)
        
        # 算法参数
        self.agent_width = env_config.get('agent', {}).get('car_width', 5)
        self.sight_width = env_config.get('agent', {}).get('sight_width', 24)
        self.sight_length = env_config.get('agent', {}).get('sight_length', 24)
        self.width = env_config.get('environment', {}).get('width', 600)
        self.height = env_config.get('environment', {}).get('height', 600)
        
        # JUMP特定参数
        jump_params = config.get('parameters', {})
        self.jump_threshold = jump_params.get('jump_threshold', PathConstants.JUMP_THRESHOLD)
        self.safety_margin = jump_params.get('safety_margin', PathConstants.SAFETY_MARGIN)
        
        # 路径生成状态
        self.farm_vertices = None
        self.path_points = []
        self.current_path_index = 0
        self.y_offset = 0
        self.turn_direction = jump_params.get('initial_turn_direction', AlgorithmDefaults.INITIAL_TURN_DIRECTION)
        self.real_radians = 0
        self.diagonal_length = 0
        self.longest_edge = None
        self.polygon_mask = None
        self.turning_radius = PathConstants.DEFAULT_TURNING_RADIUS  # Default value, will be updated from initial_state
        
    def reset(self, initial_state: Dict[str, Any]):
        """重置JUMP算法状态"""
        super().reset(initial_state)
        
        # 从环境状态获取农场边界
        farm_vertices = initial_state.get('farm_vertices')
        if farm_vertices is not None:
            # 确保farm_vertices是numpy数组
            farm_vertices = np.array(farm_vertices) if not isinstance(farm_vertices, np.ndarray) else farm_vertices
            # 坐标系转换：环境坐标[x,y]转为算法坐标[y,x]
            # 与旧版一致：env.min_area_rect[0][:, 0, ::-1]
            self.farm_vertices = farm_vertices[:, ::-1] if farm_vertices.ndim == 2 else farm_vertices
            self._initialize_coverage_pattern()
        else:
            self.farm_vertices = None
            
        # 从initial_state获取turning_radius
        if 'turning_radius' in initial_state:
            self.turning_radius = initial_state['turning_radius']
            
        self.current_path_index = 0
        self.y_offset = -self.diagonal_length + self.agent_width / 2
        # 保持初始化时的turn_direction值，不在reset中重置
        
    def _initialize_coverage_pattern(self):
        """初始化覆盖模式"""
        # 计算最长边和方向
        self.longest_edge = self._find_longest_edge(self.farm_vertices)
        dx = self.longest_edge[1][0] - self.longest_edge[0][0]
        dy = self.longest_edge[1][1] - self.longest_edge[0][1]
        self.real_radians = np.arctan2(dy, dx)
        self.real_radians = self.real_radians % (2 * np.pi) if self.real_radians >= 0 else (self.real_radians + 2 * np.pi) % (2 * np.pi)
        
        # 计算对角线长度
        min_x, min_y = self.farm_vertices.min(axis=0)
        max_x, max_y = self.farm_vertices.max(axis=0)
        self.diagonal_length = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)
        
        # 创建多边形掩码
        self._create_polygon_mask()
        
    def _find_longest_edge(self, vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """找到多边形的最长边"""
        max_length = 0
        longest_edge = None
        
        for i in range(len(vertices)):
            start = vertices[i]
            end = vertices[(i + 1) % len(vertices)]
            length = np.linalg.norm(end - start)
            if length > max_length:
                max_length = length
                longest_edge = (start, end)
                
        return longest_edge
        
    def _create_polygon_mask(self):
        """创建多边形掩码用于路径验证"""
        poly_path = Path(self.farm_vertices)
        y, x = np.mgrid[:self.height, :self.width]
        coords = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        self.polygon_mask = np.zeros((self.height, self.width))
        self.polygon_mask[poly_path.contains_points(coords).reshape(self.height, self.width)] = 1
        
    def _generate_path_line(self) -> List[Tuple[float, float]]:
        """生成当前y_offset对应的路径线"""
        start = [0, 0]
        end = np.array([100 * np.cos(self.real_radians), 100 * np.sin(self.real_radians)])
        
        new_start = [
            start[0] + self.y_offset * np.cos(self.real_radians + np.pi / 2) - self.diagonal_length * np.cos(self.real_radians),
            start[1] + self.y_offset * np.sin(self.real_radians + np.pi / 2) - self.diagonal_length * np.sin(self.real_radians)
        ]
        new_end = [
            end[0] + self.y_offset * np.cos(self.real_radians + np.pi / 2) + self.diagonal_length * np.cos(self.real_radians),
            end[1] + self.y_offset * np.sin(self.real_radians + np.pi / 2) + self.diagonal_length * np.sin(self.real_radians)
        ]
        
        # 生成线上的点
        line_points = []
        direction = np.array(new_end) - np.array(new_start)
        length = np.linalg.norm(direction)
        
        for i in np.arange(0, length, 1):
            interpolated_point = np.array(new_start) + (i / length) * direction
            line_points.append(interpolated_point)
            
        # 过滤有效点
        valid_points = [
            point for point in line_points 
            if (0 <= int(point[1]) < self.height and 
                0 <= int(point[0]) < self.width and 
                self.polygon_mask[int(point[1]), int(point[0])] == 1)
        ]
        
        # 根据转向方向调整顺序
        if not self.turn_direction:
            valid_points = valid_points[::-1]
            
        return valid_points
        
    def _find_jump_target(self, forward_weeds: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        """寻找跳跃目标点"""
        if not forward_weeds:
            return None
            
        # 使用原始的find_nearest_point_jump逻辑
        radians = -self.real_radians if not self.turn_direction else -(self.real_radians + np.pi)
        radians = radians % (2 * np.pi)
        
        rotation_matrix = np.array([
            [np.cos(radians), -np.sin(radians)],
            [np.sin(radians), np.cos(radians)]
        ])
        
        p_rotated = np.dot(rotation_matrix, np.array(self.current_position))
        rotated_coords = [np.dot(rotation_matrix, np.array(c)) for c in forward_weeds]
        
        if not rotated_coords:
            return None
            
        nearest_index = min(range(len(rotated_coords)), key=lambda i: abs(rotated_coords[i][0] - p_rotated[0]))
        return forward_weeds[nearest_index]
        
    def _get_forward_weeds(self) -> List[Tuple[float, float]]:
        """获取前方的杂草点"""
        if not self.discovered_weeds:
            return []
            
        # 计算当前方向向量
        current_rad = self.real_radians if self.turn_direction else (self.real_radians + np.pi)
        rad_vector = np.array([np.cos(current_rad), np.sin(current_rad)])
        
        # 垂直方向
        vertical_rad = self.real_radians + np.pi / 2
        vertical_rad = vertical_rad - 2 * np.pi if vertical_rad > np.pi else vertical_rad
        vertical_vector = np.array([np.cos(vertical_rad), np.sin(vertical_rad)])
        
        # 筛选前方和垂直方向的杂草
        agent_pos = np.array(self.current_position)
        forward_weeds = []
        
        for weed in self.discovered_weeds:
            weed_pos = np.array(weed)
            to_weed = weed_pos - agent_pos
            
            # 检查是否在前方
            if np.dot(to_weed, rad_vector) > 0:
                # 检查是否在垂直范围内
                if np.dot(to_weed, vertical_vector) > 0:
                    forward_weeds.append(weed)
                    
        return forward_weeds
        
    def plan_next_waypoint(self, current_state: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """规划下一个路径点"""
        # 更新状态
        self.update_state(current_state)
        
        # 检查终止条件
        if self.should_terminate(current_state):
            return None
            
        # 如果当前路径用完，生成新的路径线
        if self.current_path_index >= len(self.path_points):
            self.path_points = self._generate_path_line()
            self.current_path_index = 0
            self.turn_direction = not self.turn_direction
            self.y_offset += self.sight_width / 2
            
            # 检查是否超出边界
            if self.y_offset >= self.diagonal_length:
                return None
                
        # 如果没有有效路径点，继续下一条线
        if not self.path_points:
            return self.plan_next_waypoint(current_state)
            
        # JUMP逻辑：检查是否需要跳跃到杂草
        forward_weeds = self._get_forward_weeds()
        jump_target = self._find_jump_target(forward_weeds)
        
        if jump_target is not None:
            # 使用dubins路径跳跃到杂草
            current_rad = np.pi / 2 - np.radians(current_state['agent_direction'])
            target_rad = self.real_radians if not self.turn_direction else self.real_radians + np.pi
            
            # 生成dubins路径
            path_points = self.generate_dubins_path(
                (current_state['agent_position'][0], 
                 current_state['agent_position'][1],
                 current_rad),
                (jump_target[0], jump_target[1], target_rad),  # 保持[y,x]格式
                self.turning_radius,
                0.5  # 采样间隔
            )
            
            # 返回路径列表
            if path_points:
                return ('path', path_points)
            
        # 正常沿路径前进 - 使用navigate分解
        if self.current_path_index < len(self.path_points):
            # 获取下一批路径点（可以一次处理多个点）
            batch_size = min(5, len(self.path_points) - self.current_path_index)
            batch_points = []
            
            for i in range(batch_size):
                if self.current_path_index < len(self.path_points):
                    next_point = self.path_points[self.current_path_index]
                    self.current_path_index += 1
                    # 分解路径为小步
                    if i == 0:
                        # 第一个点需要从当前位置开始分解
                        sub_path = self.decompose_path(
                            current_state['agent_position'],
                            next_point,  # 直接使用，保持[y,x]格式
                            2.0  # step_size
                        )
                    else:
                        # 后续点从前一个点开始分解
                        prev_point = self.path_points[self.current_path_index - 2]
                        sub_path = self.decompose_path(
                            prev_point,
                            next_point,
                            2.0
                        )
                    batch_points.extend(sub_path)
            
            # 返回路径列表
            if batch_points:
                return ('path', batch_points)
            
        return None
        
    def should_terminate(self, current_state: Dict[str, Any]) -> bool:
        """判断是否应该终止"""
        # 检查覆盖率
        coverage_rate = current_state.get('coverage_rate', 0.0)
        if coverage_rate >= 0.98:  # 98%覆盖率
            return True
            
        # 检查超时
        if self.check_timeout():
            return True
            
        # 检查最大迭代次数  
        if self.check_max_iterations():
            return True
            
        # 检查是否完成所有路径
        if self.y_offset >= self.diagonal_length and self.current_path_index >= len(self.path_points):
            return True
            
        return False