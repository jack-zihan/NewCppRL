"""
SNAKE和R_SNAKE算法实现
"""
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from matplotlib.path import Path
from shapely.geometry import Point, Polygon

from .base_algorithm import BasePathPlanner
from .constants import PathConstants, AlgorithmDefaults
from ..utils.coordinate_system import CoordinateSystem


class SnakePlanner(BasePathPlanner):
    """
    SNAKE算法实现 - 贪婪前向搜索
    
    特点：
    - 贪婪搜索最近的前方杂草
    - 保持当前行进方向
    - 动态重新生成路径
    """
    
    def __init__(self, config: Dict[str, Any], env_config: Dict[str, Any]):
        super().__init__(config, env_config)
        
        # 基础参数
        self.agent_width = env_config.get('agent', {}).get('car_width', 5)
        self.sight_width = env_config.get('agent', {}).get('sight_width', 24)
        self.width = env_config.get('environment', {}).get('width', 600)
        self.height = env_config.get('environment', {}).get('height', 600)
        
        # SNAKE特定参数
        snake_params = config.get('parameters', {})
        self.greedy_search = snake_params.get('greedy_search', AlgorithmDefaults.SNAKE_GREEDY_SEARCH)
        self.forward_only = snake_params.get('forward_only', AlgorithmDefaults.SNAKE_FORWARD_ONLY)
        
        # 路径状态
        self.farm_vertices = None
        self.path_points = []
        self.current_path_index = 0
        self.y_offset = 0
        self.turn_direction = snake_params.get('initial_turn_direction', AlgorithmDefaults.INITIAL_TURN_DIRECTION)
        self.real_radians = 0
        self.diagonal_length = 0
        self.polygon = None
        self.polygon_mask = None
        self.turning_radius = PathConstants.DEFAULT_TURNING_RADIUS  # Default value, will be updated from initial_state
        
    def reset(self, initial_state: Dict[str, Any]):
        """重置SNAKE算法状态"""
        super().reset(initial_state)
        
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
        # 找到最长边
        longest_edge = self._find_longest_edge(self.farm_vertices)
        dx = longest_edge[1][0] - longest_edge[0][0]
        dy = longest_edge[1][1] - longest_edge[0][1]
        self.real_radians = np.arctan2(dy, dx)
        self.real_radians = self.real_radians % (2 * np.pi) if self.real_radians >= 0 else (self.real_radians + 2 * np.pi) % (2 * np.pi)
        
        # 计算对角线长度
        min_x, min_y = self.farm_vertices.min(axis=0)
        max_x, max_y = self.farm_vertices.max(axis=0)
        self.diagonal_length = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)
        
        # 创建多边形
        self.polygon = Polygon(self.farm_vertices)
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
        """创建多边形掩码"""
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
        
    def _get_forward_weeds(self) -> List[Tuple[float, float]]:
        """获取前方的杂草点（SNAKE版本）"""
        if not self.discovered_weeds:
            return []
            
        # 计算当前方向向量
        current_rad = self.real_radians if self.turn_direction else (self.real_radians + np.pi)
        rad_vector = np.array([np.cos(current_rad), np.sin(current_rad)])
        
        # 筛选前方杂草
        agent_pos = np.array(self.current_position)
        forward_weeds = []
        
        for weed in self.discovered_weeds:
            weed_pos = np.array(weed)
            to_weed = weed_pos - agent_pos
            
            # 检查是否在前方
            if np.dot(to_weed, rad_vector) > 0:
                forward_weeds.append(weed)
                
        return forward_weeds
        
    def _find_nearest_weed(self, weeds: List[Tuple[float, float]], turning_radius: float = 5.0) -> Optional[Tuple[float, float]]:
        """寻找最近的有效杂草点"""
        if not weeds:
            return None
            
        agent_pos = np.array(self.current_position)
        valid_weeds = []
        
        for weed in weeds:
            weed_pos = np.array(weed)
            distance = np.linalg.norm(weed_pos - agent_pos)
            
            # 检查距离是否满足最小转弯半径要求
            if distance >= 2 * turning_radius:
                valid_weeds.append((weed, distance))
                
        if not valid_weeds:
            return None
            
        # 返回最近的杂草
        nearest_weed = min(valid_weeds, key=lambda x: x[1])
        return nearest_weed[0]
        
    def _regenerate_path_from_current_position(self, current_rad: float) -> List[Tuple[float, float]]:
        """从当前位置重新生成路径"""
        start_point = self.current_position
        points = []
        
        # 沿当前方向生成点，直到超出多边形
        step_size = 1.0
        current_point = np.array(start_point)
        direction = np.array([np.cos(current_rad), np.sin(current_rad)])
        
        while self.polygon.contains(Point(current_point[0], current_point[1])):
            points.append(tuple(current_point))
            current_point += step_size * direction
            
        return points
        
    def plan_next_waypoint(self, current_state: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """规划下一个路径点"""
        # 更新状态
        self.update_state(current_state)
        
        # 检查终止条件
        if self.should_terminate(current_state):
            return None
            
        # SNAKE核心逻辑：优先寻找前方杂草
        current_rad = self.real_radians if not self.turn_direction else (self.real_radians + np.pi)
        forward_weeds = self._get_forward_weeds()
        nearest_weed = self._find_nearest_weed(forward_weeds)
        
        if nearest_weed is not None:
            # 找到杂草，使用dubins路径前往
            agent_rad = np.pi / 2 - np.radians(current_state['agent_direction'])
            
            # 生成dubins路径到杂草位置
            path_points = self.generate_dubins_path(
                (current_state['agent_position'][0], 
                 current_state['agent_position'][1],
                 agent_rad),
                (nearest_weed[0], nearest_weed[1], current_rad),  # 保持[y,x]格式
                self.turning_radius,
                0.5  # 采样间隔
            )
            
            # 到达杂草后，重新生成前进路径
            self.path_points = self._regenerate_path_from_current_position(current_rad)
            self.current_path_index = 0
            
            # 添加一些前进路径点（使用navigate分解）
            if self.path_points:
                forward_points = []
                for i in range(min(5, len(self.path_points))):
                    if i == 0:
                        sub_path = self.decompose_path(
                            nearest_weed,
                            self.path_points[i],
                            2.0
                        )
                    else:
                        sub_path = self.decompose_path(
                            self.path_points[i-1],
                            self.path_points[i],
                            2.0
                        )
                    forward_points.extend(sub_path)
                path_points.extend(forward_points)
            
            return ('path', path_points)
            
        # 没有杂草，继续沿当前路径
        if self.current_path_index >= len(self.path_points):
            # 当前路径用完，生成新路径线
            self.path_points = self._generate_path_line()
            self.current_path_index = 0
            self.turn_direction = not self.turn_direction
            self.y_offset += self.sight_width / 2
            
            if self.y_offset >= self.diagonal_length:
                return None
                
        if not self.path_points:
            return self.plan_next_waypoint(current_state)
            
        # 正常牛耕式覆盖，返回路径列表
        batch_points = []
        batch_size = min(10, len(self.path_points) - self.current_path_index)
        
        for i in range(batch_size):
            if self.current_path_index < len(self.path_points):
                next_point = self.path_points[self.current_path_index]
                self.current_path_index += 1
                
                # 分解路径为小步
                if i == 0:
                    sub_path = self.decompose_path(
                        current_state['agent_position'],
                        next_point,  # 直接使用，保持[y,x]格式
                        2.0
                    )
                else:
                    prev_point = self.path_points[self.current_path_index - 2]
                    sub_path = self.decompose_path(
                        prev_point,
                        next_point,
                        2.0
                    )
                batch_points.extend(sub_path)
        
        if batch_points:
            return ('path', batch_points)
            
        return None
        
    def should_terminate(self, current_state: Dict[str, Any]) -> bool:
        """判断是否应该终止"""
        coverage_rate = current_state.get('coverage_rate', 0.0)
        if coverage_rate >= 0.98:
            return True
            
        if self.check_timeout():
            return True
            
        if self.check_max_iterations():
            return True
            
        if self.y_offset >= self.diagonal_length and self.current_path_index >= len(self.path_points):
            return True
            
        return False


class RSnakePlanner(SnakePlanner):
    """
    R_SNAKE算法实现 - 受限的SNAKE算法
    
    相比SNAKE算法，增加了垂直约束
    """
    
    def __init__(self, config: Dict[str, Any], env_config: Dict[str, Any]):
        super().__init__(config, env_config)
        
        # R_SNAKE特定参数
        r_snake_params = config.get('parameters', {})
        self.constraint_width = r_snake_params.get('constraint_width', 1.5)
        self.vertical_constraint = r_snake_params.get('vertical_constraint', True)
        
    def _get_forward_weeds(self) -> List[Tuple[float, float]]:
        """获取前方的杂草点（R_SNAKE版本，增加垂直约束）"""
        if not self.discovered_weeds:
            return []
            
        # 计算方向向量
        current_rad = self.real_radians if self.turn_direction else (self.real_radians + np.pi)
        forward_vector = np.array([np.cos(current_rad), np.sin(current_rad)])
        upward_vector = np.array([np.cos(self.real_radians + np.pi / 2), np.sin(self.real_radians + np.pi / 2)])
        
        # 筛选前方和垂直约束范围内的杂草
        agent_pos = np.array(self.current_position)
        forward_weeds = []
        
        for weed in self.discovered_weeds:
            weed_pos = np.array(weed)
            to_weed = weed_pos - agent_pos
            
            # 检查是否在前方
            forward_projection = np.dot(to_weed, forward_vector)
            if forward_projection > 0:
                # 检查垂直约束
                if self.vertical_constraint:
                    upward_projection = np.dot(to_weed, upward_vector)
                    constraint_limit = -self.constraint_width * self.sight_width
                    if upward_projection > constraint_limit:
                        forward_weeds.append(weed)
                else:
                    forward_weeds.append(weed)
                    
        return forward_weeds