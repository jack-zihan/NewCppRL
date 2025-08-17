"""
BCP算法实现 - 基础覆盖规划
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
    SAFETY_MARGIN = 2.0
    DEFAULT_TURNING_RADIUS = 5.0

class AlgorithmDefaults:
    INITIAL_TURN_DIRECTION = True


class BcpPlanner(BasePathPlanner):
    """
    BCP (Basic Coverage Planning) 算法实现
    
    特点：
    - 简单的顺序扫描
    - 基础牛耕式覆盖
    - 最小化转弯次数
    """
    
    def __init__(self, config: Dict[str, Any], env_config: Dict[str, Any]):
        super().__init__(config, env_config)
        
        # 基础参数
        self.agent_width = env_config.get('agent', {}).get('car_width', 5)
        self.sight_width = env_config.get('agent', {}).get('sight_width', 24)
        self.width = env_config.get('environment', {}).get('width', 600)
        self.height = env_config.get('environment', {}).get('height', 600)
        
        # BCP特定参数
        bcp_params = config.get('parameters', {})
        self.simple_coverage = bcp_params.get('simple_coverage', True)
        self.sequential_scan = bcp_params.get('sequential_scan', True)
        self.straight_line_preference = bcp_params.get('straight_line_preference', True)
        
        # 路径生成状态
        self.farm_vertices = None
        self.path_points = []
        self.current_path_index = 0
        self.y_offset = 0
        self.turn_direction = bcp_params.get('initial_turn_direction', AlgorithmDefaults.INITIAL_TURN_DIRECTION)
        self.real_radians = 0
        self.diagonal_length = 0
        self.polygon_mask = None
        self.turning_radius = PathConstants.DEFAULT_TURNING_RADIUS  # Default value, will be updated from initial_state
        
    def reset(self, initial_state: Dict[str, Any]):
        """重置BCP算法状态"""
        super().reset(initial_state)
        
        farm_vertices = initial_state.get('farm_vertices')
        if farm_vertices is not None:
            # 确保farm_vertices是numpy数组
            farm_vertices = np.array(farm_vertices) if not isinstance(farm_vertices, np.ndarray) else farm_vertices
            # 使用统一坐标系统转换：环境坐标[x,y]转为算法坐标[y,x]
            # 与旧版一致：env.min_area_rect[0][:, 0, ::-1]
            if farm_vertices.ndim == 2:
                # 标准化每个顶点
                self.farm_vertices = np.array([CS.normalize([v[1], v[0]]) for v in farm_vertices])
            else:
                self.farm_vertices = farm_vertices
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
        longest_edge = self._find_longest_edge(self.farm_vertices)
        dx = longest_edge[1][0] - longest_edge[0][0]
        dy = longest_edge[1][1] - longest_edge[0][1]
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
        
        if length > 0:
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
        
    def plan_next_waypoint(self, current_state: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """规划下一个路径点"""
        # 更新状态
        self.update_state(current_state)
        
        # 检查终止条件
        if self.should_terminate(current_state):
            return None
            
        # BCP核心逻辑：简单的顺序覆盖，不考虑杂草跳跃
        
        # 如果当前路径用完，生成新的路径线
        if self.current_path_index >= len(self.path_points):
            self.path_points = self._generate_path_line()
            self.current_path_index = 0
            self.turn_direction = not self.turn_direction
            self.y_offset += self.agent_width  # BCP使用较小的步进，确保完全覆盖
            
            # 检查是否超出边界
            if self.y_offset >= self.diagonal_length:
                return None
                
        # 如果没有有效路径点，继续下一条线
        if not self.path_points:
            return self.plan_next_waypoint(current_state)
            
        # BCP算法：简单顺序覆盖，返回路径列表
        batch_points = []
        batch_size = min(10, len(self.path_points) - self.current_path_index)
        
        for i in range(batch_size):
            if self.current_path_index < len(self.path_points):
                next_point = self.path_points[self.current_path_index]
                self.current_path_index += 1
                
                # 分解路径为小步（使用统一坐标系统）
                if i == 0:
                    # 第一个点从当前位置开始，next_point已经是[y,x]格式
                    sub_path = self.decompose_path(
                        current_state['agent_position'],
                        next_point,  # 直接使用，已经是[y,x]格式
                        2.0  # step_size
                    )
                else:
                    # 后续点从前一个点开始
                    prev_point = self.path_points[self.current_path_index - 2]
                    sub_path = self.decompose_path(
                        prev_point,  # 直接使用，已经是[y,x]格式
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