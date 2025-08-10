"""
REACT算法实现 - 反应式路径规划
"""
import math
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from matplotlib.path import Path
from shapely.geometry import Point, Polygon, LineString

from .base_algorithm import BasePathPlanner
from ..core import CoordinateSystem as CS


class ReactPlanner(BasePathPlanner):
    """
    REACT算法实现 - 反应式随机探索
    
    特点：
    - 随机生成目标点
    - 反应式地响应发现的杂草
    - 路径验证和碰撞检测
    """
    
    def __init__(self, config: Dict[str, Any], env_config: Dict[str, Any]):
        super().__init__(config, env_config)
        
        # 基础参数
        self.width = env_config.get('environment', {}).get('width', 600)
        self.height = env_config.get('environment', {}).get('height', 600)
        
        # REACT特定参数
        react_params = config.get('parameters', {})
        self.max_attempts = react_params.get('max_attempts', 50)
        self.random_exploration = react_params.get('random_exploration', True)
        self.goal_bounds = react_params.get('goal_generation_bounds', [600, 600])
        
        # 状态变量
        self.farm_vertices = None
        self.polygon = None
        self.polygon_mask = None
        self.current_goal = None
        self.goal_path = []
        self.path_index = 0
        self.attempts_count = 0
        self.turning_radius = 5.0  # Default value, will be updated from initial_state
        
        # 随机数生成器
        self.rng = random.Random()
        
    def reset(self, initial_state: Dict[str, Any]):
        """重置REACT算法状态"""
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
            self.polygon = Polygon(self.farm_vertices)
            self._create_polygon_mask()
        else:
            self.farm_vertices = None
            self.polygon = None
            
        # 从initial_state获取turning_radius
        if 'turning_radius' in initial_state:
            self.turning_radius = initial_state['turning_radius']
            
        self.current_goal = None
        self.goal_path = []
        self.path_index = 0
        self.attempts_count = 0
        
        # 设置随机种子（如果提供）
        seed = initial_state.get('seed')
        if seed is not None:
            self.rng.seed(seed)
            
    def _create_polygon_mask(self):
        """创建多边形掩码"""
        poly_path = Path(self.farm_vertices)
        y, x = np.mgrid[:self.height, :self.width]
        coords = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        self.polygon_mask = np.zeros((self.height, self.width))
        self.polygon_mask[poly_path.contains_points(coords).reshape(self.height, self.width)] = 1
        
    def _generate_random_goal(self) -> Tuple[float, float]:
        """生成随机目标点"""
        max_attempts = 100
        for _ in range(max_attempts):
            x = self.rng.uniform(0, self.goal_bounds[0])
            y = self.rng.uniform(0, self.goal_bounds[1])
            
            # 检查点是否在农场边界内
            if self.polygon and self.polygon.contains(Point(x, y)):
                return (x, y)
                
        # 如果无法找到有效点，返回农场中心
        if self.farm_vertices is not None:
            center_x = np.mean(self.farm_vertices[:, 0])
            center_y = np.mean(self.farm_vertices[:, 1])
            return (center_x, center_y)
            
        return (self.width // 2, self.height // 2)
        
    def _generate_path_to_goal(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        """生成从起点到目标的路径"""
        # 创建直线路径
        start_point = np.array(start)
        goal_point = np.array(goal)
        
        # 计算路径点
        direction = goal_point - start_point
        distance = np.linalg.norm(direction)
        
        if distance < 1.0:
            return [goal]
            
        # 以1单位间隔生成路径点
        num_points = int(distance)
        path_points = []
        
        for i in range(1, num_points + 1):
            interpolated_point = start_point + (i / num_points) * direction
            path_points.append(tuple(interpolated_point))
            
        # 添加最终目标点
        path_points.append(goal)
        
        # 过滤有效点（在农场边界内）
        valid_points = []
        for point in path_points:
            if (0 <= int(point[1]) < self.height and 
                0 <= int(point[0]) < self.width and 
                self.polygon_mask[int(point[1]), int(point[0])] == 1):
                valid_points.append(point)
                
        return valid_points
        
    def _find_nearest_weed(self) -> Optional[Tuple[float, float]]:
        """寻找最近的杂草点"""
        if not self.discovered_weeds:
            return None
            
        agent_pos = np.array(self.current_position)
        min_distance = float('inf')
        nearest_weed = None
        
        for weed in self.discovered_weeds:
            weed_pos = np.array(weed)
            distance = np.linalg.norm(weed_pos - agent_pos)
            
            if distance < min_distance:
                min_distance = distance
                nearest_weed = weed
                
        return nearest_weed
        
    def _validate_path(self, path: List[Tuple[float, float]]) -> bool:
        """验证路径的有效性（简化版碰撞检测）"""
        if not path:
            return False
            
        # 检查路径上的点是否都在有效区域内
        for point in path:
            x, y = int(point[0]), int(point[1])
            if (0 <= x < self.width and 0 <= y < self.height):
                if self.polygon_mask[y, x] == 0:  # 在无效区域
                    return False
            else:
                return False  # 超出边界
                
        return True
        
    def plan_next_waypoint(self, current_state: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """规划下一个路径点"""
        # 更新状态
        self.update_state(current_state)
        
        # 检查终止条件
        if self.should_terminate(current_state):
            return None
            
        # REACT核心逻辑：优先响应发现的杂草
        nearest_weed = self._find_nearest_weed() 
        
        if nearest_weed is not None:
            # 发现杂草，使用dubins路径快速响应
            agent_rad = np.pi / 2 - np.radians(current_state['agent_direction'])
            target_rad = math.atan2(
                nearest_weed[1] - current_state['agent_position'][1],
                nearest_weed[0] - current_state['agent_position'][0]
            )
            
            # 生成dubins路径到杂草（nearest_weed已经是[y,x]格式）
            path_points = self.generate_dubins_path(
                (current_state['agent_position'][0],
                 current_state['agent_position'][1],
                 agent_rad),
                (nearest_weed[0], nearest_weed[1], target_rad),  # 直接使用，已经是[y,x]
                self.turning_radius,
                0.5
            )
            
            if path_points:
                return ('path', path_points)
                
        # 没有杂草，继续随机探索
        if not self.goal_path or self.path_index >= len(self.goal_path):
            # 需要新目标
            if self.attempts_count >= self.max_attempts:
                return None  # 达到最大尝试次数
                
            # 生成新的随机目标
            self.current_goal = self._generate_random_goal()
            if self.current_goal:
                # 生成到目标的路径（使用LineString采样）
                line = LineString([current_state['agent_position'], self.current_goal])
                path_points = []
                
                # 采样路径点
                for i in np.arange(0, min(line.length, 50), 1):  # 限制最大长度
                    point = line.interpolate(i).coords[0]
                    # 验证点的有效性
                    x, y = int(point[0]), int(point[1])
                    if 0 <= x < self.width and 0 <= y < self.height:
                        if self.polygon_mask[y, x] == 1:  # 在有效区域
                            path_points.append(list(point))
                        else:
                            break  # 遇到无效区域，停止
                    else:
                        break  # 超出边界
                
                if path_points:
                    # 使用dubins到达第一个有效点
                    goal_rad = math.atan2(
                        path_points[0][1] - current_state['agent_position'][1],
                        path_points[0][0] - current_state['agent_position'][0]
                    )
                    
                    dubins_path = self.generate_dubins_path(
                        (current_state['agent_position'][0],
                         current_state['agent_position'][1],
                         np.pi / 2 - np.radians(current_state['agent_direction'])),
                        (path_points[0][0], path_points[0][1], goal_rad),
                        self.turning_radius,
                        0.5
                    )
                    
                    # 分解后续路径点
                    for i in range(1, min(10, len(path_points))):
                        sub_path = self.decompose_path(
                            path_points[i-1],
                            path_points[i],
                            2.0
                        )
                        dubins_path.extend(sub_path)
                    
                    self.goal_path = []  # 清空目标路径
                    self.path_index = 0
                    self.attempts_count += 1
                    
                    return ('path', dubins_path)
                    
            # 如果无法生成有效路径，增加尝试次数并重试
            self.attempts_count += 1
            if self.attempts_count < self.max_attempts:
                return self.plan_next_waypoint(current_state)
                
        return None
        
    def should_terminate(self, current_state: Dict[str, Any]) -> bool:
        """判断是否应该终止"""
        # 检查覆盖率
        coverage_rate = current_state.get('coverage_rate', 0.0)
        if coverage_rate >= 0.98:
            return True
            
        # 检查超时
        if self.check_timeout():
            return True
            
        # 检查最大迭代次数
        if self.check_max_iterations():
            return True
            
        # 检查最大尝试次数（REACT特有）
        if self.attempts_count >= self.max_attempts:
            return True
            
        return False