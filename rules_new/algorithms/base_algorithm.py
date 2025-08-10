"""
算法基类 - 定义所有路径规划算法的通用接口

使用统一的坐标系统和异常处理
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import time
import logging

# 导入核心组件
try:
    from ..core import CoordinateSystem as CS, AlgorithmError, handle_errors
except ImportError:
    # 如果核心模块不存在，使用默认实现
    class CS:
        @staticmethod
        def normalize(pos):
            return (float(pos[0]), float(pos[1]))
        @staticmethod
        def distance(pos1, pos2):
            p1 = CS.normalize(pos1)
            p2 = CS.normalize(pos2)
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    AlgorithmError = Exception
    def handle_errors(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)


class BasePathPlanner(ABC):
    """
    路径规划算法基类
    
    提供所有算法的通用接口和基础功能：
    - 配置管理
    - 状态跟踪
    - 性能监控
    - 结果收集
    """
    
    def __init__(self, config: Dict[str, Any], env_config: Dict[str, Any]):
        """
        初始化路径规划器
        
        Args:
            config: 算法特定配置
            env_config: 环境配置
        """
        self.config = config
        self.env_config = env_config
        self.algorithm_name = config.get('algorithm', {}).get('name', 'Unknown')
        
        # 性能监控
        self.start_time = None
        self.total_distance = 0.0
        self.coverage_milestones = {}  # 存储90%, 95%, 98%覆盖率对应的距离
        self.coverage_history = []
        self.distance_history = []
        
        # 状态跟踪
        self.current_position = None
        self.current_direction = None
        self.discovered_weeds = []
        self.iteration_count = 0
        
    @handle_errors(AlgorithmError, save_checkpoint=True, reraise=True)
    def reset(self, initial_state: Dict[str, Any]):
        """重置算法状态（使用统一坐标系统）"""
        self.start_time = time.time()
        self.total_distance = 0.0 
        self.coverage_milestones = {}
        self.coverage_history = []
        self.distance_history = []
        self.iteration_count = 0
        
        # 从初始状态提取位置和方向（使用统一坐标系统）
        raw_position = initial_state.get('agent_position', [0, 0])
        self.current_position = CS.normalize(raw_position)  # 统一为 (y, x) 元组
        self.current_direction = initial_state.get('agent_direction', 0)
        
        # 标准化杂草位置
        raw_weeds = initial_state.get('discovered_weeds', [])
        self.discovered_weeds = [CS.normalize(w) for w in raw_weeds]
        
    @handle_errors(AlgorithmError, save_checkpoint=False, reraise=True)
    def update_state(self, new_state: Dict[str, Any]):
        """更新算法状态（使用统一坐标系统）"""
        if self.current_position is not None:
            # 使用统一的距离计算
            raw_new_pos = new_state['agent_position']
            new_pos = CS.normalize(raw_new_pos)
            distance = CS.distance(self.current_position, new_pos)
            self.total_distance += distance
            
        # 更新位置（统一格式）
        self.current_position = CS.normalize(new_state['agent_position'])
        self.current_direction = new_state['agent_direction']
        
        # 更新杂草位置（统一格式）
        raw_weeds = new_state.get('discovered_weeds', [])
        self.discovered_weeds = [CS.normalize(w) for w in raw_weeds]
        
        # 记录覆盖率里程碑
        coverage_rate = new_state.get('coverage_rate', 0.0)
        self.coverage_history.append(coverage_rate)
        self.distance_history.append(self.total_distance)
        
        for milestone in [0.90, 0.95, 0.98]:
            if coverage_rate >= milestone and milestone not in self.coverage_milestones:
                self.coverage_milestones[milestone] = self.total_distance
                
        self.iteration_count += 1
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        metrics = {
            'algorithm': self.algorithm_name,
            'total_distance': self.total_distance,
            'coverage_90': self.coverage_milestones.get(0.90, -1),
            'coverage_95': self.coverage_milestones.get(0.95, -1), 
            'coverage_98': self.coverage_milestones.get(0.98, -1),
            'final_coverage': self.coverage_history[-1] if self.coverage_history else 0.0,
            'iterations': self.iteration_count,
            'runtime': time.time() - self.start_time if self.start_time else 0.0
        }
        return metrics
        
    def check_timeout(self) -> bool:
        """检查是否超时"""
        timeout = self.config.get('performance', {}).get('timeout_seconds', 300)
        if self.start_time and (time.time() - self.start_time) > timeout:
            return True
        return False
        
    def check_max_iterations(self) -> bool:
        """检查是否达到最大迭代次数"""
        max_iter = self.config.get('performance', {}).get('max_iterations', 5000)
        return self.iteration_count >= max_iter
    
    @abstractmethod
    def plan_next_waypoint(self, current_state: Dict[str, Any]) -> Optional[Any]:
        """
        规划下一个路径点
        
        Args:
            current_state: 当前环境状态，包含：
                - agent_position: [x, y]
                - agent_direction: 弧度
                - discovered_weeds: [[x, y], ...]
                - maps: 各种地图信息
                - coverage_rate: 当前覆盖率
                
        Returns:
            下一个路径点，格式可以是：
            - None: 没有更多路径点
            - (x, y): 单个路径点
            - ('path', [(x1,y1), (x2,y2), ...]): 路径点列表
            - [x, y]: 单个路径点（列表格式）
        """
        pass
        
    @abstractmethod 
    def should_terminate(self, current_state: Dict[str, Any]) -> bool:
        """
        判断是否应该终止规划
        
        Args:
            current_state: 当前环境状态
            
        Returns:
            是否应该终止
        """
        pass
    
    # ========== 路径工具方法 ==========
    
    @staticmethod
    def decompose_path(start: List[float], target: List[float], step_size: float = 2.0) -> List[Tuple[float, float]]:
        """
        将路径分解为小步（优化版本，使用向量化计算）
        
        Args:
            start: 起始点 [y, x]
            target: 目标点 [y, x]
            step_size: 分解步长
            
        Returns:
            路径点列表（统一格式的元组）
        """
        # 统一坐标格式
        start_pos = CS.normalize(start)
        target_pos = CS.normalize(target)
        
        # 计算向量和距离
        vector = np.array(target_pos) - np.array(start_pos)
        distance = np.linalg.norm(vector)
        
        if distance < 1e-6:
            return [target_pos]
        
        # 向量化计算所有路径点（性能优化）
        num_steps = max(1, int(distance // step_size))
        t = np.linspace(0, 1, num_steps + 1)[1:]  # 不包括起点
        
        # 向量化生成路径点
        waypoints = start_pos + t[:, np.newaxis] * vector
        
        # 转换为元组列表（统一格式）
        result = [tuple(point) for point in waypoints]
        
        return result
    
    @staticmethod
    def generate_dubins_path(start_pose: Tuple[float, float, float], 
                            end_pose: Tuple[float, float, float],
                            turning_radius: float,
                            sample_interval: float = 0.5) -> List[List[float]]:
        """
        生成dubins平滑路径点
        
        Args:
            start_pose: 起始位姿 (x, y, angle)
            end_pose: 目标位姿 (x, y, angle)
            turning_radius: 转弯半径
            sample_interval: 采样间隔
            
        Returns:
            路径点列表 [[x, y], ...]
        """
        try:
            import dubins
            path = dubins.shortest_path(start_pose, end_pose, turning_radius)
            configurations, _ = path.sample_many(sample_interval)
            # 返回路径点（跳过第一个点，因为是当前位置）
            return [list(point[:2]) for point in configurations[1:]]
        except ImportError:
            # 如果没有安装dubins库，返回直线路径
            return BasePathPlanner.decompose_path(
                list(start_pose[:2]), 
                list(end_pose[:2]), 
                sample_interval
            )