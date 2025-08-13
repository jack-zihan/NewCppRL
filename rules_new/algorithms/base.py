#!/usr/bin/env python3
"""
路径规划算法基类

定义所有路径规划算法的通用接口。
简化版本，去除过度的异常处理和坐标系统封装。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class BasePathPlanner(ABC):
    """
    路径规划算法基类
    
    所有算法必须继承此类并实现核心方法。
    坐标统一使用[y, x]格式。
    """
    
    def __init__(self, **kwargs):
        """
        初始化算法
        
        Args:
            **kwargs: 算法特定参数
        """
        self.name = self.__class__.__name__
        self.params = kwargs
        
        # 算法状态
        self.current_position = None
        self.trajectory = []
        self.is_initialized = False
    
    @abstractmethod
    def get_action(self, observation: Dict[str, Any]) -> Any:
        """
        根据观察获取下一步动作
        
        这是算法的核心方法，必须实现。
        
        Args:
            observation: 环境观察，包含agent_position等信息
            
        Returns:
            动作（格式取决于具体环境）
        """
        pass
    
    def reset(self):
        """重置算法状态"""
        self.current_position = None
        self.trajectory = []
        self.is_initialized = False
    
    def update_position(self, position: List[float]):
        """
        更新当前位置
        
        Args:
            position: 新位置[y, x]
        """
        self.current_position = position
        self.trajectory.append(tuple(position))
    
    def get_trajectory(self) -> List[Tuple[float, float]]:
        """获取完整轨迹"""
        return self.trajectory.copy()
    
    def calculate_distance(self, p1: List[float], p2: List[float]) -> float:
        """
        计算两点间距离（工具方法）
        
        Args:
            p1: 第一个点[y, x]
            p2: 第二个点[y, x]
            
        Returns:
            欧几里得距离
        """
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取算法信息
        
        Returns:
            包含算法状态和参数的字典
        """
        return {
            'name': self.name,
            'params': self.params,
            'trajectory_length': len(self.trajectory),
            'is_initialized': self.is_initialized
        }


class SimpleNavigator(BasePathPlanner):
    """
    简单导航器示例
    
    展示如何实现基类接口。
    """
    
    def __init__(self, step_size: float = 1.0, **kwargs):
        super().__init__(step_size=step_size, **kwargs)
        self.step_size = step_size
    
    def get_action(self, observation: Dict[str, Any]) -> List[float]:
        """
        简单的向目标移动策略
        
        Args:
            observation: 包含agent_position和goal_position
            
        Returns:
            下一个目标位置
        """
        current_pos = observation.get('agent_position', [0, 0])
        goal_pos = observation.get('goal_position', [100, 100])
        
        # 计算方向
        direction = np.array(goal_pos) - np.array(current_pos)
        distance = np.linalg.norm(direction)
        
        if distance < self.step_size:
            # 已到达目标
            next_pos = goal_pos
        else:
            # 向目标移动一步
            direction = direction / distance  # 单位向量
            next_pos = current_pos + direction * self.step_size
        
        self.update_position(list(next_pos))
        return list(next_pos)