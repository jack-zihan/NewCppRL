"""
统一坐标系处理模块

解决环境坐标系[x,y]与算法坐标系[y,x]之间的转换问题
"""
from typing import Tuple, List, Union
import numpy as np


class CoordinateSystem:
    """
    统一坐标系管理器
    
    负责环境坐标系和算法坐标系之间的转换:
    - 环境坐标系: [x, y] 格式
    - 算法坐标系: [y, x] 格式（行列索引）
    """
    
    @staticmethod
    def env_to_algo(pos: Union[Tuple, List, np.ndarray]) -> Tuple[float, float]:
        """
        环境坐标[x,y]到算法坐标[y,x]的转换
        
        Args:
            pos: 环境坐标系中的位置 [x, y]
            
        Returns:
            算法坐标系中的位置 (y, x)
        """
        if isinstance(pos, (tuple, list)):
            return (pos[1], pos[0])
        elif isinstance(pos, np.ndarray):
            if pos.shape == (2,):
                return (pos[1], pos[0])
            else:
                # 批量转换
                return pos[:, [1, 0]]
        else:
            raise ValueError(f"Unsupported position type: {type(pos)}")
    
    @staticmethod
    def algo_to_env(pos: Union[Tuple, List, np.ndarray]) -> Union[List, np.ndarray]:
        """
        算法坐标[y,x]到环境坐标[x,y]的转换
        
        Args:
            pos: 算法坐标系中的位置 (y, x)
            
        Returns:
            环境坐标系中的位置 [x, y]
        """
        if isinstance(pos, tuple):
            return [pos[1], pos[0]]
        elif isinstance(pos, list):
            return [pos[1], pos[0]]
        elif isinstance(pos, np.ndarray):
            if pos.shape == (2,):
                return np.array([pos[1], pos[0]])
            else:
                # 批量转换
                return pos[:, [1, 0]]
        else:
            raise ValueError(f"Unsupported position type: {type(pos)}")
    
    @staticmethod
    def batch_env_to_algo(positions: List[Union[Tuple, List]]) -> List[Tuple]:
        """批量转换环境坐标到算法坐标"""
        return [CoordinateSystem.env_to_algo(pos) for pos in positions]
    
    @staticmethod
    def batch_algo_to_env(positions: List[Union[Tuple, List]]) -> List[List]:
        """批量转换算法坐标到环境坐标"""
        return [CoordinateSystem.algo_to_env(pos) for pos in positions]