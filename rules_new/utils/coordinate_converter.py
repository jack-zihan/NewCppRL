"""
坐标系统转换工具类

确保新旧版本坐标系一致性
旧版统一使用[y,x]格式，新版需要保持一致
"""
import numpy as np
from typing import List, Tuple, Union, Any


class CoordinateConverter:
    """
    坐标系转换器
    
    统一处理环境坐标和算法坐标之间的转换
    确保与旧版rules保持一致的[y,x]格式
    """
    
    @staticmethod
    def ensure_yx_format(pos: Union[List, Tuple, np.ndarray]) -> List[float]:
        """
        确保坐标是[y,x]格式
        
        Args:
            pos: 输入坐标，可能是[x,y]或[y,x]
            
        Returns:
            [y,x]格式的坐标
        """
        if len(pos) != 2:
            raise ValueError(f"坐标必须是2维的，但收到: {pos}")
        return [float(pos[0]), float(pos[1])]
    
    @staticmethod
    def env_xy_to_algo_yx(env_pos: Union[List, Tuple]) -> List[float]:
        """
        环境坐标[x,y] -> 算法坐标[y,x]
        
        用于从环境提取坐标时的转换
        env.agent.x, env.agent.y -> [y, x]
        """
        return [float(env_pos[1]), float(env_pos[0])]
    
    @staticmethod
    def algo_yx_to_env_xy(algo_pos: Union[List, Tuple]) -> List[float]:
        """
        算法坐标[y,x] -> 环境坐标[x,y]
        
        用于传递给环境时的转换（如果需要）
        """
        return [float(algo_pos[1]), float(algo_pos[0])]
    
    @staticmethod
    def batch_convert_to_yx(positions: List[Union[List, Tuple]]) -> List[List[float]]:
        """
        批量转换坐标到[y,x]格式
        
        Args:
            positions: 坐标列表
            
        Returns:
            [y,x]格式的坐标列表
        """
        return [CoordinateConverter.ensure_yx_format(pos) for pos in positions]
    
    @staticmethod
    def verify_farm_vertices(vertices: np.ndarray) -> np.ndarray:
        """
        验证并转换农场顶点坐标
        
        确保顶点坐标是正确的[y,x]格式
        与旧版的env.min_area_rect[0][:, 0, ::-1]保持一致
        """
        if vertices.ndim == 2:
            # 如果是2维数组，确保是[y,x]格式
            return vertices
        elif vertices.ndim == 3:
            # 如果是3维数组（从min_area_rect提取），进行相应转换
            return vertices[:, 0, ::-1]
        else:
            raise ValueError(f"农场顶点维度错误: {vertices.ndim}")
    
    @staticmethod
    def array_index_yx(y: int, x: int) -> Tuple[int, int]:
        """
        生成数组索引[y,x]
        
        用于访问地图数组：map[y, x]
        """
        return (int(y), int(x))
    
    @staticmethod
    def validate_coordinate_consistency(pos1: Any, pos2: Any, tolerance: float = 1e-6) -> bool:
        """
        验证两个坐标是否一致
        
        Args:
            pos1: 第一个坐标
            pos2: 第二个坐标
            tolerance: 容差
            
        Returns:
            是否一致
        """
        try:
            p1 = CoordinateConverter.ensure_yx_format(pos1)
            p2 = CoordinateConverter.ensure_yx_format(pos2)
            
            diff = np.linalg.norm(np.array(p1) - np.array(p2))
            return diff < tolerance
        except:
            return False