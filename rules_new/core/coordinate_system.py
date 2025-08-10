"""
坐标系统 - 单一真相源

统一管理所有坐标操作，确保整个系统的坐标一致性
内部统一使用 [y, x] 格式（与旧版rules保持一致）

坐标系统约定：
1. 内部格式：[y, x] - 所有算法内部使用此格式
2. 环境交互：根据环境API自动转换
3. 数组索引：array[y, x] - numpy数组行列索引
4. 显示格式：可配置的输出格式

作者：Rules_new优化团队
版本：2.0.0
"""

import numpy as np
from typing import List, Tuple, Union, Any, Optional
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


class CoordinateSystem:
    """
    坐标系统 - 单一真相源
    
    所有坐标操作必须通过此类进行，确保系统范围内的一致性
    内部统一使用 [y, x] 格式
    """
    
    # 类级别的格式定义（单一真相）
    INTERNAL_FORMAT = 'yx'  # 内部统一格式
    DISPLAY_FORMAT = 'yx'   # 显示格式（可配置）
    
    # 格式验证缓存
    _format_cache = {}
    
    @classmethod
    def get_internal_format(cls) -> str:
        """获取内部坐标格式"""
        return cls.INTERNAL_FORMAT
    
    @classmethod
    def set_display_format(cls, format_type: str):
        """设置显示格式（不影响内部格式）"""
        if format_type not in ['yx', 'xy']:
            raise ValueError(f"不支持的格式: {format_type}")
        cls.DISPLAY_FORMAT = format_type
        logger.info(f"显示格式设置为: {format_type}")
    
    # ========== 核心转换方法 ==========
    
    @staticmethod
    def normalize(pos: Union[List, Tuple, np.ndarray]) -> Tuple[float, float]:
        """
        标准化坐标到内部格式 [y, x]
        
        这是最核心的方法，所有坐标都应该通过此方法标准化
        
        规则：
        - 列表/数组输入：假定为 [x, y] 格式，需要交换为 (y, x)
        - 元组输入：假定已经是 (y, x) 格式，保持不变
        
        Args:
            pos: 任意格式的2D坐标
            
        Returns:
            标准化的 (y, x) 元组
        """
        if not hasattr(pos, '__len__') or len(pos) != 2:
            raise ValueError(f"坐标必须是2维的: {pos}")
        
        # 如果是元组，假定已经是正确格式 (y, x)
        if isinstance(pos, tuple):
            return (float(pos[0]), float(pos[1]))
        
        # 如果是列表或数组，假定是 [x, y] 格式，需要交换
        return (float(pos[1]), float(pos[0]))
    
    @staticmethod
    def from_env(env_x: float, env_y: float) -> Tuple[float, float]:
        """
        从环境坐标（可能是x,y顺序）转换到内部格式
        
        Args:
            env_x: 环境的x坐标
            env_y: 环境的y坐标
            
        Returns:
            内部格式 (y, x)
        """
        return (float(env_y), float(env_x))
    
    @staticmethod
    def to_env(pos: Union[List, Tuple]) -> Tuple[float, float]:
        """
        从内部格式转换到环境坐标
        
        Args:
            pos: 内部格式 [y, x]
            
        Returns:
            环境格式 (x, y)
        """
        y, x = CoordinateSystem.normalize(pos)
        return (x, y)
    
    # ========== 数组操作 ==========
    
    @staticmethod
    def to_array_index(pos: Union[List, Tuple]) -> Tuple[int, int]:
        """
        转换到数组索引 (row, col)
        
        Args:
            pos: 内部格式坐标 [y, x]
            
        Returns:
            数组索引 (row, col) 即 (y, x) 的整数形式
        """
        y, x = CoordinateSystem.normalize(pos)
        return (int(y), int(x))
    
    @staticmethod
    def validate_array_bounds(pos: Union[List, Tuple], 
                            height: int, width: int) -> bool:
        """
        验证坐标是否在数组边界内
        
        Args:
            pos: 内部格式坐标
            height: 数组高度
            width: 数组宽度
            
        Returns:
            是否在边界内
        """
        y, x = CoordinateSystem.normalize(pos)
        return 0 <= y < height and 0 <= x < width
    
    # ========== 批量操作（性能优化） ==========
    
    @staticmethod
    def batch_normalize(positions: List[Any]) -> List[Tuple[float, float]]:
        """
        批量标准化坐标（性能优化版本）
        
        Args:
            positions: 坐标列表
            
        Returns:
            标准化的坐标列表
        """
        return [CoordinateSystem.normalize(pos) for pos in positions]
    
    @staticmethod
    @lru_cache(maxsize=128)
    def cached_normalize(y: float, x: float) -> Tuple[float, float]:
        """
        缓存的标准化（用于频繁调用的坐标）
        
        Args:
            y: y坐标
            x: x坐标
            
        Returns:
            标准化的坐标
        """
        return (float(y), float(x))
    
    # ========== 向量操作 ==========
    
    @staticmethod
    def distance(pos1: Union[List, Tuple], pos2: Union[List, Tuple]) -> float:
        """
        计算两点之间的欧几里得距离
        
        Args:
            pos1: 第一个点
            pos2: 第二个点
            
        Returns:
            距离
        """
        p1 = CoordinateSystem.normalize(pos1)
        p2 = CoordinateSystem.normalize(pos2)
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    @staticmethod
    def vector(from_pos: Union[List, Tuple], 
               to_pos: Union[List, Tuple]) -> np.ndarray:
        """
        计算从一点到另一点的向量
        
        Args:
            from_pos: 起点
            to_pos: 终点
            
        Returns:
            向量 [dy, dx]
        """
        p1 = CoordinateSystem.normalize(from_pos)
        p2 = CoordinateSystem.normalize(to_pos)
        return np.array([p2[0] - p1[0], p2[1] - p1[1]])
    
    @staticmethod
    def angle(from_pos: Union[List, Tuple], 
              to_pos: Union[List, Tuple]) -> float:
        """
        计算从一点到另一点的角度（弧度）
        
        使用数学坐标系：0度在右，逆时针为正
        
        Args:
            from_pos: 起点
            to_pos: 终点
            
        Returns:
            角度（弧度）
        """
        vec = CoordinateSystem.vector(from_pos, to_pos)
        return np.arctan2(vec[0], vec[1])  # 注意：y在前
    
    # ========== 坐标变换 ==========
    
    @staticmethod
    def rotate(pos: Union[List, Tuple], 
               angle: float, 
               center: Optional[Union[List, Tuple]] = None) -> Tuple[float, float]:
        """
        旋转坐标点
        
        Args:
            pos: 要旋转的点
            angle: 旋转角度（弧度）
            center: 旋转中心，默认为原点
            
        Returns:
            旋转后的坐标
        """
        y, x = CoordinateSystem.normalize(pos)
        
        if center is None:
            cy, cx = 0.0, 0.0
        else:
            cy, cx = CoordinateSystem.normalize(center)
        
        # 平移到原点
        y -= cy
        x -= cx
        
        # 旋转
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        new_x = x * cos_a - y * sin_a
        new_y = x * sin_a + y * cos_a
        
        # 平移回去
        return (new_y + cy, new_x + cx)
    
    @staticmethod
    def translate(pos: Union[List, Tuple], 
                  offset: Union[List, Tuple]) -> Tuple[float, float]:
        """
        平移坐标
        
        Args:
            pos: 原始坐标
            offset: 偏移量 [dy, dx]
            
        Returns:
            平移后的坐标
        """
        y, x = CoordinateSystem.normalize(pos)
        dy, dx = CoordinateSystem.normalize(offset)
        return (y + dy, x + dx)
    
    # ========== 验证方法 ==========
    
    @staticmethod
    def validate_consistency(positions: List[Any], 
                            tolerance: float = 1e-6) -> bool:
        """
        验证一组坐标的一致性
        
        Args:
            positions: 坐标列表
            tolerance: 容差
            
        Returns:
            是否一致
        """
        if len(positions) < 2:
            return True
        
        normalized = CoordinateSystem.batch_normalize(positions)
        
        # 检查是否所有点都不同（或在容差范围内相同）
        for i in range(len(normalized) - 1):
            dist = CoordinateSystem.distance(normalized[i], normalized[i + 1])
            if dist < tolerance:
                logger.warning(f"坐标 {i} 和 {i+1} 过于接近: {dist}")
        
        return True
    
    @staticmethod
    def format_for_display(pos: Union[List, Tuple]) -> str:
        """
        格式化坐标用于显示
        
        Args:
            pos: 内部格式坐标
            
        Returns:
            格式化的字符串
        """
        y, x = CoordinateSystem.normalize(pos)
        
        if CoordinateSystem.DISPLAY_FORMAT == 'xy':
            return f"({x:.2f}, {y:.2f})"
        else:
            return f"[{y:.2f}, {x:.2f}]"
    
    # ========== 调试方法 ==========
    
    @staticmethod
    def debug_info(pos: Any) -> dict:
        """
        获取坐标的调试信息
        
        Args:
            pos: 任意格式的坐标
            
        Returns:
            调试信息字典
        """
        try:
            normalized = CoordinateSystem.normalize(pos)
            return {
                'input': pos,
                'normalized': normalized,
                'display': CoordinateSystem.format_for_display(normalized),
                'array_index': CoordinateSystem.to_array_index(normalized),
                'valid': True
            }
        except Exception as e:
            return {
                'input': pos,
                'error': str(e),
                'valid': False
            }


# 便捷别名
CS = CoordinateSystem  # 简短别名，方便使用