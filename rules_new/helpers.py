#!/usr/bin/env python3
"""
辅助函数集合 - 简化的工具函数

提供常用的工具函数，避免重复代码。
设计原则：函数简单直接，不过度封装。
"""

import numpy as np
import logging
from typing import List, Tuple, Union, Any, Optional, Dict
from pathlib import Path
import yaml
import json


# ==================== 坐标转换函数 ====================
# 简单直接，不需要146行的CoordinateSystem类

def to_yx(position: Union[List, Tuple, np.ndarray]) -> Tuple[float, float]:
    """
    转换为[y, x]格式
    
    Args:
        position: 输入坐标
        
    Returns:
        (y, x)格式的元组
    """
    if len(position) != 2:
        raise ValueError(f"坐标必须是2维的: {position}")
    
    # 如果已经是(y, x)格式（根据上下文判断）
    # 这里简化处理，假设输入是[x, y]需要转换
    return (float(position[1]), float(position[0]))


def to_xy(position: Union[List, Tuple, np.ndarray]) -> Tuple[float, float]:
    """
    转换为[x, y]格式
    
    Args:
        position: 输入坐标
        
    Returns:
        (x, y)格式的元组
    """
    if len(position) != 2:
        raise ValueError(f"坐标必须是2维的: {position}")
    
    # 如果输入是[y, x]格式，转换为[x, y]
    return (float(position[1]), float(position[0]))


def normalize_position(position: Any) -> List[float]:
    """
    标准化位置格式
    
    Args:
        position: 各种格式的位置
        
    Returns:
        标准化的[y, x]列表
    """
    if isinstance(position, (list, tuple)):
        return [float(position[0]), float(position[1])]
    elif isinstance(position, np.ndarray):
        return [float(position[0]), float(position[1])]
    else:
        raise TypeError(f"不支持的位置类型: {type(position)}")


# ==================== 几何计算函数 ====================

def calculate_distance(p1: Union[List, Tuple], p2: Union[List, Tuple]) -> float:
    """
    计算两点间的欧几里得距离
    
    Args:
        p1: 第一个点
        p2: 第二个点
        
    Returns:
        距离
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    return float(np.linalg.norm(p2 - p1))


def calculate_angle(p1: Union[List, Tuple], p2: Union[List, Tuple], p3: Union[List, Tuple]) -> float:
    """
    计算三点形成的角度
    
    Args:
        p1: 第一个点
        p2: 中心点
        p3: 第三个点
        
    Returns:
        角度（弧度）
    """
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    cos_angle = np.clip(cos_angle, -1, 1)
    
    return float(np.arccos(cos_angle))


def point_in_rectangle(point: Union[List, Tuple], 
                       rect_center: Union[List, Tuple], 
                       rect_size: Union[List, Tuple]) -> bool:
    """
    判断点是否在矩形内
    
    Args:
        point: 要检查的点
        rect_center: 矩形中心
        rect_size: 矩形尺寸[height, width]
        
    Returns:
        是否在矩形内
    """
    px, py = point[1], point[0]  # 转换为x, y
    cx, cy = rect_center[1], rect_center[0]
    width, height = rect_size[1], rect_size[0]
    
    return (abs(px - cx) <= width/2) and (abs(py - cy) <= height/2)


def line_intersects_rectangle(p1: Union[List, Tuple], p2: Union[List, Tuple],
                             rect_center: Union[List, Tuple], 
                             rect_size: Union[List, Tuple]) -> bool:
    """
    判断线段是否与矩形相交
    
    简化版本，用于基本碰撞检测
    
    Args:
        p1: 线段起点
        p2: 线段终点
        rect_center: 矩形中心
        rect_size: 矩形尺寸
        
    Returns:
        是否相交
    """
    # 简化检查：如果任一端点在矩形内，则相交
    if point_in_rectangle(p1, rect_center, rect_size):
        return True
    if point_in_rectangle(p2, rect_center, rect_size):
        return True
    
    # 更精确的检查可以后续添加
    return False


# ==================== 日志设置函数 ====================

def setup_logging(log_file: Optional[Path] = None, 
                 level: int = logging.INFO) -> logging.Logger:
    """
    设置日志系统
    
    Args:
        log_file: 日志文件路径
        level: 日志级别
        
    Returns:
        配置好的logger
    """
    logger = logging.getLogger('rules_new')
    logger.setLevel(level)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定）
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# ==================== 配置文件处理 ====================

def load_yaml(file_path: Path) -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        配置字典
    """
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], file_path: Path):
    """
    保存YAML配置文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
    """
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_json(file_path: Path) -> Dict[str, Any]:
    """
    加载JSON文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        数据字典
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: Path, indent: int = 2):
    """
    保存JSON文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
        indent: 缩进级别
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent)


# ==================== 路径处理函数 ====================

def ensure_directory(path: Path) -> Path:
    """
    确保目录存在
    
    Args:
        path: 目录路径
        
    Returns:
        创建好的路径
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """
    获取项目根目录
    
    Returns:
        项目根目录路径
    """
    current = Path(__file__).resolve()
    # 向上查找直到找到rules_new目录
    while current.name != 'rules_new' and current.parent != current:
        current = current.parent
    return current


# ==================== 数据处理函数 ====================

def moving_average(data: List[float], window_size: int = 5) -> List[float]:
    """
    计算移动平均
    
    Args:
        data: 数据列表
        window_size: 窗口大小
        
    Returns:
        平滑后的数据
    """
    if len(data) < window_size:
        return data
    
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2 + 1)
        smoothed.append(np.mean(data[start:end]))
    
    return smoothed


def normalize_angle(angle: float) -> float:
    """
    标准化角度到[-π, π]
    
    Args:
        angle: 输入角度（弧度）
        
    Returns:
        标准化后的角度
    """
    # 处理正负角度
    angle = angle % (2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    elif angle < -np.pi:
        angle += 2 * np.pi
    return angle


def clip_value(value: float, min_val: float, max_val: float) -> float:
    """
    限制值在指定范围内
    
    Args:
        value: 输入值
        min_val: 最小值
        max_val: 最大值
        
    Returns:
        限制后的值
    """
    return max(min_val, min(value, max_val))


# ==================== 性能计时器 ====================

class Timer:
    """
    简单的性能计时器
    
    使用方法:
        with Timer("操作名称"):
            # 要计时的代码
    """
    
    def __init__(self, name: str = "Operation", logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        import time
        elapsed = time.time() - self.start_time
        self.logger.info(f"{self.name} 耗时: {elapsed:.3f}秒")


# ==================== 批处理工具 ====================

def batch_process(items: List[Any], batch_size: int = 32) -> List[List[Any]]:
    """
    将列表分批处理
    
    Args:
        items: 要处理的项目列表
        batch_size: 批大小
        
    Returns:
        分批后的列表
    """
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    return batches


def parallel_map(func, items, max_workers: int = 4):
    """
    并行映射函数
    
    Args:
        func: 要应用的函数
        items: 输入项目列表
        max_workers: 最大工作进程数
        
    Returns:
        结果列表
    """
    from concurrent.futures import ProcessPoolExecutor
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(func, items))
    
    return results