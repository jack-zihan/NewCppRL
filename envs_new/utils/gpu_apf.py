"""GPU加速的APF计算"""
import numpy as np
import cupy as cp
from cupyx.scipy import ndimage as cp_ndimage


def gpu_apf_bool(binary_map: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    GPU版本的APF欧几里得距离计算
    使用标准欧几里得距离变换，物理上更正确
    
    Args:
        binary_map: 二值地图，1表示障碍物
        
    Returns:
        distance_map: 欧几里得距离图
        is_empty: 是否为空地图
    """
    # 转到GPU
    map_gpu = cp.asarray(binary_map)
    
    # 检查是否为空
    is_empty = not cp.any(map_gpu).item()
    
    if is_empty:
        return np.zeros_like(binary_map, dtype=np.float32), True
    
    # 计算标准欧几里得距离（使用EDT算法）
    inverted_map = ~map_gpu.astype(bool)
    distance_map_gpu = cp_ndimage.distance_transform_edt(
        inverted_map
    ).astype(cp.float32)
    
    # 返回CPU
    return distance_map_gpu.get(), is_empty