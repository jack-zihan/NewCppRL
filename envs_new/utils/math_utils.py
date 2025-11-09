"""
数学工具函数模块
"""
from __future__ import annotations
import math
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

def total_variation(image: np.ndarray) -> int:
    """计算图像的总变差，用于前沿边缘检测。"""
    pixel_dif1 = image[1:, :] - image[:-1, :]
    pixel_dif2 = image[:, 1:] - image[:, :-1]
    return int(np.abs(pixel_dif1).sum() + np.abs(pixel_dif2).sum())


def total_variation_mat(mat: np.ndarray) -> np.ndarray:
    """计算总变差矩阵，标记与邻域不同的像素位置。"""
    tv = np.zeros_like(mat, dtype=bool)
    tv[1:, :] |= (mat[1:, :] != mat[:-1, :])
    tv[:-1, :] |= (mat[1:, :] != mat[:-1, :])
    tv[:, 1:] |= (mat[:, 1:] != mat[:, :-1])
    tv[:, :-1] |= (mat[:, 1:] != mat[:, :-1])
    return tv

# ========== 观察向量工具函数 ==========
# 这些函数被v4/v5复用，用于处理历史序列、坐标变换、角度编码

def _pad_history(history: Union[List, np.ndarray], length: int,
                 pad_value: Union[float, np.ndarray]) -> np.ndarray:
    """通用历史序列padding（numpy优化版本）支持：- 1D数组：标量序列 [a, b, c]，- 2D数组：向量序列 [[x1,y1], [x2,y2]]，- List输入：自动转换为numpy数组
        Args: history: List或ndarray，length: 目标长度，pad_value: 填充值（标量或数组）
        Returns: (length,) or (length, D) numpy数组"""

    if isinstance(history, list): history = np.array(history, dtype=np.float32) # 统一转换为numpy
    recent = history[-length:] if len(history) >= length else history # 取最后length项

    padding_needed = length - len(recent)
    if padding_needed == 0: return recent # 不需要padding，直接返回

    # pad_value转numpy
    if isinstance(pad_value, (list, tuple)): pad_value = np.array(pad_value, dtype=np.float32)

    # 创建padding
    if recent.ndim == 1:
        padding = np.full(padding_needed, pad_value, dtype=np.float32)
    else:  # 2D
        padding = np.tile(pad_value, (padding_needed, 1))
    return np.concatenate([padding, recent], axis=0)


def _to_ego_frame(positions: List[Tuple], ref_pos: Tuple[float, float],
                   ref_heading: float, map_size: Tuple[int, int]) -> np.ndarray:
    """转换位置序列到自我坐标系并归一化（numpy优化版本），positions: 世界坐标位置列表 [(x,y), ...]， ref_pos: 参考位置（当前位置）， ref_heading: 参考朝向（度，图像坐标系），map_size: 地图尺寸 (width, height) 用于归一化"""
    pos_array = np.array(positions, dtype=np.float32) # 转换为numpy数组 (N, 2)
    rel_pos = pos_array - np.array(ref_pos, dtype=np.float32) # 世界系相对位移

    # 构造2D旋转矩阵（与extract_ego_patch保持一致：旋转角=90+direction）
    theta = np.radians(90.0 + ref_heading)
    cos_a, sin_a = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[cos_a, sin_a], [-sin_a, cos_a]], dtype=np.float32)

    # 矩阵乘法 - 一次性完成所有位置的旋转变换
    # [dx_ego]   [cos  sin ] [dx_world]
    # [dy_ego] = [-sin cos ] [dy_world]
    ego_pos = rel_pos @ rotation_matrix.T  # (N,2) @ (2,2) = (N,2)
    ego_norm = ego_pos / np.array(map_size, dtype=np.float32) # 归一化到[-1, 1]
    return ego_norm


def _angles_to_sincos(angles: List[float], ref_angle: float) -> np.ndarray:
    """相对角度的sin/cos编码（°跳变）， angles: 角度序列（度），ref_angle: 参考角度（当前朝向，度）， 循环向量都用三角函数来表示，防止跳变"""
    delta_rad = np.radians(np.array(angles, dtype=np.float32) - ref_angle) # 向量化计算: 角度差 → 弧度 → sin/cos
    return np.stack([np.sin(delta_rad), np.cos(delta_rad)], axis=1) # 堆叠sin和cos为(N, 2)数组