"""
数学工具函数模块
"""
from __future__ import annotations

import numpy as np


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