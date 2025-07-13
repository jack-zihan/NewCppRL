"""
Utility functions for environment components.
"""
from __future__ import annotations

import math
import numpy as np
from typing import Dict, Any, Tuple, Optional


def total_variation(images: np.ndarray) -> int:
    """
    Calculate total variation of images.
    https://github.com/tensorflow/tensorflow/blob/v2.7.0/tensorflow/python/ops/image_ops_impl.py#L3213-L3282
    """
    pixel_dif1 = images[1:, :] - images[:-1, :]
    pixel_dif2 = images[:, 1:] - images[:, :-1]
    tot_var = np.abs(pixel_dif1).sum() + np.abs(pixel_dif2).sum()
    return tot_var


def total_variation_mat(mat: np.ndarray) -> np.ndarray:
    """
    Calculate total variation matrix indicating where pixels differ from neighbors.
    
    Args:
        mat: Input matrix
        
    Returns:
        Boolean matrix indicating positions where values differ from neighbors
    """
    tv = np.zeros_like(mat, dtype=bool)
    tv[1:, :] |= (mat[1:, :] != mat[:-1, :])
    tv[:-1, :] |= (mat[1:, :] != mat[:-1, :])
    tv[:, 1:] |= (mat[:, 1:] != mat[:, :-1])
    tv[:, :-1] |= (mat[:, 1:] != mat[:, :-1])
    return tv