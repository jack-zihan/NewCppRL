from __future__ import annotations

import math

import numpy as np
from typing import Dict, Any, Tuple, Optional

def total_variation(images: np.ndarray) -> int:
    """https://github.com/tensorflow/tensorflow/blob/v2.7.0/tensorflow/python/ops/image_ops_impl.py#L3213-L3282"""
    pixel_dif1 = images[1:, :] - images[:-1, :]
    pixel_dif2 = images[:, 1:] - images[:, :-1]
    tot_var = np.abs(pixel_dif1).sum() + np.abs(pixel_dif2).sum()
    return tot_var


def total_variation_mat(images: np.ndarray) -> np.ndarray:
    mask_tv_cols = images[1:, :] - images[:-1, :] != 0
    mask_tv_cols = np.pad(mask_tv_cols, pad_width=[[0, 1], [0, 0]], mode='constant')
    mask_tv_rows = images[:, 1:] - images[:, :-1] != 0
    mask_tv_rows = np.pad(mask_tv_rows, pad_width=[[0, 0], [0, 1]], mode='constant')
    mask_tv = np.logical_or(mask_tv_rows, mask_tv_cols)
    return mask_tv


def get_map_pasture_larger(map_pasture: np.ndarray):
    map_pasture_larger = map_pasture.copy()
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for shift in shifts:
        map_pasture_larger = np.logical_or(
            map_pasture_larger,
            np.roll(map_pasture, shift, axis=(0, 1))
        )
    return map_pasture_larger

def apply_mask_with_color(base_map: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int],alpha: float = 1.0) -> np.ndarray:
    """
    将掩码为True的区域上色到 color。支持alpha混合：
      - alpha=1.0 时，直接替换为 color
      - alpha<1.0 时，采用 alpha*color + (1-alpha)*base_map 进行混合
    """
    mask_3d = np.expand_dims(mask, axis=-1)  # (H, W, 1)
    if alpha >= 1.0:
        return np.where(mask_3d, color, base_map)
    else:
        color_arr = np.array(color, dtype=np.float32)
        blended_region = ((1 - alpha) * base_map.astype(np.float32) + alpha * color_arr).astype(np.uint8)
        return np.where(mask_3d, blended_region, base_map).astype(np.uint8)

class NumericalRange:
    """
    用于表示一个有上下限并具有“mode”(即可由上下限计算出的有效范围)的数值区间。
    """
    def __init__(self, min: float, max: float):
        self.min = min
        self.max = max
    @property
    def mode(self) -> float:
        return self.max - self.min

class MowerAgent:
    # 这种放在前面的是类变量，各个类是相通的
    width = 4
    length = 6
    occupancy = math.hypot(width, length)
    lw_ratio = math.degrees(math.atan2(width / occupancy, length / occupancy))

    vision_length = 28.0
    vision_angle  = 75.0

    def __init__(
        self,
        position: tuple[float, float] = (0.0, 0.0),
        direction: float = 0.0,
    ):
        self.x, self.y = position #
        self.direction = direction % 360.0  # 确保方向在 0-360 度之间
        self.last_speed: float = 0.0
        self.last_steer: float = 0.0

    @property
    def position(self):
        return self.x, self.y

    @property
    def position_discrete(self):
        return round(self.x), round(self.y)

    @property
    def convex_hull(self):
        return np.array([
            (self.x + 1.0 * self.width * math.cos(math.radians(self.direction + 0 + self.lw_ratio)),
             self.y + 1.0 * self.width * math.sin(math.radians(self.direction + 0 + self.lw_ratio))),
            (self.x + 1.0 * self.width * math.cos(math.radians(self.direction + 180 - self.lw_ratio)),
             self.y + 1.0 * self.width * math.sin(math.radians(self.direction + 180 - self.lw_ratio))),
            (self.x + 1.0 * self.width * math.cos(math.radians(self.direction + 180 + self.lw_ratio)),
             self.y + 1.0 * self.width * math.sin(math.radians(self.direction + 180 + self.lw_ratio))),
            (self.x + 1.0 * self.width * math.cos(math.radians(self.direction + 0 - self.lw_ratio)),
             self.y + 1.0 * self.width * math.sin(math.radians(self.direction + 0 - self.lw_ratio))),
        ])

    def control(self, speed: float, steer: float):
        """
        根据线速度和角速度以及动力学模型，计算新的x,y
        """
        self.last_speed, self.last_steer = speed, steer
        self.direction = (self.direction + steer) % 360
        dx = speed * math.cos(math.radians(self.direction))
        dy = speed * math.sin(math.radians(self.direction))
        self.x += dx
        self.y += dy

    def reset(self, position: tuple[float, float], direction: float):
        self.x, self.y = position
        self.direction = direction % 360.0
        self.last_speed, self.last_steer = 0.0, 0.0


class RealAgent(MowerAgent):
    def control(self, new_position: tuple[float, float], new_direction: float):
        """
        直接设置机器人的新位置和方向。
        """
        self.x, self.y = new_position
        self.direction = new_direction % 360  # 确保方向在 0-360 度之间
        self.last_acc, self.last_steer = 0., 0.
