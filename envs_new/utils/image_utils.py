"""
共享图像处理工具
"""
import math
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any


def enlarge_map_features(map_data: np.ndarray) -> np.ndarray:
    """将地图特征向所有方向扩大一个像素"""
    map_larger = map_data.copy()
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for shift in shifts:
        map_larger = np.logical_or(
            map_larger,
            np.roll(map_data, shift, axis=(0, 1))
        )
    return map_larger


def apply_channel_padding(image: np.ndarray, pad_values: List[float], 
                         pad_length: int) -> np.ndarray:
    """为每个通道应用不同的填充值"""
    height, width, channels = image.shape
    padded_channels = []
    
    for channel_idx in range(channels):
        channel = image[..., channel_idx]
        pad_value = pad_values[channel_idx]
        padded_channel = np.pad(
            channel,
            pad_width=((pad_length, pad_length), (pad_length, pad_length)),
            mode='constant',
            constant_values=pad_value
        )
        padded_channels.append(padded_channel)
    
    return np.stack(padded_channels, axis=-1)


def extract_ego_patch(maps: np.ndarray, pad_values: List[float],
                     center_y: float, center_x: float, direction_deg: float,
                     patch_size: Tuple[int, int]) -> np.ndarray:
    """
    提取以智能体为中心的自我中心观察补丁
    
    Args:
        maps: 堆叠的地图 (H, W, C)
        pad_values: 每个通道的填充值
        center_y: 中心Y坐标
        center_x: 中心X坐标
        direction_deg: 方向角度
        patch_size: 输出补丁大小 (高度, 宽度)
        
    Returns:
        旋转和裁剪后的观察补丁
    """
    patch_height, patch_width = patch_size
    
    # 计算旋转所需的对角线填充长度
    diagonal_length = math.ceil(max(patch_height, patch_width) / 2 * math.sqrt(2))
    
    # 应用填充
    padded_maps = apply_channel_padding(maps, pad_values, diagonal_length)
    
    # 调整中心坐标以适应填充
    center_y_padded = center_y + diagonal_length
    center_x_padded = center_x + diagonal_length
    
    # 围绕中心裁剪正方形区域
    top = int(round(center_y_padded - diagonal_length))
    bottom = int(round(center_y_padded + diagonal_length))
    left = int(round(center_x_padded - diagonal_length))
    right = int(round(center_x_padded + diagonal_length))
    
    cropped_maps = padded_maps[top:bottom, left:right, :]
    
    if cropped_maps.size == 0:
        raise ValueError("裁剪区域为空")
    
    # 旋转以使智能体方向向上对齐
    # 180度是因为要将agent的前进方向（默认向右0度）旋转到图像上方
    rotation_angle = 180 + direction_deg
    rotation_center = (diagonal_length, diagonal_length)
    rotation_matrix = cv2.getRotationMatrix2D(rotation_center, rotation_angle, 1.0)
    
    rotated_maps = cv2.warpAffine(
        cropped_maps,
        rotation_matrix,
        (cropped_maps.shape[1], cropped_maps.shape[0])
    )
    
    # 确保为3D
    if rotated_maps.ndim == 2:
        rotated_maps = rotated_maps[..., np.newaxis]
    
    # 最终裁剪到所需的补丁大小
    rotated_height, rotated_width = rotated_maps.shape[:2]
    start_y = max(0, (rotated_height - patch_height) // 2)
    start_x = max(0, (rotated_width - patch_width) // 2)
    
    final_patch = rotated_maps[start_y:start_y + patch_height, 
                             start_x:start_x + patch_width, :]
    
    return final_patch


def stack_maps(maps_dict: Dict[str, Dict[str, Any]]) -> Tuple[np.ndarray, List[float]]:
    """
    将字典中的地图堆叠成单个数组
    Returns:
        (堆叠的地图, 填充值) 元组
    """
    if not maps_dict:
        raise ValueError("地图字典为空")
    
    # 获取第一个地图以确定形状
    first_key = next(iter(maps_dict))
    base_shape = maps_dict[first_key]["map"].shape
    
    channels = []
    pad_values = []
    
    for map_name, map_info in maps_dict.items():
        map_array = map_info["map"]
        pad_value = map_info.get("pad", 0.0)
        
        if map_array.shape != base_shape:
            raise ValueError(f"地图 {map_name} 形状不一致: "
                           f"{map_array.shape} vs {base_shape}")
        
        channels.append(map_array)
        pad_values.append(pad_value)
    
    stacked_maps = np.stack(channels, axis=-1)
    return stacked_maps, pad_values


def apply_noise_to_pose(y: float, x: float, direction: float,
                       position_noise: float, direction_noise: float,
                       rng: np.random.Generator) -> Tuple[float, float, float]:
    """对智能体姿态应用噪声"""
    if position_noise > 0:
        y_noise = rng.normal(0, position_noise)
        x_noise = rng.normal(0, position_noise)
        y += np.clip(y_noise, -position_noise, position_noise)
        x += np.clip(x_noise, -position_noise, position_noise)
    
    if direction_noise > 0:
        dir_noise = rng.normal(0, direction_noise)
        direction = (direction + np.clip(dir_noise, -direction_noise, direction_noise)) % 360
    
    return y, x, direction