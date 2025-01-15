# observation_manager.py

from __future__ import annotations

import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Sequence, Tuple, Optional, Dict, Any

from envs.components.utils import MowerAgent


# TODO: 未来可以优化多尺度的可扩展性，使得变得很好用
class ObservationManager:
    def __init__(
            self,
            state_size: Tuple[int, int] = (128, 128), state_downsize: Tuple[int, int] = (128, 128),
            use_multiscale: bool = True, n_scales: int = 4, multiscale_feature_size: int = 16,
            use_global_features: bool = True,
            position_noise: float = 0.0, direction_noise: float = 0.0,
            rng: Optional[np.random.Generator] = None,
    ):
        """
        2D BEV Observation 提取器

        参数:
            state_size (Tuple[int, int]): 状态尺寸，默认为 (128, 128)。
            state_downsize (Tuple[int, int]): 状态下采样尺寸，默认为 (128, 128)。
            use_multiscale (bool): 是否使用多尺度特征，默认为 True。
            n_scales (int): 多尺度特征的尺度数量，默认为 4。
            multiscale_feature_size (int): 全局特征尺寸，默认为 16。
            use_global_features (bool): 是否使用全局特征，默认为 True。
            position_noise (float): 位置噪声，默认为 0.0。
            direction_noise (float): 方向噪声，默认为 0.0。
            rng (Optional[np.random.Generator]): 随机数生成器，默认为 None。
        """
        self.state_size = state_size
        self.state_downsize = state_downsize
        self.use_multiscale = use_multiscale
        self.n_scales = n_scales
        self.use_global_features = use_global_features
        self.multiscale_feature_size = multiscale_feature_size
        self.position_noise = position_noise
        self.direction_noise = direction_noise
        self.rng = rng

    def generate_observation(
            self,
            agent: MowerAgent,
            maps_dict: Dict[str, Dict[str, Any]]
            # eg. {"map1": {"map": map1, "pad": 0}, "map2": {"map": map2, "pad": 0}}
    ) -> np.ndarray:
        """
         主入口：根据 agent 位姿和 maps_dict 生成观测图 (C,H,W)。
         逻辑概括：
           1) 将maps_dict堆叠 => (H,W,C)
           2) 对位姿注入噪声后 => 裁剪 & 旋转 => 得到 (state_size)
           3) 再resize到 state_downsize
           4) 如果 use_multiscale => multi-scale pooling; 如果 use_global_features => 提取全局特征 => concat
           5) 返回 float32 (C,H,W)
         """
        stacked_maps, pad_values = self.stack_maps(maps_dict)  # 堆叠地图: (H,W,C) & 其padding
        noisy_y, noisy_x, noisy_direction = self._get_noisy_pose(agent)  # 位姿注入噪声
        local_observation = self.extract_ego_observation(
            maps=stacked_maps, pad_values=pad_values,
            center_y=noisy_y, center_x=noisy_x, direction_deg=noisy_direction,
            patch_size=self.state_size
        )
        local_observation_downsampled = cv2.resize(
            local_observation,
            (self.state_downsize[1], self.state_downsize[0]),
            interpolation=cv2.INTER_NEAREST
        )
        local_patch_downsampled = local_observation_downsampled.transpose(2, 0, 1)

        if self.use_multiscale:
            local_features = self._generate_multiscale_features(local_patch_downsampled)
        else:
            local_features = local_patch_downsampled

        if self.use_global_features:  # TODO，这里现在可能有尺度不匹配的风险，要注意 确实有，这个地方要好好注意一下
            global_features = self._extract_global_patch(
                maps=stacked_maps, pad_values=pad_values,
                center_y=noisy_y, center_x=noisy_x, direction_deg=noisy_direction
            )
            final_observation = np.concatenate([local_features, global_features], axis=0)
        else:
            final_observation = local_features

        return final_observation.astype(np.float32)

    @staticmethod
    def stack_maps(maps_dict: Dict[str, Dict[str, Any]]) -> Tuple[np.ndarray, Sequence[float]]:
        first_key = next(iter(maps_dict))
        base_shape = maps_dict[first_key]["map"].shape

        channels = []
        pad_values = []

        for map_name, map_info in maps_dict.items():
            map_array = map_info["map"]
            pad_value = map_info.get("pad", 0)
            if map_array.shape != base_shape:
                raise ValueError(
                    f"地图 {map_name} 大小 {map_array.shape} 与其它地图不一致: {base_shape}。"
                )
            channels.append(map_array)
            pad_values.append(pad_value)

        stacked_maps = np.stack(channels, axis=-1)
        return stacked_maps, pad_values

    def _get_noisy_pose(self, agent: MowerAgent) -> Tuple[float, float, float]:
        y, x, direction = agent.y, agent.x, agent.direction
        if self.position_noise > 0:
            y_noise = self.rng.normal(0, self.position_noise)
            x_noise = self.rng.normal(0, self.position_noise)
            y += np.clip(y_noise, -self.position_noise, self.position_noise)
            x += np.clip(x_noise, -self.position_noise, self.position_noise)
        if self.direction_noise > 0:
            direction_noise = self.rng.normal(0, self.direction_noise)
            direction = (direction + np.clip(direction_noise, -self.direction_noise, self.direction_noise)) % 360
        return y, x, direction

    @staticmethod
    def _apply_padding_per_channel(image: np.ndarray, pad_values: Sequence[float], pad_lenth: int) -> np.ndarray:
        height, width, channels = image.shape
        padded_channels = []
        for channel_index in range(channels):
            channel = image[..., channel_index]
            pad_value = pad_values[channel_index]
            padded_channel = np.pad(channel, pad_width=((pad_lenth, pad_lenth), (pad_lenth, pad_lenth)),
                                    mode='constant', constant_values=pad_value)
            padded_channels.append(padded_channel)
        return np.stack(padded_channels, axis=-1)

    def extract_ego_observation(
            self, maps: np.ndarray, pad_values: Sequence[float],
            center_y: float, center_x: float, direction_deg: float, patch_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        提取以输入位姿为中心，输出朝向朝上的局部图像crop patch。
        """
        patch_height, patch_width = patch_size

        diagonal_length = math.ceil(max(patch_height, patch_width) / 2 * math.sqrt(2))
        padded_maps = self._apply_padding_per_channel(maps, pad_values, pad_lenth=diagonal_length)

        # corp出足够大的ego为中心的patch
        center_y_padded, center_x_padded = center_y + diagonal_length, center_x + diagonal_length
        top, bottom = int(round(center_y_padded - diagonal_length)), int(round(center_y_padded + diagonal_length))
        left, right = int(round(center_x_padded - diagonal_length)), int(round(center_x_padded + diagonal_length))
        cropped_maps = padded_maps[top:bottom, left:right, :]
        assert cropped_maps.size != 0

        # 旋转patch，使得agent方向朝上
        rotation_angle = 180 + direction_deg
        rotation_center = (diagonal_length, diagonal_length)
        rotation_matrix = cv2.getRotationMatrix2D(rotation_center, rotation_angle, 1.0)

        rotated_maps = cv2.warpAffine(
            cropped_maps,
            rotation_matrix,
            (cropped_maps.shape[1], cropped_maps.shape[0])
        )
        if rotated_maps.ndim == 2:
            rotated_maps = rotated_maps[..., np.newaxis]

        # crop出最终目标尺寸patch
        rotated_map_height, rotated_map_width, _ = rotated_maps.shape
        start_y = max(0, (rotated_map_height - patch_height) // 2)
        start_x = max(0, (rotated_map_width - patch_width) // 2)

        final_patch = rotated_maps[start_y:start_y + patch_height, start_x:start_x + patch_width, :]

        return final_patch

    # 和gpt详细讨论后，认可目前的实现
    def _generate_multiscale_features(self, observation: np.ndarray) -> np.ndarray:
        """
        将observation根据multiscale_feature_size进行多尺度特征提取。
        """
        channels, height, width = observation.shape

        observation_tensor = torch.from_numpy(observation).unsqueeze(0)  # (1, channels, height, width)
        scale_features = []

        for scale_index in range(self.n_scales):
            crop_length = (2 ** scale_index) * self.multiscale_feature_size
            crop_length = min(crop_length, height, width)

            # Center crop
            top, left = (height - crop_length) // 2, (width - crop_length) // 2
            bottom, right = top + crop_length, left + crop_length
            cropped = observation_tensor[:, :, top:bottom, left:right]

            # Max pooling to get multiscale size feature
            kernel_size = max(1, crop_length // self.multiscale_feature_size)
            pooled = F.max_pool2d(cropped, kernel_size=kernel_size, stride=kernel_size)

            scale_features.append(pooled)

        multiscale_features = torch.cat(scale_features, dim=1).squeeze(0)
        return multiscale_features.numpy()


    def _extract_global_patch(  # 现在的global feature没考虑非多尺度情况下的计算问题
            self, maps: np.ndarray, pad_values: Sequence[float],
            center_y: float, center_x: float, direction_deg: float
    ) -> np.ndarray:
        """
        提取全局特征, 且尺寸为map的尺寸，再resize到multiscale_feature_size。
        """
        multiscale_feature_size = self.multiscale_feature_size
        height, width, channels = maps.shape
        global_crop_size = max(height, width)

        large_patch = self.extract_ego_observation(
            maps=maps, pad_values=pad_values,
            center_y=center_y, center_x=center_x, direction_deg=direction_deg,
            patch_size=(global_crop_size, global_crop_size)
        )
        # Max pooling to get multiscale size feature
        large_patch_tensor = large_patch.transpose(2, 0, 1)[np.newaxis, ...]
        kernel_size = max(1, global_crop_size // multiscale_feature_size)
        pooled = F.max_pool2d(torch.from_numpy(large_patch_tensor), kernel_size=(kernel_size, kernel_size), stride=kernel_size)
        pooled_array = pooled.squeeze(0).numpy()
        # Crop the center multiscale_feature_size x multiscale_feature_size
        _, pooled_height, pooled_width = pooled_array.shape
        if pooled_height < self.multiscale_feature_size or pooled_width < self.multiscale_feature_size:
            raise ValueError(f"全局特征 {pooled_array.shape} 小于 {self.multiscale_feature_size}")
        start_y = max(0, (pooled_height - multiscale_feature_size) // 2)
        start_x = max(0, (pooled_width - multiscale_feature_size) // 2)
        final_global_features = pooled_array[:, start_y:start_y + multiscale_feature_size,
                                start_x:start_x + multiscale_feature_size]
        return final_global_features
