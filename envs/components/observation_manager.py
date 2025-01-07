# observation_manager.py

from __future__ import annotations

import math
from typing import Sequence, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from envs.components.utils import MowerAgent


class ObservationManager:
    """
    负责根据机器人位置与方向，对地图进行裁剪、旋转，生成观测。
    """

    def __init__(
        self,
        state_size: Tuple[int, int] = (128, 128),
        state_downsize: Tuple[int, int] = (128, 128),
        vision_length: int = 28,
        vision_angle: float = 75.0,
        use_multiscale: bool = True,
        use_global_features: bool = True,
        global_feature_size: int = 16,
        position_noise: float = 0.0,
        direction_noise: float = 0.0,
        rng: Optional[np.random.Generator] = None
    ):
        """
        初始化观测管理器。

        :param state_size: 观测图像的尺寸 (宽, 高)。
        :param state_downsize: 缩小后的观测图像尺寸 (宽, 高)。
        :param vision_length: 机器人的视野长度。
        :param vision_angle: 机器人的视野角度。
        :param use_multiscale: 是否使用多尺度特征。
        :param use_global_features: 是否使用全局特征。
        :param global_feature_size: 全局特征的尺寸。
        :param position_noise: 位置噪声强度。
        :param direction_noise: 方向噪声强度。
        :param rng: 随机数生成器。
        """
        self.state_size = state_size
        self.state_downsize = state_downsize
        self.vision_length = vision_length
        self.vision_angle = vision_angle
        self.use_multiscale = use_multiscale
        self.use_global_features = use_global_features
        self.global_feature_size = global_feature_size

        self.position_noise = position_noise
        self.direction_noise = direction_noise
        self.rng = rng if rng else np.random.default_rng()

    def generate_observation( # TODO: 以后扩展任意地图的concat一起，提高可扩展性
        self,
        agent: MowerAgent,
        field_frontier_map: np.ndarray,
        mist_map: np.ndarray,
        weed_map: np.ndarray,
        obstacle_map: np.ndarray,
        trajectory_map: np.ndarray
    ) -> np.ndarray:
        """
        生成观测，包括局部视野和（可选）全局特征。

        :param agent: 机器人对象。
        :param field_frontier_map: 农田前沿地图。
        :param mist_map: 迷雾地图。
        :param weed_map: 当前杂草地图。
        :param obstacle_map: 障碍物地图。
        :param trajectory_map: 轨迹地图。
        :return: 生成的观测图像。
        """
        # 将各地图层叠加为多通道图像
        # TODO: 这里现在逻辑需要修改，要暴露接口给不同环境使用，或者变成一个对外参数
        maps = np.stack(
            [
                field_frontier_map,
                mist_map,
                weed_map,
                obstacle_map,
                trajectory_map
            ],
            axis=-1
        )

        # 定义填充值（假设背景为0）
        pad_value = [0] * maps.shape[-1]

        # 提取局部观测
        local_observation = self._extract_local_observation(maps, pad_value, agent)

        # 缩放到 state_downsize
        resized_observation = cv2.resize(local_observation, self.state_downsize, interpolation=cv2.INTER_NEAREST)

        # 转置为 (C, H, W) 格式
        obs = resized_observation.transpose(2, 0, 1)

        if self.use_multiscale:
            # 生成多尺度特征
            obs = self._generate_multiscale_features(obs, maps, pad_value, agent)

        return obs.astype(np.float32)

    def _extract_local_observation( # TDOO: 没看到对observation和vector的记录，等待后续envbase找一找
        self,
        maps: np.ndarray,
        pad_value: Sequence[float],
        agent: MowerAgent
    ) -> np.ndarray:
        """
        根据机器人位置和方向，从 maps 中裁剪出局部视野并旋转对正。

        :param maps: 多通道地图数据。
        :param pad_value: 边界填充的值。
        :param agent: 机器人对象。
        :return: 裁剪并旋转后的局部观测图像。
        """
        agent_y, agent_x = agent.y, agent.x
        agent_direction = agent.direction

        # 添加位置噪声
        if self.position_noise > 0.0:
            delta_y = self.rng.normal(0, self.position_noise)
            delta_x = self.rng.normal(0, self.position_noise)
            agent_y += np.clip(delta_y, -self.position_noise, self.position_noise)
            agent_x += np.clip(delta_x, -self.position_noise, self.position_noise)

        # 添加方向噪声
        if self.direction_noise > 0.0:
            delta_dir = self.rng.normal(0, self.direction_noise)
            agent_direction = (agent_direction + np.clip(delta_dir, -self.direction_noise, self.direction_noise)) % 360

        diag_r = self.state_size[0] / 2 * np.sqrt(2)
        diag_r_int = np.ceil(diag_r).astype(np.int32)

        # 使用 cv2.copyMakeBorder 函数在 maps 图像的边界周围添加边框
        # 边框的宽度由 diag_r_int 决定，填充值由 pad_value 指定
        padded_maps = cv2.copyMakeBorder(
            maps,
            diag_r_int, diag_r_int, diag_r_int, diag_r_int,
            cv2.BORDER_CONSTANT,
            value=pad_value
        )

        # 计算裁剪坐标
        left = int(round(agent_y))
        right = left + 2 * diag_r_int
        up = int(round(agent_x))
        down = up + 2 * diag_r_int

        # 裁剪地图
        cropped_maps = padded_maps[left:right, up:down]

        # 旋转裁剪后的地图以对齐机器人的方向
        rotation_matrix = cv2.getRotationMatrix2D((diag_r, diag_r), 180 + agent_direction, 1.0)
        rotated_maps = cv2.warpAffine(
            cropped_maps.astype(np.float32),
            rotation_matrix,
            (2 * diag_r_int, 2 * diag_r_int),
            flags=cv2.INTER_NEAREST
        )

        # 截取为 state_size 大小
        delta_l = diag_r_int - self.state_size[0] // 2
        delta_r = delta_l + self.state_size[0]
        final_observation = rotated_maps[delta_l:delta_r, delta_l:delta_r]

        # 确保观测图像有通道维度
        if final_observation.ndim == 2:
            final_observation = final_observation[..., np.newaxis]

        return final_observation

    def _generate_multiscale_features( # TODO: 这个函数还没有完全弄清楚，等待再学习
        self,
        local_obs: np.ndarray,
        maps: np.ndarray,
        pad_value: Sequence[float],
        agent: MowerAgent
    ) -> np.ndarray:
        """
        生成多尺度特征和全局特征。

        :param local_obs: 局部观测图像。
        :param maps: 多通道地图数据。
        :param pad_value: 边界填充的值。
        :param agent: 机器人对象。
        :return: 包含多尺度特征和全局特征的观测图像。
        """
        feature_list = [local_obs]
        temp = torch.from_numpy(local_obs).float()

        # 多尺度池化
        for _ in range(4):
            temp = F.max_pool2d(temp, kernel_size=2, stride=2)
            feature_list.append(temp.numpy())

        if self.use_global_features:
            global_feature = self._extract_global_observation(maps, pad_value, agent)
            feature_list.append(global_feature)

        # 按通道维度连接
        return np.concatenate(feature_list, axis=0)

    def _extract_global_observation(
        self,
        maps: np.ndarray,
        pad_value: Sequence[float],
        agent: MowerAgent
    ) -> np.ndarray:
        """
        生成全局特征，旋转对正后下采样到指定尺寸。

        :param maps: 多通道地图数据。
        :param pad_value: 边界填充的值。
        :param agent: 机器人对象。
        :return: 全局特征图像。
        """
        diag_r = self.state_size[0] / 2 * math.sqrt(2)
        diag_r_int = int(math.ceil(diag_r))

        agent_y, agent_x = agent.y, agent.x
        agent_direction = agent.direction

        # 添加位置噪声
        if self.position_noise > 0.0:
            delta_y = self.rng.normal(0, self.position_noise)
            delta_x = self.rng.normal(0, self.position_noise)
            agent_y += np.clip(delta_y, -self.position_noise, self.position_noise)
            agent_x += np.clip(delta_x, -self.position_noise, self.position_noise)

        # 添加方向噪声
        if self.direction_noise > 0.0:
            delta_dir = self.rng.normal(0, self.direction_noise)
            agent_direction = (agent_direction + np.clip(delta_dir, -self.direction_noise, self.direction_noise)) % 360

        # 边界填充
        padded_maps = cv2.copyMakeBorder(
            maps,
            diag_r_int, diag_r_int, diag_r_int, diag_r_int,
            cv2.BORDER_CONSTANT,
            value=pad_value
        )

        # 计算裁剪坐标
        left = int(round(agent_y))
        right = left + 2 * diag_r_int
        up = int(round(agent_x))
        down = up + 2 * diag_r_int

        # 裁剪地图
        cropped_maps = padded_maps[left:right, up:down]

        # 旋转裁剪后的地图以对齐机器人的方向
        rotation_matrix = cv2.getRotationMatrix2D((diag_r, diag_r), 180 + agent_direction, 1.0)
        rotated_maps = cv2.warpAffine(
            cropped_maps.astype(np.float32),
            rotation_matrix,
            (2 * diag_r_int, 2 * diag_r_int),
            flags=cv2.INTER_NEAREST
        )

        # 截取为原大小
        delta_l = diag_r_int - maps.shape[1] // 2
        delta_r = delta_l + maps.shape[1]
        final_global_obs = rotated_maps[delta_l:delta_r, delta_l:delta_r]

        # 确保观测图像有通道维度
        if final_global_obs.ndim == 2:
            final_global_obs = final_global_obs[..., np.newaxis]

        # 下采样到 global_feature_size
        resized_global_obs = cv2.resize(final_global_obs, (self.global_feature_size, self.global_feature_size), interpolation=cv2.INTER_NEAREST)
        resized_global_obs = resized_global_obs.transpose(2, 0, 1)  # (C, H, W)

        return resized_global_obs
