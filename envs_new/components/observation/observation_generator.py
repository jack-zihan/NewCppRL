"""
观察生成系统，支持第一人称视角和多尺度观察。
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Any, Union, List

from envs_new.components.config.environment_config import EnvironmentConfig
from envs_new.components.entity.agent import Agent
from envs_new.utils.image_utils import extract_ego_patch, stack_maps, apply_noise_to_pose


class ObservationGenerator:
    """统一的观察生成器"""

    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.rng: np.random.Generator = None

        if config.use_multiscale:
            self._validate_multiscale_config() # 预验证多尺度配置

    def generate_observation(self, agent: Agent,
                             maps_dict: Union[Dict[str, Dict[str, Any]], Dict[str, np.ndarray]], noisy_pose=None) -> np.ndarray:
        """
        生成观察；
        Args: agent: 智能体；maps_input: 地图输入（包含map和pad信息的字典）
        Returns: 观察数组，格式为 (C, H, W)
        """
        # 1. 预处理地图
        stacked_maps, pad_values = stack_maps(maps_dict)

        # 2. 应用位置和方向噪声
        noisy_pose = noisy_pose if noisy_pose is not None else apply_noise_to_pose(agent.y, agent.x, agent.direction,
                                                               self.config.position_noise, self.config.direction_noise,
                                                               self.rng)
        # 3. 生成基础ego-centric观测
        base_observation = self._extract_base_observation(noisy_pose, stacked_maps, pad_values)

        # 4. 根据配置进行不同的处理
        return self._apply_multiscale_transform(base_observation) if self.config.use_multiscale else base_observation


    def _extract_base_observation(self, pose, stacked_maps: np.ndarray, pad_values: List[float]) -> np.ndarray:
        """提取基础的ego-centric观测，统一处理噪声和基础变换"""
        # 提取位置和方向噪声
        y, x, direction = pose

        # 提取ego-centric patch
        ego_observation = extract_ego_patch(maps=stacked_maps, pad_values=pad_values, patch_size=self.config.state_size,
                                            center_y=y, center_x=x, direction_deg=direction, )
        # Resize到目标尺寸（如果需要）
        if self.config.state_downsize != self.config.state_size:
            ego_observation = cv2.resize(ego_observation,
                (self.config.state_downsize[1], self.config.state_downsize[0]),interpolation=cv2.INTER_NEAREST)

        return ego_observation.transpose(2, 0, 1).astype(np.float32) # 转换为 (C, H, W) 格式

    def _apply_multiscale_transform(self, base_observation: np.ndarray) -> np.ndarray:
        """
        在基础观测上应用多尺度变换（SGCNN风格）
        
        设计理念：近处精细、远处粗糙的视觉系统
        - 第1层：高分辨率局部细节（用于精确控制）
        - 第2-4层：逐渐扩大的感受野（捕获更远的障碍物和目标）
        - 全局层（可选）：整体环境认知
        使用max_pool2d确保远处的关键特征（如单个杂草）不会丢失
        """
        obs_list = []
        obs_current = base_observation.copy()
        center_size = self.config.state_downsize[0] // 2
        feature_size = self.config.multiscale_feature_size

        with torch.no_grad():
            # 生成 n_scales 个尺度
            for _ in range(self.config.n_scales):
                # 从中心裁剪
                half_size = feature_size // 2
                cropped = obs_current[:,
                          center_size - half_size:center_size + half_size,
                          center_size - half_size:center_size + half_size]
                obs_list.append(cropped)

                # 池化准备下一层
                obs_current = F.max_pool2d(torch.from_numpy(obs_current), (2, 2), 2).numpy()
                center_size //= 2

            # 如果启用，添加全局观察
            if self.config.use_global_features:
                # 计算需要的池化核大小以达到目标尺寸
                kernel_size = int(np.round(self.config.state_downsize[0] / feature_size))
                assert kernel_size >= 1, f"Invalid kernel_size: {kernel_size}" # 这些检查应该在初始化时就通过了，这里用断言确保

                # 使用max_pool2d保持与其他层的一致性, 复用base_observation，它已经包含了正确的噪声和旋转
                obs_global = F.max_pool2d(torch.from_numpy(base_observation),
                                          (kernel_size, kernel_size),kernel_size).numpy()
                obs_list.append(obs_global)

        return np.concatenate(obs_list, axis=0, dtype=np.float32) # 连接所有尺度

    def get_observation_shape(self, num_map_channels: int) -> Tuple[int, int, int]:
        """获取期望的观察形状 (C, H, W); Args: num_map_channels: 地图通道数   """
        if self.config.use_multiscale:
            # 多尺度观察使用 n_scales 个尺度
            multiscale_channels = num_map_channels * self.config.n_scales
            total_channels = multiscale_channels + num_map_channels if self.config.use_global_features else multiscale_channels

            return (total_channels, self.config.multiscale_feature_size, self.config.multiscale_feature_size)
        else:
            # 标准第一人称观察
            return (num_map_channels, self.config.state_downsize[0], self.config.state_downsize[1])

    def set_random_generator(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def _validate_multiscale_config(self) -> None:
        """预验证多尺度配置的有效性"""
        # 验证多尺度池化配置, 经过 n_scales-1 次池化后的最小尺寸
        n_pooling = self.config.n_scales - 1
        min_size_after_pooling = self.config.state_downsize[0] // (2 ** n_pooling)

        if min_size_after_pooling < self.config.multiscale_feature_size:
            raise ValueError(f"多尺度配置无效：state_downsize={self.config.state_downsize[0]} "
                             f"经过{n_pooling}次池化后尺寸为{min_size_after_pooling}，" f"小于feature_size={self.config.multiscale_feature_size}")

        # 验证全局观察池化配置（如果启用）
        if self.config.use_global_features:
            current_size, feature_size = self.config.state_downsize[0], self.config.multiscale_feature_size
            kernel_size = int(np.round(current_size / feature_size))

            if kernel_size < 1:
                raise ValueError(f"全局观察配置无效：state_downsize={current_size} " f"太小，"
                                 f"无法池化到feature_size={feature_size}")

            # 计算池化后的实际尺寸
            pooled_size = current_size // kernel_size

            # 验证池化后尺寸是否精确匹配
            if pooled_size != feature_size:
                raise ValueError(f"全局观察池化尺寸不匹配：state_downsize={current_size} "
                    f"使用kernel_size={kernel_size}池化后得到{pooled_size}，"f"不等于期望的feature_size={feature_size}。"
                    f"请调整state_downsize或feature_size使其能够整除")
