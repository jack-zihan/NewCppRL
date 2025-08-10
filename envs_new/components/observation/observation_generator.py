"""
观察生成系统，支持第一人称视角和多尺度观察。
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Any, Union

from envs_new.components.config.environment_config import EnvironmentConfig
from envs_new.components.entity.agent import Agent
from envs_new.utils.image_utils import extract_ego_patch, stack_maps, apply_noise_to_pose


class ObservationGenerator:
    """统一的观察生成器"""
    
    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.rng: np.random.Generator = None
        
        # 预验证多尺度配置
        if config.use_multiscale:
            self._validate_multiscale_config()
    
    def set_random_generator(self, rng: np.random.Generator) -> None:
        self.rng = rng
    
    def _validate_multiscale_config(self) -> None:
        """预验证多尺度配置的有效性"""
        # 经过3次池化后的最小尺寸（第4次不需要裁剪中心）
        min_size_after_pooling = self.config.state_downsize[0] // (2**3)
        
        if min_size_after_pooling < self.config.multiscale_feature_size:
            raise ValueError(
                f"多尺度配置无效：state_downsize={self.config.state_downsize[0]} "
                f"经过3次池化后尺寸为{min_size_after_pooling}，"
                f"小于feature_size={self.config.multiscale_feature_size}"
            )
    
    def generate_observation(self, agent: Agent, maps_input: Union[Dict[str, Dict[str, Any]], Dict[str, np.ndarray]]) -> np.ndarray:
        """
        生成观察
        
        Args:
            agent: 智能体
            maps_input: 地图输入，可以是：
                       - Dict[str, Dict[str, Any]]: 包含map和pad信息的字典（旧格式）
                       - Dict[str, np.ndarray]: 直接的地图数组字典（新格式）
            
        Returns:
            观察数组，格式为 (C, H, W)
        """
        # 1. 预处理地图
        maps_dict = self._preprocess_maps(maps_input)
        stacked_maps, pad_values = stack_maps(maps_dict)
        
        # 2. 生成基础ego-centric观测（包含噪声）
        base_observation = self._extract_base_observation(agent, stacked_maps, pad_values)
        
        # 3. 根据配置进行不同的处理
        if self.config.use_multiscale:
            return self._apply_multiscale_transform(base_observation, agent, stacked_maps, pad_values)
        else:
            return base_observation
    
    def _preprocess_maps(self, maps_input: Union[Dict[str, Dict[str, Any]], Dict[str, np.ndarray]]) -> Dict[str, Dict[str, Any]]:
        """预处理地图输入，确保格式统一"""
        # 兼容两种输入格式
        if maps_input and isinstance(next(iter(maps_input.values())), dict):
            return maps_input
        else:
            return {
                name: {'map': map_array, 'pad': 0.0}
                for name, map_array in maps_input.items()
            }
    
    def _extract_base_observation(self, agent: Agent, stacked_maps: np.ndarray, 
                                pad_values: List[float]) -> np.ndarray:
        """提取基础的ego-centric观测，统一处理噪声和基础变换"""
        # 应用位置和方向噪声
        noisy_y, noisy_x, noisy_direction = apply_noise_to_pose(
            agent.y, agent.x, agent.direction,
            self.config.position_noise, self.config.direction_noise,
            self.rng or np.random.default_rng()
        )
        
        # 提取ego-centric patch
        ego_observation = extract_ego_patch(
            maps=stacked_maps,
            pad_values=pad_values,
            center_y=noisy_y,
            center_x=noisy_x,
            direction_deg=noisy_direction,
            patch_size=self.config.state_size
        )
        
        # Resize到目标尺寸（如果需要）
        if self.config.state_downsize != self.config.state_size:
            ego_observation = cv2.resize(
                ego_observation,
                (self.config.state_downsize[1], self.config.state_downsize[0]),
                interpolation=cv2.INTER_NEAREST
            )
        
        # 转换为 (C, H, W) 格式
        return ego_observation.transpose(2, 0, 1).astype(np.float32)
    
    def _apply_multiscale_transform(self, base_observation: np.ndarray, agent: Agent,
                                  stacked_maps: np.ndarray, pad_values: List[float]) -> np.ndarray:
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
        feature_size = self.config.multiscale_feature_size
        
        with torch.no_grad():
            # 生成4个尺度
            for scale in range(4):
                if scale == 0:
                    # 第一个尺度：直接从原始观测中心裁剪
                    center = self.config.state_downsize[0] // 2
                    half_size = feature_size // 2
                    cropped = obs_current[:, 
                                        center-half_size:center+half_size,
                                        center-half_size:center+half_size]
                else:
                    # 后续尺度：先池化，再从中心裁剪
                    obs_current = F.max_pool2d(
                        torch.from_numpy(obs_current), (2, 2), 2
                    ).numpy()
                    
                    current_size = obs_current.shape[1]
                    center = current_size // 2
                    half_size = min(feature_size // 2, center)
                    
                    # 由于已经预验证，这里不需要边界检查
                    cropped = obs_current[:, 
                                        center-half_size:center+half_size,
                                        center-half_size:center+half_size]
                    
                    # 如果最后一个尺度过小，需要resize
                    if scale == 3 and cropped.shape[1] < feature_size:
                        cropped_tensor = torch.from_numpy(cropped).unsqueeze(0)
                        resized = F.interpolate(
                            cropped_tensor, 
                            size=(feature_size, feature_size), 
                            mode='nearest'
                        )
                        cropped = resized.squeeze(0).numpy()
                
                obs_list.append(cropped)
            
            # 如果启用，添加全局观察
            if self.config.use_global_features:
                # 全局观测需要包含噪声，复用_extract_base_observation的噪声计算
                noisy_y, noisy_x, noisy_direction = apply_noise_to_pose(
                    agent.y, agent.x, agent.direction,
                    self.config.position_noise, self.config.direction_noise,
                    self.rng or np.random.default_rng()
                )
                
                # 使用完整地图尺寸提取全局视角
                global_observation = extract_ego_patch(
                    maps=stacked_maps,
                    pad_values=pad_values,
                    center_y=noisy_y,
                    center_x=noisy_x,
                    direction_deg=noisy_direction,
                    patch_size=(stacked_maps.shape[0], stacked_maps.shape[1])
                )
                
                # Resize到feature_size
                global_resized = cv2.resize(
                    global_observation,
                    (feature_size, feature_size),
                    interpolation=cv2.INTER_AREA
                )
                
                # 转换为 (C, H, W) 格式
                obs_global = global_resized.transpose(2, 0, 1)
                obs_list.append(obs_global)
        
        # 连接所有尺度
        return np.concatenate(obs_list, axis=0, dtype=np.float32)
    
    
    def get_observation_shape(self, num_map_channels: int) -> Tuple[int, int, int]:
        """
        获取期望的观察形状 (C, H, W)
        
        Args:
            num_map_channels: 地图通道数
        """
        if self.config.use_multiscale:
            # 多尺度观察总是使用4个尺度
            multiscale_channels = num_map_channels * 4
            
            if self.config.use_global_features:
                total_channels = multiscale_channels + num_map_channels
            else:
                total_channels = multiscale_channels
            
            return (total_channels, self.config.multiscale_feature_size, self.config.multiscale_feature_size)
        else:
            # 标准第一人称观察
            return (num_map_channels, self.config.state_downsize[0], self.config.state_downsize[1])