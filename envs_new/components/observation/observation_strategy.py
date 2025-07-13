"""
Observation strategies for the mowing robot environment.
Provides different observation generation approaches using strategy pattern.
"""
from __future__ import annotations

import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any

from envs_new.components.config.environment_config import ObservationConfig
from envs_new.components.entity.agent import Agent


class NoiseInjector:
    """Handles noise injection for position and direction."""
    
    def __init__(self, position_noise: float = 0.0, direction_noise: float = 0.0):
        """
        Initialize noise injector.
        
        Args:
            position_noise: Standard deviation of position noise
            direction_noise: Standard deviation of direction noise
        """
        self.position_noise = position_noise
        self.direction_noise = direction_noise
        self.rng: Optional[np.random.Generator] = None
    
    def set_random_generator(self, rng: np.random.Generator) -> None:
        """Set random number generator."""
        self.rng = rng
    
    def apply_noise(self, agent: Agent) -> Tuple[float, float, float]:
        """
        Apply noise to agent pose.
        
        Args:
            agent: Agent to get pose from
            
        Returns:
            Tuple of (noisy_y, noisy_x, noisy_direction)
        """
        y, x, direction = agent.y, agent.x, agent.direction
        
        if self.position_noise > 0 and self.rng is not None:
            y_noise = self.rng.normal(0, self.position_noise)
            x_noise = self.rng.normal(0, self.position_noise)
            y += np.clip(y_noise, -self.position_noise, self.position_noise)
            x += np.clip(x_noise, -self.position_noise, self.position_noise)
        
        if self.direction_noise > 0 and self.rng is not None:
            dir_noise = self.rng.normal(0, self.direction_noise)
            direction = (direction + np.clip(dir_noise, -self.direction_noise, self.direction_noise)) % 360
        
        return y, x, direction


class ObservationStrategy(ABC):
    """Abstract base class for observation generation strategies."""
    
    def __init__(self, config: ObservationConfig):
        """
        Initialize observation strategy.
        
        Args:
            config: Observation configuration
        """
        self.config = config
        self.noise_injector = NoiseInjector(config.position_noise, config.direction_noise)
    
    def set_random_generator(self, rng: np.random.Generator) -> None:
        """Set random number generator."""
        self.noise_injector.set_random_generator(rng)
    
    @abstractmethod
    def generate_observation(self, agent: Agent, maps_dict: Dict[str, Dict[str, Any]]) -> np.ndarray:
        """
        Generate observation from agent and maps.
        
        Args:
            agent: Agent to observe from
            maps_dict: Dictionary of maps with padding info
            
        Returns:
            Observation array in (C, H, W) format
        """
        pass
    
    @staticmethod
    def stack_maps(maps_dict: Dict[str, Dict[str, Any]]) -> Tuple[np.ndarray, List[float]]:
        """
        Stack maps from dictionary into single array.
        
        Args:
            maps_dict: Dictionary with map arrays and padding values
            
        Returns:
            Tuple of (stacked_maps, pad_values)
        """
        if not maps_dict:
            raise ValueError("Maps dictionary is empty")
        
        # Get first map to determine shape
        first_key = next(iter(maps_dict))
        base_shape = maps_dict[first_key]["map"].shape
        
        channels = []
        pad_values = []
        
        for map_name, map_info in maps_dict.items():
            map_array = map_info["map"]
            pad_value = map_info.get("pad", 0.0)
            
            if map_array.shape != base_shape:
                raise ValueError(f"Map {map_name} has inconsistent shape: "
                               f"{map_array.shape} vs {base_shape}")
            
            channels.append(map_array)
            pad_values.append(pad_value)
        
        stacked_maps = np.stack(channels, axis=-1)
        return stacked_maps, pad_values
    
    @staticmethod
    def apply_padding_per_channel(image: np.ndarray, pad_values: List[float], 
                                pad_length: int) -> np.ndarray:
        """
        Apply different padding values to each channel.
        
        Args:
            image: Input image (H, W, C)
            pad_values: Padding value for each channel
            pad_length: Padding length
            
        Returns:
            Padded image
        """
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
    
    def extract_ego_observation(self, maps: np.ndarray, pad_values: List[float],
                              center_y: float, center_x: float, direction_deg: float,
                              patch_size: Tuple[int, int]) -> np.ndarray:
        """
        Extract ego-centric observation patch.
        
        Args:
            maps: Stacked maps (H, W, C)
            pad_values: Padding values for each channel
            center_y: Center Y coordinate
            center_x: Center X coordinate
            direction_deg: Direction in degrees
            patch_size: Size of output patch (height, width)
            
        Returns:
            Rotated and cropped observation patch
        """
        patch_height, patch_width = patch_size
        
        # Calculate diagonal padding needed
        diagonal_length = math.ceil(max(patch_height, patch_width) / 2 * math.sqrt(2))
        
        # Apply padding
        padded_maps = self.apply_padding_per_channel(maps, pad_values, diagonal_length)
        
        # Adjust center coordinates for padding
        center_y_padded = center_y + diagonal_length
        center_x_padded = center_x + diagonal_length
        
        # Crop square region around center
        top = int(round(center_y_padded - diagonal_length))
        bottom = int(round(center_y_padded + diagonal_length))
        left = int(round(center_x_padded - diagonal_length))
        right = int(round(center_x_padded + diagonal_length))
        
        cropped_maps = padded_maps[top:bottom, left:right, :]
        
        if cropped_maps.size == 0:
            raise ValueError("Cropped region is empty")
        
        # Rotate to align agent direction upward
        rotation_angle = 180 + direction_deg
        rotation_center = (diagonal_length, diagonal_length)
        rotation_matrix = cv2.getRotationMatrix2D(rotation_center, rotation_angle, 1.0)
        
        rotated_maps = cv2.warpAffine(
            cropped_maps,
            rotation_matrix,
            (cropped_maps.shape[1], cropped_maps.shape[0])
        )
        
        # Ensure 3D
        if rotated_maps.ndim == 2:
            rotated_maps = rotated_maps[..., np.newaxis]
        
        # Final crop to desired patch size
        rotated_height, rotated_width = rotated_maps.shape[:2]
        start_y = max(0, (rotated_height - patch_height) // 2)
        start_x = max(0, (rotated_width - patch_width) // 2)
        
        final_patch = rotated_maps[start_y:start_y + patch_height, 
                                 start_x:start_x + patch_width, :]
        
        return final_patch


class FirstPersonObservation(ObservationStrategy):
    """Standard first-person observation strategy."""
    
    def generate_observation(self, agent: Agent, maps_dict: Dict[str, Dict[str, Any]]) -> np.ndarray:
        """Generate first-person observation."""
        # Stack maps
        stacked_maps, pad_values = self.stack_maps(maps_dict)
        
        # Apply noise to pose
        noisy_y, noisy_x, noisy_direction = self.noise_injector.apply_noise(agent)
        
        # Extract ego observation
        local_observation = self.extract_ego_observation(
            maps=stacked_maps,
            pad_values=pad_values,
            center_y=noisy_y,
            center_x=noisy_x,
            direction_deg=noisy_direction,
            patch_size=self.config.state_size
        )
        
        # Resize if needed
        if self.config.state_downsize != self.config.state_size:
            local_observation = cv2.resize(
                local_observation,
                (self.config.state_downsize[1], self.config.state_downsize[0]),
                interpolation=cv2.INTER_NEAREST
            )
        
        # Convert to (C, H, W) format
        observation = local_observation.transpose(2, 0, 1)
        
        return observation.astype(np.float32)


class MultiScaleObservation(ObservationStrategy):
    """Multi-scale observation strategy with global features."""
    
    def generate_observation(self, agent: Agent, maps_dict: Dict[str, Dict[str, Any]]) -> np.ndarray:
        """Generate multi-scale observation with global features."""
        # Stack maps
        stacked_maps, pad_values = self.stack_maps(maps_dict)
        
        # Apply noise to pose
        noisy_y, noisy_x, noisy_direction = self.noise_injector.apply_noise(agent)
        
        # Extract local observation
        local_observation = self.extract_ego_observation(
            maps=stacked_maps,
            pad_values=pad_values,
            center_y=noisy_y,
            center_x=noisy_x,
            direction_deg=noisy_direction,
            patch_size=self.config.state_size
        )
        
        # Resize local observation
        if self.config.state_downsize != self.config.state_size:
            local_observation = cv2.resize(
                local_observation,
                (self.config.state_downsize[1], self.config.state_downsize[0]),
                interpolation=cv2.INTER_NEAREST
            )
        
        # Convert to (C, H, W) format
        local_patch = local_observation.transpose(2, 0, 1)
        
        # Generate multi-scale features
        multiscale_features = self._generate_multiscale_features(local_patch)
        
        # Add global features if enabled
        if self.config.use_global_features:
            global_features = self._extract_global_features(
                stacked_maps, pad_values, noisy_y, noisy_x, noisy_direction
            )
            final_observation = np.concatenate([multiscale_features, global_features], axis=0)
        else:
            final_observation = multiscale_features
        
        return final_observation.astype(np.float32)
    
    def _generate_multiscale_features(self, observation: np.ndarray) -> np.ndarray:
        """
        Generate multi-scale features from observation.
        
        Args:
            observation: Input observation (C, H, W)
            
        Returns:
            Multi-scale features
        """
        channels, height, width = observation.shape
        observation_tensor = torch.from_numpy(observation).unsqueeze(0)  # (1, C, H, W)
        scale_features = []
        
        for scale_idx in range(self.config.n_scales):
            # Calculate crop size for this scale
            crop_length = (2 ** scale_idx) * self.config.multiscale_feature_size
            crop_length = min(crop_length, height, width)
            
            # Center crop
            top = (height - crop_length) // 2
            left = (width - crop_length) // 2
            bottom = top + crop_length
            right = left + crop_length
            
            cropped = observation_tensor[:, :, top:bottom, left:right]
            
            # Max pooling to target size
            kernel_size = max(1, crop_length // self.config.multiscale_feature_size)
            pooled = F.max_pool2d(cropped, kernel_size=kernel_size, stride=kernel_size)
            
            scale_features.append(pooled)
        
        # Concatenate all scales
        multiscale_features = torch.cat(scale_features, dim=1).squeeze(0)
        return multiscale_features.numpy()
    
    def _extract_global_features(self, maps: np.ndarray, pad_values: List[float],
                                center_y: float, center_x: float, 
                                direction_deg: float) -> np.ndarray:
        """
        Extract global features from full map.
        
        Args:
            maps: Full maps array
            pad_values: Padding values
            center_y: Center Y coordinate
            center_x: Center X coordinate
            direction_deg: Direction in degrees
            
        Returns:
            Global features array
        """
        height, width = maps.shape[:2]
        global_crop_size = max(height, width)
        
        # Extract large patch covering full map
        large_patch = self.extract_ego_observation(
            maps=maps,
            pad_values=pad_values,
            center_y=center_y,
            center_x=center_x,
            direction_deg=direction_deg,
            patch_size=(global_crop_size, global_crop_size)
        )
        
        # Convert to tensor and pool down
        large_patch_tensor = torch.from_numpy(large_patch.transpose(2, 0, 1)).unsqueeze(0)
        kernel_size = max(1, global_crop_size // self.config.multiscale_feature_size)
        
        pooled = F.max_pool2d(
            large_patch_tensor,
            kernel_size=kernel_size,
            stride=kernel_size
        )
        
        pooled_array = pooled.squeeze(0).numpy()
        
        # Ensure correct output size by center cropping
        _, pooled_height, pooled_width = pooled_array.shape
        target_size = self.config.multiscale_feature_size
        
        if pooled_height < target_size or pooled_width < target_size:
            raise ValueError(f"Global features too small: {pooled_array.shape}")
        
        start_y = (pooled_height - target_size) // 2
        start_x = (pooled_width - target_size) // 2
        
        global_features = pooled_array[:, start_y:start_y + target_size, 
                                     start_x:start_x + target_size]
        
        return global_features


class ObservationManager:
    """Manages observation generation using different strategies."""
    
    def __init__(self, config: ObservationConfig):
        """
        Initialize observation manager.
        
        Args:
            config: Observation configuration
        """
        self.config = config
        
        # Select strategy based on configuration
        if config.use_multiscale:
            self.strategy = MultiScaleObservation(config)
        else:
            self.strategy = FirstPersonObservation(config)
    
    def set_random_generator(self, rng: np.random.Generator) -> None:
        """Set random number generator."""
        self.strategy.set_random_generator(rng)
    
    def generate_observation(self, agent: Agent, maps_dict: Dict[str, Dict[str, Any]]) -> np.ndarray:
        """Generate observation using current strategy."""
        return self.strategy.generate_observation(agent, maps_dict)
    
    def get_observation_shape(self) -> Tuple[int, int, int]:
        """Get expected observation shape (C, H, W)."""
        if self.config.use_multiscale:
            # Calculate number of channels
            base_channels = len(self.config.state_size)  # This should be updated based on map channels
            multiscale_channels = base_channels * self.config.n_scales
            
            if self.config.use_global_features:
                total_channels = multiscale_channels + base_channels
            else:
                total_channels = multiscale_channels
            
            return (total_channels, self.config.multiscale_feature_size, self.config.multiscale_feature_size)
        else:
            # Standard first-person observation
            return (len(self.config.state_size), self.config.state_downsize[0], self.config.state_downsize[1])
    
    def set_strategy(self, strategy: ObservationStrategy) -> None:
        """Change observation strategy."""
        self.strategy = strategy