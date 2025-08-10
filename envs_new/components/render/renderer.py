"""
简化的渲染系统
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, Tuple, Optional

from envs_new.components.config.environment_config import EnvironmentConfig
from envs_new.components.entity.agent import Agent
from envs_new.utils.image_utils import enlarge_map_features, extract_ego_patch


# 渲染颜色配置
RENDER_COLORS = {
    'background': (255, 255, 255),
    'field_frontier': (76, 187, 23),
    'covered_farmland': (112, 173, 7),
    'obstacle': (30, 75, 130),
    'obstacle_edge': (47, 82, 143),
    'weed_undiscovered': (0, 0, 0),
    'weed_discovered': (255, 0, 0),
    'weed_covered': (0, 0, 0),
    'trajectory': (255, 38, 255),
    'agent': (255, 0, 0),
    'agent_vision': (192, 192, 192),
    'mist': (128, 128, 128)
}

# 渲染透明度配置
COVERED_FARMLAND_ALPHA = 0.25  # 已覆盖农田的透明度
COVERED_WEED_ALPHA = 0.1       # 已清除杂草的透明度
MIST_EFFECT_ALPHA = 0.7        # 迷雾效果强度



class Renderer:
    """统一的环境渲染器"""
    
    def __init__(self, config: EnvironmentConfig):
        """初始化渲染器"""
        self.config = config
    
    def render(self, maps_dict: Dict[str, np.ndarray], agent: Agent,
              dimensions: Tuple[int, int], mode: str = "map",
              observation_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        渲染环境
        
        Args:
            maps_dict: 环境地图字典
            agent: 当前智能体
            dimensions: 地图尺寸
            mode: 渲染模式 ("map" 或 "first_person")
            observation_size: 第一人称视图大小
            
        Returns:
            渲染后的图像
        """
        if mode == "first_person":
            rendered = self._render_first_person(maps_dict, agent, dimensions, observation_size)
        else:
            rendered = self._render_map(maps_dict, agent, dimensions)
        
        # 应用缩放
        if self.config.render_repeat_times > 1:
            rendered = rendered.repeat(self.config.render_repeat_times, axis=0).repeat(
                self.config.render_repeat_times, axis=1
            )
        
        return rendered
    
    def _render_map(self, maps_dict: Dict[str, np.ndarray], agent: Agent,
                   dimensions: Tuple[int, int]) -> np.ndarray:
        """渲染完整环境地图"""
        width, height = dimensions
        rendered_map = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # 渲染顺序：从底层到顶层（背景→地形→物体→智能体→效果）
        
        # 渲染field frontier
        if 'field_frontier' in maps_dict:
            rendered_map[maps_dict['field_frontier'].astype(bool)] = RENDER_COLORS['field_frontier']
        
        # 渲染covered farmland
        if (self.config.render_covered_farmland and 
            'original_field_frontier' in maps_dict and 
            'field_frontier' in maps_dict):
            
            covered_mask = np.logical_and(
                maps_dict['original_field_frontier'],
                np.logical_not(maps_dict['field_frontier'])
            )
            if covered_mask.any():
                rendered_map[covered_mask] = (
                    COVERED_FARMLAND_ALPHA * np.array(RENDER_COLORS['covered_farmland']) +
                    (1 - COVERED_FARMLAND_ALPHA) * rendered_map[covered_mask]
                ).astype(np.uint8)
        
        # 渲染obstacles
        if 'obstacle' in maps_dict:
            rendered_map[maps_dict['obstacle'].astype(bool)] = RENDER_COLORS['obstacle']
        
        # 渲染agent vision
        cv2.ellipse(
            img=rendered_map,
            center=agent.position_discrete,
            axes=(int(agent.vision_length), int(agent.vision_length)),
            angle=agent.direction,
            startAngle=-int(agent.vision_angle / 2),
            endAngle=int(agent.vision_angle / 2),
            color=RENDER_COLORS['agent_vision'],
            thickness=-1
        )
        
        # 渲染weeds
        self._render_weeds(rendered_map, maps_dict)
        
        # 渲染trajectory
        if 'trajectory' in maps_dict:
            rendered_map[maps_dict['trajectory'].astype(bool)] = RENDER_COLORS['trajectory']
        
        # 渲染agent
        convex_hull = agent.convex_hull.round().astype(np.int32)
        cv2.fillPoly(rendered_map, [convex_hull], color=RENDER_COLORS['agent'])
        
        # 应用mist效果
        if self.config.render_mist and 'mist' in maps_dict:
            rendered_map[maps_dict['mist'].astype(bool)] = (
                rendered_map[maps_dict['mist'].astype(bool)] * MIST_EFFECT_ALPHA
            ).astype(np.uint8)
        
        return rendered_map
    
    def _render_weeds(self, rendered_map: np.ndarray, 
                     maps_dict: Dict[str, np.ndarray]) -> None:
        """渲染不同状态的杂草"""
        if 'weed' not in maps_dict:
            return
        
        weed_map = maps_dict['weed']
        
        if 'field_frontier' in maps_dict:
            # 未发现的杂草（在frontier区域内）
            # enlarge_map_features：将1像素的杂草扩大，使其在视觉上更明显
            weed_undiscovered = enlarge_map_features(
                np.logical_and(weed_map, maps_dict['field_frontier'])
            )
            rendered_map[weed_undiscovered] = RENDER_COLORS['weed_undiscovered']
            
            # 已发现的杂草（不在frontier区域内）
            weed_discovered = enlarge_map_features(
                np.logical_and(weed_map, np.logical_not(maps_dict['field_frontier']))
            )
            rendered_map[weed_discovered] = RENDER_COLORS['weed_discovered']
        else:
            # 所有杂草都作为已发现处理
            rendered_map[enlarge_map_features(weed_map.astype(bool))] = RENDER_COLORS['weed_discovered']
        
        # 渲染已清除的杂草
        if (self.config.render_covered_weed and 
            'original_weed' in maps_dict):
            weed_covered = enlarge_map_features(
                np.logical_and(
                    maps_dict['original_weed'],
                    np.logical_not(weed_map)
                )
            )
            if weed_covered.any():
                rendered_map[weed_covered] = (
                    (1 - COVERED_WEED_ALPHA) * np.array(RENDER_COLORS['weed_covered']) +
                    COVERED_WEED_ALPHA * rendered_map[weed_covered]
                ).astype(np.uint8)
    
    def _render_first_person(self, maps_dict: Dict[str, np.ndarray], agent: Agent,
                            dimensions: Tuple[int, int], 
                            observation_size: Tuple[int, int]) -> np.ndarray:
        """渲染第一人称视图"""
        # 获取基础渲染地图
        base_map = self._render_map(maps_dict, agent, dimensions)
        
        # 提取第一人称补丁 - 使用共享工具函数
        patch = extract_ego_patch(
            maps=base_map,
            pad_values=[128] * 3,  # RGB channels with gray padding
            center_y=agent.y,
            center_x=agent.x,
            direction_deg=agent.direction,
            patch_size=observation_size
        )
        
        return patch.astype(np.uint8)