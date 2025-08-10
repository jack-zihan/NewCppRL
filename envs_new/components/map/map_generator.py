"""
地图生成协调器 - 组件化场景生成系统。

核心设计：
1. 组件化架构：每个Creator负责生成特定的场景元素
2. 依赖管理系统：自动解析组件间依赖，确保正确的生成顺序
3. 状态共享机制：通过字典传递共享状态，避免复杂的参数传递

组件依赖关系：
- frontier: 无依赖，创建基础地图
- obstacle: 依赖frontier，在地图上生成障碍物
- weed: 依赖obstacle，避免在障碍物上生成杂草
- agent: 依赖obstacle，确保初始位置安全
- trajectory: 依赖agent，从机器人位置开始生成轨迹
- mist: 依赖agent，场景已完整后添加迷雾效果
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

from envs_new.components.config.environment_config import EnvironmentConfig
from envs_new.components.state.environment_state import EnvironmentState
from envs_new.components.entity.agent import Agent
from envs_new.components.map.map_components import (
    FrontierCreator, AgentCreator, ObstacleCreator, WeedCreator, TrajectoryCreator, MistCreator
)
from envs_new.utils.dependency_sorter import sort_components_by_dependencies


class ScenarioGenerator:
    """场景生成器 - 灵活的组件化地图生成。"""
    
    # 所有可用组件
    COMPONENTS = {
        'frontier': FrontierCreator(),
        'agent': AgentCreator(),
        'obstacle': ObstacleCreator(),
        'weed': WeedCreator(),
        'trajectory': TrajectoryCreator(),
        'mist': MistCreator()
    }
    
    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.rng: Optional[np.random.Generator] = None
        
        # 构建基于配置的默认组件列表
        self._build_default_components()
    
    def _build_default_components(self) -> None:
        """根据配置构建默认启用的组件列表"""
        # 核心组件：地图、机器人、障碍物、杂草
        self.default_components = ['frontier', 'agent', 'obstacle', 'weed']
        
        # 根据配置添加可选组件
        if self.config.use_traj:
            self.default_components.append('trajectory')
        if self.config.use_mist:
            self.default_components.append('mist')
    
    def set_random_generator(self, rng: np.random.Generator) -> None:
        """设置随机数生成器"""
        self.rng = rng
    
    def generate_scenario(self, 
                         enabled_components: Optional[List[str]] = None,
                         map_id: Optional[int] = None,
                         weed_distribution: str = "uniform",
                         weed_count: int = 100,
                         scenario_directory: Optional[Union[str, Path]] = None,
                         initial_position: Optional[Tuple[float, float]] = None,
                         initial_direction: Optional[float] = None,
                         **additional_options) -> Tuple[Agent, Dict[str, np.ndarray], EnvironmentState]:
        """
        场景生成核心函数
        
        Args:
            enabled_components: 启用的组件列表，默认根据map_config创建全部合法组件
            map_id: 地图ID
            weed_distribution: 杂草分布类型 ("uniform", "gaussian")
            weed_count: 杂草数量
            scenario_directory: 预制场景目录
            initial_position: 初始位置覆盖
            initial_direction: 初始方向覆盖
            **additional_options: 额外选项
            
        Returns:
            (agent, maps_dict, env_state) 元组

        """
        if self.rng is None:
            raise ValueError("Random generator not set. Call set_random_generator() first.")
        
        # 使用智能默认组件列表而非所有组件
        if enabled_components is None:
            enabled_components = self.default_components
        
        # 验证组件名称
        invalid_components = set(enabled_components) - set(self.COMPONENTS.keys())
        if invalid_components:
            raise ValueError(f"Unknown components: {invalid_components}")
        
        # 构建共享状态字典，所有组件通过该字典交互
        state = {
            'agent': None,           # 机器人实体（由AgentCreator创建）
            'maps_dict': {},         # 地图数据（各组件逐步添加）
            'env_state': EnvironmentState(),  # 环境状态
            'config': self.config,   # 统一的环境配置
            'dimensions': None,      # 地图尺寸（由FrontierCreator设置）
            'options': {
                'map_id': map_id,
                'weed_distribution': weed_distribution,
                'weed_count': weed_count,
                'scenario_directory': scenario_directory,
                'initial_position': initial_position,
                'initial_direction': initial_direction,
                **additional_options
            }
        }
        
        # 组件间存在依赖关系，拓扑排序确保正确执行顺序
        execution_order = self._sort_components(enabled_components)
        
        for component_name in execution_order:
            component = self.COMPONENTS[component_name]
            try:
                component.generate(state, self.rng)
            except Exception as e:
                raise RuntimeError(f"Component '{component_name}' failed: {str(e)}") from e
        
        return state['agent'], state['maps_dict'], state['env_state']
    
    def _sort_components(self, enabled_components: List[str]) -> List[str]:
        """
        根据依赖关系排序组件。
        
        使用拓扑排序算法处理组件间的依赖关系，确保：
        1. 无循环依赖
        2. 依赖项在被依赖项之前执行
        """
        component_classes = {}
        for name in enabled_components:
            component_classes[name] = self.COMPONENTS[name].__class__
        
        return sort_components_by_dependencies(component_classes, enabled_components)
    
    def get_available_components(self) -> List[str]:
        """获取所有可用组件名称"""
        return list(self.COMPONENTS.keys())