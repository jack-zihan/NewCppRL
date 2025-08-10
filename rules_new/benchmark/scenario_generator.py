"""
场景生成器 - 确保所有算法在相同场景下测试

保证相同seed生成完全一致的场景配置
"""

import random
import numpy as np
import torch
from typing import Dict, List, Any, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ScenarioGenerator:
    """
    场景生成器
    
    确保给定相同的seed和参数，生成完全一致的场景
    """
    
    def __init__(self, base_config: Dict[str, Any]):
        """
        初始化场景生成器
        
        Args:
            base_config: 基础配置，包含环境参数
        """
        self.base_config = base_config
        self.scenarios_cache = {}  # 缓存已生成的场景
        
    def generate_scenario(self, 
                         seed: int,
                         difficulty: str,
                         map_id: int,
                         weed_distribution: str,
                         noise_level: str = 'no_noise') -> Dict[str, Any]:
        """
        生成确定性场景
        
        Args:
            seed: 随机种子
            difficulty: 难度等级 (easy/medium/hard)
            map_id: 地图ID
            weed_distribution: 杂草分布类型 (gaussian/uniform)
            noise_level: 噪声级别
            
        Returns:
            场景配置字典
        """
        # 生成场景唯一ID
        scenario_id = f"s{seed}_{difficulty}_m{map_id}_{weed_distribution[0]}"
        
        # 检查缓存
        if scenario_id in self.scenarios_cache:
            logger.debug(f"使用缓存场景: {scenario_id}")
            return self.scenarios_cache[scenario_id]
        
        # 固定所有随机种子
        self._set_seeds(seed)
        
        # 获取难度配置
        difficulty_config = self.base_config['difficulty_levels'][difficulty]
        
        # 生成场景参数
        scenario = {
            'seed': seed,
            'scenario_id': scenario_id,
            'difficulty': difficulty,
            'map_id': map_id,
            'weed_distribution': weed_distribution,
            'noise_level': noise_level,
            
            # 环境参数
            'obstacle_range': difficulty_config['obstacle_range'],
            'weed_num': difficulty_config['weed_num'],
            'noise_params': self.base_config['noise_sets'][noise_level],
            
            # 农场参数（确保一致性）
            'farm_vertices': self._generate_farm_vertices(map_id, seed),
            
            # 障碍物位置（确定性生成）
            'obstacle_positions': self._generate_obstacles(
                difficulty_config['obstacle_range'], 
                seed
            ),
            
            # 杂草位置（确定性生成）
            'weed_positions': self._generate_weeds(
                weed_distribution,
                difficulty_config['weed_num'],
                seed
            )
        }
        
        # 缓存场景
        self.scenarios_cache[scenario_id] = scenario
        
        logger.info(f"生成场景: {scenario_id}")
        return scenario
    
    def generate_all_scenarios(self, 
                              seeds: List[int],
                              difficulties: List[str],
                              weed_distributions: List[str],
                              noise_levels: List[str] = None) -> List[Dict[str, Any]]:
        """
        生成所有场景组合
        
        Args:
            seeds: 种子列表
            difficulties: 难度列表
            weed_distributions: 杂草分布类型列表
            noise_levels: 噪声级别列表
            
        Returns:
            场景列表
        """
        if noise_levels is None:
            noise_levels = ['no_noise']
            
        scenarios = []
        
        for seed in seeds:
            for difficulty in difficulties:
                # 获取该难度对应的地图
                map_ids = self.base_config['difficulty_levels'][difficulty]['map_ids']
                
                for map_id in map_ids:
                    for weed_dist in weed_distributions:
                        for noise_level in noise_levels:
                            scenario = self.generate_scenario(
                                seed, difficulty, map_id, 
                                weed_dist, noise_level
                            )
                            scenarios.append(scenario)
        
        logger.info(f"生成 {len(scenarios)} 个场景")
        return scenarios
    
    def _set_seeds(self, seed: int):
        """设置所有随机数生成器的种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def _generate_farm_vertices(self, map_id: int, seed: int) -> List[List[float]]:
        """
        生成农场顶点（确定性）
        
        Args:
            map_id: 地图ID
            seed: 随机种子
            
        Returns:
            农场顶点列表
        """
        # 固定种子
        np.random.seed(seed + map_id)
        
        # 基于map_id生成不同形状的农场
        if map_id % 3 == 0:
            # 矩形农场
            width = 400 + (map_id % 5) * 40
            height = 400 + ((map_id + 1) % 5) * 40
            vertices = [
                [100, 100],
                [100 + width, 100],
                [100 + width, 100 + height],
                [100, 100 + height]
            ]
        elif map_id % 3 == 1:
            # 五边形农场
            center = [300, 300]
            radius = 200 + (map_id % 4) * 30
            angles = np.linspace(0, 2 * np.pi, 6)[:-1]
            vertices = []
            for angle in angles:
                x = center[0] + radius * np.cos(angle)
                y = center[1] + radius * np.sin(angle)
                vertices.append([x, y])
        else:
            # 不规则四边形
            base_vertices = np.array([
                [150, 150],
                [450, 120],
                [480, 420],
                [120, 450]
            ])
            # 添加一些随机扰动
            perturbation = np.random.randn(4, 2) * 20
            vertices = (base_vertices + perturbation).tolist()
        
        return vertices
    
    def _generate_obstacles(self, 
                           obstacle_range: Tuple[int, int],
                           seed: int) -> List[Dict[str, Any]]:
        """
        生成障碍物位置（确定性）
        
        Args:
            obstacle_range: 障碍物数量范围
            seed: 随机种子
            
        Returns:
            障碍物位置列表
        """
        np.random.seed(seed * 2)  # 使用不同的种子避免与其他生成重复
        
        min_obs, max_obs = obstacle_range
        num_obstacles = np.random.randint(min_obs, max_obs + 1) if max_obs > min_obs else min_obs
        
        obstacles = []
        for i in range(num_obstacles):
            obstacle = {
                'position': [
                    np.random.uniform(150, 450),
                    np.random.uniform(150, 450)
                ],
                'radius': np.random.uniform(10, 30),
                'type': 'static'
            }
            obstacles.append(obstacle)
        
        return obstacles
    
    def _generate_weeds(self,
                       distribution: str,
                       num_weeds: int,
                       seed: int) -> List[List[float]]:
        """
        生成杂草位置（确定性）
        
        Args:
            distribution: 分布类型 (gaussian/uniform)
            num_weeds: 杂草数量
            seed: 随机种子
            
        Returns:
            杂草位置列表
        """
        np.random.seed(seed * 3)  # 使用不同的种子
        
        weeds = []
        
        if distribution == 'gaussian':
            # 高斯分布：杂草聚集在几个中心点
            num_centers = min(5, max(1, num_weeds // 20))
            centers = np.random.uniform(150, 450, (num_centers, 2))
            
            weeds_per_center = num_weeds // num_centers
            for center in centers:
                for _ in range(weeds_per_center):
                    # 在中心点周围生成杂草
                    offset = np.random.randn(2) * 50
                    weed_pos = center + offset
                    # 确保在边界内
                    weed_pos = np.clip(weed_pos, 100, 500)
                    weeds.append(weed_pos.tolist())
                    
        else:  # uniform
            # 均匀分布：杂草随机分布在整个区域
            for _ in range(num_weeds):
                weed_pos = [
                    np.random.uniform(100, 500),
                    np.random.uniform(100, 500)
                ]
                weeds.append(weed_pos)
        
        return weeds
    
    def get_scenario_info(self, scenario_id: str) -> Dict[str, Any]:
        """
        获取场景信息
        
        Args:
            scenario_id: 场景ID
            
        Returns:
            场景信息字典
        """
        if scenario_id in self.scenarios_cache:
            return self.scenarios_cache[scenario_id]
        else:
            logger.warning(f"场景 {scenario_id} 不在缓存中")
            return None
    
    def clear_cache(self):
        """清空场景缓存"""
        self.scenarios_cache.clear()
        logger.info("场景缓存已清空")