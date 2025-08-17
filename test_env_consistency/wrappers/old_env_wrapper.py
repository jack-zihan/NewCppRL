"""
旧环境包装器
统一管理旧环境中零散的状态记录，便于与新环境对齐
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from envs.cpp_env_base_copy import CppEnvBase
from test_env_consistency.configs.state_mapping import (
    STATE_MAPPING, 
    SCATTERED_STATE_TRACKING,
    DYNAMICS_STATE_CHANGES
)


class OldEnvWrapper(CppEnvBase):
    """
    旧环境包装器
    继承旧环境并添加统一的状态管理接口
    """
    
    def __init__(self, *args, **kwargs):
        """初始化包装器"""
        super().__init__(*args, **kwargs)
        
        # 统一的状态追踪字典
        self.unified_state = {
            'agent': {},
            'maps': {},
            'dynamics': {},
            'metrics': {},
            'history': {},
            'config': {}
        }
        
        # 历史记录缓冲区
        self.state_history = {
            'positions': [],
            'directions': [],
            'actions': [],
            'velocities': [],
            'rewards': [],
            'maps_checksums': []
        }
        
        # 最大历史记录长度
        self.max_history_length = 100
        
    def reset(self, seed: Optional[int] = None, **kwargs):
        """重置环境并捕获初始状态"""
        obs, info = super().reset(seed=seed, **kwargs)
        
        # 捕获初始状态
        self._capture_initial_states()
        
        return obs, info
    
    def step(self, action):
        """执行动作并捕获状态变化"""
        # 记录step前的状态
        self._capture_pre_step_states()
        
        # 调用原始step
        obs, reward, terminated, truncated, info = super().step(action)
        
        # 捕获step后的状态变化
        self._capture_post_step_states(action, reward)
        
        return obs, reward, terminated, truncated, info
    
    def _capture_initial_states(self):
        """捕获reset后的初始状态"""
        # Agent状态
        self.unified_state['agent'] = {
            'position': (self.agent.x, self.agent.y),
            'direction': self.agent.direction,
            'velocity': getattr(self, 'velocity', 0.0),
            'angular_velocity': getattr(self, 'w', 0.0)
        }
        
        # 地图状态
        self.unified_state['maps'] = {
            'weed': self.map_weed.copy() if hasattr(self, 'map_weed') else None,
            'weed_noisy': self.map_weed_noisy.copy() if hasattr(self, 'map_weed_noisy') else None,
            'weed_ori': self.map_weed_ori.copy() if hasattr(self, 'map_weed_ori') else None,
            'frontier': self.map_frontier.copy() if hasattr(self, 'map_frontier') else None,
            'obstacle': self.map_obstacle.copy() if hasattr(self, 'map_obstacle') else None,
            'boundary': self.map_boundary.copy() if hasattr(self, 'map_boundary') else None,
            'trajectory': self.map_trajectory.copy() if hasattr(self, 'map_trajectory') else None,
            'mist': self.map_mist.copy() if hasattr(self, 'map_mist') else None
        }
        
        # 度量状态
        self.unified_state['metrics'] = {
            'weed_count': self.weed_num_t if hasattr(self, 'weed_num_t') else 0,
            'initial_weed_count': self.weed_num if hasattr(self, 'weed_num') else 0,
            'frontier_count': self.frontier_num_t if hasattr(self, 'frontier_num_t') else 0,
            'initial_frontier_count': self.frontier_num if hasattr(self, 'frontier_num') else 0,
            'collision_count': self.collision_num if hasattr(self, 'collision_num') else 0,
            'step_count': self.current_step if hasattr(self, 'current_step') else 0
        }
        
        # 配置信息
        self.unified_state['config'] = {
            'dimensions': self.dimensions,
            'action_space_type': self.action_space_type,
            'use_sgcnn': self.use_sgcnn,
            'use_trajectory': self.use_traj if hasattr(self, 'use_traj') else False
        }
        
        # 清空历史记录
        self.state_history = {k: [] for k in self.state_history.keys()}
    
    def _capture_pre_step_states(self):
        """捕获step执行前的状态（用于记录last值）"""
        # 记录当前位置作为last position
        if hasattr(self, 'agent'):
            self.unified_state['dynamics']['last_position'] = (self.agent.x, self.agent.y)
            self.unified_state['dynamics']['last_direction'] = self.agent.direction
    
    def _capture_post_step_states(self, action, reward):
        """捕获step执行后的状态变化"""
        # 更新Agent状态
        self.unified_state['agent'] = {
            'position': (self.agent.x, self.agent.y),
            'direction': self.agent.direction,
            'velocity': getattr(self, 'velocity', 0.0),
            'angular_velocity': getattr(self, 'w', 0.0)
        }
        
        # 更新地图状态（只更新变化的）
        if hasattr(self, 'map_weed'):
            self.unified_state['maps']['weed'] = self.map_weed.copy()
        if hasattr(self, 'map_trajectory'):
            self.unified_state['maps']['trajectory'] = self.map_trajectory.copy()
        if hasattr(self, 'map_mist'):
            self.unified_state['maps']['mist'] = self.map_mist.copy()
        
        # 更新度量
        self.unified_state['metrics']['weed_count'] = self.weed_num_t if hasattr(self, 'weed_num_t') else 0
        self.unified_state['metrics']['frontier_count'] = self.frontier_num_t if hasattr(self, 'frontier_num_t') else 0
        self.unified_state['metrics']['collision_count'] = self.collision_num if hasattr(self, 'collision_num') else 0
        self.unified_state['metrics']['step_count'] = self.current_step if hasattr(self, 'current_step') else 0
        
        # 添加到历史记录
        self._update_history(action, reward)
    
    def _update_history(self, action, reward):
        """更新历史记录"""
        # 限制历史记录长度
        for key in self.state_history:
            if len(self.state_history[key]) >= self.max_history_length:
                self.state_history[key].pop(0)
        
        # 添加新记录
        self.state_history['positions'].append(self.unified_state['agent']['position'])
        self.state_history['directions'].append(self.unified_state['agent']['direction'])
        self.state_history['actions'].append(action)
        self.state_history['velocities'].append((
            self.unified_state['agent']['velocity'],
            self.unified_state['agent']['angular_velocity']
        ))
        self.state_history['rewards'].append(reward)
        
        # 计算地图校验和（用于快速检测地图变化）
        maps_checksum = {}
        for map_name, map_data in self.unified_state['maps'].items():
            if map_data is not None:
                maps_checksum[map_name] = np.sum(map_data)
        self.state_history['maps_checksums'].append(maps_checksum)
    
    def get_unified_state(self) -> Dict[str, Any]:
        """获取统一格式的状态"""
        return self.unified_state.copy()
    
    def set_unified_state(self, state_dict: Dict[str, Any]):
        """
        设置环境状态（从新环境同步）
        这是核心功能，需要将新环境的状态映射到旧环境的各个变量
        """
        # 设置Agent状态
        if 'agent' in state_dict:
            agent_state = state_dict['agent']
            if 'position' in agent_state:
                self.agent.x, self.agent.y = agent_state['position']
            if 'direction' in agent_state:
                self.agent.direction = agent_state['direction']
            if 'velocity' in agent_state:
                self.velocity = agent_state['velocity']
            if 'angular_velocity' in agent_state:
                self.w = agent_state['angular_velocity']
        
        # 设置地图状态
        if 'maps' in state_dict:
            maps_state = state_dict['maps']
            for map_name, map_data in maps_state.items():
                if map_data is not None:
                    setattr(self, f'map_{map_name}', map_data.copy())
        
        # 设置度量状态
        if 'metrics' in state_dict:
            metrics = state_dict['metrics']
            if 'weed_count' in metrics:
                self.weed_num_t = metrics['weed_count']
            if 'initial_weed_count' in metrics:
                self.weed_num = metrics['initial_weed_count']
            if 'frontier_count' in metrics:
                self.frontier_num_t = metrics['frontier_count']
            if 'initial_frontier_count' in metrics:
                self.frontier_num = metrics['initial_frontier_count']
            if 'collision_count' in metrics:
                self.collision_num = metrics['collision_count']
            if 'step_count' in metrics:
                self.current_step = metrics['step_count']
        
        # 更新统一状态缓存
        self._capture_initial_states()
    
    def get_state_by_path(self, path: str) -> Any:
        """根据路径获取状态值（支持嵌套访问）"""
        parts = path.split('.')
        obj = self
        
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                return None
        
        return obj
    
    def compare_with_new_env_state(self, new_env_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        与新环境状态进行对比
        返回差异报告
        """
        differences = {}
        
        # 对比各个类别的状态
        for category in ['agent', 'maps', 'metrics']:
            if category in new_env_state and category in self.unified_state:
                cat_diff = self._compare_category(
                    self.unified_state[category],
                    new_env_state[category],
                    category
                )
                if cat_diff:
                    differences[category] = cat_diff
        
        return differences
    
    def _compare_category(self, old_cat: Dict, new_cat: Dict, category: str) -> Dict:
        """对比某个类别的状态"""
        diff = {}
        
        for key in new_cat:
            if key not in old_cat:
                diff[key] = {'status': 'missing_in_old', 'new_value': new_cat[key]}
            elif isinstance(new_cat[key], np.ndarray):
                if not np.array_equal(old_cat[key], new_cat[key]):
                    diff[key] = {
                        'status': 'different',
                        'difference_count': np.sum(old_cat[key] != new_cat[key])
                    }
            elif isinstance(new_cat[key], (int, float)):
                if abs(old_cat[key] - new_cat[key]) > 1e-6:
                    diff[key] = {
                        'status': 'different',
                        'old': old_cat[key],
                        'new': new_cat[key],
                        'diff': new_cat[key] - old_cat[key]
                    }
            elif old_cat[key] != new_cat[key]:
                diff[key] = {
                    'status': 'different',
                    'old': old_cat[key],
                    'new': new_cat[key]
                }
        
        return diff if diff else None