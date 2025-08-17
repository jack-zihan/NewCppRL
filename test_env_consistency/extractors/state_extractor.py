"""
状态提取器
从新环境中提取完整的标准化状态
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from typing import Dict, Any, Optional, List, Tuple


class NewEnvStateExtractor:
    """
    新环境状态提取器
    从组件化的新环境中提取完整状态
    """
    
    def __init__(self):
        """初始化提取器"""
        self.state_cache = {}
        
    def extract_complete_state(self, env) -> Dict[str, Any]:
        """
        从新环境提取完整状态
        新环境使用组件化架构，需要从各个管理器中提取
        """
        state = {
            'agent': self._extract_agent_state(env),
            'maps': self._extract_maps_state(env),
            'metrics': self._extract_metrics_state(env),
            'dynamics': self._extract_dynamics_state(env),
            'config': self._extract_config_state(env),
            'env_state': self._extract_env_state_info(env)
        }
        
        # 缓存状态用于后续对比
        self.state_cache = state.copy()
        
        return state
    
    def _extract_agent_state(self, env) -> Dict[str, Any]:
        """提取Agent相关状态"""
        agent_state = {}
        
        # 从agent_manager提取
        if hasattr(env, 'agent_manager'):
            agent_mgr = env.agent_manager
            if hasattr(agent_mgr, 'agent'):
                agent = agent_mgr.agent
                agent_state['position'] = (agent.position[0], agent.position[1])
                agent_state['direction'] = agent.direction
                agent_state['velocity'] = getattr(agent, 'velocity', 0.0)
                agent_state['angular_velocity'] = getattr(agent, 'angular_velocity', 0.0)
            
            agent_state['collision_count'] = getattr(agent_mgr, 'collision_count', 0)
        
        return agent_state
    
    def _extract_maps_state(self, env) -> Dict[str, Any]:
        """提取地图相关状态"""
        maps_state = {}
        
        # 从map_manager提取
        if hasattr(env, 'map_manager'):
            map_mgr = env.map_manager
            
            # 提取各种地图
            map_names = ['weed_map', 'noisy_weed_map', 'original_weed_map',
                        'frontier_map', 'obstacle_map', 'boundary_map',
                        'trajectory_map', 'mist_map']
            
            for map_name in map_names:
                if hasattr(map_mgr, map_name):
                    map_data = getattr(map_mgr, map_name)
                    if map_data is not None:
                        # 复制地图数据以避免引用问题
                        maps_state[map_name.replace('_map', '')] = map_data.copy()
        
        return maps_state
    
    def _extract_metrics_state(self, env) -> Dict[str, Any]:
        """提取度量相关状态"""
        metrics_state = {}
        
        # 从metrics_manager提取
        if hasattr(env, 'metrics_manager'):
            metrics_mgr = env.metrics_manager
            
            if hasattr(metrics_mgr, 'current_metrics'):
                metrics = metrics_mgr.current_metrics
                metrics_state['weed_count'] = getattr(metrics, 'weed_count', 0)
                metrics_state['initial_weed_count'] = getattr(metrics, 'initial_weed_count', 0)
                metrics_state['frontier_count'] = getattr(metrics, 'frontier_count', 0)
                metrics_state['initial_frontier_count'] = getattr(metrics, 'initial_frontier_count', 0)
            
            # 其他度量
            metrics_state['collision_count'] = getattr(env.agent_manager, 'collision_count', 0)
            metrics_state['step_count'] = env.current_step if hasattr(env, 'current_step') else 0
        
        return metrics_state
    
    def _extract_dynamics_state(self, env) -> Dict[str, Any]:
        """提取动力学相关状态"""
        dynamics_state = {}
        
        # 从env_state提取历史信息
        if hasattr(env, 'env_state'):
            env_state = env.env_state
            
            # 提取位置历史
            if hasattr(env_state, 'get_info'):
                pos_info = env_state.get_info('agent_position')
                if pos_info:
                    dynamics_state['last_position'] = pos_info.last
                    dynamics_state['current_position'] = pos_info.current
                
                # 提取方向历史
                dir_info = env_state.get_info('agent_direction')
                if dir_info:
                    dynamics_state['last_direction'] = dir_info.last
                    dynamics_state['current_direction'] = dir_info.current
        
        return dynamics_state
    
    def _extract_config_state(self, env) -> Dict[str, Any]:
        """提取配置相关状态"""
        config_state = {}
        
        # 从config提取
        if hasattr(env, 'config'):
            config = env.config
            config_state['dimensions'] = config.dimensions
            config_state['action_space_type'] = config.action_space_type
            config_state['use_sgcnn'] = config.use_sgcnn
            config_state['use_trajectory'] = config.use_trajectory
            
            # 其他配置
            config_state['obstacle_num'] = config.obstacle_num
            config_state['weed_num'] = config.weed_num
            config_state['weed_dist'] = config.weed_dist
        
        return config_state
    
    def _extract_env_state_info(self, env) -> Dict[str, Any]:
        """提取环境状态管理器中的信息"""
        env_state_info = {}
        
        if hasattr(env, 'env_state'):
            state_mgr = env.env_state
            
            # 获取所有tracked的状态信息
            if hasattr(state_mgr, '_state_infos'):
                for key, state_var in state_mgr._state_infos.items():
                    env_state_info[key] = {
                        'current': state_var.current,
                        'last': state_var.last,
                        'history_length': len(state_var._history)
                    }
        
        return env_state_info
    
    def get_state_diff(self, env) -> Dict[str, Any]:
        """
        获取与上次提取状态的差异
        用于检测状态变化
        """
        current_state = self.extract_complete_state(env)
        
        if not self.state_cache:
            return {'status': 'no_previous_state'}
        
        differences = {}
        
        for category in ['agent', 'maps', 'metrics', 'dynamics']:
            if category in current_state and category in self.state_cache:
                cat_diff = self._compare_states(
                    self.state_cache[category],
                    current_state[category],
                    category
                )
                if cat_diff:
                    differences[category] = cat_diff
        
        return differences
    
    def _compare_states(self, old_state: Dict, new_state: Dict, category: str) -> Optional[Dict]:
        """比较两个状态字典"""
        diff = {}
        
        for key in new_state:
            if key not in old_state:
                diff[key] = {'change': 'added', 'value': new_state[key]}
            elif isinstance(new_state[key], np.ndarray):
                if not np.array_equal(old_state[key], new_state[key]):
                    diff[key] = {
                        'change': 'modified',
                        'pixels_changed': np.sum(old_state[key] != new_state[key])
                    }
            elif isinstance(new_state[key], (int, float)):
                if abs(old_state[key] - new_state[key]) > 1e-6:
                    diff[key] = {
                        'change': 'modified',
                        'old': old_state[key],
                        'new': new_state[key],
                        'delta': new_state[key] - old_state[key]
                    }
            elif old_state[key] != new_state[key]:
                diff[key] = {
                    'change': 'modified',
                    'old': old_state[key],
                    'new': new_state[key]
                }
        
        return diff if diff else None
    
    def validate_state_completeness(self, state: Dict[str, Any]) -> Dict[str, bool]:
        """
        验证状态的完整性
        检查所有必要的状态组件是否存在
        """
        required_components = {
            'agent': ['position', 'direction'],
            'maps': ['weed', 'frontier', 'obstacle'],
            'metrics': ['weed_count', 'step_count'],
            'config': ['dimensions', 'action_space_type']
        }
        
        validation_result = {}
        
        for category, required_fields in required_components.items():
            if category not in state:
                validation_result[category] = False
                continue
            
            category_valid = True
            for field in required_fields:
                if field not in state[category] or state[category][field] is None:
                    category_valid = False
                    break
            
            validation_result[category] = category_valid
        
        validation_result['overall'] = all(validation_result.values())
        
        return validation_result