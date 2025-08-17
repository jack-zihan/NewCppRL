"""
状态同步器
实现新环境到旧环境的状态同步
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from test_env_consistency.extractors.state_extractor import NewEnvStateExtractor
from test_env_consistency.wrappers.old_env_wrapper import OldEnvWrapper
from test_env_consistency.configs.state_mapping import STATE_MAPPING, TOLERANCE_CONFIG


class StateSynchronizer:
    """
    状态同步器
    负责将新环境的状态同步到旧环境
    """
    
    def __init__(self, tolerance_config: Optional[Dict] = None):
        """
        初始化同步器
        
        Args:
            tolerance_config: 容差配置，用于浮点数比较
        """
        self.extractor = NewEnvStateExtractor()
        self.tolerance = tolerance_config or TOLERANCE_CONFIG
        self.sync_history = []
        self.last_sync_state = None
        
    def sync_new_to_old(self, new_env, old_env_wrapper: OldEnvWrapper) -> Dict[str, Any]:
        """
        将新环境状态同步到旧环境
        
        Args:
            new_env: 新环境实例
            old_env_wrapper: 旧环境包装器实例
            
        Returns:
            同步报告，包含同步的状态和任何问题
        """
        # 提取新环境状态
        new_state = self.extractor.extract_complete_state(new_env)
        
        # 验证状态完整性
        completeness = self.extractor.validate_state_completeness(new_state)
        if not completeness['overall']:
            return {
                'success': False,
                'error': 'Incomplete new environment state',
                'details': completeness
            }
        
        # 执行同步
        sync_result = self._perform_sync(new_state, old_env_wrapper)
        
        # 验证同步结果
        verification = self._verify_sync(new_env, old_env_wrapper)
        
        # 记录同步历史
        self.sync_history.append({
            'new_state': new_state,
            'sync_result': sync_result,
            'verification': verification
        })
        
        self.last_sync_state = new_state
        
        return {
            'success': verification['success'],
            'sync_result': sync_result,
            'verification': verification,
            'state_summary': self._get_state_summary(new_state)
        }
    
    def _perform_sync(self, new_state: Dict, old_env: OldEnvWrapper) -> Dict[str, Any]:
        """
        执行实际的状态同步
        """
        sync_report = {
            'agent': {},
            'maps': {},
            'metrics': {},
            'issues': []
        }
        
        # 同步Agent状态
        if 'agent' in new_state:
            for key, value in new_state['agent'].items():
                try:
                    if key == 'position':
                        old_env.agent.x, old_env.agent.y = value
                        sync_report['agent'][key] = 'synced'
                    elif key == 'direction':
                        old_env.agent.direction = value
                        sync_report['agent'][key] = 'synced'
                    elif key == 'velocity':
                        old_env.velocity = value
                        sync_report['agent'][key] = 'synced'
                    elif key == 'angular_velocity':
                        old_env.w = value
                        sync_report['agent'][key] = 'synced'
                    elif key == 'collision_count':
                        old_env.collision_num = value
                        sync_report['agent'][key] = 'synced'
                except Exception as e:
                    sync_report['issues'].append(f"Failed to sync agent.{key}: {str(e)}")
        
        # 同步地图状态
        if 'maps' in new_state:
            for map_name, map_data in new_state['maps'].items():
                try:
                    if map_data is not None:
                        # 映射地图名称
                        old_map_name = f'map_{map_name}' if not map_name.startswith('map_') else map_name
                        setattr(old_env, old_map_name, map_data.copy())
                        sync_report['maps'][map_name] = 'synced'
                except Exception as e:
                    sync_report['issues'].append(f"Failed to sync map {map_name}: {str(e)}")
        
        # 同步度量状态
        if 'metrics' in new_state:
            metrics_mapping = {
                'weed_count': 'weed_num_t',
                'initial_weed_count': 'weed_num',
                'frontier_count': 'frontier_num_t',
                'initial_frontier_count': 'frontier_num',
                'collision_count': 'collision_num',
                'step_count': 'current_step'
            }
            
            for new_key, old_key in metrics_mapping.items():
                if new_key in new_state['metrics']:
                    try:
                        setattr(old_env, old_key, new_state['metrics'][new_key])
                        sync_report['metrics'][new_key] = 'synced'
                    except Exception as e:
                        sync_report['issues'].append(f"Failed to sync metric {new_key}: {str(e)}")
        
        # 同步随机数生成器状态（如果需要）
        if hasattr(new_env, '_np_random') and hasattr(old_env, 'np_random'):
            try:
                # 获取新环境的随机数生成器状态
                new_rng_state = new_env._np_random.bit_generator.state
                # 设置到旧环境
                old_env.np_random.bit_generator.state = new_rng_state
                sync_report['rng'] = 'synced'
            except Exception as e:
                sync_report['issues'].append(f"Failed to sync RNG state: {str(e)}")
        
        # 更新旧环境的统一状态缓存
        old_env._capture_initial_states()
        
        return sync_report
    
    def _verify_sync(self, new_env, old_env: OldEnvWrapper) -> Dict[str, Any]:
        """
        验证同步是否成功
        """
        verification = {
            'success': True,
            'mismatches': [],
            'warnings': []
        }
        
        # 提取两个环境的当前状态
        new_state = self.extractor.extract_complete_state(new_env)
        old_state = old_env.get_unified_state()
        
        # 验证Agent状态
        if 'agent' in new_state and 'agent' in old_state:
            agent_new = new_state['agent']
            agent_old = old_state['agent']
            
            # 验证位置
            if 'position' in agent_new and 'position' in agent_old:
                pos_new = agent_new['position']
                pos_old = agent_old['position']
                pos_diff = np.sqrt((pos_new[0] - pos_old[0])**2 + (pos_new[1] - pos_old[1])**2)
                
                if pos_diff > self.tolerance['position_tolerance']:
                    verification['mismatches'].append({
                        'field': 'agent.position',
                        'new': pos_new,
                        'old': pos_old,
                        'difference': pos_diff
                    })
                    verification['success'] = False
            
            # 验证方向
            if 'direction' in agent_new and 'direction' in agent_old:
                dir_diff = abs(agent_new['direction'] - agent_old['direction'])
                if dir_diff > self.tolerance['angle_tolerance']:
                    verification['mismatches'].append({
                        'field': 'agent.direction',
                        'new': agent_new['direction'],
                        'old': agent_old['direction'],
                        'difference': dir_diff
                    })
                    verification['success'] = False
        
        # 验证地图状态
        if 'maps' in new_state and 'maps' in old_state:
            for map_name in new_state['maps']:
                if map_name in old_state['maps']:
                    map_new = new_state['maps'][map_name]
                    map_old = old_state['maps'][map_name]
                    
                    if map_new is not None and map_old is not None:
                        if not np.array_equal(map_new, map_old):
                            diff_count = np.sum(map_new != map_old)
                            verification['mismatches'].append({
                                'field': f'maps.{map_name}',
                                'pixels_different': diff_count
                            })
                            verification['success'] = False
        
        # 验证度量状态
        if 'metrics' in new_state and 'metrics' in old_state:
            for metric_name in ['weed_count', 'frontier_count', 'step_count']:
                if metric_name in new_state['metrics'] and metric_name in old_state['metrics']:
                    val_new = new_state['metrics'][metric_name]
                    val_old = old_state['metrics'][metric_name]
                    
                    if val_new != val_old:
                        verification['mismatches'].append({
                            'field': f'metrics.{metric_name}',
                            'new': val_new,
                            'old': val_old
                        })
                        verification['success'] = False
        
        return verification
    
    def _get_state_summary(self, state: Dict) -> Dict:
        """
        获取状态摘要信息
        """
        summary = {}
        
        if 'agent' in state:
            summary['agent_position'] = state['agent'].get('position', 'N/A')
            summary['agent_direction'] = state['agent'].get('direction', 'N/A')
        
        if 'metrics' in state:
            summary['weed_count'] = state['metrics'].get('weed_count', 'N/A')
            summary['step_count'] = state['metrics'].get('step_count', 'N/A')
        
        if 'maps' in state:
            summary['maps_available'] = list(state['maps'].keys())
        
        return summary
    
    def get_sync_history(self) -> List[Dict]:
        """获取同步历史记录"""
        return self.sync_history
    
    def clear_history(self):
        """清空同步历史"""
        self.sync_history = []
        self.last_sync_state = None