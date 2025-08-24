"""
环境状态同步工具

用于在新旧版本环境之间复制和同步状态，确保动力学测试的一致性
"""
import numpy as np
import cv2
from typing import Dict, Any, Tuple
from pathlib import Path


class EnvironmentStateSynchronizer:
    """环境状态同步器，用于在新旧环境之间同步状态"""
    
    @staticmethod
    def extract_new_env_state(new_env) -> Dict[str, Any]:
        """
        从新版环境中提取完整状态信息
        
        Args:
            new_env: 新版环境实例
            
        Returns:
            包含所有状态信息的字典
        """
        state_info = {
            # 地图信息
            'maps': {
                'field': new_env.maps_dict['field'].copy(),
                'obstacle': new_env.maps_dict['obstacle'].copy(), 
                'weed': new_env.maps_dict['weed'].copy(),
                'trajectory': new_env.maps_dict.get('trajectory', np.zeros_like(new_env.maps_dict['field'])).copy(),
                'mist': new_env.maps_dict.get('mist', np.ones_like(new_env.maps_dict['field'])).copy(),
            },
            
            # 智能体状态
            'agent': {
                'x': float(new_env.agent.x),
                'y': float(new_env.agent.y), 
                'direction': float(new_env.agent.direction),
                'last_steer': float(new_env.agent.last_steer),
                'position_discrete': tuple(new_env.agent.position_discrete),
                'convex_hull': new_env.agent.convex_hull.copy()
            },
            
            # 环境状态
            'env_state': {
                'dimensions': new_env.env_state.dimensions,
                'current_step': new_env.env_state.current_step,
                'max_steps': new_env.env_state.max_steps,
                'field_area': new_env.env_state.field_area if hasattr(new_env.env_state, 'field_area') else 0,
                'field_variation': new_env.env_state.field_variation if hasattr(new_env.env_state, 'field_variation') else 0,
                'weed_count': new_env.env_state.weed_count if hasattr(new_env.env_state, 'weed_count') else 0,
                'total_weed_count': new_env.env_state.total_weed_count,
                'agent_position': new_env.env_state.agent_position if hasattr(new_env.env_state, 'agent_position') else new_env.agent.position,
                'agent_steer': new_env.env_state.agent_steer if hasattr(new_env.env_state, 'agent_steer') else 0,
                'crashed': new_env.env_state.crashed,
                'finished': new_env.env_state.finished,
                'timeout': new_env.env_state.timeout
            },
            
            # 配置信息
            'config': {
                'action_type': new_env.config.action_type if hasattr(new_env.config, 'action_type') else 'discrete',
                'max_episode_steps': new_env.config.max_episode_steps,
                'vision_length': new_env.agent.vision_length if hasattr(new_env.agent, 'vision_length') else 28,
                'vision_angle': new_env.agent.vision_angle if hasattr(new_env.agent, 'vision_angle') else 75,
                'v_range': (0.0, 3.5),  # 默认值
                'w_range': (-28.6, 28.6),  # 默认值
                'nvec': (7, 21)  # 默认值
            }
        }
        
        return state_info
    
    @staticmethod 
    def sync_old_env_state(old_env, state_info: Dict[str, Any]) -> None:
        """
        将状态信息同步到旧版环境
        
        Args:
            old_env: 旧版环境实例
            state_info: 从新版环境提取的状态信息
        """
        # 同步地图
        maps = state_info['maps']
        old_env.map_frontier = maps['field'].copy()
        old_env.map_obstacle = maps['obstacle'].copy()
        old_env.map_weed = maps['weed'].copy()
        old_env.map_trajectory = maps['trajectory'].copy()  
        old_env.map_mist = maps['mist'].copy()
        
        # 同步原始地图（用于渲染）
        old_env.map_frontier_full = maps['field'].copy()
        if hasattr(old_env, 'map_weed_ori'):
            old_env.map_weed_ori = maps['weed'].copy()
        
        # 同步智能体状态
        agent_info = state_info['agent']
        old_env.agent.x = agent_info['x']
        old_env.agent.y = agent_info['y']
        old_env.agent.direction = agent_info['direction']  
        old_env.agent.last_steer = agent_info['last_steer']
        
        # 同步环境状态变量
        env_state = state_info['env_state']
        old_env.dimensions = env_state['dimensions']
        old_env.t = env_state['current_step']
        
        # 同步状态追踪变量（用于奖励计算）
        old_env.frontier_area_t = env_state['field_area']
        old_env.frontier_tv_t = env_state['field_variation'] 
        old_env.weed_num_t = env_state['weed_count']
        old_env.steer_t = env_state['agent_steer']
        old_env.weed_num = env_state['total_weed_count']
        
        # 同步配置信息
        config = state_info['config']
        old_env.action_type = config['action_type']
        old_env.vision_length = int(config['vision_length'])
        old_env.vision_angle = int(config['vision_angle'])
        
        # 同步动作范围
        from envs_new.components.config.environment_config import NumericalRange
        old_env.v_range = NumericalRange(config['v_range'][0], config['v_range'][1])
        old_env.w_range = NumericalRange(config['w_range'][0], config['w_range'][1])
        old_env.nvec = config['nvec']
        
        print("✓ Old environment state synchronized")
    
    @staticmethod
    def sync_new_env_state(new_env, state_info: Dict[str, Any]) -> None:
        """
        将状态信息同步到新版环境 (特别是previous position)
        
        Args:
            new_env: 新版环境实例  
            state_info: 从新版环境提取的状态信息（可用于设置previous值）
        """
        env_state = state_info['env_state']
        # 设置previous position为当前position（这样在step时trajectory就从当前位置开始）
        new_env.env_state._previous_agent_position = new_env.env_state.agent_position
        print("✓ New environment previous position synchronized")
    
    @staticmethod
    def compare_states_before_step(old_env, new_env) -> Dict[str, Any]:
        """
        比较两个环境在执行step前的状态
        
        Returns:
            状态比较结果
        """
        comparison = {
            'agent_position_diff': np.linalg.norm([
                old_env.agent.x - new_env.agent.x,
                old_env.agent.y - new_env.agent.y
            ]),
            'agent_direction_diff': abs(old_env.agent.direction - new_env.agent.direction),
            'agent_steer_diff': abs(old_env.agent.last_steer - new_env.agent.last_steer),
            'field_area_diff': abs(old_env.frontier_area_t - new_env.env_state.field_area),
            'weed_count_diff': abs(old_env.weed_num_t - new_env.env_state.weed_count),
            'step_diff': abs(old_env.t - new_env.env_state.current_step),
            'maps_equal': {
                'field': np.array_equal(old_env.map_frontier, new_env.maps_dict['field']),
                'obstacle': np.array_equal(old_env.map_obstacle, new_env.maps_dict['obstacle']),
                'weed': np.array_equal(old_env.map_weed, new_env.maps_dict['weed']),
                'trajectory': np.array_equal(old_env.map_trajectory, new_env.maps_dict.get('trajectory', np.zeros_like(old_env.map_trajectory))),
                'mist': np.array_equal(old_env.map_mist, new_env.maps_dict.get('mist', np.ones_like(old_env.map_mist)))
            }
        }
        
        return comparison
    
    @staticmethod  
    def compare_states_after_step(old_env, new_env, old_obs, new_obs, old_reward, new_reward,
                                 old_done, new_done, old_info, new_info) -> Dict[str, Any]:
        """
        比较两个环境在执行step后的状态
        
        Returns:
            详细的状态比较结果
        """
        comparison = {
            # 智能体状态比较
            'agent_position_diff': np.linalg.norm([
                old_env.agent.x - new_env.agent.x,
                old_env.agent.y - new_env.agent.y
            ]),
            'agent_direction_diff': abs(old_env.agent.direction - new_env.agent.direction),
            'agent_steer_diff': abs(old_env.agent.last_steer - new_env.agent.last_steer),
            
            # 环境状态比较
            'field_area_diff': abs(old_env.frontier_area_t - new_env.env_state.field_area),
            'field_tv_diff': abs(old_env.frontier_tv_t - new_env.env_state.field_variation),
            'weed_count_diff': abs(old_env.weed_num_t - new_env.env_state.weed_count),
            'step_diff': abs(old_env.t - new_env.env_state.current_step),
            
            # 观测比较
            'obs_shape_match': old_obs['observation'].shape == new_obs['observation'].shape,
            'vector_diff': float(np.abs(old_obs['vector'] - new_obs['vector']).max()),
            'weed_ratio_diff': float(np.abs(old_obs['weed_ratio'] - new_obs['weed_ratio']).max()),
            
            # 奖励和完成状态比较
            'reward_diff': abs(float(old_reward) - float(new_reward)),
            'done_match': bool(old_done) == bool(new_done[0] or new_done[1]),  # terminated or truncated
            
            # 地图状态比较
            'maps_equal': {
                'field': np.array_equal(old_env.map_frontier, new_env.maps_dict['field']),
                'weed': np.array_equal(old_env.map_weed, new_env.maps_dict['weed']),
                'trajectory': np.array_equal(old_env.map_trajectory, new_env.maps_dict.get('trajectory', np.zeros_like(old_env.map_trajectory))),
                'mist': np.array_equal(old_env.map_mist, new_env.maps_dict.get('mist', np.ones_like(old_env.map_mist)))
            },
            
            # 地图变化量比较
            'map_changes': {
                'field_pixels_changed': int(np.sum(old_env.map_frontier != new_env.maps_dict['field'])),
                'weed_pixels_changed': int(np.sum(old_env.map_weed != new_env.maps_dict['weed'])),
                'trajectory_pixels_changed': int(np.sum(old_env.map_trajectory != new_env.maps_dict.get('trajectory', np.zeros_like(old_env.map_trajectory)))),
                'mist_pixels_changed': int(np.sum(old_env.map_mist != new_env.maps_dict.get('mist', np.ones_like(old_env.map_mist))))
            }
        }
        
        return comparison
    
    @staticmethod
    def print_comparison_summary(comparison: Dict[str, Any], step_type: str = "after_step") -> None:
        """打印比较结果摘要"""
        print(f"\n{'='*60}")
        print(f"STATE COMPARISON SUMMARY ({step_type.upper()})")
        print(f"{'='*60}")
        
        # 智能体状态
        print(f"Agent Position Diff: {comparison['agent_position_diff']:.6f}")
        print(f"Agent Direction Diff: {comparison['agent_direction_diff']:.6f}°")
        print(f"Agent Steer Diff: {comparison['agent_steer_diff']:.6f}")
        
        # 环境状态
        print(f"Field Area Diff: {comparison.get('field_area_diff', comparison.get('frontier_area_diff', 0))}")
        if 'field_tv_diff' in comparison:
            print(f"Field TV Diff: {comparison['field_tv_diff']}")
        elif 'frontier_tv_diff' in comparison:
            print(f"Field TV Diff: {comparison['frontier_tv_diff']}")
        print(f"Weed Count Diff: {comparison['weed_count_diff']}")
        print(f"Step Diff: {comparison['step_diff']}")
        
        # 地图一致性
        maps_equal = comparison['maps_equal']
        print(f"\nMap Consistency:")
        for map_name, is_equal in maps_equal.items():
            status = "✓" if is_equal else "✗"
            print(f"  {status} {map_name}: {'Equal' if is_equal else 'Different'}")
        
        if step_type == "after_step":
            # 观测和奖励
            print(f"\nObservation & Reward:")
            print(f"Obs Shape Match: {'✓' if comparison['obs_shape_match'] else '✗'}")
            print(f"Vector Diff: {comparison['vector_diff']:.6f}")
            print(f"Weed Ratio Diff: {comparison['weed_ratio_diff']:.6f}")
            print(f"Reward Diff: {comparison['reward_diff']:.6f}")
            print(f"Done Match: {'✓' if comparison['done_match'] else '✗'}")
            
            # 地图变化量
            if 'map_changes' in comparison:
                print(f"\nMap Changes:")
                for map_name, pixels_changed in comparison['map_changes'].items():
                    print(f"  {map_name}: {pixels_changed} pixels changed")