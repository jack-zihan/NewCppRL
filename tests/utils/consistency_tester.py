"""
统一一致性测试器 - 整合动力学、奖励、观测一致性测试

提供一个统一的接口来测试新旧环境之间的完全一致性，
替代分散在多个文件中的重复功能。
"""
import sys
import os
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import json

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


class UnifiedConsistencyTester:
    """统一一致性测试器 - 集成所有一致性测试功能"""
    
    def __init__(self, tolerance: float = 1e-12, verbose: bool = True):
        """
        初始化统一测试器
        
        Args:
            tolerance: 数值比较容差
            verbose: 是否输出详细信息
        """
        self.tolerance = tolerance
        self.verbose = verbose
        self.test_results = {
            'dynamics': [],
            'rewards': [],
            'observations': [],
            'overall_stats': {}
        }
        
        # 延迟导入避免循环依赖
        self._synchronizer = None
    
    @property
    def synchronizer(self):
        """延迟加载环境状态同步器"""
        if self._synchronizer is None:
            from tests.utils.environment_state_synchronizer import EnvironmentStateSynchronizer
            self._synchronizer = EnvironmentStateSynchronizer()
        return self._synchronizer
    
    def get_environment_classes(self, version: str):
        """
        动态获取环境类
        
        Args:
            version: 环境版本 ('v1', 'v2', 'v3')
            
        Returns:
            (旧环境类, 新环境类) 元组
        """
        try:
            if version == 'v1':
                from envs.cpp_env_v1 import CppEnv as OldEnv
                from envs_new.cpp_env_v1 import CppEnv as NewEnv
            elif version == 'v2':
                from envs.cpp_env_v2 import CppEnv as OldEnv
                from envs_new.cpp_env_v2 import CppEnv as NewEnv
            elif version == 'v3':
                from envs.cpp_env_v3 import CppEnv as OldEnv
                from envs_new.cpp_env_v3 import CppEnv as NewEnv
            else:
                raise ValueError(f"Unsupported version: {version}")
            
            return OldEnv, NewEnv
            
        except ImportError as e:
            raise ImportError(f"Failed to import environment classes for {version}: {e}")
    
    def compare_agent_states(self, old_env, new_env) -> Dict[str, Any]:
        """比较智能体状态"""
        result = {
            'consistent': True,
            'differences': {}
        }
        
        # 位置比较
        pos_diff = np.linalg.norm([old_env.agent.x - new_env.agent.x, 
                                  old_env.agent.y - new_env.agent.y])
        if pos_diff > self.tolerance:
            result['consistent'] = False
            result['differences']['position'] = pos_diff
        
        # 方向比较
        dir_diff = abs(old_env.agent.direction - new_env.agent.direction)
        if dir_diff > self.tolerance:
            result['consistent'] = False
            result['differences']['direction'] = dir_diff
        
        # 转向比较
        steer_diff = abs(old_env.agent.last_steer - new_env.agent.last_steer)
        if steer_diff > self.tolerance:
            result['consistent'] = False
            result['differences']['steer'] = steer_diff
        
        return result
    
    def compare_map_states(self, old_env, new_env) -> Dict[str, Any]:
        """比较地图状态"""
        result = {
            'consistent': True,
            'differences': {}
        }
        
        # 地图映射关系
        map_mappings = {
            'frontier': ('map_frontier', 'field_frontier'),
            'obstacle': ('map_obstacle', 'obstacle'),
            'weed': ('map_weed', 'weed'),
            'trajectory': ('map_trajectory', 'trajectory'),
            'mist': ('map_mist', 'mist')
        }
        
        for map_name, (old_attr, new_key) in map_mappings.items():
            if hasattr(old_env, old_attr) and new_key in new_env.maps_dict:
                old_map = getattr(old_env, old_attr)
                new_map = new_env.maps_dict[new_key]
                
                if not np.array_equal(old_map, new_map):
                    max_diff = np.abs(old_map - new_map).max()
                    if max_diff > self.tolerance:
                        result['consistent'] = False
                        result['differences'][map_name] = {
                            'max_diff': float(max_diff),
                            'changed_pixels': int(np.sum(old_map != new_map)),
                            'total_pixels': int(old_map.size)
                        }
        
        return result
    
    def compare_environment_states(self, old_env, new_env) -> Dict[str, Any]:
        """比较环境状态变量"""
        result = {
            'consistent': True,
            'differences': {}
        }
        
        # 状态变量映射
        state_mappings = {
            'frontier_area': ('frontier_area_t', 'frontier_area'),
            'frontier_variation': ('frontier_tv_t', 'frontier_variation'),
            'weed_count': ('weed_num_t', 'weed_count'),
            'agent_steer': ('steer_t', 'agent_steer'),
            'current_step': ('t', 'current_step')
        }
        
        for var_name, (old_attr, new_attr) in state_mappings.items():
            if hasattr(old_env, old_attr):
                old_val = getattr(old_env, old_attr)
                new_val = getattr(new_env.env_state, new_attr, None)
                
                if new_val is not None:
                    diff = abs(old_val - new_val)
                    if diff > self.tolerance:
                        result['consistent'] = False
                        result['differences'][var_name] = {
                            'diff': float(diff),
                            'old_value': float(old_val),
                            'new_value': float(new_val)
                        }
        
        return result
    
    def compare_observations(self, obs1: Dict, obs2: Dict) -> Dict[str, Any]:
        """比较观测值"""
        result = {
            'consistent': True,
            'differences': {}
        }
        
        for key in obs1.keys():
            if key not in obs2:
                result['consistent'] = False
                result['differences'][key] = "Missing in obs2"
                continue
            
            if isinstance(obs1[key], np.ndarray):
                if obs1[key].shape != obs2[key].shape:
                    result['consistent'] = False
                    result['differences'][key] = f"Shape mismatch: {obs1[key].shape} vs {obs2[key].shape}"
                    continue
                
                max_diff = np.abs(obs1[key] - obs2[key]).max()
                if max_diff > self.tolerance:
                    result['consistent'] = False
                    result['differences'][key] = f"Max diff: {max_diff:.8f}"
            
            elif isinstance(obs1[key], (int, float)):
                diff = abs(obs1[key] - obs2[key])
                if diff > self.tolerance:
                    result['consistent'] = False
                    result['differences'][key] = f"Diff: {diff:.8f}"
        
        return result
    
    def test_dynamics_consistency(self, version: str = 'v2', seeds: List[int] = None, 
                                 num_steps: int = 10) -> Dict[str, Any]:
        """
        测试动力学一致性
        
        Args:
            version: 环境版本
            seeds: 测试种子列表
            num_steps: 每个种子的测试步数
            
        Returns:
            测试结果详情
        """
        if seeds is None:
            seeds = [0, 1, 2, 3, 4]
        
        if self.verbose:
            print(f"🔄 开始动力学一致性测试 (版本: {version})")
            print(f"   种子: {seeds}, 步数: {num_steps}")
        
        OldEnv, NewEnv = self.get_environment_classes(version)
        results = {
            'version': version,
            'test_type': 'dynamics',
            'seeds_tested': len(seeds),
            'steps_per_seed': num_steps,
            'seed_results': {},
            'overall_consistent': True,
            'error_summary': {}
        }
        
        for seed in seeds:
            if self.verbose:
                print(f"\n🧪 测试种子 {seed}")
            
            # 创建环境
            old_env = OldEnv(render_mode=None)
            new_env = NewEnv()
            
            try:
                # 重置环境
                new_obs, new_info = new_env.reset(seed=seed)
                old_obs, old_info = old_env.reset(seed=999999)
                
                # 同步状态
                new_state = self.synchronizer.extract_new_env_state(new_env)
                self.synchronizer.sync_old_env_state(old_env, new_state)
                
                seed_result = {
                    'consistent': True,
                    'step_results': [],
                    'total_steps': 0,
                    'consistent_steps': 0
                }
                
                # 执行测试步骤
                for step in range(num_steps):
                    action = random.randint(0, new_env.action_space.n - 1)
                    
                    # 执行动作
                    new_obs, new_reward, new_done, new_trunc, new_info = new_env.step(action)
                    old_obs, old_reward, old_done, old_trunc, old_info = old_env.step(action)
                    
                    # 比较状态
                    agent_check = self.compare_agent_states(old_env, new_env)
                    map_check = self.compare_map_states(old_env, new_env)
                    env_check = self.compare_environment_states(old_env, new_env)
                    obs_check = self.compare_observations(old_obs, new_obs)
                    
                    step_consistent = (agent_check['consistent'] and 
                                     map_check['consistent'] and 
                                     env_check['consistent'] and 
                                     obs_check['consistent'])
                    
                    step_result = {
                        'step': step,
                        'action': action,
                        'consistent': step_consistent,
                        'agent_check': agent_check,
                        'map_check': map_check,
                        'env_check': env_check,
                        'obs_check': obs_check
                    }
                    
                    seed_result['step_results'].append(step_result)
                    seed_result['total_steps'] += 1
                    
                    if step_consistent:
                        seed_result['consistent_steps'] += 1
                    else:
                        seed_result['consistent'] = False
                        if self.verbose:
                            print(f"   ❌ 步骤 {step} 不一致")
                    
                    if new_done or new_trunc or old_done or old_trunc:
                        break
                
                if seed_result['consistent']:
                    if self.verbose:
                        print(f"   ✅ 种子 {seed} 完全一致 ({seed_result['consistent_steps']}/{seed_result['total_steps']})")
                else:
                    results['overall_consistent'] = False
                    if self.verbose:
                        print(f"   ❌ 种子 {seed} 存在不一致 ({seed_result['consistent_steps']}/{seed_result['total_steps']})")
                
                results['seed_results'][seed] = seed_result
                
            finally:
                old_env.close()
                new_env.close()
        
        # 生成错误摘要
        if not results['overall_consistent']:
            results['error_summary'] = self._generate_error_summary(results['seed_results'])
        
        self.test_results['dynamics'].append(results)
        return results
    
    def test_rewards_consistency(self, version: str = 'v2', seeds: List[int] = None,
                                num_steps: int = 10) -> Dict[str, Any]:
        """
        测试奖励一致性
        
        Args:
            version: 环境版本
            seeds: 测试种子列表
            num_steps: 每个种子的测试步数
            
        Returns:
            测试结果详情
        """
        if seeds is None:
            seeds = [0, 1, 2, 3, 4]
        
        if self.verbose:
            print(f"🎁 开始奖励一致性测试 (版本: {version})")
            print(f"   种子: {seeds}, 步数: {num_steps}")
        
        OldEnv, NewEnv = self.get_environment_classes(version)
        results = {
            'version': version,
            'test_type': 'rewards',
            'seeds_tested': len(seeds),
            'steps_per_seed': num_steps,
            'seed_results': {},
            'overall_consistent': True,
            'reward_statistics': {
                'total_comparisons': 0,
                'consistent_comparisons': 0,
                'max_difference': 0.0,
                'avg_difference': 0.0
            }
        }
        
        all_differences = []
        
        for seed in seeds:
            if self.verbose:
                print(f"\n🧪 测试种子 {seed}")
            
            old_env = OldEnv(render_mode=None)
            new_env = NewEnv()
            
            try:
                # 重置环境
                new_obs, new_info = new_env.reset(seed=seed)
                old_obs, old_info = old_env.reset(seed=999999)
                
                # 同步状态
                new_state = self.synchronizer.extract_new_env_state(new_env)
                self.synchronizer.sync_old_env_state(old_env, new_state)
                
                seed_result = {
                    'consistent': True,
                    'reward_comparisons': [],
                    'total_steps': 0,
                    'consistent_steps': 0,
                    'max_reward_diff': 0.0
                }
                
                # 执行测试步骤
                for step in range(num_steps):
                    action = random.randint(0, new_env.action_space.n - 1)
                    
                    # 执行动作
                    new_obs, new_reward, new_done, new_trunc, new_info = new_env.step(action)
                    old_obs, old_reward, old_done, old_trunc, old_info = old_env.step(action)
                    
                    # 比较奖励
                    reward_diff = abs(float(new_reward) - float(old_reward))
                    reward_consistent = reward_diff <= self.tolerance
                    
                    comparison = {
                        'step': step,
                        'action': action,
                        'old_reward': float(old_reward),
                        'new_reward': float(new_reward),
                        'difference': reward_diff,
                        'consistent': reward_consistent
                    }
                    
                    seed_result['reward_comparisons'].append(comparison)
                    seed_result['total_steps'] += 1
                    all_differences.append(reward_diff)
                    
                    if reward_consistent:
                        seed_result['consistent_steps'] += 1
                    else:
                        seed_result['consistent'] = False
                        results['overall_consistent'] = False
                        if self.verbose:
                            print(f"   ❌ 步骤 {step} 奖励不一致: {old_reward:.6f} vs {new_reward:.6f} (差异: {reward_diff:.6f})")
                    
                    seed_result['max_reward_diff'] = max(seed_result['max_reward_diff'], reward_diff)
                    
                    if new_done or new_trunc or old_done or old_trunc:
                        break
                
                if seed_result['consistent']:
                    if self.verbose:
                        print(f"   ✅ 种子 {seed} 奖励完全一致 ({seed_result['consistent_steps']}/{seed_result['total_steps']})")
                else:
                    if self.verbose:
                        print(f"   ❌ 种子 {seed} 奖励存在差异 ({seed_result['consistent_steps']}/{seed_result['total_steps']})")
                
                results['seed_results'][seed] = seed_result
                
            finally:
                old_env.close()
                new_env.close()
        
        # 计算统计信息
        results['reward_statistics'] = {
            'total_comparisons': len(all_differences),
            'consistent_comparisons': sum(1 for d in all_differences if d <= self.tolerance),
            'max_difference': max(all_differences) if all_differences else 0.0,
            'avg_difference': np.mean(all_differences) if all_differences else 0.0
        }
        
        self.test_results['rewards'].append(results)
        return results
    
    def test_observations_consistency(self, version: str = 'v2', seeds: List[int] = None,
                                    num_steps: int = 10) -> Dict[str, Any]:
        """
        测试观测一致性
        
        Args:
            version: 环境版本
            seeds: 测试种子列表
            num_steps: 每个种子的测试步数
            
        Returns:
            测试结果详情
        """
        if seeds is None:
            seeds = [0, 1, 2, 3, 4]
        
        if self.verbose:
            print(f"👁️ 开始观测一致性测试 (版本: {version})")
            print(f"   种子: {seeds}, 步数: {num_steps}")
        
        OldEnv, NewEnv = self.get_environment_classes(version)
        results = {
            'version': version,
            'test_type': 'observations',
            'seeds_tested': len(seeds),
            'steps_per_seed': num_steps,
            'seed_results': {},
            'overall_consistent': True,
            'observation_statistics': {
                'total_comparisons': 0,
                'consistent_comparisons': 0,
                'failed_keys': {}
            }
        }
        
        for seed in seeds:
            if self.verbose:
                print(f"\n🧪 测试种子 {seed}")
            
            old_env = OldEnv(render_mode=None)
            new_env = NewEnv()
            
            try:
                # 重置环境
                new_obs, new_info = new_env.reset(seed=seed)
                old_obs, old_info = old_env.reset(seed=999999)
                
                # 同步状态
                new_state = self.synchronizer.extract_new_env_state(new_env)
                self.synchronizer.sync_old_env_state(old_env, new_state)
                
                seed_result = {
                    'consistent': True,
                    'observation_comparisons': [],
                    'total_steps': 0,
                    'consistent_steps': 0
                }
                
                # 执行测试步骤
                for step in range(num_steps):
                    action = random.randint(0, new_env.action_space.n - 1)
                    
                    # 执行动作
                    new_obs, new_reward, new_done, new_trunc, new_info = new_env.step(action)
                    old_obs, old_reward, old_done, old_trunc, old_info = old_env.step(action)
                    
                    # 比较观测
                    obs_check = self.compare_observations(old_obs, new_obs)
                    
                    comparison = {
                        'step': step,
                        'action': action,
                        'consistent': obs_check['consistent'],
                        'differences': obs_check['differences']
                    }
                    
                    seed_result['observation_comparisons'].append(comparison)
                    seed_result['total_steps'] += 1
                    
                    if obs_check['consistent']:
                        seed_result['consistent_steps'] += 1
                    else:
                        seed_result['consistent'] = False
                        results['overall_consistent'] = False
                        
                        # 记录失败的键
                        for key in obs_check['differences']:
                            if key not in results['observation_statistics']['failed_keys']:
                                results['observation_statistics']['failed_keys'][key] = 0
                            results['observation_statistics']['failed_keys'][key] += 1
                        
                        if self.verbose:
                            print(f"   ❌ 步骤 {step} 观测不一致: {list(obs_check['differences'].keys())}")
                    
                    if new_done or new_trunc or old_done or old_trunc:
                        break
                
                if seed_result['consistent']:
                    if self.verbose:
                        print(f"   ✅ 种子 {seed} 观测完全一致 ({seed_result['consistent_steps']}/{seed_result['total_steps']})")
                else:
                    if self.verbose:
                        print(f"   ❌ 种子 {seed} 观测存在差异 ({seed_result['consistent_steps']}/{seed_result['total_steps']})")
                
                results['seed_results'][seed] = seed_result
                
            finally:
                old_env.close()
                new_env.close()
        
        # 计算统计信息
        total_comparisons = sum(result['total_steps'] for result in results['seed_results'].values())
        consistent_comparisons = sum(result['consistent_steps'] for result in results['seed_results'].values())
        
        results['observation_statistics']['total_comparisons'] = total_comparisons
        results['observation_statistics']['consistent_comparisons'] = consistent_comparisons
        
        self.test_results['observations'].append(results)
        return results
    
    def test_all_consistency(self, version: str = 'v2', seeds: List[int] = None,
                           num_steps: int = 10) -> Dict[str, Any]:
        """
        运行所有一致性测试
        
        Args:
            version: 环境版本
            seeds: 测试种子列表
            num_steps: 每个种子的测试步数
            
        Returns:
            综合测试结果
        """
        if self.verbose:
            print(f"🚀 开始全面一致性测试 (版本: {version})")
        
        # 运行各项测试
        dynamics_result = self.test_dynamics_consistency(version, seeds, num_steps)
        rewards_result = self.test_rewards_consistency(version, seeds, num_steps)
        observations_result = self.test_observations_consistency(version, seeds, num_steps)
        
        # 综合结果
        overall_result = {
            'version': version,
            'test_timestamp': datetime.now().isoformat(),
            'test_config': {
                'seeds': seeds or [0, 1, 2, 3, 4],
                'num_steps': num_steps,
                'tolerance': self.tolerance
            },
            'results': {
                'dynamics': dynamics_result,
                'rewards': rewards_result,
                'observations': observations_result
            },
            'overall_consistent': (dynamics_result['overall_consistent'] and 
                                 rewards_result['overall_consistent'] and 
                                 observations_result['overall_consistent']),
            'summary': {
                'tests_run': 3,
                'tests_passed': sum([
                    dynamics_result['overall_consistent'],
                    rewards_result['overall_consistent'],
                    observations_result['overall_consistent']
                ])
            }
        }
        
        if self.verbose:
            self.print_comprehensive_report(overall_result)
        
        return overall_result
    
    def _generate_error_summary(self, seed_results: Dict) -> Dict[str, Any]:
        """生成错误摘要"""
        error_types = {
            'agent_errors': 0,
            'map_errors': 0,
            'env_state_errors': 0,
            'observation_errors': 0
        }
        
        for seed, result in seed_results.items():
            for step_result in result['step_results']:
                if not step_result['agent_check']['consistent']:
                    error_types['agent_errors'] += 1
                if not step_result['map_check']['consistent']:
                    error_types['map_errors'] += 1
                if not step_result['env_check']['consistent']:
                    error_types['env_state_errors'] += 1
                if not step_result['obs_check']['consistent']:
                    error_types['observation_errors'] += 1
        
        return error_types
    
    def print_comprehensive_report(self, results: Dict[str, Any]):
        """打印综合测试报告"""
        print(f"\n{'='*60}")
        print("全面一致性测试报告")
        print(f"{'='*60}")
        print(f"版本: {results['version']}")
        print(f"时间: {results['test_timestamp']}")
        print(f"总体一致性: {'✅ 通过' if results['overall_consistent'] else '❌ 失败'}")
        print(f"通过测试: {results['summary']['tests_passed']}/{results['summary']['tests_run']}")
        
        print("\n📊 详细结果:")
        for test_type, test_result in results['results'].items():
            status = "✅" if test_result['overall_consistent'] else "❌"
            print(f"  {status} {test_type.capitalize()}: {'通过' if test_result['overall_consistent'] else '失败'}")
            
            if test_type == 'rewards' and 'reward_statistics' in test_result:
                stats = test_result['reward_statistics']
                print(f"    - 比较次数: {stats['total_comparisons']}")
                print(f"    - 一致次数: {stats['consistent_comparisons']}")
                print(f"    - 最大差异: {stats['max_difference']:.8f}")
                print(f"    - 平均差异: {stats['avg_difference']:.8f}")
        
        print(f"{'='*60}")
    
    def save_results(self, filepath: str):
        """保存测试结果到文件"""
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'tolerance': self.tolerance,
            'test_results': self.test_results
        }
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        if self.verbose:
            print(f"✅ 测试结果已保存到: {filepath}")