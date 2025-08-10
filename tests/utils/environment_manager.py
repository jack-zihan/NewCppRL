"""
环境管理器 - 统一环境创建、同步、比较功能

提供一个统一的接口来管理测试环境的生命周期，
替代分散在各个测试文件中的重复环境操作逻辑。
"""
import sys
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Type
from contextlib import contextmanager

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


class EnvironmentManager:
    """环境管理器 - 统一环境操作接口"""
    
    def __init__(self, verbose: bool = True):
        """
        初始化环境管理器
        
        Args:
            verbose: 是否输出详细信息
        """
        self.verbose = verbose
        self._synchronizer = None
        self._env_cache = {}
        
    @property
    def synchronizer(self):
        """延迟加载环境状态同步器"""
        if self._synchronizer is None:
            from tests.utils.environment_state_synchronizer import EnvironmentStateSynchronizer
            self._synchronizer = EnvironmentStateSynchronizer()
        return self._synchronizer
    
    def get_environment_classes(self, version: str) -> Tuple[Type, Type]:
        """
        获取环境类
        
        Args:
            version: 环境版本 ('v1', 'v2', 'v3')
            
        Returns:
            (旧环境类, 新环境类) 元组
        """
        if version in self._env_cache:
            return self._env_cache[version]
        
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
            
            self._env_cache[version] = (OldEnv, NewEnv)
            return OldEnv, NewEnv
            
        except ImportError as e:
            raise ImportError(f"Failed to import environment classes for {version}: {e}")
    
    @contextmanager
    def create_environment_pair(self, version: str, seed: int = 0, sync: bool = True):
        """
        创建环境对的上下文管理器
        
        Args:
            version: 环境版本
            seed: 随机种子
            sync: 是否同步环境状态
            
        Yields:
            (old_env, new_env) 环境对
        """
        OldEnv, NewEnv = self.get_environment_classes(version)
        
        # 创建环境实例
        old_env = OldEnv(render_mode=None)
        new_env = NewEnv()
        
        try:
            if self.verbose:
                print(f"🏗️ 创建环境对 (版本: {version}, 种子: {seed})")
            
            # 重置环境
            new_obs, new_info = new_env.reset(seed=seed)
            old_obs, old_info = old_env.reset(seed=999999)  # 使用固定种子避免随机影响
            
            # 同步状态
            if sync:
                new_state = self.synchronizer.extract_new_env_state(new_env)
                self.synchronizer.sync_old_env_state(old_env, new_state)
                
                if self.verbose:
                    print("🔄 环境状态已同步")
            
            yield old_env, new_env
            
        finally:
            # 确保环境被正确关闭
            old_env.close()
            new_env.close()
            
            if self.verbose:
                print("🗑️ 环境已清理")
    
    def create_multiple_environment_pairs(self, versions: List[str], seeds: List[int], 
                                        sync: bool = True) -> Dict[str, Dict[int, Tuple]]:
        """
        批量创建环境对（非上下文管理器版本，需要手动清理）
        
        Args:
            versions: 环境版本列表
            seeds: 种子列表
            sync: 是否同步状态
            
        Returns:
            {version: {seed: (old_env, new_env)}} 的嵌套字典
        """
        env_pairs = {}
        
        for version in versions:
            env_pairs[version] = {}
            
            for seed in seeds:
                OldEnv, NewEnv = self.get_environment_classes(version)
                
                old_env = OldEnv(render_mode=None)
                new_env = NewEnv()
                
                # 重置环境
                new_obs, new_info = new_env.reset(seed=seed)
                old_obs, old_info = old_env.reset(seed=999999)
                
                # 同步状态
                if sync:
                    new_state = self.synchronizer.extract_new_env_state(new_env)
                    self.synchronizer.sync_old_env_state(old_env, new_state)
                
                env_pairs[version][seed] = (old_env, new_env)
        
        if self.verbose:
            total_pairs = sum(len(seeds_dict) for seeds_dict in env_pairs.values())
            print(f"🏗️ 已创建 {total_pairs} 个环境对")
        
        return env_pairs
    
    def cleanup_environment_pairs(self, env_pairs: Dict[str, Dict[int, Tuple]]):
        """
        清理批量创建的环境对
        
        Args:
            env_pairs: 环境对字典
        """
        total_cleaned = 0
        
        for version, seeds_dict in env_pairs.items():
            for seed, (old_env, new_env) in seeds_dict.items():
                old_env.close()
                new_env.close()
                total_cleaned += 1
        
        if self.verbose:
            print(f"🗑️ 已清理 {total_cleaned} 个环境对")
    
    def verify_environment_sync(self, old_env, new_env, tolerance: float = 1e-12) -> Dict[str, Any]:
        """
        验证环境同步状态
        
        Args:
            old_env: 旧环境实例
            new_env: 新环境实例
            tolerance: 数值比较容差
            
        Returns:
            同步验证结果
        """
        verification = {
            'synchronized': True,
            'differences': {},
            'details': {}
        }
        
        # 验证智能体状态
        agent_diff = {
            'position': np.linalg.norm([old_env.agent.x - new_env.agent.x, 
                                       old_env.agent.y - new_env.agent.y]),
            'direction': abs(old_env.agent.direction - new_env.agent.direction),
            'steer': abs(old_env.agent.last_steer - new_env.agent.last_steer)
        }
        
        for key, diff in agent_diff.items():
            if diff > tolerance:
                verification['synchronized'] = False
                verification['differences'][f'agent_{key}'] = diff
        
        verification['details']['agent'] = agent_diff
        
        # 验证地图状态
        map_checks = {}
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
                    map_checks[map_name] = max_diff
                    
                    if max_diff > tolerance:
                        verification['synchronized'] = False
                        verification['differences'][f'map_{map_name}'] = max_diff
                else:
                    map_checks[map_name] = 0.0
        
        verification['details']['maps'] = map_checks
        
        # 验证环境状态变量
        state_checks = {}
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
                    state_checks[var_name] = diff
                    
                    if diff > tolerance:
                        verification['synchronized'] = False
                        verification['differences'][f'state_{var_name}'] = diff
        
        verification['details']['state'] = state_checks
        
        return verification
    
    def compare_environment_step_results(self, old_result: Tuple, new_result: Tuple, 
                                       tolerance: float = 1e-12) -> Dict[str, Any]:
        """
        比较环境step结果
        
        Args:
            old_result: 旧环境step结果 (obs, reward, done, trunc, info)
            new_result: 新环境step结果 (obs, reward, done, trunc, info)
            tolerance: 数值比较容差
            
        Returns:
            比较结果
        """
        old_obs, old_reward, old_done, old_trunc, old_info = old_result
        new_obs, new_reward, new_done, new_trunc, new_info = new_result
        
        comparison = {
            'consistent': True,
            'differences': {},
            'details': {}
        }
        
        # 比较观测
        obs_comparison = self._compare_observations(old_obs, new_obs, tolerance)
        comparison['details']['observations'] = obs_comparison
        
        if not obs_comparison['consistent']:
            comparison['consistent'] = False
            comparison['differences']['observations'] = obs_comparison['differences']
        
        # 比较奖励
        reward_diff = abs(float(old_reward) - float(new_reward))
        comparison['details']['reward'] = {
            'old_reward': float(old_reward),
            'new_reward': float(new_reward),
            'difference': reward_diff,
            'consistent': reward_diff <= tolerance
        }
        
        if reward_diff > tolerance:
            comparison['consistent'] = False
            comparison['differences']['reward'] = reward_diff
        
        # 比较完成状态
        done_consistent = (old_done == new_done) and (old_trunc == new_trunc)
        comparison['details']['done'] = {
            'old_done': old_done,
            'new_done': new_done,
            'old_trunc': old_trunc,
            'new_trunc': new_trunc,
            'consistent': done_consistent
        }
        
        if not done_consistent:
            comparison['consistent'] = False
            comparison['differences']['done'] = "Terminal states don't match"
        
        return comparison
    
    def _compare_observations(self, obs1: Dict, obs2: Dict, tolerance: float) -> Dict[str, Any]:
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
                if max_diff > tolerance:
                    result['consistent'] = False
                    result['differences'][key] = f"Max diff: {max_diff:.8f}"
            
            elif isinstance(obs1[key], (int, float)):
                diff = abs(obs1[key] - obs2[key])
                if diff > tolerance:
                    result['consistent'] = False
                    result['differences'][key] = f"Diff: {diff:.8f}"
        
        return result
    
    def run_environment_compatibility_test(self, versions: List[str], 
                                         num_steps: int = 5) -> Dict[str, Any]:
        """
        运行环境兼容性测试
        
        Args:
            versions: 要测试的版本列表
            num_steps: 每个版本的测试步数
            
        Returns:
            兼容性测试结果
        """
        if self.verbose:
            print(f"🔧 开始环境兼容性测试")
            print(f"   版本: {versions}")
            print(f"   步数: {num_steps}")
        
        results = {
            'test_type': 'compatibility',
            'versions_tested': versions,
            'steps_per_version': num_steps,
            'version_results': {},
            'overall_compatible': True
        }
        
        for version in versions:
            if self.verbose:
                print(f"\n🧪 测试版本 {version}")
            
            version_result = {
                'version': version,
                'compatible': True,
                'creation_success': False,
                'reset_success': False,
                'step_success': False,
                'sync_success': False,
                'errors': []
            }
            
            try:
                # 测试环境创建
                with self.create_environment_pair(version, seed=0, sync=True) as (old_env, new_env):
                    version_result['creation_success'] = True
                    version_result['reset_success'] = True
                    
                    # 测试同步
                    sync_verification = self.verify_environment_sync(old_env, new_env)
                    version_result['sync_success'] = sync_verification['synchronized']
                    
                    if not sync_verification['synchronized']:
                        version_result['compatible'] = False
                        version_result['errors'].append(f"Sync failed: {sync_verification['differences']}")
                    
                    # 测试步骤执行
                    step_errors = []
                    for step in range(num_steps):
                        try:
                            action = old_env.action_space.sample()
                            
                            old_result = old_env.step(action)
                            new_result = new_env.step(action)
                            
                            step_comparison = self.compare_environment_step_results(old_result, new_result)
                            
                            if not step_comparison['consistent']:
                                step_errors.append(f"Step {step}: {step_comparison['differences']}")
                            
                            if old_result[2] or old_result[3]:  # done or truncated
                                break
                                
                        except Exception as e:
                            step_errors.append(f"Step {step} error: {str(e)}")
                    
                    if step_errors:
                        version_result['compatible'] = False
                        version_result['errors'].extend(step_errors)
                    else:
                        version_result['step_success'] = True
                    
                    if self.verbose:
                        if version_result['compatible']:
                            print(f"   ✅ 版本 {version} 兼容性测试通过")
                        else:
                            print(f"   ❌ 版本 {version} 兼容性测试失败")
                            for error in version_result['errors'][:3]:  # 只显示前3个错误
                                print(f"      - {error}")
            
            except Exception as e:
                version_result['compatible'] = False
                version_result['errors'].append(f"Environment creation failed: {str(e)}")
                
                if self.verbose:
                    print(f"   ❌ 版本 {version} 环境创建失败: {str(e)}")
            
            if not version_result['compatible']:
                results['overall_compatible'] = False
            
            results['version_results'][version] = version_result
        
        if self.verbose:
            status = "✅ 通过" if results['overall_compatible'] else "❌ 失败"
            print(f"\n🏆 兼容性测试总体结果: {status}")
        
        return results
    
    def get_environment_info(self, version: str) -> Dict[str, Any]:
        """
        获取环境信息
        
        Args:
            version: 环境版本
            
        Returns:
            环境信息字典
        """
        try:
            OldEnv, NewEnv = self.get_environment_classes(version)
            
            # 创建临时环境获取信息
            with self.create_environment_pair(version, seed=0, sync=False) as (old_env, new_env):
                info = {
                    'version': version,
                    'old_env': {
                        'class_name': old_env.__class__.__name__,
                        'action_space_type': type(old_env.action_space).__name__,
                        'action_space_shape': getattr(old_env.action_space, 'shape', None),
                        'action_space_n': getattr(old_env.action_space, 'n', None),
                        'observation_space_type': type(old_env.observation_space).__name__,
                        'observation_space_shape': getattr(old_env.observation_space, 'shape', None),
                    },
                    'new_env': {
                        'class_name': new_env.__class__.__name__,
                        'action_space_type': type(new_env.action_space).__name__,
                        'action_space_shape': getattr(new_env.action_space, 'shape', None),
                        'action_space_n': getattr(new_env.action_space, 'n', None),
                        'observation_space_type': type(new_env.observation_space).__name__,
                        'observation_space_shape': getattr(new_env.observation_space, 'shape', None),
                    }
                }
                
                return info
                
        except Exception as e:
            return {
                'version': version,
                'error': str(e),
                'available': False
            }