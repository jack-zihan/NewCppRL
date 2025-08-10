"""
完整的一致性检查工具集合

提供环境一致性测试的所有核心功能，包括状态比较、观测比较、奖励比较等
"""
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import json
from datetime import datetime


class ComprehensiveConsistencyChecker:
    """全面的一致性检查器"""
    
    def __init__(self, tolerance: float = 1e-6, verbose: bool = True):
        self.tolerance = tolerance
        self.verbose = verbose
        self.test_results = {
            'total_comparisons': 0,
            'passed_comparisons': 0,
            'failed_comparisons': 0,
            'error_details': []
        }
    
    def compare_agent_states(self, agent1, agent2) -> Dict[str, Any]:
        """比较两个智能体状态"""
        result = {
            'consistent': True,
            'differences': {}
        }
        
        # 位置比较
        pos_diff = np.linalg.norm([agent1.x - agent2.x, agent1.y - agent2.y])
        if pos_diff > self.tolerance:
            result['consistent'] = False
            result['differences']['position'] = f"Position diff: {pos_diff:.8f}"
        
        # 方向比较
        dir_diff = abs(agent1.direction - agent2.direction)
        if dir_diff > self.tolerance:
            result['consistent'] = False
            result['differences']['direction'] = f"Direction diff: {dir_diff:.8f}"
        
        # 转向比较
        steer_diff = abs(agent1.last_steer - agent2.last_steer)
        if steer_diff > self.tolerance:
            result['consistent'] = False
            result['differences']['steer'] = f"Steer diff: {steer_diff:.8f}"
        
        return result
    
    def compare_map_states(self, maps1: Dict, maps2: Dict) -> Dict[str, Any]:
        """比较两个地图状态"""
        result = {
            'consistent': True,
            'differences': {}
        }
        
        common_keys = set(maps1.keys()) & set(maps2.keys())
        
        for key in common_keys:
            if isinstance(maps1[key], np.ndarray) and isinstance(maps2[key], np.ndarray):
                if not np.array_equal(maps1[key], maps2[key]):
                    max_diff = np.abs(maps1[key] - maps2[key]).max()
                    changed_pixels = np.sum(maps1[key] != maps2[key])
                    total_pixels = maps1[key].size
                    
                    if max_diff > self.tolerance:
                        result['consistent'] = False
                        result['differences'][key] = {
                            'max_diff': float(max_diff),
                            'changed_pixels': int(changed_pixels),
                            'change_ratio': float(changed_pixels / total_pixels)
                        }
        
        return result
    
    def compare_observations(self, obs1: Dict, obs2: Dict) -> Dict[str, Any]:
        """比较两个观测"""
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
                # 数组比较
                if obs1[key].shape != obs2[key].shape:
                    result['consistent'] = False
                    result['differences'][key] = f"Shape mismatch: {obs1[key].shape} vs {obs2[key].shape}"
                    continue
                
                max_diff = np.abs(obs1[key] - obs2[key]).max()
                if max_diff > self.tolerance:
                    result['consistent'] = False
                    result['differences'][key] = f"Max diff: {max_diff:.8f}"
            
            elif isinstance(obs1[key], (int, float)):
                # 标量比较
                diff = abs(obs1[key] - obs2[key])
                if diff > self.tolerance:
                    result['consistent'] = False
                    result['differences'][key] = f"Diff: {diff:.8f}"
        
        return result
    
    def compare_rewards(self, reward1: float, reward2: float, 
                       info1: Dict = None, info2: Dict = None) -> Dict[str, Any]:
        """比较两个奖励值"""
        result = {
            'consistent': True,
            'reward_diff': abs(reward1 - reward2),
            'reward1': float(reward1),
            'reward2': float(reward2)
        }
        
        if result['reward_diff'] > self.tolerance:
            result['consistent'] = False
        
        # 如果提供了info，也比较奖励组件
        if info1 and info2 and 'reward_components' in info1 and 'reward_components' in info2:
            result['component_comparison'] = self.compare_reward_components(
                info1['reward_components'], info2['reward_components']
            )
        
        return result
    
    def compare_reward_components(self, components1: Dict, components2: Dict) -> Dict[str, Any]:
        """比较奖励组件"""
        result = {
            'consistent': True,
            'differences': {}
        }
        
        all_keys = set(components1.keys()) | set(components2.keys())
        
        for key in all_keys:
            val1 = components1.get(key, 0.0)
            val2 = components2.get(key, 0.0)
            diff = abs(val1 - val2)
            
            if diff > self.tolerance:
                result['consistent'] = False
                result['differences'][key] = {
                    'diff': float(diff),
                    'value1': float(val1),
                    'value2': float(val2)
                }
        
        return result
    
    def run_comprehensive_comparison(self, env1, env2, action_sequence: List[int], 
                                   initial_seed: int = 0) -> Dict[str, Any]:
        """运行全面的环境比较测试"""
        if self.verbose:
            print(f"🔍 开始全面比较测试 (seed={initial_seed}, steps={len(action_sequence)})")
        
        # 重置环境
        obs1, info1 = env1.reset(seed=initial_seed)
        obs2, info2 = env2.reset(seed=initial_seed)
        
        step_results = []
        overall_consistent = True
        
        # 初始状态比较
        initial_obs_check = self.compare_observations(obs1, obs2)
        if not initial_obs_check['consistent']:
            overall_consistent = False
            step_results.append({
                'step': -1,
                'action': None,
                'observation_check': initial_obs_check
            })
        
        # 逐步执行和比较
        for step, action in enumerate(action_sequence):
            # 执行动作
            obs1, reward1, done1, trunc1, info1 = env1.step(action)
            obs2, reward2, done2, trunc2, info2 = env2.step(action)
            
            # 比较结果
            obs_check = self.compare_observations(obs1, obs2)
            reward_check = self.compare_rewards(reward1, reward2, info1, info2)
            done_check = (done1 == done2) and (trunc1 == trunc2)
            
            step_result = {
                'step': step,
                'action': action,
                'observation_check': obs_check,
                'reward_check': reward_check,
                'done_check': done_check,
                'step_consistent': obs_check['consistent'] and reward_check['consistent'] and done_check
            }
            
            if not step_result['step_consistent']:
                overall_consistent = False
            
            step_results.append(step_result)
            
            if done1 or done2:
                break
        
        return {
            'overall_consistent': overall_consistent,
            'total_steps': len(step_results),
            'consistent_steps': sum(1 for r in step_results if r.get('step_consistent', True)),
            'step_details': step_results,
            'summary': self._generate_comparison_summary(step_results, overall_consistent)
        }
    
    def _generate_comparison_summary(self, step_results: List[Dict], overall_consistent: bool) -> Dict[str, Any]:
        """生成比较测试摘要"""
        total_steps = len(step_results)
        consistent_steps = sum(1 for r in step_results if r.get('step_consistent', True))
        
        obs_failures = sum(1 for r in step_results if not r.get('observation_check', {}).get('consistent', True))
        reward_failures = sum(1 for r in step_results if not r.get('reward_check', {}).get('consistent', True))
        done_failures = sum(1 for r in step_results if not r.get('done_check', True))
        
        return {
            'overall_consistent': overall_consistent,
            'total_steps': total_steps,
            'consistent_steps': consistent_steps,
            'consistency_rate': consistent_steps / total_steps if total_steps > 0 else 0.0,
            'failure_breakdown': {
                'observation_failures': obs_failures,
                'reward_failures': reward_failures,
                'done_failures': done_failures
            }
        }
    
    def print_comparison_report(self, results: Dict[str, Any]):
        """打印比较测试报告"""
        print(f"\n{'='*60}")
        print("全面一致性测试报告")
        print(f"{'='*60}")
        
        summary = results['summary']
        print(f"总体一致性: {'✅ 通过' if summary['overall_consistent'] else '❌ 失败'}")
        print(f"测试步数: {summary['total_steps']}")
        print(f"一致步数: {summary['consistent_steps']}")
        print(f"一致性率: {summary['consistency_rate']*100:.1f}%")
        
        failures = summary['failure_breakdown']
        if any(failures.values()):
            print(f"\n失败分析:")
            if failures['observation_failures'] > 0:
                print(f"  观测不一致: {failures['observation_failures']} 步")
            if failures['reward_failures'] > 0:
                print(f"  奖励不一致: {failures['reward_failures']} 步")
            if failures['done_failures'] > 0:
                print(f"  完成状态不一致: {failures['done_failures']} 步")
        
        print(f"{'='*60}")
    
    def save_comparison_results(self, results: Dict[str, Any], filepath: str):
        """保存比较结果到文件"""
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'tolerance': self.tolerance,
            'results': results
        }
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        if self.verbose:
            print(f"✅ 测试结果已保存到: {filepath}")


class MultiVersionConsistencyChecker(ComprehensiveConsistencyChecker):
    """多版本一致性检查器"""
    
    def __init__(self, tolerance: float = 1e-6, verbose: bool = True):
        super().__init__(tolerance, verbose)
        self.version_results = {}
    
    def run_multi_version_test(self, env_classes: List, version_names: List[str], 
                              action_sequences: List[List[int]], seeds: List[int]) -> Dict[str, Any]:
        """运行多版本一致性测试"""
        if self.verbose:
            print(f"🚀 开始多版本一致性测试")
            print(f"   版本: {', '.join(version_names)}")
            print(f"   种子数: {len(seeds)}")
            print(f"   每种子步数: {len(action_sequences[0]) if action_sequences else 0}")
        
        all_results = {}
        
        for i, seed in enumerate(seeds):
            if self.verbose:
                print(f"\n🧪 测试种子 {seed} ({i+1}/{len(seeds)})")
            
            # 创建环境实例
            environments = [env_class() for env_class in env_classes]
            
            seed_results = {}
            
            # 比较每对版本
            for j in range(len(environments)):
                for k in range(j+1, len(environments)):
                    pair_name = f"{version_names[j]}_vs_{version_names[k]}"
                    
                    comparison_result = self.run_comprehensive_comparison(
                        environments[j], environments[k], 
                        action_sequences[i] if i < len(action_sequences) else action_sequences[0],
                        seed
                    )
                    
                    seed_results[pair_name] = comparison_result
                    
                    if self.verbose:
                        consistency_rate = comparison_result['summary']['consistency_rate']
                        status = "✅" if comparison_result['summary']['overall_consistent'] else "❌"
                        print(f"  {pair_name}: {status} {consistency_rate*100:.1f}%")
            
            all_results[f"seed_{seed}"] = seed_results
            
            # 清理环境
            for env in environments:
                env.close()
        
        # 生成总体统计
        overall_stats = self._generate_multi_version_stats(all_results, version_names)
        
        return {
            'seed_results': all_results,
            'overall_stats': overall_stats,
            'test_config': {
                'versions': version_names,
                'seeds': seeds,
                'tolerance': self.tolerance,
                'total_comparisons': len(seeds) * len(env_classes) * (len(env_classes) - 1) // 2
            }
        }
    
    def _generate_multi_version_stats(self, all_results: Dict, version_names: List[str]) -> Dict[str, Any]:
        """生成多版本测试统计"""
        stats = {}
        
        # 计算每个版本对的总体统计
        for j in range(len(version_names)):
            for k in range(j+1, len(version_names)):
                pair_name = f"{version_names[j]}_vs_{version_names[k]}"
                
                pair_consistencies = []
                pair_step_counts = []
                
                for seed_key, seed_results in all_results.items():
                    if pair_name in seed_results:
                        result = seed_results[pair_name]
                        pair_consistencies.append(result['summary']['consistency_rate'])
                        pair_step_counts.append(result['summary']['total_steps'])
                
                stats[pair_name] = {
                    'average_consistency': np.mean(pair_consistencies) if pair_consistencies else 0.0,
                    'min_consistency': np.min(pair_consistencies) if pair_consistencies else 0.0,
                    'max_consistency': np.max(pair_consistencies) if pair_consistencies else 0.0,
                    'total_seeds_tested': len(pair_consistencies),
                    'perfect_seeds': sum(1 for c in pair_consistencies if c >= 1.0),
                    'average_steps': np.mean(pair_step_counts) if pair_step_counts else 0
                }
        
        return stats