"""
测试辅助工具函数集合

提供通用的测试工具函数，包括环境比较、一致性验证、测试数据生成等
"""
import numpy as np
import random
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path


class TestDataGenerator:
    """测试数据生成器"""
    
    @staticmethod
    def generate_action_sequence(length: int, action_space_size: int, seed: Optional[int] = None) -> List[int]:
        """生成可重现的动作序列"""
        if seed is not None:
            random.seed(seed)
        return [random.randint(0, action_space_size - 1) for _ in range(length)]
    
    @staticmethod
    def generate_test_seeds(count: int, start: int = 0) -> List[int]:
        """生成测试种子列表"""
        return list(range(start, start + count))


class ConsistencyChecker:
    """一致性检查工具"""
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
    
    def compare_observations(self, obs1: Dict, obs2: Dict) -> Dict[str, Any]:
        """比较两个观测值"""
        result = {
            'consistent': True,
            'differences': {}
        }
        
        for key in obs1.keys():
            if key not in obs2:
                result['consistent'] = False
                result['differences'][key] = f"Missing in obs2"
                continue
            
            if isinstance(obs1[key], np.ndarray):
                diff = np.abs(obs1[key] - obs2[key]).max()
                if diff > self.tolerance:
                    result['consistent'] = False
                    result['differences'][key] = f"Max diff: {diff}"
            else:
                if abs(obs1[key] - obs2[key]) > self.tolerance:
                    result['consistent'] = False
                    result['differences'][key] = f"Diff: {obs1[key] - obs2[key]}"
        
        return result
    
    def compare_rewards(self, reward1: float, reward2: float) -> Dict[str, Any]:
        """比较两个奖励值"""
        diff = abs(reward1 - reward2)
        return {
            'consistent': diff <= self.tolerance,
            'difference': diff,
            'reward1': reward1,
            'reward2': reward2
        }


class TestReporter:
    """测试结果报告器"""
    
    @staticmethod
    def print_test_summary(test_name: str, results: Dict[str, Any]):
        """打印测试摘要"""
        print(f"\n{'='*60}")
        print(f"测试报告: {test_name}")
        print(f"{'='*60}")
        
        if 'total_tests' in results:
            print(f"总测试数: {results['total_tests']}")
            print(f"通过数: {results['passed']}")
            print(f"失败数: {results['failed']}")
            print(f"成功率: {results['passed']/results['total_tests']*100:.1f}%")
        
        if 'errors' in results and results['errors']:
            print(f"\n错误详情:")
            for error in results['errors'][:5]:  # 只显示前5个错误
                print(f"  - {error}")
    
    @staticmethod
    def save_test_results(results: Dict[str, Any], filepath: str):
        """保存测试结果到文件"""
        import json
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)


class EnvironmentTestSuite:
    """环境测试套件基类"""
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.checker = ConsistencyChecker(tolerance)
        self.reporter = TestReporter()
        
    def setup_test_environments(self, env_classes: List, **kwargs):
        """设置测试环境"""
        environments = []
        for env_class in env_classes:
            env = env_class(**kwargs)
            environments.append(env)
        return environments
    
    def run_consistency_test(self, environments: List, num_steps: int = 10, num_seeds: int = 3) -> Dict[str, Any]:
        """运行一致性测试"""
        results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        test_seeds = TestDataGenerator.generate_test_seeds(num_seeds)
        
        for seed in test_seeds:
            try:
                # 重置所有环境
                observations = []
                for env in environments:
                    obs, _ = env.reset(seed=seed)
                    observations.append(obs)
                
                # 运行测试步骤
                for step in range(num_steps):
                    action = random.randint(0, environments[0].action_space.n - 1)
                    
                    # 执行动作
                    step_results = []
                    for env in environments:
                        result = env.step(action)
                        step_results.append(result)
                    
                    # 检查一致性
                    for i in range(1, len(step_results)):
                        results['total_tests'] += 1
                        
                        # 比较奖励
                        reward_check = self.checker.compare_rewards(
                            step_results[0][1], step_results[i][1]
                        )
                        
                        if reward_check['consistent']:
                            results['passed'] += 1
                        else:
                            results['failed'] += 1
                            results['errors'].append(
                                f"Seed {seed}, Step {step}: Reward diff {reward_check['difference']}"
                            )
                        
                        # 比较观测
                        obs_check = self.checker.compare_observations(
                            step_results[0][0], step_results[i][0]
                        )
                        
                        if not obs_check['consistent']:
                            results['failed'] += 1
                            results['errors'].append(
                                f"Seed {seed}, Step {step}: Obs diff {obs_check['differences']}"
                            )
            
            except Exception as e:
                results['errors'].append(f"Seed {seed}: Exception {str(e)}")
        
        return results