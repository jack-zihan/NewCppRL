#!/usr/bin/env python3
"""
路径规划算法测试器 - 核心测试引擎

简洁、清晰、直接的测试系统，用于评估不同路径规划算法的性能。
合并了原benchmark和experiment系统的核心功能，去除冗余抽象。
"""

import os
import sys
import yaml
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from scenarios import ScenarioBuilder
from metrics import MetricsCalculator
from plotter import ResultPlotter
from helpers import to_yx, to_xy, setup_logging

# 算法导入
from algorithms.base import BasePathPlanner
from algorithms.jump_planner import JumpPlanner
from algorithms.snake_planner import SnakePlanner, RSnakePlanner
from algorithms.bcp_planner import BcpPlanner
from algorithms.react_planner import ReactPlanner


class PathPlannerTester:
    """
    路径规划测试器
    
    负责运行算法测试、收集指标、生成报告。
    设计原则：简单直接，避免过度抽象。
    """
    
    def __init__(self, config_path: str):
        """
        初始化测试器
        
        Args:
            config_path: 配置文件路径（支持yaml格式）
        """
        self.config = self._load_config(config_path)
        self.output_dir = self._create_output_directory()
        self.logger = self._setup_logger()
        
        # 初始化组件
        self.scenario_builder = ScenarioBuilder(self.config.get('scenarios', {}))
        self.metrics_calculator = MetricsCalculator(self.config.get('metrics', {}))
        self.plotter = ResultPlotter(self.output_dir)
        
        # 算法注册表（简单明了）
        self.algorithm_registry = {
            'JUMP': JumpPlanner,
            'SNAKE': SnakePlanner,
            'R-SNAKE': RSnakePlanner,
            'BCP': BcpPlanner,
            'REACT': ReactPlanner,
        }
        
        # 初始化算法实例
        self.algorithms = self._initialize_algorithms()
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_output_directory(self) -> Path:
        """创建输出目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"test_results/{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = setup_logging(self.output_dir / "test.log")
        return logger
    
    def _initialize_algorithms(self) -> Dict[str, BasePathPlanner]:
        """初始化启用的算法"""
        algorithms = {}
        for algo_name, algo_config in self.config.get('algorithms', {}).items():
            if algo_config.get('enabled', True):
                if algo_name in self.algorithm_registry:
                    try:
                        # 简单直接的初始化，不需要复杂的工厂模式
                        algo_class = self.algorithm_registry[algo_name]
                        algorithms[algo_name] = algo_class(**algo_config.get('params', {}))
                        self.logger.info(f"✅ 初始化算法: {algo_name}")
                    except Exception as e:
                        self.logger.warning(f"⚠️  算法 {algo_name} 初始化失败: {e}")
                else:
                    self.logger.warning(f"⚠️  算法 {algo_name} 未在注册表中找到")
        
        if not algorithms:
            self.logger.warning("⚠️  没有成功初始化任何算法！")
        
        return algorithms
    
    def run_tests(self) -> Dict[str, Any]:
        """
        运行测试 - 主入口
        
        Returns:
            测试结果字典
        """
        self.logger.info(f"开始测试，共 {len(self.algorithms)} 个算法")
        
        # 生成测试场景
        scenarios = self.scenario_builder.build_all()
        self.logger.info(f"生成 {len(scenarios)} 个测试场景")
        
        # 运行测试
        all_results = []
        for scenario in scenarios:
            scenario_results = self._test_scenario(scenario)
            all_results.extend(scenario_results)
        
        # 分析结果
        analysis = self._analyze_results(all_results)
        
        # 生成可视化
        self.plotter.plot_all(all_results, analysis)
        
        # 保存结果
        self._save_results(all_results, analysis)
        
        self.logger.info(f"测试完成，结果保存到: {self.output_dir}")
        
        return {
            'output_dir': str(self.output_dir),
            'num_algorithms': len(self.algorithms),
            'num_scenarios': len(scenarios),
            'num_tests': len(all_results),
            'analysis': analysis
        }
    
    def _test_scenario(self, scenario: Dict) -> List[Dict]:
        """
        测试单个场景
        
        Args:
            scenario: 场景配置
            
        Returns:
            该场景下所有算法的测试结果
        """
        results = []
        scenario_id = scenario['id']
        
        for algo_name, algorithm in self.algorithms.items():
            self.logger.debug(f"测试 {algo_name} 在场景 {scenario_id}")
            
            try:
                # 创建环境
                env = self._create_environment(scenario)
                
                # 运行算法
                start_time = time.time()
                trajectory = self._run_algorithm(algorithm, env, scenario)
                run_time = time.time() - start_time
                
                # 计算指标
                metrics = self.metrics_calculator.calculate(trajectory, env, scenario)
                
                # 记录结果
                result = {
                    'algorithm': algo_name,
                    'scenario_id': scenario_id,
                    'trajectory': trajectory,
                    'metrics': metrics,
                    'run_time': run_time,
                    'success': metrics.get('completed', False)
                }
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"测试失败 {algo_name}@{scenario_id}: {e}")
                results.append({
                    'algorithm': algo_name,
                    'scenario_id': scenario_id,
                    'error': str(e),
                    'success': False
                })
        
        return results
    
    def _create_environment(self, scenario: Dict):
        """
        创建测试环境
        
        简化版环境创建，去除复杂的环境管理器
        """
        # 这里根据实际环境接口来实现
        # 暂时返回模拟环境
        class SimpleEnv:
            def __init__(self, scenario):
                self.scenario = scenario
                self.agent_position = scenario.get('start_position', [0, 0])
                self.obstacles = scenario.get('obstacles', [])
                self.boundaries = scenario.get('boundaries', [[0, 0], [100, 100]])
                
            def reset(self):
                self.agent_position = self.scenario.get('start_position', [0, 0])
                return self.get_observation()
            
            def step(self, action):
                # 简化的环境步进
                self.agent_position = action  # 直接设置位置（简化版）
                obs = self.get_observation()
                done = self._check_done()
                reward = self._calculate_reward()
                return obs, reward, done, {}
            
            def get_observation(self):
                return {'agent_position': self.agent_position}
            
            def _check_done(self):
                # 简化的完成检查
                return False
            
            def _calculate_reward(self):
                return 0.0
        
        return SimpleEnv(scenario)
    
    def _run_algorithm(self, algorithm: BasePathPlanner, env, scenario: Dict) -> List[Tuple[float, float]]:
        """
        运行算法并收集轨迹
        
        Args:
            algorithm: 路径规划算法
            env: 测试环境
            scenario: 场景配置
            
        Returns:
            轨迹点列表
        """
        trajectory = []
        obs = env.reset()
        max_steps = self.config.get('max_steps', 1000)
        
        for step in range(max_steps):
            # 获取算法决策
            action = algorithm.get_action(obs)
            
            # 环境步进
            obs, reward, done, info = env.step(action)
            
            # 记录轨迹
            position = obs.get('agent_position', [0, 0])
            trajectory.append(tuple(position))
            
            if done:
                break
        
        return trajectory
    
    def _analyze_results(self, results: List[Dict]) -> Dict:
        """
        分析测试结果
        
        简化的分析逻辑，专注核心指标
        """
        analysis = {
            'summary': {},
            'rankings': {},
            'statistics': {}
        }
        
        # 按算法分组
        by_algorithm = {}
        for result in results:
            algo = result['algorithm']
            if algo not in by_algorithm:
                by_algorithm[algo] = []
            by_algorithm[algo].append(result)
        
        # 计算每个算法的统计指标
        for algo, algo_results in by_algorithm.items():
            metrics_list = [r.get('metrics', {}) for r in algo_results if r.get('success', False)]
            
            if metrics_list:
                # 计算平均值
                avg_coverage = np.mean([m.get('coverage_rate', 0) for m in metrics_list])
                avg_path_length = np.mean([m.get('path_length', 0) for m in metrics_list])
                success_rate = len(metrics_list) / len(algo_results)
                
                analysis['summary'][algo] = {
                    'average_coverage': avg_coverage,
                    'average_path_length': avg_path_length,
                    'success_rate': success_rate,
                    'num_tests': len(algo_results)
                }
        
        # 生成排名
        if analysis['summary']:
            # 按覆盖率排名
            coverage_ranking = sorted(
                analysis['summary'].items(),
                key=lambda x: x[1]['average_coverage'],
                reverse=True
            )
            analysis['rankings']['by_coverage'] = [algo for algo, _ in coverage_ranking]
            
            # 按路径长度排名（越短越好）
            path_ranking = sorted(
                analysis['summary'].items(),
                key=lambda x: x[1]['average_path_length']
            )
            analysis['rankings']['by_path_length'] = [algo for algo, _ in path_ranking]
        
        return analysis
    
    def _save_results(self, results: List[Dict], analysis: Dict):
        """保存测试结果"""
        import json
        
        # 保存原始结果
        with open(self.output_dir / 'raw_results.json', 'w') as f:
            # 转换为可序列化格式
            serializable_results = []
            for r in results:
                sr = r.copy()
                if 'trajectory' in sr:
                    sr['trajectory'] = [list(p) for p in sr['trajectory']]
                serializable_results.append(sr)
            json.dump(serializable_results, f, indent=2)
        
        # 保存分析结果
        with open(self.output_dir / 'analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # 保存配置备份
        with open(self.output_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)
        
        # 生成Markdown报告
        self._generate_report(analysis)
    
    def _generate_report(self, analysis: Dict):
        """生成测试报告"""
        report = ["# 路径规划算法测试报告\n"]
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 摘要
        report.append("## 测试摘要\n")
        report.append(f"- 测试算法数: {len(self.algorithms)}\n")
        report.append(f"- 测试场景数: {len(self.scenario_builder.build_all())}\n")
        
        # 算法性能
        report.append("\n## 算法性能\n")
        report.append("| 算法 | 平均覆盖率 | 平均路径长度 | 成功率 |\n")
        report.append("|------|------------|--------------|--------|\n")
        
        for algo, stats in analysis['summary'].items():
            report.append(f"| {algo} | {stats['average_coverage']:.2%} | "
                         f"{stats['average_path_length']:.1f} | "
                         f"{stats['success_rate']:.2%} |\n")
        
        # 排名
        report.append("\n## 算法排名\n")
        report.append(f"### 覆盖率排名\n")
        for i, algo in enumerate(analysis['rankings'].get('by_coverage', []), 1):
            report.append(f"{i}. {algo}\n")
        
        # 保存报告
        with open(self.output_dir / 'report.md', 'w') as f:
            f.writelines(report)


def main():
    """主函数 - 命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='路径规划算法测试器')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--quick', action='store_true', help='快速测试模式')
    parser.add_argument('--parallel', action='store_true', help='并行执行')
    
    args = parser.parse_args()
    
    # 运行测试
    tester = PathPlannerTester(args.config)
    results = tester.run_tests()
    
    print(f"✅ 测试完成！结果保存到: {results['output_dir']}")
    print(f"📊 测试了 {results['num_algorithms']} 个算法，{results['num_scenarios']} 个场景")


if __name__ == "__main__":
    main()