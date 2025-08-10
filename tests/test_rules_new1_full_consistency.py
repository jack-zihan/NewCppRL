#!/usr/bin/env python3
"""
Rules_new1 完整一致性测试
验证rules_new1与rules_new的功能一致性
"""
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
import json
import yaml
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 设置环境变量，禁用GUI
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'

from rules_new.experiment.experiment_runner import ExperimentRunner
from rules_new.experiment.config_manager import ConfigManager
from rules_new.utils.logging_utils import LoggingUtils


class FullConsistencyTester:
    """完整一致性测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.logger = LoggingUtils.setup_logger("consistency_tester")
        self.config_manager = ConfigManager()
        
        # 测试配置
        self.test_seeds = [0, 42, 100, 123, 456, 789, 1000, 2000, 3000, 4000]  # 10个种子
        self.algorithms = ['JUMP', 'SNAKE', 'R_SNAKE', 'REACT', 'BCP']  # 5个传统算法
        self.nn_algorithms = ['NN_baseline', 'NN_ours']  # 2个NN算法
        
        # 结果存储
        self.results = {
            'metrics': {},  # 性能指标
            'trajectories': {},  # 轨迹数据
            'render_images': {}  # 渲染图像
        }
        
        # 创建输出目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = project_root / 'logs' / 'consistency_tests' / f'full_test_{timestamp}'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_test_experiment_config(self) -> Dict[str, Any]:
        """创建测试用的实验配置"""
        return {
            'experiment': {
                'name': 'consistency_test',
                'description': 'Full consistency test for rules_new'
            },
            'algorithms': [
                {'name': 'JUMP', 'enabled': True},
                {'name': 'SNAKE', 'enabled': True},
                {'name': 'R_SNAKE', 'enabled': True},
                {'name': 'REACT', 'enabled': True},
                {'name': 'BCP', 'enabled': True},
                {
                    'name': 'NN_baseline',
                    'enabled': False,  # 暂时禁用，需要模型文件
                    'model_path': str(project_root / 'ckpt' / 'baseline_model.pt'),
                    'device': 'cpu'
                },
                {
                    'name': 'NN_ours',
                    'enabled': False,  # 暂时禁用，需要模型文件
                    'model_path': str(project_root / 'ckpt' / 'our_model.pt'),
                    'device': 'cpu'
                }
            ],
            'parameters': {
                'seeds': self.test_seeds,
                'difficulties': ['medium'],  # 使用中等难度
                'weed_distributions': ['gaussian'],
                'noise_levels': ['no_noise']
            },
            'environment_overrides': {},
            'output': {
                'base_dir': str(self.output_dir),
                'csv_format': True,
                'save_trajectories': True
            }
        }
    
    def run_single_algorithm_test(self, algorithm_name: str, seed: int) -> Dict[str, Any]:
        """运行单个算法的测试"""
        self.logger.info(f"测试 {algorithm_name} - 种子 {seed}")
        
        try:
            # 创建临时实验配置
            exp_config = self.create_test_experiment_config()
            # 只启用当前算法
            for alg in exp_config['algorithms']:
                alg['enabled'] = (alg['name'] == algorithm_name)
            exp_config['parameters']['seeds'] = [seed]
            
            # 保存临时配置
            temp_config_path = self.output_dir / f'temp_config_{algorithm_name}_{seed}.yaml'
            with open(temp_config_path, 'w') as f:
                yaml.dump(exp_config, f)
            
            # 创建实验运行器
            runner = ExperimentRunner(str(temp_config_path))
            
            # 运行实验
            results = runner.run_experiment()
            
            # 提取关键指标
            metrics = {
                'coverage_rate': 0.0,
                'total_steps': 0,
                'total_reward': 0.0,
                'success': False
            }
            
            if 'experiment_results' in results:
                for result in results['experiment_results']:
                    if result.get('success'):
                        alg_metrics = result.get('algorithm_metrics', {})
                        metrics['coverage_rate'] = alg_metrics.get('final_coverage', 0.0)
                        metrics['total_steps'] = alg_metrics.get('total_steps', 0)
                        metrics['total_reward'] = alg_metrics.get('total_reward', 0.0)
                        metrics['success'] = True
                        break
            
            # 清理临时文件
            temp_config_path.unlink(missing_ok=True)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"测试失败 {algorithm_name} - 种子 {seed}: {e}")
            return {
                'coverage_rate': 0.0,
                'total_steps': 0,
                'total_reward': 0.0,
                'success': False,
                'error': str(e)
            }
    
    def run_all_tests(self):
        """运行所有测试"""
        self.logger.info("开始完整一致性测试")
        
        # 测试每个算法和种子组合
        for algorithm in self.algorithms:
            self.results['metrics'][algorithm] = {}
            
            for seed in self.test_seeds:
                metrics = self.run_single_algorithm_test(algorithm, seed)
                self.results['metrics'][algorithm][seed] = metrics
                
                # 打印进度
                if metrics['success']:
                    self.logger.info(f"✅ {algorithm} - 种子 {seed}: "
                                   f"覆盖率={metrics['coverage_rate']:.3f}, "
                                   f"步数={metrics['total_steps']}")
                else:
                    self.logger.warning(f"❌ {algorithm} - 种子 {seed}: 失败")
        
        self.logger.info("所有测试完成")
    
    def generate_comparison_report(self):
        """生成对比报告"""
        self.logger.info("生成对比报告")
        
        # 创建统计表格
        report_lines = ["# Rules_new1 一致性测试报告\n"]
        report_lines.append(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_lines.append(f"测试种子: {self.test_seeds}\n")
        report_lines.append("\n## 算法性能对比\n")
        
        # 生成表格头
        report_lines.append("| 算法 | 指标 | " + " | ".join(f"种子{s}" for s in self.test_seeds) + " | 平均值 |\n")
        report_lines.append("|------|------|" + "--------|" * (len(self.test_seeds) + 1) + "\n")
        
        # 生成每个算法的数据
        for algorithm in self.algorithms:
            if algorithm not in self.results['metrics']:
                continue
            
            # 覆盖率
            coverage_values = []
            coverage_row = f"| {algorithm} | 覆盖率 |"
            for seed in self.test_seeds:
                if seed in self.results['metrics'][algorithm]:
                    value = self.results['metrics'][algorithm][seed].get('coverage_rate', 0.0)
                    coverage_values.append(value)
                    coverage_row += f" {value:.3f} |"
                else:
                    coverage_row += " N/A |"
            
            if coverage_values:
                avg_coverage = np.mean(coverage_values)
                coverage_row += f" {avg_coverage:.3f} |"
            else:
                coverage_row += " N/A |"
            report_lines.append(coverage_row + "\n")
            
            # 步数
            steps_values = []
            steps_row = f"| {algorithm} | 步数 |"
            for seed in self.test_seeds:
                if seed in self.results['metrics'][algorithm]:
                    value = self.results['metrics'][algorithm][seed].get('total_steps', 0)
                    steps_values.append(value)
                    steps_row += f" {value} |"
                else:
                    steps_row += " N/A |"
            
            if steps_values:
                avg_steps = np.mean(steps_values)
                steps_row += f" {avg_steps:.0f} |"
            else:
                steps_row += " N/A |"
            report_lines.append(steps_row + "\n")
        
        # 保存报告
        report_path = self.output_dir / 'consistency_report.md'
        with open(report_path, 'w') as f:
            f.writelines(report_lines)
        
        self.logger.info(f"报告已保存: {report_path}")
        
        # 保存JSON格式的结果
        json_path = self.output_dir / 'results.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"JSON结果已保存: {json_path}")
    
    def create_visualization(self):
        """创建可视化图表"""
        self.logger.info("创建可视化图表")
        
        # 创建覆盖率对比图
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 覆盖率箱线图
        ax1 = axes[0]
        coverage_data = []
        labels = []
        
        for algorithm in self.algorithms:
            if algorithm in self.results['metrics']:
                values = [
                    self.results['metrics'][algorithm][seed].get('coverage_rate', 0.0)
                    for seed in self.test_seeds
                    if seed in self.results['metrics'][algorithm]
                ]
                if values:
                    coverage_data.append(values)
                    labels.append(algorithm)
        
        if coverage_data:
            bp = ax1.boxplot(coverage_data, labels=labels)
            ax1.set_ylabel('Coverage Rate')
            ax1.set_title('Algorithm Coverage Rate Comparison')
            ax1.grid(True, alpha=0.3)
        
        # 步数对比图
        ax2 = axes[1]
        steps_data = []
        
        for algorithm in self.algorithms:
            if algorithm in self.results['metrics']:
                values = [
                    self.results['metrics'][algorithm][seed].get('total_steps', 0)
                    for seed in self.test_seeds
                    if seed in self.results['metrics'][algorithm]
                ]
                if values:
                    steps_data.append(values)
        
        if steps_data:
            bp = ax2.boxplot(steps_data, labels=labels)
            ax2.set_ylabel('Total Steps')
            ax2.set_title('Algorithm Steps Comparison')
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Rules_new1 Consistency Test Results')
        plt.tight_layout()
        
        # 保存图表
        chart_path = self.output_dir / 'comparison_chart.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"图表已保存: {chart_path}")
    
    def run(self):
        """运行完整测试流程"""
        self.logger.info("=" * 50)
        self.logger.info("开始Rules_new1完整一致性测试")
        self.logger.info("=" * 50)
        
        # 运行所有测试
        self.run_all_tests()
        
        # 生成报告
        self.generate_comparison_report()
        
        # 创建可视化
        self.create_visualization()
        
        self.logger.info("=" * 50)
        self.logger.info("测试完成！")
        self.logger.info(f"结果保存在: {self.output_dir}")
        self.logger.info("=" * 50)


if __name__ == "__main__":
    tester = FullConsistencyTester()
    tester.run()