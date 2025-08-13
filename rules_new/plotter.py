#!/usr/bin/env python3
"""
结果绘图器 - 可视化测试结果

生成各种图表和可视化，包括：
- 算法性能对比图
- 轨迹可视化
- 覆盖率热图
- 统计报告
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import seaborn as sns

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class ResultPlotter:
    """
    结果绘图器
    
    负责生成各种可视化图表。
    设计原则：每个方法生成一种图表，职责单一。
    """
    
    def __init__(self, output_dir: Path):
        """
        初始化绘图器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / 'figures'
        self.figures_dir.mkdir(exist_ok=True)
        
        # 算法颜色映射（保持一致性）
        self.color_map = {
            'JUMP': '#FF6B6B',
            'SNAKE': '#4ECDC4',
            'R-SNAKE': '#45B7D1',
            'BCP': '#96CEB4',
            'REACT': '#FECA57',
            'NN_baseline': '#9B59B6',
            'NN_ours': '#E74C3C'
        }
    
    def plot_all(self, results: List[Dict], analysis: Dict):
        """
        生成所有图表
        
        Args:
            results: 测试结果列表
            analysis: 分析结果
        """
        # 1. 算法性能对比
        self.plot_algorithm_comparison(analysis)
        
        # 2. 覆盖率曲线
        self.plot_coverage_curves(results)
        
        # 3. 路径长度分布
        self.plot_path_length_distribution(results)
        
        # 4. 效率热图
        self.plot_efficiency_heatmap(results)
        
        # 5. 选择性地绘制轨迹（只绘制前几个场景）
        self.plot_sample_trajectories(results[:3])
    
    def plot_algorithm_comparison(self, analysis: Dict):
        """
        绘制算法性能对比图
        
        Args:
            analysis: 分析结果
        """
        if not analysis.get('summary'):
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        algorithms = list(analysis['summary'].keys())
        colors = [self.color_map.get(algo, '#333333') for algo in algorithms]
        
        # 覆盖率对比
        coverages = [analysis['summary'][algo]['average_coverage'] for algo in algorithms]
        axes[0].bar(algorithms, coverages, color=colors)
        axes[0].set_title('平均覆盖率')
        axes[0].set_ylabel('覆盖率')
        axes[0].set_ylim(0, 1)
        axes[0].tick_params(axis='x', rotation=45)
        
        # 路径长度对比
        path_lengths = [analysis['summary'][algo]['average_path_length'] for algo in algorithms]
        axes[1].bar(algorithms, path_lengths, color=colors)
        axes[1].set_title('平均路径长度')
        axes[1].set_ylabel('路径长度')
        axes[1].tick_params(axis='x', rotation=45)
        
        # 成功率对比
        success_rates = [analysis['summary'][algo]['success_rate'] for algo in algorithms]
        axes[2].bar(algorithms, success_rates, color=colors)
        axes[2].set_title('成功率')
        axes[2].set_ylabel('成功率')
        axes[2].set_ylim(0, 1)
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'algorithm_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_coverage_curves(self, results: List[Dict]):
        """
        绘制覆盖率增长曲线
        
        Args:
            results: 测试结果列表
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 按算法分组
        by_algorithm = {}
        for result in results:
            if not result.get('success', False):
                continue
            algo = result['algorithm']
            if algo not in by_algorithm:
                by_algorithm[algo] = []
            by_algorithm[algo].append(result)
        
        # 绘制每个算法的覆盖率曲线
        for algo, algo_results in by_algorithm.items():
            # 提取覆盖率数据
            threshold_90 = []
            threshold_95 = []
            threshold_98 = []
            
            for r in algo_results:
                metrics = r.get('metrics', {})
                threshold_90.append(metrics.get('coverage_90_length', np.nan))
                threshold_95.append(metrics.get('coverage_95_length', np.nan))
                threshold_98.append(metrics.get('coverage_98_length', np.nan))
            
            # 计算平均值（忽略inf和nan）
            avg_90 = np.nanmean([x for x in threshold_90 if x != np.inf])
            avg_95 = np.nanmean([x for x in threshold_95 if x != np.inf])
            avg_98 = np.nanmean([x for x in threshold_98 if x != np.inf])
            
            # 绘制曲线
            coverages = [0.90, 0.95, 0.98]
            lengths = [avg_90, avg_95, avg_98]
            
            # 过滤掉nan值
            valid_points = [(c, l) for c, l in zip(coverages, lengths) if not np.isnan(l)]
            if valid_points:
                coverages, lengths = zip(*valid_points)
                ax.plot(coverages, lengths, marker='o', 
                       label=algo, color=self.color_map.get(algo, '#333333'),
                       linewidth=2)
        
        ax.set_xlabel('覆盖率')
        ax.set_ylabel('路径长度')
        ax.set_title('覆盖率-路径长度关系')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'coverage_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_path_length_distribution(self, results: List[Dict]):
        """
        绘制路径长度分布图
        
        Args:
            results: 测试结果列表
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 收集每个算法的路径长度数据
        data_for_plot = []
        labels = []
        
        by_algorithm = {}
        for result in results:
            if not result.get('success', False):
                continue
            algo = result['algorithm']
            if algo not in by_algorithm:
                by_algorithm[algo] = []
            path_length = result.get('metrics', {}).get('path_length', 0)
            by_algorithm[algo].append(path_length)
        
        # 准备箱线图数据
        for algo, lengths in by_algorithm.items():
            if lengths:
                data_for_plot.append(lengths)
                labels.append(algo)
        
        # 绘制箱线图
        if data_for_plot:
            bp = ax.boxplot(data_for_plot, labels=labels, patch_artist=True)
            
            # 设置颜色
            for patch, label in zip(bp['boxes'], labels):
                patch.set_facecolor(self.color_map.get(label, '#333333'))
                patch.set_alpha(0.7)
        
        ax.set_xlabel('算法')
        ax.set_ylabel('路径长度')
        ax.set_title('路径长度分布')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'path_length_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_efficiency_heatmap(self, results: List[Dict]):
        """
        绘制效率热图
        
        Args:
            results: 测试结果列表
        """
        # 准备数据矩阵
        algorithms = []
        scenarios = []
        efficiency_matrix = []
        
        # 收集唯一的算法和场景
        for result in results:
            if result['algorithm'] not in algorithms:
                algorithms.append(result['algorithm'])
            if result['scenario_id'] not in scenarios:
                scenarios.append(result['scenario_id'])
        
        # 创建效率矩阵
        efficiency_matrix = np.zeros((len(algorithms), len(scenarios)))
        
        for result in results:
            if not result.get('success', False):
                continue
            
            algo_idx = algorithms.index(result['algorithm'])
            scenario_idx = scenarios.index(result['scenario_id'])
            
            # 使用综合效率评分
            efficiency = result.get('metrics', {}).get('overall_score', 0)
            efficiency_matrix[algo_idx, scenario_idx] = efficiency
        
        # 绘制热图
        fig, ax = plt.subplots(figsize=(12, 6))
        
        im = ax.imshow(efficiency_matrix, cmap='YlOrRd', aspect='auto')
        
        # 设置刻度
        ax.set_xticks(np.arange(len(scenarios)))
        ax.set_yticks(np.arange(len(algorithms)))
        ax.set_xticklabels(scenarios)
        ax.set_yticklabels(algorithms)
        
        # 旋转场景标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('效率评分', rotation=270, labelpad=15)
        
        ax.set_title('算法效率热图')
        ax.set_xlabel('场景')
        ax.set_ylabel('算法')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'efficiency_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_sample_trajectories(self, sample_results: List[Dict]):
        """
        绘制样本轨迹
        
        Args:
            sample_results: 要绘制的样本结果
        """
        for result in sample_results:
            if not result.get('success', False) or 'trajectory' not in result:
                continue
            
            self.plot_single_trajectory(
                result['trajectory'],
                result['algorithm'],
                result['scenario_id']
            )
    
    def plot_single_trajectory(self, trajectory: List[Tuple[float, float]], 
                               algorithm: str, scenario_id: str):
        """
        绘制单个轨迹
        
        Args:
            trajectory: 轨迹点列表
            algorithm: 算法名称
            scenario_id: 场景ID
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 提取轨迹坐标
        if trajectory:
            ys, xs = zip(*trajectory)
            
            # 绘制轨迹
            ax.plot(xs, ys, color=self.color_map.get(algorithm, '#333333'), 
                   linewidth=2, alpha=0.7, label=algorithm)
            
            # 标记起点和终点
            ax.scatter(xs[0], ys[0], color='green', s=100, marker='o', label='起点', zorder=5)
            ax.scatter(xs[-1], ys[-1], color='red', s=100, marker='s', label='终点', zorder=5)
        
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        ax.set_title(f'轨迹: {algorithm} @ {scenario_id}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 保存图片
        filename = f'trajectory_{algorithm}_{scenario_id}.png'
        plt.tight_layout()
        plt.savefig(self.figures_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_scenario_with_trajectory(self, scenario: Dict, trajectory: List[Tuple[float, float]], 
                                     algorithm: str, save_path: Optional[Path] = None):
        """
        绘制带场景元素的轨迹图
        
        Args:
            scenario: 场景配置
            trajectory: 轨迹点列表
            algorithm: 算法名称
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 绘制边界
        boundaries = scenario.get('boundaries', [])
        if boundaries:
            boundary_xs = [p[1] for p in boundaries]
            boundary_ys = [p[0] for p in boundaries]
            ax.plot(boundary_xs, boundary_ys, 'k-', linewidth=2, label='边界')
        
        # 绘制障碍物
        obstacles = scenario.get('obstacles', [])
        for obstacle in obstacles:
            pos = obstacle['position']
            size = obstacle['size']
            rect = patches.Rectangle(
                (pos[1] - size[1]/2, pos[0] - size[0]/2),
                size[1], size[0],
                linewidth=1, edgecolor='r', facecolor='red', alpha=0.3
            )
            ax.add_patch(rect)
        
        # 绘制轨迹
        if trajectory:
            ys, xs = zip(*trajectory)
            ax.plot(xs, ys, color=self.color_map.get(algorithm, '#333333'),
                   linewidth=2, alpha=0.7, label=algorithm)
            
            # 起点和终点
            ax.scatter(xs[0], ys[0], color='green', s=150, marker='o', 
                      label='起点', zorder=5, edgecolors='black', linewidth=2)
            ax.scatter(xs[-1], ys[-1], color='red', s=150, marker='s', 
                      label='终点', zorder=5, edgecolors='black', linewidth=2)
        
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        ax.set_title(f'{algorithm} - 场景 {scenario["id"]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 保存
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(self.figures_dir / f'scenario_{scenario["id"]}_{algorithm}.png', 
                       dpi=150, bbox_inches='tight')
        plt.close()