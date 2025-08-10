"""
结果分析器 - 自动分析和汇总基准测试结果

提供统计分析、排名和性能对比
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ResultAnalyzer:
    """
    结果分析器
    
    分析基准测试结果，生成统计报告和排名
    """
    
    def __init__(self, output_dir: Path):
        """
        初始化结果分析器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.analysis_dir = self.output_dir / 'analysis'
        self.analysis_dir.mkdir(exist_ok=True)
        
        logger.info(f"结果分析器初始化 - 输出目录: {output_dir}")
    
    def analyze(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析测试结果
        
        Args:
            results: 测试结果列表
            
        Returns:
            分析摘要
        """
        if not results:
            logger.warning("没有结果可分析")
            return {}
        
        # 转换为DataFrame便于分析
        df = self._results_to_dataframe(results)
        
        # 保存原始数据
        df.to_csv(self.analysis_dir / 'raw_results.csv', index=False)
        
        analysis = {
            'overall_statistics': self._compute_overall_statistics(df),
            'algorithm_rankings': self._compute_algorithm_rankings(df),
            'scenario_analysis': self._analyze_by_scenario(df),
            'difficulty_analysis': self._analyze_by_difficulty(df),
            'performance_matrix': self._create_performance_matrix(df),
            'best_worst_cases': self._find_best_worst_cases(df)
        }
        
        # 保存分析结果
        self._save_analysis(analysis)
        
        return analysis
    
    def _results_to_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """将结果转换为DataFrame"""
        data = []
        
        for result in results:
            if 'error' in result.get('metrics', {}):
                continue  # 跳过错误结果
                
            metrics = result['metrics']
            scenario = result['scenario']
            
            row = {
                'algorithm': result['algorithm'],
                'scenario_id': scenario['scenario_id'],
                'seed': scenario['seed'],
                'difficulty': scenario['difficulty'],
                'weed_distribution': scenario['weed_distribution'],
                'noise_level': scenario['noise_level'],
                'runtime': result['runtime'],
                
                # 覆盖率指标
                'final_coverage': metrics.get('final_coverage', 0),
                'coverage_90_distance': metrics.get('coverage_90_distance', -1),
                'coverage_95_distance': metrics.get('coverage_95_distance', -1),
                'coverage_98_distance': metrics.get('coverage_98_distance', -1),
                
                # 路径指标
                'total_distance': metrics.get('total_distance', 0),
                'total_steps': metrics.get('total_steps', 0),
                'path_smoothness': metrics.get('path_smoothness', 0),
                
                # 碰撞指标
                'collision_occurred': metrics.get('collision_occurred', False),
                'collision_distance': metrics.get('collision_distance', -1),
                
                # 效率指标
                'coverage_efficiency': metrics.get('coverage_efficiency', 0),
                'time_efficiency': metrics.get('time_efficiency', 0),
                'overall_efficiency_score': metrics.get('overall_efficiency_score', 0)
            }
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _compute_overall_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算整体统计信息"""
        stats = {}
        
        # 按算法分组
        for alg in df['algorithm'].unique():
            alg_df = df[df['algorithm'] == alg]
            
            stats[alg] = {
                'total_tests': len(alg_df),
                'success_rate': (alg_df['final_coverage'] >= 0.98).mean(),
                'collision_rate': alg_df['collision_occurred'].mean(),
                
                # 覆盖率统计
                'coverage_mean': alg_df['final_coverage'].mean(),
                'coverage_std': alg_df['final_coverage'].std(),
                'coverage_min': alg_df['final_coverage'].min(),
                'coverage_max': alg_df['final_coverage'].max(),
                
                # 路径长度统计（达到各覆盖率的平均路径）
                'path_90_mean': alg_df[alg_df['coverage_90_distance'] > 0]['coverage_90_distance'].mean(),
                'path_95_mean': alg_df[alg_df['coverage_95_distance'] > 0]['coverage_95_distance'].mean(),
                'path_98_mean': alg_df[alg_df['coverage_98_distance'] > 0]['coverage_98_distance'].mean(),
                
                # 效率统计
                'efficiency_mean': alg_df['overall_efficiency_score'].mean(),
                'efficiency_std': alg_df['overall_efficiency_score'].std(),
                
                # 运行时间统计
                'runtime_mean': alg_df['runtime'].mean(),
                'runtime_std': alg_df['runtime'].std()
            }
        
        return stats
    
    def _compute_algorithm_rankings(self, df: pd.DataFrame) -> Dict[str, List[Tuple[str, float]]]:
        """计算算法排名"""
        rankings = {}
        
        # 关键指标排名
        ranking_metrics = [
            ('success_rate', 'desc'),  # 成功率（高到低）
            ('collision_rate', 'asc'),  # 碰撞率（低到高）
            ('path_98_mean', 'asc'),    # 达到98%覆盖的平均路径（短到长）
            ('efficiency_score', 'desc'),  # 效率分数（高到低）
            ('runtime_mean', 'asc')     # 平均运行时间（快到慢）
        ]
        
        for metric_name, order in ranking_metrics:
            metric_values = []
            
            for alg in df['algorithm'].unique():
                alg_df = df[df['algorithm'] == alg]
                
                if metric_name == 'success_rate':
                    value = (alg_df['final_coverage'] >= 0.98).mean()
                elif metric_name == 'collision_rate':
                    value = alg_df['collision_occurred'].mean()
                elif metric_name == 'path_98_mean':
                    valid_df = alg_df[alg_df['coverage_98_distance'] > 0]
                    value = valid_df['coverage_98_distance'].mean() if len(valid_df) > 0 else float('inf')
                elif metric_name == 'efficiency_score':
                    value = alg_df['overall_efficiency_score'].mean()
                elif metric_name == 'runtime_mean':
                    value = alg_df['runtime'].mean()
                else:
                    value = 0
                
                metric_values.append((alg, value))
            
            # 排序
            reverse = (order == 'desc')
            metric_values.sort(key=lambda x: x[1], reverse=reverse)
            
            rankings[metric_name] = metric_values
        
        # 计算综合排名
        rankings['overall'] = self._compute_overall_ranking(df)
        
        return rankings
    
    def _compute_overall_ranking(self, df: pd.DataFrame) -> List[Tuple[str, float]]:
        """计算综合排名"""
        scores = {}
        
        # 权重配置
        weights = {
            'success_rate': 0.3,
            'collision_rate': 0.2,
            'path_efficiency': 0.2,
            'coverage_efficiency': 0.2,
            'runtime': 0.1
        }
        
        for alg in df['algorithm'].unique():
            alg_df = df[df['algorithm'] == alg]
            
            # 计算各项得分（归一化到0-1）
            success_score = (alg_df['final_coverage'] >= 0.98).mean()
            collision_score = 1 - alg_df['collision_occurred'].mean()  # 碰撞越少越好
            
            # 路径效率（路径越短越好）
            valid_path = alg_df[alg_df['coverage_98_distance'] > 0]['coverage_98_distance']
            if len(valid_path) > 0:
                # 归一化：假设最好的路径是500，最差是2000
                path_score = max(0, min(1, (2000 - valid_path.mean()) / 1500))
            else:
                path_score = 0
            
            coverage_efficiency_score = alg_df['coverage_efficiency'].mean()
            
            # 运行时间得分（时间越短越好）
            # 归一化：假设最快0.1秒，最慢10秒
            runtime_score = max(0, min(1, (10 - alg_df['runtime'].mean()) / 9.9))
            
            # 计算加权总分
            total_score = (
                weights['success_rate'] * success_score +
                weights['collision_rate'] * collision_score +
                weights['path_efficiency'] * path_score +
                weights['coverage_efficiency'] * coverage_efficiency_score +
                weights['runtime'] * runtime_score
            )
            
            scores[alg] = total_score
        
        # 排序返回
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    def _analyze_by_scenario(self, df: pd.DataFrame) -> Dict[str, Any]:
        """按场景分析"""
        scenario_analysis = {}
        
        for scenario_id in df['scenario_id'].unique():
            scenario_df = df[df['scenario_id'] == scenario_id]
            
            # 找出该场景下表现最好的算法
            best_algorithm = scenario_df.loc[
                scenario_df['overall_efficiency_score'].idxmax(), 'algorithm'
            ] if len(scenario_df) > 0 else None
            
            scenario_analysis[scenario_id] = {
                'algorithms_tested': list(scenario_df['algorithm'].unique()),
                'best_algorithm': best_algorithm,
                'avg_coverage': scenario_df['final_coverage'].mean(),
                'collision_count': scenario_df['collision_occurred'].sum(),
                'difficulty': scenario_df['difficulty'].iloc[0] if len(scenario_df) > 0 else 'unknown'
            }
        
        return scenario_analysis
    
    def _analyze_by_difficulty(self, df: pd.DataFrame) -> Dict[str, Any]:
        """按难度分析"""
        difficulty_analysis = {}
        
        for difficulty in df['difficulty'].unique():
            diff_df = df[df['difficulty'] == difficulty]
            
            # 计算每个算法在该难度下的表现
            algorithm_performance = {}
            for alg in diff_df['algorithm'].unique():
                alg_diff_df = diff_df[diff_df['algorithm'] == alg]
                
                algorithm_performance[alg] = {
                    'success_rate': (alg_diff_df['final_coverage'] >= 0.98).mean(),
                    'avg_coverage': alg_diff_df['final_coverage'].mean(),
                    'collision_rate': alg_diff_df['collision_occurred'].mean(),
                    'avg_distance': alg_diff_df['total_distance'].mean()
                }
            
            # 找出该难度下最佳算法
            if algorithm_performance:
                best_alg = max(
                    algorithm_performance.items(),
                    key=lambda x: x[1]['success_rate']
                )[0]
            else:
                best_alg = None
            
            difficulty_analysis[difficulty] = {
                'algorithm_performance': algorithm_performance,
                'best_algorithm': best_alg,
                'overall_success_rate': (diff_df['final_coverage'] >= 0.98).mean(),
                'overall_collision_rate': diff_df['collision_occurred'].mean()
            }
        
        return difficulty_analysis
    
    def _create_performance_matrix(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """创建性能矩阵"""
        matrix = {}
        
        algorithms = df['algorithm'].unique()
        difficulties = df['difficulty'].unique()
        
        for alg in algorithms:
            matrix[alg] = {}
            for diff in difficulties:
                subset = df[(df['algorithm'] == alg) & (df['difficulty'] == diff)]
                
                if len(subset) > 0:
                    # 使用成功率作为性能指标
                    performance = (subset['final_coverage'] >= 0.98).mean()
                else:
                    performance = 0
                
                matrix[alg][diff] = performance
        
        return matrix
    
    def _find_best_worst_cases(self, df: pd.DataFrame) -> Dict[str, Any]:
        """找出最佳和最差案例"""
        best_worst = {
            'best_cases': [],
            'worst_cases': []
        }
        
        # 最佳案例：效率分数最高的前5个
        best_idx = df.nlargest(5, 'overall_efficiency_score').index
        for idx in best_idx:
            row = df.loc[idx]
            best_worst['best_cases'].append({
                'algorithm': row['algorithm'],
                'scenario': row['scenario_id'],
                'coverage': row['final_coverage'],
                'distance': row['total_distance'],
                'efficiency': row['overall_efficiency_score']
            })
        
        # 最差案例：发生碰撞或覆盖率最低的前5个
        worst_df = df[(df['collision_occurred'] == True) | (df['final_coverage'] < 0.5)]
        if len(worst_df) > 0:
            worst_idx = worst_df.nsmallest(min(5, len(worst_df)), 'final_coverage').index
            for idx in worst_idx:
                row = df.loc[idx]
                best_worst['worst_cases'].append({
                    'algorithm': row['algorithm'],
                    'scenario': row['scenario_id'],
                    'coverage': row['final_coverage'],
                    'collision': row['collision_occurred'],
                    'efficiency': row['overall_efficiency_score']
                })
        
        return best_worst
    
    def _save_analysis(self, analysis: Dict[str, Any]):
        """保存分析结果"""
        # 保存为JSON
        json_path = self.analysis_dir / 'analysis_results.json'
        
        # 创建可序列化的副本
        serializable_analysis = {}
        for key, value in analysis.items():
            if isinstance(value, dict):
                serializable_analysis[key] = self._make_serializable(value)
            else:
                serializable_analysis[key] = value
        
        with open(json_path, 'w') as f:
            json.dump(serializable_analysis, f, indent=2, default=self._json_converter)
        
        logger.info(f"分析结果已保存: {json_path}")
        
        # 生成Markdown报告
        self._generate_markdown_report(analysis)
    
    def _make_serializable(self, obj: Any) -> Any:
        """递归转换对象为可序列化格式"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj) if 'pd' in globals() else False:
            return None
        else:
            return obj
    
    def _json_converter(self, obj):
        """JSON默认转换器"""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return str(obj)
    
    def _generate_markdown_report(self, analysis: Dict[str, Any]):
        """生成Markdown格式的报告"""
        report_path = self.analysis_dir / 'analysis_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# 基准测试分析报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 算法排名
            f.write("## 算法综合排名\n\n")
            if 'overall' in analysis.get('algorithm_rankings', {}):
                f.write("| 排名 | 算法 | 综合得分 |\n")
                f.write("|------|------|----------|\n")
                for i, (alg, score) in enumerate(analysis['algorithm_rankings']['overall'], 1):
                    f.write(f"| {i} | {alg} | {score:.3f} |\n")
                f.write("\n")
            
            # 整体统计
            f.write("## 整体性能统计\n\n")
            if 'overall_statistics' in analysis:
                f.write("| 算法 | 成功率 | 碰撞率 | 平均覆盖率 | 平均效率 |\n")
                f.write("|------|--------|--------|------------|----------|\n")
                for alg, stats in analysis['overall_statistics'].items():
                    f.write(f"| {alg} | {stats['success_rate']:.1%} | "
                           f"{stats['collision_rate']:.1%} | "
                           f"{stats['coverage_mean']:.1%} | "
                           f"{stats['efficiency_mean']:.3f} |\n")
                f.write("\n")
            
            # 难度分析
            f.write("## 不同难度下的表现\n\n")
            if 'difficulty_analysis' in analysis:
                for diff, diff_data in analysis['difficulty_analysis'].items():
                    f.write(f"### {diff}难度\n\n")
                    f.write(f"- 最佳算法: {diff_data.get('best_algorithm', 'N/A')}\n")
                    f.write(f"- 整体成功率: {diff_data.get('overall_success_rate', 0):.1%}\n")
                    f.write(f"- 整体碰撞率: {diff_data.get('overall_collision_rate', 0):.1%}\n\n")
            
            # 最佳和最差案例
            f.write("## 典型案例\n\n")
            if 'best_worst_cases' in analysis:
                f.write("### 最佳案例\n\n")
                for case in analysis['best_worst_cases'].get('best_cases', []):
                    f.write(f"- **{case['algorithm']}** 在 {case['scenario']}: "
                           f"覆盖率 {case['coverage']:.1%}, "
                           f"效率 {case['efficiency']:.3f}\n")
                
                f.write("\n### 最差案例\n\n")
                for case in analysis['best_worst_cases'].get('worst_cases', []):
                    f.write(f"- **{case['algorithm']}** 在 {case['scenario']}: "
                           f"覆盖率 {case['coverage']:.1%}, "
                           f"碰撞 {case['collision']}\n")
        
        logger.info(f"Markdown报告已生成: {report_path}")