#!/usr/bin/env python3
"""
指标收集器
统一收集和处理算法执行的各项指标
"""

import json
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, log_dir: Path):
        """初始化收集器"""
        self.log_dir = log_dir
        self.metrics_dir = log_dir / 'metrics'
        self.metrics_dir.mkdir(exist_ok=True)
        
        # 指标定义
        self.metric_definitions = {
            'total_reward': {'name': '总奖励', 'unit': '', 'format': '.2f'},
            'coverage_rate': {'name': '覆盖率', 'unit': '%', 'format': '.2%'},
            'steps': {'name': '执行步数', 'unit': '步', 'format': 'd'},
            'execution_time': {'name': '执行时间', 'unit': '秒', 'format': '.2f'},
            'trajectory_length': {'name': '轨迹长度', 'unit': '米', 'format': '.2f'},
            'weeds_found': {'name': '发现杂草数', 'unit': '个', 'format': 'd'},
            'avg_reward_per_step': {'name': '平均步奖励', 'unit': '', 'format': '.4f'}
        }
        
        # 收集的所有指标
        self.collected_metrics = []
    
    def collect_from_rules_new(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """从rules_new结果中收集指标"""
        if not result:
            return None
        
        metrics = {
            'source': 'rules',
            'algorithm': result.get('algorithm', 'unknown'),
            'total_reward': result.get('total_reward', 0),
            'coverage_rate': result.get('coverage_rate', 0),
            'steps': result.get('steps', 0),
            'execution_time': result.get('execution_time', 0),
            'trajectory': result.get('trajectory', []),
            'actions': result.get('actions', []),
            'rewards': result.get('rewards', []),
            'final_frame': result.get('final_frame', None),
            'discovered_weeds': result.get('discovered_weeds', [])
        }
        
        # 计算衍生指标
        metrics['trajectory_length'] = self._calculate_trajectory_length(metrics['trajectory'])
        metrics['weeds_found'] = len(metrics['discovered_weeds'])
        metrics['avg_reward_per_step'] = metrics['total_reward'] / max(metrics['steps'], 1)
        
        # 计算统计信息
        if metrics['rewards']:
            metrics['reward_stats'] = {
                'min': np.min(metrics['rewards']),
                'max': np.max(metrics['rewards']),
                'mean': np.mean(metrics['rewards']),
                'std': np.std(metrics['rewards'])
            }
        
        self.collected_metrics.append(metrics)
        return metrics
    
    def collect_from_rules_new1(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """从rules_new1结果中收集指标"""
        # rules_new1的结果格式应该与适配器返回的格式一致
        if not result:
            return None
        
        metrics = {
            'source': 'rules_new',
            'algorithm': result.get('algorithm', 'unknown'),
            'total_reward': result.get('total_reward', 0),
            'coverage_rate': result.get('coverage_rate', 0),
            'steps': result.get('steps', 0),
            'execution_time': result.get('execution_time', 0),
            'trajectory': result.get('trajectory', []),
            'actions': result.get('actions', []),
            'rewards': result.get('rewards', []),
            'final_frame': result.get('final_frame', None),
            'discovered_weeds': result.get('discovered_weeds', [])
        }
        
        # 计算衍生指标
        metrics['trajectory_length'] = self._calculate_trajectory_length(metrics['trajectory'])
        metrics['weeds_found'] = len(metrics['discovered_weeds'])
        metrics['avg_reward_per_step'] = metrics['total_reward'] / max(metrics['steps'], 1)
        
        # 计算统计信息
        if metrics['rewards']:
            metrics['reward_stats'] = {
                'min': np.min(metrics['rewards']),
                'max': np.max(metrics['rewards']),
                'mean': np.mean(metrics['rewards']),
                'std': np.std(metrics['rewards'])
            }
        
        self.collected_metrics.append(metrics)
        return metrics
    
    def _calculate_trajectory_length(self, trajectory: List[List[float]]) -> float:
        """计算轨迹总长度"""
        if not trajectory or len(trajectory) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(trajectory)):
            p1 = np.array(trajectory[i-1])
            p2 = np.array(trajectory[i])
            total_length += np.linalg.norm(p2 - p1)
        
        return total_length
    
    def calculate_similarity(self, metrics1: Dict, metrics2: Dict) -> Dict[str, float]:
        """计算两组指标的相似度"""
        if not metrics1 or not metrics2:
            return {'overall': 0.0}
        
        similarity = {}
        
        # 1. 奖励相似度
        reward_diff = abs(metrics1['total_reward'] - metrics2['total_reward'])
        max_reward = max(abs(metrics1['total_reward']), abs(metrics2['total_reward']), 1)
        similarity['reward'] = max(0, 1 - reward_diff / max_reward) * 100
        
        # 2. 覆盖率相似度
        coverage_diff = abs(metrics1['coverage_rate'] - metrics2['coverage_rate'])
        similarity['coverage'] = max(0, 1 - coverage_diff) * 100
        
        # 3. 步数相似度
        steps_diff = abs(metrics1['steps'] - metrics2['steps'])
        max_steps = max(metrics1['steps'], metrics2['steps'], 1)
        similarity['steps'] = max(0, 1 - steps_diff / max_steps) * 100
        
        # 4. 轨迹相似度
        if metrics1['trajectory'] and metrics2['trajectory']:
            traj_sim = self._calculate_trajectory_similarity(
                metrics1['trajectory'], 
                metrics2['trajectory']
            )
            similarity['trajectory'] = traj_sim
        else:
            similarity['trajectory'] = 0.0
        
        # 5. 执行时间相似度（不太重要，权重较低）
        time_diff = abs(metrics1['execution_time'] - metrics2['execution_time'])
        max_time = max(metrics1['execution_time'], metrics2['execution_time'], 1)
        similarity['execution_time'] = max(0, 1 - time_diff / max_time) * 100
        
        # 计算加权平均相似度
        weights = {
            'reward': 0.3,
            'coverage': 0.3,
            'steps': 0.2,
            'trajectory': 0.15,
            'execution_time': 0.05
        }
        
        overall = sum(similarity[key] * weights[key] for key in weights)
        similarity['overall'] = overall
        
        return similarity
    
    def _calculate_trajectory_similarity(self, traj1: List, traj2: List) -> float:
        """计算轨迹相似度（使用DTW或简单的点对点距离）"""
        # 简化版：使用点对点平均距离
        min_len = min(len(traj1), len(traj2))
        if min_len == 0:
            return 0.0
        
        # 采样对齐
        sample_indices = np.linspace(0, min_len-1, min(100, min_len), dtype=int)
        
        total_distance = 0.0
        for idx in sample_indices:
            p1 = np.array(traj1[idx])
            p2 = np.array(traj2[idx])
            total_distance += np.linalg.norm(p1 - p2)
        
        avg_distance = total_distance / len(sample_indices)
        
        # 转换为相似度（假设平均距离10以内是很相似的）
        similarity = max(0, 1 - avg_distance / 10) * 100
        
        return similarity
    
    def save_metrics(self, algorithm: str, seed: int, metrics: Dict):
        """保存单个算法的指标"""
        filename = self.metrics_dir / f'{algorithm}_seed{seed}_metrics.json'
        
        # 移除不可序列化的内容
        save_data = {k: v for k, v in metrics.items() 
                    if k not in ['final_frame', 'trajectory', 'actions', 'rewards']}
        
        # 保存轨迹和动作为numpy文件
        if metrics.get('trajectory'):
            np.save(self.metrics_dir / f'{algorithm}_seed{seed}_trajectory.npy', 
                   metrics['trajectory'])
        
        if metrics.get('actions'):
            np.save(self.metrics_dir / f'{algorithm}_seed{seed}_actions.npy', 
                   metrics['actions'])
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
    
    def export_summary(self, all_results: Dict) -> Path:
        """导出CSV格式的汇总报告"""
        csv_path = self.log_dir / 'metrics_comparison.csv'
        
        rows = []
        headers = ['算法', '种子', '版本', '总奖励', '覆盖率(%)', '步数', 
                  '执行时间(s)', '轨迹长度', '平均步奖励', '一致性(%)']
        
        for algorithm, alg_results in all_results.items():
            for seed, seed_results in alg_results.get('seeds', {}).items():
                # rules_new指标
                if seed_results.get('metrics_old'):
                    m = seed_results['metrics_old']
                    rows.append([
                        algorithm, seed, 'rules',
                        f"{m.get('total_reward', 0):.2f}",
                        f"{m.get('coverage_rate', 0)*100:.2f}",
                        m.get('steps', 0),
                        f"{m.get('execution_time', 0):.2f}",
                        f"{m.get('trajectory_length', 0):.2f}",
                        f"{m.get('avg_reward_per_step', 0):.4f}",
                        '-'
                    ])
                
                # rules_new1指标
                if seed_results.get('metrics_new'):
                    m = seed_results['metrics_new']
                    consistency = seed_results.get('comparison', {}).get('similarity_score', 0)
                    rows.append([
                        algorithm, seed, 'rules_new',
                        f"{m.get('total_reward', 0):.2f}",
                        f"{m.get('coverage_rate', 0)*100:.2f}",
                        m.get('steps', 0),
                        f"{m.get('execution_time', 0):.2f}",
                        f"{m.get('trajectory_length', 0):.2f}",
                        f"{m.get('avg_reward_per_step', 0):.4f}",
                        f"{consistency:.1f}"
                    ])
        
        # 写入CSV
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        
        return csv_path
    
    def get_summary_stats(self) -> Dict:
        """获取汇总统计信息"""
        if not self.collected_metrics:
            return {}
        
        stats = {
            'total_metrics_collected': len(self.collected_metrics),
            'algorithms': list(set(m['algorithm'] for m in self.collected_metrics)),
            'sources': list(set(m['source'] for m in self.collected_metrics))
        }
        
        # 按算法分组统计
        for algorithm in stats['algorithms']:
            alg_metrics = [m for m in self.collected_metrics if m['algorithm'] == algorithm]
            
            if alg_metrics:
                stats[algorithm] = {
                    'count': len(alg_metrics),
                    'avg_reward': np.mean([m['total_reward'] for m in alg_metrics]),
                    'avg_coverage': np.mean([m['coverage_rate'] for m in alg_metrics]),
                    'avg_steps': np.mean([m['steps'] for m in alg_metrics]),
                    'avg_execution_time': np.mean([m['execution_time'] for m in alg_metrics])
                }
        
        return stats