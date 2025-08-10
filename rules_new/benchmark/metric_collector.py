"""
指标收集器 - 收集和计算统一的性能指标

包括覆盖率、路径长度、碰撞检测等关键指标
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class MetricCollector:
    """
    统一的指标收集器
    
    收集所有算法的标准化性能指标
    """
    
    def __init__(self, coverage_thresholds: List[float] = None):
        """
        初始化指标收集器
        
        Args:
            coverage_thresholds: 覆盖率阈值列表，默认[0.90, 0.95, 0.98]
        """
        if coverage_thresholds is None:
            coverage_thresholds = [0.90, 0.95, 0.98]
        
        self.coverage_thresholds = sorted(coverage_thresholds)
        self.metrics_history = []
        
    def collect_metrics(self, 
                       trajectory_data: Dict[str, Any],
                       env_info: Dict[str, Any],
                       algorithm_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        收集完整的性能指标
        
        Args:
            trajectory_data: 轨迹数据，包含coverage_history, distance_history等
            env_info: 环境信息
            algorithm_info: 算法信息
            
        Returns:
            性能指标字典
        """
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'algorithm': algorithm_info.get('name', 'unknown'),
            'scenario_id': env_info.get('scenario_id', 'unknown'),
            
            # 基础信息
            'seed': env_info.get('seed', 0),
            'difficulty': env_info.get('difficulty', 'unknown'),
            'map_id': env_info.get('map_id', 0),
            'weed_distribution': env_info.get('weed_distribution', 'unknown'),
            'noise_level': env_info.get('noise_level', 'no_noise'),
        }
        
        # 收集覆盖率相关指标
        coverage_metrics = self._collect_coverage_metrics(trajectory_data)
        metrics.update(coverage_metrics)
        
        # 收集路径相关指标
        path_metrics = self._collect_path_metrics(trajectory_data)
        metrics.update(path_metrics)
        
        # 收集碰撞相关指标
        collision_metrics = self._collect_collision_metrics(trajectory_data)
        metrics.update(collision_metrics)
        
        # 收集时间相关指标
        time_metrics = self._collect_time_metrics(trajectory_data)
        metrics.update(time_metrics)
        
        # 收集效率相关指标
        efficiency_metrics = self._calculate_efficiency_metrics(
            coverage_metrics, path_metrics, time_metrics
        )
        metrics.update(efficiency_metrics)
        
        # 保存到历史记录
        self.metrics_history.append(metrics)
        
        logger.debug(f"收集指标: {algorithm_info.get('name')} - "
                    f"最终覆盖率: {metrics.get('final_coverage', 0):.3f}")
        
        return metrics
    
    def _collect_coverage_metrics(self, trajectory_data: Dict[str, Any]) -> Dict[str, float]:
        """收集覆盖率相关指标"""
        coverage_history = trajectory_data.get('coverage_history', [])
        distance_history = trajectory_data.get('distance_history', [])
        
        if not coverage_history:
            logger.warning("没有覆盖率历史数据")
            return {
                'final_coverage': 0.0,
                'coverage_90_distance': -1,
                'coverage_95_distance': -1,
                'coverage_98_distance': -1,
            }
        
        metrics = {
            'final_coverage': coverage_history[-1] if coverage_history else 0.0,
            'max_coverage': max(coverage_history) if coverage_history else 0.0,
        }
        
        # 计算达到各个覆盖率阈值时的路径长度
        for threshold in self.coverage_thresholds:
            key = f'coverage_{int(threshold * 100)}_distance'
            distance = self._find_distance_at_coverage(
                coverage_history, distance_history, threshold
            )
            metrics[key] = distance
            
            # 同时记录达到该覆盖率的步数
            step_key = f'coverage_{int(threshold * 100)}_steps'
            steps = self._find_steps_at_coverage(coverage_history, threshold)
            metrics[step_key] = steps
        
        # 计算覆盖率增长速度
        if len(coverage_history) > 1:
            metrics['coverage_growth_rate'] = self._calculate_growth_rate(coverage_history)
        else:
            metrics['coverage_growth_rate'] = 0.0
        
        return metrics
    
    def _collect_path_metrics(self, trajectory_data: Dict[str, Any]) -> Dict[str, float]:
        """收集路径相关指标"""
        distance_history = trajectory_data.get('distance_history', [])
        position_history = trajectory_data.get('position_history', [])
        
        metrics = {
            'total_distance': distance_history[-1] if distance_history else 0.0,
            'total_steps': len(distance_history),
        }
        
        # 计算路径平滑度
        if position_history and len(position_history) > 2:
            metrics['path_smoothness'] = self._calculate_path_smoothness(position_history)
        else:
            metrics['path_smoothness'] = 0.0
        
        # 计算平均步长
        if len(distance_history) > 1:
            step_distances = np.diff(distance_history)
            metrics['avg_step_distance'] = np.mean(step_distances)
            metrics['std_step_distance'] = np.std(step_distances)
        else:
            metrics['avg_step_distance'] = 0.0
            metrics['std_step_distance'] = 0.0
        
        return metrics
    
    def _collect_collision_metrics(self, trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """收集碰撞相关指标"""
        collision_info = trajectory_data.get('collision_info', {})
        
        metrics = {
            'collision_occurred': collision_info.get('occurred', False),
            'collision_step': collision_info.get('step', -1),
            'collision_distance': collision_info.get('distance', -1),
            'collision_position': collision_info.get('position', None),
            'collision_type': collision_info.get('type', 'none'),  # boundary/obstacle/none
        }
        
        # 如果发生碰撞，计算碰撞时的覆盖率
        if metrics['collision_occurred']:
            coverage_history = trajectory_data.get('coverage_history', [])
            collision_step = metrics['collision_step']
            if 0 <= collision_step < len(coverage_history):
                metrics['collision_coverage'] = coverage_history[collision_step]
            else:
                metrics['collision_coverage'] = 0.0
        else:
            metrics['collision_coverage'] = -1
        
        return metrics
    
    def _collect_time_metrics(self, trajectory_data: Dict[str, Any]) -> Dict[str, float]:
        """收集时间相关指标"""
        time_info = trajectory_data.get('time_info', {})
        
        metrics = {
            'total_time': time_info.get('total_time', 0.0),
            'planning_time': time_info.get('planning_time', 0.0),
            'execution_time': time_info.get('execution_time', 0.0),
            'avg_step_time': time_info.get('avg_step_time', 0.0),
        }
        
        # 计算规划和执行的时间比例
        if metrics['total_time'] > 0:
            metrics['planning_time_ratio'] = metrics['planning_time'] / metrics['total_time']
            metrics['execution_time_ratio'] = metrics['execution_time'] / metrics['total_time']
        else:
            metrics['planning_time_ratio'] = 0.0
            metrics['execution_time_ratio'] = 0.0
        
        return metrics
    
    def _calculate_efficiency_metrics(self,
                                     coverage_metrics: Dict[str, float],
                                     path_metrics: Dict[str, float],
                                     time_metrics: Dict[str, float]) -> Dict[str, float]:
        """计算效率相关指标"""
        metrics = {}
        
        # 覆盖效率：覆盖率/路径长度
        if path_metrics['total_distance'] > 0:
            metrics['coverage_efficiency'] = (
                coverage_metrics['final_coverage'] / path_metrics['total_distance']
            )
        else:
            metrics['coverage_efficiency'] = 0.0
        
        # 时间效率：覆盖率/总时间
        if time_metrics['total_time'] > 0:
            metrics['time_efficiency'] = (
                coverage_metrics['final_coverage'] / time_metrics['total_time']
            )
        else:
            metrics['time_efficiency'] = 0.0
        
        # 步数效率：覆盖率/总步数
        if path_metrics['total_steps'] > 0:
            metrics['step_efficiency'] = (
                coverage_metrics['final_coverage'] / path_metrics['total_steps']
            )
        else:
            metrics['step_efficiency'] = 0.0
        
        # 计算综合效率分数（归一化后的加权平均）
        weights = {'coverage': 0.4, 'path': 0.3, 'time': 0.3}
        
        # 归一化各项指标
        normalized_coverage = coverage_metrics['final_coverage']
        normalized_path = 1.0 / (1.0 + path_metrics['total_distance'] / 1000.0)
        normalized_time = 1.0 / (1.0 + time_metrics['total_time'] / 100.0)
        
        metrics['overall_efficiency_score'] = (
            weights['coverage'] * normalized_coverage +
            weights['path'] * normalized_path +
            weights['time'] * normalized_time
        )
        
        return metrics
    
    def _find_distance_at_coverage(self,
                                   coverage_history: List[float],
                                   distance_history: List[float],
                                   threshold: float) -> float:
        """
        找到达到指定覆盖率时的路径长度
        
        Args:
            coverage_history: 覆盖率历史
            distance_history: 距离历史
            threshold: 覆盖率阈值
            
        Returns:
            达到阈值时的路径长度，如果未达到返回-1
        """
        if not coverage_history or not distance_history:
            return -1
        
        for i, coverage in enumerate(coverage_history):
            if coverage >= threshold:
                if i < len(distance_history):
                    return distance_history[i]
                else:
                    return distance_history[-1]
        
        return -1  # 未达到阈值
    
    def _find_steps_at_coverage(self,
                                coverage_history: List[float],
                                threshold: float) -> int:
        """找到达到指定覆盖率时的步数"""
        for i, coverage in enumerate(coverage_history):
            if coverage >= threshold:
                return i + 1
        return -1
    
    def _calculate_growth_rate(self, coverage_history: List[float]) -> float:
        """计算覆盖率增长速度"""
        if len(coverage_history) < 2:
            return 0.0
        
        # 计算每步的增长率
        growth_rates = np.diff(coverage_history)
        
        # 返回平均增长率
        return np.mean(growth_rates)
    
    def _calculate_path_smoothness(self, position_history: List[Tuple[float, float]]) -> float:
        """
        计算路径平滑度（角度变化的标准差）
        
        Args:
            position_history: 位置历史
            
        Returns:
            路径平滑度分数（越小越平滑）
        """
        if len(position_history) < 3:
            return 0.0
        
        angles = []
        for i in range(1, len(position_history) - 1):
            p1 = np.array(position_history[i - 1])
            p2 = np.array(position_history[i])
            p3 = np.array(position_history[i + 1])
            
            v1 = p2 - p1
            v2 = p3 - p2
            
            # 计算角度变化
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                angles.append(angle)
        
        if angles:
            # 返回角度变化的标准差（弧度）
            return np.std(angles)
        else:
            return 0.0
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        获取所有收集指标的汇总统计
        
        Returns:
            汇总统计字典
        """
        if not self.metrics_history:
            return {}
        
        # 按算法分组
        algorithms = {}
        for metric in self.metrics_history:
            alg_name = metric['algorithm']
            if alg_name not in algorithms:
                algorithms[alg_name] = []
            algorithms[alg_name].append(metric)
        
        # 计算每个算法的统计信息
        summary = {}
        for alg_name, alg_metrics in algorithms.items():
            summary[alg_name] = self._calculate_algorithm_statistics(alg_metrics)
        
        return summary
    
    def _calculate_algorithm_statistics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算单个算法的统计信息"""
        stats = {}
        
        # 需要统计的数值字段
        numeric_fields = [
            'final_coverage', 'total_distance', 'total_steps', 'total_time',
            'coverage_90_distance', 'coverage_95_distance', 'coverage_98_distance',
            'coverage_efficiency', 'time_efficiency', 'overall_efficiency_score'
        ]
        
        for field in numeric_fields:
            values = [m.get(field, 0) for m in metrics_list if field in m and m[field] != -1]
            if values:
                stats[f'{field}_mean'] = np.mean(values)
                stats[f'{field}_std'] = np.std(values)
                stats[f'{field}_min'] = np.min(values)
                stats[f'{field}_max'] = np.max(values)
        
        # 碰撞率
        collision_count = sum(1 for m in metrics_list if m.get('collision_occurred', False))
        stats['collision_rate'] = collision_count / len(metrics_list) if metrics_list else 0
        
        # 成功率（达到98%覆盖率）
        success_count = sum(1 for m in metrics_list if m.get('final_coverage', 0) >= 0.98)
        stats['success_rate'] = success_count / len(metrics_list) if metrics_list else 0
        
        return stats