"""
性能监控器

监控算法执行性能，识别瓶颈
提供性能分析和优化建议

作者：Rules_new优化团队
版本：2.0.0
"""

import time
import logging
import psutil
import numpy as np
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from functools import wraps
from collections import defaultdict

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    性能监控器
    
    功能：
    - 执行时间跟踪
    - 内存使用监控
    - 性能瓶颈识别
    - 优化建议生成
    """
    
    def __init__(self, name: str = "default"):
        """
        初始化性能监控器
        
        Args:
            name: 监控器名称
        """
        self.name = name
        self.metrics = defaultdict(list)
        self.timers = {}
        self.start_time = None
        self.memory_baseline = None
        
        # 性能阈值
        self.thresholds = {
            'max_execution_time': 0.1,  # 单次操作最大时间（秒）
            'max_memory_usage': 500,    # 最大内存使用（MB）
            'max_cpu_usage': 80,        # 最大CPU使用率（%）
        }
        
        # 性能统计
        self.stats = {
            'total_operations': 0,
            'slow_operations': 0,
            'memory_peaks': [],
            'bottlenecks': []
        }
    
    def start_timer(self, operation: str):
        """
        开始计时
        
        Args:
            operation: 操作名称
        """
        self.timers[operation] = time.perf_counter()
    
    def stop_timer(self, operation: str) -> float:
        """
        停止计时并记录
        
        Args:
            operation: 操作名称
            
        Returns:
            执行时间（秒）
        """
        if operation not in self.timers:
            logger.warning(f"Timer for {operation} was not started")
            return 0.0
        
        elapsed = time.perf_counter() - self.timers[operation]
        del self.timers[operation]
        
        # 记录指标
        self.metrics[f"time_{operation}"].append(elapsed)
        self.stats['total_operations'] += 1
        
        # 检查是否为慢操作
        if elapsed > self.thresholds['max_execution_time']:
            self.stats['slow_operations'] += 1
            self.stats['bottlenecks'].append({
                'operation': operation,
                'time': elapsed,
                'timestamp': datetime.now().isoformat()
            })
            logger.warning(f"Slow operation detected: {operation} took {elapsed:.3f}s")
        
        return elapsed
    
    def measure_memory(self) -> Dict[str, float]:
        """
        测量当前内存使用
        
        Returns:
            内存使用信息
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        
        memory_mb = memory_info.rss / 1024 / 1024  # 转换为MB
        
        result = {
            'rss_mb': memory_mb,
            'vms_mb': memory_info.vms / 1024 / 1024 if hasattr(memory_info, 'vms') else 0,
            'percent': process.memory_percent()
        }
        
        # 记录内存峰值
        if memory_mb > self.thresholds['max_memory_usage']:
            self.stats['memory_peaks'].append({
                'memory_mb': memory_mb,
                'timestamp': datetime.now().isoformat()
            })
            logger.warning(f"High memory usage: {memory_mb:.2f} MB")
        
        self.metrics['memory_mb'].append(memory_mb)
        
        return result
    
    def measure_cpu(self) -> float:
        """
        测量CPU使用率
        
        Returns:
            CPU使用率（%）
        """
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.metrics['cpu_percent'].append(cpu_percent)
        
        if cpu_percent > self.thresholds['max_cpu_usage']:
            logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
        
        return cpu_percent
    
    @staticmethod
    def profile(operation_name: Optional[str] = None):
        """
        性能分析装饰器
        
        Args:
            operation_name: 操作名称（默认使用函数名）
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                # 获取监控器实例
                if hasattr(self, 'performance_monitor'):
                    monitor = self.performance_monitor
                else:
                    # 创建临时监控器
                    monitor = PerformanceMonitor(func.__name__)
                
                name = operation_name or func.__name__
                
                # 开始监控
                monitor.start_timer(name)
                memory_before = monitor.measure_memory()
                
                try:
                    # 执行函数
                    result = func(self, *args, **kwargs)
                    
                    # 结束监控
                    elapsed = monitor.stop_timer(name)
                    memory_after = monitor.measure_memory()
                    memory_delta = memory_after['rss_mb'] - memory_before['rss_mb']
                    
                    # 记录性能数据
                    monitor.metrics[f"memory_delta_{name}"].append(memory_delta)
                    
                    if elapsed > 0.01:  # 只记录超过10ms的操作
                        logger.debug(f"{name}: {elapsed:.3f}s, Δmem: {memory_delta:.1f}MB")
                    
                    return result
                    
                except Exception as e:
                    monitor.stop_timer(name)
                    logger.error(f"Error in {name}: {e}")
                    raise
            
            return wrapper
        return decorator
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取性能统计
        
        Returns:
            性能统计信息
        """
        stats = self.stats.copy()
        
        # 计算平均值
        for key, values in self.metrics.items():
            if values:
                stats[f"{key}_mean"] = np.mean(values)
                stats[f"{key}_std"] = np.std(values)
                stats[f"{key}_max"] = np.max(values)
                stats[f"{key}_min"] = np.min(values)
        
        # 识别主要瓶颈
        if self.stats['bottlenecks']:
            # 按时间排序找出最慢的操作
            sorted_bottlenecks = sorted(
                self.stats['bottlenecks'],
                key=lambda x: x['time'],
                reverse=True
            )
            stats['top_bottlenecks'] = sorted_bottlenecks[:5]
        
        return stats
    
    def generate_report(self) -> str:
        """
        生成性能报告
        
        Returns:
            性能报告文本
        """
        stats = self.get_statistics()
        
        report = []
        report.append(f"=== 性能监控报告: {self.name} ===")
        report.append(f"总操作数: {stats['total_operations']}")
        report.append(f"慢操作数: {stats['slow_operations']}")
        
        # 时间性能
        if 'time_mean' in stats:
            report.append(f"\n时间性能:")
            for key in stats:
                if key.startswith('time_') and key.endswith('_mean'):
                    op_name = key[5:-5]
                    mean_time = stats[key]
                    max_time = stats.get(f"time_{op_name}_max", 0)
                    report.append(f"  {op_name}: 平均 {mean_time:.3f}s, 最大 {max_time:.3f}s")
        
        # 内存性能
        if 'memory_mb_mean' in stats:
            report.append(f"\n内存使用:")
            report.append(f"  平均: {stats['memory_mb_mean']:.1f} MB")
            report.append(f"  峰值: {stats.get('memory_mb_max', 0):.1f} MB")
        
        # 瓶颈分析
        if 'top_bottlenecks' in stats:
            report.append(f"\n主要瓶颈:")
            for bottleneck in stats['top_bottlenecks']:
                report.append(f"  - {bottleneck['operation']}: {bottleneck['time']:.3f}s")
        
        # 优化建议
        suggestions = self.generate_optimization_suggestions(stats)
        if suggestions:
            report.append(f"\n优化建议:")
            for suggestion in suggestions:
                report.append(f"  - {suggestion}")
        
        return "\n".join(report)
    
    def generate_optimization_suggestions(self, stats: Dict[str, Any]) -> List[str]:
        """
        生成优化建议
        
        Args:
            stats: 性能统计
            
        Returns:
            优化建议列表
        """
        suggestions = []
        
        # 基于慢操作比例
        if stats['total_operations'] > 0:
            slow_ratio = stats['slow_operations'] / stats['total_operations']
            if slow_ratio > 0.1:
                suggestions.append(f"超过{slow_ratio:.1%}的操作较慢，建议优化算法")
        
        # 基于内存使用
        if 'memory_mb_max' in stats and stats['memory_mb_max'] > self.thresholds['max_memory_usage']:
            suggestions.append(f"内存峰值达到{stats['memory_mb_max']:.1f}MB，建议优化内存使用")
        
        # 基于具体瓶颈
        if 'top_bottlenecks' in stats:
            for bottleneck in stats['top_bottlenecks'][:3]:
                if 'path' in bottleneck['operation'].lower():
                    suggestions.append(f"路径操作'{bottleneck['operation']}'较慢，考虑使用向量化计算")
                elif 'loop' in bottleneck['operation'].lower():
                    suggestions.append(f"循环操作'{bottleneck['operation']}'较慢，考虑使用numpy批量操作")
        
        return suggestions
    
    def reset(self):
        """重置监控器"""
        self.metrics.clear()
        self.timers.clear()
        self.stats = {
            'total_operations': 0,
            'slow_operations': 0,
            'memory_peaks': [],
            'bottlenecks': []
        }


# 便捷的上下文管理器
class TimedOperation:
    """计时上下文管理器"""
    
    def __init__(self, monitor: PerformanceMonitor, operation: str):
        self.monitor = monitor
        self.operation = operation
    
    def __enter__(self):
        self.monitor.start_timer(self.operation)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.stop_timer(self.operation)