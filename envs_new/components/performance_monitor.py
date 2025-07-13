"""
Performance monitoring utilities for environment components.
"""
from __future__ import annotations

import time
import functools
from typing import Dict, List, Callable, Any
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    last_time: float = 0.0
    
    @property
    def average_time(self) -> float:
        """Average execution time."""
        return self.total_time / self.call_count if self.call_count > 0 else 0.0
    
    def update(self, execution_time: float) -> None:
        """Update metrics with new execution time."""
        self.call_count += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.last_time = execution_time
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.call_count = 0
        self.total_time = 0.0
        self.min_time = float('inf')
        self.max_time = 0.0
        self.last_time = 0.0


class PerformanceMonitor:
    """Global performance monitoring system."""
    
    def __init__(self):
        self._metrics: Dict[str, PerformanceMetrics] = {}
        self._enabled = True
    
    def enable(self) -> None:
        """Enable performance monitoring."""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable performance monitoring."""
        self._enabled = False
    
    @contextmanager
    def measure(self, name: str):
        """Context manager for measuring execution time."""
        if not self._enabled:
            yield
            return
        
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            if name not in self._metrics:
                self._metrics[name] = PerformanceMetrics()
            
            self._metrics[name].update(execution_time)
    
    def monitor_method(self, name: str = None):
        """Decorator for monitoring method performance."""
        def decorator(func: Callable) -> Callable:
            method_name = name or f"{func.__qualname__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.measure(method_name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def get_metrics(self, name: str = None) -> Dict[str, PerformanceMetrics]:
        """Get performance metrics."""
        if name:
            return {name: self._metrics.get(name, PerformanceMetrics())}
        return self._metrics.copy()
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary."""
        summary = {}
        for name, metrics in self._metrics.items():
            summary[name] = {
                'calls': metrics.call_count,
                'total_time': metrics.total_time,
                'avg_time': metrics.average_time,
                'min_time': metrics.min_time if metrics.min_time != float('inf') else 0.0,
                'max_time': metrics.max_time,
            }
        return summary
    
    def reset_metrics(self, name: str = None) -> None:
        """Reset performance metrics."""
        if name:
            if name in self._metrics:
                self._metrics[name].reset()
        else:
            for metrics in self._metrics.values():
                metrics.reset()
    
    def print_summary(self, sort_by: str = 'total_time') -> None:
        """Print performance summary."""
        summary = self.get_summary()
        if not summary:
            print("No performance data available.")
            return
        
        # Sort by specified metric
        sorted_items = sorted(summary.items(), 
                            key=lambda x: x[1].get(sort_by, 0), 
                            reverse=True)
        
        print("\n=== Performance Summary ===")
        print(f"{'Method':<40} {'Calls':<8} {'Total(s)':<10} {'Avg(ms)':<10} {'Min(ms)':<10} {'Max(ms)':<10}")
        print("-" * 98)
        
        for name, metrics in sorted_items:
            print(f"{name:<40} "
                  f"{metrics['calls']:<8} "
                  f"{metrics['total_time']:<10.4f} "
                  f"{metrics['avg_time']*1000:<10.2f} "
                  f"{metrics['min_time']*1000:<10.2f} "
                  f"{metrics['max_time']*1000:<10.2f}")


# Global performance monitor instance
performance_monitor = PerformanceMonitor()

# Convenience functions
def measure_time(name: str):
    """Context manager for measuring execution time."""
    return performance_monitor.measure(name)

def monitor_performance(name: str = None):
    """Decorator for monitoring method performance."""
    return performance_monitor.monitor_method(name)

def get_performance_summary():
    """Get performance summary."""
    return performance_monitor.get_summary()

def print_performance_summary(sort_by: str = 'total_time'):
    """Print performance summary."""
    performance_monitor.print_summary(sort_by)

def reset_performance_metrics():
    """Reset all performance metrics."""
    performance_monitor.reset_metrics()