"""
标准化实验评估系统

用于运行和对比多个算法在相同场景下的性能表现
支持7个算法：JUMP、SNAKE、R_SNAKE、BCP、REACT、NN_baseline、NN_ours

主要特性：
- 确定性场景生成：相同seed生成完全一致的场景
- save_finished_picture参数：场景完成时保存渲染图片
- config_dir参数：灵活切换配置目录
- 并行执行：多进程加速测试
- 自动分析：生成排名和统计报告

作者：Rules_new团队
版本：1.0.0
"""

from .scenario_generator import ScenarioGenerator
from .metric_collector import MetricCollector
from .visualization_manager import VisualizationManager
from .result_analyzer import ResultAnalyzer
from .benchmark_runner import BenchmarkRunner

__version__ = '1.0.0'

__all__ = [
    'BenchmarkRunner',
    'ScenarioGenerator',
    'MetricCollector',
    'ResultAnalyzer',
    'VisualizationManager'
]