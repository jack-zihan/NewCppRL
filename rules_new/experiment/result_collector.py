"""
结果收集器 - 收集和管理实验结果
"""
import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from ..utils.path_utils import PathUtils
from ..utils.logging_utils import CSVResultCollector, ExperimentMetrics


class ResultCollector:
    """实验结果收集器 - 支持时间戳分层目录结构"""
    
    def __init__(self, experiment_name: str, output_config: Dict[str, Any], experiment_dir: Path):
        """
        初始化结果收集器
        
        Args:
            experiment_name: 实验名称
            output_config: 输出配置
            experiment_dir: 实验专用目录路径
        """
        self.experiment_name = experiment_name
        self.output_config = output_config
        self.logger = logging.getLogger(__name__)
        
        # 使用传入的实验目录
        self.experiment_dir = experiment_dir
        
        # 创建子目录结构
        self.subdirs = PathUtils.create_experiment_subdirectories(experiment_dir)
        
        # 确定输出格式
        self.csv_format = output_config.get('csv_format', True)
        self.include_trajectory = output_config.get('include_trajectory', False)
        
        # 设置指标列表
        self.metrics = output_config.get('metrics', [
            'coverage_90', 'coverage_95', 'coverage_98', 
            'total_coverage', 'path_length', 'collision_rate'
        ])
        
        # CSV收集器（每个算法一个文件）
        self.csv_collectors: Dict[str, CSVResultCollector] = {}
        
        # 实验指标收集器
        self.experiment_metrics = ExperimentMetrics()
        
        self.logger.info(f"结果收集器初始化完成: {experiment_dir}")
        
    def get_results_directory(self) -> Path:
        """获取结果目录路径"""
        return self.subdirs['results']
        
    def get_trajectories_directory(self) -> Path:
        """获取轨迹目录路径"""
        return self.subdirs['trajectories']
        
    def get_logs_directory(self) -> Path:
        """获取日志目录路径"""
        return self.subdirs['logs']
        
    def _get_csv_headers(self) -> List[str]:
        """获取CSV表头"""
        base_headers = [
            'experiment_name',
            'algorithm',
            'seed',
            'difficulty',
            'weed_distribution',
            'noise_level', 
            'map_id',
            'timestamp',
            'runtime_seconds'
        ]
        
        # 添加性能指标
        base_headers.extend(self.metrics)
        
        # 如果包含轨迹信息
        if self.include_trajectory:
            base_headers.extend([
                'trajectory_length',
                'coverage_history',
                'distance_history'
            ])
            
        return base_headers
    
    def _get_csv_collector(self, key: str) -> CSVResultCollector:
        """获取或创建CSV收集器"""
        if key not in self.csv_collectors:
            # 使用results子目录
            csv_file = self.subdirs['results'] / f"{key}_results.csv"
            headers = self._get_csv_headers()
            self.csv_collectors[key] = CSVResultCollector(csv_file, headers)
            
        return self.csv_collectors[key]
    
    def collect_result(self, experiment_info: Dict[str, Any], algorithm_metrics: Dict[str, Any]):
        """
        收集单次实验结果
        
        Args:
            experiment_info: 实验信息（种子、难度、算法等）
            algorithm_metrics: 算法性能指标
        """
        # 准备结果数据
        result_data = {
            'experiment_name': experiment_info.get('experiment_name', 'unknown'),
            'algorithm': experiment_info.get('algorithm', 'unknown'),
            'seed': experiment_info.get('seed', 0),
            'difficulty': experiment_info.get('difficulty', 'unknown'),
            'weed_distribution': experiment_info.get('weed_distribution', 'unknown'),
            'noise_level': experiment_info.get('noise_level', 'unknown'),
            'map_id': experiment_info.get('map_id', 0),
            'timestamp': datetime.now().isoformat(),
            'runtime_seconds': algorithm_metrics.get('runtime', 0.0)
        }
        
        # 添加性能指标
        for metric in self.metrics:
            result_data[metric] = algorithm_metrics.get(metric, -1)
        
        # 添加轨迹信息（如果需要）
        if self.include_trajectory:
            result_data['trajectory_length'] = len(algorithm_metrics.get('coverage_history', []))
            result_data['coverage_history'] = algorithm_metrics.get('coverage_history', [])
            result_data['distance_history'] = algorithm_metrics.get('distance_history', [])
        
        # 保存到CSV
        if self.csv_format:
            # 使用算法名作为文件区分
            csv_key = experiment_info.get('algorithm', 'all')
            csv_collector = self._get_csv_collector(csv_key)
            csv_collector.append_row(result_data)
        
        # 记录到实验指标
        self.experiment_metrics.record_metric('result', result_data)
        
        self.logger.info(f"收集结果: {result_data['algorithm']} - "
                        f"覆盖率: {result_data.get('total_coverage', 0):.3f} - "
                        f"运行时间: {result_data['runtime_seconds']:.2f}s")
    
    def collect_batch_results(self, results: List[Dict[str, Any]]):
        """批量收集结果"""
        for result in results:
            experiment_info = result.get('experiment_info', {})
            algorithm_metrics = result.get('algorithm_metrics', {})
            self.collect_result(experiment_info, algorithm_metrics)
    
    def export_summary(self, summary_file: Optional[Path] = None) -> Path:
        """导出实验摘要"""
        if summary_file is None:
            summary_file = self.experiment_dir / "summary.json"
        
        # 收集摘要信息
        summary_data = {
            'experiment_name': self.experiment_name,
            'export_time': datetime.now().isoformat(),
            'experiment_directory': str(self.experiment_dir),
            'subdirectories': {
                'results': str(self.subdirs['results']),
                'trajectories': str(self.subdirs['trajectories']),
                'logs': str(self.subdirs['logs'])
            },
            'metrics_collected': self.metrics,
            'csv_files': {
                key: str(collector.csv_file_path.relative_to(self.experiment_dir))
                for key, collector in self.csv_collectors.items()
            },
            'result_counts': {
                key: collector.get_row_count() 
                for key, collector in self.csv_collectors.items()
            },
            'experiment_metrics_summary': {
                metric_name: self.experiment_metrics.get_metric_summary(metric_name)
                for metric_name in ['result']
            }
        }
        
        # 导出到JSON文件
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"实验摘要已导出: {summary_file}")
            return summary_file
            
        except IOError as e:
            self.logger.error(f"导出摘要失败: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            'total_results': 0,
            'algorithms_tested': set(),
            'csv_files_created': len(self.csv_collectors),
            'experiment_directory': str(self.experiment_dir)
        }
        
        for key, collector in self.csv_collectors.items():
            row_count = collector.get_row_count()
            stats['total_results'] += row_count
            stats[f'{key}_results'] = row_count
            
            # 读取数据获取算法信息
            try:
                data = collector.read_all_data()
                for row in data:
                    if 'algorithm' in row:
                        stats['algorithms_tested'].add(row['algorithm'])
            except:
                pass
        
        stats['algorithms_tested'] = list(stats['algorithms_tested'])
        return stats
    
    def cleanup(self):
        """清理资源"""
        # 导出最终的实验指标
        metrics_file = self.subdirs['logs'] / "experiment_metrics.json"
        try:
            self.experiment_metrics.export_metrics(metrics_file)
            self.logger.info(f"实验指标已保存: {metrics_file}")
        except Exception as e:
            self.logger.warning(f"保存实验指标失败: {e}")
        
        # 清空收集器
        self.csv_collectors.clear()
        self.logger.info("结果收集器已清理")