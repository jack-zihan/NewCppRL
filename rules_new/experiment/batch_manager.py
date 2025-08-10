"""
批量管理器 - 处理多个实验的批量执行
"""
import asyncio
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from .experiment_runner import ExperimentRunner
from .config_manager import ConfigManager
from ..utils.path_utils import PathUtils
from ..utils.logging_utils import LoggingUtils, PerformanceTimer


class BatchManager:
    """批量实验管理器"""
    
    def __init__(self, max_workers: int = 4):
        """
        初始化批量管理器
        
        Args:
            max_workers: 最大并行工作线程数
        """
        self.max_workers = max_workers
        self.logger = LoggingUtils.get_experiment_logger("batch_manager")
        self.config_manager = ConfigManager()
        
        # 实验队列
        self.experiment_queue: List[str] = []
        self.completed_experiments: List[Dict[str, Any]] = []
        self.failed_experiments: List[Dict[str, Any]] = []
        
    def add_experiment(self, experiment_config_path: str):
        """添加实验到队列"""
        # 验证配置文件存在
        try:
            config_path = PathUtils.get_experiment_config_path(experiment_config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"实验配置文件不存在: {config_path}")
                
            self.experiment_queue.append(experiment_config_path)
            self.logger.info(f"实验已添加到队列: {experiment_config_path}")
            
        except Exception as e:
            self.logger.error(f"添加实验失败: {e}")
            raise
    
    def add_multiple_experiments(self, experiment_configs: List[str]):
        """批量添加实验"""
        for config in experiment_configs:
            try:
                self.add_experiment(config)
            except Exception as e:
                self.logger.warning(f"跳过无效实验配置 {config}: {e}")
    
    def discover_experiments(self, pattern: str = "*.yaml") -> List[str]:
        """自动发现实验配置文件"""
        experiments_dir = PathUtils.get_project_root() / "rules_new" / "configs" / "experiments"
        
        discovered = []
        for config_file in experiments_dir.glob(pattern):
            # 提取相对路径
            relative_path = config_file.relative_to(experiments_dir)
            discovered.append(str(relative_path))
            
        self.logger.info(f"发现 {len(discovered)} 个实验配置: {discovered}")
        return discovered
    
    def _run_single_experiment_batch(self, experiment_config: str) -> Dict[str, Any]:
        """运行单个实验（用于批量执行）"""
        try:
            self.logger.info(f"开始批量实验: {experiment_config}")
            
            # 创建实验运行器
            runner = ExperimentRunner(experiment_config)
            
            # 运行实验
            with PerformanceTimer(f"批量实验-{experiment_config}") as timer:
                result = runner.run_experiment()
                
            # 清理资源
            runner.cleanup()
            
            # 添加批量执行信息
            result['config_path'] = experiment_config
            result['batch_execution'] = True
            result['batch_runtime'] = timer.get_elapsed_time()
            
            self.logger.info(f"批量实验完成: {experiment_config}")
            return result
            
        except Exception as e:
            error_result = {
                'config_path': experiment_config,
                'error': str(e),
                'success': False,
                'batch_execution': True
            }
            self.logger.error(f"批量实验失败 {experiment_config}: {e}")
            return error_result
    
    def run_sequential(self) -> Dict[str, Any]:
        """顺序执行所有实验"""
        self.logger.info(f"开始顺序执行 {len(self.experiment_queue)} 个实验")
        
        start_time = datetime.now()
        
        for i, experiment_config in enumerate(self.experiment_queue, 1):
            self.logger.info(f"执行实验 {i}/{len(self.experiment_queue)}: {experiment_config}")
            
            result = self._run_single_experiment_batch(experiment_config)
            
            if result.get('success', False):
                self.completed_experiments.append(result)
            else:
                self.failed_experiments.append(result)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        return self._generate_batch_summary(total_time, "sequential")
    
    def run_parallel(self) -> Dict[str, Any]:
        """并行执行所有实验"""
        self.logger.info(f"开始并行执行 {len(self.experiment_queue)} 个实验 (最大线程数: {self.max_workers})")
        
        start_time = datetime.now()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_config = {
                executor.submit(self._run_single_experiment_batch, config): config
                for config in self.experiment_queue
            }
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    
                    if result.get('success', False):
                        self.completed_experiments.append(result)
                    else:
                        self.failed_experiments.append(result)
                        
                except Exception as e:
                    error_result = {
                        'config_path': config,
                        'error': str(e),
                        'success': False,
                        'batch_execution': True
                    }
                    self.failed_experiments.append(error_result)
                    self.logger.error(f"并行实验异常 {config}: {e}")
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        return self._generate_batch_summary(total_time, "parallel")
    
    def _generate_batch_summary(self, total_time: float, execution_mode: str) -> Dict[str, Any]:
        """生成批量执行摘要"""
        total_experiments = len(self.completed_experiments) + len(self.failed_experiments)
        success_rate = len(self.completed_experiments) / total_experiments if total_experiments > 0 else 0.0
        
        summary = {
            'execution_mode': execution_mode,
            'total_experiments': total_experiments,
            'successful_experiments': len(self.completed_experiments),
            'failed_experiments': len(self.failed_experiments),
            'success_rate': success_rate,
            'total_runtime_seconds': total_time,
            'average_runtime_per_experiment': total_time / total_experiments if total_experiments > 0 else 0.0,
            'completed_configs': [exp['config_path'] for exp in self.completed_experiments],
            'failed_configs': [exp['config_path'] for exp in self.failed_experiments],
            'timestamp': datetime.now().isoformat()
        }
        
        # 导出摘要到文件
        self._export_batch_summary(summary)
        
        self.logger.info(f"批量执行完成: {execution_mode}")
        self.logger.info(f"成功: {len(self.completed_experiments)}/{total_experiments} ({success_rate:.1%})")
        self.logger.info(f"总耗时: {total_time:.2f} 秒")
        
        return summary
    
    def _export_batch_summary(self, summary: Dict[str, Any]):
        """导出批量执行摘要"""
        try:
            output_dir = PathUtils.get_project_root() / "logs" / "batch_experiments"
            PathUtils.ensure_directory_exists(output_dir)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = output_dir / f"batch_summary_{timestamp}.json"
            
            import json
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"批量摘要已导出: {summary_file}")
            
        except Exception as e:
            self.logger.warning(f"导出批量摘要失败: {e}")
    
    def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态"""
        return {
            'queued_experiments': len(self.experiment_queue),
            'completed_experiments': len(self.completed_experiments),
            'failed_experiments': len(self.failed_experiments),
            'queue_configs': self.experiment_queue.copy(),
            'max_workers': self.max_workers
        }
    
    def clear_queue(self):
        """清空实验队列"""
        self.experiment_queue.clear()
        self.completed_experiments.clear()
        self.failed_experiments.clear()
        self.logger.info("实验队列已清空")
    
    def clear_results(self):
        """清空执行结果"""
        self.completed_experiments.clear()
        self.failed_experiments.clear()
        self.logger.info("执行结果已清空")