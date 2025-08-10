"""
日志工具 - 提供日志记录和CSV文件管理功能
"""
import csv
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from .path_utils import PathUtils


class LoggingUtils:
    """日志工具类"""
    
    @staticmethod
    def setup_logger(name: str, log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(name)
        
        # 避免重复添加处理器
        if logger.handlers:
            return logger
            
        logger.setLevel(level)
        
        # 创建格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 文件处理器（如果提供了日志文件）
        if log_file:
            PathUtils.ensure_directory_exists(log_file.parent)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
        return logger
    
    @staticmethod
    def get_experiment_logger(experiment_name: str) -> logging.Logger:
        """获取实验专用的日志记录器"""
        log_dir = PathUtils.get_project_root() / "logs" / "experiments"
        PathUtils.ensure_directory_exists(log_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{experiment_name}_{timestamp}.log"
        
        return LoggingUtils.setup_logger(f"experiment.{experiment_name}", log_file)


class CSVResultCollector:
    """CSV结果收集器"""
    
    def __init__(self, csv_file_path: Path, headers: List[str]):
        """
        初始化CSV收集器
        
        Args:
            csv_file_path: CSV文件路径
            headers: CSV表头
        """
        self.csv_file_path = csv_file_path
        self.headers = headers
        
        # 确保目录存在
        PathUtils.ensure_directory_exists(csv_file_path.parent)
        
        # 创建CSV文件（如果不存在）
        self._ensure_csv_exists()
    
    def _ensure_csv_exists(self):
        """确保CSV文件存在并有正确的表头"""
        if not self.csv_file_path.exists():
            self.write_headers()
    
    def write_headers(self):
        """写入CSV表头"""
        try:
            with open(self.csv_file_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(self.headers)
        except IOError as e:
            raise IOError(f"无法写入CSV文件头: {e}")
    
    def append_row(self, data: Dict[str, Any]):
        """追加一行数据"""
        try:
            with open(self.csv_file_path, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                
                # 按headers顺序提取数据
                row = []
                for header in self.headers:
                    value = data.get(header, '')
                    # 处理列表类型的数据（如coverage和distance历史）
                    if isinstance(value, list):
                        value = ','.join(map(str, value))
                    row.append(value)
                    
                writer.writerow(row)
                
        except IOError as e:
            raise IOError(f"无法写入CSV数据: {e}")
    
    def batch_append_rows(self, data_list: List[Dict[str, Any]]):
        """批量追加多行数据"""
        try:
            with open(self.csv_file_path, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                
                for data in data_list:
                    row = []
                    for header in self.headers:
                        value = data.get(header, '')
                        if isinstance(value, list):
                            value = ','.join(map(str, value))
                        row.append(value)
                    writer.writerow(row)
                    
        except IOError as e:
            raise IOError(f"无法批量写入CSV数据: {e}")
    
    def read_all_data(self) -> List[Dict[str, str]]:
        """读取所有CSV数据"""
        if not self.csv_file_path.exists():
            return []
            
        try:
            with open(self.csv_file_path, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                return list(reader)
                
        except IOError as e:
            raise IOError(f"无法读取CSV文件: {e}")
    
    def get_file_size(self) -> int:
        """获取CSV文件大小（字节）"""
        if self.csv_file_path.exists():
            return self.csv_file_path.stat().st_size
        return 0
    
    def get_row_count(self) -> int:
        """获取CSV文件行数（不包括表头）"""
        if not self.csv_file_path.exists():
            return 0
            
        try:
            with open(self.csv_file_path, 'r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                row_count = sum(1 for _ in reader) - 1  # 减去表头
                return max(0, row_count)
                
        except IOError:
            return 0


class PerformanceTimer:
    """性能计时器"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        
    def start(self):
        """开始计时"""
        self.start_time = time.time()
        
    def stop(self) -> float:
        """停止计时，返回耗时（秒）"""
        if self.start_time is None:
            raise RuntimeError("计时器未启动")
            
        self.end_time = time.time()
        return self.end_time - self.start_time
    
    def get_elapsed_time(self) -> float:
        """获取已经过时间（秒）"""
        if self.start_time is None:
            return 0.0
            
        current_time = time.time()
        return current_time - self.start_time
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        elapsed = self.stop()
        print(f"{self.name} 耗时: {elapsed:.3f} 秒")


class ExperimentMetrics:
    """实验指标收集器"""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()
        
    def record_metric(self, name: str, value: Any, timestamp: Optional[float] = None):
        """记录指标"""
        if timestamp is None:
            timestamp = time.time()
            
        if name not in self.metrics:
            self.metrics[name] = []
            
        self.metrics[name].append({
            'value': value,
            'timestamp': timestamp,
            'relative_time': timestamp - self.start_time
        })
    
    def get_metric_summary(self, name: str) -> Dict[str, Any]:
        """获取指标摘要"""
        if name not in self.metrics:
            return {}
            
        values = [entry['value'] for entry in self.metrics[name]]
        
        if not values:
            return {}
            
        # 数值类型的统计
        if all(isinstance(v, (int, float)) for v in values):
            return {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': sum(values) / len(values),
                'first': values[0],
                'last': values[-1]
            }
        else:
            return {
                'count': len(values),
                'first': values[0],
                'last': values[-1]
            }
    
    def export_metrics(self, file_path: Path):
        """导出指标到文件"""
        import json
        
        export_data = {
            'experiment_start': self.start_time,
            'export_time': time.time(),
            'metrics': self.metrics
        }
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            raise IOError(f"无法导出指标: {e}")