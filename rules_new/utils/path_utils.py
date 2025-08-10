"""
路径处理工具 - 提供路径相关的实用功能
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class PathUtils:
    """路径处理工具类"""
    
    @staticmethod
    def get_project_root() -> Path:
        """获取项目根目录"""
        # 从当前文件开始，向上查找包含CLAUDE.md的目录
        current = Path(__file__).resolve()
        
        for parent in current.parents:
            if (parent / "CLAUDE.md").exists():
                return parent
                
        # 如果找不到，返回当前目录的上三级（rules_new1的上级）
        return current.parents[3]
    
    @staticmethod
    def resolve_relative_path(relative_path: str, base_path: Optional[Path] = None) -> Path:
        """解析相对路径为绝对路径"""
        if base_path is None:
            base_path = PathUtils.get_project_root()
            
        return base_path / relative_path
    
    @staticmethod
    def ensure_directory_exists(path: Path) -> Path:
        """确保目录存在，如果不存在则创建"""
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def get_config_path(config_name: str) -> Path:
        """获取配置文件的绝对路径"""
        project_root = PathUtils.get_project_root()
        config_path = project_root / "rules_new" / "configs" / config_name
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        return config_path
    
    @staticmethod
    def get_algorithm_config_path(algorithm_name: str) -> Path:
        """获取算法配置文件路径"""
        return PathUtils.get_config_path(f"algorithms/{algorithm_name.lower()}.yaml")
    
    @staticmethod
    def get_experiment_config_path(experiment_name: str) -> Path:
        """获取实验配置文件路径"""
        return PathUtils.get_config_path(f"experiments/{experiment_name}.yaml")
    
    @staticmethod
    def get_experiment_output_directory(experiment_name: str, config: Dict[str, Any]) -> Path:
        """
        获取实验专用的时间戳输出目录
        格式：logs/experiments/{experiment_name}_{timestamp}/
        """
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建实验专用目录名
        experiment_dir_name = f"{experiment_name}_{timestamp}"
        
        # 获取基础目录（默认logs）
        output_config = config.get('output', {})
        base_dir = output_config.get('base_dir', 'logs')
        
        # 构建完整路径：logs/experiments/{experiment_name}_{timestamp}/
        full_path = PathUtils.resolve_relative_path(f"{base_dir}/experiments/{experiment_dir_name}")
        
        # 确保目录存在
        PathUtils.ensure_directory_exists(full_path)
        
        return full_path
    
    @staticmethod
    def create_experiment_subdirectories(experiment_dir: Path) -> Dict[str, Path]:
        """
        在实验目录下创建标准子目录结构
        
        返回目录路径字典：
        - results: CSV结果文件
        - trajectories: 轨迹可视化文件  
        - logs: 日志文件
        """
        subdirs = {
            'results': experiment_dir / 'results',
            'trajectories': experiment_dir / 'trajectories', 
            'logs': experiment_dir / 'logs'
        }
        
        # 创建所有子目录
        for subdir_path in subdirs.values():
            PathUtils.ensure_directory_exists(subdir_path)
            
        return subdirs

    @staticmethod
    def get_output_directory(config: Dict[str, Any]) -> Path:
        """根据配置获取输出目录"""
        output_config = config.get('output', {})
        base_dir = output_config.get('base_dir', 'logs')
        
        # 解析相对路径
        output_dir = PathUtils.resolve_relative_path(base_dir)
        
        # 确保目录存在
        return PathUtils.ensure_directory_exists(output_dir)
    
    @staticmethod
    def get_log_file_path(experiment_name: str, algorithm_name: str, difficulty: str) -> Path:
        """获取日志文件路径"""
        project_root = PathUtils.get_project_root()
        log_dir = project_root / "logs"
        PathUtils.ensure_directory_exists(log_dir)
        
        filename = f"coverage_results_{algorithm_name}_{difficulty}.csv"
        return log_dir / filename
    
    @staticmethod
    def create_unique_filename(base_path: Path, suffix: str = "") -> Path:
        """创建唯一的文件名（如果文件存在则添加数字后缀）"""
        if not base_path.exists():
            return base_path
            
        stem = base_path.stem
        extension = base_path.suffix
        parent = base_path.parent
        
        counter = 1
        while True:
            new_name = f"{stem}_{counter}{suffix}{extension}"
            new_path = parent / new_name
            if not new_path.exists():
                return new_path
            counter += 1
    
    @staticmethod
    def normalize_path_separators(path_str: str) -> str:
        """标准化路径分隔符（将反斜杠转换为正斜杠）"""
        return path_str.replace('\\\\', '/').replace('\\', '/')
    
    @staticmethod
    def get_relative_path_from_project_root(absolute_path: Path) -> str:
        """获取相对于项目根目录的相对路径"""
        project_root = PathUtils.get_project_root()
        try:
            return str(absolute_path.relative_to(project_root))
        except ValueError:
            # 路径不在项目根目录下，返回绝对路径
            return str(absolute_path)
    
    @staticmethod
    def is_path_safe(path: Path) -> bool:
        """检查路径是否安全（不包含危险的路径遍历）"""
        try:
            # 检查是否有路径遍历攻击
            resolved = path.resolve()
            project_root = PathUtils.get_project_root().resolve()
            
            # 路径必须在项目根目录下或其子目录中
            try:
                resolved.relative_to(project_root)
                return True
            except ValueError:
                return False
                
        except (OSError, ValueError):
            return False
    
    @staticmethod
    def backup_file(file_path: Path, backup_suffix: str = ".backup") -> Optional[Path]:
        """备份文件"""
        if not file_path.exists():
            return None
            
        backup_path = file_path.with_suffix(file_path.suffix + backup_suffix)
        
        try:
            # 如果备份文件已存在，创建带时间戳的备份
            if backup_path.exists():
                import time
                timestamp = int(time.time())
                backup_path = file_path.with_suffix(f".{timestamp}{backup_suffix}")
                
            backup_path.write_bytes(file_path.read_bytes())
            return backup_path
            
        except (OSError, IOError):
            return None