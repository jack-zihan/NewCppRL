"""
配置验证工具函数。
"""
from __future__ import annotations

from typing import Tuple, Set, Dict, Any, Union
from pathlib import Path


def validate_positive(value: Union[int, float], name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_non_negative(value: Union[int, float], name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def validate_range(value: Union[int, float], min_val: Union[int, float], 
                  max_val: Union[int, float], name: str) -> None:
    if not min_val <= value <= max_val:
        raise ValueError(f"{name} must be in [{min_val}, {max_val}], got {value}")


def validate_tuple_positive(values: Tuple, name: str) -> None:
    if any(val <= 0 for val in values):
        raise ValueError(f"All {name} dimensions must be positive, got {values}")


def validate_tuple_range(values: Tuple[Union[int, float], Union[int, float]], name: str) -> None:
    """验证元组表示有效范围（min <= max）"""
    if len(values) != 2:
        raise ValueError(f"{name} must be a tuple of length 2, got {values}")
    min_val, max_val = values
    if min_val >= max_val:
        raise ValueError(f"{name} min ({min_val}) must be less than max ({max_val})")


def validate_required_keys(data: Dict[str, Any], required_keys: Set[str], name: str) -> None:
    missing_keys = required_keys - set(data.keys())
    if missing_keys:
        raise ValueError(f"Missing {name} keys: {missing_keys}")


def validate_path_exists(path: Union[str, Path], name: str) -> Path:
    path_obj = Path(path)
    if not path_obj.exists():
        raise ValueError(f"{name} does not exist: {path_obj}")
    return path_obj


def get_project_root() -> Path:
    # 从当前文件位置向上3级找到项目根目录
    config_file_path = Path(__file__)  # envs_new/utils/validation_utils.py
    project_root = config_file_path.parent.parent.parent
    return project_root


def validate_map_directory(map_dir: str) -> Path:
    """验证地图目录并返回绝对路径。"""
    if not Path(map_dir).is_absolute():
        project_root = get_project_root()
        map_path = project_root / map_dir
    else:
        map_path = Path(map_dir)
    
    if not map_path.exists():
        raise ValueError(f"Map directory does not exist: {map_path} (from config: {map_dir})")
    
    return map_path