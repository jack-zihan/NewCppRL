"""
配置管理器 - 处理YAML配置文件的加载和验证
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from ..utils.path_utils import PathUtils


class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._loaded_configs = {}  # 配置缓存
        
    def load_base_config(self) -> Dict[str, Any]:
        """加载基础配置"""
        return self.load_config("base_config.yaml")
    
    def load_algorithm_config(self, algorithm_name: str) -> Dict[str, Any]:
        """加载算法配置"""
        config_path = f"algorithms/{algorithm_name.lower()}.yaml"
        return self.load_config(config_path)
    
    def load_experiment_config(self, experiment_name: str) -> Dict[str, Any]:
        """加载实验配置"""
        config_path = f"experiments/{experiment_name}.yaml"
        return self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        # 检查缓存
        if config_path in self._loaded_configs:
            return self._loaded_configs[config_path]
        
        try:
            full_path = PathUtils.get_config_path(config_path)
            
            with open(full_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                
            # 验证配置
            validated_config = self._validate_config(config, config_path)
            
            # 缓存配置
            self._loaded_configs[config_path] = validated_config
            
            self.logger.info(f"成功加载配置: {config_path}")
            return validated_config
            
        except FileNotFoundError:
            self.logger.error(f"配置文件未找到: {config_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"YAML格式错误 {config_path}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"加载配置失败 {config_path}: {e}")
            raise
    
    def _validate_config(self, config: Dict[str, Any], config_path: str) -> Dict[str, Any]:
        """验证配置文件内容"""
        if config is None:
            raise ValueError(f"配置文件为空: {config_path}")
        
        # 基础配置验证
        if "base_config.yaml" in config_path:
            return self._validate_base_config(config)
        
        # 算法配置验证
        elif "algorithms/" in config_path:
            return self._validate_algorithm_config(config)
        
        # 实验配置验证
        elif "experiments/" in config_path:
            return self._validate_experiment_config(config)
        
        return config
    
    def _validate_base_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证基础配置"""
        required_sections = ['environment', 'agent', 'paths']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"基础配置缺少必需的部分: {section}")
        
        # 环境参数验证
        env_config = config['environment']
        if env_config['width'] <= 0 or env_config['height'] <= 0:
            raise ValueError("环境尺寸必须大于0")
        
        # 智能体参数验证
        agent_config = config['agent']
        if agent_config['car_width'] <= 0:
            raise ValueError("车辆宽度必须大于0")
        
        return config
    
    def _validate_algorithm_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证算法配置"""
        if 'algorithm' not in config:
            raise ValueError("算法配置缺少algorithm部分")
        
        algorithm_info = config['algorithm']
        if 'name' not in algorithm_info:
            raise ValueError("算法配置缺少name字段")
        
        # 验证性能参数
        if 'performance' in config:
            perf_config = config['performance']
            if 'timeout_seconds' in perf_config and perf_config['timeout_seconds'] <= 0:
                raise ValueError("超时时间必须大于0")
            if 'max_iterations' in perf_config and perf_config['max_iterations'] <= 0:
                raise ValueError("最大迭代次数必须大于0")
        
        return config
    
    def _validate_experiment_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证实验配置"""
        required_sections = ['experiment', 'algorithms']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"实验配置缺少必需的部分: {section}")
        
        # 验证实验信息
        exp_config = config['experiment']
        if 'name' not in exp_config:
            raise ValueError("实验配置缺少name字段")
        
        # 验证算法列表
        algorithms = config['algorithms']
        if not algorithms:
            raise ValueError("实验配置必须包含至少一个算法")
        
        for alg in algorithms:
            if 'name' not in alg:
                raise ValueError("算法配置必须包含name字段")
            
            # 神经网络算法需要model_path，传统算法需要config_path
            alg_name = alg['name']
            if alg_name.startswith('NN_'):
                if 'model_path' not in alg:
                    raise ValueError(f"神经网络算法 {alg_name} 必须包含model_path字段")
            else:
                if 'config_path' not in alg:
                    raise ValueError(f"传统算法 {alg_name} 必须包含config_path字段")
        
        return config
    
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """合并配置（深度合并）"""
        def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
            result = base.copy()
            
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            
            return result
        
        return deep_merge(base_config, override_config)
    
    def clear_cache(self):
        """清空配置缓存"""
        self._loaded_configs.clear()
        self.logger.info("配置缓存已清空")
    
    def get_cached_configs(self) -> Dict[str, Dict[str, Any]]:
        """获取所有缓存的配置"""
        return self._loaded_configs.copy()