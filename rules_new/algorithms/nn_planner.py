"""
神经网络规划器 - 基于SAC模型的路径规划算法
"""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

from .base_algorithm import BasePathPlanner
from ..core import CoordinateSystem as CS


class NNPlanner(BasePathPlanner):
    """
    神经网络路径规划器
    
    特点：
    - 基于SAC训练的神经网络模型
    - 支持baseline和ours两种模型
    - 直接输出环境action而非waypoint
    """
    
    def __init__(self, config: Dict[str, Any], env_config: Dict[str, Any]):
        # 首先初始化logger
        self.logger = logging.getLogger(__name__)
        
        super().__init__(config, env_config)
        
        # 神经网络特定参数
        self.model_path = config.get('model_path')
        self.device = config.get('device', 'cpu')
        
        if not self.model_path:
            raise ValueError("NNPlanner需要model_path配置")
        
        # 加载模型  
        self.actor = self._load_model()
        
        # 标记这是神经网络算法（返回action而非waypoint）
        self.algorithm_type = 'neural_network'
        
        # 当前观测状态
        self.current_observation = None
        self.current_vector = None
        
        self.logger.info(f"NNPlanner初始化完成: {self.algorithm_name}, 模型: {self.model_path}")
        
    def _load_model(self) -> torch.nn.Module:
        """加载SAC模型"""
        try:
            # 如果是相对路径，需要基于项目根目录解析
            if not Path(self.model_path).is_absolute():
                # 获取项目根目录
                project_root = Path(__file__).parents[2]
                model_path = project_root / self.model_path
            else:
                model_path = Path(self.model_path)
            
            if not model_path.exists():
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            # 加载完整的actor-critic模型，提取actor部分
            actor_critic = torch.load(model_path, map_location=self.device)
            actor = actor_critic[0].to(self.device)  # 第一个元素是actor
            
            # 设置为评估模式
            actor.eval()
            
            self.logger.info(f"成功加载模型: {model_path}")
            return actor
            
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            raise
    
    def reset(self, initial_state: Dict[str, Any]):
        """重置神经网络算法状态"""
        super().reset(initial_state)
        
        # 重置观测状态
        self.current_observation = None
        self.current_vector = None
        
    def _process_observation(self, current_state: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        处理环境状态，转换为神经网络输入格式
        
        老版本envs系统提供：与训练时完全一致的观测格式
        - observation: (25, 16, 16) numpy数组
        - vector: 标量值
        """
        try:
            # 验证观测数据存在
            if 'observation' not in current_state:
                available_keys = list(current_state.keys())
                raise ValueError(
                    f"观测数据缺失！current_state中没有'observation'键。\n"
                    f"可用键: {available_keys}\n"
                    f"这表明环境观测数据传递失败。"
                )
            
            # 直接使用老版本环境的观测数据，但需要进行通道转换
            observation_data = current_state['observation']
            
            # 验证观测数据格式
            if not isinstance(observation_data, np.ndarray):
                raise TypeError(f"观测数据应该是numpy数组，实际类型: {type(observation_data)}")
            
            # 检测模型需要的通道数（如果还没有检测过）
            if not hasattr(self, '_model_channels'):
                self._detect_model_channels()
            
            if observation_data.shape != (25, 16, 16):
                raise ValueError(f"观测数据形状错误: {observation_data.shape}, 期望: (25, 16, 16)")
            
            # 根据模型需求调整通道数
            if self._model_channels == 20:
                observation_data = observation_data[:20, :, :]  # NN_baseline需要20通道
                self.logger.debug(f"观测数据转换: {observation_data.shape} (25->20通道)")
            elif self._model_channels == 25:
                # NN_ours需要25通道，保持原样
                self.logger.debug(f"观测数据保持: {observation_data.shape} (25通道)")
            else:
                self.logger.warning(f"未知的模型通道数: {self._model_channels}，默认转换为20通道")
                observation_data = observation_data[:20, :, :]
            
            # 处理向量数据（与sac_cont_test.py一致）
            vector_data = current_state.get('vector', 0.0)
            if not isinstance(vector_data, (int, float, np.number)):
                # 如果是数组，取第一个元素
                if hasattr(vector_data, '__len__') and len(vector_data) > 0:
                    vector_data = float(vector_data[0])
                else:
                    vector_data = 0.0
            
            # 转换为torch tensor（与sac_cont_test.py完全一致）
            observation_tensor = torch.from_numpy(observation_data).float().to(self.device).unsqueeze(0)
            vector_tensor = torch.tensor([vector_data]).float().to(self.device).unsqueeze(0)
            
            self.logger.debug(f"张量转换完成: observation_shape={observation_tensor.shape}, vector_shape={vector_tensor.shape}")
            
            return observation_tensor, vector_tensor
            
        except Exception as e:
            self.logger.error(f"观测处理失败: {e}")
            # 不要掩盖问题！重新抛出异常让开发者知道有问题
            raise e
    
    def plan_next_waypoint(self, current_state: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """
        神经网络推理获取下一个动作
        
        注意：NNPlanner返回的是action而非waypoint
        这将在ExperimentRunner中特殊处理
        """
        # 更新状态
        self.update_state(current_state)
        
        # 检查终止条件
        if self.should_terminate(current_state):
            return None
        
        try:
            # 处理观测
            observation, vector = self._process_observation(current_state)
            
            # 存储当前观测（用于调试）
            self.current_observation = observation
            self.current_vector = vector
            
            # 神经网络推理
            with torch.no_grad():
                logits = self.actor(observation=observation, vector=vector)
                action = logits[2][0].tolist()  # 提取action，参考sac_cont_test.py
            
            # 返回action（而非waypoint）
            # 这里我们返回一个特殊格式来标识这是action
            return ('action', tuple(action))
            
        except Exception as e:
            self.logger.error(f"神经网络推理失败: {e}")
            return None
    
    def should_terminate(self, current_state: Dict[str, Any]) -> bool:
        """判断是否应该终止"""
        # 检查覆盖率
        coverage_rate = current_state.get('coverage_rate', 0.0)
        if coverage_rate >= 0.98:
            return True
            
        # 检查超时
        if self.check_timeout():
            return True
            
        # 检查最大迭代次数
        if self.check_max_iterations():
            return True
            
        return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_path': self.model_path,
            'device': self.device,
            'algorithm_type': self.algorithm_type,
            'model_loaded': self.actor is not None
        }
    
    def _detect_model_channels(self):
        """检测模型需要的输入通道数"""
        try:
            # 通过查看模型的第一层来检测通道数
            first_layer = None
            for module in self.actor.modules():
                if hasattr(module, 'weight') and len(module.weight.shape) == 4:  # 卷积层
                    first_layer = module
                    break
            
            if first_layer is not None:
                self._model_channels = first_layer.weight.shape[1]  # 输入通道数
                self.logger.info(f"检测到模型 {self.algorithm_name} 需要 {self._model_channels} 个输入通道")
            else:
                # 默认使用20通道（向后兼容）
                self._model_channels = 20
                self.logger.warning(f"无法检测模型 {self.algorithm_name} 通道数，默认使用20通道")
                
        except Exception as e:
            self.logger.error(f"检测模型 {self.algorithm_name} 通道数失败: {e}")
            self._model_channels = 20  # 默认值