"""
割草机器人环境奖励系统。

采用策略模式实现可组合的奖励组件，每个组件负责计算特定方面的奖励：
- 基础惩罚：鼓励快速完成任务
- 杂草清除：奖励清除杂草的正向行为
- 田地覆盖：奖励覆盖田地区域
- 转向平滑性：鼓励平滑的运动轨迹
- 碰撞/完成：终止条件的大额奖惩
"""
from __future__ import annotations

from typing import Dict, List, Any
import numpy as np

from envs_new.components.config.environment_config import EnvironmentConfig
from envs_new.components.state.environment_state import EnvironmentState


class RewardCalculator:
    """奖励计算器基类 - 无状态纯函数设计"""
    group = None  # 默认不属于任何组
    
    @classmethod
    def calculate(cls, env_state: EnvironmentState, coefficient: float, config: EnvironmentConfig = None, **kwargs) -> float:
        """子类需要重写此方法 - config作为显式参数"""
        return 0.0


class BaseCalculator(RewardCalculator):
    """基础惩罚计算器 - 鼓励机器人快速完成任务"""
    
    @classmethod
    def calculate(cls, env_state: EnvironmentState, coefficient: float, config: EnvironmentConfig = None, **kwargs) -> float:
        return coefficient


class WeedRemovalCalculator(RewardCalculator):
    """杂草清除奖励计算器 - 主要任务目标"""
    
    @classmethod
    def calculate(cls, env_state: EnvironmentState, coefficient: float, config: EnvironmentConfig = None, **kwargs) -> float:
        weed_info = env_state.get_info('weed_count')
        if not weed_info:
            return 0.0
        # 奖励与清除数量成正比
        weed_removed = -weed_info.change()
        return weed_removed * coefficient


class FieldCoverageCalculator(RewardCalculator):
    """田地覆盖奖励计算器 - 鼓励覆盖田地区域"""
    group = 'field'  # 属于field组
    
    @classmethod
    def calculate(cls, env_state: EnvironmentState, coefficient: float, config: EnvironmentConfig = None, **kwargs) -> float:
        if not config:
            return 0.0
        
        # 归一化：根据机器人宽度和最大速度计算预期每步覆盖量
        normalization = 2 * config.agent_width * config.v_max
        
        field_info = env_state.get_info('field_area')
        if not field_info:
            return 0.0
        
        field_covered = -field_info.change()
        return field_covered / normalization * coefficient


class FieldVariationCalculator(RewardCalculator):
    """田地变化奖励计算器 - 鼓励减少田地复杂度"""
    group = 'field'  # 属于field组
    
    @classmethod
    def calculate(cls, env_state: EnvironmentState, coefficient: float, config: EnvironmentConfig = None, **kwargs) -> float:
        if not config:
            return 0.0
        
        field_variation_info = env_state.get_info('field_variation')
        if not field_variation_info:
            return 0.0
        
        # 田地复杂度减少意味着机器人正在清理面积、合并分散区域
        variation_reduction = -field_variation_info.change()
        return variation_reduction / config.v_max * coefficient


class TurningPenaltyCalculator(RewardCalculator):
    """转向加速度惩罚 - 鼓励平滑的转向动作"""
    group = 'turning'  # 属于turning组
    
    @classmethod
    def calculate(cls, env_state: EnvironmentState, coefficient: float, config: EnvironmentConfig = None, **kwargs) -> float:
        if not config:
            return 0.0
        
        steer_info = env_state.get_info('agent_steer')
        if not steer_info or steer_info.last is None:
            return 0.0

        normalized_change = abs(steer_info.change()) / config.w_max
        return normalized_change * coefficient


class DirectionChangePenaltyCalculator(RewardCalculator):
    """转向方向改变惩罚 - 避免频繁左右摇摆"""
    group = 'turning'  # 属于turning组
    
    @classmethod
    def calculate(cls, env_state: EnvironmentState, coefficient: float, config: EnvironmentConfig = None, **kwargs) -> float:
        steer_info = env_state.get_info('agent_steer')
        if not steer_info or steer_info.last is None:
            return 0.0
        
        # 当转向方向反转时施加惩罚（符号相反）
        return coefficient if steer_info.current * steer_info.last < 0 else 0.0


class SteeringSmoothnessCalculator(RewardCalculator):
    """转向平滑性奖励 - 鼓励较小的转向角度"""
    group = 'turning'  # 属于turning组
    
    @classmethod
    def calculate(cls, env_state: EnvironmentState, coefficient: float, config: EnvironmentConfig = None, **kwargs) -> float:
        if not config:
            return 0.0
        
        # 非线性奖励公式：转向越小奖励越高
        # 0.4 - sqrt(|转向/最大转向|) 确保直线行驶获得最大奖励
        current_steer = env_state.agent_steer or 0.0
        normalized_steer = abs(current_steer / config.w_max)
        smoothness_reward = 0.4 - (normalized_steer ** 0.5)
        return smoothness_reward * coefficient


class CollisionPenaltyCalculator(RewardCalculator):
    """碰撞惩罚 - 大额负奖励以避免碰撞"""
    
    @classmethod
    def calculate(cls, env_state: EnvironmentState, coefficient: float, config: EnvironmentConfig = None, **kwargs) -> float:
        return coefficient if env_state.crashed else 0.0


class CompletionBonusCalculator(RewardCalculator):
    """任务完成奖励 - 成功完成任务的大额正奖励"""
    
    @classmethod
    def calculate(cls, env_state: EnvironmentState, coefficient: float, config: EnvironmentConfig = None, **kwargs) -> float:
        return coefficient if env_state.finished else 0.0


class RewardSystem:
    """智能奖励系统 - 自动化组件管理和计算"""
    
    AVAILABLE_CALCULATORS = {
        'base_penalty': BaseCalculator,  # 更改key名称以匹配配置
        'weed_removal': WeedRemovalCalculator,
        'field_coverage': FieldCoverageCalculator,
        'field_variation': FieldVariationCalculator,
        'turning_penalty': TurningPenaltyCalculator,
        'direction_change_penalty': DirectionChangePenaltyCalculator,
        'steering_smoothness': SteeringSmoothnessCalculator,
        'collision_penalty': CollisionPenaltyCalculator,
        'completion_bonus': CompletionBonusCalculator
    }
    
    # 简化的组系数映射
    REWARD_GROUPS = {'field': 'reward_field_group_coef', 'turning': 'reward_turning_group_coef'}
    
    def __init__(self, config: EnvironmentConfig):
        self.config = config
    
    def calculate_reward(self, env_state: EnvironmentState, **kwargs) -> float:
        """计算总奖励值 - 直接从配置读取并传递系数"""
        total_reward = 0.0
        
        for name, calc_class in self.AVAILABLE_CALCULATORS.items():
            # 1. 从config获取基础系数
            config_attr = f"reward_{name}"
            coefficient = getattr(self.config, config_attr)
            
            # 2. 如果属于组，应用组系数
            if calc_class.group and calc_class.group in self.REWARD_GROUPS:
                group_coef = getattr(self.config, self.REWARD_GROUPS[calc_class.group])
                coefficient *= group_coef
            
            # 3. 调用calculate，传入最终系数和config
            component_reward = calc_class.calculate(env_state, coefficient, self.config, **kwargs)
            total_reward += component_reward
        
        return 0.0 if abs(total_reward) < 1e-8 else total_reward
    
    
    def get_reward_breakdown(self, env_state: EnvironmentState, **kwargs) -> Dict[str, Any]:
        """
        获取奖励分解信息 - 动态生成，自动适应当前激活的组件
        
        Returns:
            包含breakdown（各组件奖励）和total（总奖励）的字典
        """
        # 计算所有激活组件的奖励
        breakdown = {}  # 使用breakdown替代components，更符合业务语义
        
        for name, calc_class in self.AVAILABLE_CALCULATORS.items():
            # 与calculate_reward保持一致的逻辑
            config_attr = f"reward_{name}"
            coefficient = getattr(self.config, config_attr, 0.0)
            
            if calc_class.group and calc_class.group in self.REWARD_GROUPS:
                group_coef = getattr(self.config, self.REWARD_GROUPS[calc_class.group], 1.0)
                coefficient *= group_coef
            
            component_reward = calc_class.calculate(env_state, coefficient, self.config, **kwargs)
            breakdown[name] = component_reward
        
        # 计算总和
        total = sum(breakdown.values())
        
        # 返回简洁的结构：只包含breakdown和total
        return {
            'breakdown': breakdown,  # 所有组件的奖励分解
            'total': 0.0 if abs(total) < 1e-8 else total
        }
    
    def add_calculator(self, name: str, calculator_class: type) -> None:
        """动态添加新的Calculator"""
        self.AVAILABLE_CALCULATORS[name] = calculator_class
    
    def remove_calculator(self, name: str) -> None:
        """移除指定的Calculator"""
        if name in self.AVAILABLE_CALCULATORS:
            del self.AVAILABLE_CALCULATORS[name]
    
    def set_active_calculators(self, calculator_names: List[str]) -> None:
        """设置激活的Calculator子集（保留以兼容API）"""
        to_remove = set(self.AVAILABLE_CALCULATORS.keys()) - set(calculator_names)
        for name in to_remove:
            del self.AVAILABLE_CALCULATORS[name]
    
    def update_coefficients(self, new_coefficients: Dict[str, float]) -> None:
        """动态更新奖励系数 - 直接更新config即可"""
        for key, value in new_coefficients.items():
            config_attr = f"reward_{key}"
            if hasattr(self.config, config_attr):
                setattr(self.config, config_attr, value)


# HIFCalculator已移至envs_new/cpp_env_v5.py作为内部类，提高代码内聚性

