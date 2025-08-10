"""
割草机器人环境奖励系统。

采用策略模式实现可组合的奖励组件，每个组件负责计算特定方面的奖励：
- 基础惩罚：鼓励快速完成任务
- 杂草清除：奖励清除杂草的正向行为
- 前沿覆盖：奖励探索未知区域
- 转向平滑性：鼓励平滑的运动轨迹
- 碰撞/完成：终止条件的大额奖惩
"""
from __future__ import annotations

from typing import Dict, List, Any
import numpy as np

from envs_new.components.config.environment_config import EnvironmentConfig
from envs_new.components.state.environment_state import EnvironmentState


class BaseCalculator:
    """基础惩罚计算器 - 鼓励机器人快速完成任务"""
    coefficient = -0.1
    
    @staticmethod
    def calculate(env_state: EnvironmentState, **kwargs) -> float:
        return BaseCalculator.coefficient


class WeedRemovalCalculator:
    """杂草清除奖励计算器 - 主要任务目标"""
    coefficient = 20.0  # 高系数体现核心任务重要性
    
    @staticmethod
    def calculate(env_state: EnvironmentState, **kwargs) -> float:
        weed_info = env_state.get_info('weed_count')
        if not weed_info:
            return 0.0
        # 奖励与清除数量成正比
        weed_removed = -weed_info.change()
        return float(weed_removed) * WeedRemovalCalculator.coefficient


class FrontierCoverageCalculator:
    """前沿覆盖奖励计算器 - 鼓励探索未知区域"""
    coefficient = 1.0
    
    @staticmethod
    def calculate(env_state: EnvironmentState, **kwargs) -> float:
        config = kwargs.get('config')
        
        if config is None:
            return 0.0
        
        # 归一化：根据机器人宽度和最大速度计算预期每步覆盖量
        normalization = 2 * config.agent_width * config.v_max
        
        frontier_info = env_state.get_info('frontier_area')
        if not frontier_info:
            return 0.0
        frontier_covered = -frontier_info.change()
        
        return float(frontier_covered) / normalization * FrontierCoverageCalculator.coefficient


class FrontierVariationCalculator:
    """前沿变化奖励计算器 - 鼓励减少前沿复杂度"""
    coefficient = 0.5
    
    @staticmethod
    def calculate(env_state: EnvironmentState, **kwargs) -> float:
        config = kwargs.get('config')
        
        if config is None:
            return 0.0
        
        frontier_variation_info = env_state.get_info('frontier_variation')
        if not frontier_variation_info:
            return 0.0
        # 前沿复杂度减少意味着机器人正在清理面积、合并分散区域
        variation_reduction = -frontier_variation_info.change()
        
        return float(variation_reduction) / config.v_max * FrontierVariationCalculator.coefficient


class TurningPenaltyCalculator:
    """转向加速度惩罚 - 鼓励平滑的转向动作"""
    coefficient = -0.5
    
    @staticmethod
    def calculate(env_state: EnvironmentState, **kwargs) -> float:
        config = kwargs.get('config')
        
        if config is None:
            return 0.0
        
        steer_info = env_state.get_info('agent_steer')
        if not steer_info or steer_info.last is None:
            return 0.0
            
        steer_change = abs(steer_info.change())
        normalized_change = steer_change / config.w_max
        
        return normalized_change * TurningPenaltyCalculator.coefficient


class DirectionChangePenaltyCalculator:
    """转向方向改变惩罚 - 避免频繁左右摇摆"""
    coefficient = -0.30
    
    @staticmethod
    def calculate(env_state: EnvironmentState, **kwargs) -> float:
        steer_info = env_state.get_info('agent_steer')
        if not steer_info or steer_info.last is None:
            return 0.0
            
        current_steer = steer_info.current
        previous_steer = steer_info.last
        
        # 当转向方向反转时施加惩罚（符号相反）
        if current_steer * previous_steer < 0:
            return DirectionChangePenaltyCalculator.coefficient
        else:
            return 0.0


class SteeringSmoothnessCalculator:
    """转向平滑性奖励 - 鼓励较小的转向角度"""
    coefficient = 0.25
    
    @staticmethod
    def calculate(env_state: EnvironmentState, **kwargs) -> float:
        config = kwargs.get('config')
        
        if config is None:
            return 0.0
        
        # 非线性奖励公式：转向越小奖励越高
        # 0.4 - sqrt(|转向/最大转向|) 确保直线行驶获得最大奖励
        current_steer = env_state.agent_steer or 0.0
        normalized_steer = abs(current_steer / config.w_max)
        smoothness_reward = 0.4 - (normalized_steer ** 0.5)
        
        return smoothness_reward * SteeringSmoothnessCalculator.coefficient


class CollisionPenaltyCalculator:
    """碰撞惩罚 - 大额负奖励以避免碰撞"""
    coefficient = -399.0  # 近似-400，抵消大部分积累奖励
    
    @staticmethod
    def calculate(env_state: EnvironmentState, **kwargs) -> float:
        crashed = env_state.crashed
        return CollisionPenaltyCalculator.coefficient if crashed else 0.0


class CompletionBonusCalculator:
    """任务完成奖励 - 成功完成任务的大额正奖励"""
    coefficient = 500.0
    
    @staticmethod
    def calculate(env_state: EnvironmentState, **kwargs) -> float:
        finished = env_state.finished
        return CompletionBonusCalculator.coefficient if finished else 0.0


class RewardSystem:
    """智能奖励系统 - 自动化组件管理和计算"""
    
    AVAILABLE_CALCULATORS = {
        'base': BaseCalculator,
        'weed_removal': WeedRemovalCalculator,
        'frontier_coverage': FrontierCoverageCalculator,
        'frontier_variation': FrontierVariationCalculator,
        'turning_penalty': TurningPenaltyCalculator,
        'direction_change_penalty': DirectionChangePenaltyCalculator,
        'steering_smoothness': SteeringSmoothnessCalculator,
        'collision_penalty': CollisionPenaltyCalculator,
        'completion_bonus': CompletionBonusCalculator
    }
    
    def __init__(self, config: EnvironmentConfig):
        self.config = config
        
        # 根据配置更新Calculator系数
        self._update_coefficients()
        
        # 激活的Calculator（可配置）
        self.active_calculators = self._determine_active_calculators()
        
        # 组系数映射（用于应用组级别系数）
        self.group_coefficients = {
            'frontier_coverage': 'frontier_total_coef',
            'frontier_variation': 'frontier_total_coef', 
            'turning_penalty': 'turn_total_coef',
            'direction_change_penalty': 'turn_total_coef',
            'steering_smoothness': 'turn_total_coef'
        }
    
    def _update_coefficients(self) -> None:
        """根据配置更新所有Calculator的系数"""
        coefficient_mapping = {
            'base': self.config.reward_base_penalty,
            'weed_removal': self.config.reward_weed_removal_coef,
            'frontier_coverage': self.config.reward_frontier_coverage_coef,
            'frontier_variation': self.config.reward_frontier_tv_coef,
            'turning_penalty': self.config.reward_turn_gap_coef,
            'direction_change_penalty': self.config.reward_turn_direction_coef,
            'steering_smoothness': self.config.reward_turn_self_coef,
            'collision_penalty': self.config.reward_collision_penalty,
            'completion_bonus': self.config.reward_completion_bonus
        }
        
        for calc_name, coef_value in coefficient_mapping.items():
            if calc_name in self.AVAILABLE_CALCULATORS:
                self.AVAILABLE_CALCULATORS[calc_name].coefficient = coef_value
    
    def _determine_active_calculators(self) -> List[str]:
        """确定激活的Calculator（未来可扩展为配置驱动）"""
        # 默认激活所有
        return list(self.AVAILABLE_CALCULATORS.keys())
    
    def calculate_reward(self, env_state: EnvironmentState, **kwargs) -> float:
        """计算总奖励值 - 简化后的直接循环实现"""
        calc_kwargs = {
            'config': self.config,
            **kwargs
        }
        
        total_reward = 0.0
        
        for name in self.active_calculators:
            if name in self.AVAILABLE_CALCULATORS:
                component_reward = self.AVAILABLE_CALCULATORS[name].calculate(env_state, **calc_kwargs)
                
                # 应用组系数：frontier和turn相关组件有额外的组级系数
                if name in self.group_coefficients:
                    group_coef_key = self.group_coefficients[name]
                    # 使用get_reward_coefficients方法获取系数
                    coefficients = self.config.get_reward_coefficients()
                    group_coefficient = coefficients.get(group_coef_key, 1.0)
                    component_reward *= group_coefficient
                
                total_reward += component_reward
        
        if abs(total_reward) < 1e-8:
            total_reward = 0.0
        
        return float(total_reward)
    
    
    def get_reward_breakdown(self, env_state: EnvironmentState, **kwargs) -> Dict[str, Any]:
        """获取奖励分解和汇总信息"""
        calc_kwargs = {
            'config': self.config,
            **kwargs
        }
        
        # 计算所有组件奖励（含组系数）
        components = {}
        components_raw = {}  # 原始奖励（不含组系数）
        
        for name in self.active_calculators:
            if name in self.AVAILABLE_CALCULATORS:
                raw_reward = self.AVAILABLE_CALCULATORS[name].calculate(env_state, **calc_kwargs)
                components_raw[name] = raw_reward
                
                # 应用组系数
                if name in self.group_coefficients:
                    group_coef_key = self.group_coefficients[name]
                    # 使用get_reward_coefficients方法获取系数
                    coefficients = self.config.get_reward_coefficients()
                    group_coefficient = coefficients.get(group_coef_key, 1.0)
                    components[name] = raw_reward * group_coefficient
                else:
                    components[name] = raw_reward
        
        # 分组统计用于向后兼容
        turning_names = ['turning_penalty', 'direction_change_penalty', 'steering_smoothness']
        frontier_names = ['frontier_coverage', 'frontier_variation']
        
        turning_total = sum(components.get(name, 0.0) for name in turning_names)
        frontier_total = sum(components.get(name, 0.0) for name in frontier_names)
        
        return {
            'components': components,
            'components_raw': components_raw,  # 提供原始奖励用于调试
            'total': self.calculate_reward(env_state, **kwargs),
            'turning_total': turning_total,
            'frontier_total': frontier_total,
            'base': components.get('base', 0.0),
            'weed_removal': components.get('weed_removal', 0.0),
            'collision_penalty': components.get('collision_penalty', 0.0),
            'completion_bonus': components.get('completion_bonus', 0.0)
        }
    
    def add_calculator(self, name: str, calculator_class: type) -> None:
        self.AVAILABLE_CALCULATORS[name] = calculator_class
        if name not in self.active_calculators:
            self.active_calculators.append(name)
    
    def remove_calculator(self, name: str) -> None:
        if name in self.AVAILABLE_CALCULATORS:
            del self.AVAILABLE_CALCULATORS[name]
        if name in self.active_calculators:
            self.active_calculators.remove(name)
    
    def set_active_calculators(self, calculator_names: List[str]) -> None:
        invalid_names = set(calculator_names) - set(self.AVAILABLE_CALCULATORS.keys())
        if invalid_names:
            raise ValueError(f"Unknown calculators: {invalid_names}")
        self.active_calculators = calculator_names
    
    def update_coefficients(self, new_coefficients: Dict[str, float]) -> None:
        """动态更新奖励系数，用于在线调试或自适应策略。"""
        # 将旧系数名称映射到新的扁平化配置属性
        coefficient_mapping = {
            'base_penalty': 'reward_base_penalty',
            'weed_removal_coef': 'reward_weed_removal_coef',
            'frontier_coverage_coef': 'reward_frontier_coverage_coef',
            'frontier_tv_coef': 'reward_frontier_tv_coef',
            'turn_gap_coef': 'reward_turn_gap_coef',
            'turn_direction_coef': 'reward_turn_direction_coef',
            'turn_self_coef': 'reward_turn_self_coef',
            'collision_penalty': 'reward_collision_penalty',
            'completion_bonus': 'reward_completion_bonus'
        }
        
        for key, value in new_coefficients.items():
            if key in coefficient_mapping:
                setattr(self.config, coefficient_mapping[key], value)
        
        self._update_coefficients()


# 为了向后兼容，保留旧的接口别名
CompositeReward = RewardSystem