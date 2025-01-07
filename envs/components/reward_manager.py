from __future__ import annotations
from typing import Optional, Dict
from envs.components.utils import NumericalRange, MowerAgent

class RewardManager:
    """
    管理奖励的计算逻辑。
    """

    def __init__(
        self,
        speed_range: NumericalRange,
        angular_range: NumericalRange,
        coefficients: Optional[Dict[str, float]] = None
    ):
        """
        初始化奖励管理器。

        :param speed_range: 速度数值区间。
        :param angular_range: 角速度数值区间。
        :param coefficients: 奖励系数字典。
        """
        self.speed_range = speed_range
        self.angular_range = angular_range

        # 默认奖励系数
        default_coefficients = {
            'turn_total_coef': 0, # 转向角总惩罚
            'turn_gap_coef': -1, #-0.5, # 转向角加速度惩罚
            'turn_direction_coef': -0.30, # 方向相反系数
            'turn_self_coef': 0.25, # 转向角速度自身大小惩罚
            'frontier_total_coef': 0.125, # 农田探索系数
            'frontier_coverage_coef': 1, # 农田探索系数
            'frontier_tv_coef': 0.5, # 农田前沿总变异系数
            'base_penalty': -0.1,  # 基础惩罚
            'weed_removal_coef': 20.0,  # 杂草移除奖励
            'collision_penalty': -399.0, # 碰撞惩罚
            'completion_bonus': 500.0 # 完成奖励
        }

        # 更新系数
        self.coefficients = default_coefficients
        if coefficients:
            self.coefficients.update(coefficients)

    def calculate_step_reward( # TODO: 现在没有考虑碰撞惩罚和完成奖励
        self,
        current_steer: float,
        previous_steer: float,
        current_frontier_area: int,
        previous_frontier_area: int,
        current_frontier_tv: float,
        previous_frontier_tv: float,
        current_weeds: int,
        previous_weeds: int
    ) -> float:
        """
        计算单步奖励。

        :param current_steer: 当前转向角速度。
        :param previous_steer: 上一步转向角速度。
        :param current_frontier_area: 当前农田前沿面积。
        :param previous_frontier_area: 上一步农田前沿面积。
        :param current_frontier_tv: 当前农田前沿的总变异。
        :param previous_frontier_tv: 上一步农田前沿的总变异。
        :param current_weeds: 当前杂草数量。
        :param previous_weeds: 上一步杂草数量。
        :return: 本步奖励。
        """
        reward = self.coefficients['base_penalty']

        # 转向相关奖励
        turn_gap_reward = self.coefficients['turn_gap_coef'] * abs(current_steer - previous_steer) / self.angular_range.mode
        # turn_direction_reward = self.coefficients['turn_direction_coef'] * (
        #     0.0 if (current_steer * previous_steer >= 0 or (current_steer == 0 and previous_steer == 0)) else 1.0
        # ) # 其实后面的判断可以简化为 current_steer * previous_steer >= 0
        turn_direction_reward = self.coefficients['turn_direction_coef'] * (
            0.0 if current_steer * previous_steer >= 0  else 1.0)
        turn_self_reward = self.coefficients['turn_self_coef'] * (0.4 - abs(current_steer / self.angular_range.max) ** 0.5)
        reward += self.coefficients['turn_total_coef']*(turn_gap_reward + turn_direction_reward + turn_self_reward)

        # 农田探索相关奖励
        frontier_coverage_reward = self.coefficients['frontier_coverage_coef'] * (previous_frontier_area - current_frontier_area) / (2 * MowerAgent.width * self.speed_range.max)
        frontier_tv_reward = self.coefficients['frontier_tv_coef'] * (previous_frontier_tv - current_frontier_tv) / self.speed_range.max
        frontier_reward = self.coefficients['frontier_total_coef'] * (frontier_coverage_reward + frontier_tv_reward)
        reward += frontier_reward

        # 杂草移除相关奖励
        weed_removal_reward = self.coefficients['weed_removal_coef'] * (previous_weeds - current_weeds)
        reward += weed_removal_reward

        # 奖励归零处理
        if abs(reward) < 1e-8:
            reward = 0.0

        return float(reward)