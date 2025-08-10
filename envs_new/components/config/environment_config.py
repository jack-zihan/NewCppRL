"""
Environment configuration management for the mowing robot simulation.
Provides a unified, flattened configuration for all environment components.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any
from pathlib import Path


@dataclass
class EnvironmentConfig:
    """完全扁平化的环境配置，简洁高效"""
    
    # 核心环境参数
    action_type: str = "discrete"
    max_episode_steps: int = 3000
    
    # 地图配置参数（原MapConfig）
    map_dir: str = "envs/maps/1-400"
    num_obstacles_range: Tuple[int, int] = (5, 8)
    obstacle_size_range: Tuple[int, int] = (10, 25)
    use_box_boundary: bool = True
    weed_noise: float = 0.0
    use_traj: bool = True
    use_mist: bool = True
    use_apf: bool = True
    ensure_frontier_visibility: bool = True
    exclude_weeds_near_obstacles: bool = True
    
    # 智能体配置参数（原AgentConfig）
    agent_width: float = 4.0
    agent_length: float = 6.0
    agent_vision_length: float = 28.0
    agent_vision_angle: float = 75.0
    
    # 动作空间配置参数（原ActionConfig）
    v_min: float = 0.0
    v_max: float = 3.5
    w_min: float = -28.6
    w_max: float = 28.6
    action_nvec: Tuple[int, int] = (7, 21)
    
    # 观察配置参数（原ObservationConfig）
    state_size: Tuple[int, int] = (128, 128)
    state_downsize: Tuple[int, int] = (128, 128)
    use_multiscale: bool = True
    n_scales: int = 4
    multiscale_feature_size: int = 16
    use_global_features: bool = True
    use_trajectory: bool = True
    obs_use_mist: bool = True  # 重命名避免与map的use_mist冲突
    position_noise: float = 0.0
    direction_noise: float = 0.0
    
    # 奖励配置参数（原RewardConfig.coefficients）
    reward_turn_total_coef: float = 0.0
    reward_turn_gap_coef: float = -0.5
    reward_turn_direction_coef: float = -0.30
    reward_turn_self_coef: float = 0.25
    reward_frontier_total_coef: float = 0.125
    reward_frontier_coverage_coef: float = 1.0
    reward_frontier_tv_coef: float = 0.5
    reward_base_penalty: float = -0.1
    reward_weed_removal_coef: float = 20.0
    reward_collision_penalty: float = -399.0
    reward_completion_bonus: float = 500.0
    
    # 渲染配置参数（原RenderConfig）
    render_modes: Tuple[str, ...] = ("rgb_array", "first_person")
    render_fps: int = 50
    render_repeat_times: int = 2
    render_first_person: bool = False  # 控制rgb_array模式下是否渲染第一人称视角
    render_tv: bool = False
    render_mist: bool = False
    render_covered_weed: bool = True
    render_covered_farmland: bool = True
    
    def __post_init__(self):
        """最小化验证，只检查关键约束"""
        if self.v_min >= self.v_max:
            raise ValueError(f"v_min ({self.v_min}) must be less than v_max ({self.v_max})")
        if self.w_min >= self.w_max:
            raise ValueError(f"w_min ({self.w_min}) must be less than w_max ({self.w_max})")
    
    def get_reward_coefficients(self) -> Dict[str, float]:
        """获取奖励系数字典，保持与原RewardConfig兼容"""
        return {
            'turn_total_coef': self.reward_turn_total_coef,
            'turn_gap_coef': self.reward_turn_gap_coef,
            'turn_direction_coef': self.reward_turn_direction_coef,
            'turn_self_coef': self.reward_turn_self_coef,
            'frontier_total_coef': self.reward_frontier_total_coef,
            'frontier_coverage_coef': self.reward_frontier_coverage_coef,
            'frontier_tv_coef': self.reward_frontier_tv_coef,
            'base_penalty': self.reward_base_penalty,
            'weed_removal_coef': self.reward_weed_removal_coef,
            'collision_penalty': self.reward_collision_penalty,
            'completion_bonus': self.reward_completion_bonus
        }
    
    def get_absolute_map_dir(self) -> Path:
        """获取地图目录的绝对路径"""
        if Path(self.map_dir).is_absolute():
            return Path(self.map_dir)
        else:
            # 使用简单的路径拼接，避免复杂的项目根目录查找
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent  # 回到项目根目录
            return project_root / self.map_dir
    
    @property
    def v_range(self) -> 'NumericalRange':
        """为了兼容性提供v_range属性"""
        return NumericalRange(self.v_min, self.v_max)
    
    @property
    def w_range(self) -> 'NumericalRange':
        """为了兼容性提供w_range属性"""
        return NumericalRange(self.w_min, self.w_max)


@dataclass
class NumericalRange:
    """保留此类以保持兼容性"""
    min: float
    max: float
    
    @property
    def mode(self) -> float:
        """Returns the range span (max - min)."""
        return self.max - self.min