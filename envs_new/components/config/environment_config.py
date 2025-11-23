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
    action_type: str = "continuous" #"discrete"
    max_episode_steps: int = 30000
    state_history_length: int = 2 # 观察历史长度
    use_history_vector: bool = False  # 是否使用历史序列向量模式（v4/v6启用）
    
    # 地图配置参数（原MapConfig）
    map_dir: str = "envs_new/maps/weed_coverage"  # 默认指向weed_coverage根目录
    num_obstacles_range: Tuple[int, int] = (5, 8)
    obstacle_size_range: Tuple[int, int] = (10, 25)
    use_box_boundary: bool = True
    weed_noise: float = 0.0
    use_trajectory: bool = True
    use_mist: bool = True
    use_apf: bool = True
    apf_interpolate: bool = False #True  # APF场值双线性插值开关（使用连续坐标而非离散网格）
    ensure_field_visibility: bool = True
    exclude_weeds_near_obstacles: bool = True

    # 场景生成参数
    obstacle_expand_pixels: int = 15          # 障碍物周围不生成农田的像素数
    obstacle_min_distance_to_edge: int = 100  # 障碍物离地图边缘最小像素数
    obstacle_min_distance_to_agent: float = 2.0  # 障碍物离智能体最少几倍体长
    boundary_expand_ratio: float = 1.2        # 障碍物边界扩展比例（1.2=扩大20%）
    boundary_min_expand_pixels: int = 60      # 边界最小扩展像素数
    weed_avoid_obstacle_pixels: int = 29      # 障碍物周围不生成杂草的像素数
    
    # 智能体配置参数（原AgentConfig）
    agent_width: float = 4.0
    agent_length: float = 6.0
    agent_vision_length: float = 28.0
    agent_vision_angle: float = 75.0
    coverage_extended_px: float = 1.0 # 先用2试试，后面看1，至少向外扩1个像素

    # 动作空间配置参数（原ActionConfig）
    v_min: float = 0.0
    v_max: float = 3.5
    w_min: float = -28.6
    w_max: float = 28.6
    action_nvec: Tuple[int, int] = (7, 21)
    
    # 观察配置参数（原ObservationConfig）
    state_size: Tuple[int, int] = (128, 128) # (256, 256) #(128, 128)
    state_downsize: Tuple[int, int] = (128, 128) #(128, 128)
    use_multiscale: bool = True
    n_scales: int = 4
    multiscale_feature_size: int = 16 # 16
    use_global_features: bool = True #True
    position_noise: float = 0.0
    direction_noise: float = 0.0
    
    # 奖励配置参数（清晰命名，与Calculator名称一致）
    reward_base_penalty: float = -0.1  # 基础惩罚，鼓励快速完成任务
    reward_weed_removal: float = 20.0  # 杂草清除奖励

    reward_field_coverage: float = 1.0  # 田地覆盖奖励
    reward_field_variation: float = 0.5  # 田地变化奖励（原tv_coef）
    reward_field_group_coef: float = 0.125  # 田地相关奖励组级别系数

    reward_turning_penalty: float = -0.5  # 转向加速度惩罚（原turn_gap_coef）
    reward_direction_change_penalty: float = -0.30  # 方向改变惩罚（原turn_direction_coef）
    reward_steering_smoothness: float = 0.25  # 转向平滑性奖励（原turn_self_coef）
    reward_turning_group_coef: float = 0.0  # 转向相关组级别系数

    reward_collision_penalty: float = -399.0  # 碰撞惩罚
    reward_completion_bonus: float = 500.0  # 任务完成奖励
    reward_overlap_penalty: float = 0.0  # 重复覆盖惩罚（默认关闭）

    overlap_tolerance: float = 1.0 # 定义重复覆盖容忍度观测归一化的上限，同时隐式编码奖励比值 R* = field_group / |overlap_penalty|

    reward_apf: float = 1.0  # APF奖励系数（v2环境特有）
    reward_hif: float = 0.0  # HIF方向引导奖励系数（v5环境特有）
    
    # 渲染配置参数（原RenderConfig）
    render_modes: Tuple[str, ...] = ("rgb_array", "first_person")
    render_fps: int = 50
    render_repeat_times: int = 2 # 控制渲染时返回的渲染图片分辨率，基础分辨率是传入图像的尺寸（一般为400*400）
    render_first_person: bool = False  # 控制rgb_array模式下是否渲染第一人称视角
    render_tv: bool = False # 暂时没有渲染TV的功能，这个功能在老环境的Warpper和uitils/visualizer中实现
    render_mist: bool = False
    render_covered_weed: bool = True
    render_covered_field: bool = True

    # HIF可视化参数
    render_hif_lines: bool = True  # HIF线条绘制开关
    hif_line_stride: int = 12  # 采样密度（像素间隔）
    hif_line_length: float = 8.0  # 线段基础长度（会乘以render_scale）
    hif_line_thickness: int = 1  # 线条基础粗细（会按render_scale缩放）
    hif_line_color: Tuple[int, int, int] = (0, 255, 255)  # BGR颜色（默认青色）
    hif_line_confidence_threshold: float = 0.2  # 最小置信度阈值

    # hif相关内容
    include_confidence: bool = True  # 是否在观察中包含置信度通道（hif环境特有）
    hif_min_epsilon: float = 1e-6 # 防止除零错误的最小置信度值（hif环境特有）
    temporal_decay_rate: float = 0.85 # 置信度的时间衰减率（hif环境特有）
    spatial_spread_sigma: float = 3.0 # 置信度的空间扩散标准差（hif环境特有）

    # 课程学习：田地面积缩放（默认关闭）
    # 当启用时，在 reset 时按区间随机采样一个缩放因子 s∈[min,max]，
    # 以图像中心为原点对 field（以及 v5/v6 的 HIF）做同心等比缩放，保持画布尺寸不变。
    field_scale_enabled: bool = False
    field_scale_range: Tuple[float, float] = (1.0, 1.0)

    def __post_init__(self):
        """最小化验证，只检查关键约束"""
        if self.v_min >= self.v_max:
            raise ValueError(f"v_min ({self.v_min}) must be less than v_max ({self.v_max})")
        if self.w_min >= self.w_max:
            raise ValueError(f"w_min ({self.w_min}) must be less than w_max ({self.w_max})")
    
    def get_reward_coefficients(self) -> Dict[str, float]:
        """获取所有奖励系数的字典形式"""
        return {
            'base_penalty': self.reward_base_penalty,
            'weed_removal': self.reward_weed_removal,
            'field_coverage': self.reward_field_coverage,
            'field_variation': self.reward_field_variation,
            'field_group_coef': self.reward_field_group_coef,
            'turning_penalty': self.reward_turning_penalty,
            'direction_change_penalty': self.reward_direction_change_penalty,
            'steering_smoothness': self.reward_steering_smoothness,
            'turning_group_coef': self.reward_turning_group_coef,
            'collision_penalty': self.reward_collision_penalty,
            'completion_bonus': self.reward_completion_bonus,
            'overlap_penalty': self.reward_overlap_penalty,
            "apf": self.reward_apf,
            "hif": self.reward_hif
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
        return NumericalRange(self.v_min, self.v_max)
    
    @property
    def w_range(self) -> 'NumericalRange':
        return NumericalRange(self.w_min, self.w_max)


@dataclass
class NumericalRange:
    min: float
    max: float
    
    @property
    def mode(self) -> float:
        """Returns the range span (max - min)."""
        return self.max - self.min
