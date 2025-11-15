"""
V6环境：时空鲁棒的HIF引导
通过历史轨迹的时间衰减和空间扩散，生成稳定的方向引导奖励
"""
from __future__ import annotations
import numpy as np
import cv2
import math
from typing import Dict, Any, Optional, List, Tuple
from gymnasium.wrappers import HumanRendering
from envs_new.cpp_env_v4 import CppEnv as CppEnvV4
from envs_new.cpp_env_v5 import CppEnv as CppEnvV5
from envs_new.components.reward.reward_system import RewardCalculator
from envs_new.utils.image_utils import extract_ego_patch, apply_noise_to_pose


class CppEnv(CppEnvV5):
    """V6环境：增强的时空HIF引导，action从(v, w)变为轨迹, 并考虑课程学习，之后v7版本蒋输出动作转换为基元
    目标：通过多点平均降低单点噪声影响
    实现：1. 轨迹权重地图：历史轨迹的时间衰减和空间扩散 2. 增强HIF奖励：使用时空加权的方向差计算
    """

    def __init__(self, render_mode="rgb_array", **kwargs):
        v6_defaults = {'reward_hif': 0.2, 'use_history_vector': True, "state_history_length": 16,
                       # v6显式声明继承v4的历史向量模式， 使用更长的历史（vector维度: 2 + 6*16 = 98）
                       # 'map_dir': "envs_new/maps/field_coverage"
                       }
        final_kwargs = {**v6_defaults, **kwargs}
        super().__init__(render_mode=render_mode, **final_kwargs)

        self.reward_system.add_calculator('hif', EnhancedHIFCalculator)  # 替换为时空扩散版奖励

    def _get_observation_maps(self) -> Dict[str, Dict[str, Any]]:
        """时空轨迹权重地图 + v4基础地图field, obstacle, coverlap, coverage with time serial. trajectory"""
        obs = CppEnvV4._get_observation_maps(self)  # v6不包含v5的HIF观察通道，仅保留v4基础观察
        self._generate_trajectory_weight_maps()  # 生成轨迹权重相关的三张地图('trajectory_weights', 'trajectory_cos', 'trajectory_sin')存入map_dict

        # 添加轨迹权重到观测（归一化到[0,1]）
        # ps, 轨迹朝向信息不用编码进来，因为转换后都变成了自身坐标系轨迹信息，这个和
        weight_map = self.maps_dict['trajectory_weights']
        normalized_weight = self._normalize_weight_map(weight_map)
        obs['trajectory_weights'] = {'map': normalized_weight, 'pad': 0.0}

        return obs

    def _generate_trajectory_weight_maps(self):
        """构建轨迹权重和方向地图  trajectory_cos: 方向余弦分量（用于计算平均方向）, trajectory_sin: 方向正弦分量（用于计算平均方向）,trajectory_weights: 时空权重分布
           选择时空扩散完再归一化的原因
          - 我们要的是“在一个邻域里，按可信度加权后的平均方向”。可信度来自时间衰减/多次经过，体现在 weight 上。先对每个像素把 cos/weight、sin/weight 做成“单位方向”，再做卷积，相当于把所有像素都当成“同等可信”的
            方向再平均，丢掉了原本的权重信息。
          - 正确做法是先把“加权后的分子”和“权重分母”分别叠加并扩散（卷积），最后再相除，这样邻域平均自然会同时考虑“方向”和“这个方向有多靠谱”。
            卷积是线性操作，除法是非线性的，Conv(cos/weight) ≠ Conv(cos)/Conv(weight)。我们要的结果恰好是后者这种“分子分母分别卷积再相除”的形式。
        """
        height, width = self.maps_dict['field'].shape

        trajectory = self._get_trajectory_history()  # 获取历史轨迹
        if not trajectory:  # 第一帧创建轨迹地图，这个信号是trajectory的map形式附属空间信号，不在map_gernaror和updater中操作
            for tj_map in ['trajectory_weights', 'trajectory_cos', 'trajectory_sin']:
                self.maps_dict[tj_map] = np.zeros((height, width), dtype=np.float32)
            return

        # 第一步：获得时间扩散累计地图
        time_diffused_maps = self._generate_time_diffusion_maps(trajectory, (height, width))

        # 第二步：通过高斯卷积进行空间扩散获得累加地图
        kernel, borderType = self._create_gaussian_kernel(self.config.spatial_spread_sigma), cv2.BORDER_CONSTANT
        for tj_map in ['trajectory_weights', 'trajectory_cos', 'trajectory_sin']:
            self.maps_dict[tj_map] = cv2.filter2D(time_diffused_maps[tj_map], -1, kernel, borderType=borderType)
        # 权重在2D高斯滤波时候多项极小值卷积可能变负导致不通过Gym观测空间检查，因此添加一个强制阶段的数值保障
        self.maps_dict['trajectory_weights'] = np.maximum(self.maps_dict['trajectory_weights'], 0.0).astype(np.float32)

        # 第三步，cos/sin加权累积值除以权重得到平均方向
        for tj_map in ['trajectory_cos', 'trajectory_sin']:
            self.maps_dict[tj_map] = self.maps_dict[tj_map]  / (self.maps_dict['trajectory_weights'] + 1e-6)
            self.maps_dict[tj_map][self.maps_dict['trajectory_weights']<1e-6] = 0.0  # 同样去除无效值区域对卷积后数值精度的误差来的方向误差引导

    def _get_trajectory_history(self) -> List[Tuple[int, int, float]]:
        """获取历史轨迹数据 [(x, y, direction_deg), ...] 从旧到新排列"""
        positions = list(self.env_state.get_info('agent_position').history)
        directions = list(self.env_state.get_info('agent_direction').history)
        trajectory = [(int(pos[0]), int(pos[1]), float(dir_deg))
                      for pos, dir_deg in zip(positions, directions)]
        return trajectory

    def _generate_time_diffusion_maps(self, trajectory: List[Tuple[int, int, float]],
                                      shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """将轨迹点投射到稀疏地图上，考虑时间衰减和方向编码"""
        height, width = shape
        weight_map = np.zeros((height, width), dtype=np.float32)
        cos_map = np.zeros((height, width), dtype=np.float32)
        sin_map = np.zeros((height, width), dtype=np.float32)

        # 遍历轨迹点，reversed使得最新的点在前面，time_idx=0对应当前时刻从近到远
        for time_idx, (x, y, direction_deg) in enumerate(reversed(trajectory)):
            if not (0 <= x < width and 0 <= y < height): continue  # 边界检查
            time_weight = self.config.temporal_decay_rate ** time_idx  # 时间衰减权重（当前为1，越久远越小）
            double_angle = 2.0 * math.radians(direction_deg) % math.pi  # 图像坐标系轴向角度（双倍角编码处理无向性）

            # 累加到稀疏地图（注意：可能有多个历史点落在同一位置）
            weight_map[y, x] += time_weight
            cos_map[y, x] += time_weight * math.cos(double_angle)
            sin_map[y, x] += time_weight * math.sin(double_angle)
        return {'trajectory_weights': weight_map, 'trajectory_cos': cos_map, 'trajectory_sin': sin_map}

    def _create_gaussian_kernel(self, sigma: float) -> np.ndarray:
        """创建高斯卷积核, sigma: 高斯分布标准差, 2D高斯核（归一化）"""
        # 核半径取3倍标准差（覆盖99.7%的分布）
        kernel_radius = int(math.ceil(3 * sigma))
        x = np.arange(-kernel_radius, kernel_radius + 1, dtype=np.float32)

        # 1D高斯分布
        gaussian_1d = np.exp(-(x ** 2) / (2 * sigma ** 2))
        gaussian_1d /= gaussian_1d.sum()

        # 2D高斯核（外积）
        kernel_2d = np.outer(gaussian_1d, gaussian_1d)
        return kernel_2d.astype(np.float32)

    def _normalize_weight_map(self, weight_map: np.ndarray) -> np.ndarray:
        """归一化权重地图到[0,1] 使用99分位数归一化，避免异常值影响 """
        if weight_map.max() <= 0: return weight_map

        # 获取正权重的99分位值作为归一化范围，避免异常值
        percentile_99 = np.percentile(weight_map[weight_map > 0], 99)
        normalized = np.minimum(weight_map / (percentile_99 + 1e-6), 1.0)
        return normalized.astype(np.float32)

    def _get_observation_channels(self) -> int:
        """通道数：4基础 (field, obstacle, time_series_coveraged_field, overlap) + trajectory + trajectory_weights"""
        return 4 + int(self.config.use_trajectory) + 1


class EnhancedHIFCalculator(RewardCalculator):
    """增强的HIF方向引导奖励计算器，使用时空加权的方向差计算，对噪声和-1区域更加鲁棒"""

    @classmethod
    def calculate(cls, env_state, coefficient: float, config=None, **kwargs) -> float:
        """
        计算时空加权的HIF引导奖励
        Args: env_state: 环境状态；coefficient: 奖励系数；config: 环境配置；**kwargs: 包含map_dict的额外参数
        """
        map_dict = kwargs.get('map_dict')
        hif_map, weight_map = map_dict['hif'], map_dict['trajectory_weights']

        # 获取HIF不为-1且有足够的轨迹权重的有效位置(取0.01为最小阈值)
        valid_mask = (hif_map >= 0) & (weight_map > 0.01)
        if not np.any(valid_mask): return 0.0  # 无有效位置返回0奖励

        # 提取有效hif和轨迹方向值, 并双倍角编码
        valid_traj_weights = weight_map[valid_mask]
        hif_cos, hif_sin = np.cos(2 * hif_map[valid_mask]), np.sin(2 * hif_map[valid_mask])
        trajectory_cos, trajectory_sin = map_dict['trajectory_cos'][valid_mask], map_dict['trajectory_sin'][valid_mask]

        # 计算轴向相似度（使用向量点积）, 这个在论文中需要写公式简单推导一下啊
        #   - 记 m = [trajectory_cos, trajectory_sin] 是“单位向量在邻域的按权平均”，因此范数 ||m|| ∈ [0, 1]。
        #   - 记 h = [hif_cos, hif_sin]，它是单位向量，||h|| = 1。
        #   - 则 cos_similarity = m·h。由柯西-施瓦茨不等式：|m·h| ≤ ||m||·||h|| ≤ 1，因此 cos_similarity ∈ [-1, 1]。
        #   - 直观等价：m 的模 r 表示方向一致性，m·h = r·cos(2Δ)，r∈[0,1]、cos(2Δ)∈[-1,1]，乘积自然在 [-1,1]。
        cos_similarity = np.clip(trajectory_cos * hif_cos + trajectory_sin * hif_sin, -1.0, 1.0) # 但是为了防止浮点误差，还是clip一下

        # 轴向角度差（范围[0, π/2]）
        # 注：先点积再取 arccos 是必要的， 它把“非线性的相似度”变成“线性的角度差”，
        # 轨迹场是“单位向量在邻域的按权平均”，得到均值向量 m=[cos̄2, sin̄2]，其模长 r=||m||∈[0,1] 反映“方向一致性/集中度”。
        # 点积同时编码了两件事 —— 方向差 和 方向一致性（r 越大，点积越接近真实角度差，r 越小，点积越接近0，表示方向混乱无意义）。点积= r·cos(2Δ)，r 随数据分散度变化，直接用它做奖励，系数标尺会随场景内局部一致性漂移，导致调参不稳。用角差，再单独用权重图做二次加权，能把“方向差”和“可信度/样本量”解耦，语义更清晰。
        # 若用 1−cos_similarity 或直接 cos_similarity，靠近最优 Δ≈0 时，cos(2Δ)≈1−2Δ²，梯度近零（惩罚是二次型），对微小偏差的驱动力很弱；而用角差 Δ 本身是线性的，对齐附近也有稳定梯度，收敛更灵敏。当前“点积→arccos→折半→线性按权平均→归一化”的实现更可靠、更易调
        angle_differences = 0.5 * np.arccos(cos_similarity) # 将相似度转化为轴向角度差（0-pi/2）, 因为是以2Δ角进行cos编码的，转化后*0.5得到Δ
        mean_angle_differences = np.sum(valid_traj_weights * angle_differences) / np.sum(valid_traj_weights)  # 按权重加权平均角度差
        normalized_reward = -mean_angle_differences / (0.5 * math.pi) # 归一化到[-1, 0]，差异越小奖励越高（接近0）
        return coefficient * normalized_reward  # 返回负奖励（差异越大惩罚越大）

if __name__ == "__main__":
    if_render = True
    """简单的功能测试"""
    print("创建V6环境进行测试...")

    # 创建环境
    env = CppEnv(
        render_mode="rgb_array",
        field_scale_enabled=False,
        field_scale_range=(1.0, 1.0),  # (1.0, 1.0)
        # render_first_person=True,  # 控制渲染第一人称视角
        # map_dir='envs_new/maps/field_coverage'
    )
    if if_render: env = HumanRendering(env)

    # 重置环境
    obs, info = env.reset(seed=99)
    env.action_space.seed(7)
    print(f"环境重置成功，观测形状: {obs}")

    # 执行几步
    for step in range(300):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {step + 1}: Reward = {reward:.4f}, Done = {done}")

        if done:
            break
        if if_render:
            env.render()
    # 检查轨迹权重地图
    if 'trajectory_weights' in env.maps_dict:
        weight_map = env.maps_dict['trajectory_weights']
        print(f"轨迹权重地图形状: {weight_map.shape}")
        print(f"权重范围: [{weight_map.min():.4f}, {weight_map.max():.4f}]")

    print("V6环境测试完成！")
