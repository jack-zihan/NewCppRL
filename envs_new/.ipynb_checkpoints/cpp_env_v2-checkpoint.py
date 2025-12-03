"""
CppEnv v2 - APF增强观察与奖励环境
基于新的模块化架构，优化后的简洁实现
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional
from gymnasium.wrappers import HumanRendering
from scipy.ndimage import map_coordinates

from envs_new.cpp_env_base import CppEnvBase
from envs_new.components.config.environment_config import EnvironmentConfig
from envs_new.components.state.environment_state import EnvironmentState
from envs_new.components.reward.reward_system import RewardCalculator
from envs_new.utils.math_utils import total_variation_mat
from cpu_apf import cpu_apf_bool
import cupy as cp
from envs_new.utils.gpu_apf import gpu_apf_bool

from envs_new.components.map.map_components import OverlapMapCreator
from envs_new.components.dynamics import CoverageOverlapUpdater

class APFCalculator(RewardCalculator):
    """基于APF势场变化的奖励计算器"""

    @classmethod
    def calculate(cls, env_state: EnvironmentState, coefficient: float,
                  config: EnvironmentConfig = None, **kwargs) -> float:
        """计算基于势场变化的奖励"""
        agent_pos_info = env_state.get_info('agent_position')
        if len(agent_pos_info) <= 1 or 'map_dict' not in kwargs: return 0.0

        apf = kwargs['map_dict']['apf']

        if config.apf_interpolate:
            # 获得当前帧和上一帧的连续坐标（减0.5以对齐像素中心）
            coords = np.array([[agent_pos_info.current[1], agent_pos_info.last[1]],
                               [agent_pos_info.current[0], agent_pos_info.last[0]]]) - 0.5
            
            # 批量插值：4个势场 × 2个位置, prefilter=False 表明线性插值不需要预滤波，加速插值
            field_vals = map_coordinates(apf[0], coords, order=1, mode='nearest', prefilter=False)
            obstacle_vals = map_coordinates(apf[2], coords, order=1, mode='nearest', prefilter=False)
            weed_vals = map_coordinates(apf[3], coords, order=1, mode='nearest', prefilter=False)
            
            # 计算奖励增量
            reward_field = 1.0 * (field_vals[0] - field_vals[1])
            reward_obstacle = min(-0.3 * (obstacle_vals[0] - obstacle_vals[1]), 0)
            reward_weed = 3.0 * (weed_vals[0] - weed_vals[1])
            
            # 轨迹奖励（可选）
            if len(apf) > 4:
                traj_vals = map_coordinates(apf[-1], coords, order=1, mode='nearest', prefilter=False)
                reward_trajectory = min(-0.0 * (traj_vals[0] - traj_vals[1]), 0)
            else:
                reward_trajectory = 0
        else: # 原有离散逻辑
            x_curr, y_curr = int(agent_pos_info.current[0]), int(agent_pos_info.current[1])
            x_prev, y_prev = int(agent_pos_info.last[0]), int(agent_pos_info.last[1])

            # 计算各势场的变化量
            reward_field = 1.0 * (apf[0][y_curr, x_curr] - apf[0][y_prev, x_prev])
            reward_obstacle = min(-0.3 * (apf[2][y_curr, x_curr] - apf[2][y_prev, x_prev]), 0)  # 只保留惩罚负奖励
            reward_weed = 3.0 * (apf[3][y_curr, x_curr] - apf[3][y_prev, x_prev])

            # 轨迹奖励（可选）
            reward_trajectory = min(-0.0 * (apf[-1][y_curr, x_curr] - apf[-1][y_prev, x_prev]), 0) \
                if len(apf) > 4 else 0

        return coefficient * (reward_field + reward_obstacle + reward_weed + reward_trajectory)


class CppEnv(CppEnvBase):
    """v2环境 - 使用APF(人工势场)进行观察增强和奖励计算"""

    def __init__(self, render_mode="rgb_array", **kwargs): # 目前只支持 rgb_array, human由warpper实现
        """初始化v2环境，配置APF特性"""
        # v2核心特性：apf观察和奖励, 一定要有mist
        v2_configs = {'reward_apf': 1.0, 'use_apf': True, 'use_trajectory': True, 'use_mist': True, 'render_mist': True,
                      # "state_size":(768, 768),  "state_downsize":(768, 768), "multiscale_feature_size":96,
                      }

        super().__init__(render_mode=render_mode, **{**v2_configs, **kwargs})
        if self.config.use_apf: self.reward_system.add_calculator("apf", APFCalculator)  # 注册APF奖励计算器
        #
        self.scenario_generator.add_component('overlap', OverlapMapCreator())
        self.env_dynamics.add_updater('coverage_overlap', CoverageOverlapUpdater())

    def _get_observation_channels(self) -> int:
        """v2环境的观察通道数：field, mist_inv, obstacle, weed, (trajectory)
        11月17日加上了overlap, time_series_coveraged_field"""
        return 4 + 1 # time_series_coveraged_field去掉了 int(self.config.use_trajectory)

        # return 4 + int(self.config.use_trajectory)

    def get_discounted_apf(self, binary_map: np.ndarray, propagate_distance: int,
                           eps: Optional[float] = None, pad: bool = False) -> np.ndarray:
        """将二值地图转换为距离衰减势场"""
        if pad: binary_map = np.pad(binary_map, [[1, 1], [1, 1]], mode='constant', constant_values=1)
        # 训练时根据环境的device属性（由GymWrapper设置）配置Apf计算设备
        device = getattr(self, 'device', None)
        if device is not None and 'cuda' in str(device):
            device_id = int(str(device).split(':')[1]) # GPU版本：使用gpu_apf_bool， 并确保在正确的GPU设备上运行
            with cp.cuda.Device(device_id):
                distance_map, is_empty = gpu_apf_bool(binary_map)
        else:
            distance_map, is_empty = cpu_apf_bool(binary_map) # CPU版本：使用cpu_apf_bool'

        if not is_empty:
            # 指数衰减：势能 = gamma^距离
            gamma = (propagate_distance - 1) / propagate_distance
            potential_field = gamma ** distance_map

            # 截断过小值
            if eps is None: eps = gamma ** propagate_distance
            potential_field = np.where(potential_field < eps, 0., potential_field)
        else:
            potential_field = distance_map

        if pad: potential_field = potential_field[1:-1, 1:-1]
        return potential_field

    def _get_observation_maps(self) -> Dict[str, Dict[str, Any]]:
        """生成APF增强的观察地图"""
        # 提取原始地图
        field, obstacle, mist = self.maps_dict['field'], self.maps_dict['obstacle'], self.maps_dict['mist']
        trajectory = self.maps_dict['trajectory'] if self.config.use_trajectory else np.zeros(
            (self.env_state.dimensions))

        weed = self.maps_dict['weed_noisy'] if (self.config.weed_noise and self.np_random.uniform() < self.config.weed_noise) \
            else self.maps_dict['weed']

        # 提取边界
        field_edges = np.logical_and(total_variation_mat(field), mist)
        obstacle_edges = np.logical_and(total_variation_mat(obstacle), mist)
        weed_filtered = np.logical_and(weed, mist)

        # apf变换或直接使用
        if self.config.use_apf:
            obs_field = self.get_discounted_apf(field_edges, 30) # 原版30 64
            obs_obstacle = self.get_discounted_apf(obstacle_edges, 3, pad=True)
            obs_obstacle = np.maximum(obs_obstacle, np.logical_and(obstacle, mist))
            obs_weed = self.get_discounted_apf(weed_filtered, 40, 1e-2) # 原版40
            obs_trajectory = self.get_discounted_apf(trajectory, 4) if self.config.use_trajectory else trajectory
        else:
            obs_field = field_edges.astype(float)
            obs_obstacle = obstacle_edges.astype(float)
            obs_weed = weed_filtered.astype(float)
            obs_trajectory = trajectory.astype(float)

        # 组装观察数组供奖励计算
        layers = [obs_field, np.logical_not(mist).astype(float), obs_obstacle, obs_weed]
        # if self.config.use_trajectory: layers.append(obs_trajectory)

        # 保存APF数组供奖励计算
        self.maps_dict['apf'] = np.stack(layers, axis=0) if self.config.use_apf else None

        # 返回字典格式供观察生成器使用
        obs_maps = {
            'field': {'map': obs_field, 'pad': 0.0},
            'mist_inv': {'map': np.logical_not(mist), 'pad': 0.0},
            'obstacle': {'map': obs_obstacle, 'pad': 1.0},
            'weed': {'map': obs_weed, 'pad': 0.0},
        }
        # if self.config.use_trajectory:
        #     obs_maps['trajectory'] = {'map': obs_trajectory, 'pad': 0.0}

        # 重复覆盖热图：归一化到[0,1]，上限由overlap_tolerance控制（对应当前阶段的冗余预算R*）
        normalized_overlap = (np.clip(self.maps_dict['overlap'] + 1, 0, self.config.overlap_tolerance + 1)
                              / (self.config.overlap_tolerance + 1))  # 当观测=1.0时，表示该区域已达到奖励breakeven阈值
        obs_maps['overlap'] = {'map': normalized_overlap.astype(np.float32), 'pad': 0.0}

        # # 覆盖顺序秩通道（稳定秩）：rank = order_label / total_field_area，未覆盖=0
        # normalized_order_map = (self.maps_dict['time_series_coveraged_field'].astype(np.float32) /
        #                         float(int(self.env_state.total_field_area))).astype(np.float32)
        # obs_maps['time_series_coveraged_field'] = {'map': normalized_order_map, 'pad': 0.0}

        return obs_maps

    def _get_step_info(self) -> Dict[str, Any]:
        """base基础上增加overlap_count信息"""
        step_info = super()._get_step_info() # 继承base的字段
        step_info['overlap_count'] = np.array(self.env_state.overlap_count, dtype=np.float32)
        return step_info

if __name__ == "__main__":
    if_render = True
    episodes = 3
    real_map_dir = '/home/lzh/NewCppRL/envs_new/maps/weed_coverage/real_true'
    env = CppEnv(
        # render_first_person=True,  # 控制渲染第一人称视角
        # use_multiscale=False, # 是否使用多尺度观察
        # use_global_features=False,
        # num_obstacles_range = [0, 0]
        # map_dir = "envs_new/maps/weed_coverage"  # 默认指向weed_coverage根目录
        # map_dir = "envs_new/maps/indoor_coverage",
        boundary_source = "field"
    )
    # Note: HumanRendering wrapper would be added here if available
    if if_render: env: CppEnv = HumanRendering(env)  # 封装后，只接收render_mode="rgb_array"的env，使得step和reset的时候展示渲染图像

    for _ in range(episodes):
        # env.set_obstacle_range([0,0])
        obs, info = env.reset(seed=120, options={
            'weed_distribution': 'gaussian',  # Updated parameter name
            # 'map_id': 80,
            "weed_count": 100,  # Updated parameter name
            # "specific_scenario_dir": real_map_dir
        })
        env.action_space.seed(66)
        done = False
        while not done:
            action = env.action_space.sample()
            # action = 1 * 21 + 10
            obs, reward, done, truncated, info = env.step(action)
            # obs, reward, done, _, info = env.step((0, 4))
            print(reward)
            if if_render:
                img = env.render()

    env.close()
