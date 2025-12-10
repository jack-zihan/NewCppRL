"""
CppEnv v4 - 移除杂草相关逻辑，专注于田地覆盖任务， 保留所有配置灵活性（mist、trajectory、多尺度等）
"""
from __future__ import annotations

import math
import numpy as np
from typing import Dict, Any, List
from gymnasium.wrappers import HumanRendering

from envs_new.cpp_env_base import CppEnvBase
from envs_new.components.dynamics import FieldCoverageUpdater, FieldTaskStatusUpdater, CoverageOverlapUpdater
from envs_new.components.map.map_components import OverlapMapCreator
from envs_new.utils.math_utils import _pad_history, _to_ego_frame, _angles_to_sincos


class CppEnv(CppEnvBase):
    """
    v4环境 - 田地覆盖任务
    """

    def __init__(self, render_mode="rgb_array", **kwargs):
        v4_defaults = {'use_mist': False, 'use_apf': False, 'use_trajectory': True, 'render_mist': False,
                       "num_obstacles_range": (0, 0), "reward_completion_bonus": 15000, "reward_field_group_coef": 10,
                       # "reward_completion_bonus": 15000 设置为最大步数的两倍 (0.8*400)*(0.8*400)/(w*v~3.5*4)
                       "state_size": (768, 768), "state_downsize": (768, 768), "multiscale_feature_size": 96,
                       "use_global_features": False, 'use_history_vector': True, 'state_history_length': 2,
                       # 启用历史向量模式（vector维度: 2 + 6*L）# 显式声明历史长度
                       'map_dir': "envs_new/maps/weed_coverage" # field_coverage
                       # 'map_dir': "envs_new/maps/indoor_coverage"
                       }
        final_kwargs = {**v4_defaults, **kwargs}
        super().__init__(render_mode=render_mode, **final_kwargs)

        # 移除weed相关组件
        self.scenario_generator.remove_component('weed')
        self.reward_system.remove_calculator('weed_removal')
        self.env_dynamics.remove_updater('weed')

        # 替换field相关updater： Field探索模式转为覆盖模式, 任务统计变为field任务
        self.env_dynamics.remove_updater('field')
        self.env_dynamics.add_updater('field', FieldCoverageUpdater())
        self.env_dynamics.remove_updater('status')
        self.env_dynamics.add_updater('status', FieldTaskStatusUpdater())
        # 启用覆盖统计：map组件 + updater（默认仅v4开启，其他版本不受影响）
        self.scenario_generator.add_component('overlap', OverlapMapCreator())
        self.env_dynamics.add_updater('coverage_overlap', CoverageOverlapUpdater())

    def _get_observation_channels(self) -> int:
        """v4环境的观察通道数：field, obstacle, time_series_coveraged_field, overlap[重复覆盖热图],(trajectory)"""
        return 4 + int(self.config.use_trajectory)

    def _get_observation_maps(self) -> Dict[str, Dict[str, Any]]:
        """
        field覆盖观测：field, obstacle, time_series_coveraged_field(覆盖顺序秩，归一化)，(optional) trajectory，不包含weed and mist
        """
        obs_maps = {
            'field': {'map': self.maps_dict['field'], 'pad': 0.0},
            'obstacle': {'map': self.maps_dict['obstacle'], 'pad': 1.0},  # 边界视为障碍物
        }

        # 重复覆盖热图：归一化到[0,1]，上限由overlap_tolerance控制（对应当前阶段的冗余预算R*）
        normalized_overlap = (np.clip(self.maps_dict['overlap'] + 1, 0, self.config.overlap_tolerance + 1)
                              / (self.config.overlap_tolerance + 1))  # 当观测=1.0时，表示该区域已达到奖励breakeven阈值
        obs_maps['overlap'] = {'map': normalized_overlap.astype(np.float32), 'pad': 0.0}

        # 覆盖顺序秩通道（稳定秩）：rank = order_label / total_field_area，未覆盖=0
        normalized_order_map = (self.maps_dict['time_series_coveraged_field'].astype(np.float32) /
                                float(int(self.env_state.total_field_area))).astype(np.float32)
        obs_maps['time_series_coveraged_field'] = {'map': normalized_order_map, 'pad': 0.0}

        if self.config.use_trajectory and 'trajectory' in self.maps_dict:  # 可选轨迹地图
            obs_maps['trajectory'] = {'map': self.maps_dict['trajectory'], 'pad': 0.0}

        return obs_maps

    def _get_completion_ratio(self) -> np.ndarray:
        return np.array([self.env_state.field_coverage_ratio], dtype=np.float32)  # 田地覆盖完成率 [0, 1]

    def _get_step_info(self) -> Dict[str, Any]:
        return {
            'crashed': np.array(self.env_state.crashed, dtype=np.bool_),
            'finished': np.array(self.env_state.finished, dtype=np.bool_),
            'timeout': np.array(self.env_state.timeout, dtype=np.bool_),
            'field_area': np.array(self.env_state.field_area, dtype=np.float32),
            'field_coverage_ratio': np.array(self.env_state.field_coverage_ratio, dtype=np.float32),
            'overlap_count': np.array(self.env_state.overlap_count, dtype=np.float32),
        }

    def _get_observation_vector(self) -> np.ndarray:
        """观察向量：覆盖率 + 时间进度 + 动作与位姿历史（自我坐标系）

        返回固定长度向量：[coverage, time_progress, steer[L], speed[L],
                         dx_ego[L], dy_ego[L], heading_sin[L], heading_cos[L]]
        """
        # 1. 标量特征
        coverage = self.env_state.field_coverage_ratio
        time_prog = self.env_state.current_step / self.env_state.max_steps

        # 2. 读取历史序列（直接访问，fail-fast）
        L = self.config.state_history_length

        position_history = list(self.env_state.get_info('agent_position').history)
        direction_history = list(self.env_state.get_info('agent_direction').history)
        speed_history = list(self.env_state.get_info('agent_speed').history)
        steer_history = list(self.env_state.get_info('agent_steer').history)

        # 3. 归一化并填充动作历史
        normalized_steer = [w / self.config.w_max for w in steer_history]
        normalized_speed = [v / self.config.v_max for v in speed_history]
        padded_steer = _pad_history(normalized_steer, L, 0.0).tolist()
        padded_speed = _pad_history(normalized_speed, L, 0.0).tolist()

        # 4. 位姿历史转自我坐标系（信任数据确定性）
        ref_pos = position_history[-1]
        ref_heading = direction_history[-1]

        ego_positions = _to_ego_frame(position_history, ref_pos, ref_heading, self.env_state.dimensions)
        padded_ego_positions = _pad_history(ego_positions, L, np.array([0.0, 0.0]))
        dx_ego = padded_ego_positions[:, 0].tolist()
        dy_ego = padded_ego_positions[:, 1].tolist()

        # 5. 朝向相对编码（sin/cos避免跳变，numpy优化版本）
        heading_sincos = _angles_to_sincos(direction_history, ref_heading)
        padded_heading = _pad_history(heading_sincos, L, np.array([0.0, 1.0]))
        heading_sin = padded_heading[:, 0].tolist()
        heading_cos = padded_heading[:, 1].tolist()

        # 6. 组装返回
        return np.array([coverage, time_prog] + padded_steer + padded_speed +
                        dx_ego + dy_ego + heading_sin + heading_cos, dtype=np.float32)


if __name__ == "__main__":
    # 测试v4环境
    if_render = True
    episodes = 3

    # 创建环境，展示配置灵活性
    env = CppEnv(
        use_mist=False,  # 可选：使用迷雾
        render_mist=False,  # 可选：渲染迷雾
        use_trajectory=True,  # 可选：记录轨迹
        use_apf=False,  # 可选：不使用APF（v4默认不需要）
        use_multiscale=True,  # 可选：不使用多尺度
        num_obstacles_range=[3, 5],  # 可配置障碍物数量
        field_scale_enabled=False,
        field_scale_range=(1.0, 1.0)
    )

    if if_render:
        env = HumanRendering(env)

    for episode in range(episodes):
        print(f"\n--- Episode {episode + 1} ---")
        obs, info = env.reset(seed=120 + episode, options={
            'map_id': None,  # 使用随机地图
            'initial_position': None,
            'initial_direction': None,
        })

        print(f"Observation shape: {obs['observation'].shape}")
        print(f"Initial field coverage: {obs['completion_ratio'][0]:.2%}")

        env.action_space.seed(66)
        done = False
        step_count = 0
        total_reward = 0

        while not done:  # 限制步数用于测试
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)

            step_count += 1
            total_reward += reward

            if step_count % 20 == 0:
                print(f"  Step {step_count}: reward={reward:.4f}, "
                      f"coverage={obs['completion_ratio'][0]:.2%}")

            if if_render:
                env.render()

            if done or truncated:
                print(f"\nEpisode finished after {step_count} steps!")
                print(f"  Total reward: {total_reward:.2f}")
                print(f"  Final coverage: {obs['completion_ratio'][0]:.2%}")
                if info.get('crashed'):
                    print("  Termination: Crashed")
                elif info.get('finished'):
                    print("  Termination: Field fully covered!")
                elif info.get('timeout'):
                    print("  Termination: Timeout")
                break

    env.close()
    print("\n✅ v4 Environment test completed successfully!")
