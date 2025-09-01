"""
CppEnv v5 - 田地覆盖任务 + HIF方向引导
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from gymnasium.wrappers import HumanRendering

from envs_new.cpp_env_v4 import CppEnv as CppEnvV4
from envs_new.components.state.environment_state import EnvironmentState
from envs_new.components.config.environment_config import EnvironmentConfig
from envs_new.components.reward.reward_system import RewardCalculator


class HIFCreator:
    """
    加载人类意图场(Human Intention Field)，用于引导agent的方向正则，HIF像素包描述方向值（弧度），表示该位置的期望行进方向，特殊值-1表示该位置无方向引导。
    """

    @classmethod
    def get_dependencies(cls) -> List[str]:
        return ['field']  # HIF需要在field创建后加载，以获取地图尺寸

    def generate(self, state: Dict[str, Any], rng: np.random.Generator) -> None:
        """
        加载HIF地图并验证尺寸匹配， 场景模式：从scenario_directory加载，普通模式：从{map_dir}/hif/human_intent_field_{map_id+1}.npy加载
        """
        dimensions = state['env_state'].get_static_info('dimensions')
        scenario_directory = state['options'].get('scenario_directory')

        # 根据模式选择加载方式
        state['maps_dict']['hif'] = self._load_from_directory(scenario_directory, dimensions) if scenario_directory else \
            self._load_from_file(state['env_state'].get_static_info('map_id'), state['config'], dimensions)

    def _load_from_directory(self, directory: Union[str, Path], dimensions: Tuple[int, int]) -> np.ndarray:
        """从场景目录加载HIF地图"""
        hif_file = Path(directory) / 'map_hif.npy'
        if not hif_file.exists(): raise FileNotFoundError(f"Scenario HIF file not found: {hif_file}\n" )

        hif_map = np.load(str(hif_file)).astype(np.float32)
        if hif_map.shape != dimensions: raise ValueError(f"Scenario HIF dimensions {hif_map.shape} don't match expected {dimensions}")
        return hif_map


    def _load_from_file(self, map_id: Optional[int], config, dimensions: Tuple[int, int]) -> np.ndarray:
        """从标准位置加载HIF地图， config.map_dir指向包含field/和hif/的父目录"""
        map_root = Path(config.get_absolute_map_dir())
        hif_file = map_root / 'hif' / f'human_intent_field_{map_id + 1}.npy' # TODO: 图片是从1开始编号的，因此差1，以后有时间去除

        if not hif_file.exists(): raise FileNotFoundError(f"HIF file required but not found: {hif_file}\n")

        hif_map = np.load(str(hif_file)).astype(np.float32)
        if hif_map.shape != dimensions:
            raise ValueError(f"HIF map dimensions {hif_map.shape} don't match expected {dimensions}")
        return hif_map


class HIFCalculator(RewardCalculator):
    """
    人类意图场(HIF)方向引导奖励, 根据agent当前朝向与HIF指定方向的差异计算奖励, 角度差异越小，奖励越高（实际是负惩罚越小）
    """

    @classmethod
    def calculate(cls, env_state: EnvironmentState, coefficient: float,
                  config: EnvironmentConfig = None, **kwargs) -> float:
        """
        计算HIF方向引导奖励, 方向差异惩罚（负值）
        """
        # 获取agent位置和方向信息
        agent_position, agent_direction = env_state.get_info('agent_position'), env_state.get_info('agent_direction')
        if len(agent_position) <= 1 or 'map_dict' not in kwargs:
            return 0.0

        hif_map = kwargs['map_dict']['hif']

        # 获取agent当前和上一步的网格坐标
        x, y = int(agent_position.current[0]), int(agent_position.current[1])
        x_last, y_last = int(agent_position.last[0]), int(agent_position.last[1])

        # 计算与HIF方向差异
        weight_current, weight_last = 0.3, 0.7
        angle_diff = weight_current * cls._compute_angle_difference(agent_direction.current, hif_map[y, x]) \
            if hif_map[y, x] >= 0 else 0
        angle_diff_last = weight_last * cls._compute_angle_difference(agent_direction.current,hif_map[y_last, x_last]) \
            if hif_map[y_last, x_last] >= 0 else 0

        return -coefficient * (angle_diff + angle_diff_last)  # 返回负奖励（惩罚）

    @staticmethod
    def _compute_angle_difference(agent_direction: float, hif_direction: float) -> float:
        """
            计算无向场中agent朝向与HIF方向的角度差异

            坐标系映射关系：
            - Agent系统：0°=3点钟(东), 90°=6点钟(南), 180°=9点钟(西), 270°=12点钟(北)
            - HIF系统：0rad=9点钟(西), π/2rad=6点钟(南), πrad=3点钟(东)
            - 转换原理：两系统相差180°相位

            Args:
                agent_direction: agent朝向，单位：度（Agent类保证[0,360)但此处防御性处理）
                hif_direction: HIF期望方向，单位：弧度，范围[0,π]

            Returns:
                角度差异，单位：度，范围[0,90]（无向场最大差异为90°）
        """
        agent_direction_normalized = agent_direction % 360  # Step 1: 防御性归一化 - 确保agent方向在[0, 360)范围
        hif_direction_degrees = np.degrees(hif_direction)  # Step 2: 单位转换 - HIF从弧度转为度
        hif_in_agent_coordinates = (hif_direction_degrees + 180) % 360  # Step 3: 坐标系对齐 - 将HIF坐标系映射到Agent坐标系（180°相位差） HIF 0(9点钟) → Agent 180°, HIF π(3点钟) → Agent 0°
        angular_difference = abs(agent_direction_normalized - hif_in_agent_coordinates)  # Step 4: 计算原始角度差
        if angular_difference > 180: angular_difference = 360 - angular_difference  # Step 5: 周期性优化 - 选择最短路径角度 例如：350°和10°的差异应该是20°而不是340°
        if angular_difference > 90: angular_difference = 180 - angular_difference  # Step 6: 无向场归一化 - 将[0,180]映射到[0,90] 在无向场中，相反方向(180°差)等价于同向(0°差)
        return angular_difference


class CppEnv(CppEnvV4):
    """
    v5环境 - Field覆盖 + HIF方向引导
    """
    def __init__(self, render_mode="rgb_array", **kwargs):
        v5_defaults = {'reward_hif': 0.01, 'map_dir': "envs_new/maps/field_coverage"}  # 默认HIF奖励权重
        final_kwargs = {**v5_defaults, **kwargs}

        super().__init__(render_mode=render_mode, **final_kwargs)

        # 添加HIF组件
        self.scenario_generator.add_component('hif', HIFCreator())
        self.reward_system.add_calculator('hif', HIFCalculator)

    def _get_observation_channels(self) -> int:
        """v5环境的观察通道数：field, obstacle, trajectory, hif"""
        # v5 在 v4 的基础上添加了 hif
        return 3 + int(self.config.use_trajectory)  # field, obstacle, hif, (trajectory)

    def _get_observation_maps(self) -> Dict[str, Dict[str, Any]]:
        obs_maps = super()._get_observation_maps() # 获取v4的基础观察地图
        obs_maps['hif'] = {'map': self.maps_dict['hif'], 'pad': -1.0}  # 添加HIF观察,-1表示无方向引导
        return obs_maps


if __name__ == "__main__":
    if_render = True
    episodes = 3

    print("=" * 60)
    print("Testing CppEnv v5 - Field Coverage + HIF Guidance")
    print("=" * 60)

    # 创建v5环境，展示HIF特性
    env = CppEnv(use_multiscale=True, use_global_features=True,
                 # render_first_person=True,  # 控制渲染第一人称视角
                 )
    # 默认map_dir现在指向field_coverage，自动寻找field/和hif/子目录

    if if_render: env = HumanRendering(env)

    for episode in range(episodes):
        print(f"\n--- Episode {episode + 1} ---")

        # 使用特定地图ID以加载HIF（如果有的话）
        # env.update_config({'num_obstacles_range':[0,0]})
        obs, info = env.reset(seed=120 + episode, options={
            # 'map_id': 0,  # 使用地图0（如果存在对应的HIF文件）
            # 'initial_position': None,
            # 'initial_direction': None,
        })

        print(f"Observation shape: {obs['observation'].shape}")
        print(f"Initial field coverage: {obs['completion_ratio'][0]:.2%}")

        # 检查是否成功加载HIF（通过observation判断）
        if 'observation' in obs and obs['observation'].shape[0] > 3:
            print(f"HIF observation channel detected (total channels: {obs['observation'].shape[0]})")

        env.action_space.seed(66)
        done = False
        step_count = 0
        total_reward = 0
        hif_rewards = []

        while not done:  # 限制步数用于测试
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)

            step_count += 1
            total_reward += reward

            # 尝试分解奖励以查看HIF贡献
            reward_breakdown = env.reward_system.get_reward_breakdown(
                env.env_state, map_dict=env.maps_dict)
            if 'hif' in reward_breakdown['breakdown']:
                hif_rewards.append(reward_breakdown['breakdown']['hif'])

            if step_count % 20 == 0:
                print(f"  Step {step_count}: reward={reward:.4f}, "
                      f"coverage={obs['completion_ratio'][0]:.2%}")
                if hif_rewards:
                    print(f"    HIF reward contribution: {hif_rewards[-1]:.4f}")

            if if_render:
                env.render()

            if done or truncated:
                print(f"\nEpisode finished after {step_count} steps!")
                print(f"  Total reward: {total_reward:.2f}")
                print(f"  Final coverage: {obs['completion_ratio'][0]:.2%}")
                if hif_rewards:
                    avg_hif = np.mean(hif_rewards)
                    print(f"  Average HIF reward: {avg_hif:.4f}")
                if info.get('crashed'):
                    print("  Termination: Crashed")
                elif info.get('finished'):
                    print("  Termination: Field fully covered!")
                elif info.get('timeout'):
                    print("  Termination: Timeout")
                break

    env.close()
    print("\n✅ v5 Environment test completed successfully!")
