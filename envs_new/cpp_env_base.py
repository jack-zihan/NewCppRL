"""
模块化的割草/覆盖机器人环境基础类。

采用组件化架构设计，提供更好的可维护性、可扩展性和可测试性。
所有环境变体（v1, v2, v3, v4）都继承此基类。
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

from envs_new.components.config.environment_config import EnvironmentConfig, EnvironmentConfigOriginal
from envs_new.components.map.map_generator import ScenarioGenerator
from envs_new.components.observation.observation_generator import ObservationGenerator
from envs_new.components.dynamics.environment_dynamics import EnvironmentDynamics
from envs_new.components.dynamics.action_processor import ActionProcessor
from envs_new.components.reward.reward_system import RewardSystem
from envs_new.components.render.renderer import Renderer

class CppEnvBase(gym.Env):
    """
    模块化割草/覆盖机器人环境，未来可能拓展到Isaac Agile上去。
    
    核心特性：
    - 可配置的组件系统：场景生成、动力学、观察、奖励、渲染
    - 清晰的关注点分离：每个组件独立负责特定功能
    
    通用参数约定：
    - maps_dict: 地图数据字典，包含各种地图层（obstacle, weed, frontier等）
    - env_state: 环境状态对象，管理所有状态变量和历史
    - agent: 机器人实体，包含位置、方向、凸包等属性
    """

    metadata = {
        "render_modes": ["rgb_array", "first_person"],  # 渲染正常图片或者第一人称观测
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None, **kwargs):
        super().__init__()

        # 极简配置创建：直接使用kwargs
        # self.config = EnvironmentConfig(**kwargs)
        self.config = EnvironmentConfigOriginal(**kwargs) # 使用原版v2观测

        self._initialize_components()
        self._initialize_spaces()

        # Environment state
        self.agent = None
        self.maps_dict = {}
        self.env_state = None  # Will be created by scenario generator

        # Rendering
        self.is_open = True
        self.render_mode = render_mode  # 目前只支持 rgb_array, human由warpper实现

    def _initialize_components(self) -> None:
        """Initialize all environment components."""
        # 所有组件使用同一配置对象
        self.scenario_generator = ScenarioGenerator(self.config)

        # Action processing, 动作解析解耦于动力学之外增加灵活性
        self.action_processor = ActionProcessor(self.config)

        # Environment dynamics。
        self.env_dynamics = EnvironmentDynamics(self.config, self.action_processor)

        # Observation generation
        self.observation_generator = ObservationGenerator(self.config)

        # Reward calculation
        self.reward_system = RewardSystem(self.config)

        # Rendering
        self.renderer = Renderer(self.config)

    def _get_observation_channels(self) -> int:
        """获取预期观察通道数（int），子类可重写此方法声明自己的通道数"""
        return 3 + int(self.config.use_trajectory) # 基础环境：field, obstacle, weed, (trajectory)

    def _get_expected_vector_length(self) -> int:
        """预期的observation vector维度，包含当前时间t和完成度c, 以及当前坐标系下的轨迹信息（x, y , steer, speed, der_cos, dre_sin）, 为了不影响除草，设置是否开启的开关，覆盖问题才开启"""
        return 2 + 6 * self.config.state_history_length if self.config.use_history_vector else 4

    def _initialize_spaces(self) -> None:
        """Initialize action and observation spaces."""
        if self.config.action_type == "discrete":
            self.action_space = gym.spaces.Discrete(self.action_processor.get_action_space_size())
        elif self.config.action_type == "multi_discrete":
            self.action_space = gym.spaces.MultiDiscrete(self.config.action_nvec)
        elif self.config.action_type == "continuous":
            bounds = self.action_processor.get_action_bounds()
            self.action_space = gym.spaces.Box(
                low=np.array([bounds[0][0], bounds[1][0]], dtype=np.float32),
                high=np.array([bounds[0][1], bounds[1][1]], dtype=np.float32),
                shape=(2,), dtype=np.float32)
        else:
            raise ValueError(f"Unsupported action type: {self.config.action_type}")

        # 使用声明的通道数而非估算
        actual_channels = self._get_observation_channels()
        obs_shape = self.observation_generator.get_observation_shape(actual_channels)  # 计算观察形状

        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(low=0.0, high=1.0, shape=obs_shape, dtype=np.float32),
            "vector": gym.spaces.Box(low=-1.0, high=1.0, shape=(self._get_expected_vector_length(),), dtype=np.float32),
            "completion_ratio": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        })  # 这里还是叫completion_ratio, 这样在未来除了覆盖任务保持统一性

    def _count_observation_channels(self, obs_maps: Dict[str, Dict[str, Any]]) -> int:
        """遍历obs_maps，累加每个条目的通道贡献：有 'num_channels' 字段 → 使用该值（复合负载，如HIF的2-3通道）无 'num_channels' 字段 → 默认为1（单通道地图）"""
        return sum(entry.get('num_channels', 1) for entry in obs_maps.values())

    def _update_observation_space(self) -> None:
        """基于实际地图更新观察空间（精确通道计数）"""
        # 获取实际的观察地图
        obs_maps = self._get_observation_maps()
        actual_channels = self._count_observation_channels(obs_maps)

        # 获取正确的观察形状
        obs_shape = self.observation_generator.get_observation_shape(actual_channels)

        # 更新观察空间
        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(low=0.0, high=1.0, shape=obs_shape, dtype=np.float32),
            "vector": gym.spaces.Box(low=-1.0, high=1.0, shape=(len(self._get_observation_vector()),), dtype=np.float32), # 基于当前场景的实际向量长度更新vector形状
            "completion_ratio": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        })

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[
        Dict[str, np.ndarray], Dict[str, Any]]:
        """重置环境开始新回合。"""
        super().reset(seed=seed)

        self.scenario_generator.set_random_generator(self.np_random)
        self.observation_generator.set_random_generator(self.np_random)

        # Generate complete scenario environemnt state (agent, maps, env_state), options参数包含特定的场景生成选项
        options = options or {}
        self.agent, self.maps_dict, self.env_state = self.scenario_generator.generate_scenario(
            map_id=options.get('map_id'), scenario_directory=options.get('specific_scenario_dir'),
            weed_count=options.get('weed_count', self.config.weed_count), weed_distribution=options.get('weed_distribution', 'uniform'),
            initial_position=options.get('initial_position'), initial_direction=options.get('initial_direction')
        )

        self.env_dynamics.reset(self.agent, self.maps_dict, self.env_state)

        # update observation components
        self._update_observation_space()
        observation = self._generate_observation()

        return observation, {}

    def step(self, action: Union[int, Tuple]) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        """
        # Apply dynamics， 传入action_type使得action_processor是无状态的函数式模式
        self.agent, self.maps_dict, self.env_state = self.env_dynamics.step(
            self.agent, self.maps_dict, self.env_state, action, self.config.action_type
        )

        # Generate observation
        observation = self._generate_observation()

        # Calculate reward
        reward = self.reward_system.calculate_reward(self.env_state, map_dict=self.maps_dict)

        # Check episode termination
        terminated = self.env_state.crashed or self.env_state.finished
        truncated = self.env_state.timeout

        # Generate info using extractable method
        info = self._get_step_info()

        return observation, float(reward), bool(terminated), bool(truncated), info

    def _get_step_info(self) -> Dict[str, Any]:
        """
        Generate info dictionary for step return.
        Subclasses can override this to return different information. 需要封装未np.array, 方便并行环境的GymGrapper封装并行化
        """
        return {
            'crashed': np.array(self.env_state.crashed, dtype=np.bool_),
            'finished': np.array(self.env_state.finished, dtype=np.bool_),
            'timeout': np.array(self.env_state.timeout, dtype=np.bool_),
            'weed_count': np.array(self.env_state.weed_count, dtype=np.float32),
            'weed_ratio': np.array(self.env_state.weed_coverage_ratio, dtype=np.float32)
        }

    def _generate_observation(self) -> Dict[str, np.ndarray]:
        """生成观察"""
        # 获取环境特定的观察地图
        obs_maps = self._get_observation_maps()

        # 使用统一的观察生成器处理
        visual_obs = self.observation_generator.generate_observation(self.agent, obs_maps)

        return {
            'observation': visual_obs,
            'vector': self._get_observation_vector(),
            'completion_ratio': self._get_completion_ratio()
        }

    def _get_observation_maps(self) -> Dict[str, Dict[str, Any]]:
        """
        获取观察所需的地图。子类应该重写此方法以提供特定的地图集合。
        
        Returns:
            地图字典，键为地图名称，值为包含'map'和'pad'的字典
        """
        # 默认实现：返回基础地图，注意obstacle的pad值为1.0（表示边界为障碍物）
        obs_maps = {
            'field': {'map': self.maps_dict['field'], 'pad': 0.0},
            'obstacle': {'map': self.maps_dict['obstacle'], 'pad': 1.0},  # 边界视为障碍物
            'weed': {'map': self.maps_dict['weed'], 'pad': 0.0}
        }

        # 可选轨迹地图
        if self.config.use_trajectory and 'trajectory' in self.maps_dict:
            obs_maps['trajectory'] = {'map': self.maps_dict['trajectory'], 'pad': 0.0}

        return obs_maps

    def _get_observation_vector(self) -> np.ndarray:
        """归一化转向值, 四维向量：[speed, steer, coverage_ratio, time_progress]"""
        return np.array([self.agent.last_speed / self.config.v_max, self.agent.last_steer / self.config.w_max,
                         self.env_state.weed_coverage_ratio,
                         self.env_state.current_step / self.env_state.max_steps],dtype=np.float32)

    def _get_completion_ratio(self) -> np.ndarray:
        """默认获取杂草完成率"""
        return np.array([self.env_state.weed_coverage_ratio], dtype=np.float32)

    def render(self) -> Optional[np.ndarray]:
        """
        Render environment.
        
        Returns:
            Rendered image if render_mode is set, None otherwise
        """
        if self.render_mode is None:
            return None

        if self.render_mode not in self.metadata["render_modes"]:
            gym.logger.warn(f"Render mode {self.render_mode} not supported")
            return None

        # Determine render mode
        render_mode = "first_person" if self.config.render_first_person else "map"

        return self.renderer.render(
            self.maps_dict, self.agent, self.env_state.dimensions,
            mode=render_mode,
            observation_size=self.config.state_size if render_mode == "first_person" else None
        )

    def close(self) -> None:
        """Close environment and cleanup resources."""
        self.is_open = False

    def get_reward_breakdown(self) -> Dict[str, float]:
        """Get detailed reward breakdown for analysis."""
        return self.reward_system.get_reward_breakdown(self.env_state)

    def get_state_info(self) -> Any:
        """Get current environment state object."""
        return self.env_state

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update environment configuration dynamically."""
        # 动态更新config对象的属性, 由于所有组件都持有config的引用，更新会自动传播
        for key, value in new_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # 对于需要特殊处理的配置，保留兼容性
        if 'reward_coefficients' in new_config:
            self.reward_system.update_coefficients(new_config['reward_coefficients'])

    def set_action_type(self, action_type: str) -> None:
        """Change action type and reinitialize action space."""
        if action_type not in ["discrete", "continuous", "multi_discrete"]:
            raise ValueError(f"Unsupported action type: {action_type}")

        self.config.action_type = action_type
        self._initialize_spaces()

    def get_collision_info(self) -> Dict[str, bool]:
        """Get detailed collision information."""
        return self.env_dynamics.get_collision_info(self.agent, self.maps_dict)


if __name__ == "__main__":
    # Test the new environment
    env = CppEnvBase(render_mode='rgb_array')

    obs, info = env.reset()
    print(f"Observation shape: {obs['observation'].shape}")
    print(f"Vector shape: {obs['vector'].shape}")
    print(f"Weed ratio: {obs['completion_ratio']}")

    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step reward: {reward:.3f}, Done: {terminated or truncated}")

        if terminated or truncated:
            break

    env.close()
    print("Environment test completed successfully!")
