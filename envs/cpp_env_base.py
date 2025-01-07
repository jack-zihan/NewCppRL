# main_env.py

from __future__ import annotations

from typing import Optional, Tuple, Union, Dict, Any

import cv2
import gymnasium as gym
import numpy as np
import math
import pygame
from pygame import gfxdraw

import logging

from envs.components.utils import MowerAgent, NumericalRange, total_variation_mat
from envs.components.map_manager import MapManager
from envs.components.reward_manager import RewardManager
from envs.components.observation_manager import ObservationManager


class CppEnvBase(gym.Env):
    """
    主环境类，管理环境交互、渲染和状态更新。
    """
    metadata = {
        "render_modes": ["rgb_array", "state_pixels"],
        "render_fps": 50,
    }

    def __init__(
        self,
        action_type: str = "discrete",
        render_mode: Optional[str] = None,
        # 观测管理
        state_pixels: bool = False,
        state_size: Tuple[int, int] = (128, 128),
        state_downsize: Tuple[int, int] = (128, 128),
        # 随机因素
        position_noise: float = 0.0,
        direction_noise: float = 0.0,
        weed_noise: float = 0.0,
        # 地图相关
        map_dir: str = "envs/maps/1-400",
        use_box_boundary: bool = True,
        num_obstacles_range: Tuple[int, int] = (5, 8),
        obstacle_size_range: Tuple[int, int] = (10, 25),
        # 其余
        use_multiscale: bool = True,
        use_global_features: bool = True,
        global_feature_size: int = 16,
        render_repeat_times: int = 2,
    ):
        """
        初始化主环境。

        :param action_type: 动作类型，'discrete', 'continuous', 或 'multi_discrete'。
        :param render_mode: 渲染模式，'rgb_array' 或 'state_pixels'。
        :param state_pixels: 是否使用第一人称视角渲染。
        :param state_size: 观测图像的尺寸 (宽, 高)。
        :param state_downsize: 缩小后的观测图像尺寸 (宽, 高)。
        :param position_noise: 位置噪声强度。
        :param direction_noise: 方向噪声强度。
        :param weed_noise: 杂草噪声强度。
        :param map_dir: 地图文件夹路径。
        :param use_box_boundary: 是否使用边界框作为障碍物。
        :param num_obstacles_range: 障碍物数量范围。
        :param obstacle_size_range: 障碍物尺寸范围。
        :param use_multiscale: 是否使用多尺度特征。
        :param use_global_features: 是否使用全局特征。
        :param global_feature_size: 全局特征的尺寸。
        :param render_repeat_times: 渲染时图像重复倍数。
        """
        super().__init__()
        self.action_type = action_type
        self.render_mode = render_mode
        self.state_pixels = state_pixels
        self.render_repeat_times = render_repeat_times

        # RNG
        self.rng = np.random.default_rng()

        # 数值区间
        speed_range = NumericalRange(0.0, 3.5)
        angular_range = NumericalRange(-28.6, 28.6)

        # 地图管理
        self.map_manager = MapManager(
            map_dir=map_dir,
            num_obstacles_range=num_obstacles_range,
            obstacle_size_range=obstacle_size_range,
            use_box_boundary=use_box_boundary,
            weed_noise=weed_noise,
            rng=self.rng
        )

        # 观测管理
        self.obs_manager = ObservationManager(
            state_size=state_size,
            state_downsize=state_downsize,
            vision_length=28,
            vision_angle=75.0,
            use_multiscale=use_multiscale,
            use_global_features=use_global_features,
            global_feature_size=global_feature_size,
            position_noise=position_noise,
            direction_noise=direction_noise,
            rng=self.rng
        )

        # 奖励管理
        self.reward_manager = RewardManager(
            speed_range=speed_range,
            angular_range=angular_range
        )

        # 初始化 action_space 和 observation_space
        self._initialize_action_space()
        self._initialize_observation_space()

        # 机器人对象
        self.agent = MowerAgent()

        # 其他地图或渲染用数据
        self.trajectory_map: np.ndarray = np.zeros((1, 1), dtype=np.uint8)
        self.mist_map: np.ndarray = np.zeros((1, 1), dtype=np.uint8)

        # 渲染
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.is_open: bool = True

        # 时间步
        self.current_step: int = 0
        # 记录上一步农田/杂草/转向等信息，用于奖励
        self.previous_weed_count: int = 0
        self.previous_frontier_area: int = 0
        self.previous_frontier_tv: float = 0.0
        self.previous_steer: float = 0.0

    def _initialize_action_space(self):
        """
        初始化动作空间。
        """
        if self.action_type == 'discrete':
            # 假设动作空间基于障碍物数量范围进行离散化
            self.action_space = gym.spaces.Discrete(self.map_manager.num_obstacles_range[1] * self.map_manager.num_obstacles_range[1])
        elif self.action_type == 'continuous':
            self.action_space = gym.spaces.Box(
                low=np.array([self.reward_manager.speed_range.min, self.reward_manager.angular_range.min], dtype=np.float32),
                high=np.array([self.reward_manager.speed_range.max, self.reward_manager.angular_range.max], dtype=np.float32),
                shape=(2,),
                dtype=np.float32,
            )
        elif self.action_type == 'multi_discrete':
            self.action_space = gym.spaces.MultiDiscrete(nvec=self.map_manager.num_obstacles_range)
        else:
            raise NotImplementedError(f"不支持的动作类型: {self.action_type}")



    def _initialize_observation_space(self):
        """
        初始化观测空间。
        """
        if self.obs_manager.use_multiscale:
            sgcnn_channels = 5 + 4  # 原地图通道 + 多尺度 + 全局特征
        else:
            sgcnn_channels = 5  # 原地图通道

        obs_shape = (sgcnn_channels, self.obs_manager.state_downsize[0], self.obs_manager.state_downsize[1])
        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(low=0., high=1., shape=obs_shape, dtype=np.float32),
            "vector": gym.spaces.Box(low=-1., high=1., shape=(1,), dtype=np.float32),
            "weed_ratio": gym.spaces.Box(low=0., high=1., shape=(1,), dtype=np.float32)
        })

    def step(
        self,
        action: Union[int, Tuple[int, int], Tuple[float, float]]
    ) -> Tuple[Dict[str, Union[np.ndarray, float]], float, bool, bool, Dict[str, Any]]:
        """
        执行动作并返回观测、奖励、是否结束、是否超时及额外信息。

        :param action: 执行动作。
        :return: 观测、奖励、是否结束、是否超时、额外信息。
        """
        # 解析动作
        speed, steer = self._parse_action(action)



        # 记录当前状态以便奖励计算
        previous_position = self.agent.position_discrete
        previous_steer = self.agent.last_steer

        # 执行控制
        self.agent.control(speed, steer)

        # 割除杂草
        self._cut_weeds()

        # 更新农田前沿区域
        self._update_field_frontier()

        # 更新迷雾区域
        self._update_mist()

        # 检测碰撞
        crashed = self._check_collision()

        # 绘制轨迹
        self._update_trajectory(previous_position)

        # 计算奖励
        reward = self._calculate_reward(steer, previous_steer)

        # 处理碰撞惩罚
        if crashed:
            reward += self.reward_manager.coefficients['collision_penalty']

        # 增加步数
        self.current_step += 1
        time_out = (self.current_step >= 3000)

        # 检查是否完成（所有杂草已割）
        finish = (self.previous_weed_count <= 0)
        if finish:
            reward += self.reward_manager.coefficients['completion_bonus']

        done = crashed or finish

        # 获取观测
        observation = self._get_observation()

        return observation, float(reward), done, time_out, {}

    def _parse_action(self, action: Union[int, Tuple[int, int], Tuple[float, float]]) -> Tuple[float, float]:
        """
        根据 action_type 解析动作为 (speed, steer)。

        :param action: 执行动作。
        :return: (速度, 角速度)。
        """
        if self.action_type == 'discrete':
            acc = action // self.map_manager.num_obstacles_range[1]
            speed = self.reward_manager.speed_range.min + (acc + 1) / self.map_manager.num_obstacles_range[1] * self.reward_manager.speed_range.max
            steer = action % self.map_manager.num_obstacles_range[1]
            steer = self.reward_manager.angular_range.min + steer / (self.map_manager.num_obstacles_range[1] - 1) * self.reward_manager.angular_range.max
        elif self.action_type == 'continuous':
            speed, steer = action
        elif self.action_type == 'multi_discrete':
            acc, steer = action
            speed = self.reward_manager.speed_range.min + (acc + 1) / self.map_manager.num_obstacles_range[1] * self.reward_manager.speed_range.max
            steer = self.reward_manager.angular_range.min + steer / (self.map_manager.num_obstacles_range[1] - 1) * self.reward_manager.angular_range.max
        else:
            raise NotImplementedError(f"不支持的动作类型: {self.action_type}")
        return speed, steer

    def _cut_weeds(self):
        """
        割除机器人当前所在区域的杂草。
        """
        convex_hull = self.agent.convex_hull.round().astype(np.int32)
        map_cutter = np.zeros_like(self.map_manager.weed_map)
        cv2.fillPoly(map_cutter, [convex_hull], color=1)
        weeds_cut = np.sum(np.logical_and(map_cutter, self.map_manager.weed_map))
        self.map_manager.weed_map = np.where(map_cutter, 0, self.map_manager.weed_map)

    def _update_field_frontier(self):
        """
        更新农田前沿区域。
        """
        self.map_manager.field_frontier_map = cv2.ellipse(
            self.map_manager.field_frontier_map,
            center=self.agent.position_discrete,
            axes=(self.obs_manager.vision_length, self.obs_manager.vision_length),
            angle=self.agent.direction,
            startAngle=-self.obs_manager.vision_angle / 2,
            endAngle=self.obs_manager.vision_angle / 2,
            color=0,
            thickness=-1
        )

    def _update_mist(self):
        """
        更新迷雾区域。
        """
        self.mist_map = cv2.ellipse(
            self.mist_map,
            center=self.agent.position_discrete,
            axes=(self.obs_manager.vision_length + 1, self.obs_manager.vision_length + 1),
            angle=self.agent.direction,
            startAngle=-self.obs_manager.vision_angle / 2,
            endAngle=self.obs_manager.vision_angle / 2,
            color=1,
            thickness=-1
        )

        # 如果视野内没有任何农田，则自动扩大可见区域
        if not np.any(np.logical_and(self.map_manager.field_frontier_map, self.mist_map)):
            distance = cv2.pointPolygonTest(self.map_manager.initial_bounding_box[0], self.agent.position_discrete, True)
            radius = int(math.ceil(abs(distance) + 5))
            self.mist_map = cv2.circle(
                self.mist_map,
                self.agent.position_discrete,
                radius=radius,
                color=1,
                thickness=-1
            )

    def _check_collision(self) -> bool:
        """
        检测是否超出边界或与障碍物碰撞。

        :return: 是否发生碰撞。
        """
        convex_hull = self.agent.convex_hull
        dims = self.map_manager.dimensions

        # 生成机器人地图
        agent_map = np.zeros((dims[1], dims[0]), dtype=np.uint8)
        cv2.fillPoly(agent_map, [convex_hull.round().astype(np.int32)], color=1)

        # 边界碰撞
        out_of_bounds = not (
            (convex_hull[:, 0] > 0).all() and
            (convex_hull[:, 0] < dims[0]).all() and
            (convex_hull[:, 1] > 0).all() and
            (convex_hull[:, 1] < dims[1]).all()
        )

        # 障碍物碰撞
        crashed_obstacles = np.any(np.logical_and(agent_map, self.map_manager.obstacle_map))

        # 位置修正
        self.agent.x = float(np.clip(self.agent.x, 0, dims[0]))
        self.agent.y = float(np.clip(self.agent.y, 0, dims[1]))

        return out_of_bounds or crashed_obstacles

    def _update_trajectory(self, previous_position: Tuple[int, int]):
        """
        更新机器人轨迹。

        :param previous_position: 机器人上一步的位置。
        """
        current_position = self.agent.position_discrete
        x_prev, y_prev = previous_position
        x_curr, y_curr = current_position

        # 确保坐标在有效范围内
        x_prev = np.clip(x_prev, 0, self.map_manager.dimensions[0] - 1)
        y_prev = np.clip(y_prev, 0, self.map_manager.dimensions[1] - 1)
        x_curr = np.clip(x_curr, 0, self.map_manager.dimensions[0] - 1)
        y_curr = np.clip(y_curr, 0, self.map_manager.dimensions[1] - 1)

        cv2.line(self.trajectory_map, (x_prev, y_prev), (x_curr, y_curr), color=1, thickness=1)

    def _calculate_reward(self, current_steer: float, previous_steer: float) -> float:
        """
        计算本步奖励。

        :param current_steer: 当前转向角速度。
        :param previous_steer: 上一步转向角速度。
        :return: 本步奖励。
        """
        current_weed_count = self.map_manager.weed_map.sum(dtype=np.int32)
        current_frontier_area = self.map_manager.field_frontier_map.sum(dtype=np.int32)
        current_frontier_tv = total_variation_mat(self.map_manager.field_frontier_map).sum()

        step_reward = self.reward_manager.calculate_step_reward(
            current_steer=current_steer,
            previous_steer=previous_steer,
            current_frontier_area=current_frontier_area,
            previous_frontier_area=self.previous_frontier_area,
            current_frontier_tv=current_frontier_tv,
            previous_frontier_tv=self.previous_frontier_tv,
            current_weeds=current_weed_count,
            previous_weeds=self.previous_weed_count
        )

        # 更新奖励相关状态
        self.previous_weed_count = current_weed_count
        self.previous_frontier_area = current_frontier_area
        self.previous_frontier_tv = current_frontier_tv
        self.previous_steer = current_steer

        return step_reward

    def _get_observation(self) -> Dict[str, Union[np.ndarray, float]]:
        """
        生成观测，包括图像观测、向量信息和杂草覆盖率。

        :return: 观测字典。
        """
        obs_image = self.obs_manager.generate_observation(
            agent=self.agent,
            field_frontier_map=self.map_manager.field_frontier_map,
            mist_map=self.mist_map,
            weed_map=self.map_manager.weed_map,
            obstacle_map=self.map_manager.obstacle_map,
            trajectory_map=self.trajectory_map
        )

        # 向量信息：归一化的转向
        vector_info = self.agent.last_steer / self.reward_manager.angular_range.mode

        # 杂草覆盖率
        weed_ratio = 1.0 - (self.previous_weed_count / max(1, self.map_manager.total_weed_count))


        return {
            "observation": obs_image,
            "vector": float(vector_info),
            "weed_ratio": float(weed_ratio)
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[Dict[str, Union[np.ndarray, float]], Dict[str, Any]]:
        """
        重置环境，返回初始观测和额外信息。

        :param seed: 随机种子。
        :param options: 重置选项。
        :return: 初始观测和额外信息。
        """
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        opts = options or {}
        weed_distribution = opts.get("weed_distribution", "uniform")
        weed_count = opts.get("weed_count", 100)
        map_id = opts.get("map_id", self.rng.integers(0, len(self.map_manager.map_names)))
        specific_scenario_dir = opts.get("specific_scenario_dir", None)
        initial_position = opts.get("initial_position", None)
        initial_direction = opts.get("initial_direction", None)

        # 加载或随机生成地图
        if specific_scenario_dir:
            self.map_manager.load_maps_from_directory(specific_scenario_dir)
            initial_pos, initial_dir = self.map_manager.find_initial_bounding_box()
        else:
            self.map_manager.load_field_frontier_map(map_id)
            initial_pos, initial_dir = self.map_manager.find_initial_bounding_box()
            self.map_manager.initialize_obstacle_map(agent_position=initial_pos)
            self.map_manager.initialize_weed_distribution(weed_distribution, weed_count)

        # 初始化机器人
        self.agent.reset(
            position=initial_position if initial_position else initial_pos,
            direction=initial_direction if initial_direction else initial_dir
        )

        # 初始化额外地图
        dims = self.map_manager.dimensions
        self.trajectory_map = np.zeros((dims[1], dims[0]), dtype=np.uint8)
        self.mist_map = np.zeros((dims[1], dims[0]), dtype=np.uint8)

        # 更新农田前沿和迷雾区域
        self._update_maps_after_reset()

        # 初始化奖励相关状态
        self.previous_weed_count = self.map_manager.weed_map.sum(dtype=np.int32)
        self.previous_frontier_area = self.map_manager.field_frontier_map.sum(dtype=np.int32)
        self.previous_frontier_tv = total_variation_mat(self.map_manager.field_frontier_map).sum()
        self.current_step = 1
        self.previous_steer = 0.0

        # 获取初始观测
        observation = self._get_observation()

        return observation, {}

    def _update_maps_after_reset(self):
        """
        根据机器人初始位置更新农田前沿和迷雾。
        """
        # 割除初始位置的杂草
        self._cut_weeds()

        # 更新农田前沿区域
        self._update_field_frontier()

        # 更新迷雾区域
        self._update_mist()

    def render(self) -> Optional[np.ndarray]:
        """
        渲染环境，返回 RGB 图像数组。

        :return: 渲染后的图像数组，或 None 如果未指定渲染模式。
        """
        if self.render_mode not in self.metadata["render_modes"]:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return None

        try:
            import pygame
        except ImportError as e:
            raise gym.error.DependencyNotInstalled(
                "pygame 未安装，请运行 `pip install pygame`。"
            ) from e

        if self.screen is None:
            pygame.init()
            if self.state_pixels:
                width = self.obs_manager.state_size[0] * self.render_repeat_times
                height = self.obs_manager.state_size[1] * self.render_repeat_times
                self.screen = pygame.Surface((width, height))
            else:
                map_width, map_height = self.map_manager.dimensions
                self.screen = pygame.Surface((map_width * self.render_repeat_times, map_height * self.render_repeat_times))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.state_pixels:
            img = self._render_first_person_view()
        else:
            img = self._render_full_map()

        # 按比例缩放图像
        img = img.repeat(self.render_repeat_times, axis=0).repeat(self.render_repeat_times, axis=1)

        # 转换为 Pygame Surface 并绘制
        surf = pygame.surfarray.make_surface(img)
        self.screen.blit(surf, (0, 0))

        # 返回图像数组
        rendered_image = np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), (1, 0, 2))


        return rendered_image

    def _render_full_map(self) -> np.ndarray:
        """
        渲染完整地图，返回 RGB 图像数组。

        :return: 渲染后的完整地图图像。
        """
        map_width, map_height = self.map_manager.dimensions
        rendered_map = np.ones((map_height, map_width, 3), dtype=np.uint8) * 255  # 白色背景

        # 农田前沿
        frontier_map = (self.map_manager.field_frontier_map > 0)
        rendered_map[frontier_map] = [76, 187, 23]  # 绿色

        # 障碍物
        obstacle_map = (self.map_manager.obstacle_map > 0)
        rendered_map[obstacle_map] = [30, 75, 130]  # 蓝色

        # 杂草
        weed_map = (self.map_manager.weed_map > 0)
        rendered_map[weed_map] = [255, 0, 0]  # 红色

        # 轨迹
        trajectory_map = (self.trajectory_map > 0)
        rendered_map[trajectory_map] = [255, 38, 255]  # 紫色

        # 绘制机器人
        convex_hull = self.agent.convex_hull.round().astype(np.int32)
        cv2.fillPoly(rendered_map, [convex_hull], color=(0, 0, 255))  # 蓝色表示机器人

        # 添加障碍物边缘高亮
        obstacle_edges = total_variation_mat(self.map_manager.obstacle_map)
        rendered_map[obstacle_edges] = [47, 82, 143]  # 深蓝色


        return rendered_map

    def _render_first_person_view(self) -> np.ndarray:
        """
        渲染第一人称视角的图像，返回 RGB 图像数组。

        :return: 渲染后的第一人称视角图像。
        """
        # 获取旋转裁剪后的观测图像
        obs_dict = self._get_observation()
        img_tensor = obs_dict["observation"]  # (C, H, W)

        # 转换为 (H, W, C)
        img = img_tensor.transpose(1, 2, 0)

        # 确保图像有 3 个通道
        if img.shape[2] < 3:
            img = np.repeat(img, 3, axis=2)
        elif img.shape[2] > 3:
            img = img[:, :, :3]

        # 转换为 uint8
        img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)


        return img_uint8

    def close(self):
        """
        关闭环境，释放资源。
        """
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.is_open = False
