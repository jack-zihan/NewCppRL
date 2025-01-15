from __future__ import annotations

import cv2
import numpy as np
import gymnasium as gym

from gymnasium.wrappers import HumanRendering
from typing import Optional, Tuple, Union, Dict, Any

try:
    import pygame
    from pygame import gfxdraw
except ImportError:
    pygame = None

from envs.components.map_manager import MapManager
from envs.components.reward_manager import RewardManager
from envs.components.observation_manager import ObservationManager
from envs.components.dynamic import SceneDynamic
from envs.components.utils import MowerAgent, NumericalRange, total_variation_mat, get_map_pasture_larger, \
    apply_mask_with_color


class CppEnvBase(gym.Env):
    """
    主环境类，管理环境交互、渲染和状态更新，使用新的 MapManager / RewardManager / ObservationManager。
    """
    metadata = {
        "render_modes": ["rgb_array", "state_pixels"],
        "render_fps": 50,
    }

    # 对速度与角速度的离散化: nvec=(7,21)
    v_range = NumericalRange(0.0, 3.5)
    w_range = NumericalRange(-28.6, 28.6)
    nvec = (7, 21)

    # 其它用于渲染的静态属性
    render_repeat_times = 2
    render_tv = False
    render_mist = False
    render_covered_weed = True
    render_covered_farmland = True

    def __init__(
            self,
            action_type: str = "discrete",
            render_mode: Optional[str] = None,
            state_pixels: bool = False,  # 是否渲染第一视角
            state_size: Tuple[int, int] = (128, 128), state_downsize: Tuple[int, int] = (128, 128),
            max_episode_steps: int = 3000,
            # 地图/障碍物/杂草参数
            map_dir: str = "envs/maps/1-400", use_box_boundary: bool = True,
            num_obstacles_range: Tuple[int, int] = (5, 8), obstacle_size_range: Tuple[int, int] = (10, 25),

            # Observation Manager
            use_traj: bool = True,
            use_mist: bool = True,
            use_global_features: bool = True,
            use_multiscale: bool = True, multiscale_feature_size: int = 16, n_scales: int = 4,

            position_noise: float = 0.0, direction_noise: float = 0.0, weed_noise: float = 0.0,

    ):
        super().__init__()

        self.action_type = action_type
        self.render_mode = render_mode
        self.state_pixels = state_pixels
        self.state_size = state_size
        self.state_downsize = state_downsize
        self.max_episode_steps = max_episode_steps

        self.current_step: int = 0

        self.maps_dict: Dict[str, np.ndarray] = {}
        self.pad_values: Dict[str, float] = {}
        self.state_info: Dict[str, Any] = {}

        # 用于渲染
        self.screen = None
        self.clock = None
        self.is_open = True

        self.map_manager = MapManager(
            map_dir=map_dir,
            num_obstacles_range=num_obstacles_range,
            obstacle_size_range=obstacle_size_range,
            use_box_boundary=use_box_boundary,
            weed_noise=weed_noise,
            use_traj=use_traj,
            use_mist=use_mist,
        )

        self.reward_manager = RewardManager(
            speed_range=self.v_range,
            angular_range=self.w_range
        )

        self.observation_manager = ObservationManager(
            state_size=state_size,
            state_downsize=state_downsize,
            use_multiscale=use_multiscale,
            multiscale_feature_size=multiscale_feature_size,
            use_global_features=use_global_features,
            position_noise=position_noise,
            direction_noise=direction_noise,
            n_scales=n_scales
        )

        self.scene_dynamic = SceneDynamic()
        self.agent: MowerAgent = MowerAgent()

        if self.action_type == 'discrete':
            self.action_space = gym.spaces.Discrete(self.nvec[0] * self.nvec[1])  # 7*21=147
        elif self.action_type == 'multi_discrete':
            self.action_space = gym.spaces.MultiDiscrete(self.nvec)
        elif self.action_type == 'continuous':
            self.action_space = gym.spaces.Box(
                low=np.array([self.v_range.min, self.w_range.min], dtype=np.float32),
                high=np.array([self.v_range.max, self.w_range.max], dtype=np.float32),
                shape=(2,),
                dtype=np.float32
            )
        else:
            raise NotImplementedError(
                f"不支持动作类型: {action_type}, 仅支持['discrete','continuous','multi_discrete']")
        map_shape = (multiscale_feature_size, multiscale_feature_size) if use_multiscale else state_downsize
        obs_shape = (self.map_manager.get_map_channel() * (1 + use_global_features + n_scales if use_multiscale else 0),
                     *map_shape)
        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(low=0., high=1., shape=obs_shape, dtype=np.float32),
            "vector": gym.spaces.Box(low=-1., high=1., shape=(1,), dtype=np.float32),
            "weed_ratio": gym.spaces.Box(low=0., high=1., shape=(1,), dtype=np.float32)
        })

    # -----------------------------
    # reset
    # -----------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) \
            -> Tuple[Dict[str, Union[np.ndarray, float]], Dict[str, Any]]:
        super().reset(seed=seed)

        self.map_manager.rng = self.np_random
        self.observation_manager.rng = self.np_random

        opts = options or {}
        weed_dist = opts.get("weed_distribution", "uniform")
        weed_count = opts.get("weed_count", 100)
        map_id = opts.get("map_id", None)
        specific_scenario_dir = opts.get("specific_scenario_dir", None)
        initial_position, initial_direction = opts.get("initial_position", None), opts.get("initial_direction", None)

        # 加载或随机生成地图
        self.maps_dict, agent_dict, self.pad_info, self.state_info = self.map_manager.generate_scenario(
            map_id=map_id,
            weed_dist=weed_dist,
            weed_count=weed_count,
            specific_scenario_dir=specific_scenario_dir,
        )

        self.agent = agent_dict["agent_1"]
        if initial_position and initial_direction:
            self.agent.reset(position=initial_position, direction=initial_direction)

        self.scene_dynamic.reset(self.maps_dict, self.agent, self.state_info)
        self.current_step = 0

        observation = self._get_observation()
        return observation, {}

    def step(self, action: Union[int, Tuple[int, int], Tuple[float, float]]) \
            -> Tuple[Dict[str, Union[np.ndarray, float]], float, bool, bool, Dict[str, Any]]:
        linear_vel, angular_vel = self._parse_action(action)
        self.maps_dict, self.agent, self.state_info = self.scene_dynamic.dynamic(maps_dict=self.maps_dict,
                                                                                 agent=self.agent,
                                                                                 state_info=self.state_info,
                                                                                 linear_velocity=linear_vel,
                                                                                 angular_velocity=angular_vel)

        reward = self.reward_manager.calculate_step_reward(state_info=self.state_info)
        crashed, finished = self.state_info["crashed"], self.state_info["finished"]
        time_out = (self.current_step >= self.max_episode_steps)
        done = crashed or finished or time_out
        obs = self._get_observation()

        self.current_step += 1
        return obs, float(reward), done, time_out, {}

    def _parse_action(self, action: Union[int, Tuple[int, int], Tuple[float, float]]) -> Tuple[float, float]:
        """
        使用 nvec=(7,21)，v_range=(0,3.5), w_range=(-28.6,28.6)
        """
        if self.action_type == 'discrete':
            acc = action // self.nvec[1]  # [0..6]
            steer_idx = action % self.nvec[1]  # [0..20]
            linear_vel = (self.v_range.min + (acc + 1) / self.nvec[0] * self.v_range.mode)  # 7档
            angular_vel = (self.w_range.min + steer_idx / (self.nvec[1] - 1) * self.w_range.mode)
        elif self.action_type == 'multi_discrete':
            acc, steer_idx = action
            linear_vel = (self.v_range.min + (acc + 1) / self.nvec[0] * self.v_range.mode)
            angular_vel = (self.w_range.min + steer_idx / (self.nvec[1] - 1) * self.w_range.mode)
        elif self.action_type == 'continuous':
            linear_vel, angular_vel = action
        else:
            raise NotImplementedError(f"不支持动作类型: {self.action_type}")

        return float(linear_vel), float(angular_vel)

    # 要实现不同的观测模式，还真要apf的在这里加载才可以
    def _get_observation(self) -> Dict[str, Union[np.ndarray, float]]:
        """
        基于新的 ObservationManager, 将 map_manager 中的地图打包为 maps_dict 并生成观测。
        """
        # TODO: 现在这样获取也是不对的，应该从padding_info中获取，这个地方应该开放可拓展性，这里本质还是在给一个map_input_dict，怎么给就有很多方法了，但是不同任务的observation空间是不同的，这里就存在observation还要思考修改的地方
        map_input_dict = {
            key: {"map": self.maps_dict[key], "pad": 0.0 if key != "obstacle" else 1.0}
            for key in ["field_frontier", "obstacle", "weed", "trajectory"]
        }

        obs_img = self.observation_manager.generate_observation(
            agent=self.agent,
            maps_dict=map_input_dict
        )

        vector_obs = self.agent.last_steer / self.w_range.max

        total_weed_count = self.map_manager.total_weed_count
        current_weed_count = self.state_info["weed_count"]
        weed_ratio = 1.0 - float(current_weed_count / total_weed_count) if total_weed_count > 0 else 1.0

        return {
            "observation": obs_img.astype(np.float32),
            "vector": np.float32(vector_obs),
            "weed_ratio": np.float32(weed_ratio),
        }

    # -----------------------------
    # render
    # -----------------------------
    def render(self) -> Optional[np.ndarray]:
        if self.render_mode not in self.metadata["render_modes"]:
            if hasattr(gym.logger, "warn"):
                gym.logger.warn("You are calling render without specifying render_mode. Default is None.")
            return None

        if pygame is None:
            raise gym.error.DependencyNotInstalled("pygame 未安装，请使用 pip install pygame。")

        if self.screen is None:
            pygame.init()
            if self.state_pixels:
                w = self.state_size[0] * self.render_repeat_times
                h = self.state_size[1] * self.render_repeat_times
                self.screen = pygame.Surface((w, h))
            else:
                w, h = self.state_info["dimensions"]
                self.screen = pygame.Surface((w * self.render_repeat_times, h * self.render_repeat_times))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.state_pixels:
            img = self._render_first_person_view()
        else:
            img = self._render_map()

        # repeat
        img = img.repeat(self.render_repeat_times, axis=0).repeat(self.render_repeat_times, axis=1)

        surf = pygame.surfarray.make_surface(img)
        self.screen.blit(surf, (0, 0))

        arr3d = pygame.surfarray.pixels3d(self.screen)
        rendered_image = np.transpose(arr3d, (1, 0, 2))
        return rendered_image

    def _render_map(self) -> np.ndarray:
        """
        渲染地图，包括障碍物、杂草、机器人轨迹、机器人视野、机器人位置等。
        """
        h, w = self.state_info["dimensions"][1], self.state_info["dimensions"][0]
        rendered_map = np.ones((h, w, 3), dtype=np.uint8) * 255

        color_map = {
            "field_frontier": (76, 187, 23), "covered_farmland_blend": (112, 173, 7), "obstacle": (30, 75, 130),
            "weed": (255, 0, 0), "weed_undiscovered": (0, 0, 0), "weed_discovered": (255, 0, 0),
            "covered_weed_blend": (0, 0, 0), "trajectory": (255, 38, 255), "robot_poly": (255, 0, 0),
            "robot_vision": (192, 192, 192), "tv_frontier": (255, 38, 255), "tv_obstacle": (47, 82, 143)
        }

        base_render_items = [("field_frontier", "field_frontier", 1.0), ("obstacle", "obstacle", 1.0),
                             ("trajectory", "trajectory", 1.0)]
        for map_key, color_key, alpha_val in base_render_items:
            if map_key in self.maps_dict:
                rendered_map = apply_mask_with_color(rendered_map, self.maps_dict[map_key], color_map[color_key],
                                                     alpha=alpha_val)

        if self.render_covered_farmland and "original_field_frontier" in self.maps_dict and "field_frontier" in self.maps_dict:
            covered_mask = np.logical_and(self.maps_dict["original_field_frontier"],
                                          np.logical_not(self.maps_dict["field_frontier"]))
            rendered_map = apply_mask_with_color(rendered_map, covered_mask, color_map["covered_farmland_blend"],
                                                 alpha=0.25)

        cv2.ellipse(img=rendered_map, center=self.agent.position_discrete,
                    axes=(int(self.agent.vision_length), int(self.agent.vision_length)),
                    angle=self.agent.direction, startAngle=-int(self.agent.vision_angle / 2), endAngle=int(self.agent.vision_angle / 2),
                    color=color_map["robot_vision"], thickness=-1)



        if "weed" in self.maps_dict and "field_frontier" in self.maps_dict:
            weed_undiscovered = get_map_pasture_larger(
                np.logical_and(self.maps_dict["weed"], self.maps_dict["field_frontier"]))
            weed_discovered = get_map_pasture_larger(
                np.logical_and(self.maps_dict["weed"], np.logical_not(self.maps_dict["field_frontier"])))
            rendered_map = apply_mask_with_color(rendered_map, weed_undiscovered, color_map["weed_undiscovered"])
            rendered_map = apply_mask_with_color(rendered_map, weed_discovered, color_map["weed_discovered"])

        if self.render_covered_weed and "original_weed" in self.maps_dict and "weed" in self.maps_dict:
            weed_covered = get_map_pasture_larger(
                np.logical_and(self.maps_dict["original_weed"], np.logical_not(self.maps_dict["weed"])))
            rendered_map = apply_mask_with_color(rendered_map, weed_covered, color_map["covered_weed_blend"], alpha=0.9)

        cv2.fillPoly(rendered_map, [self.agent.convex_hull.round().astype(np.int32)], color_map["robot_poly"])

        if self.render_tv:
            if "field_frontier" in self.maps_dict:
                mask_tv = total_variation_mat(self.maps_dict["field_frontier"])
                rendered_map = apply_mask_with_color(rendered_map, mask_tv, color_map["tv_frontier"])
            if "map_obstacle" in self.maps_dict:
                mask_tv = total_variation_mat(self.maps_dict["map_obstacle"])
                rendered_map = apply_mask_with_color(rendered_map, mask_tv, color_map["tv_obstacle"])

        if self.render_mist and "mist" in self.maps_dict:
            mist_map = self.maps_dict["mist"]
            rendered_map = np.where(np.expand_dims(mist_map, axis=-1) == 1, (rendered_map * 0.7).astype(np.uint8),
                                    rendered_map)

        return rendered_map

    def _render_first_person_view(self) -> np.ndarray:
        """
        截取并旋转地图到第一人称视角
        """
        base_map = self._render_map().astype(np.float32)
        noisy_y, noisy_x, noisy_dir = self.observation_manager._get_noisy_pose(self.agent)

        map_dict_for_render = {
            "R": {"map": base_map[..., 0], "pad": 128.},
            "G": {"map": base_map[..., 1], "pad": 128.},
            "B": {"map": base_map[..., 2], "pad": 128.},
        }

        stacked_maps, pad_values = self.observation_manager.stack_maps(map_dict_for_render)
        final_obs = self.observation_manager.extract_ego_observation(
            maps=stacked_maps,
            pad_values=pad_values,
            center_y=noisy_y,
            center_x=noisy_x,
            direction_deg=noisy_dir,
            patch_size=self.state_size
        )

        return final_obs.astype(np.uint8)

    def close(self):
        if self.screen is not None and pygame is not None:
            pygame.display.quit()
            pygame.quit()
            self.is_open = False


if __name__ == "__main__":
    if_render = True
    episodes = 3
    env = CppEnvBase(
        render_mode='rgb_array' if if_render else None,
        # state_pixels=True,
        state_pixels=False,
    )
    env: CppEnvBase = HumanRendering(env)

    for _ in range(episodes):
        # reset_state = {
        #     'seed': 120, 'options': {
        #         'weed_dist': 'gaussian',
        #         # 'map_id': 80,
        #         "weed_num": 100
        #     }
        # }
        reset_state = {}
        obs, info = env.reset(**reset_state)
        done = False
        while not done:
            action = env.action_space.sample()
            # action = 1 * 21 + 10
            obs, reward, done, _, info = env.step(action)
            # obs, reward, done, _, info = env.step((0, 4))
            print(reward)
            if if_render:
                env.render()

    env.close()
