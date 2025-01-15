from __future__ import annotations

import os
import cv2
import math
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict, Any

from envs.components.utils import MowerAgent


class MapManager:
    """
    管理与地图相关的操作，包括地图加载、生成、障碍物随机化和杂草初始化。
    """

    def __init__(
            self,
            map_dir: str = 'envs/maps/1-400',
            num_obstacles_range: Tuple[int, int] = (5, 8),
            obstacle_size_range: Tuple[int, int] = (10, 25),
            use_box_boundary: bool = True,
            weed_noise: float = 0.0,
            use_traj: bool = True,
            use_mist: bool = True,
            rng: Optional[np.random.Generator] = None
    ):
        """
        初始化地图管理器。

        :param map_dir: 地图文件夹路径。
        :param num_obstacles_range: 障碍物数量范围。
        :param obstacle_size_range: 障碍物尺寸范围。
        :param use_box_boundary: 是否使用边界框作为障碍物。
        :param weed_noise: 杂草噪声系数。
        :param rng: 随机数生成器。
        """
        self.map_dir = Path(map_dir)
        self.map_names = sorted(os.listdir(self.map_dir))
        self.num_obstacles_range = num_obstacles_range
        self.obstacle_size_range = obstacle_size_range
        self.use_box_boundary = use_box_boundary
        self.use_traj = use_traj
        self.use_mist = use_mist
        self.weed_noise = weed_noise
        self.rng = rng

        # 地图属性初始化
        self.dimensions: Tuple[int, int] = (0, 0)  # (width, height)
        self.field_frontier_map: np.ndarray = np.zeros((1, 1), dtype=np.uint8)
        self.original_field_frontier_map: np.ndarray = np.zeros((1, 1), dtype=np.uint8)
        self.obstacle_map: np.ndarray = np.zeros((1, 1), dtype=np.uint8)
        self.weed_map: np.ndarray = np.zeros((1, 1), dtype=np.uint8)
        self.weed_noise_map: np.ndarray = np.zeros((1, 1), dtype=np.uint8)
        self.original_weed_map: np.ndarray = np.zeros((1, 1), dtype=np.uint8)
        self.initial_bounding_box: List[np.ndarray] = []
        self.total_weed_count: int = 0

    def generate_scenario(
            self,
            map_id: Optional[int] = None,
            weed_dist: str = "uniform", weed_count: int = 100,
            specific_scenario_dir: Optional[str] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], Dict[str, float], Dict[str, Any]]:
        """
        生成一个场景，包括地图、机器人和填充信息。
        """

        if specific_scenario_dir:
            self.load_maps_from_directory(specific_scenario_dir)
            init_pos, init_dir = self.initialize_boudingbox_and_position()
        else:
            map_id = map_id or self.rng.integers(0, len(self.map_names))
            self.generate_frontier_maps(map_id)
            init_pos, init_dir = self.initialize_boudingbox_and_position()
            self.generate_obstacle_map(init_pos)
            self.generate_weed_map(weed_dist, weed_count)

        if self.use_box_boundary:
            self._draw_obstacle_map_boxboundary()

        agent = MowerAgent(position=init_pos, direction=init_dir)

        maps_dict = {
            "field_frontier": self.field_frontier_map, "original_field_frontier": self.original_field_frontier_map,
            "obstacle": self.obstacle_map,
            "weed": self.weed_map, "weed_noise": self.weed_noise_map, "original_weed": self.original_weed_map
        }
        if self.use_traj: maps_dict["trajectory"] = np.ones((self.dimensions[1], self.dimensions[0]), dtype=np.uint8)
        if self.use_mist: maps_dict["mist"] = np.ones((self.dimensions[1], self.dimensions[0]), dtype=np.uint8)

        agent_dict = {"agent_1": agent}
        padding_info = {"field_frontier": 0.0, "obstacle": 1.0, "weed": 0.0, "trajectory": 0.0, "mist": 1}
        state_info = {"initial_bounding_box": self.initial_bounding_box, "dimensions": self.dimensions}

        return maps_dict, agent_dict, padding_info, state_info

    def generate_frontier_maps(self, map_id: int):
        """
        生成指定 ID 的农田前沿地图，并更新相关属性。
        """
        self.load_field_frontier_map(map_id)

    def load_field_frontier_map(self, map_id: int):
        """
        加载指定 ID 的农田前沿地图，并更新相关属性。

        :param map_id: 地图 ID。
        :raises ValueError: 如果 map_id 超出范围。
        :raises FileNotFoundError: 如果地图文件未找到。
        """
        if not (0 <= map_id < len(self.map_names)):
            raise ValueError(f"map_id {map_id} 超出范围。")

        map_name = self.map_names[map_id]
        map_path = self.map_dir / map_name
        loaded_image = cv2.imread(str(map_path))
        if loaded_image is None:
            raise FileNotFoundError(f"未找到地图文件: {map_path}")

        self.field_frontier_map = (loaded_image.sum(axis=-1) > 0).astype(np.uint8)
        self.dimensions = self.field_frontier_map.shape[::-1]  # (宽, 高)
        self.original_field_frontier_map = self.field_frontier_map.copy()

    def load_maps_from_directory(self, directory: Union[str, Path]) -> None:
        """
        从指定目录加载农田前沿地图、障碍物地图和杂草地图。

        :param directory: 地图文件所在目录。
        :raises FileNotFoundError: 如果必要的地图文件未找到。
        """
        directory = Path(directory)
        required_files = ['frontier_map.png', 'obstacle_map.png', 'weed_map.png']
        for file in required_files:
            if not (directory / file).exists():
                raise FileNotFoundError(f"必需的地图文件 '{file}' 未在 '{directory}' 中找到。")

        self.field_frontier_map = (cv2.imread(str(directory / 'frontier_map.png'), cv2.IMREAD_GRAYSCALE) > 0).astype(
            np.uint8)
        self.obstacle_map = (cv2.imread(str(directory / 'obstacle_map.png'), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        self.weed_map = (cv2.imread(str(directory / 'weed_map.png'), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)

        # 确保杂草仅存在于农田前沿区域
        self.weed_map[self.field_frontier_map == 0] = 0

        # 膨胀障碍物地图以防止杂草靠近障碍物
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29))
        dilated_obstacles = cv2.dilate(self.obstacle_map, kernel, iterations=1)
        self.weed_map[dilated_obstacles > 0] = 0

        # 更新其他属性
        self.dimensions = self.field_frontier_map.shape[::-1]
        self.original_field_frontier_map = self.field_frontier_map.copy()
        self.total_weed_count = self.weed_map.sum()

        # 初始化杂草噪声地图和原始杂草地图
        self._initialize_weed_maps()

    def initialize_boudingbox_and_position(self) -> Tuple[Tuple[float, float], float]:
        """
        基于农田前沿地图找到最大轮廓并提取最小包围矩形，作为机器人初始位置和方向的参考。

        :return: 初始位置和方向角度。
        :raises ValueError: 如果未找到任何轮廓。
        """
        contours, _ = cv2.findContours(self.field_frontier_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("在 field_frontier_map 中未找到任何轮廓。")

        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)  # (center, (w, h), angle)
        box = cv2.boxPoints(rect)
        start_idx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - start_idx, 0).astype(int).reshape((-1, 1, 2))
        self.initial_bounding_box = [box]

        # 选择较长的边作为方向参考
        edge1 = box[1, 0] - box[0, 0]
        edge2 = box[2, 0] - box[1, 0]
        if math.hypot(*edge1) > math.hypot(*edge2):
            reference_edge = edge1
        else:
            reference_edge = edge2

        initial_position = (float(box[0, 0, 0]), float(box[0, 0, 1]))
        direction_vector = reference_edge / math.hypot(*reference_edge)
        initial_direction = math.degrees(math.atan2(direction_vector[1], direction_vector[0]))

        return initial_position, initial_direction

    def _draw_obstacle_map_boxboundary(self):
        """仅绘制外边界障碍，不进行随机采样。"""
        assert self.initial_bounding_box and self.obstacle_map is not None
        box = self.initial_bounding_box[0]

        # 计算 box 的中心、宽高与旋转角度
        r_center = 0.5 * (box[0, 0] + box[2, 0])
        vecs = [box[0, 0] - box[1, 0], box[1, 0] - box[2, 0]]
        wd_i = 0
        if abs(vecs[1][1]) < abs(vecs[1][0]):
            wd_i = 0
        ht_i = (wd_i + 1) % 2
        angle = math.atan2(vecs[wd_i][1], vecs[wd_i][0]) * 180.0 / math.pi
        width = math.hypot(*vecs[wd_i])
        height = math.hypot(*vecs[ht_i])

        # 对宽高进行放大
        width = max(width * 1.2, width + 60)
        height = max(height * 1.2, height + 60)
        rect_expanded = ((r_center[0], r_center[1]), (width, height), angle)
        box_points = cv2.boxPoints(rect_expanded)
        start_idx = box_points.sum(axis=1).argmin()
        box_points = np.roll(box_points, 4 - start_idx, 0)
        box_points = box_points.reshape((-1, 1, 2)).astype(np.int32)

        obstacle_map_with_boundary = np.ones_like(self.obstacle_map)
        cv2.fillPoly(obstacle_map_with_boundary, [box_points], color=(0,))
        self.obstacle_map = np.where(obstacle_map_with_boundary == 1, 1, self.obstacle_map)

    def generate_obstacle_map(self, agent_position: Tuple[float, float]):  # 这里初始化的超参比较多，以后考虑用config文件
        """
        随机生成障碍物，并更新农田前沿地图。
        :param agent_position: 机器人初始位置 (x, y)，用于与障碍物保持距离。
        """
        # 如果启用边界框，并且已有 bounding box 信息，则先填充为全障碍物，再清空该 box 区域
        self.obstacle_map = np.zeros((self.dimensions[1], self.dimensions[0]), dtype=np.uint8)

        # 决定要放置的障碍物数量
        num_obstacles = self.rng.integers(*self.num_obstacles_range) if self.num_obstacles_range[1] > 0 else 0
        current_obstacle_num = 0
        agent_x, agent_y = agent_position

        while current_obstacle_num < num_obstacles:
            # 随机生成障碍物中心
            o_x = self.rng.uniform(100, self.dimensions[0] - 100)
            o_y = self.rng.uniform(100, self.dimensions[1] - 100)

            # 如果该点已被障碍物占据，直接继续
            if self.obstacle_map[int(o_y), int(o_x)]:
                continue

            # 随机障碍物的长宽和旋转角度，生成障碍物顶点坐标
            o_len = self.rng.uniform(*self.obstacle_size_range)
            o_wid = self.rng.uniform(*self.obstacle_size_range)
            angle = self.rng.uniform(0., 360.)

            rotated_rect = cv2.RotatedRect(
                center=(o_x, o_y), size=(o_len, o_wid), angle=angle
            )
            pts = np.array(rotated_rect.points(), dtype=np.int32).reshape((-1, 1, 2))

            # 检测与机器人初始位置的距离，保持足够安全距离
            dist2player = cv2.pointPolygonTest(pts, (agent_x, agent_y), True)
            if dist2player < -2.0 * MowerAgent.length:
                current_obstacle_num += 1
                cv2.fillPoly(self.obstacle_map, [pts], color=(1,))

                # 同时在 field_frontier_map 中填充一个略大的区域，用于防止杂草靠近
                expanded_rect = cv2.RotatedRect(
                    center=(o_x, o_y),
                    size=(o_len + 15, o_wid + 15),
                    angle=angle
                )
                pts_expanded = np.array(expanded_rect.points(), dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(self.field_frontier_map, [pts_expanded], color=(0,))


    def _draw_box_boundary_if_needed(self):
        """仅绘制外边界障碍，不进行随机采样。"""
        if self.use_box_boundary and self.initial_bounding_box:
            self.obstacle_map = np.ones((self.dimensions[1], self.dimensions[0]), dtype=np.uint8)
            box = self.initial_bounding_box[0]

            # 计算 box 的中心、宽高与旋转角度
            r_center = 0.5 * (box[0, 0] + box[2, 0])
            vecs = [box[0, 0] - box[1, 0], box[1, 0] - box[2, 0]]
            wd_i = 0
            if abs(vecs[1][1]) < abs(vecs[1][0]):
                wd_i = 0
            ht_i = (wd_i + 1) % 2
            angle = math.atan2(vecs[wd_i][1], vecs[wd_i][0]) * 180.0 / math.pi
            width = math.hypot(*vecs[wd_i])
            height = math.hypot(*vecs[ht_i])

            # 对宽高进行放大
            width = max(width * 1.2, width + 60)
            height = max(height * 1.2, height + 60)
            rect_expanded = ((r_center[0], r_center[1]), (width, height), angle)
            box_points = cv2.boxPoints(rect_expanded)
            start_idx = box_points.sum(axis=1).argmin()
            box_points = np.roll(box_points, 4 - start_idx, 0)
            box_points = box_points.reshape((-1, 1, 2)).astype(np.int32)

            # 填充为障碍物后，再在 box 区域清空
            cv2.fillPoly(self.obstacle_map, [box_points], color=(0,))


    def generate_weed_map(self, distribution: str, weed_count: int):
        """
        初始化杂草分布，支持均匀分布和高斯分布。

        :param distribution: 分布类型，'uniform' 或 'gaussian'。
        :param weed_count: 杂草数量。
        :raises ValueError: 如果分布类型不受支持。
        """
        self.weed_map = np.zeros((self.dimensions[1], self.dimensions[0]), dtype=np.uint8)

        if isinstance(weed_count, float):
            weed_count = math.ceil(self.original_field_frontier_map.sum() * weed_count)

        self.total_weed_count = weed_count

        if distribution == 'uniform':
            self._populate_weeds_uniform(weed_count)
        elif distribution == 'gaussian':
            self._populate_weeds_gaussian(weed_count)
        else:
            raise ValueError(f"不支持的杂草分布类型: {distribution}")

        self._initialize_weed_maps()


    def _populate_weeds_uniform(self, weed_count: int):
        """
        使用均匀分布填充杂草。

        :param weed_count: 杂草数量。
        """
        possible_positions = np.argwhere(self.field_frontier_map)
        self.rng.shuffle(possible_positions)
        selected_positions = possible_positions[:weed_count]
        self.weed_map[selected_positions[:, 0], selected_positions[:, 1]] = 1


    def _populate_weeds_gaussian(self, weed_count: int):  # TODO：高斯分布未来调整地更完善一些
        """
        使用高斯分布填充杂草。

        :param weed_count: 杂草数量。
        """
        center_x, center_y = self.dimensions[0] / 2, self.dimensions[1] / 2
        scale_x, scale_y = self.dimensions[0] * 0.35, self.dimensions[1] * 0.35
        candidates = self.rng.normal(loc=[center_x, center_y], scale=[scale_x, scale_y], size=(weed_count * 5, 2))
        candidates = np.round(candidates).astype(int)
        candidates = np.clip(candidates, [0, 0], [self.dimensions[0] - 1, self.dimensions[1] - 1])

        # 获取唯一位置
        unique_candidates = np.unique(candidates, axis=0)
        # 过滤在农田前沿且未被占用的位置
        valid_candidates = unique_candidates[
            np.logical_and(self.field_frontier_map[unique_candidates[:, 1], unique_candidates[:, 0]] == 1,
                           self.weed_map[unique_candidates[:, 1], unique_candidates[:, 0]] == 0)
        ]

        selected_positions = valid_candidates[:weed_count]
        self.weed_map[selected_positions[:, 1], selected_positions[:, 0]] = 1


    def _initialize_weed_maps(self):
        """
        初始化噪声杂草地图和原始杂草地图。
        """
        if self.weed_noise > 0.0:
            self.weed_noise_map = self._apply_weed_noise(self.weed_map)
        else:
            self.weed_noise_map = self.weed_map.copy()
        self.original_weed_map = self.weed_map.copy()


    def _apply_weed_noise(self, weed_map: np.ndarray) -> np.ndarray:
        """
        应用噪声到杂草地图，通过随机位移扰动现有杂草位置。

        :param weed_map: 原始杂草地图。
        :return: 扰动后的杂草地图。
        """
        # 获取现有杂草位置
        weed_positions = np.argwhere(weed_map)
        num_weeds = weed_positions.shape[0]

        if num_weeds == 0:
            return np.zeros_like(weed_map)

        # 生成随机位移（-1, 0, 1）应用于所有杂草位置
        shifts = self.rng.integers(-1, 2, size=(num_weeds, 2))
        shifted_positions = weed_positions + shifts

        # 保持杂草位置在地图范围内
        shifted_positions = np.clip(shifted_positions, [0, 0], [self.dimensions[1] - 1, self.dimensions[0] - 1])

        # 移除重复位置
        shifted_positions = np.unique(shifted_positions, axis=0)

        # 创建新的噪声杂草地图
        weed_noise_map = np.zeros_like(weed_map)
        weed_noise_map[shifted_positions[:, 0], shifted_positions[:, 1]] = 1

        # 确保杂草仅存在于农田前沿区域
        weed_noise_map = np.logical_and(weed_noise_map, self.field_frontier_map).astype(np.uint8)
        return weed_noise_map


    def get_map_dimensions(self) -> Tuple[int, int]:
        """
        获取地图的宽度和高度。
        """
        return self.dimensions

    def get_map_channel(self) -> int:
        """
        获取地图的通道数。
        """
        return 3 + self.use_traj + self.use_mist