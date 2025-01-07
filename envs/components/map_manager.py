from __future__ import annotations

import os
import cv2
import math
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Union

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
        self.weed_noise = weed_noise
        self.rng = rng if rng else np.random.default_rng()

        # 地图属性初始化
        self.dimensions: Tuple[int, int] = (0, 0)  # (width, height)
        self.field_frontier_map: np.ndarray = np.zeros((1, 1), dtype=np.uint8)
        self.original_field_frontier_map: np.ndarray = np.zeros((1, 1), dtype=np.uint8)
        self.obstacle_map: np.ndarray = np.zeros((1, 1), dtype=np.uint8)
        self.weed_map: np.ndarray = np.zeros((1, 1), dtype=np.uint8)
        self.weed_noise_map: np.ndarray = np.zeros((1, 1), dtype=np.uint8)
        self.original_weed_map: np.ndarray = np.zeros((1, 1), dtype=np.uint8)
        self.contours: List[np.ndarray] = []
        self.initial_bounding_box: List[np.ndarray] = []
        self.total_weed_count: int = 0

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

        self.field_frontier_map = (cv2.imread(str(directory / 'frontier_map.png'), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        self.obstacle_map = (cv2.imread(str(directory / 'obstacle_map.png'), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        self.weed_map = (cv2.imread(str(directory / 'weed_map.png'), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)

        # 确保杂草仅存在于农田前沿区域
        self.weed_map[self.field_frontier_map == 0] = 0

        # 膨胀障碍物地图以防止杂草靠近障碍物
        kernel_size = (29, 29)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        dilated_obstacles = cv2.dilate(self.obstacle_map, kernel, iterations=1)
        self.weed_map[dilated_obstacles > 0] = 0

        # 更新其他属性
        self.dimensions = self.field_frontier_map.shape[::-1]
        self.original_field_frontier_map = self.field_frontier_map.copy()
        self.total_weed_count = self.weed_map.sum()

        # 初始化杂草噪声地图和原始杂草地图
        self._initialize_weed_maps()

    def generate_frontier_maps(self, map_id: int):
        """
        通过加载指定 ID 的地图生成农田前沿地图。

        :param map_id: 地图 ID。
        """
        self.load_field_frontier_map(map_id)

    def initialize_boudingbox(self) -> Tuple[Tuple[float, float], float]:
        """
        基于农田前沿地图找到最大轮廓并提取最小包围矩形，作为初始位置和方向的参考。

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

    def initialize_obstacle_map(self, agent_position: Tuple[float, float]):
        """
        随机生成障碍物，并更新农田前沿地图。

        :param agent_position: 机器人初始位置。
        """
        self.obstacle_map = np.zeros((self.dimensions[1], self.dimensions[0]), dtype=np.uint8)

        if self.use_box_boundary and self.initial_bounding_box:
            # 填充整张地图为障碍物
            self.obstacle_map.fill(1)
            # 清空农田区域
            box = self.initial_bounding_box[0]
            cv2.fillPoly(self.obstacle_map, [box], color=0)

        # 随机生成离散障碍物
        num_obstacles = self.rng.integers(*self.num_obstacles_range) if self.num_obstacles_range[1] > 0 else 0
        agent_x, agent_y = agent_position

        for _ in range(num_obstacles):
            o_x = self.rng.uniform(100, self.dimensions[0] - 100)
            o_y = self.rng.uniform(100, self.dimensions[1] - 100)
            if self.obstacle_map[int(o_y), int(o_x)]:
                continue

            o_length = self.rng.uniform(*self.obstacle_size_range)
            o_width = self.rng.uniform(*self.obstacle_size_range)
            angle = self.rng.uniform(0., 360.)
            rotated_rect = ((o_x, o_y), (o_length, o_width), angle)
            obstacle_points = cv2.boxPoints(rotated_rect).astype(int).reshape((-1, 1, 2))

            # 确保与机器人保持足够距离
            distance = cv2.pointPolygonTest(obstacle_points, (agent_x, agent_y), True)
            if distance < -2.0 * MowerAgent.occupancy:
                self.obstacle_map = cv2.fillPoly(self.obstacle_map, [obstacle_points], color=1)
                # 扩展障碍物区域，防止杂草靠近
                expanded_rect = ((o_x, o_y), (o_length + 15, o_width + 15), angle)
                expanded_points = cv2.boxPoints(expanded_rect).astype(int).reshape((-1, 1, 2))
                self.field_frontier_map = cv2.fillPoly(self.field_frontier_map, [expanded_points], color=0)


    def randomize_obstacles(self, agent_position: Tuple[float, float]): # TODO: 这里逻辑有问题，注意修改
        """
        根据 use_box_boundary, num_obstacles_range 等在 map_obstacle 上添加障碍物，
        并更新 map_frontier 相应区域。
        """
        if self.use_box_boundary and self.min_area_rect:
            # 初始化 map_obstacle 为全1（表示全部为障碍物）
            self.map_obstacle = np.ones((self.dimensions[1], self.dimensions[0]), dtype=np.uint8)
            box = self.min_area_rect[0]
            r_center = np.mean(box[:, 0, :], axis=0)

            # 计算宽度和高度方向
            vecs = [box[0, 0] - box[1, 0], box[1, 0] - box[2, 0]]
            wd_i = 0 if abs(vecs[1][1]) < abs(vecs[1][0]) else 1
            ht_i = (wd_i + 1) % 2
            angle = math.degrees(math.atan2(vecs[wd_i][1], vecs[wd_i][0]))
            width = max(math.hypot(*vecs[wd_i]) * 1.2, math.hypot(*vecs[wd_i]) + 60)
            height = max(math.hypot(*vecs[ht_i]) * 1.2, math.hypot(*vecs[ht_i]) + 60)
            rect_expanded = ((r_center[0], r_center[1]), (width, height), angle)
            box_expanded = cv2.boxPoints(rect_expanded)
            start_idx = box_expanded.sum(axis=1).argmin()
            box_expanded = np.roll(box_expanded, 4 - start_idx, 0)
            box_expanded = box_expanded.reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(self.map_obstacle, [box_expanded], color=(0,))  # 清空农田区域

        else:
            self.map_obstacle = np.zeros((self.dimensions[1], self.dimensions[0]), dtype=np.uint8)

        # 随机生成离散障碍物
        num_obstacles = self.np_random.integers(*self.num_obstacles_range) if self.num_obstacles_range[1] > 0 else 0
        agent_x, agent_y = agent_position

        for _ in range(num_obstacles):
            # 随机尝试放置障碍物
            o_x = self.np_random.uniform(100, self.dimensions[0] - 100)
            o_y = self.np_random.uniform(100, self.dimensions[1] - 100)
            if self.map_obstacle[int(o_y), int(o_x)]:
                continue
            o_len = self.np_random.uniform(*self.obstacle_size_range)
            o_wid = self.np_random.uniform(*self.obstacle_size_range)
            angle = self.np_random.uniform(0., 360.)
            rotated_rect = ((o_x, o_y), (o_len, o_wid), angle)
            pts = cv2.boxPoints(rotated_rect).astype(np.int32).reshape((-1, 1, 2))

            # 保持与 agent 的初始位置足够距离
            dist2player = cv2.pointPolygonTest(pts, (agent_x, agent_y), True)
            if dist2player < -2.0 * MowerAgent.occupancy:
                # 填充障碍物
                self.map_obstacle = cv2.fillPoly(self.map_obstacle, [pts], color=(1,))
                # 扩展障碍物区域，防止杂草靠近
                pts_expanded = cv2.boxPoints(((o_x, o_y), (o_len + 15, o_wid + 15), angle)).astype(np.int32).reshape((-1, 1, 2))
                self.map_frontier = cv2.fillPoly(self.map_frontier, [pts_expanded], color=(0,))

    def initialize_weed_distribution(self, distribution: str, weed_count: int):
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

    def _populate_weeds_gaussian(self, weed_count: int): # TODO：高斯分布有问题，应该是再boudingbox中心而不是地图中心
        """
        使用高斯分布填充杂草。

        :param weed_count: 杂草数量。
        """
        center_x, center_y = self.dimensions[0] / 2, self.dimensions[1] / 2
        scale_x, scale_y = self.dimensions[0] * 0.35, self.dimensions[1] * 0.35
        candidates = self.rng.normal(loc=[center_x, center_y], scale=[scale_x, scale_y], size=(weed_count * 2, 2))
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
