"""
极简地图生成组件 - 无继承、无抽象的组件设计
每个组件包含完整的业务逻辑，通过简单字典状态进行通信
"""
import math
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

from envs_new.components.entity.agent import AgentFactory


def load_map_from_directory(directory: Union[str, Path], map_type: str) -> np.ndarray:
    """从目录加载指定类型的地图文件"""
    file_path = Path(directory) / f"map_{map_type}.png"
    if not file_path.exists():
        raise FileNotFoundError(f"Map file not found: {file_path}")

    image = cv2.imread(str(file_path),
                       cv2.IMREAD_COLOR if map_type == 'field' else cv2.IMREAD_GRAYSCALE)  # field用彩色图像，其他用灰度图
    # 彩图或者灰度图处理逻辑稍微差异
    return (image.sum(axis=-1) > 0).astype(np.uint8) if len(image.shape) == 3 else (image > 0).astype(np.uint8)


class FieldCreator:
    """加载田地地图并建立环境基础"""

    @classmethod
    def get_dependencies(cls) -> List[str]:
        return []

    def generate(self, state: Dict[str, Any], rng: np.random.Generator) -> None:
        options = state['options']
        config = state['config']

        # 在本回合首次确定缩放因子（单一真相），写入 env_state
        if config.field_scale_enabled:
            s_min, s_max = tuple(config.field_scale_range)
            scale = float(rng.uniform(s_min, s_max))
        else:
            scale = 1.0


        # 统一的3元组返回值 - 两种模式都返回 (map, dimensions, field_id)
        field_map, dimensions, field_id = (self._load_from_directory(options['scenario_directory'])
                                           if options.get('scenario_directory') else
                                           self._load_from_file(options.get('map_id'), config, rng))

        # 若启用缩放：以图像中心为原点做同心等比仿射缩放（保持画布尺寸不变）
        if scale != 1.0: field_map = self._scale_binary_map_center(field_map, scale)
        bounding_box, field_contours = self._extract_geometry(field_map)  # 提取几何信息, 并存入maps_dicts

        # 存储field_id和field_sacle到env_state供其他组件使用 (HFI需要和filed地图匹配)
        state['env_state'].set_static_info('field_id', field_id)
        state['env_state'].set_static_info('field_scale', scale)

        state['maps_dict']['field'] = field_map
        state['maps_dict']['original_field'] = field_map.copy()
        state['maps_dict']['time_series_coveraged_field'] = np.zeros_like(field_map, dtype=np.uint32) # 初始化覆盖顺序秩标签图：未覆盖=0，首次覆盖时写入递增标签, 使用无符号整型存储更大范围的标签值

        state['env_state'].set_static_info('dimensions', dimensions)
        state['env_state'].set_static_info('bounding_box', bounding_box)
        state['env_state'].set_static_info('field_contours', field_contours)
        state['env_state'].set_static_info('total_field_area', int(field_map.sum()))  # 设置田地总面积

    def _load_from_directory(self, directory: Union[str, Path]) -> Tuple[np.ndarray, Tuple[int, int], Optional[int]]:
        """从预制场景目录加载田地地图"""
        field_map = load_map_from_directory(directory, 'field')
        dimensions = field_map.shape[::-1]  # (width, height)
        return field_map, dimensions, None  # 场景模式没有field_id

    def _load_from_file(self, map_id: Optional[int], config, rng: np.random.Generator) -> Tuple[
        np.ndarray, Tuple[int, int], int]:
        """从地图文件加载"""

        # 添加field子目录（config.map_dir应指向父目录）
        map_dir = config.get_absolute_map_dir() / 'field'
        if not map_dir.exists(): raise FileNotFoundError(f"Map directory does not exist: {map_dir}")

        map_files = sorted([f for f in map_dir.iterdir() if f.suffix.lower() == '.png'])
        if not map_files: raise ValueError(f"No PNG map files found in directory: {map_dir}")

        # 选择地图并转换为二值田地地图（任何非黑色像素都是田地）
        if map_id is None:
            map_id = rng.integers(0, len(map_files))
        image = cv2.imread(str(map_files[map_id]))
        field_map = (image.sum(axis=-1) > 0).astype(np.uint8)
        dimensions = field_map.shape[::-1]  # (width, height)
        
        # 提取field文件名中的实际编号
        field_id = int(map_files[map_id].stem.split('_')[1])

        return field_map, dimensions, field_id

    def _extract_geometry(self, field_map: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """提取边界框和轮廓"""
        contours, _ = cv2.findContours(field_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if not contours: raise ValueError("No contours found in field map")

        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_contour = sorted_contours[0]
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)

        # box顺序是按顺时针方向排列的四个顶点，起点在左上角
        start_idx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - start_idx, 0).astype(int)
        box = box.reshape((-1, 1, 2))

        return [box], sorted_contours

    def _scale_binary_map_center(self, binary_map: np.ndarray, scale: float) -> np.ndarray:
        """以图像中心为原点对二值掩码做同心等比缩放，保持画布尺寸不变。

        使用最近邻插值以确保二值性不被破坏。
        """
        if not isinstance(binary_map, np.ndarray) or binary_map.ndim != 2:
            raise ValueError("binary_map must be a 2D numpy array")
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        if abs(scale - 1.0) < 1e-6:
            return binary_map

        h, w = binary_map.shape
        cx, cy = w / 2.0, h / 2.0
        M = np.array([[scale, 0.0, (1.0 - scale) * cx],
                      [0.0, scale, (1.0 - scale) * cy]], dtype=np.float32)

        scaled = cv2.warpAffine(binary_map.astype(np.uint8), M, (w, h),
                                flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0)
        return scaled.astype(np.uint8)


class AgentCreator:
    """创建并定位Agent，根据boundary_source选择生成策略"""

    @classmethod
    def get_dependencies(cls) -> List[str]:
        return ['field']

    def generate(self, state: Dict[str, Any], rng: np.random.Generator) -> None:
        bounding_box = state['env_state'].get_static_info('bounding_box')
        dimensions = state['env_state'].get_static_info('dimensions')
        field_map = state['maps_dict']['field']

        position, direction = self._calculate_pose(bounding_box, dimensions, field_map, state['config'], rng,
                                                   state['options'].get('initial_position'),
                                                   state['options'].get('initial_direction'))

        state['agent'] = AgentFactory.create_mower_agent(state['config'], position, direction)

    def _calculate_pose(self, bounding_box: List[np.ndarray], dimensions: Tuple[int, int],
                        field_map: np.ndarray, config, rng: np.random.Generator,
                        override_position: Optional[Tuple[float, float]],
                        override_direction: Optional[float]) -> Tuple[Tuple[float, float], float]:
        """计算agent初始位置和方向，根据boundary_source分支处理"""
        if config.boundary_source == "field": #在field安全区域内随机生成agent位置和方向
            # 创建安全区域（腐蚀field，确保agent完全在内）
            safe_margin = int(config.agent_length * 2)  # 安全边距 = 2倍agent长度
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (safe_margin, safe_margin))
            safe_zone = cv2.erode(field_map, kernel, iterations=1)
            safe_positions = np.argwhere(safe_zone > 0)

            # 若安全区域为空，则在field质心处放置agent
            if len(safe_positions) == 0:
                field_positions = np.argwhere(field_map > 0)
                center = field_positions.mean(axis=0)
                position =  (float(center[1]), float(center[0]))
            else: # 否则从安全区任取一个位置作为起点
                y, x = safe_positions[rng.integers(0, len(safe_positions))]
                position = (float(x), float(y))
            direction = float(rng.uniform(0, 360))
        else:
            box = bounding_box[0]
            edge1, edge2 = box[1, 0] - box[0, 0], box[2, 0] - box[1, 0]
            pos_index, direction_vector = (0, edge1) if math.hypot(*edge1) > math.hypot(*edge2) else (1, edge2)
            position = (float(box[pos_index, 0, 0]), float(box[pos_index, 0, 1]))
            direction_vector = direction_vector / math.hypot(*direction_vector)
            direction = math.degrees(math.atan2(direction_vector[1], direction_vector[0]))

        # 用户覆盖（支持部分覆盖）
        if override_position is not None:
            position = override_position
        if override_direction is not None:
            direction = override_direction

        return position, direction

class ObstacleCreator:
    """生成随机障碍物"""

    @classmethod
    def get_dependencies(cls) -> List[str]:
        return ['field', 'agent']

    def generate(self, state: Dict[str, Any], rng: np.random.Generator) -> None:
        config = state['config']
        scenario_directory = state['options'].get('scenario_directory')
        dimensions = state['env_state'].get_static_info('dimensions')
        field_id = state['env_state'].get_static_info('field_id')
        scale = float(state['env_state'].get_static_info('field_scale', 1.0))
        obstacle_file = Path(config.get_absolute_map_dir()) / 'obstacle' / f'obstacle_{field_id}.png'

        # 判断加载模式
        if scenario_directory: # 若指定地址则直接加载
            obstacles = self._load_from_directory(scenario_directory, dimensions)
        elif obstacle_file.exists(): # 若指定field_id对应的obstalce文件存在则加载
            obstacles = (cv2.imread(str(obstacle_file), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8) # 加载预制obstacle
            if abs(scale - 1.0) > 1e-6: obstacles = self._scale_obstacle_map_center(obstacles, scale) # 处理缩放
            assert obstacles.shape == (dimensions[1], dimensions[0])
        else: # 否则随机生成障碍物
            obstacles = self._generate_obstacles(dimensions, state['maps_dict']['field'], state['agent'].position,
                                                 state['agent'].length, config, rng)
        if config.boundary_source:
            boundary = self._generate_boundary(state['env_state'].get_static_info('bounding_box'),
                                               state['maps_dict']['field'], config)
            obstacles = np.logical_or(obstacles, boundary).astype(np.uint8)
        state['maps_dict']['obstacle'] = obstacles

    def _load_from_directory(self, directory: Union[str, Path], dimensions: Tuple[int, int]) -> np.ndarray:
        """从预制场景目录加载obstacle地图"""
        obstacle_map = load_map_from_directory(directory, 'obstacle')

        width, height = dimensions
        if obstacle_map.shape != (height, width):  # 验证尺寸匹配
            raise ValueError(f"Obstacle map dimensions {obstacle_map.shape} don't match "
                             f"expected dimensions {(height, width)}")
        return obstacle_map

    def _generate_obstacles(self, dimensions: Tuple[int, int], field_map: np.ndarray,
                            agent_position: Tuple[float, float], agent_length: float, config,
                            rng: np.random.Generator) -> np.ndarray:
        """生成随机障碍物"""
        width, height = dimensions
        min_obstacles, max_obstacles = config.num_obstacles_range
        num_obstacles = rng.integers(min_obstacles, max_obstacles + 1)

        obstacle_map = np.zeros((height, width), dtype=np.uint8)
        if num_obstacles == 0 or max_obstacles <= 0: return obstacle_map

        # 10倍障碍物添加尝试
        obstacles_placed_count = 0
        max_attempts = num_obstacles * 10
        for _ in range(max_attempts):
            if obstacles_placed_count >= num_obstacles:
                break

            obstacle = self._create_random_obstacle(dimensions, config, rng)
            if self._is_valid_placement(obstacle, agent_position, obstacle_map, agent_length, config):
                # 如果障碍物有效，则添加到障碍物地图，并清除靠近obstalce的field
                cv2.fillPoly(obstacle_map, [obstacle], color=(1,))

                # 简化的扩展逻辑
                rect = cv2.minAreaRect(obstacle)
                center, (w, h), angle = rect
                expanded_box = cv2.boxPoints((center, 
                                             (w + config.obstacle_expand_pixels, 
                                              h + config.obstacle_expand_pixels), 
                                             angle))
                cv2.fillPoly(field_map, [expanded_box.astype(np.int32)], 0)
                obstacles_placed_count += 1
        return obstacle_map

    def _is_valid_placement(self, obstacle: np.ndarray, agent_position: Tuple[float, float],
                            existing_obstacles: np.ndarray, agent_length: float, config) -> bool:
        """检查障碍物放置是否有效"""
        center = obstacle.mean(axis=0)[0]
        center_int = (int(center[1]), int(center[0]))  # 使用int()进行网格索引（floor操作），获取中心所在的网格

        # 中心位置越界或者中心位置在已有障碍物，则无效
        if (0 <= center_int[0] < existing_obstacles.shape[0] and 0 <= center_int[1] < existing_obstacles.shape[1]
                and existing_obstacles[center_int]):
            return False

        # cv2.pointPolygonTest测点在多边形内部的距离，为负则表示点在多边形外部，要求大于两倍agent_lenth
        agent_x, agent_y = agent_position
        distance_to_agent = - cv2.pointPolygonTest(obstacle, (agent_x, agent_y), True)
        min_distance = config.obstacle_min_distance_to_agent * agent_length

        return distance_to_agent > min_distance

    def _generate_boundary(self, bounding_box: List[np.ndarray],
                           field_map: np.ndarray, config) -> np.ndarray:
        """根据boundary_source生成边界障碍物，"field": field外部为障碍（精确轮廓），"box": bounding box外部为障碍（带扩展） """
        if config.boundary_source == "field":
            return (field_map == 0).astype(np.uint8)
        elif config.boundary_source == "box":
            h, w = field_map.shape
            boundary_map = np.ones((h, w), dtype=np.uint8)
            expanded_box = self._calculate_expanded_box(bounding_box[0], config)
            cv2.fillPoly(boundary_map, [expanded_box], color=(0,))
            return boundary_map
        else:
            return np.zeros_like(field_map)

    def _create_random_obstacle(self, dimensions: Tuple[int, int], config, rng: np.random.Generator) -> np.ndarray:
        """创建随机障碍物"""
        width, height = dimensions
        margin = config.obstacle_min_distance_to_edge

        center_x = rng.uniform(margin, width - margin)
        center_y = rng.uniform(margin, height - margin)

        min_size, max_size = config.obstacle_size_range
        obs_length = rng.uniform(min_size, max_size)
        obs_width = rng.uniform(min_size, max_size)
        angle = rng.uniform(0., 360.)

        # 直接使用元组表示法生成顶点
        box_points = cv2.boxPoints(((center_x, center_y), (obs_length, obs_width), angle))
        return box_points.reshape((-1, 1, 2)).astype(np.int32)

    def _calculate_expanded_box(self, box: np.ndarray, config) -> np.ndarray:
        """计算扩展边界框 - 极简版本"""
        # 直接从顶点获取旋转矩形参数
        center, (width, height), angle = cv2.minAreaRect(box)

        # 按配置扩展尺寸，并生成矩形
        expanded_width = max(width * config.boundary_expand_ratio, width + config.boundary_min_expand_pixels)
        expanded_height = max(height * config.boundary_expand_ratio, height + config.boundary_min_expand_pixels)
        expanded_box = cv2.boxPoints(((center), (expanded_width, expanded_height), angle))

        # 保持原顺序（左上角开始）
        start_idx = expanded_box.sum(axis=1).argmin()
        expanded_box = np.roll(expanded_box, 4 - start_idx, 0)
        return expanded_box.reshape((-1, 1, 2)).astype(np.int32)

    def _scale_obstacle_map_center(self, obstacle_map: np.ndarray, scale: float) -> np.ndarray:
        """以图像中心为原点对obstacle地图做同心等比缩放（保持画布尺寸不变）"""
        h, w = obstacle_map.shape
        center_x, center_y = w / 2.0, h / 2.0

        # 构建以中心为原点的仿射变换矩阵
        transform_matrix = np.array([[scale, 0.0, (1.0 - scale) * center_x],
                                     [0.0, scale, (1.0 - scale) * center_y]], dtype=np.float32)

        # 对二值图使用最近邻插值保持二值特性
        scaled_map = cv2.warpAffine(obstacle_map.astype(np.uint8),transform_matrix,(w, h),
            borderMode=cv2.BORDER_CONSTANT,borderValue=0,flags=cv2.INTER_NEAREST)
        return scaled_map.astype(np.uint8)


class WeedCreator:
    """生成杂草分布"""

    @classmethod
    def get_dependencies(cls) -> List[str]:
        return ['field', 'obstacle']

    def generate(self, state: Dict[str, Any], rng: np.random.Generator) -> None:
        options, config = state['options'], state['config']

        if options.get('scenario_directory'): # 判断是否从目录加载
            weed_map = self._load_from_directory(options['scenario_directory'],
                                                 state['env_state'].get_static_info('dimensions'))
        else:
            weed_map = self._generate_weed_distribution(state['maps_dict']['field'],
                                                        options.get('weed_distribution', 'uniform'),
                                                        options.get('weed_count', 100), rng)
        # 应用障碍物排除
        if config.exclude_weeds_near_obstacles:
            weed_map = self._apply_obstacle_exclusion(weed_map, state['maps_dict']['obstacle'], config)

        state['maps_dict']['weed'] = weed_map
        state['maps_dict']['original_weed'] = state['maps_dict']['weed'].copy()
        state['env_state'].set_static_info('total_weed_count', int(weed_map.sum()))

        if config.weed_noise:
            noisy_weed_map = self._apply_weed_noise(weed_map, rng)
            state['maps_dict']['weed_noisy'] = noisy_weed_map

    def _load_from_directory(self, directory: Union[str, Path], dimensions: Tuple[int, int]) -> np.ndarray:
        """从预制场景目录加载weed地图"""
        weed_map = load_map_from_directory(directory, 'weed')

        # 验证尺寸匹配
        if weed_map.shape != dimensions:
            raise ValueError(f"Weed map dimensions {weed_map.shape} don't match " f"expected dimensions {dimensions}")
        return weed_map

    def _generate_weed_distribution(self, field_map: np.ndarray, distribution: str,
                                    weed_count: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """生成杂草分布"""
        if distribution == "uniform":
            weed_map = self._generate_uniform_distribution(field_map, weed_count, rng)
        elif distribution == "gaussian":
            weed_map = self._generate_gaussian_distribution(field_map, weed_count, rng)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
        return weed_map

    def _generate_uniform_distribution(self, field_map: np.ndarray,
                                       weed_count: int, rng: np.random.Generator) -> np.ndarray:
        """生成均匀分布"""
        weed_map = np.zeros_like(field_map, dtype=np.uint8)
        possible_positions = np.argwhere(field_map)

        if len(possible_positions) == 0: return weed_map

        # 随机选择weed的位置并生成weed_map
        actual_count = min(weed_count, len(possible_positions))
        selected_indices = rng.choice(len(possible_positions), size=actual_count, replace=False)
        selected_positions = possible_positions[selected_indices]
        weed_map[selected_positions[:, 0], selected_positions[:, 1]] = 1
        return weed_map

    def _generate_gaussian_distribution(self, field_map: np.ndarray,
                                        weed_count: int, rng: np.random.Generator) -> np.ndarray:
        """生成高斯分布"""
        weed_map = np.zeros_like(field_map, dtype=np.uint8)
        height, width = field_map.shape

        center_y, center_x = height / 2, width / 2
        scale_y, scale_x = height * 0.35, width * 0.35

        candidates = rng.normal(loc=[center_y, center_x], scale=[scale_y, scale_x], size=(weed_count * 5, 2))

        candidates = np.round(candidates).astype(int)
        candidates = np.clip(candidates, [0, 0], [height - 1, width - 1])
        unique_candidates = np.unique(candidates, axis=0)

        valid_mask = field_map[unique_candidates[:, 0], unique_candidates[:, 1]] == 1
        valid_candidates = unique_candidates[valid_mask]

        actual_count = min(weed_count, len(valid_candidates))
        if actual_count > 0:
            selected_positions = valid_candidates[:actual_count]
            weed_map[selected_positions[:, 0], selected_positions[:, 1]] = 1

        return weed_map

    def _apply_weed_noise(self, weed_map: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """应用杂草噪声"""
        height, width = weed_map.shape
        weed_positions = np.argwhere(weed_map)
        num_weeds = len(weed_positions)

        if num_weeds == 0: return np.zeros_like(weed_map)

        shifts = rng.integers(-1, 2, size=(num_weeds, 2))
        shifted_positions = weed_positions + shifts
        shifted_positions = np.clip(shifted_positions, [0, 0], [height - 1, width - 1])
        unique_positions = np.unique(shifted_positions, axis=0)

        noisy_weed_map = np.zeros_like(weed_map)
        noisy_weed_map[unique_positions[:, 0], unique_positions[:, 1]] = 1

        return noisy_weed_map

    def _apply_obstacle_exclusion(self, weed_map: np.ndarray, obstacle_map: np.ndarray, config) -> np.ndarray:
        """从障碍物附近移除杂草"""
        kernel_size = config.weed_avoid_obstacle_pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated_obstacles = cv2.dilate(obstacle_map, kernel, iterations=1)

        cleaned_weed_map = weed_map.copy()
        cleaned_weed_map[dilated_obstacles > 0] = 0

        return cleaned_weed_map


class TrajectoryCreator:
    """轨迹记录地图组件"""

    @classmethod
    def get_dependencies(cls) -> List[str]:
        return ['field']

    def generate(self, state: Dict[str, Any], rng: np.random.Generator) -> None:
        if not state['config'].use_trajectory:
            return

        width, height = state['env_state'].get_static_info('dimensions')
        state['maps_dict']['trajectory'] = np.zeros((height, width), dtype=np.uint8)


class MistCreator:
    """雾效地图组件（包含可见性调整）"""

    @classmethod
    def get_dependencies(cls) -> List[str]:
        return ['field', 'agent']

    def generate(self, state: Dict[str, Any], rng: np.random.Generator) -> None:
        if not state['config'].use_mist: return

        width, height = state['env_state'].get_static_info('dimensions')
        state['maps_dict']['mist'] = np.zeros((height, width), dtype=np.uint8)

        # 集成可见性调整功能
        if state['config'].ensure_field_visibility:
            self._ensure_field_visibility(state['maps_dict'], state['agent'].position,  # agent当前位置
                state['env_state'].get_static_info('field_contours'))  # 地图边界轮廓

    def _ensure_field_visibility(self, maps_dict: Dict[str, np.ndarray],
                                 agent_position: Tuple[float, float],
                                 field_contours: List[np.ndarray]) -> None:
        """确保田地在agent初始视野中可见"""
        field_in_vision = np.logical_and(maps_dict['field'], maps_dict['mist'])
        if not field_in_vision.any():
            largest_contour = field_contours[0]
            dist_to_field = cv2.pointPolygonTest(largest_contour, agent_position, True)

            radius = math.ceil(abs(dist_to_field) + 5)
            agent_position_discrete = (int(agent_position[0]), int(agent_position[1]))
            cv2.circle(img=maps_dict['mist'], center=agent_position_discrete, radius=radius, color=(1,), thickness=-1)

# HIFCreator已移至envs_new/cpp_env_v5.py作为内部类，提高代码内聚性


class OverlapMapCreator:
    """重复覆盖统计地图组件（overlap map）。

    初始化重复覆盖计数图：
    - 田地区域置为 -1（允许一次无惩罚覆盖）
    - 非田地区域置为 0（理想情况下不应覆盖）
    """

    @classmethod
    def get_dependencies(cls) -> List[str]:
        return ['field']

    def generate(self, state: Dict[str, Any], rng: np.random.Generator) -> None:
        field_map = state['maps_dict']['field']
        h, w = field_map.shape
        overlap_map = np.zeros((h, w), dtype=np.int16)
        overlap_map[field_map == 1] = -1
        state['maps_dict']['overlap'] = overlap_map
