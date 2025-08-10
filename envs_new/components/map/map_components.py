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
    """
    从目录加载指定类型的地图文件
    
    Args:
        directory: 场景目录路径
        map_type: 地图类型 ('frontier', 'obstacle', 'weed')
        
    Returns:
        二值化的地图数组
        
    Raises:
        FileNotFoundError: 地图文件不存在
        ValueError: 加载失败或地图类型无效
    """
    directory = Path(directory)
    
    # 地图类型到文件名的映射
    filename_map = {
        'frontier': 'map_frontier.png',
        'obstacle': 'map_obstacle.png',
        'weed': 'map_weed.png'
    }
    
    if map_type not in filename_map:
        raise ValueError(f"Invalid map type: {map_type}")
    
    file_path = directory / filename_map[map_type]
    
    if not file_path.exists():
        raise FileNotFoundError(f"Map file not found: {file_path}")
    
    # 加载图像
    if map_type == 'frontier':
        # frontier需要彩色图像来检测任何非黑色像素
        image = cv2.imread(str(file_path))
        if image is None:
            raise ValueError(f"Failed to load image: {file_path}")
        # 任何非黑色像素都是农田
        binary_map = (image.sum(axis=-1) > 0).astype(np.uint8)
    else:
        # obstacle和weed使用灰度图
        image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {file_path}")
        # 二值化
        binary_map = (image > 0).astype(np.uint8)
    
    return binary_map


class FrontierCreator:
    """加载frontier地图并建立环境基础"""
    
    @classmethod
    def get_dependencies(cls) -> List[str]:
        return []
    
    def generate(self, state: Dict[str, Any], rng: np.random.Generator) -> None:
        options = state['options']
        config = state['config']
        
        if options.get('scenario_directory'):
            frontier_map, dimensions = self._load_from_directory(
                options['scenario_directory']
            )
        else:
            frontier_map, dimensions = self._load_from_file(
                options.get('map_id'), config, rng
            )
        
        # 提取几何信息
        bounding_box, frontier_contours = self._extract_geometry(frontier_map)
        
        # 设置共享dimensions供其他组件使用
        state['dimensions'] = dimensions
        
        # 唯一权威存储 - 消除冗余
        state['maps_dict']['field_frontier'] = frontier_map
        state['maps_dict']['original_field_frontier'] = frontier_map.copy()
        state['env_state'].set_static_info('dimensions', dimensions)
        state['env_state'].set_static_info('bounding_box', bounding_box)
        state['env_state'].set_static_info('frontier_contours', frontier_contours)
    
    def _load_from_directory(self, directory: Union[str, Path]) -> Tuple[np.ndarray, Tuple[int, int]]:
        """从预制场景目录加载frontier地图"""
        frontier_map = load_map_from_directory(directory, 'frontier')
        dimensions = frontier_map.shape[::-1]  # (width, height)
        return frontier_map, dimensions
    
    def _load_from_file(self, map_id: Optional[int], config, rng: np.random.Generator) -> Tuple[np.ndarray, Tuple[int, int]]:
        """从地图文件加载"""
        map_dir = config.get_absolute_map_dir()
        
        # 获取所有地图文件
        if not map_dir.exists():
            raise FileNotFoundError(f"Map directory does not exist: {map_dir}")
        
        map_files = sorted([f for f in map_dir.iterdir() if f.suffix.lower() == '.png'])
        if not map_files:
            raise ValueError(f"No PNG map files found in directory: {map_dir}")
        
        # 选择地图
        if map_id is None:
            map_id = rng.integers(0, len(map_files))
        
        if not (0 <= map_id < len(map_files)):
            raise ValueError(f"Map ID {map_id} out of range [0, {len(map_files)-1}]")
        
        map_path = map_files[map_id]
        
        # 加载图像
        image = cv2.imread(str(map_path))
        if image is None:
            raise ValueError(f"Failed to load image from: {map_path}")
        
        # 转换为二值frontier地图（任何非黑色像素都是农田）
        frontier_map = (image.sum(axis=-1) > 0).astype(np.uint8)
        dimensions = frontier_map.shape[::-1]  # (width, height)
        
        return frontier_map, dimensions
    
    def _extract_geometry(self, frontier_map: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """提取边界框和轮廓"""
        contours, _ = cv2.findContours(frontier_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("No contours found in frontier map")
        
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_contour = sorted_contours[0]
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        
        start_idx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - start_idx, 0).astype(int)
        box = box.reshape((-1, 1, 2))
        
        return [box], sorted_contours


class AgentCreator:
    """创建并定位Agent"""
    
    @classmethod
    def get_dependencies(cls) -> List[str]:
        return ['frontier']
    
    def generate(self, state: Dict[str, Any], rng: np.random.Generator) -> None:
        options = state['options']
        config = state['config']
        
        # 使用共享的dimensions数据
        bounding_box = state['env_state'].get_static_info('bounding_box')
        dimensions = state['dimensions']
        
        position, direction = self._calculate_pose(
            bounding_box,
            dimensions,
            options.get('initial_position'),
            options.get('initial_direction')
        )
        
        agent = AgentFactory.create_mower_agent(config, position, direction)
        
        # 唯一权威存储 - agent位置通过agent.position获取
        state['agent'] = agent

    def _calculate_pose(self, bounding_box: List[np.ndarray], dimensions: Tuple[int, int],
                       override_position: Optional[Tuple[float, float]],
                       override_direction: Optional[float]) -> Tuple[Tuple[float, float], float]:
        """计算agent初始位置和方向"""
        if override_position is not None and override_direction is not None:
            return override_position, override_direction
        
        if not bounding_box:
            width, height = dimensions
            return (width / 2, height / 2), 0.0
        
        box = bounding_box[0]
        edge1 = box[1, 0] - box[0, 0]
        edge2 = box[2, 0] - box[1, 0]
        
        if math.hypot(*edge1) > math.hypot(*edge2):
            pos_index = 0
            direction_vector = edge1
        else:
            pos_index = 1
            direction_vector = edge2
        
        position = (float(box[pos_index, 0, 0]), float(box[pos_index, 0, 1]))
        
        direction_vector = direction_vector / math.hypot(*direction_vector)
        direction = math.degrees(math.atan2(direction_vector[1], direction_vector[0]))
        
        if override_position is not None:
            position = override_position
        if override_direction is not None:
            direction = override_direction
        
        return position, direction


class ObstacleCreator:
    """生成随机障碍物"""
    
    @classmethod
    def get_dependencies(cls) -> List[str]:
        return ['frontier', 'agent']
    
    def generate(self, state: Dict[str, Any], rng: np.random.Generator) -> None:
        options = state['options']
        config = state['config']
        
        # 判断是否从目录加载
        if options.get('scenario_directory'):
            obstacles = self._load_from_directory(
                options['scenario_directory'], 
                state['dimensions']
            )
        else:
            obstacles = self._generate_obstacles(
                state['dimensions'],  # 使用共享的dimensions
                state['maps_dict']['field_frontier'],
                state['agent'].position,
                state['agent'].length,
                config,
                rng
            )
        
        # 边界处理逻辑更清晰
        if config.use_box_boundary:
            boundary = self._generate_boundary(
                state['dimensions'],
                state['env_state'].get_static_info('bounding_box'),
                config
            )
            obstacles = np.logical_or(obstacles, boundary).astype(np.uint8)

        state['maps_dict']['obstacle'] = obstacles
    
    def _load_from_directory(self, directory: Union[str, Path], dimensions: Tuple[int, int]) -> np.ndarray:
        """从预制场景目录加载obstacle地图"""
        obstacle_map = load_map_from_directory(directory, 'obstacle')
        
        # 验证尺寸匹配
        width, height = dimensions
        if obstacle_map.shape != (height, width):
            raise ValueError(f"Obstacle map dimensions {obstacle_map.shape} don't match "
                           f"expected dimensions {(height, width)}")
        
        return obstacle_map
    
    def _generate_obstacles(self, dimensions: Tuple[int, int], frontier_map: np.ndarray,
                          agent_position: Tuple[float, float], agent_length: float, config, rng: np.random.Generator) -> np.ndarray:
        """生成随机障碍物"""
        width, height = dimensions
        obstacle_map = np.zeros((height, width), dtype=np.uint8)
        
        min_obstacles, max_obstacles = config.num_obstacles_range
        if max_obstacles <= 0:
            return obstacle_map
            
        num_obstacles = rng.integers(min_obstacles, max_obstacles + 1)
        if num_obstacles == 0:
            return obstacle_map
        
        obstacles_placed = 0
        max_attempts = num_obstacles * 10
        
        for _ in range(max_attempts):
            if obstacles_placed >= num_obstacles:
                break
            
            obstacle = self._create_random_obstacle(dimensions, config, rng)
            
            if self._is_valid_placement(obstacle, agent_position, obstacle_map, agent_length):
                cv2.fillPoly(obstacle_map, [obstacle], color=(1,))
                expanded = self._expand_obstacle(obstacle, 15)
                cv2.fillPoly(frontier_map, [expanded], color=(0,))
                obstacles_placed += 1
        
        return obstacle_map
    
    def _generate_boundary(self, dimensions: Tuple[int, int], 
                          bounding_box: List[np.ndarray], config) -> np.ndarray:
        """生成边界障碍物"""
        if not config.use_box_boundary or not bounding_box:
            return np.zeros(dimensions[::-1], dtype=np.uint8)
        
        width, height = dimensions
        boundary_map = np.ones((height, width), dtype=np.uint8)
        
        box = bounding_box[0]
        expanded_box = self._calculate_expanded_box(box)
        cv2.fillPoly(boundary_map, [expanded_box], color=(0,))
        
        return boundary_map
    
    def _create_random_obstacle(self, dimensions: Tuple[int, int], config, rng: np.random.Generator) -> np.ndarray:
        """创建随机障碍物"""
        width, height = dimensions
        margin = 100
        
        center_x = rng.uniform(margin, width - margin)
        center_y = rng.uniform(margin, height - margin)
        
        min_size, max_size = config.obstacle_size_range
        obs_length = rng.uniform(min_size, max_size)
        obs_width = rng.uniform(min_size, max_size)
        angle = rng.uniform(0., 360.)
        
        rotated_rect = cv2.RotatedRect(
            center=(center_x, center_y),
            size=(obs_length, obs_width),
            angle=angle
        )
        
        return np.array(rotated_rect.points(), dtype=np.int32).reshape((-1, 1, 2))
    
    def _is_valid_placement(self, obstacle: np.ndarray, agent_position: Tuple[float, float],
                          existing_obstacles: np.ndarray, agent_length: float) -> bool:
        """检查障碍物放置是否有效"""
        center = obstacle.mean(axis=0)[0]
        center_int = (int(center[1]), int(center[0]))
        
        if (0 <= center_int[0] < existing_obstacles.shape[0] and 
            0 <= center_int[1] < existing_obstacles.shape[1] and
            existing_obstacles[center_int]):
            return False
        
        agent_x, agent_y = agent_position
        distance_to_agent = cv2.pointPolygonTest(obstacle, (agent_x, agent_y), True)
        min_distance = -2.0 * agent_length
        
        return distance_to_agent < min_distance
    
    def _expand_obstacle(self, obstacle: np.ndarray, expansion: float) -> np.ndarray:
        """扩大障碍物"""
        rect = cv2.minAreaRect(obstacle)
        center, (width, height), angle = rect
        
        expanded_rect = cv2.RotatedRect(
            center=center,
            size=(width + expansion, height + expansion),
            angle=angle
        )
        
        return np.array(expanded_rect.points(), dtype=np.int32).reshape((-1, 1, 2))
    
    def _calculate_expanded_box(self, box: np.ndarray) -> np.ndarray:
        """计算扩展边界框"""
        # 扩展旋转矩形：提取边向量→计算旋转角→扩展尺寸→重建矩形
        center = 0.5 * (box[0, 0] + box[2, 0])
        edge1 = box[0, 0] - box[1, 0]
        edge2 = box[1, 0] - box[2, 0]
        
        if abs(edge2[1]) < abs(edge2[0]):
            width_vec, height_vec = edge1, edge2
        else:
            width_vec, height_vec = edge2, edge1
        
        angle = math.atan2(width_vec[1], width_vec[0]) * 180.0 / math.pi
        width = math.hypot(*width_vec)
        height = math.hypot(*height_vec)
        
        expanded_width = max(width * 1.2, width + 60)
        expanded_height = max(height * 1.2, height + 60)
        
        expanded_rect = cv2.RotatedRect(
            center=(center[0], center[1]),
            size=(expanded_width, expanded_height),
            angle=angle
        )
        
        box_points = cv2.boxPoints(expanded_rect)
        start_idx = box_points.sum(axis=1).argmin()
        box_points = np.roll(box_points, 4 - start_idx, 0)
        
        return box_points.reshape((-1, 1, 2)).astype(np.int32)


class WeedCreator:
    """生成杂草分布"""
    
    @classmethod
    def get_dependencies(cls) -> List[str]:
        return ['frontier', 'obstacle']
    
    def generate(self, state: Dict[str, Any], rng: np.random.Generator) -> None:
        options = state['options']
        config = state['config']
        
        # 判断是否从目录加载
        if options.get('scenario_directory'):
            weed_map = self._load_from_directory(
                options['scenario_directory'],
                state['dimensions']
            )
            
            # 从加载的地图创建噪声版本
            noisy_weed_map = self._apply_weed_noise(weed_map, rng)
            
            # 应用障碍物排除（如果配置了）
            if config.weed_noise > 0:
                weed_map = self._apply_obstacle_exclusion(weed_map, state['maps_dict']['obstacle'])
                noisy_weed_map = self._apply_obstacle_exclusion(noisy_weed_map, state['maps_dict']['obstacle'])
            
            state['maps_dict']['weed'] = weed_map
            state['maps_dict']['weed_noisy'] = noisy_weed_map
            total_weed_count = int(weed_map.sum())
        else:
            weed_map, noisy_weed_map = self._generate_weed_distribution(
                state['maps_dict']['field_frontier'],  # 直接使用，无需中间变量
                options.get('weed_distribution', 'uniform'),
                options.get('weed_count', 100),
                rng
            )
            
            # 应用障碍物排除
            if config.exclude_weeds_near_obstacles:
                weed_map = self._apply_obstacle_exclusion(weed_map, state['maps_dict']['obstacle'])
                noisy_weed_map = self._apply_obstacle_exclusion(noisy_weed_map, state['maps_dict']['obstacle'])
            
            state['maps_dict']['weed'] = weed_map
            state['maps_dict']['weed_noisy'] = noisy_weed_map
            total_weed_count = int(weed_map.sum())
        
        # 保存原始杂草地图用于渲染已清除区域（与FrontierCreator保持一致）
        state['maps_dict']['original_weed'] = state['maps_dict']['weed'].copy()
        
        state['env_state'].set_static_info('total_weed_count', total_weed_count)
    
    def _load_from_directory(self, directory: Union[str, Path], dimensions: Tuple[int, int]) -> np.ndarray:
        """从预制场景目录加载weed地图"""
        weed_map = load_map_from_directory(directory, 'weed')
        
        # 验证尺寸匹配
        width, height = dimensions
        if weed_map.shape != (height, width):
            raise ValueError(f"Weed map dimensions {weed_map.shape} don't match "
                           f"expected dimensions {(height, width)}")
        
        return weed_map
    
    def _generate_weed_distribution(self, frontier_map: np.ndarray, distribution: str,
                                  weed_count: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """生成杂草分布"""
        if distribution == "uniform":
            weed_map = self._generate_uniform_distribution(frontier_map, weed_count, rng)
        elif distribution == "gaussian":
            weed_map = self._generate_gaussian_distribution(frontier_map, weed_count, rng)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
        
        # 应用噪声
        noisy_weed_map = self._apply_weed_noise(weed_map, rng)
        
        return weed_map, noisy_weed_map
    
    def _generate_uniform_distribution(self, frontier_map: np.ndarray, 
                                     weed_count: int, rng: np.random.Generator) -> np.ndarray:
        """生成均匀分布"""
        weed_map = np.zeros_like(frontier_map, dtype=np.uint8)
        possible_positions = np.argwhere(frontier_map)
        
        if len(possible_positions) == 0:
            return weed_map
        
        actual_count = min(weed_count, len(possible_positions))
        rng.shuffle(possible_positions)
        selected_positions = possible_positions[:actual_count]
        
        weed_map[selected_positions[:, 0], selected_positions[:, 1]] = 1
        return weed_map
    
    def _generate_gaussian_distribution(self, frontier_map: np.ndarray,
                                      weed_count: int, rng: np.random.Generator) -> np.ndarray:
        """生成高斯分布"""
        weed_map = np.zeros_like(frontier_map, dtype=np.uint8)
        height, width = frontier_map.shape
        
        center_y, center_x = height / 2, width / 2
        scale_y, scale_x = height * 0.35, width * 0.35
        
        candidates = rng.normal(
            loc=[center_y, center_x],
            scale=[scale_y, scale_x],
            size=(weed_count * 5, 2)
        )
        
        candidates = np.round(candidates).astype(int)
        candidates = np.clip(candidates, [0, 0], [height - 1, width - 1])
        unique_candidates = np.unique(candidates, axis=0)
        
        valid_mask = frontier_map[unique_candidates[:, 0], unique_candidates[:, 1]] == 1
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
        
        if num_weeds == 0:
            return np.zeros_like(weed_map)
        
        shifts = rng.integers(-1, 2, size=(num_weeds, 2))
        shifted_positions = weed_positions + shifts
        shifted_positions = np.clip(shifted_positions, [0, 0], [height - 1, width - 1])
        unique_positions = np.unique(shifted_positions, axis=0)
        
        noisy_weed_map = np.zeros_like(weed_map)
        noisy_weed_map[unique_positions[:, 0], unique_positions[:, 1]] = 1
        
        return noisy_weed_map
    
    def _apply_obstacle_exclusion(self, weed_map: np.ndarray, obstacle_map: np.ndarray) -> np.ndarray:
        """从障碍物附近移除杂草"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29))
        dilated_obstacles = cv2.dilate(obstacle_map, kernel, iterations=1)
        
        cleaned_weed_map = weed_map.copy()
        cleaned_weed_map[dilated_obstacles > 0] = 0
        
        return cleaned_weed_map


class TrajectoryCreator:
    """轨迹记录地图组件"""
    
    @classmethod
    def get_dependencies(cls) -> List[str]:
        return ['frontier']
    
    def generate(self, state: Dict[str, Any], rng: np.random.Generator) -> None:
        if not state['config'].use_traj:
            return
            
        width, height = state['dimensions']
        state['maps_dict']['trajectory'] = np.zeros((height, width), dtype=np.uint8)


class MistCreator:
    """雾效地图组件（包含可见性调整）"""
    
    @classmethod
    def get_dependencies(cls) -> List[str]:
        return ['frontier', 'agent']
    
    def generate(self, state: Dict[str, Any], rng: np.random.Generator) -> None:
        if not state['config'].use_mist:
            return
            
        width, height = state['dimensions']
        state['maps_dict']['mist'] = np.ones((height, width), dtype=np.uint8)
        
        # 集成可见性调整功能
        if state['config'].ensure_frontier_visibility:
            self._ensure_frontier_visibility(
                state['maps_dict'],
                state['agent'].position,  # agent当前位置
                state['env_state'].get_static_info('frontier_contours')  # 地图边界轮廓
            )
    
    def _ensure_frontier_visibility(self, maps_dict: Dict[str, np.ndarray], 
                                   agent_position: Tuple[float, float],
                                   frontier_contours: List[np.ndarray]) -> None:
        """确保frontier在agent初始视野中可见"""
        frontier_in_vision = np.logical_and(maps_dict['field_frontier'], maps_dict['mist'])
        if not frontier_in_vision.any():
            largest_contour = frontier_contours[0]
            dist_to_frontier = cv2.pointPolygonTest(largest_contour, agent_position, True)
            
            radius = math.ceil(abs(dist_to_frontier) + 5)
            agent_position_discrete = (int(round(agent_position[0])), int(round(agent_position[1])))
            cv2.circle(
                img=maps_dict['mist'],
                center=agent_position_discrete,
                radius=radius,
                color=(1,),
                thickness=-1
            )