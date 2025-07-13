# refine_real_world_optimized_fixed_v2.py
"""
2025-07-12
v1-v2-v3: 独立的Refined视频渲染系统, tracking_data.json实现像素级渐进式标注显示和交互调整功能
v4: 包含所有编辑功能、视频保存功能和相机运动插值矫正
v5: 包含 正确恢复探索地图状态, 预计算缓存系统、快速渲染和编辑功能
合并三个版本共称原版
"""

import cv2
import numpy as np
import json
import time
import os
from enum import Enum
from collections import deque
import math
from datetime import datetime
import copy


class RefineState(Enum):
    """Refined播放器状态"""
    LOADING = 0
    PLAYING = 1
    PAUSED = 2
    EDITING_ANNOTATION = 3
    EDITING_ORIENTATION = 4
    EDITING_CAMERA = 5
    EDITING_CAMERA_INTERPOLATION = 6
    PREPROCESSING = 7  # 预处理状态


class CacheManager:
    """缓存管理器 - 负责预计算和缓存探索地图、标注数据和覆盖信息"""

    def __init__(self, data, video_info, map_scale=2.0, skip_coverage_cache=False):
        self.data = data
        self.video_info = video_info
        self.map_scale = map_scale
        self.skip_coverage_cache = skip_coverage_cache  # 是否跳过覆盖缓存
        self.trajectory_data = data['trajectory_summary']
        self.annotations = data['annotations']
        self.render_config = data['render_config']
        self.camera_motion = data.get('camera_motion', {})

        # 重建数据结构
        self.trajectory_positions = [np.array(pos) for pos in self.trajectory_data['trajectory_positions']]
        self.direction_angles = self.trajectory_data['direction_angles']
        self.polygons = [[[float(p[0]), float(p[1])] for p in poly['points']]
                         for poly in self.annotations['polygons']]
        self.circles = [(tuple(circle['center']), circle['radius'])
                        for circle in self.annotations['circles']]
        self.transform_matrices = [np.array(transform) for transform in
                                   self.camera_motion.get('transform_matrices', [])]

        # 缓存路径
        video_dir = os.path.dirname(video_info['path'])
        scale_suffix = f"_scale{int(map_scale * 10)}" if map_scale != 2.0 else ""

        # 使用JSON中指定的文件名（如果有的话）
        self.explore_cache_path = os.path.join(video_dir,
                                               video_info.get('explore_map_cache',
                                                              f'explore_map_cache{scale_suffix}.mp4'))
        self.annotation_cache_path = os.path.join(video_dir,
                                                  video_info.get('annotation_cache',
                                                                 f'annotation_cache{scale_suffix}.npz'))
        self.coverage_cache_path = os.path.join(video_dir,
                                                video_info.get('coverage_cache',
                                                               f'coverage_cache{scale_suffix}.npz'))

        # 探索地图管理器
        self.exploration_manager = None
        self._init_exploration_manager()

        # 探索地图缓存读取器
        self.explore_reader = None

    def _init_exploration_manager(self):
        """初始化探索地图管理器"""
        frame_size = (self.video_info['width'], self.video_info['height'])
        self.exploration_manager = OptimizedExplorationManager(
            self.trajectory_data['bounds'],
            frame_size,
            self.render_config['sector_radius'],
            map_scale=self.map_scale
        )

        # 添加所有标注到探索地图
        for polygon in self.polygons:
            self.exploration_manager.add_annotation_mask('polygon', polygon)

        for center, radius in self.circles:
            self.exploration_manager.add_annotation_mask('circle', center, radius)

    def need_cache_rebuild(self):
        """检查是否需要重建缓存"""
        # 检查缓存文件是否存在
        if not os.path.exists(self.explore_cache_path) or not os.path.exists(self.annotation_cache_path):
            print("缓存文件不存在，需要重建")
            return True

        # 检查覆盖缓存（如果不跳过）
        if not self.skip_coverage_cache and not os.path.exists(self.coverage_cache_path):
            print("覆盖缓存不存在，需要重建")
            return True

        # 检查缓存的scale是否匹配
        if os.path.exists(self.annotation_cache_path):
            try:
                cache_data = np.load(self.annotation_cache_path, allow_pickle=True)
                cached_scale = float(cache_data.get('map_scale', 2.0))
                if abs(cached_scale - self.map_scale) > 0.01:
                    print(f"缓存的map_scale({cached_scale})与当前设置({self.map_scale})不匹配，需要重建缓存")
                    return True
            except Exception as e:
                print(f"读取缓存文件时出错: {e}")
                return True

        # 检查收割参数是否改变（如果不跳过覆盖缓存）
        if not self.skip_coverage_cache and os.path.exists(self.coverage_cache_path):
            try:
                coverage_data = np.load(self.coverage_cache_path, allow_pickle=True)
                cached_height = float(coverage_data.get('weeding_height', 0))
                cached_width = float(coverage_data.get('weeding_width', 0))
                current_height = self.render_config.get('weeding_height', 20)
                current_width = self.render_config.get('weeding_width', 100)

                if abs(cached_height - current_height) > 0.1 or abs(cached_width - current_width) > 0.1:
                    print(f"收割参数已改变，需要重建覆盖缓存")
                    return True

            except Exception as e:
                print(f"读取覆盖缓存时出错: {e}")
                return True

        # 检查缓存文件的时间戳是否比JSON文件更新
        json_path = getattr(self, 'json_path', None)
        if json_path and os.path.exists(json_path):
            json_mtime = os.path.getmtime(json_path)
            explore_mtime = os.path.getmtime(self.explore_cache_path)
            annotation_mtime = os.path.getmtime(self.annotation_cache_path)

            if json_mtime > explore_mtime or json_mtime > annotation_mtime:
                print("JSON文件比缓存文件更新，需要重建缓存")
                return True

        print("缓存文件有效，无需重建")
        return False

    def _precompute_transform_scales(self):
        """预计算每帧变换矩阵的缩放因子"""
        transform_scales = []
        for transform in self.transform_matrices:
            try:
                scale_x = np.sqrt(transform[0, 0] ** 2 + transform[1, 0] ** 2)
                scale_y = np.sqrt(transform[0, 1] ** 2 + transform[1, 1] ** 2)
                avg_scale = (scale_x + scale_y) / 2.0
                transform_scales.append(avg_scale)
            except:
                transform_scales.append(1.0)
        return transform_scales

    def build_explore_cache(self, progress_callback=None):
        """构建探索地图缓存"""
        print("正在构建探索地图缓存...")

        total_frames = len(self.trajectory_positions)
        transform_scales = self._precompute_transform_scales()

        # 尝试使用不同的编码器
        map_height, map_width = self.exploration_manager.map_size[1], self.exploration_manager.map_size[0]

        codecs = [
            ('HEVC', 'H265'),  # H.265/HEVC
            ('hvc1', 'H265'),  # 另一种H265格式
            ('H264', 'H264'),  # H.264/AVC
            ('mp4v', 'MPEG4')  # MPEG-4
        ]

        writer = None
        used_codec = None

        for codec, codec_name in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(
                    self.explore_cache_path,
                    fourcc,
                    30.0,  # 固定帧率，用于缓存
                    (map_width, map_height),
                    isColor=False  # 灰度图
                )
                if writer.isOpened():
                    used_codec = codec_name
                    print(f"使用{codec_name}编码器创建探索地图缓存")
                    break
                else:
                    writer.release()
                    writer = None
            except:
                continue

        if writer is None or not writer.isOpened():
            raise RuntimeError(f"无法创建探索地图缓存文件: {self.explore_cache_path}")

        try:
            # 重置探索地图
            self.exploration_manager.exploration_map.fill(0)

            for frame_idx in range(total_frames):
                if frame_idx < len(self.direction_angles):
                    # 更新探索地图
                    robot_pos = self.trajectory_positions[frame_idx]
                    robot_angle = self.direction_angles[frame_idx]

                    scale_factor = transform_scales[frame_idx] if frame_idx < len(transform_scales) else 1.0
                    sector_radius_global = self.render_config['sector_radius'] * scale_factor

                    self.exploration_manager.update_exploration(
                        (float(robot_pos[0]), float(robot_pos[1])),
                        float(robot_angle),
                        sector_radius_global,
                        np.radians(self.render_config['sector_angle'])
                    )

                # 保存当前探索地图状态
                current_map = self.exploration_manager.exploration_map.copy()
                writer.write(current_map)

                # 进度回调
                if progress_callback and frame_idx % 50 == 0:
                    progress_callback(f"探索地图缓存: {frame_idx}/{total_frames}")

        finally:
            writer.release()

        print(f"探索地图缓存已保存: {self.explore_cache_path} (使用{used_codec}编码)")

    def build_annotation_cache(self, progress_callback=None):
        """构建标注掩码缓存"""
        print("正在构建标注掩码缓存...")

        total_frames = len(self.trajectory_positions)
        num_polygons = len(self.polygons)
        num_circles = len(self.circles)

        # 准备存储结构
        polygon_masks = []  # 每帧每个多边形的掩码数据
        circle_masks = []  # 每帧每个圆形的掩码数据

        # 读取探索地图缓存
        explore_reader = cv2.VideoCapture(self.explore_cache_path)
        if not explore_reader.isOpened():
            raise RuntimeError(f"无法读取探索地图缓存: {self.explore_cache_path}")

        try:
            for frame_idx in range(total_frames):
                ret, explore_map = explore_reader.read()
                if not ret:
                    # 如果读取失败，使用空的探索地图
                    explore_map = np.zeros((self.exploration_manager.map_size[1],
                                            self.exploration_manager.map_size[0]), dtype=np.uint8)
                else:
                    # 转换为灰度图
                    if len(explore_map.shape) == 3:
                        explore_map = cv2.cvtColor(explore_map, cv2.COLOR_BGR2GRAY)

                # 更新探索管理器的地图状态
                self.exploration_manager.exploration_map = explore_map

                # 获取当前帧各标注的可见掩码
                frame_polygon_masks = []
                for poly_idx in range(num_polygons):
                    roi_data = self.exploration_manager.get_explored_annotation_pixels('polygon', poly_idx)
                    if roi_data is not None and np.sum(roi_data['mask']) > 0:
                        # 存储ROI数据：掩码、偏移和尺寸
                        mask_data = {
                            'mask': roi_data['mask'].astype(np.uint8),
                            'offset': roi_data['offset'],
                            'size': roi_data['size']
                        }
                        frame_polygon_masks.append(mask_data)
                    else:
                        frame_polygon_masks.append(None)

                frame_circle_masks = []
                for circle_idx in range(num_circles):
                    roi_data = self.exploration_manager.get_explored_annotation_pixels('circle', circle_idx)
                    if roi_data is not None and np.sum(roi_data['mask']) > 0:
                        mask_data = {
                            'mask': roi_data['mask'].astype(np.uint8),
                            'offset': roi_data['offset'],
                            'size': roi_data['size']
                        }
                        frame_circle_masks.append(mask_data)
                    else:
                        frame_circle_masks.append(None)

                polygon_masks.append(frame_polygon_masks)
                circle_masks.append(frame_circle_masks)

                # 进度回调
                if progress_callback and frame_idx % 50 == 0:
                    progress_callback(f"标注掩码缓存: {frame_idx}/{total_frames}")

        finally:
            explore_reader.release()

        # 保存到npz文件
        save_data = {
            'polygon_masks': polygon_masks,
            'circle_masks': circle_masks,
            'num_frames': total_frames,
            'num_polygons': num_polygons,
            'num_circles': num_circles,
            'map_bounds': self.exploration_manager.map_bounds,
            'map_scale': self.exploration_manager.map_scale
        }

        np.savez_compressed(self.annotation_cache_path, **save_data)
        print(f"标注掩码缓存已保存: {self.annotation_cache_path}")

    def build_coverage_cache(self, progress_callback=None):
        """构建覆盖信息缓存"""
        if self.skip_coverage_cache:
            print("跳过覆盖缓存构建")
            return

        print("正在构建覆盖信息缓存...")

        total_frames = len(self.trajectory_positions)
        num_circles = len(self.circles)

        # 获取收割参数
        weeding_height = self.render_config.get('weeding_height', 20)
        weeding_width = self.render_config.get('weeding_width', 100)

        # 存储每个圆形第一次被覆盖的帧索引
        circle_coverage_frames = [-1] * num_circles  # -1表示未被覆盖

        for frame_idx in range(total_frames):
            if frame_idx >= len(self.direction_angles):
                continue

            # 获取当前位置和朝向
            robot_pos = self.trajectory_positions[frame_idx]
            robot_angle = self.direction_angles[frame_idx]

            # 计算收割矩形的位置（在全局坐标系中）
            # 矩形在箭头末端，垂直于朝向
            weed_react_distance_to_robot = self.render_config['weed_react_distance_to_robot']
            rect_center_x = robot_pos[0] + weed_react_distance_to_robot * np.cos(robot_angle)
            rect_center_y = robot_pos[1] + weed_react_distance_to_robot * np.sin(robot_angle)

            # 检查每个圆形是否与矩形相交
            for circle_idx, (circle_center, circle_radius) in enumerate(self.circles):
                # 如果已经被覆盖，跳过
                if circle_coverage_frames[circle_idx] != -1:
                    continue

                # 计算矩形与圆形是否相交
                if self._check_rect_circle_intersection(
                        (rect_center_x, rect_center_y), robot_angle,
                        weeding_width, weeding_height,
                        circle_center, circle_radius):
                    circle_coverage_frames[circle_idx] = frame_idx

            # 进度回调
            if progress_callback and frame_idx % 50 == 0:
                progress_callback(f"覆盖信息缓存: {frame_idx}/{total_frames}")

        # 保存覆盖信息
        save_data = {
            'circle_coverage_frames': circle_coverage_frames,
            'num_circles': num_circles,
            'weeding_height': weeding_height,
            'weeding_width': weeding_width
        }

        np.savez_compressed(self.coverage_cache_path, **save_data)
        print(f"覆盖信息缓存已保存: {self.coverage_cache_path}")

    def _check_rect_circle_intersection(self, rect_center, rect_angle, rect_width, rect_height,
                                        circle_center, circle_radius):
        """检查矩形与圆形是否相交（使用全局坐标）"""
        # 将圆心转换到矩形的局部坐标系
        dx = circle_center[0] - rect_center[0]
        dy = circle_center[1] - rect_center[1]

        # 旋转到矩形的局部坐标系（矩形是垂直于朝向的）
        perpendicular_angle = rect_angle + np.pi / 2
        cos_angle = np.cos(-perpendicular_angle)
        sin_angle = np.sin(-perpendicular_angle)

        local_x = dx * cos_angle - dy * sin_angle
        local_y = dx * sin_angle + dy * cos_angle

        # 找到矩形上最近的点
        closest_x = np.clip(local_x, -rect_width / 2, rect_width / 2)
        closest_y = np.clip(local_y, -rect_height / 2, rect_height / 2)

        # 计算最近点到圆心的距离
        dist_x = local_x - closest_x
        dist_y = local_y - closest_y
        distance = np.sqrt(dist_x ** 2 + dist_y ** 2)

        # 如果距离小于半径，则相交
        return distance <= circle_radius + 5

    def build_all_caches(self, progress_callback=None):
        """构建所有缓存"""
        self.build_explore_cache(progress_callback)
        self.build_annotation_cache(progress_callback)
        if not self.skip_coverage_cache:
            self.build_coverage_cache(progress_callback)

    def load_cached_annotation_data(self, frame_idx):
        """加载指定帧的标注掩码数据"""
        if not hasattr(self, '_annotation_cache'):
            # 懒加载标注缓存
            if os.path.exists(self.annotation_cache_path):
                cache_data = np.load(self.annotation_cache_path, allow_pickle=True)
                # 转换为标准字典以便访问
                self._annotation_cache = {}
                for key in cache_data.files:
                    data = cache_data[key]
                    # 如果是0维数组（标量），提取其值
                    if isinstance(data, np.ndarray) and data.ndim == 0:
                        data = data.item()
                    self._annotation_cache[key] = data

                # 确保map_scale是float类型
                if 'map_scale' in self._annotation_cache:
                    self._annotation_cache['map_scale'] = float(self._annotation_cache['map_scale'])

                # 确保map_bounds是字典
                if 'map_bounds' in self._annotation_cache and isinstance(self._annotation_cache['map_bounds'],
                                                                         np.ndarray):
                    # 如果是numpy数组，转换回字典
                    bounds_item = self._annotation_cache['map_bounds'].item()
                    if isinstance(bounds_item, dict):
                        self._annotation_cache['map_bounds'] = bounds_item
            else:
                return [], []

        if frame_idx >= self._annotation_cache['num_frames']:
            return [], []

        polygon_masks = self._annotation_cache['polygon_masks'][frame_idx]
        circle_masks = self._annotation_cache['circle_masks'][frame_idx]

        return polygon_masks, circle_masks

    def load_coverage_data(self):
        """加载覆盖信息"""
        if self.skip_coverage_cache:
            return None

        if not hasattr(self, '_coverage_cache'):
            if os.path.exists(self.coverage_cache_path):
                cache_data = np.load(self.coverage_cache_path, allow_pickle=True)
                self._coverage_cache = {
                    'circle_coverage_frames': cache_data['circle_coverage_frames'].tolist(),
                    'num_circles': int(cache_data['num_circles'])
                }
            else:
                return None

        return self._coverage_cache

    def open_explore_reader(self):
        """打开探索地图缓存读取器"""
        if self.explore_reader is None:
            self.explore_reader = cv2.VideoCapture(self.explore_cache_path)
            if not self.explore_reader.isOpened():
                raise RuntimeError(f"无法打开探索地图缓存: {self.explore_cache_path}")

    def close_explore_reader(self):
        """关闭探索地图缓存读取器"""
        if self.explore_reader is not None:
            self.explore_reader.release()
            self.explore_reader = None

    def load_exploration_map_for_frame(self, frame_idx):
        """加载指定帧的探索地图状态"""
        if self.explore_reader is None:
            self.open_explore_reader()

        # 设置到指定帧
        self.explore_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, explore_map = self.explore_reader.read()

        if ret:
            # 转换为灰度图
            if len(explore_map.shape) == 3:
                explore_map = cv2.cvtColor(explore_map, cv2.COLOR_BGR2GRAY)

            # 更新exploration_manager的地图状态
            self.exploration_manager.exploration_map = explore_map
            self.exploration_manager._roi_cache.clear()  # 清除ROI缓存

            return True
        else:
            print(f"警告：无法读取帧{frame_idx}的探索地图")
            return False


class OptimizedExplorationManager:
    """优化的探索地图管理器"""

    def __init__(self, trajectory_bounds, frame_size, base_sector_radius, map_scale=2.0):
        self.frame_width, self.frame_height = frame_size
        self.base_sector_radius = base_sector_radius
        self.map_scale = map_scale

        self.map_bounds, self.map_size = self._calculate_optimal_map_size(
            trajectory_bounds, frame_size, base_sector_radius
        )

        print(f"探索地图分辨率缩放: {self.map_scale}x")
        print(f"优化后的探索地图尺寸: {self.map_size}")
        print(f"地图边界: {self.map_bounds}")

        self.exploration_map = np.zeros((self.map_size[1], self.map_size[0]), dtype=np.uint8)

        self.annotation_masks = {
            'polygons': [],
            'circles': []
        }

        self.last_sector_params = None
        self._roi_cache = {}

    def _calculate_optimal_map_size(self, bounds, frame_size, sector_radius):
        buffer = sector_radius * 1.5

        map_bounds = {
            'min_x': float(bounds['min_x'] - buffer),
            'min_y': float(bounds['min_y'] - buffer),
            'max_x': float(bounds['max_x'] + buffer),
            'max_y': float(bounds['max_y'] + buffer)
        }

        map_width = int((map_bounds['max_x'] - map_bounds['min_x']) * self.map_scale)
        map_height = int((map_bounds['max_y'] - map_bounds['min_y']) * self.map_scale)

        map_width = max(map_width, int(frame_size[0] * self.map_scale))
        map_height = max(map_height, int(frame_size[1] * self.map_scale))

        return map_bounds, (map_width, map_height)

    def global_to_map_coords(self, global_x, global_y):
        map_x = int(round((float(global_x) - self.map_bounds['min_x']) * self.map_scale))
        map_y = int(round((float(global_y) - self.map_bounds['min_y']) * self.map_scale))

        map_x = max(0, min(map_x, self.map_size[0] - 1))
        map_y = max(0, min(map_y, self.map_size[1] - 1))

        return map_x, map_y

    def update_exploration(self, robot_position_global, robot_facing, sector_radius_global, sector_angle):
        center_map_x, center_map_y = self.global_to_map_coords(
            robot_position_global[0], robot_position_global[1]
        )

        radius_in_map = int(sector_radius_global * self.map_scale)

        half_angle = sector_angle / 2
        start_angle = robot_facing - half_angle
        end_angle = robot_facing + half_angle
        start_deg = np.degrees(start_angle)
        end_deg = np.degrees(end_angle)

        temp_map = np.zeros_like(self.exploration_map)

        if start_deg > end_deg:
            cv2.ellipse(temp_map, (center_map_x, center_map_y),
                        (radius_in_map, radius_in_map), 0, start_deg, 360, 255, -1)
            cv2.ellipse(temp_map, (center_map_x, center_map_y),
                        (radius_in_map, radius_in_map), 0, 0, end_deg, 255, -1)
        else:
            cv2.ellipse(temp_map, (center_map_x, center_map_y),
                        (radius_in_map, radius_in_map), 0, start_deg, end_deg, 255, -1)

        self.exploration_map = cv2.bitwise_or(self.exploration_map, temp_map)
        self._roi_cache.clear()

    def add_annotation_mask(self, annotation_type, points_or_center, radius=None):
        mask = np.zeros((self.map_size[1], self.map_size[0]), dtype=np.uint8)

        if annotation_type == 'polygon':
            map_points = []
            for px, py in points_or_center:
                mx, my = self.global_to_map_coords(px, py)
                map_points.append([mx, my])

            if len(map_points) >= 3:
                map_points = np.array(map_points, dtype=np.int32)
                cv2.fillPoly(mask, [map_points], 255)

                x_coords = map_points[:, 0]
                y_coords = map_points[:, 1]
                bbox = (np.min(x_coords), np.min(y_coords),
                        np.max(x_coords), np.max(y_coords))

                self.annotation_masks['polygons'].append((mask, bbox))
                return len(self.annotation_masks['polygons']) - 1

        elif annotation_type == 'circle':
            center_map_x, center_map_y = self.global_to_map_coords(
                points_or_center[0], points_or_center[1]
            )

            radius_in_map = max(1, int(radius * self.map_scale))

            cv2.circle(mask, (center_map_x, center_map_y), radius_in_map, 255, -1)

            bbox = (center_map_x - radius_in_map, center_map_y - radius_in_map,
                    center_map_x + radius_in_map, center_map_y + radius_in_map)

            self.annotation_masks['circles'].append((mask, bbox))
            return len(self.annotation_masks['circles']) - 1

        return -1

    def get_explored_annotation_pixels(self, annotation_type, annotation_idx):
        cache_key = f"{annotation_type}_{annotation_idx}"

        if cache_key in self._roi_cache:
            return self._roi_cache[cache_key]

        try:
            if annotation_type == 'polygon':
                if annotation_idx >= len(self.annotation_masks['polygons']):
                    return None
                annotation_mask, bbox = self.annotation_masks['polygons'][annotation_idx]
            elif annotation_type == 'circle':
                if annotation_idx >= len(self.annotation_masks['circles']):
                    return None
                annotation_mask, bbox = self.annotation_masks['circles'][annotation_idx]
            else:
                return None

            x_min = max(0, int(bbox[0]))
            y_min = max(0, int(bbox[1]))
            x_max = min(self.map_size[0], int(bbox[2]) + 1)
            y_max = min(self.map_size[1], int(bbox[3]) + 1)

            if x_max <= x_min or y_max <= y_min:
                return None

            roi_annotation = annotation_mask[y_min:y_max, x_min:x_max]
            roi_exploration = self.exploration_map[y_min:y_max, x_min:x_max]
            roi_result = cv2.bitwise_and(roi_annotation, roi_exploration)

            result = {
                'mask': roi_result,
                'offset': (x_min, y_min),
                'size': (x_max - x_min, y_max - y_min)
            }

            self._roi_cache[cache_key] = result
            return result

        except Exception as e:
            print(f"获取已探索像素失败: {e}")
            return None


class VideoSaver:
    """视频保存管理器"""

    def __init__(self, video_info, render_config):
        self.video_info = video_info
        self.render_config = render_config
        self.is_saving = False
        self.video_writer = None
        self.output_path = None

    def start_saving(self, prefix="refined"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.mp4"

        video_dir = os.path.dirname(self.video_info['path'])
        self.output_path = os.path.join(video_dir, filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.video_info['fps'],
            (self.video_info['width'], self.video_info['height'])
        )

        if self.video_writer.isOpened():
            self.is_saving = True
            print(f"开始保存视频到: {self.output_path}")
            return True
        else:
            print("创建视频写入器失败")
            return False

    def save_frame(self, frame):
        if self.is_saving and self.video_writer:
            self.video_writer.write(frame)

    def stop_saving(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.is_saving = False
            print(f"视频已保存到: {self.output_path}")


class InteractiveEditor:
    """交互式编辑器"""

    def __init__(self, player):
        self.player = player
        self.dragging_annotation = None
        self.drag_offset = (0, 0)
        self.orientation_edit_step = 5
        self.camera_edit_step = 5
        self.camera_angle_step = 1

        self.camera_calibration_history = {}
        self.last_calibrated_frame = None

    def handle_mouse_event(self, event, x, y, flags, param):
        if self.player.current_state == RefineState.EDITING_ANNOTATION:
            self._handle_annotation_drag(event, x, y, flags)

    def _handle_annotation_drag(self, event, x, y, flags):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_annotation = self._find_annotation_at_point(x, y)
            if clicked_annotation:
                self.dragging_annotation = clicked_annotation
                ann_type, ann_idx = clicked_annotation
                if ann_type == 'polygon':
                    center = self._get_polygon_center(ann_idx)
                else:
                    center = self._get_circle_center(ann_idx)
                self.drag_offset = (x - center[0], y - center[1])
                print(f"开始拖拽 {ann_type} {ann_idx}")

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging_annotation:
            self._update_annotation_position(self.dragging_annotation, x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging_annotation:
                print(f"完成拖拽 {self.dragging_annotation}")
                self.dragging_annotation = None
                self.drag_offset = (0, 0)

    def _find_annotation_at_point(self, x, y):
        global_pos = self.player._screen_to_global_coords(x, y)
        if global_pos is None:
            return None

        for i, (center, radius) in enumerate(self.player.circles):
            distance = np.sqrt((global_pos[0] - center[0]) ** 2 + (global_pos[1] - center[1]) ** 2)
            if distance <= radius:
                return ('circle', i)

        for i, polygon in enumerate(self.player.polygons):
            poly_array = np.array(polygon, dtype=np.float32)
            if cv2.pointPolygonTest(poly_array, (float(global_pos[0]), float(global_pos[1])), False) >= 0:
                return ('polygon', i)

        return None

    def _get_polygon_center(self, poly_idx):
        polygon = self.player.polygons[poly_idx]
        center_global = np.mean(polygon, axis=0)
        return self.player._global_to_screen_coords(center_global)

    def _get_circle_center(self, circle_idx):
        center_global, _ = self.player.circles[circle_idx]
        return self.player._global_to_screen_coords(center_global)

    def _update_annotation_position(self, annotation_info, screen_x, screen_y):
        ann_type, ann_idx = annotation_info

        adjusted_x = screen_x - self.drag_offset[0]
        adjusted_y = screen_y - self.drag_offset[1]
        new_global_pos = self.player._screen_to_global_coords(adjusted_x, adjusted_y)

        if new_global_pos is None:
            return

        if ann_type == 'polygon':
            old_center = np.mean(self.player.polygons[ann_idx], axis=0)
            offset = np.array(new_global_pos) - old_center

            for i in range(len(self.player.polygons[ann_idx])):
                self.player.polygons[ann_idx][i] = [
                    self.player.polygons[ann_idx][i][0] + offset[0],
                    self.player.polygons[ann_idx][i][1] + offset[1]
                ]

        elif ann_type == 'circle':
            old_center, radius = self.player.circles[ann_idx]
            self.player.circles[ann_idx] = (new_global_pos, radius)

    def handle_orientation_editing(self, key):
        if self.player.current_frame_idx >= len(self.player.direction_angles):
            return

        step_rad = np.radians(self.orientation_edit_step)

        if key == ord('a') or key == 81:
            self.player.direction_angles[self.player.current_frame_idx] -= step_rad
        elif key == ord('d') or key == 83:
            self.player.direction_angles[self.player.current_frame_idx] += step_rad
        elif key == 82:
            self.orientation_edit_step = min(self.orientation_edit_step + 1, 45)
            print(f"朝向编辑步长: {self.orientation_edit_step}°")
        elif key == 84:
            self.orientation_edit_step = max(self.orientation_edit_step - 1, 1)
            print(f"朝向编辑步长: {self.orientation_edit_step}°")
        elif key == 13:
            print(f"朝向已更新为: {np.degrees(self.player.direction_angles[self.player.current_frame_idx]):.1f}°")
            self.player.save_current_data()
            self.player.current_state = RefineState.PAUSED
        elif key == 27:
            self.player.direction_angles[self.player.current_frame_idx] = self.player._original_angle
            self.player.current_state = RefineState.PAUSED
            print("朝向编辑已取消")

    def _extract_transform_offset(self, original_transform, calibrated_transform):
        dx = calibrated_transform[0, 2] - original_transform[0, 2]
        dy = calibrated_transform[1, 2] - original_transform[1, 2]

        original_angle = np.arctan2(original_transform[1, 0], original_transform[0, 0])
        calibrated_angle = np.arctan2(calibrated_transform[1, 0], calibrated_transform[0, 0])
        dangle = calibrated_angle - original_angle

        while dangle > np.pi:
            dangle -= 2 * np.pi
        while dangle < -np.pi:
            dangle += 2 * np.pi

        return (dx, dy, dangle)

    def _apply_transform_offset(self, transform, offset):
        dx, dy, dangle = offset

        new_transform = transform.copy()

        new_transform[0, 2] += dx
        new_transform[1, 2] += dy

        if abs(dangle) > 1e-6:
            current_angle = np.arctan2(transform[1, 0], transform[0, 0])
            new_angle = current_angle + dangle

            cos_angle = np.cos(new_angle)
            sin_angle = np.sin(new_angle)

            scale = np.sqrt(transform[0, 0] ** 2 + transform[1, 0] ** 2)
            new_transform[0, 0] = cos_angle * scale
            new_transform[0, 1] = -sin_angle * scale
            new_transform[1, 0] = sin_angle * scale
            new_transform[1, 1] = cos_angle * scale

        return new_transform

    def handle_camera_editing(self, key):
        if self.player.current_frame_idx >= len(self.player.transform_matrices):
            return

        transform = self.player.transform_matrices[self.player.current_frame_idx].copy()

        if key == ord('i'):
            transform[1, 2] -= self.camera_edit_step
        elif key == ord('k'):
            transform[1, 2] += self.camera_edit_step
        elif key == ord('j'):
            transform[0, 2] -= self.camera_edit_step
        elif key == ord('l'):
            transform[0, 2] += self.camera_edit_step
        elif key == ord('u'):
            angle_rad = np.radians(self.camera_angle_step)
            rotation = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad), np.cos(angle_rad), 0],
                [0, 0, 1]
            ])
            transform = rotation @ transform
        elif key == ord('p'):
            angle_rad = np.radians(-self.camera_angle_step)
            rotation = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad), np.cos(angle_rad), 0],
                [0, 0, 1]
            ])
            transform = rotation @ transform
        elif key == ord('['):
            self.camera_edit_step = max(1, self.camera_edit_step - 1)
            self.camera_angle_step = max(0.1, self.camera_angle_step - 0.1)
            print(f"相机编辑步长: 位置={self.camera_edit_step}px, 角度={self.camera_angle_step}°")
        elif key == ord(']'):
            self.camera_edit_step = min(20, self.camera_edit_step + 1)
            self.camera_angle_step = min(10, self.camera_angle_step + 0.1)
            print(f"相机编辑步长: 位置={self.camera_edit_step}px, 角度={self.camera_angle_step}°")
        elif key == ord('='):
            if self.last_calibrated_frame is not None and self.last_calibrated_frame < self.player.current_frame_idx:
                current_offset = self._extract_transform_offset(
                    self.player._original_transform,
                    transform
                )

                last_calibration = self.camera_calibration_history[self.last_calibrated_frame]
                last_offset = last_calibration['offset']

                frame_count = self.player.current_frame_idx - self.last_calibrated_frame
                print(f"\n准备在帧{self.last_calibrated_frame}到帧{self.player.current_frame_idx}之间进行插值矫正")
                print(
                    f"起始偏移: dx={last_offset[0]:.1f}, dy={last_offset[1]:.1f}, dθ={np.degrees(last_offset[2]):.1f}°")
                print(
                    f"结束偏移: dx={current_offset[0]:.1f}, dy={current_offset[1]:.1f}, dθ={np.degrees(current_offset[2]):.1f}°")
                print(f"将影响{frame_count}帧")
                print("按Enter确认插值，按ESC取消")

                self.player.current_state = RefineState.EDITING_CAMERA_INTERPOLATION
                self.player._interpolation_data = {
                    'start_frame': self.last_calibrated_frame,
                    'end_frame': self.player.current_frame_idx,
                    'start_offset': last_offset,
                    'end_offset': current_offset,
                    'current_transform': transform
                }
                return
            else:
                print("没有可用的前一个矫正点，无法进行插值")
        elif key == 13:
            frame_idx = self.player.current_frame_idx

            offset = self._extract_transform_offset(self.player._original_transform, transform)

            self.camera_calibration_history[frame_idx] = {
                'original': self.player._original_transform.copy(),
                'calibrated': transform.copy(),
                'offset': offset
            }

            self.last_calibrated_frame = frame_idx

            print(f"相机运动已矫正 - 帧{frame_idx}")
            print(f"偏移量: dx={offset[0]:.1f}, dy={offset[1]:.1f}, dθ={np.degrees(offset[2]):.1f}°")

            self.player.save_current_data()
            self.player.current_state = RefineState.PAUSED

        elif key == 27:
            self.player.transform_matrices[self.player.current_frame_idx] = self.player._original_transform
            self.player.current_state = RefineState.PAUSED
            print("相机编辑已取消")

        if key not in [13, 27, ord('=')]:
            self.player.transform_matrices[self.player.current_frame_idx] = transform

    def handle_camera_interpolation_confirmation(self, key):
        if key == 13:
            data = self.player._interpolation_data
            start_frame = data['start_frame']
            end_frame = data['end_frame']
            start_offset = data['start_offset']
            end_offset = data['end_offset']

            frame_count = end_frame - start_frame

            for i in range(start_frame + 1, end_frame):
                t = (i - start_frame) / frame_count

                dx = start_offset[0] + t * (end_offset[0] - start_offset[0])
                dy = start_offset[1] + t * (end_offset[1] - start_offset[1])
                dangle = start_offset[2] + t * (end_offset[2] - start_offset[2])

                interpolated_offset = (dx, dy, dangle)

                if i < len(self.player.data['camera_motion']['transform_matrices']):
                    original_transform = np.array(self.player.data['camera_motion']['transform_matrices'][i])

                    new_transform = self._apply_transform_offset(original_transform, interpolated_offset)
                    self.player.transform_matrices[i] = new_transform

                    self.camera_calibration_history[i] = {
                        'original': original_transform.copy(),
                        'calibrated': new_transform.copy(),
                        'offset': interpolated_offset,
                        'interpolated': True
                    }

            self.player.transform_matrices[end_frame] = data['current_transform']
            self.camera_calibration_history[end_frame] = {
                'original': self.player._original_transform.copy(),
                'calibrated': data['current_transform'].copy(),
                'offset': end_offset
            }

            self.last_calibrated_frame = end_frame

            print(f"插值矫正完成：帧{start_frame}到帧{end_frame}")
            print(f"共矫正{frame_count}帧")

            self.player.save_current_data()
            self.player.current_state = RefineState.PAUSED

            self.player._interpolation_data = None

        elif key == 27:
            print("取消插值矫正")

            self.player.transform_matrices[self.player.current_frame_idx] = self.player._original_transform

            self.player.current_state = RefineState.EDITING_CAMERA

            self.player._interpolation_data = None


class OptimizedRefineVideoPlayer:
    """优化的Refined视频播放器 - 使用缓存机制提升性能"""

    def __init__(self, tracking_data_path, map_scale=2.0, debug_first_frame=False, render_boudingbox=False):
        self.current_state = RefineState.LOADING
        self.tracking_data_path = tracking_data_path
        self.map_scale = map_scale
        self.debug_first_frame = debug_first_frame  # 调试模式
        self.render_bounding_box_flag = render_boudingbox

        # 加载数据
        self.load_tracking_data()

        # 初始化视频
        self.setup_video()

        # 初始化缓存管理器
        self.cache_manager = CacheManager(self.data, self.video_info,
                                          map_scale=self.map_scale,
                                          skip_coverage_cache=self.debug_first_frame)
        self.cache_manager.json_path = tracking_data_path
        # 检查并构建缓存
        self._ensure_cache_available()
        # 加载覆盖信息
        self.coverage_data = self.cache_manager.load_coverage_data()

        # 创建编辑器
        self.editor = InteractiveEditor(self)
        # 创建视频保存器
        self.video_saver = VideoSaver(self.video_info, self.render_config)

        # 播放控制
        self.current_frame_idx = 0
        self.frame_cache = {}

        # 插值数据存储
        self._interpolation_data = None

        # 创建窗口和设置回调
        self.window_name = "Optimized Refined Video Player"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.editor.handle_mouse_event)

        print("\n=== 优化的Refined播放器初始化完成 ===")
        if self.debug_first_frame:
            print("=== 调试模式：仅显示第一帧 ===")
        self._print_controls()

    def _print_controls(self):
        print("控制说明：")
        if not self.debug_first_frame:
            print("  空格: 暂停/播放")
            print("  W: 保存当前为止的视频（暂停时）")
            print("  A: 进入标注编辑模式（暂停时）")
            print("  R: 进入朝向编辑模式（暂停时）")
            print("  S: 进入相机运动编辑模式（暂停时）")
        print("  Q: 退出程序")
        if not self.debug_first_frame:
            print("\n编辑模式操作：")
            print("  标注编辑: 拖拽移动标注")
            print("  朝向编辑: ←/→调整角度，↑/↓调整步长")
            print("  相机编辑: IJKL移动位置，U/P旋转，[/]调整步长")
            print("           =键进行插值矫正（需要先有一个矫正点）")
            print("  所有编辑: Enter保存，ESC取消")

    def _ensure_cache_available(self):
        """确保缓存可用"""
        if self.cache_manager.need_cache_rebuild():
            print("\n缓存文件不存在或已过期，开始预处理...")
            self.current_state = RefineState.PREPROCESSING

            def progress_callback(message):
                print(f"预处理进度: {message}")

            try:
                self.cache_manager.build_all_caches(progress_callback)
                print("预处理完成！")
            except Exception as e:
                print(f"预处理失败: {e}")
                raise
        else:
            print("缓存文件已存在，跳过预处理")

    def load_tracking_data(self):
        try:
            with open(self.tracking_data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)

            self.video_info = self.data['video_info']
            self.trajectory_data = self.data['trajectory_summary']
            self.annotations = self.data['annotations']
            self.render_config = self.data['render_config']
            self.camera_motion = self.data.get('camera_motion', {})
            self.tracking_boxes = self.data.get('tracking_boxes', [])

            self.trajectory_positions = [np.array(pos) for pos in self.trajectory_data['trajectory_positions']]
            self.direction_angles = self.trajectory_data['direction_angles']
            self.polygons = [[[float(p[0]), float(p[1])] for p in poly['points']]
                             for poly in self.annotations['polygons']]
            self.circles = [(tuple(circle['center']), circle['radius'])
                            for circle in self.annotations['circles']]

            self.transform_matrices = [np.array(transform) for transform in
                                       self.camera_motion.get('transform_matrices', [])]

            self._ensure_correct_colors()

            # 确保收割参数存在
            if 'weeding_height' not in self.render_config:
                self.render_config['weeding_height'] = 20
            if 'weeding_width' not in self.render_config:
                self.render_config['weeding_width'] = 80
            if 'weeding_color' not in self.render_config:
                self.render_config['weeding_color'] = (200, 200, 0)  # 青色
            if 'weeding_alpha' not in self.render_config:
                self.render_config['weeding_alpha'] = 0.6

            print(f"成功加载追踪数据: {len(self.trajectory_positions)}帧")

        except Exception as e:
            print(f"加载追踪数据失败: {e}")
            raise

    def _ensure_correct_colors(self):
        correct_colors = {
            'trajectory_color': (0, 0, 220),
            'direction_arrow_color': (0, 255, 0),
            'sector_color': (229, 153, 51),
            'polygon_fill_color': (0, 0, 220),
            'polygon_border_color': (0, 0, 160),
            'circle_fill_color': (0, 220, 0),
            'circle_border_color': (0, 150, 0),
            'bbox_color': (255, 0, 0),
        }

        for key, value in correct_colors.items():
            if key in self.render_config:
                self.render_config[key] = value

    def setup_video(self):
        self.cap = cv2.VideoCapture(self.video_info['path'])
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.video_info['path']}")

        self.fps = self.video_info['fps']
        self.frame_width = self.video_info['width']
        self.frame_height = self.video_info['height']
        self.total_frames = min(
            int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            len(self.trajectory_positions),
            len(self.transform_matrices)
        )

    def _screen_to_global_coords(self, screen_x, screen_y):
        try:
            if self.current_frame_idx < len(self.transform_matrices):
                transform = self.transform_matrices[self.current_frame_idx]
            else:
                return None

            point_array = np.array([[float(screen_x), float(screen_y)]], dtype=np.float32).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(point_array, transform)
            return (transformed[0, 0][0], transformed[0, 0][1])
        except:
            return None

    def _global_to_screen_coords(self, global_pos):
        try:
            if self.current_frame_idx < len(self.transform_matrices):
                inv_transform = np.linalg.inv(self.transform_matrices[self.current_frame_idx])
            else:
                return (0, 0)

            point_array = np.array([[float(global_pos[0]), float(global_pos[1])]], dtype=np.float32).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(point_array, inv_transform)
            return (int(transformed[0, 0][0]), int(transformed[0, 0][1]))
        except:
            return (0, 0)

    def _transform_mask_to_frame(self, roi_data, inv_transform, frame_shape):
        """优化的ROI掩码变换"""
        if roi_data is None:
            return None

        roi_mask = roi_data['mask']
        offset_x, offset_y = roi_data['offset']

        # 从缓存管理器获取正确的map_scale和map_bounds
        if hasattr(self.cache_manager, '_annotation_cache'):
            map_scale = float(self.cache_manager._annotation_cache.get('map_scale', self.map_scale))
            map_bounds = self.cache_manager._annotation_cache.get('map_bounds')
        else:
            map_scale = self.cache_manager.exploration_manager.map_scale
            map_bounds = self.cache_manager.exploration_manager.map_bounds

        # 构建ROI到全局的变换
        map_to_global = np.eye(3, dtype=np.float32)
        map_to_global[0, 2] = map_bounds['min_x'] + offset_x / map_scale
        map_to_global[1, 2] = map_bounds['min_y'] + offset_y / map_scale
        map_to_global[0, 0] = 1.0 / map_scale
        map_to_global[1, 1] = 1.0 / map_scale

        # 组合变换
        combined_transform = inv_transform @ map_to_global

        # 变换ROI掩码到帧坐标系
        local_mask = cv2.warpPerspective(
            roi_mask, combined_transform,
            (frame_shape[1], frame_shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        _, local_mask = cv2.threshold(local_mask, 128, 255, cv2.THRESH_BINARY)

        return local_mask

    def render_refined_annotations(self, frame, frame_idx):
        """使用实时计算渲染标注（与原版保持一致）"""
        if frame_idx >= len(self.transform_matrices):
            return

        try:
            inv_transform = np.linalg.inv(self.transform_matrices[frame_idx])
        except:
            return

        # 渲染多边形
        for i, polygon in enumerate(self.polygons):
            roi_data = self.cache_manager.exploration_manager.get_explored_annotation_pixels('polygon', i)
            if roi_data is None or np.sum(roi_data['mask']) == 0:
                continue

            local_mask = self._transform_mask_to_frame(roi_data, inv_transform, frame.shape)
            if local_mask is not None:
                self._render_masked_annotation(frame, local_mask, 'polygon')

        # 渲染圆形
        for i, (center, radius) in enumerate(self.circles):
            # 检查是否已被覆盖
            is_covered = False
            if self.coverage_data and i < self.coverage_data['num_circles']:
                coverage_frame = self.coverage_data['circle_coverage_frames'][i]
                if coverage_frame != -1 and frame_idx >= coverage_frame:
                    is_covered = True

            roi_data = self.cache_manager.exploration_manager.get_explored_annotation_pixels('circle', i)
            if roi_data is None or np.sum(roi_data['mask']) == 0:
                continue

            local_mask = self._transform_mask_to_frame(roi_data, inv_transform, frame.shape)
            if local_mask is not None:
                self._render_masked_annotation(frame, local_mask, 'circle')

                # 如果已被覆盖，绘制红叉
                if is_covered:
                    self._render_cross_on_circle(frame, center, radius, inv_transform)

    def _render_cross_on_circle(self, frame, circle_center, circle_radius, inv_transform):
        """在圆形上绘制红色十字叉"""
        # 转换圆心到屏幕坐标
        point_array = np.array([[float(circle_center[0]), float(circle_center[1])]],
                               dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(point_array, inv_transform)
        screen_center = (int(transformed[0, 0][0]), int(transformed[0, 0][1]))

        # 计算屏幕上的半径（近似）
        # 通过转换圆上的一个点来估算
        edge_point = np.array([[float(circle_center[0] + circle_radius), float(circle_center[1])]],
                              dtype=np.float32).reshape(-1, 1, 2)
        edge_transformed = cv2.perspectiveTransform(edge_point, inv_transform)
        screen_radius = int(np.sqrt((edge_transformed[0, 0][0] - screen_center[0]) ** 2 +
                                    (edge_transformed[0, 0][1] - screen_center[1]) ** 2))

        # 绘制红色十字叉
        cross_size = int(screen_radius * 0.7)  # 十字大小为半径的70%
        cross_color = (0, 0, 255)  # 红色
        cross_thickness = max(2, int(screen_radius * 0.1))  # 线条粗细

        # 绘制十字的两条线
        cv2.line(frame,
                 (screen_center[0] - cross_size, screen_center[1] - cross_size),
                 (screen_center[0] + cross_size, screen_center[1] + cross_size),
                 cross_color, cross_thickness)
        cv2.line(frame,
                 (screen_center[0] - cross_size, screen_center[1] + cross_size),
                 (screen_center[0] + cross_size, screen_center[1] - cross_size),
                 cross_color, cross_thickness)

    def _render_masked_annotation(self, frame, mask, annotation_type):
        """渲染被掩码限制的标注"""
        if annotation_type == 'polygon':
            fill_color = self.render_config['polygon_fill_color']
            border_color = self.render_config['polygon_border_color']
            border_thickness = self.render_config['polygon_border_thickness']
            alpha = self.render_config['polygon_alpha']
        else:
            fill_color = self.render_config['circle_fill_color']
            border_color = self.render_config['circle_border_color']
            border_thickness = self.render_config['circle_border_thickness']
            alpha = self.render_config['circle_alpha']

        valid_mask = mask > 0
        if np.sum(valid_mask) == 0:
            return

        color_mask = np.zeros_like(frame, dtype=np.uint8)
        color_mask[valid_mask] = fill_color

        temp_frame = frame.copy()
        cv2.addWeighted(temp_frame, 1.0, color_mask, alpha, 0, temp_frame)
        frame[valid_mask] = temp_frame[valid_mask]

        try:
            kernel = np.ones((3, 3), np.uint8)
            smooth_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(smooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                cv2.drawContours(frame, contours, -1, border_color, border_thickness)
        except Exception as e:
            print(f"边框绘制失败: {e}")

    def _get_visible_trajectory_range(self, frame_idx, inv_transform):
        """获取视野内的轨迹范围"""
        # 获取视野的四个角点
        corners = np.array([
            [0, 0],
            [self.frame_width, 0],
            [self.frame_width, self.frame_height],
            [0, self.frame_height]
        ], dtype=np.float32).reshape(-1, 1, 2)

        # 转换到全局坐标
        global_corners = cv2.perspectiveTransform(corners, self.transform_matrices[frame_idx])

        # 获取全局坐标的边界
        min_x = np.min(global_corners[:, 0, 0])
        max_x = np.max(global_corners[:, 0, 0])
        min_y = np.min(global_corners[:, 0, 1])
        max_y = np.max(global_corners[:, 0, 1])

        # 添加一些缓冲区
        buffer = 100
        min_x -= buffer
        max_x += buffer
        min_y -= buffer
        max_y += buffer

        # 找到在视野内的轨迹点范围
        start_idx = 0
        end_idx = min(frame_idx + 1, len(self.trajectory_positions))

        # 从后往前找起始点
        for i in range(frame_idx, -1, -1):
            pos = self.trajectory_positions[i]
            if pos[0] < min_x or pos[0] > max_x or pos[1] < min_y or pos[1] > max_y:
                # 继续检查前面几个点，确保轨迹连续
                out_count = 0
                for j in range(max(0, i - 10), i):
                    check_pos = self.trajectory_positions[j]
                    if check_pos[0] < min_x or check_pos[0] > max_x or check_pos[1] < min_y or check_pos[1] > max_y:
                        out_count += 1
                if out_count > 8:  # 如果前10个点中有8个都在视野外，则认为可以截断
                    start_idx = i
                    break

        return start_idx, end_idx

    def render_trajectory_and_orientation(self, frame, frame_idx):
        """渲染轨迹和朝向（优化版本）"""
        if frame_idx >= len(self.trajectory_positions) or frame_idx >= len(self.transform_matrices):
            return

        try:
            inv_transform = np.linalg.inv(self.transform_matrices[frame_idx])
        except:
            return

        # 获取视野内的轨迹范围
        start_idx, end_idx = self._get_visible_trajectory_range(frame_idx, inv_transform)

        binary_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        trajectory_points = []
        for i in range(start_idx, end_idx):
            pos = self.trajectory_positions[i]
            point_array = np.array([[float(pos[0]), float(pos[1])]], dtype=np.float32).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(point_array, inv_transform)
            screen_pos = (int(transformed[0, 0][0]), int(transformed[0, 0][1]))
            trajectory_points.append(screen_pos)

        if len(trajectory_points) > 1:
            for i in range(1, len(trajectory_points)):
                cv2.line(binary_mask, trajectory_points[i - 1], trajectory_points[i], 255,
                         thickness=self.render_config['trajectory_thickness'])

        if np.sum(binary_mask) > 0:
            trajectory_color_mask = np.zeros_like(frame)
            trajectory_color_mask[:] = self.render_config['trajectory_color']

            trajectory_alpha = self.render_config.get('trajectory_alpha', 1.0)

            overlay = np.where(
                binary_mask[:, :, np.newaxis] > 0,
                cv2.addWeighted(
                    frame,
                    1.0 - trajectory_alpha,
                    trajectory_color_mask,
                    trajectory_alpha,
                    0
                ),
                frame
            )
            frame[:] = overlay

        if frame_idx < len(self.direction_angles):
            curr_pos = self.trajectory_positions[frame_idx]
            curr_angle = self.direction_angles[frame_idx]

            point_array = np.array([[float(curr_pos[0]), float(curr_pos[1])]], dtype=np.float32).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(point_array, inv_transform)
            curr_screen = (int(transformed[0, 0][0]), int(transformed[0, 0][1]))

            arrow_length = self.render_config['direction_arrow_length']
            arrow_end = (
                int(curr_screen[0] + arrow_length * np.cos(curr_angle)),
                int(curr_screen[1] + arrow_length * np.sin(curr_angle))
            )

            arrow_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            cv2.arrowedLine(arrow_mask, curr_screen, arrow_end, 255, 2, tipLength=0.3)

            if np.sum(arrow_mask) > 0:
                arrow_alpha = self.render_config.get('arrow_alpha', self.render_config.get('trajectory_alpha', 1.0))
                arrow_color_mask = np.zeros_like(frame)
                arrow_color_mask[:] = self.render_config['direction_arrow_color']

                overlay = np.where(
                    arrow_mask[:, :, np.newaxis] > 0,
                    cv2.addWeighted(
                        frame,
                        1.0 - arrow_alpha,
                        arrow_color_mask,
                        arrow_alpha,
                        0
                    ),
                    frame
                )
                frame[:] = overlay

            self._render_sector(frame, curr_screen, self.render_config['sector_radius'],
                                self.render_config['sector_angle'], curr_angle)

            # 渲染收割矩形
            weed_react_distance_to_robot = self.render_config['weed_react_distance_to_robot']
            if 30 < frame_idx < 100:
                weed_react_distance_to_robot = self.render_config['weed_react_distance_to_robot'] * 0.7

            if frame_idx >= 100:
                    weed_react_distance_to_robot = self.lerp(100, self.render_config['weed_react_distance_to_robot'] * 0.7,
                                                         120,
                                                         self.render_config['weed_react_distance_to_robot'], frame_idx)
            weeding_react_center = (
                int(curr_screen[0] + weed_react_distance_to_robot * np.cos(curr_angle)),
                int(curr_screen[1] + weed_react_distance_to_robot * np.sin(curr_angle))
            )
            self._render_weeding_rect(frame, weeding_react_center, curr_angle, inv_transform)

    def lerp(self, t1, v1, t2, v2, t):
        """线性插值: t < t1 返回 v1, t > t2 返回 v2, 否则线性插值"""
        return v1 if t <= t1 else v2 if t >= t2 else v1 + (v2 - v1) * (t - t1) / (t2 - t1)

    def _render_weeding_rect(self, frame, arrow_end, robot_angle, inv_transform):
        """渲染收割矩形"""
        weeding_height = self.render_config.get('weeding_height', 20)
        weeding_width = self.render_config.get('weeding_width', 100)
        weeding_color = self.render_config.get('weeding_color', (200, 200, 0))
        weeding_alpha = self.render_config.get('weeding_alpha', 0.6)

        # 计算矩形的四个角点（垂直于朝向）
        perpendicular_angle = robot_angle + np.pi / 2

        # 矩形的四个角（在屏幕坐标系中）
        half_width = weeding_width / 2
        half_height = weeding_height / 2

        # 计算矩形中心点（箭头末端）
        center_x, center_y = arrow_end

        # 计算四个角点
        corners = []
        for dx, dy in [(-half_width, -half_height), (half_width, -half_height),
                       (half_width, half_height), (-half_width, half_height)]:
            # 旋转角点
            x = center_x + dx * np.cos(perpendicular_angle) - dy * np.sin(perpendicular_angle)
            y = center_y + dx * np.sin(perpendicular_angle) + dy * np.cos(perpendicular_angle)
            corners.append([int(x), int(y)])

        corners = np.array(corners, dtype=np.int32)

        # 创建矩形掩码
        rect_mask = np.zeros_like(frame)
        cv2.fillPoly(rect_mask, [corners], weeding_color)

        # 叠加到frame上
        cv2.addWeighted(frame, 1.0, rect_mask, weeding_alpha, 0, frame)

        # 绘制边框
        cv2.polylines(frame, [corners], True, weeding_color, 2)
        # cv2.polylines(frame, [corners], True, (255, 255, 255), 2)

    def _render_sector(self, frame, center, radius, angle_deg, facing_angle):
        """渲染扇形视野"""
        half_angle = np.radians(angle_deg) / 2
        start_angle = facing_angle - half_angle
        end_angle = facing_angle + half_angle

        points = [center]
        num_points = 20
        for i in range(num_points + 1):
            current_angle = start_angle + i * (end_angle - start_angle) / num_points
            x = center[0] + radius * np.cos(current_angle)
            y = center[1] + radius * np.sin(current_angle)
            points.append((int(x), int(y)))

        sector_mask = np.zeros_like(frame)
        cv2.fillPoly(sector_mask, [np.array(points, dtype=np.int32)], self.render_config['sector_color'])

        sector_alpha = self.render_config['sector_alpha']
        cv2.addWeighted(frame, 1.0, sector_mask, sector_alpha, 0, frame)

    def render_bounding_box(self, frame, frame_idx):
        """渲染追踪框"""
        if (frame_idx >= len(self.tracking_boxes) or
                not self.render_config.get('show_bbox', True)):
            return

        bbox = self.tracking_boxes[frame_idx]
        x = int(bbox['x'])
        y = int(bbox['y'])
        w = int(bbox['width'])
        h = int(bbox['height'])

        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      self.render_config['bbox_color'],
                      self.render_config.get('bbox_thickness', 2))

    def render_clean_frame(self, frame_idx):
        """渲染干净的帧（恢复探索地图状态）"""
        if frame_idx >= self.total_frames:
            return None

        # 读取帧
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return None

        # 关键修复：从缓存中恢复当前帧的探索地图状态
        if not self.cache_manager.load_exploration_map_for_frame(frame_idx):
            print(f"警告：无法加载帧{frame_idx}的探索地图状态")
            return frame

        # 渲染所有可视化元素
        self.render_trajectory_and_orientation(frame, frame_idx)
        if self.render_bounding_box_flag: self.render_bounding_box(frame, frame_idx)
        self.render_refined_annotations(frame, frame_idx)  # 使用实时计算而非缓存

        return frame

    def process_frame(self, frame_idx):
        """处理指定帧（包含状态信息）"""
        frame = self.render_clean_frame(frame_idx)
        if frame is None:
            return None

        self._draw_status_info(frame)

        return frame

    def _draw_status_info(self, frame):
        """绘制状态信息"""
        info_text = f"Frame: {self.current_frame_idx}/{self.total_frames - 1}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if self.debug_first_frame:
            debug_text = "DEBUG MODE - First Frame Only"
            cv2.putText(frame, debug_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            weeding_info = f"Weeding: {self.render_config['weeding_width']}x{self.render_config['weeding_height']}"
            cv2.putText(frame, weeding_info, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return

        state_map = {
            RefineState.PLAYING: "播放中",
            RefineState.PAUSED: "暂停",
            RefineState.EDITING_ANNOTATION: "标注编辑",
            RefineState.EDITING_ORIENTATION: "朝向编辑",
            RefineState.EDITING_CAMERA: "相机编辑",
            RefineState.EDITING_CAMERA_INTERPOLATION: "相机插值确认",
            RefineState.PREPROCESSING: "预处理中"
        }
        state_text = f"状态: {state_map.get(self.current_state, '未知')}"
        cv2.putText(frame, state_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if self.render_config.get('show_fps', False):
            fps_text = f"FPS: {self.fps:.1f}"
            cv2.putText(frame, fps_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if self.current_state == RefineState.EDITING_ORIENTATION:
            if self.current_frame_idx < len(self.direction_angles):
                angle_text = f"朝向: {np.degrees(self.direction_angles[self.current_frame_idx]):.1f}°"
                cv2.putText(frame, angle_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                step_text = f"步长: {self.editor.orientation_edit_step}°"
                cv2.putText(frame, step_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        elif self.current_state == RefineState.EDITING_ANNOTATION:
            edit_text = "拖拽移动标注"
            cv2.putText(frame, edit_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        elif self.current_state == RefineState.EDITING_CAMERA:
            pos_text = f"位置步长: {self.editor.camera_edit_step}px"
            angle_text = f"角度步长: {self.editor.camera_angle_step:.1f}°"
            cv2.putText(frame, pos_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(frame, angle_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            if self.editor.last_calibrated_frame is not None:
                history_text = f"上次矫正: 帧{self.editor.last_calibrated_frame}"
                cv2.putText(frame, history_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        elif self.current_state == RefineState.EDITING_CAMERA_INTERPOLATION:
            interp_text = "插值矫正模式"
            cv2.putText(frame, interp_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

            if self._interpolation_data:
                start_frame = self._interpolation_data['start_frame']
                end_frame = self._interpolation_data['end_frame']
                range_text = f"范围: 帧{start_frame} → 帧{end_frame}"
                cv2.putText(frame, range_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                confirm_text = "Enter确认, ESC取消"
                cv2.putText(frame, confirm_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

    def save_partial_video(self):
        """保存当前帧为止的视频"""
        if self.current_state != RefineState.PAUSED:
            print("请先暂停视频")
            return

        if self.video_saver.start_saving("refined_partial"):
            for i in range(self.current_frame_idx + 1):
                frame = self.render_clean_frame(i)
                if frame is not None:
                    self.video_saver.save_frame(frame)

                if i % 30 == 0:
                    print(f"保存进度: {i}/{self.current_frame_idx}")

            self.video_saver.stop_saving()

    def save_complete_video(self):
        """保存完整视频"""
        if self.video_saver.start_saving("refined_complete"):
            for i in range(self.total_frames):
                frame = self.render_clean_frame(i)
                if frame is not None:
                    self.video_saver.save_frame(frame)

                if i % 30 == 0:
                    print(f"保存进度: {i}/{self.total_frames - 1}")

            self.video_saver.stop_saving()

    def save_current_data(self):
        """保存当前修改的数据"""
        try:
            updated_data = copy.deepcopy(self.data)

            updated_data['trajectory_summary']['direction_angles'] = [float(angle) for angle in self.direction_angles]

            updated_data['annotations']['polygons'] = [
                {'points': [[float(p[0]), float(p[1])] for p in poly], 'global_coords': True}
                for poly in self.polygons
            ]

            updated_data['annotations']['circles'] = [
                {'center': [float(center[0]), float(center[1])], 'radius': float(radius), 'global_coords': True}
                for center, radius in self.circles
            ]

            # 保存map_scale设置
            updated_data['render_config']['last_used_map_scale'] = self.map_scale

            updated_data['camera_motion']['transform_matrices'] = [
                transform.tolist() for transform in self.transform_matrices
            ]

            backup_path = self.tracking_data_path.replace('.json', '_backup.json')

            if not os.path.exists(backup_path):
                import shutil
                shutil.copy2(self.tracking_data_path, backup_path)
                print(f"原数据已备份到: {backup_path}")

            with open(self.tracking_data_path, 'w', encoding='utf-8') as f:
                json.dump(updated_data, f, indent=2, ensure_ascii=False)

            print(f"数据已更新保存到: {self.tracking_data_path}")
            print("注意：如果修改了轨迹或标注数据，请删除缓存文件以重新生成")

            self.data = updated_data

        except Exception as e:
            print(f"保存数据失败: {e}")

    def handle_key_press(self, key):
        """处理按键事件"""
        if key == ord('q') or key == 27:
            if self.current_state in [RefineState.EDITING_ANNOTATION,
                                      RefineState.EDITING_ORIENTATION,
                                      RefineState.EDITING_CAMERA,
                                      RefineState.EDITING_CAMERA_INTERPOLATION]:
                self.current_state = RefineState.PAUSED
                print("退出编辑模式")
                return True
            else:
                return False

        if self.debug_first_frame:
            # 调试模式下只允许退出
            return True

        elif key == ord(' '):
            if self.current_state == RefineState.PLAYING:
                self.current_state = RefineState.PAUSED
                print("暂停")
            elif self.current_state == RefineState.PAUSED:
                self.current_state = RefineState.PLAYING
                print("继续播放")

        elif key == ord('w') or key == ord('W'):
            if self.current_state == RefineState.PAUSED:
                print("开始保存视频...")
                self.save_partial_video()
            else:
                print("请先暂停视频")

        elif key == ord('a') or key == ord('A'):
            if self.current_state == RefineState.PAUSED:
                self.current_state = RefineState.EDITING_ANNOTATION
                print("进入标注编辑模式 - 拖拽鼠标移动标注")
            else:
                print("请先暂停视频")

        elif key == ord('r') or key == ord('R'):
            if self.current_state == RefineState.PAUSED:
                self.current_state = RefineState.EDITING_ORIENTATION
                self._original_angle = self.direction_angles[self.current_frame_idx]
                print("进入朝向编辑模式")
                print("使用←/→调整角度，↑/↓调整步长，Enter保存，ESC取消")
            else:
                print("请先暂停视频")

        elif key == ord('s') or key == ord('S'):
            if self.current_state == RefineState.PAUSED:
                self.current_state = RefineState.EDITING_CAMERA
                self._original_transform = self.transform_matrices[self.current_frame_idx].copy()
                print("进入相机运动编辑模式")
                print("IJKL: 移动位置，U/P: 旋转，[/]: 调整步长")
                print("=: 插值矫正（需要先有矫正点）")
                print("Enter保存，ESC取消")
            else:
                print("请先暂停视频")

        return True

    def run(self):
        """运行播放器主循环"""
        if self.debug_first_frame:
            # 调试模式：只显示第一帧
            print("\n调试模式：显示第一帧")
            frame = self.process_frame(0)
            if frame is not None:
                cv2.imshow(self.window_name, frame)

            print("\n当前收割参数：")
            print(f"  宽度: {self.render_config['weeding_width']}")
            print(f"  高度: {self.render_config['weeding_height']}")
            print(f"  颜色: {self.render_config['weeding_color']}")
            print("\n按Q退出...")

            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
        else:
            # 正常播放模式
            self.current_state = RefineState.PLAYING
            clock = time.time()

            print("\n开始播放...")

            try:
                while True:
                    current_time = time.time()

                    if self.current_state == RefineState.EDITING_ORIENTATION:
                        key = cv2.waitKey(1) & 0xFF
                        if key != 255:
                            self.editor.handle_orientation_editing(key)
                            frame = self.process_frame(self.current_frame_idx)
                            if frame is not None:
                                cv2.imshow(self.window_name, frame)

                    elif self.current_state == RefineState.EDITING_CAMERA:
                        key = cv2.waitKey(1) & 0xFF
                        if key != 255:
                            self.editor.handle_camera_editing(key)
                            frame = self.process_frame(self.current_frame_idx)
                            if frame is not None:
                                cv2.imshow(self.window_name, frame)

                    elif self.current_state == RefineState.EDITING_CAMERA_INTERPOLATION:
                        key = cv2.waitKey(1) & 0xFF
                        if key != 255:
                            self.editor.handle_camera_interpolation_confirmation(key)
                            frame = self.process_frame(self.current_frame_idx)
                            if frame is not None:
                                cv2.imshow(self.window_name, frame)

                    else:
                        key = cv2.waitKey(1) & 0xFF
                        if key != 255:
                            if not self.handle_key_press(key):
                                break

                    if self.current_state == RefineState.PLAYING:
                        frame_interval = 1.0 / self.fps
                        if current_time - clock >= frame_interval:
                            frame = self.process_frame(self.current_frame_idx)
                            if frame is None:
                                print("\n播放完成，自动保存视频...")
                                self.save_complete_video()
                                self.current_state = RefineState.PAUSED
                            else:
                                cv2.imshow(self.window_name, frame)
                                self.current_frame_idx += 1
                                clock = current_time

                    elif self.current_state in [RefineState.PAUSED, RefineState.EDITING_ANNOTATION]:
                        frame = self.process_frame(self.current_frame_idx)
                        if frame is not None:
                            cv2.imshow(self.window_name, frame)

            finally:
                # 清理资源
                self.cap.release()
                self.cache_manager.close_explore_reader()
                cv2.destroyAllWindows()

        # 清理资源
        self.cap.release()
        self.cache_manager.close_explore_reader()
        cv2.destroyAllWindows()


def main():
    """主函数"""
    tracking_data_path = "tracking_data_5_29_good.json"

    # 可以从命令行参数或配置文件读取map_scale
    import sys
    map_scale = 1.5  # 1.5  # 默认值
    debug_mode = True  # 默认关闭调试模式
    render_boudingbox = False

    # 解析命令行参数
    if len(sys.argv) > 1:
        try:
            map_scale = float(sys.argv[1])
            print(f"使用map_scale: {map_scale}")
        except:
            print("无效的map_scale参数，使用默认值1.5")

    if len(sys.argv) > 2:
        debug_mode = sys.argv[2].lower() == 'debug'
        if debug_mode:
            print("启用调试模式（仅显示第一帧）")

    if not os.path.exists(tracking_data_path):
        print(f"文件不存在: {tracking_data_path}")
        print("请先运行标注阶段生成tracking_data_5_22.json文件")
        return
    try:
        player = OptimizedRefineVideoPlayer(tracking_data_path, map_scale=map_scale, debug_first_frame=debug_mode,
                                            render_boudingbox=render_boudingbox)
        player.run()
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
