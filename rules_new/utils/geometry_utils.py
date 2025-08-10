"""
几何工具函数 - 提供几何计算和路径处理功能
"""
import math
import numpy as np
from typing import List, Tuple, Optional
from matplotlib.path import Path
from shapely.geometry import Point, Polygon, LineString


class GeometryUtils:
    """几何计算工具类"""
    
    @staticmethod
    def find_longest_edge(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """找到多边形的最长边"""
        max_length = 0
        longest_edge = None
        
        for i in range(len(vertices)):
            start = vertices[i]
            end = vertices[(i + 1) % len(vertices)]
            length = np.linalg.norm(end - start)
            if length > max_length:
                max_length = length
                longest_edge = (start, end)
                
        return longest_edge
    
    @staticmethod
    def is_point_in_polygon(point: Tuple[float, float], vertices: np.ndarray) -> bool:
        """检查点是否在多边形内"""
        path = Path(vertices)
        return path.contains_point(point)
    
    @staticmethod
    def create_polygon_mask(vertices: np.ndarray, width: int, height: int) -> np.ndarray:
        """创建多边形掩码"""
        poly_path = Path(vertices)
        y, x = np.mgrid[:height, :width]
        coords = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        mask = np.zeros((height, width))
        mask[poly_path.contains_points(coords).reshape(height, width)] = 1
        return mask
    
    @staticmethod
    def calculate_angle(start: Tuple[float, float], end: Tuple[float, float]) -> float:
        """计算两点间的角度（弧度）"""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        return math.atan2(dy, dx)
    
    @staticmethod
    def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """计算两点间的欧几里得距离"""
        return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
    
    @staticmethod
    def interpolate_line(start: Tuple[float, float], end: Tuple[float, float], step: float = 1.0) -> List[Tuple[float, float]]:
        """在两点间插值生成路径点"""
        start_point = np.array(start)
        end_point = np.array(end)
        
        direction = end_point - start_point
        distance = np.linalg.norm(direction)
        
        if distance < step:
            return [end]
            
        num_steps = int(distance / step)
        points = []
        
        for i in range(1, num_steps + 1):
            interpolated_point = start_point + (i / num_steps) * direction
            points.append(tuple(interpolated_point))
            
        points.append(end)
        return points
    
    @staticmethod
    def rotate_point(point: Tuple[float, float], angle: float, center: Tuple[float, float] = (0, 0)) -> Tuple[float, float]:
        """绕中心点旋转点"""
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        
        # 平移到原点
        translated_x = point[0] - center[0]
        translated_y = point[1] - center[1]
        
        # 旋转
        rotated_x = translated_x * cos_angle - translated_y * sin_angle
        rotated_y = translated_x * sin_angle + translated_y * cos_angle
        
        # 平移回原位置
        final_x = rotated_x + center[0]
        final_y = rotated_y + center[1]
        
        return (final_x, final_y)
    
    @staticmethod
    def find_offset(start: Tuple[float, float], end: Tuple[float, float], point: Tuple[float, float], 
                   angle: Optional[float] = None) -> float:
        """计算点到直线的垂直距离（带符号）"""
        start = np.array(start)
        end = np.array(end)
        point = np.array(point)
        
        if angle is None:
            # 使用start到end的向量
            line_vec = end - start
        else:
            # 根据角度计算方向向量
            line_length = np.linalg.norm(end - start)
            line_vec = np.array([np.cos(angle) * line_length, np.sin(angle) * line_length])
        
        # 计算从start到point的向量
        point_vec = point - start
        
        # 计算叉积（保留符号信息）
        cross_product = np.cross(line_vec, point_vec)
        
        # 计算垂直距离
        norm_line_vec = np.linalg.norm(line_vec)
        if norm_line_vec == 0:
            return 0.0
            
        distance = cross_product / norm_line_vec
        return distance
    
    @staticmethod
    def find_nearest_point_with_rotation(target_point: Tuple[float, float], angle: float, 
                                       candidates: List[Tuple[float, float]]) -> Tuple[Optional[Tuple[float, float]], int]:
        """使用旋转变换找到最近点（用于JUMP算法）"""
        if not candidates:
            return None, -1
            
        # 准备旋转矩阵
        radians = -angle % (2 * np.pi)  # 矩阵坐标系转换
        
        rotation_matrix = np.array([
            [np.cos(radians), -np.sin(radians)],
            [np.sin(radians), np.cos(radians)]
        ])
        
        # 旋转目标点
        target_rotated = np.dot(rotation_matrix, np.array(target_point))
        
        # 旋转候选点并找到最近的
        rotated_candidates = [np.dot(rotation_matrix, np.array(c)) for c in candidates]
        nearest_index = min(range(len(rotated_candidates)), key=lambda i: abs(rotated_candidates[i][0] - target_rotated[0]))
        
        return candidates[nearest_index], nearest_index
    
    @staticmethod
    def filter_points_by_direction(points: List[Tuple[float, float]], reference_point: Tuple[float, float], 
                                 direction_vector: np.ndarray, constraint_vector: Optional[np.ndarray] = None,
                                 constraint_value: float = 0.0) -> List[Tuple[float, float]]:
        """根据方向向量筛选点"""
        filtered_points = []
        ref_point = np.array(reference_point)
        
        for point in points:
            point_array = np.array(point)
            to_point = point_array - ref_point
            
            # 检查方向约束
            if np.dot(to_point, direction_vector) > 0:
                # 如果有额外约束
                if constraint_vector is not None:
                    if np.dot(to_point, constraint_vector) > constraint_value:
                        filtered_points.append(point)
                else:
                    filtered_points.append(point)
                    
        return filtered_points
    
    @staticmethod
    def validate_path_in_polygon(path: List[Tuple[float, float]], polygon_mask: np.ndarray, 
                               width: int, height: int) -> bool:
        """验证路径是否在多边形内"""
        for point in path:
            x, y = int(point[0]), int(point[1])
            if not (0 <= x < width and 0 <= y < height):
                return False
            if polygon_mask[y, x] == 0:
                return False
        return True