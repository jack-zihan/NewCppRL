"""
Rules_new Waypoint提取器 - 捕获算法生成的waypoints而不执行实际移动
"""
import sys
import math
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'tests'))  # 让rules_new能找到env_make


class RulesNewExtractor:
    """提取rules_new算法生成的waypoints"""
    
    def __init__(self, task_type: str = 'JUMP', seed: int = 42):
        self.task_type = task_type
        self.seed = seed
        self.waypoints = []
        self.env = None
        self.original_navigate = None
        self.original_go = None
        self.original_dubins_navigate = None
        
    def setup_environment(self):
        """设置环境和参数"""
        # 导入env_make
        from env_make import get_test_env
        
        # 创建环境
        self.env, _ = get_test_env(seed=self.seed)
        
        # 设置Config参数
        from rules.config import Config
        Config.TASK_TYPE = self.task_type
        Config.RANDOM_SEED = self.seed
        
        return self.env
        
    def patch_functions(self):
        """动态替换navigate等函数，只记录waypoints"""
        import rules.jump_path as jp
        
        # 保存原始函数
        self.original_navigate = jp.navigate
        self.original_go = jp.go
        if hasattr(jp, 'dubins_navigate'):
            self.original_dubins_navigate = jp.dubins_navigate
        
        # 替换函数
        jp.navigate = self.mock_navigate
        jp.go = self.mock_go
        jp.dubins_navigate = self.mock_dubins_navigate
        
        # 设置全局变量
        jp.env = self.env
        jp.task_type = self.task_type
        
    def restore_functions(self):
        """恢复原始函数"""
        if self.original_navigate:
            import rules.jump_path as jp
            jp.navigate = self.original_navigate
            jp.go = self.original_go
            if self.original_dubins_navigate:
                jp.dubins_navigate = self.original_dubins_navigate
                
    def mock_navigate(self, goal):
        """Mock navigate函数 - 只记录waypoint"""
        # 记录目标点（注意rules_new使用[y,x]格式）
        self.waypoints.append(tuple(goal))
        
        # 更新agent位置（模拟移动）
        import rules.jump_path as jp
        jp.agent_position = goal
        
    def mock_go(self, p2):
        """Mock go函数 - 记录低级移动"""
        # go函数用于更细粒度的移动，这里也记录
        pass  # 暂时跳过，主要关注navigate级别的waypoints
        
    def mock_dubins_navigate(self, goal_pose, turning_radius):
        """Mock dubins_navigate函数"""
        # Dubins路径的目标点
        if len(goal_pose) >= 2:
            self.waypoints.append((goal_pose[0], goal_pose[1]))
            
    def extract_waypoints(self, max_waypoints: int = 100) -> List[Tuple[float, float]]:
        """
        提取算法生成的waypoints
        
        Args:
            max_waypoints: 最大waypoint数量
            
        Returns:
            waypoints列表
        """
        self.waypoints = []
        
        # 设置环境
        env = self.setup_environment()
        
        # Patch函数
        self.patch_functions()
        
        try:
            # 导入并运行算法
            import rules.jump_path as jp
            
            # 重新初始化关键变量
            jp.agent_width = 5
            jp.sight_width = 24
            jp.sight_length = 24
            jp.W = 600
            jp.H = 600
            jp.agent_position = [float(env.agent.y), float(env.agent.x)]
            jp.turning_radius = env.v_range.max / (abs(env.w_range.max) * (math.pi / 180))
            jp.farm_vertices = env.min_area_rect[0][:, 0, ::-1]
            
            # 运行算法主循环的简化版本
            self.run_algorithm_loop(jp, max_waypoints)
            
        finally:
            # 恢复原始函数
            self.restore_functions()
            
            # 关闭环境
            if self.env:
                self.env.close()
        
        return self.waypoints[:max_waypoints]
    
    def run_algorithm_loop(self, jp, max_waypoints: int):
        """运行算法主循环的简化版本"""
        from matplotlib.path import Path
        from shapely.geometry import LineString, Polygon
        
        # 获取必要的变量
        farm_vertices = jp.farm_vertices
        agent_width = jp.agent_width
        sight_width = jp.sight_width
        
        # 计算最长边和方向
        def find_longest_edge(vertices):
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
        
        max_edge_points = find_longest_edge(farm_vertices)
        dx = max_edge_points[1][0] - max_edge_points[0][0]
        dy = max_edge_points[1][1] - max_edge_points[0][1]
        real_radians = np.arctan2(dy, dx)
        real_radians = real_radians % (2 * np.pi) if real_radians >= 0 else (real_radians + 2 * np.pi) % (2 * np.pi)
        
        # 创建多边形mask
        poly_path = Path(farm_vertices)
        y, x = np.mgrid[:jp.H, :jp.W]
        coor = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        mask = np.zeros((jp.H, jp.W))
        mask[poly_path.contains_points(coor).reshape(jp.H, jp.W)] = 1
        
        # 计算对角线长度
        min_x, min_y = farm_vertices.min(axis=0)
        max_x, max_y = farm_vertices.max(axis=0)
        diag_length = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)
        
        # 初始化
        y_offset = -diag_length + agent_width / 2
        turn = False
        start = [0, 0]
        end = np.array([100 * np.cos(real_radians), 100 * np.sin(real_radians)])
        
        waypoint_count = 0
        
        # 主循环 - 生成路径线
        while y_offset < diag_length and waypoint_count < max_waypoints:
            x_points = []
            new_start = [start[0] + y_offset * np.cos(real_radians + np.pi / 2) - diag_length * np.cos(real_radians),
                         start[1] + y_offset * np.sin(real_radians + np.pi / 2) - diag_length * np.sin(real_radians)]
            new_end = [end[0] + y_offset * np.cos(real_radians + np.pi / 2) + diag_length * np.cos(real_radians),
                       end[1] + y_offset * np.sin(real_radians + np.pi / 2) + diag_length * np.sin(real_radians)]
            
            line = LineString([new_start, new_end])
            
            # 生成线上的点
            for i in np.arange(0, line.length, 1):
                interpolated_point = np.array(line.interpolate(i).coords[0])
                x_points.append(interpolated_point)
            
            # 过滤有效点
            valid_points = [point for point in x_points if
                            0 <= int(point[1]) < jp.H and 0 <= int(point[0]) < jp.W and mask[int(point[1]), int(point[0])] == 1]
            
            if valid_points:
                turn = not turn
                
            # 根据方向调整顺序
            valid_points = valid_points if not turn else valid_points[::-1]
            
            # 记录waypoints
            if valid_points:
                # 对于JUMP算法，返回第一个有效点
                if self.task_type in ['JUMP', 'BCP']:
                    # 简化处理：返回路径线的第一个点
                    if waypoint_count == 0:  # 第一次
                        self.waypoints.append(tuple(valid_points[0]))
                        waypoint_count += 1
                    else:
                        # 后续路径线，根据算法逻辑处理
                        for point in valid_points[::10]:  # 每10个点取一个，简化
                            if waypoint_count >= max_waypoints:
                                break
                            self.waypoints.append(tuple(point))
                            waypoint_count += 1
                            
            y_offset += sight_width / 2
            
            if waypoint_count >= max_waypoints:
                break