#!/usr/bin/env python3
"""
简化的rules_new运行器 - 只提取核心路径生成逻辑
"""
import sys
import math
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
from matplotlib.path import Path as MPath
from shapely.geometry import LineString, Polygon

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'tests'))
sys.path.insert(0, str(project_root / 'envs'))


class RulesNewSimpleRunner:
    """运行rules_new核心算法逻辑，不依赖完整的jump_path.py"""
    
    def __init__(self, task_type: str = 'JUMP', seed: int = 42):
        self.task_type = task_type
        self.seed = seed
        self.waypoints = []
        
        # 算法参数
        self.agent_width = 5
        self.sight_width = 24
        self.sight_length = 24
        self.W = 600
        self.H = 600
        
    def find_longest_edge(self, vertices):
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
    
    def generate_waypoints(self, max_waypoints: int = 20) -> List[Tuple[float, float]]:
        """生成waypoints - rules_new的核心逻辑"""
        
        # 创建环境获取参数
        from env_make import get_test_env
        env, _ = get_test_env(seed=self.seed)
        
        # 获取环境参数
        farm_vertices = env.min_area_rect[0][:, 0, ::-1]  # [y, x] -> [x, y]
        turning_radius = env.v_range.max / (abs(env.w_range.max) * (math.pi / 180))
        agent_position = [float(env.agent.y), float(env.agent.x)]
        
        # 计算最长边和方向
        max_edge_points = self.find_longest_edge(farm_vertices)
        dx = max_edge_points[1][0] - max_edge_points[0][0]
        dy = max_edge_points[1][1] - max_edge_points[0][1]
        real_radians = np.arctan2(dy, dx)
        real_radians = real_radians % (2 * np.pi) if real_radians >= 0 else (real_radians + 2 * np.pi) % (2 * np.pi)
        
        # 创建多边形mask
        poly_path = MPath(farm_vertices)
        y, x = np.mgrid[:self.H, :self.W]
        coor = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        mask = np.zeros((self.H, self.W))
        mask[poly_path.contains_points(coor).reshape(self.H, self.W)] = 1
        
        # 计算对角线长度
        min_x, min_y = farm_vertices.min(axis=0)
        max_x, max_y = farm_vertices.max(axis=0)
        diag_length = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)
        
        # 初始化
        y_offset = -diag_length + self.agent_width / 2
        turn = False  # 初始为False（与rules_new一致）
        start = [0, 0]
        end = np.array([100 * np.cos(real_radians), 100 * np.sin(real_radians)])
        
        waypoints = []
        
        # 主循环 - 生成路径线
        while y_offset < diag_length and len(waypoints) < max_waypoints:
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
                            0 <= int(point[1]) < self.H and 0 <= int(point[0]) < self.W and mask[int(point[1]), int(point[0])] == 1]
            
            if valid_points:
                turn = not turn
                
            # 根据方向调整顺序
            valid_points = valid_points if not turn else valid_points[::-1]
            
            # 根据算法类型处理waypoints
            if self.task_type == 'JUMP':
                # JUMP算法：正常前进，偶尔跳跃
                if valid_points:
                    # 第一次设置起始位置
                    if not waypoints:
                        # rules_new会设置: env.agent.x = valid_points[0][1], env.agent.y = valid_points[0][0]
                        # 所以第一个waypoint是交换后的坐标
                        waypoints.append((valid_points[0][1], valid_points[0][0]))
                    else:
                        # 后续的waypoints
                        for i in range(0, len(valid_points), 20):  # 每20个点取一个
                            if len(waypoints) >= max_waypoints:
                                break
                            # rules_new的navigate会接收原始坐标
                            waypoints.append(tuple(valid_points[i]))
                            
            elif self.task_type == 'SNAKE':
                # SNAKE算法：贪婪搜索
                if valid_points:
                    if not waypoints:
                        waypoints.append((valid_points[0][1], valid_points[0][0]))
                    else:
                        for i in range(0, len(valid_points), 15):  # 密度稍高
                            if len(waypoints) >= max_waypoints:
                                break
                            waypoints.append(tuple(valid_points[i]))
                            
            elif self.task_type == 'BCP':
                # BCP算法：基础覆盖
                if valid_points:
                    if not waypoints:
                        waypoints.append((valid_points[0][1], valid_points[0][0]))
                    else:
                        for i in range(0, len(valid_points), 25):  # 稀疏覆盖
                            if len(waypoints) >= max_waypoints:
                                break
                            waypoints.append(tuple(valid_points[i]))
            
            # 更新y_offset
            y_offset += self.sight_width / 2
        
        env.close()
        
        return waypoints


def test_simple_runner():
    """测试简化运行器"""
    print("测试RulesNewSimpleRunner")
    print("-" * 60)
    
    runner = RulesNewSimpleRunner(task_type='JUMP', seed=42)
    waypoints = runner.generate_waypoints(max_waypoints=10)
    
    print(f"生成了 {len(waypoints)} 个waypoints:")
    for i, wp in enumerate(waypoints[:5]):
        print(f"  {i}: {wp}")
    
    return waypoints


def run_rules_new_simple(algorithm: str = 'JUMP', seed: int = 42, max_waypoints: int = 100) -> Dict[str, Any]:
    """
    运行rules_new算法的简化版本
    
    Args:
        algorithm: 算法名称 ('JUMP', 'SNAKE', 'R_SNAKE', 'REACT', 'BCP')
        seed: 随机种子
        max_waypoints: 最大waypoint数量
        
    Returns:
        包含waypoints和其他信息的字典
    """
    runner = RulesNewSimpleRunner(task_type=algorithm, seed=seed)
    waypoints = runner.generate_waypoints(max_waypoints=max_waypoints)
    
    return {
        'algorithm': algorithm,
        'seed': seed,
        'waypoints': waypoints,
        'num_waypoints': len(waypoints)
    }


if __name__ == "__main__":
    test_simple_runner()