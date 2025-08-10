#!/usr/bin/env python3
"""
Rules_new 适配器
将rules_new的算法执行封装成统一接口
基于实际jump_path.py的逻辑实现
"""

import sys
import os
import math
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import gymnasium as gym
import dubins
from matplotlib.path import Path as MPath
from shapely.geometry import Point, Polygon, LineString

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class RulesNewAdapter:
    """Rules_new算法适配器 - 基于实际jump_path.py实现"""
    
    def __init__(self):
        """初始化适配器"""
        self.project_root = project_root
        
        # 算法参数
        self.agent_width = 5
        self.sight_width = 24
        self.sight_length = 24
        self.W = 600
        self.H = 600
        
        # 全局状态
        self.agent_position = None
        self.rad = 0
        self.discovered = []
        self.turning_radius = None
        self.farm_vertices = None
        self.real_radians = 0
        self.diagonal_length = 0
        self.poly_path = None
        self.mask = None
        self.polygon = None
        
    def run_algorithm(self, algorithm: str, env: gym.Env, obs: Any, info: Dict,
                     max_steps: int = 1000, render: bool = False) -> Dict[str, Any]:
        """
        运行rules_new算法
        
        基于jump_path.py的实际逻辑实现
        """
        # 初始化结果存储
        result = {
            'algorithm': algorithm,
            'trajectory': [],
            'actions': [],
            'rewards': [],
            'total_reward': 0,
            'steps': 0,
            'coverage_rate': 0,
            'execution_time': 0,
            'final_frame': None,
            'discovered_weeds': []
        }
        
        start_time = time.time()
        
        # 初始化环境状态
        self._initialize_environment(env)
        
        # 收集初始位置
        result['trajectory'].append([env.agent.x, env.agent.y])
        
        # 根据算法类型执行不同逻辑
        if algorithm == 'BCP':
            result = self._run_bcp(env, result, max_steps, render)
        elif algorithm == 'JUMP':
            result = self._run_jump(env, result, max_steps, render)
        elif algorithm == 'SNAKE':
            result = self._run_snake(env, result, max_steps, render, False)
        elif algorithm == 'R_SNAKE':
            result = self._run_snake(env, result, max_steps, render, True)
        elif algorithm == 'REACT':
            result = self._run_react(env, result, max_steps, render)
        
        result['execution_time'] = time.time() - start_time
        
        return result
    
    def _initialize_environment(self, env):
        """初始化环境相关参数"""
        # 初始位置（rules_new使用[y,x]格式）
        self.agent_position = [env.agent.y, env.agent.x]
        
        # 计算turning_radius
        w_max_rad = abs(env.w_range.max) * (math.pi / 180)
        self.turning_radius = env.v_range.max / w_max_rad
        
        # 获取农场顶点
        if hasattr(env, 'min_area_rect') and env.min_area_rect is not None:
            self.farm_vertices = env.min_area_rect[0][:, 0, ::-1]
        else:
            # 使用默认矩形
            self.farm_vertices = np.array([
                [50, 50], [550, 50], [550, 550], [50, 550]
            ])
        
        # 计算农场方向和对角线长度
        self.real_radians, self.diagonal_length = self._calculate_orientation(self.farm_vertices)
        
        # 初始方向
        self.rad = np.pi / 2 - math.radians(env.agent.direction)
        
        # 创建多边形和掩码
        self.polygon = Polygon(self.farm_vertices)
        self.poly_path = MPath(self.farm_vertices)
        
        # 创建掩码
        y, x = np.mgrid[:self.H, :self.W]
        coor = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        self.mask = np.zeros((self.H, self.W))
        self.mask[self.poly_path.contains_points(coor).reshape(self.H, self.W)] = 1
        
        # 初始化发现的杂草
        self._update_discovered(env)
    
    def _update_discovered(self, env):
        """更新发现的杂草位置"""
        if hasattr(env, 'map_weed') and hasattr(env, 'map_frontier'):
            discovered = np.argwhere(np.logical_and(env.map_weed, np.logical_not(env.map_frontier)) == 1)
            self.discovered = [point for point in discovered if self._is_point_in_polygon(point, self.farm_vertices)]
        else:
            self.discovered = []
    
    def _is_point_in_polygon(self, point, vertices):
        """检查点是否在多边形内"""
        path = MPath(vertices)
        return path.contains_point(point)
    
    def _calculate_orientation(self, vertices: np.ndarray) -> Tuple[float, float]:
        """计算农场方向和对角线长度"""
        # 找最长边
        max_length = 0
        longest_edge = None
        for i in range(len(vertices)):
            start = vertices[i]
            end = vertices[(i + 1) % len(vertices)]
            length = np.linalg.norm(end - start)
            if length > max_length:
                max_length = length
                longest_edge = (start, end)
        
        # 计算方向
        dx = longest_edge[1][0] - longest_edge[0][0]
        dy = longest_edge[1][1] - longest_edge[0][1]
        real_radians = np.arctan2(dy, dx)
        real_radians = real_radians % (2 * np.pi) if real_radians >= 0 else (real_radians + 2 * np.pi) % (2 * np.pi)
        
        # 计算对角线长度
        min_x, min_y = vertices.min(axis=0)
        max_x, max_y = vertices.max(axis=0)
        diagonal_length = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)
        
        return real_radians, diagonal_length
    
    def _go(self, p2, env):
        """执行单步移动"""
        # 计算目标方向和距离
        radian = math.atan2(p2[1] - self.agent_position[1], p2[0] - self.agent_position[0])
        length = math.sqrt((p2[0] - self.agent_position[0]) ** 2 + (p2[1] - self.agent_position[1]) ** 2)
        delta_angle = -(radian - self.rad) % (2 * math.pi)
        delta_angle = delta_angle - 2 * math.pi if delta_angle > math.pi else delta_angle
        delta_angle = math.degrees(delta_angle)
        
        # 限制动作范围
        length = min(length, 3.5)
        delta_angle = max(-28.6, min(28.6, delta_angle))
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step([length, delta_angle])
        
        # 更新状态
        self.agent_position = [env.agent.y, env.agent.x]
        self.rad = np.pi / 2 - math.radians(env.agent.direction)
        self._update_discovered(env)
        
        return obs, reward, terminated, truncated, info
    
    def _navigate(self, goal, env, result, render):
        """导航到目标点（分步执行）"""
        vector = np.array(goal) - np.array(self.agent_position)
        distance = np.linalg.norm(vector)
        
        if distance < 1:
            return False
        
        num_steps = max(1, int(distance // 2))
        step_vector = vector / num_steps
        waypoints = [self.agent_position + step_vector * i for i in range(1, num_steps + 1)]
        waypoints.append(goal)
        
        for p2 in waypoints:
            # 添加最大迭代限制，避免无限循环
            max_iterations = 10
            iterations = 0
            prev_distance = float('inf')
            
            while iterations < max_iterations:
                current_distance = np.linalg.norm(np.array(p2) - np.array(self.agent_position))
                
                # 如果已经足够接近或距离没有减小，退出循环
                if current_distance < 2.0 or current_distance >= prev_distance:
                    break
                
                obs, reward, terminated, truncated, info = self._go(p2, env)
                
                # 收集数据
                result['trajectory'].append([env.agent.x, env.agent.y])
                result['actions'].append([0, 0])  # 简化动作记录
                result['rewards'].append(reward)
                result['total_reward'] += reward
                result['steps'] += 1
                result['coverage_rate'] = info.get('coverage_rate', 0)
                
                if render:
                    result['final_frame'] = env.render()
                
                if terminated or truncated:
                    return True
                
                prev_distance = current_distance
                iterations += 1
        
        return False
    
    def _dubins_navigate(self, p2, r, env, result, render):
        """使用Dubins路径导航"""
        try:
            path = dubins.shortest_path(
                (self.agent_position[0], self.agent_position[1], self.rad),
                (p2[0], p2[1], p2[2]), r
            )
            configurations, _ = path.sample_many(0.5)
            
            for point in configurations[1:]:
                done = self._navigate(list(point[:2]), env, result, render)
                if done:
                    return True
        except:
            # 如果Dubins路径失败，直接导航
            return self._navigate([p2[0], p2[1]], env, result, render)
        
        return False
    
    def _generate_path_line(self, y_offset, turn_direction):
        """生成路径线上的点"""
        start = [0, 0]
        end = np.array([100 * np.cos(self.real_radians), 100 * np.sin(self.real_radians)])
        
        new_start = [
            start[0] + y_offset * np.cos(self.real_radians + np.pi / 2) - self.diagonal_length * np.cos(self.real_radians),
            start[1] + y_offset * np.sin(self.real_radians + np.pi / 2) - self.diagonal_length * np.sin(self.real_radians)
        ]
        new_end = [
            end[0] + y_offset * np.cos(self.real_radians + np.pi / 2) + self.diagonal_length * np.cos(self.real_radians),
            end[1] + y_offset * np.sin(self.real_radians + np.pi / 2) + self.diagonal_length * np.sin(self.real_radians)
        ]
        
        # 生成线上的点
        line = LineString([new_start, new_end])
        x_points = []
        for i in np.arange(0, line.length, 1):
            interpolated_point = np.array(line.interpolate(i).coords[0])
            x_points.append(interpolated_point)
        
        # 过滤有效点
        valid_points = [
            point for point in x_points
            if 0 <= int(point[1]) < self.H and 0 <= int(point[0]) < self.W
            and self.mask[int(point[1]), int(point[0])] == 1
        ]
        
        # 根据转向方向调整顺序
        if turn_direction:
            valid_points = valid_points[::-1]
        
        return valid_points
    
    def _find_nearest_weed(self, p, coordinates, r):
        """找到最近的杂草"""
        if len(coordinates) == 0:
            return None
        
        p = np.array(p)
        coordinates = np.array(coordinates)
        distances = np.sqrt(np.sum((coordinates - p) ** 2, axis=1))
        valid_indices = np.where(distances >= 2 * r)[0]
        
        if len(valid_indices) == 0:
            return None
        
        nearest_index = valid_indices[np.argmin(distances[valid_indices])]
        return coordinates[nearest_index]
    
    def _run_bcp(self, env, result, max_steps, render):
        """运行BCP算法 - 基本覆盖路径"""
        y_offset = -self.diagonal_length + self.agent_width / 2
        turn_direction = False
        starting = False
        
        while y_offset < self.diagonal_length and result['steps'] < max_steps:
            # 生成路径线
            valid_points = self._generate_path_line(y_offset, turn_direction)
            
            if valid_points:
                if not starting:
                    # 初始位置设置
                    starting = True
                    rad_n = self.real_radians if not turn_direction else self.real_radians + np.pi
                else:
                    # Dubins导航到起始点
                    rad_n = self.real_radians if not turn_direction else self.real_radians + np.pi
                    rad_n = (rad_n + math.pi) % (2 * math.pi) - math.pi
                    done = self._dubins_navigate(
                        [valid_points[0][0], valid_points[0][1], rad_n],
                        self.turning_radius, env, result, render
                    )
                    if done:
                        break
                
                # 执行路径点
                p_i = 0
                while p_i < len(valid_points) and result['steps'] < max_steps:
                    done = self._navigate(valid_points[p_i], env, result, render)
                    if done:
                        break
                    p_i += 1
                
                turn_direction = not turn_direction
            
            # 移动到下一行
            y_offset += self.agent_width
        
        return result
    
    def _run_jump(self, env, result, max_steps, render):
        """运行JUMP算法 - 跳跃式覆盖"""
        y_offset = -self.diagonal_length + self.sight_width / 2
        turn_direction = False
        starting = False
        
        while y_offset < self.diagonal_length and result['steps'] < max_steps:
            # 生成路径线
            valid_points = self._generate_path_line(y_offset, turn_direction)
            
            if valid_points:
                if not starting:
                    starting = True
                    rad_n = self.real_radians if not turn_direction else self.real_radians + np.pi
                else:
                    # Dubins导航到起始点
                    rad_n = self.real_radians if not turn_direction else self.real_radians + np.pi
                    rad_n = (rad_n + math.pi) % (2 * math.pi) - math.pi
                    done = self._dubins_navigate(
                        [valid_points[0][0], valid_points[0][1], rad_n],
                        self.turning_radius, env, result, render
                    )
                    if done:
                        break
                
                # 执行路径点，检查是否需要跳跃到杂草
                p_i = 0
                while p_i < len(valid_points) and result['steps'] < max_steps:
                    # 检查前方是否有杂草
                    if len(self.discovered) > 0:
                        # 简化：找最近的杂草
                        weed = self._find_nearest_weed(self.agent_position, self.discovered, 0)
                        if weed is not None and np.random.random() < 0.3:  # 30%概率跳跃
                            done = self._dubins_navigate(
                                [weed[0], weed[1], rad_n],
                                self.turning_radius, env, result, render
                            )
                            if done:
                                break
                            # 跳跃后继续路径
                            if p_i + 4 < len(valid_points):
                                done = self._dubins_navigate(
                                    [valid_points[p_i + 4][0], valid_points[p_i + 4][1], rad_n],
                                    self.turning_radius, env, result, render
                                )
                                if done:
                                    break
                                p_i += 5
                                continue
                    
                    done = self._navigate(valid_points[p_i], env, result, render)
                    if done:
                        break
                    p_i += 1
                
                turn_direction = not turn_direction
            
            # 移动到下一行
            y_offset += self.sight_width / 2
        
        result['discovered_weeds'] = self.discovered.copy()
        return result
    
    def _run_snake(self, env, result, max_steps, render, is_r_snake):
        """运行SNAKE或R_SNAKE算法 - 蛇形覆盖"""
        y_offset = -self.diagonal_length + self.agent_width / 2
        turn_direction = False
        starting = False
        
        while y_offset < self.diagonal_length and result['steps'] < max_steps:
            # 生成路径线
            valid_points = self._generate_path_line(y_offset, turn_direction)
            
            if valid_points:
                if not starting:
                    starting = True
                    rad_n = self.real_radians if not turn_direction else self.real_radians + np.pi
                else:
                    # Dubins导航到起始点
                    rad_n = self.real_radians if not turn_direction else self.real_radians + np.pi
                    rad_n = (rad_n + math.pi) % (2 * math.pi) - math.pi
                    done = self._dubins_navigate(
                        [valid_points[0][0], valid_points[0][1], rad_n],
                        self.turning_radius, env, result, render
                    )
                    if done:
                        break
                
                # 执行路径点，贪心搜索杂草
                p_i = 0
                while p_i < len(valid_points) and result['steps'] < max_steps:
                    # 获取前方的杂草
                    forward = self._get_forward_weeds(self.agent_position, rad_n, is_r_snake)
                    weed = self._find_nearest_weed(self.agent_position, forward, self.turning_radius)
                    
                    if weed is not None:
                        # 贪心前往杂草
                        done = self._dubins_navigate(
                            [weed[0], weed[1], rad_n],
                            self.turning_radius, env, result, render
                        )
                        if done:
                            break
                        
                        # 重新生成前方路径
                        points = []
                        start_point = self.agent_position
                        for i in range(50):
                            next_point = [
                                start_point[0] + i * np.cos(rad_n),
                                start_point[1] + i * np.sin(rad_n)
                            ]
                            if self.polygon.contains(Point(next_point[0], next_point[1])):
                                points.append(next_point)
                            else:
                                break
                        valid_points = points
                        p_i = 0
                    
                    if p_i < len(valid_points):
                        done = self._navigate(valid_points[p_i], env, result, render)
                        if done:
                            break
                    p_i += 1
                
                turn_direction = not turn_direction
            
            # 移动到下一行
            if is_r_snake:
                y_offset += self.sight_width / 2 + self.agent_width / 2
            else:
                y_offset += self.sight_width / 2
        
        result['discovered_weeds'] = self.discovered.copy()
        return result
    
    def _get_forward_weeds(self, position, rad, is_r_snake):
        """获取前方的杂草"""
        if not self.discovered:
            return []
        
        rad_vector = np.array([np.cos(rad), np.sin(rad)])
        forward = [p for p in self.discovered if np.dot(p - position, rad_vector) > 0]
        
        if is_r_snake:
            # R_SNAKE还要检查垂直方向
            upward_vector = np.array([
                np.cos(self.real_radians + np.pi / 2),
                np.sin(self.real_radians + np.pi / 2)
            ])
            forward = [
                p for p in forward
                if np.dot(p - position, upward_vector) > -1.5 * self.sight_width
            ]
        
        return forward
    
    def _run_react(self, env, result, max_steps, render):
        """运行REACT算法 - 反应式覆盖"""
        times = 0
        max_attempts = min(50, max_steps // 20)
        
        while times < max_attempts and result['steps'] < max_steps:
            # REACT: 随机探索 + 反应式响应杂草
            if self.discovered and np.random.random() < 0.3:
                # 30%概率响应杂草
                weed = self._find_nearest_weed(self.agent_position, self.discovered, 0)
                if weed is not None:
                    goal_rad = math.atan2(
                        weed[1] - self.agent_position[1],
                        weed[0] - self.agent_position[0]
                    )
                    done = self._dubins_navigate(
                        [weed[0], weed[1], goal_rad],
                        self.turning_radius, env, result, render
                    )
                    if done:
                        break
            else:
                # 随机目标
                if self.polygon is not None:
                    min_x, min_y = self.farm_vertices.min(axis=0)
                    max_x, max_y = self.farm_vertices.max(axis=0)
                    rand_goal = [
                        np.random.uniform(min_x, max_x),
                        np.random.uniform(min_y, max_y)
                    ]
                else:
                    rand_goal = [
                        np.random.uniform(0, self.W),
                        np.random.uniform(0, self.H)
                    ]
                
                # 生成到随机目标的路径
                start = self.agent_position
                end = rand_goal
                line = LineString([start, end])
                x_points = []
                for i in np.arange(0, min(line.length, 50), 1):
                    interpolated_point = np.array(line.interpolate(i).coords[0])
                    x_points.append(interpolated_point)
                
                valid_points = [
                    point for point in x_points
                    if 0 <= int(point[1]) < self.H and 0 <= int(point[0]) < self.W
                    and self.mask[int(point[1]), int(point[0])] == 1
                ]
                
                if valid_points:
                    goal_rad = math.atan2(
                        end[1] - start[1],
                        end[0] - start[0]
                    )
                    # Dubins导航到第一个有效点
                    done = self._dubins_navigate(
                        [valid_points[0][0], valid_points[0][1], goal_rad],
                        self.turning_radius, env, result, render
                    )
                    if done:
                        break
                    
                    # 执行路径，同时检查杂草
                    for point in valid_points[1:]:
                        weed = self._find_nearest_weed(self.agent_position, self.discovered, 0)
                        if weed is not None:
                            done = self._dubins_navigate(
                                [weed[0], weed[1], self.rad],
                                self.turning_radius, env, result, render
                            )
                            if done:
                                break
                            break  # 发现杂草后重新规划
                        
                        done = self._navigate(point, env, result, render)
                        if done:
                            break
            
            times += 1
        
        result['discovered_weeds'] = self.discovered.copy()
        return result