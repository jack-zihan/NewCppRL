#!/usr/bin/env python3
"""
Rules_new vs Rules_new1 并行对比测试
同时运行两个系统，逐步对比每个环节的差异
"""
import sys
import os
import math
import numpy as np
import gymnasium as gym
import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from omegaconf import DictConfig
import json
from datetime import datetime
from matplotlib.path import Path as MPath

# 添加项目根目录
project_root = Path(__file__).parents[1]
sys.path.insert(0, str(project_root))

# 设置环境变量
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'

import envs  # 注册老版本环境
from rules_new.algorithms import JumpPlanner
from rules_new.experiment.config_manager import ConfigManager


class RulesNewSimulator:
    """模拟rules_new的执行逻辑"""
    
    def __init__(self, env, task_type='JUMP'):
        self.env = env
        self.task_type = task_type
        
        # 从环境获取参数（参考rules_new/jump_path.py）
        self.agent_width = 5  # Config.CAR_WIDTH
        self.sight_width = 24  # Config.SIGHT_WIDTH
        self.sight_length = 24  # Config.SIGHT_LENGTH
        
        # 获取环境信息
        self.agent_position = [self.env.agent.y, self.env.agent.x]  # 注意：rules_new使用[y,x]
        self.W = 600  # Config.W
        self.H = 600  # Config.H
        
        # 计算turning_radius
        w_max_rad = abs(self.env.w_range.max) * (math.pi / 180)
        self.turning_radius = self.env.v_range.max / w_max_rad
        
        # 获取farm_vertices
        self.farm_vertices = self.env.min_area_rect[0][:, 0, ::-1]
        
        # 初始化状态
        self.discovered = []
        self.rad = 0
        self.y_offset = 0
        self.turn_direction = False  # rules_new初始为False（称为turn）
        self.real_radians = 0
        self.diagonal_length = 0
        
        # 初始化覆盖模式
        self._initialize_coverage()
        
    def _initialize_coverage(self):
        """初始化覆盖模式（参考rules_new）"""
        # 找最长边
        longest_edge = self._find_longest_edge(self.farm_vertices)
        dx = longest_edge[1][0] - longest_edge[0][0]
        dy = longest_edge[1][1] - longest_edge[0][1]
        self.real_radians = np.arctan2(dy, dx)
        self.real_radians = self.real_radians % (2 * np.pi) if self.real_radians >= 0 else (self.real_radians + 2 * np.pi) % (2 * np.pi)
        
        # 计算对角线长度
        min_vals = self.farm_vertices.min(axis=0)
        max_vals = self.farm_vertices.max(axis=0)
        self.diagonal_length = np.sqrt((max_vals[0] - min_vals[0]) ** 2 + (max_vals[1] - min_vals[1]) ** 2)
        
        # 初始y_offset
        self.y_offset = -self.diagonal_length + self.agent_width / 2
        
        # 创建polygon mask
        poly_path = MPath(self.farm_vertices)
        y, x = np.mgrid[:self.H, :self.W]
        coords = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        self.polygon_mask = np.zeros((self.H, self.W))
        self.polygon_mask[poly_path.contains_points(coords).reshape(self.H, self.W)] = 1
        
    def _find_longest_edge(self, vertices):
        """找最长边"""
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
    
    def get_next_waypoint(self):
        """获取下一个路径点（简化版JUMP算法）"""
        # 生成当前行的路径
        start = [0, 0]
        end = np.array([100 * np.cos(self.real_radians), 100 * np.sin(self.real_radians)])
        
        new_start = [
            start[0] + self.y_offset * np.cos(self.real_radians + np.pi / 2) - self.diagonal_length * np.cos(self.real_radians),
            start[1] + self.y_offset * np.sin(self.real_radians + np.pi / 2) - self.diagonal_length * np.sin(self.real_radians)
        ]
        new_end = [
            end[0] + self.y_offset * np.cos(self.real_radians + np.pi / 2) + self.diagonal_length * np.cos(self.real_radians),
            end[1] + self.y_offset * np.sin(self.real_radians + np.pi / 2) + self.diagonal_length * np.sin(self.real_radians)
        ]
        
        # 生成线上的点
        line_points = []
        direction = np.array(new_end) - np.array(new_start)
        length = np.linalg.norm(direction)
        
        for i in np.arange(0, length, 1):
            interpolated_point = np.array(new_start) + (i / length) * direction
            line_points.append(interpolated_point)
        
        # 过滤有效点
        valid_points = [
            point for point in line_points 
            if (0 <= int(point[1]) < self.H and 
                0 <= int(point[0]) < self.W and 
                self.polygon_mask[int(point[1]), int(point[0])] == 1)
        ]
        
        # 根据方向调整
        if self.turn_direction:
            valid_points = valid_points[::-1]
        
        # 返回第一个有效点
        if valid_points:
            return valid_points[0]
        else:
            # 移动到下一行
            self.y_offset += self.sight_width / 2
            self.turn_direction = not self.turn_direction
            if self.y_offset < self.diagonal_length:
                return self.get_next_waypoint()  # 递归获取下一行的点
            else:
                return None  # 完成


class SideBySideComparison:
    """并行对比测试器"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.differences = []
        
        # 创建输出目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = project_root / 'logs' / 'side_by_side' / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def compare_values(self, name: str, val1: Any, val2: Any, tolerance: float = 1e-6) -> bool:
        """比较两个值"""
        equal = False
        
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            diff = abs(val1 - val2)
            equal = diff <= tolerance
            if not equal:
                self.differences.append({
                    'name': name,
                    'rules': float(val1),
                    'rules_new': float(val2),
                    'diff': float(diff)
                })
        elif isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
            if len(val1) == len(val2):
                diffs = [abs(a - b) for a, b in zip(val1, val2) if isinstance(a, (int, float))]
                max_diff = max(diffs) if diffs else 0
                equal = max_diff <= tolerance
                if not equal:
                    self.differences.append({
                        'name': name,
                        'rules': val1,
                        'rules_new': val2,
                        'diff': float(max_diff)
                    })
        elif isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            if val1.shape == val2.shape:
                max_diff = np.abs(val1 - val2).max()
                equal = np.allclose(val1, val2, atol=tolerance)
                if not equal:
                    self.differences.append({
                        'name': name,
                        'rules': 'array',
                        'rules_new': 'array',
                        'diff': float(max_diff)
                    })
        
        if equal:
            print(f"✅ {name}: 一致")
        else:
            print(f"❌ {name}: 不一致 (差异: {self.differences[-1]['diff'] if self.differences else 'N/A'})")
        
        return equal
    
    def test_initialization(self, seed: int = 42):
        """测试初始化差异"""
        print("\n" + "="*60)
        print("🔍 测试初始化")
        print("="*60)
        
        # 创建环境
        env_config_path = project_root / 'configs' / 'env_config.yaml'
        cfg = DictConfig(yaml.load(open(env_config_path), Loader=yaml.FullLoader))
        env = gym.make(render_mode='rgb_array', **cfg.env.params)
        obs, info = env.reset(seed=seed)
        
        # Rules_new系统
        rules_new_sim = RulesNewSimulator(env, task_type='JUMP')
        
        # Rules_new1系统
        base_config = self.config_manager.load_base_config()
        jump_config = self.config_manager.load_algorithm_config('jump')
        rules_new1_alg = JumpPlanner(jump_config, base_config)
        
        # 准备rules_new1的初始状态
        agent_position = [float(env.agent.x), float(env.agent.y)]  # 注意：rules_new1使用[x,y]
        farm_vertices = env.min_area_rect[0][:, 0, ::-1]
        
        initial_state = {
            'agent_position': agent_position,
            'agent_direction': float(env.agent.direction),
            'discovered_weeds': [],
            'weed_count': int(env.map_weed.sum()),
            'coverage_rate': 0.0,
            'farm_vertices': farm_vertices,
            'seed': seed,
            'maps': {}
        }
        
        rules_new1_alg.reset(initial_state)
        
        # 对比初始化参数
        print("\n初始化参数对比：")
        
        # 对比agent位置（注意坐标系差异）
        self.compare_values("agent_position_x", 
                          rules_new_sim.agent_position[1],  # rules_new的x是第二个
                          agent_position[0])  # rules_new1的x是第一个
        self.compare_values("agent_position_y",
                          rules_new_sim.agent_position[0],  # rules_new的y是第一个
                          agent_position[1])  # rules_new1的y是第二个
        
        # 对比turning_radius
        self.compare_values("turning_radius",
                          rules_new_sim.turning_radius,
                          5.0)  # rules_new1可能使用默认值
        
        # 对比real_radians
        self.compare_values("real_radians",
                          rules_new_sim.real_radians,
                          rules_new1_alg.real_radians)
        
        # 对比diagonal_length
        self.compare_values("diagonal_length",
                          rules_new_sim.diagonal_length,
                          rules_new1_alg.diagonal_length)
        
        # 对比y_offset
        self.compare_values("initial_y_offset",
                          rules_new_sim.y_offset,
                          rules_new1_alg.y_offset)
        
        # 对比turn_direction
        print(f"\nturn_direction: rules={rules_new_sim.turn_direction}, rules_new={rules_new1_alg.turn_direction}")
        
        env.close()
        return {
            'rules': {
                'agent_position': rules_new_sim.agent_position,
                'turning_radius': rules_new_sim.turning_radius,
                'real_radians': rules_new_sim.real_radians,
                'y_offset': rules_new_sim.y_offset,
                'turn_direction': rules_new_sim.turn_direction
            },
            'rules_new': {
                'agent_position': agent_position,
                'turning_radius': 5.0,
                'real_radians': rules_new1_alg.real_radians,
                'y_offset': rules_new1_alg.y_offset,
                'turn_direction': rules_new1_alg.turn_direction
            }
        }
    
    def test_first_waypoint(self, seed: int = 42):
        """测试第一个waypoint的差异"""
        print("\n" + "="*60)
        print("🔍 测试第一个Waypoint")
        print("="*60)
        
        # 创建环境
        env_config_path = project_root / 'configs' / 'env_config.yaml'
        cfg = DictConfig(yaml.load(open(env_config_path), Loader=yaml.FullLoader))
        env = gym.make(render_mode='rgb_array', **cfg.env.params)
        obs, info = env.reset(seed=seed)
        
        # Rules_new系统
        rules_new_sim = RulesNewSimulator(env, task_type='JUMP')
        waypoint_old = rules_new_sim.get_next_waypoint()
        
        # Rules_new1系统
        base_config = self.config_manager.load_base_config()
        jump_config = self.config_manager.load_algorithm_config('jump')
        rules_new1_alg = JumpPlanner(jump_config, base_config)
        
        agent_position = [float(env.agent.x), float(env.agent.y)]
        farm_vertices = env.min_area_rect[0][:, 0, ::-1]
        
        initial_state = {
            'agent_position': agent_position,
            'agent_direction': float(env.agent.direction),
            'discovered_weeds': [],
            'weed_count': int(env.map_weed.sum()),
            'coverage_rate': 0.0,
            'farm_vertices': farm_vertices,
            'seed': seed,
            'maps': {}
        }
        
        rules_new1_alg.reset(initial_state)
        waypoint_new = rules_new1_alg.plan_next_waypoint(initial_state)
        
        print(f"\nWaypoint对比：")
        print(f"  rules:  {waypoint_old}")
        print(f"  rules_new: {waypoint_new}")
        
        if waypoint_old is not None and waypoint_new is not None:
            self.compare_values("waypoint", waypoint_old, waypoint_new, tolerance=1.0)
        
        env.close()
        return {
            'rules': waypoint_old,
            'rules_new': waypoint_new
        }
    
    def test_multiple_steps(self, seed: int = 42, num_steps: int = 10):
        """测试多步执行的差异"""
        print("\n" + "="*60)
        print(f"🔍 测试{num_steps}步执行")
        print("="*60)
        
        # 创建环境
        env_config_path = project_root / 'configs' / 'env_config.yaml'
        cfg = DictConfig(yaml.load(open(env_config_path), Loader=yaml.FullLoader))
        env = gym.make(render_mode='rgb_array', **cfg.env.params)
        obs, info = env.reset(seed=seed)
        
        # 初始化两个系统
        rules_new_sim = RulesNewSimulator(env, task_type='JUMP')
        
        base_config = self.config_manager.load_base_config()
        jump_config = self.config_manager.load_algorithm_config('jump')
        rules_new1_alg = JumpPlanner(jump_config, base_config)
        
        # 记录轨迹
        trajectory_old = []
        trajectory_new = []
        
        for step in range(num_steps):
            print(f"\nStep {step}:")
            
            # Rules_new系统
            waypoint_old = rules_new_sim.get_next_waypoint()
            if waypoint_old is not None:
                trajectory_old.append(waypoint_old)
                # 更新agent位置（简化）
                rules_new_sim.agent_position = waypoint_old
            
            # Rules_new1系统
            agent_position = [float(env.agent.x), float(env.agent.y)]
            current_state = {
                'agent_position': agent_position,
                'agent_direction': float(env.agent.direction),
                'discovered_weeds': [],
                'weed_count': 0,
                'coverage_rate': 0.0,
                'farm_vertices': env.min_area_rect[0][:, 0, ::-1],
                'seed': seed,
                'maps': {}
            }
            
            if step == 0:
                rules_new1_alg.reset(current_state)
            
            waypoint_new = rules_new1_alg.plan_next_waypoint(current_state)
            if waypoint_new is not None:
                trajectory_new.append(waypoint_new)
                # 简单的环境step（实际应该转换为动作）
                # 这里简化处理，直接移动agent
                env.agent.x = waypoint_new[0]
                env.agent.y = waypoint_new[1]
            
            # 对比waypoint
            if waypoint_old is not None and waypoint_new is not None:
                dist = np.linalg.norm(np.array(waypoint_old) - np.array(waypoint_new))
                print(f"  Waypoint距离: {dist:.2f}")
            
            if waypoint_old is None and waypoint_new is None:
                print("  两个系统都终止了")
                break
        
        print(f"\n轨迹长度：")
        print(f"  rules:  {len(trajectory_old)}")
        print(f"  rules_new: {len(trajectory_new)}")
        
        env.close()
        return {
            'trajectory_old': trajectory_old[:5],  # 保存前5个点
            'trajectory_new': trajectory_new[:5]
        }
    
    def save_results(self):
        """保存结果"""
        # 保存差异
        if self.differences:
            diff_file = self.output_dir / 'differences.json'
            with open(diff_file, 'w') as f:
                json.dump(self.differences, f, indent=2, default=str)
        
        # 生成报告
        report_file = self.output_dir / 'comparison_report.txt'
        with open(report_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("Rules_new vs Rules_new1 对比报告\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"发现差异: {len(self.differences)}个\n\n")
            
            if self.differences:
                f.write("主要差异:\n")
                for diff in self.differences:
                    f.write(f"  - {diff['name']}: ")
                    f.write(f"rules={diff['rules']}, ")
                    f.write(f"rules_new={diff['rules_new']}, ")
                    f.write(f"差异={diff['diff']}\n")
        
        print(f"\n结果已保存到: {self.output_dir}")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "="*80)
        print("Rules_new vs Rules_new1 并行对比测试")
        print("="*80)
        
        # 1. 初始化测试
        init_results = self.test_initialization(seed=42)
        
        # 2. 第一个waypoint测试
        waypoint_results = self.test_first_waypoint(seed=42)
        
        # 3. 多步执行测试
        trajectory_results = self.test_multiple_steps(seed=42, num_steps=10)
        
        # 保存结果
        self.save_results()
        
        print("\n" + "="*80)
        print("测试完成！")
        print(f"发现差异: {len(self.differences)}个")
        if self.differences:
            print("\n关键差异:")
            for diff in self.differences[:5]:  # 显示前5个
                print(f"  - {diff['name']}: 差异={diff['diff']}")
        print(f"\n详细结果: {self.output_dir}")
        print("="*80)


if __name__ == "__main__":
    comparison = SideBySideComparison()
    comparison.run_all_tests()