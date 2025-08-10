#!/usr/bin/env python3
"""
Rules_new vs Rules_new1 详细对比测试
逐层递进式排查差异，从环境初始化到完整执行
"""
import sys
import os
import numpy as np
import gymnasium as gym
import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from omegaconf import DictConfig
import json
from datetime import datetime

# 添加项目根目录
project_root = Path(__file__).parents[1]
sys.path.insert(0, str(project_root))

# 设置环境变量
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'

import envs  # 注册老版本环境
from rules_new.algorithms import JumpPlanner
from rules_new.experiment.config_manager import ConfigManager


class DetailedComparison:
    """详细对比测试器"""
    
    def __init__(self, output_dir: Optional[str] = None):
        """初始化对比器"""
        self.differences = []
        self.test_results = {}
        
        # 创建输出目录
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = project_root / 'logs' / 'detailed_comparison' / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置管理器
        self.config_manager = ConfigManager()
        
    def compare_values(self, name: str, val1: Any, val2: Any, 
                      tolerance: float = 1e-6, context: str = "") -> bool:
        """
        精确对比两个值
        
        Returns:
            True if values are equal within tolerance
        """
        equal = False
        diff_info = {
            'name': name,
            'context': context,
            'rules': None,
            'rules_new': None,
            'diff': None,
            'equal': False
        }
        
        try:
            if val1 is None and val2 is None:
                equal = True
            elif val1 is None or val2 is None:
                diff_info['rules'] = val1
                diff_info['rules_new'] = val2
                diff_info['diff'] = 'One is None'
            elif isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                if val1.shape != val2.shape:
                    diff_info['rules'] = f"shape={val1.shape}"
                    diff_info['rules_new'] = f"shape={val2.shape}"
                    diff_info['diff'] = 'Shape mismatch'
                else:
                    max_diff = np.abs(val1 - val2).max()
                    equal = np.allclose(val1, val2, atol=tolerance)
                    diff_info['rules'] = val1.tolist() if val1.size < 10 else f"array{val1.shape}"
                    diff_info['rules_new'] = val2.tolist() if val2.size < 10 else f"array{val2.shape}"
                    diff_info['diff'] = float(max_diff)
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                diff = abs(val1 - val2)
                equal = diff <= tolerance
                diff_info['rules'] = float(val1)
                diff_info['rules_new'] = float(val2)
                diff_info['diff'] = float(diff)
            elif isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
                if len(val1) != len(val2):
                    diff_info['rules'] = f"len={len(val1)}"
                    diff_info['rules_new'] = f"len={len(val2)}"
                    diff_info['diff'] = 'Length mismatch'
                else:
                    diffs = [abs(a - b) for a, b in zip(val1, val2) if isinstance(a, (int, float))]
                    max_diff = max(diffs) if diffs else 0
                    equal = max_diff <= tolerance
                    diff_info['rules'] = val1
                    diff_info['rules_new'] = val2
                    diff_info['diff'] = float(max_diff)
            else:
                equal = val1 == val2
                diff_info['rules'] = str(val1)
                diff_info['rules_new'] = str(val2)
                diff_info['diff'] = 'Type or value mismatch'
        
        except Exception as e:
            diff_info['diff'] = f"Comparison error: {e}"
        
        diff_info['equal'] = equal
        
        if not equal:
            self.differences.append(diff_info)
            print(f"❌ {name}: {diff_info['diff']}")
        else:
            print(f"✅ {name}: Equal")
        
        return equal
    
    def test_environment_initialization(self, seed: int = 42) -> Dict[str, Any]:
        """
        测试环境初始化的一致性
        """
        print("\n" + "="*60)
        print("🔍 测试环境初始化")
        print("="*60)
        
        # 创建老版本环境（两个系统都使用）
        env_config_path = project_root / 'configs' / 'env_config.yaml'
        cfg = DictConfig(yaml.load(open(env_config_path), Loader=yaml.FullLoader))
        
        env = gym.make(
            render_mode='rgb_array',
            **cfg.env.params,
        )
        
        # 重置环境
        obs, info = env.reset(seed=seed)
        
        # 提取环境属性
        env_state = {
            'agent_x': float(env.agent.x),
            'agent_y': float(env.agent.y),
            'agent_direction': float(env.agent.direction),
            'map_weed_sum': int(env.map_weed.sum()) if hasattr(env, 'map_weed') else 0,
            'map_obstacle_sum': int(env.map_obstacle.sum()) if hasattr(env, 'map_obstacle') else 0,
            'min_area_rect_exists': hasattr(env, 'min_area_rect') and env.min_area_rect is not None,
            'obs_shape': obs['observation'].shape if isinstance(obs, dict) else obs.shape,
            'obs_type': type(obs).__name__
        }
        
        # 提取farm_vertices
        if hasattr(env, 'min_area_rect') and env.min_area_rect is not None:
            farm_vertices_raw = env.min_area_rect[0][:, 0, ::-1]
            env_state['farm_vertices'] = farm_vertices_raw.tolist()
        else:
            env_state['farm_vertices'] = None
        
        print("\n环境初始状态：")
        for key, value in env_state.items():
            if key != 'farm_vertices':
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value[:2] if value else None}...")  # 只显示前两个点
        
        # 保存结果
        self.test_results['env_initialization'] = env_state
        
        env.close()
        return env_state
    
    def test_algorithm_initialization(self, seed: int = 42) -> Dict[str, Any]:
        """
        测试算法初始化（rules_new1方式）
        """
        print("\n" + "="*60)
        print("🔍 测试算法初始化")
        print("="*60)
        
        # 加载配置
        base_config = self.config_manager.load_base_config()
        jump_config = self.config_manager.load_algorithm_config('jump')
        
        # 创建算法实例
        algorithm = JumpPlanner(jump_config, base_config)
        
        # 创建环境
        env_config_path = project_root / 'configs' / 'env_config.yaml'
        cfg = DictConfig(yaml.load(open(env_config_path), Loader=yaml.FullLoader))
        env = gym.make(render_mode='rgb_array', **cfg.env.params)
        
        # 重置环境
        obs, info = env.reset(seed=seed)
        
        # 准备初始状态（模拟experiment_runner的extract_state_from_environment）
        agent_position = [float(env.agent.x), float(env.agent.y)]
        agent_direction = float(env.agent.direction)
        
        if hasattr(env, 'min_area_rect') and env.min_area_rect is not None:
            farm_vertices = env.min_area_rect[0][:, 0, ::-1]
        else:
            farm_vertices = np.array([[50, 50], [350, 50], [350, 350], [50, 350]])
        
        initial_state = {
            'agent_position': agent_position,
            'agent_direction': agent_direction,
            'discovered_weeds': [],
            'weed_count': int(env.map_weed.sum()) if hasattr(env, 'map_weed') else 0,
            'coverage_rate': 0.0,
            'farm_vertices': farm_vertices,
            'seed': seed,
            'maps': {}
        }
        
        # 初始化算法
        algorithm.reset(initial_state)
        
        # 获取第一个waypoint
        waypoint = algorithm.plan_next_waypoint(initial_state)
        
        algo_state = {
            'algorithm_name': 'JUMP',
            'initial_position': agent_position,
            'initial_direction': agent_direction,
            'farm_vertices_shape': farm_vertices.shape if isinstance(farm_vertices, np.ndarray) else None,
            'first_waypoint': waypoint,
            'y_offset': algorithm.y_offset if hasattr(algorithm, 'y_offset') else None,
            'turn_direction': algorithm.turn_direction if hasattr(algorithm, 'turn_direction') else None,
            'real_radians': algorithm.real_radians if hasattr(algorithm, 'real_radians') else None
        }
        
        print("\n算法初始状态：")
        for key, value in algo_state.items():
            print(f"  {key}: {value}")
        
        # 保存结果
        self.test_results['algorithm_initialization'] = algo_state
        
        env.close()
        return algo_state
    
    def test_single_step_execution(self, seed: int = 42) -> Dict[str, Any]:
        """
        测试单步执行的一致性
        """
        print("\n" + "="*60)
        print("🔍 测试单步执行")
        print("="*60)
        
        # 初始化
        base_config = self.config_manager.load_base_config()
        jump_config = self.config_manager.load_algorithm_config('jump')
        algorithm = JumpPlanner(jump_config, base_config)
        
        # 创建环境
        env_config_path = project_root / 'configs' / 'env_config.yaml'
        cfg = DictConfig(yaml.load(open(env_config_path), Loader=yaml.FullLoader))
        env = gym.make(render_mode='rgb_array', **cfg.env.params)
        
        # 重置
        obs, info = env.reset(seed=seed)
        
        # 准备初始状态
        agent_position = [float(env.agent.x), float(env.agent.y)]
        agent_direction = float(env.agent.direction)
        farm_vertices = env.min_area_rect[0][:, 0, ::-1] if hasattr(env, 'min_area_rect') else np.array([[50, 50], [350, 50], [350, 350], [50, 350]])
        
        initial_state = {
            'agent_position': agent_position,
            'agent_direction': agent_direction,
            'discovered_weeds': [],
            'weed_count': int(env.map_weed.sum()) if hasattr(env, 'map_weed') else 0,
            'coverage_rate': 0.0,
            'farm_vertices': farm_vertices,
            'seed': seed,
            'maps': {}
        }
        
        # 初始化算法
        algorithm.reset(initial_state)
        
        # 获取waypoint
        waypoint = algorithm.plan_next_waypoint(initial_state)
        
        # 转换为动作（这里简化处理，实际需要参考experiment_runner的_waypoint_to_action）
        if waypoint:
            # 简单的连续动作：[velocity, angular_velocity]
            direction_to_waypoint = np.arctan2(waypoint[1] - agent_position[1], 
                                              waypoint[0] - agent_position[0])
            distance = np.linalg.norm(np.array(waypoint) - np.array(agent_position))
            
            # 简化的动作
            action = [min(distance * 0.1, 3.5), direction_to_waypoint * 10]
        else:
            action = [0.0, 0.0]
        
        # 执行步骤
        obs_new, reward, terminated, truncated, info_new = env.step(action)
        
        step_result = {
            'initial_position': agent_position,
            'waypoint': waypoint,
            'action': action,
            'new_position': [float(env.agent.x), float(env.agent.y)],
            'reward': float(reward),
            'terminated': terminated,
            'truncated': truncated,
            'coverage_rate': info_new.get('coverage_rate', 0.0) if info_new else 0.0
        }
        
        print("\n单步执行结果：")
        for key, value in step_result.items():
            print(f"  {key}: {value}")
        
        # 保存结果
        self.test_results['single_step'] = step_result
        
        env.close()
        return step_result
    
    def test_multiple_steps(self, seed: int = 42, num_steps: int = 10) -> Dict[str, Any]:
        """
        测试多步执行的一致性
        """
        print("\n" + "="*60)
        print(f"🔍 测试{num_steps}步执行")
        print("="*60)
        
        # 初始化
        base_config = self.config_manager.load_base_config()
        jump_config = self.config_manager.load_algorithm_config('jump')
        algorithm = JumpPlanner(jump_config, base_config)
        
        # 创建环境
        env_config_path = project_root / 'configs' / 'env_config.yaml'
        cfg = DictConfig(yaml.load(open(env_config_path), Loader=yaml.FullLoader))
        env = gym.make(render_mode='rgb_array', **cfg.env.params)
        
        # 重置
        obs, info = env.reset(seed=seed)
        
        # 准备初始状态
        def extract_state():
            return {
                'agent_position': [float(env.agent.x), float(env.agent.y)],
                'agent_direction': float(env.agent.direction),
                'discovered_weeds': [],
                'weed_count': int(env.map_weed.sum()) if hasattr(env, 'map_weed') else 0,
                'coverage_rate': info.get('coverage_rate', 0.0) if info else 0.0,
                'farm_vertices': env.min_area_rect[0][:, 0, ::-1] if hasattr(env, 'min_area_rect') else np.array([[50, 50], [350, 50], [350, 350], [50, 350]]),
                'seed': seed,
                'maps': {}
            }
        
        initial_state = extract_state()
        algorithm.reset(initial_state)
        
        # 执行多步
        trajectory = []
        total_reward = 0.0
        
        for step in range(num_steps):
            current_state = extract_state()
            
            # 获取waypoint
            waypoint = algorithm.plan_next_waypoint(current_state)
            
            if waypoint is None:
                print(f"Step {step}: Algorithm terminated")
                break
            
            # 简化的动作转换
            agent_pos = current_state['agent_position']
            direction_to_waypoint = np.arctan2(waypoint[1] - agent_pos[1], 
                                              waypoint[0] - agent_pos[0])
            distance = np.linalg.norm(np.array(waypoint) - np.array(agent_pos))
            action = [min(distance * 0.1, 3.5), direction_to_waypoint * 10]
            
            # 执行
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # 记录轨迹
            trajectory.append({
                'step': step,
                'position': [float(env.agent.x), float(env.agent.y)],
                'waypoint': waypoint,
                'reward': float(reward),
                'coverage': info.get('coverage_rate', 0.0) if info else 0.0
            })
            
            print(f"Step {step}: pos={trajectory[-1]['position']}, reward={reward:.3f}, coverage={trajectory[-1]['coverage']:.3f}")
            
            if terminated or truncated:
                print(f"Episode ended at step {step}")
                break
        
        multi_step_result = {
            'num_steps_executed': len(trajectory),
            'total_reward': total_reward,
            'final_position': trajectory[-1]['position'] if trajectory else initial_state['agent_position'],
            'final_coverage': trajectory[-1]['coverage'] if trajectory else 0.0,
            'trajectory_sample': trajectory[:3] if len(trajectory) > 3 else trajectory  # 保存前3步作为样本
        }
        
        print(f"\n{num_steps}步执行总结：")
        print(f"  执行步数: {multi_step_result['num_steps_executed']}")
        print(f"  总奖励: {multi_step_result['total_reward']:.3f}")
        print(f"  最终覆盖率: {multi_step_result['final_coverage']:.3f}")
        
        # 保存结果
        self.test_results['multiple_steps'] = multi_step_result
        
        env.close()
        return multi_step_result
    
    def save_results(self):
        """保存测试结果"""
        # 保存JSON结果
        results_file = self.output_dir / 'test_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        # 保存差异报告
        if self.differences:
            diff_file = self.output_dir / 'differences.json'
            with open(diff_file, 'w') as f:
                json.dump(self.differences, f, indent=2, default=str)
        
        # 生成摘要报告
        summary_file = self.output_dir / 'summary.txt'
        with open(summary_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("详细对比测试摘要\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"差异数量: {len(self.differences)}\n\n")
            
            if self.differences:
                f.write("主要差异:\n")
                for diff in self.differences[:10]:  # 只显示前10个
                    f.write(f"  - {diff['name']}: {diff['diff']}\n")
            else:
                f.write("✅ 未发现差异\n")
        
        print(f"\n结果已保存到: {self.output_dir}")
        
    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "="*80)
        print("开始Rules_new1详细对比测试")
        print("="*80)
        
        # 1. 环境初始化测试
        self.test_environment_initialization(seed=42)
        
        # 2. 算法初始化测试
        self.test_algorithm_initialization(seed=42)
        
        # 3. 单步执行测试
        self.test_single_step_execution(seed=42)
        
        # 4. 多步执行测试
        self.test_multiple_steps(seed=42, num_steps=10)
        
        # 保存结果
        self.save_results()
        
        print("\n" + "="*80)
        print("测试完成！")
        print(f"发现差异: {len(self.differences)}个")
        print(f"结果保存在: {self.output_dir}")
        print("="*80)


if __name__ == "__main__":
    tester = DetailedComparison()
    tester.run_all_tests()