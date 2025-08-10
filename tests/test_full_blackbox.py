#!/usr/bin/env python3
"""
完整黑盒对比测试 - rules vs rules_new
真正运行两个系统并对比waypoints
"""
import sys
import math
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
import json
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'tests'))
sys.path.insert(0, str(project_root / 'envs'))


class BlackboxComparison:
    """黑盒对比测试器"""
    
    def __init__(self):
        self.results = []
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = project_root / 'logs' / 'blackbox_comparison' / self.timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run_rules_new(self, task_type: str, seed: int, max_waypoints: int = 20) -> List[Tuple[float, float]]:
        """运行rules_new并提取waypoints"""
        print(f"  运行rules_new ({task_type}, seed={seed})...")
        
        try:
            from rules_new_simple_runner import RulesNewSimpleRunner
            
            runner = RulesNewSimpleRunner(task_type=task_type, seed=seed)
            waypoints = runner.generate_waypoints(max_waypoints=max_waypoints)
            
            print(f"    提取到 {len(waypoints)} 个waypoints")
            return waypoints
            
        except Exception as e:
            print(f"    ❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def run_rules_new1(self, task_type: str, seed: int, max_waypoints: int = 20) -> List[Tuple[float, float]]:
        """运行rules_new1并提取waypoints"""
        print(f"  运行rules_new1 ({task_type}, seed={seed})...")
        
        try:
            # 使用与rules_new相同的环境创建方法
            from env_make import get_test_env
            env, obs = get_test_env(seed=seed)
            
            # 获取环境参数
            farm_vertices = env.min_area_rect[0][:, 0, ::-1]  # [y, x] -> [x, y]
            agent_position = [float(env.agent.x), float(env.agent.y)]
            turning_radius = env.v_range.max / (abs(env.w_range.max) * (math.pi / 180))
            
            # 创建算法
            from rules_new.experiment.config_manager import ConfigManager
            
            config_manager = ConfigManager()
            algo_config = config_manager.load_algorithm_config(task_type)
            base_config = config_manager.load_base_config()
            
            # 创建planner
            if task_type == 'JUMP':
                from rules_new.algorithms import JumpPlanner
                planner = JumpPlanner(algo_config, base_config)
            elif task_type == 'SNAKE':
                from rules_new.algorithms import SnakePlanner
                planner = SnakePlanner(algo_config, base_config)
            elif task_type == 'R_SNAKE':
                from rules_new.algorithms import RSnakePlanner
                planner = RSnakePlanner(algo_config, base_config)
            elif task_type == 'BCP':
                from rules_new.algorithms import BcpPlanner
                planner = BcpPlanner(algo_config, base_config)
            elif task_type == 'REACT':
                from rules_new.algorithms import ReactPlanner
                planner = ReactPlanner(algo_config, base_config)
            else:
                raise ValueError(f"Unknown algorithm: {task_type}")
            
            # 初始化
            initial_state = {
                'agent_position': agent_position,
                'agent_direction': float(env.agent.direction),
                'discovered_weeds': [],
                'weed_count': 0,
                'farm_vertices': farm_vertices,
                'turning_radius': turning_radius,
                'coverage_rate': 0.0,
                'iteration': 0,
                'seed': seed
            }
            
            planner.reset(initial_state)
            
            # 生成waypoints
            waypoints = []
            for i in range(max_waypoints):
                current_state = {
                    'agent_position': agent_position if i == 0 else list(waypoints[-1]),
                    'agent_direction': float(env.agent.direction),
                    'discovered_weeds': [],
                    'weed_count': 0,
                    'coverage_rate': i * 0.01,
                    'iteration': i
                }
                
                waypoint = planner.plan_next_waypoint(current_state)
                if waypoint is None:
                    break
                    
                waypoints.append(waypoint)
                
            env.close()
            
            print(f"    生成了 {len(waypoints)} 个waypoints")
            return waypoints
            
        except Exception as e:
            print(f"    ❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def compare_waypoints(self, waypoints_old: List, waypoints_new: List, tolerance: float = 0.1) -> Dict[str, Any]:
        """比较两组waypoints"""
        result = {
            'total_old': len(waypoints_old),
            'total_new': len(waypoints_new),
            'matches': 0,
            'differences': [],
            'max_difference': 0,
            'avg_difference': 0
        }
        
        # 比较每个waypoint
        min_len = min(len(waypoints_old), len(waypoints_new))
        total_diff = 0
        
        for i in range(min_len):
            old = np.array(waypoints_old[i])
            new = np.array(waypoints_new[i])
            
            # 计算距离
            diff = np.linalg.norm(old - new)
            total_diff += diff
            
            if diff <= tolerance:
                result['matches'] += 1
            else:
                result['differences'].append({
                    'index': i,
                    'old': old.tolist(),
                    'new': new.tolist(),
                    'distance': float(diff)
                })
                
            result['max_difference'] = max(result['max_difference'], diff)
        
        if min_len > 0:
            result['avg_difference'] = total_diff / min_len
            
        result['match_rate'] = result['matches'] / min_len if min_len > 0 else 0
        
        return result
    
    def run_single_test(self, algorithm: str, seed: int) -> Dict[str, Any]:
        """运行单个测试案例"""
        print(f"\n测试: {algorithm} (seed={seed})")
        print("-" * 50)
        
        # 运行两个系统
        waypoints_old = self.run_rules_new(algorithm, seed)
        waypoints_new = self.run_rules_new1(algorithm, seed)
        
        # 比较结果
        comparison = self.compare_waypoints(waypoints_old, waypoints_new)
        
        # 记录结果
        result = {
            'algorithm': algorithm,
            'seed': seed,
            'waypoints_old': [(float(w[0]), float(w[1])) for w in waypoints_old[:10]],  # 只保存前10个
            'waypoints_new': [(float(w[0]), float(w[1])) for w in waypoints_new[:10]],
            'comparison': comparison
        }
        
        # 打印摘要
        print(f"  Waypoints数量: rules={len(waypoints_old)}, rules_new={len(waypoints_new)}")
        print(f"  匹配率: {comparison['match_rate']*100:.1f}%")
        print(f"  平均差异: {comparison['avg_difference']:.4f}")
        print(f"  最大差异: {comparison['max_difference']:.4f}")
        
        if waypoints_old and waypoints_new:
            print(f"  第一个waypoint:")
            print(f"    rules:  {waypoints_old[0]}")
            print(f"    rules_new: {waypoints_new[0]}")
        
        return result
    
    def run_all_tests(self):
        """运行所有测试"""
        print("="*80)
        print("🔬 Rules_new vs Rules_new1 完整黑盒对比测试")
        print("="*80)
        
        # 测试配置
        algorithms = ['JUMP', 'SNAKE', 'BCP']  # 先测试这三个稳定的算法
        seeds = [42, 100, 200]  # 测试3个种子
        
        all_results = []
        
        for algo in algorithms:
            algo_results = []
            for seed in seeds:
                result = self.run_single_test(algo, seed)
                algo_results.append(result)
                all_results.append(result)
            
            # 算法级别统计
            self.print_algorithm_summary(algo, algo_results)
        
        # 保存结果
        self.save_results(all_results)
        
        # 打印总体统计
        self.print_overall_summary(all_results)
        
    def print_algorithm_summary(self, algorithm: str, results: List[Dict]):
        """打印算法级别的统计"""
        print(f"\n{'='*60}")
        print(f"📊 {algorithm} 算法统计")
        print(f"{'='*60}")
        
        avg_match_rate = np.mean([r['comparison']['match_rate'] for r in results])
        avg_difference = np.mean([r['comparison']['avg_difference'] for r in results])
        
        print(f"  平均匹配率: {avg_match_rate*100:.1f}%")
        print(f"  平均差异: {avg_difference:.4f}")
        
    def print_overall_summary(self, results: List[Dict]):
        """打印总体统计"""
        print("\n" + "="*80)
        print("📈 总体测试结果")
        print("="*80)
        
        total_tests = len(results)
        perfect_matches = sum(1 for r in results if r['comparison']['match_rate'] == 1.0)
        high_matches = sum(1 for r in results if r['comparison']['match_rate'] >= 0.9)
        
        print(f"  总测试数: {total_tests}")
        print(f"  完美匹配: {perfect_matches} ({perfect_matches/total_tests*100:.1f}%)")
        print(f"  高度匹配(>=90%): {high_matches} ({high_matches/total_tests*100:.1f}%)")
        
        # 找出差异最大的案例
        max_diff_case = max(results, key=lambda r: r['comparison']['max_difference'])
        print(f"\n  最大差异案例:")
        print(f"    算法: {max_diff_case['algorithm']}")
        print(f"    种子: {max_diff_case['seed']}")
        print(f"    最大差异: {max_diff_case['comparison']['max_difference']:.4f}")
        
    def save_results(self, results: List[Dict]):
        """保存测试结果"""
        output_file = self.output_dir / 'blackbox_comparison_results.json'
        
        # 转换numpy类型为Python类型
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        clean_results = convert_numpy(results)
        
        with open(output_file, 'w') as f:
            json.dump(clean_results, f, indent=2)
            
        print(f"\n💾 结果已保存到: {output_file}")


if __name__ == "__main__":
    comparison = BlackboxComparison()
    comparison.run_all_tests()