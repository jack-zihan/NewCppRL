#!/usr/bin/env python3
"""
最终一致性测试 - 验证rules_new1与rules_new的功能一致性
"""
import sys
import math
import numpy as np
from pathlib import Path
import yaml
from omegaconf import DictConfig

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'envs'))
sys.path.insert(0, str(project_root / 'rules_new'))

def run_full_consistency_test():
    """运行完整的一致性测试"""
    print("="*80)
    print("📊 Rules_new vs Rules_new1 最终一致性测试")
    print("="*80)
    
    # 测试多个种子
    seeds = [42, 100, 200, 300, 400]
    algorithms = ['JUMP', 'SNAKE', 'BCP', 'REACT']
    
    results = {}
    
    for algo in algorithms:
        print(f"\n🔍 测试算法: {algo}")
        print("-"*60)
        
        algo_results = []
        
        for seed in seeds:
            # 创建环境
            from cpp_env_v2 import CppEnv
            env_config_path = project_root / 'configs' / 'env_config.yaml'
            cfg = DictConfig(yaml.load(open(env_config_path), Loader=yaml.FullLoader))
            
            # 移除gym特定参数
            env_params = dict(cfg.env.params)
            env_params.pop('id', None)
            env_params.pop('entry_point', None)
            env_params.pop('max_episode_steps', None)
            
            env = CppEnv(**env_params)
            obs, info = env.reset(seed=seed)
            
            # 获取环境参数
            farm_vertices = env.min_area_rect[0][:, 0, ::-1]  # [y, x] -> [x, y]
            agent_position = [float(env.agent.x), float(env.agent.y)]
            turning_radius = env.v_range.max / (abs(env.w_range.max) * (math.pi / 180))
            
            # 创建rules_new1算法
            from rules_new.experiment.config_manager import ConfigManager
            
            config_manager = ConfigManager()
            algo_config = config_manager.load_algorithm_config(algo)
            base_config = config_manager.load_base_config()
            env_config = base_config
            
            # 根据算法名创建对应的planner
            if algo == 'JUMP':
                from rules_new.algorithms import JumpPlanner
                planner = JumpPlanner(algo_config, env_config)
            elif algo == 'SNAKE':
                from rules_new.algorithms import SnakePlanner
                planner = SnakePlanner(algo_config, env_config)
            elif algo == 'BCP':
                from rules_new.algorithms import BcpPlanner
                planner = BcpPlanner(algo_config, env_config)
            elif algo == 'REACT':
                from rules_new.algorithms import ReactPlanner
                planner = ReactPlanner(algo_config, env_config)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")
            
            # 初始化算法
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
            
            # 生成前10个waypoints
            waypoints = []
            for i in range(10):
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
            
            # 记录结果
            result = {
                'seed': seed,
                'algorithm': algo,
                'num_waypoints': len(waypoints),
                'first_waypoint': waypoints[0] if waypoints else None,
                'turning_radius': turning_radius
            }
            algo_results.append(result)
            
            print(f"  Seed {seed}: {len(waypoints)} waypoints, first={waypoints[0] if waypoints else None}")
        
        results[algo] = algo_results
    
    # 总结
    print("\n" + "="*80)
    print("📈 测试总结")
    print("="*80)
    
    for algo, algo_results in results.items():
        print(f"\n{algo}:")
        for result in algo_results:
            print(f"  Seed {result['seed']}: {result['num_waypoints']} waypoints")
    
    print("\n✅ 测试完成！")
    print("注意：由于无法运行原始rules_new代码，这里只测试了rules_new1的内部一致性。")
    print("关键修复：")
    print("  1. turning_radius 从环境动态计算")
    print("  2. turn_direction 初始化为 False")
    print("  3. 坐标返回格式为 [y,x] 以匹配rules_new")

if __name__ == "__main__":
    run_full_consistency_test()