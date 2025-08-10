#!/usr/bin/env python3
"""
测试所有算法的输出格式一致性
"""

import sys
import numpy as np
import math
from pathlib import Path
import yaml
from omegaconf import DictConfig

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import envs  # noqa - 注册环境
import gymnasium as gym

# 导入所有算法
from rules_new.algorithms import (
    JumpPlanner, SnakePlanner, RSnakePlanner, 
    ReactPlanner, BcpPlanner
)


def create_test_environment(seed=42):
    """创建测试环境"""
    # 加载环境配置
    cfg = DictConfig(yaml.load(
        open(f'{project_root}/configs/env_config.yaml'), 
        Loader=yaml.FullLoader
    ))
    
    # 创建环境
    env = gym.make(
        render_mode=None,
        **cfg.env.params,
    )
    
    # 重置环境
    obs, info = env.reset(seed=seed)
    
    return env, obs, info


def create_initial_state(env):
    """创建算法初始状态"""
    return {
        'agent_position': [float(env.agent.x), float(env.agent.y)],
        'agent_direction': float(env.agent.direction),
        'discovered_weeds': [],
        'weed_count': 100,
        'coverage_rate': 0.0,
        'farm_vertices': env.min_area_rect[0][:, 0, ::-1] if hasattr(env, 'min_area_rect') else np.array([[50, 50], [550, 50], [550, 550], [50, 550]]),
        'seed': 42,
        'turning_radius': 7.01,
        'maps': {
            'weed': env.map_weed if hasattr(env, 'map_weed') else None,
            'obstacle': env.map_obstacle if hasattr(env, 'map_obstacle') else None,
            'frontier': env.map_frontier if hasattr(env, 'map_frontier') else None
        }
    }


def test_algorithm(AlgorithmClass, algorithm_name):
    """测试单个算法"""
    print(f"\n{'='*60}")
    print(f"测试 {algorithm_name} 算法")
    print(f"{'='*60}")
    
    # 创建环境
    env, obs, info = create_test_environment()
    
    # 创建算法配置
    algorithm_config = {
        'algorithm': {'name': algorithm_name},
        'parameters': {},
        'performance': {
            'max_iterations': 1000,
            'timeout_seconds': 60
        }
    }
    
    env_config = {
        'agent': {
            'car_width': 5,
            'sight_width': 24,
            'sight_length': 24
        },
        'environment': {
            'width': 600,
            'height': 600
        }
    }
    
    try:
        # 创建算法实例
        planner = AlgorithmClass(algorithm_config, env_config)
        
        # 准备初始状态
        initial_state = create_initial_state(env)
        
        # 重置算法
        planner.reset(initial_state)
        
        print(f"✅ 算法初始化成功")
        
        # 测试前5个决策
        print(f"\n测试前5个决策输出:")
        test_passed = True
        
        for i in range(5):
            decision = planner.plan_next_waypoint(initial_state)
            
            if decision is None:
                print(f"  步骤{i+1}: None (终止)")
                break
            elif isinstance(decision, tuple) and len(decision) == 2:
                if decision[0] == 'path':
                    print(f"  步骤{i+1}: ✅ 路径列表，包含 {len(decision[1])} 个点")
                    if decision[1]:
                        first_point = decision[1][0]
                        print(f"    第一个点: {first_point}")
                        # 验证点的格式
                        if not (isinstance(first_point, (list, tuple)) and len(first_point) == 2):
                            print(f"    ❌ 路径点格式错误: {type(first_point)}")
                            test_passed = False
                else:
                    print(f"  步骤{i+1}: ❌ 返回了单个waypoint {decision} (应该返回路径列表)")
                    test_passed = False
            else:
                print(f"  步骤{i+1}: ❌ 返回格式错误: {type(decision)} - {decision}")
                test_passed = False
            
            # 模拟进展
            initial_state['coverage_rate'] += 0.01
            
            # 对于REACT算法，模拟发现一些杂草
            if algorithm_name == 'REACT' and i == 2:
                initial_state['discovered_weeds'] = [
                    (100, 100), (150, 150), (200, 200)
                ]
                print(f"    (添加了3个杂草点用于测试)")
        
        if test_passed:
            print(f"\n✅ {algorithm_name} 算法测试通过")
        else:
            print(f"\n❌ {algorithm_name} 算法测试失败")
        
        env.close()
        return test_passed
        
    except Exception as e:
        print(f"❌ {algorithm_name} 算法出错: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return False


def main():
    """主测试函数"""
    print("开始测试所有算法的输出格式一致性")
    print("=" * 60)
    
    # 测试所有算法
    algorithms = [
        (BcpPlanner, 'BCP'),
        (SnakePlanner, 'SNAKE'),
        (RSnakePlanner, 'R_SNAKE'),
        (ReactPlanner, 'REACT'),
        (JumpPlanner, 'JUMP'),
    ]
    
    results = {}
    for AlgorithmClass, name in algorithms:
        results[name] = test_algorithm(AlgorithmClass, name)
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name:10s}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("🎉 所有算法测试通过！")
        return 0
    else:
        print("❌ 部分算法测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())