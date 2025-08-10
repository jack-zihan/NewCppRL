#!/usr/bin/env python3
"""
简化的对比测试 - 直接测试算法输出的一致性
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

# 导入rules_new1算法
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


def test_algorithm_output(AlgorithmClass, algorithm_name, seed=42, max_steps=20):
    """测试单个算法的输出"""
    print(f"\n{'='*60}")
    print(f"测试 {algorithm_name} 算法 (Seed: {seed})")
    print(f"{'='*60}")
    
    # 创建环境
    env, obs, info = create_test_environment(seed)
    
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
        initial_state = {
            'agent_position': [float(env.agent.x), float(env.agent.y)],
            'agent_direction': float(env.agent.direction),
            'discovered_weeds': [],
            'weed_count': 100,
            'coverage_rate': 0.0,
            'farm_vertices': env.min_area_rect[0][:, 0, ::-1] if hasattr(env, 'min_area_rect') else np.array([[50, 50], [550, 50], [550, 550], [50, 550]]),
            'seed': seed,
            'turning_radius': env.v_range.max / (abs(env.w_range.max) * math.pi / 180),
            'maps': {
                'weed': env.map_weed if hasattr(env, 'map_weed') else None,
                'obstacle': env.map_obstacle if hasattr(env, 'map_obstacle') else None,
                'frontier': env.map_frontier if hasattr(env, 'map_frontier') else None
            }
        }
        
        # 重置算法
        planner.reset(initial_state)
        
        print(f"✅ 算法初始化成功")
        print(f"初始位置: [{initial_state['agent_position'][0]:.2f}, {initial_state['agent_position'][1]:.2f}]")
        print(f"初始方向: {initial_state['agent_direction']:.2f}°")
        print(f"转弯半径: {initial_state['turning_radius']:.2f}")
        
        # 执行测试步骤
        total_reward = 0
        agent_trajectory = []
        
        print(f"\n执行前{max_steps}步:")
        for step in range(max_steps):
            # 获取算法决策
            decision = planner.plan_next_waypoint(initial_state)
            
            if decision is None:
                print(f"  步骤{step+1}: 算法终止")
                break
            elif isinstance(decision, tuple) and decision[0] == 'path':
                path_points = decision[1]
                print(f"  步骤{step+1}: 返回路径列表，包含{len(path_points)}个点")
                
                # 执行路径中的每个点
                for i, waypoint in enumerate(path_points[:5]):  # 只执行前5个点作为示例
                    # 转换为动作
                    agent_pos = [env.agent.x, env.agent.y]
                    agent_dir = env.agent.direction
                    agent_rad = np.pi / 2 - math.radians(agent_dir)
                    
                    target_rad = math.atan2(
                        waypoint[1] - agent_pos[1],
                        waypoint[0] - agent_pos[0]
                    )
                    
                    length = math.sqrt(
                        (waypoint[0] - agent_pos[0])**2 + 
                        (waypoint[1] - agent_pos[1])**2
                    )
                    
                    delta_angle = -(target_rad - agent_rad) % (2 * math.pi)
                    if delta_angle > math.pi:
                        delta_angle = delta_angle - 2 * math.pi
                    delta_angle = math.degrees(delta_angle)
                    
                    action = (length, delta_angle)
                    
                    # 执行动作
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    agent_trajectory.append([env.agent.x, env.agent.y])
                    
                    if terminated or truncated:
                        break
                
                # 更新状态
                initial_state['agent_position'] = [env.agent.x, env.agent.y]
                initial_state['agent_direction'] = env.agent.direction
                initial_state['coverage_rate'] = info.get('coverage_rate', 0)
                
                # 模拟发现杂草（对于需要的算法）
                if algorithm_name in ['SNAKE', 'R_SNAKE', 'REACT'] and step == 5:
                    # 在视野内寻找杂草
                    weeds_in_sight = []
                    for i in range(env.map_weed.shape[0]):
                        for j in range(env.map_weed.shape[1]):
                            if env.map_weed[i, j] > 0:
                                dist = np.sqrt((i - env.agent.y)**2 + (j - env.agent.x)**2)
                                if dist < 50:  # 50单位视野范围
                                    weeds_in_sight.append((j, i))  # x, y format
                    
                    if weeds_in_sight:
                        initial_state['discovered_weeds'] = weeds_in_sight[:3]  # 最多3个
                        print(f"    发现{len(initial_state['discovered_weeds'])}个杂草")
            else:
                print(f"  步骤{step+1}: 未知返回格式 {type(decision)}")
                break
        
        # 输出统计
        print(f"\n执行统计:")
        print(f"  总奖励: {total_reward:.4f}")
        print(f"  覆盖率: {initial_state['coverage_rate']:.2%}")
        print(f"  轨迹长度: {len(agent_trajectory)}步")
        if agent_trajectory:
            print(f"  最终位置: [{agent_trajectory[-1][0]:.2f}, {agent_trajectory[-1][1]:.2f}]")
        
        env.close()
        
        return {
            'success': True,
            'total_reward': total_reward,
            'coverage_rate': initial_state['coverage_rate'],
            'trajectory': agent_trajectory
        }
        
    except Exception as e:
        print(f"❌ {algorithm_name} 算法出错: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return {
            'success': False,
            'error': str(e)
        }


def main():
    """主测试函数"""
    print("开始测试rules_new1算法实现")
    print("=" * 60)
    
    # 测试所有算法
    algorithms = [
        (BcpPlanner, 'BCP'),
        (JumpPlanner, 'JUMP'),
        (SnakePlanner, 'SNAKE'),
        (RSnakePlanner, 'R_SNAKE'),
        (ReactPlanner, 'REACT'),
    ]
    
    results = {}
    seeds = [42, 100, 200]
    
    for AlgorithmClass, name in algorithms:
        algorithm_results = []
        for seed in seeds:
            result = test_algorithm_output(AlgorithmClass, name, seed, max_steps=10)
            algorithm_results.append(result)
        results[name] = algorithm_results
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for name, algorithm_results in results.items():
        success_count = sum(1 for r in algorithm_results if r['success'])
        total_count = len(algorithm_results)
        
        print(f"\n{name}:")
        print(f"  成功率: {success_count}/{total_count}")
        
        if success_count > 0:
            avg_reward = np.mean([r['total_reward'] for r in algorithm_results if r['success']])
            avg_coverage = np.mean([r['coverage_rate'] for r in algorithm_results if r['success']])
            print(f"  平均奖励: {avg_reward:.4f}")
            print(f"  平均覆盖率: {avg_coverage:.2%}")
        
        if success_count == total_count:
            print(f"  ✅ 所有测试通过")
        elif success_count > 0:
            print(f"  ⚠️ 部分测试通过")
        else:
            print(f"  ❌ 所有测试失败")
    
    # 最终判断
    print("\n" + "=" * 60)
    all_success = all(
        all(r['success'] for r in algorithm_results)
        for algorithm_results in results.values()
    )
    
    if all_success:
        print("🎉 所有算法测试成功！")
        print("rules_new1的算法实现工作正常。")
        return 0
    else:
        print("⚠️ 部分算法测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())