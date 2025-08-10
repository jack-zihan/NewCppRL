#!/usr/bin/env python3
"""
简单的一致性测试 - 直接测试算法输出
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


def test_continuous_action():
    """测试连续动作是否正常工作"""
    print("=" * 60)
    print("测试连续动作模式")
    print("=" * 60)
    
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
    obs, info = env.reset(seed=42)
    
    print(f"\n初始状态:")
    print(f"  Agent位置: ({env.agent.x:.2f}, {env.agent.y:.2f})")
    print(f"  Agent方向: {env.agent.direction:.2f}°")
    
    # 测试连续动作 [length, delta_angle]
    print(f"\n测试连续动作执行:")
    test_actions = [
        (10.0, 0.0),    # 直线前进10单位
        (5.0, 45.0),    # 前进5单位，右转45度
        (8.0, -30.0),   # 前进8单位，左转30度
    ]
    
    for i, (length, delta_angle) in enumerate(test_actions):
        print(f"\n步骤{i+1}: [length={length:.1f}, delta_angle={delta_angle:.1f}]")
        
        # 记录当前状态
        prev_x, prev_y = env.agent.x, env.agent.y
        prev_dir = env.agent.direction
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step((length, delta_angle))
        
        # 计算实际移动
        actual_distance = math.sqrt((env.agent.x - prev_x)**2 + (env.agent.y - prev_y)**2)
        actual_rotation = env.agent.direction - prev_dir
        
        print(f"  预期: 移动{length:.1f}, 转向{delta_angle:.1f}°")
        print(f"  实际: 移动{actual_distance:.2f}, 转向{actual_rotation:.2f}°")
        print(f"  新位置: ({env.agent.x:.2f}, {env.agent.y:.2f})")
        print(f"  新方向: {env.agent.direction:.2f}°")
        
        if terminated or truncated:
            print("  环境终止")
            break
    
    env.close()
    print("\n✅ 连续动作测试完成")
    return True


def test_jump_planner_output():
    """测试JumpPlanner的输出格式"""
    print("\n" + "=" * 60)
    print("测试JumpPlanner输出")
    print("=" * 60)
    
    from rules_new.algorithms.jump_planner import JumpPlanner
    
    # 创建环境
    cfg = DictConfig(yaml.load(
        open(f'{project_root}/configs/env_config.yaml'), 
        Loader=yaml.FullLoader
    ))
    
    env = gym.make(
        render_mode=None,
        **cfg.env.params,
    )
    
    obs, info = env.reset(seed=42)
    
    # 创建算法配置
    algorithm_config = {
        'algorithm': {'name': 'JUMP'},
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
    
    # 创建JumpPlanner
    planner = JumpPlanner(algorithm_config, env_config)
    
    # 准备初始状态
    initial_state = {
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
    
    # 重置算法
    planner.reset(initial_state)
    
    print(f"\n算法初始化完成")
    print(f"  农场顶点数: {len(planner.farm_vertices)}")
    print(f"  转弯半径: {planner.turning_radius:.2f}")
    
    # 获取前几个决策
    print(f"\n获取前5个决策:")
    for i in range(5):
        decision = planner.plan_next_waypoint(initial_state)
        
        if decision is None:
            print(f"  步骤{i+1}: None (终止)")
            break
        elif isinstance(decision, tuple) and len(decision) == 2:
            if decision[0] == 'path':
                print(f"  步骤{i+1}: 路径列表，包含 {len(decision[1])} 个点")
                if decision[1]:
                    print(f"    第一个点: {decision[1][0]}")
            else:
                print(f"  步骤{i+1}: Waypoint {decision}")
        else:
            print(f"  步骤{i+1}: {type(decision)} - {decision}")
        
        # 更新coverage_rate模拟进展
        initial_state['coverage_rate'] += 0.01
    
    env.close()
    print("\n✅ JumpPlanner测试完成")
    return True


def main():
    """主测试函数"""
    print("开始简单一致性测试")
    print("=" * 60)
    
    # 测试连续动作
    if not test_continuous_action():
        print("\n❌ 连续动作测试失败")
        return 1
    
    # 测试JumpPlanner输出
    if not test_jump_planner_output():
        print("\n❌ JumpPlanner测试失败")
        return 1
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())