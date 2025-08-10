#!/usr/bin/env python3
"""
测试NNPlanner神经网络规划器功能
"""

import sys
import numpy as np
import math
from pathlib import Path
import yaml
from omegaconf import DictConfig
import torch

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import envs  # noqa - 注册环境
import gymnasium as gym

# 导入NNPlanner
from rules_new.algorithms import NNPlanner


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


def test_nn_planner(model_name, model_path, seed=42, max_steps=10):
    """测试单个神经网络规划器"""
    print(f"\n{'='*60}")
    print(f"测试 {model_name} 神经网络规划器")
    print(f"{'='*60}")
    
    # 检查模型文件
    full_model_path = project_root / model_path
    if not full_model_path.exists():
        print(f"❌ 模型文件不存在: {full_model_path}")
        return False
    else:
        print(f"✅ 模型文件存在: {full_model_path}")
    
    # 创建环境
    env, obs, info = create_test_environment(seed)
    
    # 创建算法配置
    algorithm_config = {
        'algorithm': {'name': model_name, 'type': 'neural_network'},
        'model_path': model_path,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
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
        # 创建NNPlanner实例
        print(f"创建NNPlanner实例...")
        planner = NNPlanner(algorithm_config, env_config)
        print(f"✅ NNPlanner初始化成功")
        print(f"  使用设备: {planner.device}")
        print(f"  模型加载: {planner.actor is not None}")
        
        # 准备初始状态 - 需要包含观测数据
        initial_state = {
            'agent_position': [float(env.agent.x), float(env.agent.y)],
            'agent_direction': float(env.agent.direction),
            'discovered_weeds': [],
            'weed_count': 100,
            'coverage_rate': 0.0,
            'farm_vertices': env.min_area_rect[0][:, 0, ::-1] if hasattr(env, 'min_area_rect') else np.array([[50, 50], [550, 50], [550, 550], [50, 550]]),
            'seed': seed,
            'turning_radius': env.v_range.max / (abs(env.w_range.max) * math.pi / 180),
            'observation': obs['observation'] if isinstance(obs, dict) else obs,  # 关键：添加观测数据
            'vector': obs['vector'] if isinstance(obs, dict) and 'vector' in obs else 0.0,
            'maps': {
                'weed': env.map_weed if hasattr(env, 'map_weed') else None,
                'obstacle': env.map_obstacle if hasattr(env, 'map_obstacle') else None,
                'frontier': env.map_frontier if hasattr(env, 'map_frontier') else None
            }
        }
        
        # 验证观测数据格式
        if 'observation' in initial_state:
            obs_shape = initial_state['observation'].shape if hasattr(initial_state['observation'], 'shape') else None
            print(f"  观测数据形状: {obs_shape}")
            if obs_shape == (25, 16, 16):
                print(f"  ✅ 观测数据格式正确")
            else:
                print(f"  ⚠️ 观测数据格式可能不正确，期望 (25, 16, 16)")
        
        # 重置算法
        planner.reset(initial_state)
        print(f"✅ 算法重置成功")
        
        # 执行测试步骤
        print(f"\n执行前{max_steps}步测试:")
        total_reward = 0
        action_history = []
        
        for step in range(max_steps):
            # 更新观测数据
            if isinstance(obs, dict):
                initial_state['observation'] = obs['observation']
                initial_state['vector'] = obs.get('vector', 0.0)
            else:
                initial_state['observation'] = obs
                initial_state['vector'] = 0.0
            
            # 获取算法决策
            decision = planner.plan_next_waypoint(initial_state)
            
            if decision is None:
                print(f"  步骤{step+1}: 算法终止")
                break
            elif isinstance(decision, tuple) and decision[0] == 'action':
                action = decision[1]
                print(f"  步骤{step+1}: 动作 = [{action[0]:.3f}, {action[1]:.3f}]")
                action_history.append(action)
                
                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                # 更新状态
                initial_state['agent_position'] = [env.agent.x, env.agent.y]
                initial_state['agent_direction'] = env.agent.direction
                initial_state['coverage_rate'] = info.get('coverage_rate', 0)
                
                if terminated or truncated:
                    print(f"  环境终止: terminated={terminated}, truncated={truncated}")
                    break
            else:
                print(f"  步骤{step+1}: 未知返回格式 {type(decision)}")
                break
        
        # 输出统计
        print(f"\n执行统计:")
        print(f"  总步数: {len(action_history)}")
        print(f"  总奖励: {total_reward:.4f}")
        print(f"  覆盖率: {initial_state['coverage_rate']:.2%}")
        
        # 分析动作分布
        if action_history:
            actions_array = np.array(action_history)
            print(f"\n动作统计:")
            print(f"  线速度范围: [{actions_array[:, 0].min():.3f}, {actions_array[:, 0].max():.3f}]")
            print(f"  角速度范围: [{actions_array[:, 1].min():.3f}, {actions_array[:, 1].max():.3f}]")
            print(f"  平均线速度: {actions_array[:, 0].mean():.3f}")
            print(f"  平均角速度: {actions_array[:, 1].mean():.3f}")
        
        env.close()
        print(f"\n✅ {model_name} 测试成功")
        return True
        
    except Exception as e:
        print(f"\n❌ {model_name} 测试失败")
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return False


def main():
    """主测试函数"""
    print("开始测试NNPlanner神经网络规划器")
    print("=" * 60)
    
    # 测试配置
    nn_models = [
        {
            'name': 'NN_baseline',
            'path': 'ckpt/sac_cont/2024-09-09_01-16-14_tanhnorm_loc/sac_baseline_continuous_t[01100]_r[2570.25=2509.63~2623.36] - 副本.pt'
        },
        {
            'name': 'NN_ours',
            'path': 'ckpt/sac_cont/finetune/t[02350]_r[2782.06=2666.52~2872.77].pt'
        }
    ]
    
    results = {}
    
    # 测试每个模型
    for model_config in nn_models:
        success = test_nn_planner(
            model_config['name'], 
            model_config['path'],
            seed=42,
            max_steps=20
        )
        results[model_config['name']] = success
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for model_name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {model_name}: {status}")
    
    # 最终判断
    all_success = all(results.values())
    print("\n" + "=" * 60)
    if all_success:
        print("🎉 所有NNPlanner测试通过！")
        return 0
    else:
        print("⚠️ 部分NNPlanner测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())