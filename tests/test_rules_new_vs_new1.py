#!/usr/bin/env python3
"""
全面对比测试rules_new和rules_new1的执行结果一致性
"""

import sys
import numpy as np
import math
import time
from pathlib import Path
import yaml
from omegaconf import DictConfig

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import envs  # noqa - 注册环境
import gymnasium as gym

# 导入rules_new和rules_new1
from rules.experiment.experiment_runner import ExperimentRunner as RunnerNew
from rules_new.experiment.experiment_runner import ExperimentRunner as RunnerNew1


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


def run_algorithm_comparison(algorithm_name, seeds=[42, 100, 200], max_steps=100):
    """对比单个算法在两个系统中的执行结果"""
    print(f"\n{'='*80}")
    print(f"测试算法: {algorithm_name}")
    print(f"{'='*80}")
    
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
    
    results_comparison = []
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        
        # 运行rules_new
        env1, obs1, info1 = create_test_environment(seed)
        runner_new = RunnerNew(algorithm_config, env_config)
        
        metrics_new = {
            'total_reward': 0,
            'coverage_rate': 0,
            'agent_positions': [],
            'actions': [],
            'steps': 0
        }
        
        try:
            # 运行rules_new
            for step in range(max_steps):
                decision = runner_new.plan_next_action(obs1, info1)
                if decision is None:
                    break
                    
                obs1, reward1, terminated1, truncated1, info1 = env1.step(decision)
                metrics_new['total_reward'] += reward1
                metrics_new['coverage_rate'] = info1.get('coverage_rate', 0)
                metrics_new['agent_positions'].append([env1.agent.x, env1.agent.y])
                metrics_new['actions'].append(decision)
                metrics_new['steps'] += 1
                
                if terminated1 or truncated1:
                    break
        except Exception as e:
            print(f"  rules_new出错: {e}")
            metrics_new['error'] = str(e)
        finally:
            env1.close()
        
        # 运行rules_new1
        env2, obs2, info2 = create_test_environment(seed)
        runner_new1 = RunnerNew1(algorithm_config, env_config)
        
        metrics_new1 = {
            'total_reward': 0,
            'coverage_rate': 0,
            'agent_positions': [],
            'actions': [],
            'steps': 0
        }
        
        try:
            # 运行rules_new1
            for step in range(max_steps):
                decision = runner_new1.plan_next_action(obs2, info2)
                if decision is None:
                    break
                    
                obs2, reward2, terminated2, truncated2, info2 = env2.step(decision)
                metrics_new1['total_reward'] += reward2
                metrics_new1['coverage_rate'] = info2.get('coverage_rate', 0)
                metrics_new1['agent_positions'].append([env2.agent.x, env2.agent.y])
                metrics_new1['actions'].append(decision)
                metrics_new1['steps'] += 1
                
                if terminated2 or truncated2:
                    break
        except Exception as e:
            print(f"  rules_new1出错: {e}")
            metrics_new1['error'] = str(e)
        finally:
            env2.close()
        
        # 对比结果
        print(f"\n  rules_new:")
        print(f"    步数: {metrics_new['steps']}")
        print(f"    总奖励: {metrics_new['total_reward']:.4f}")
        print(f"    覆盖率: {metrics_new['coverage_rate']:.2%}")
        if metrics_new['agent_positions']:
            print(f"    最终位置: {metrics_new['agent_positions'][-1]}")
        
        print(f"\n  rules_new1:")
        print(f"    步数: {metrics_new1['steps']}")
        print(f"    总奖励: {metrics_new1['total_reward']:.4f}")
        print(f"    覆盖率: {metrics_new1['coverage_rate']:.2%}")
        if metrics_new1['agent_positions']:
            print(f"    最终位置: {metrics_new1['agent_positions'][-1]}")
        
        # 计算差异
        reward_diff = abs(metrics_new['total_reward'] - metrics_new1['total_reward'])
        coverage_diff = abs(metrics_new['coverage_rate'] - metrics_new1['coverage_rate'])
        steps_diff = abs(metrics_new['steps'] - metrics_new1['steps'])
        
        # 计算位置轨迹相似度（前10步）
        position_similarity = 0
        if metrics_new['agent_positions'] and metrics_new1['agent_positions']:
            min_steps = min(10, len(metrics_new['agent_positions']), len(metrics_new1['agent_positions']))
            for i in range(min_steps):
                pos1 = np.array(metrics_new['agent_positions'][i])
                pos2 = np.array(metrics_new1['agent_positions'][i])
                dist = np.linalg.norm(pos1 - pos2)
                position_similarity += dist
            position_similarity = position_similarity / min_steps if min_steps > 0 else 0
        
        print(f"\n  差异分析:")
        print(f"    奖励差异: {reward_diff:.4f}")
        print(f"    覆盖率差异: {coverage_diff:.2%}")
        print(f"    步数差异: {steps_diff}")
        print(f"    位置轨迹平均偏差(前10步): {position_similarity:.2f}")
        
        # 判断是否一致
        is_consistent = (
            reward_diff < 1.0 and  # 奖励差异小于1.0
            coverage_diff < 0.05 and  # 覆盖率差异小于5%
            steps_diff < 10 and  # 步数差异小于10
            position_similarity < 50  # 平均位置偏差小于50
        )
        
        results_comparison.append({
            'seed': seed,
            'consistent': is_consistent,
            'reward_diff': reward_diff,
            'coverage_diff': coverage_diff,
            'steps_diff': steps_diff,
            'position_similarity': position_similarity
        })
        
        if is_consistent:
            print(f"    ✅ 结果一致")
        else:
            print(f"    ❌ 结果不一致")
    
    return results_comparison


def main():
    """主测试函数"""
    print("开始全面对比测试rules_new和rules_new1")
    print("=" * 80)
    
    # 测试所有算法
    algorithms = ['BCP', 'JUMP', 'SNAKE', 'R_SNAKE', 'REACT']
    
    all_results = {}
    for algorithm in algorithms:
        results = run_algorithm_comparison(algorithm, seeds=[42, 100, 200], max_steps=50)
        all_results[algorithm] = results
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    
    for algorithm, results in all_results.items():
        consistent_count = sum(1 for r in results if r['consistent'])
        total_count = len(results)
        avg_reward_diff = np.mean([r['reward_diff'] for r in results])
        avg_coverage_diff = np.mean([r['coverage_diff'] for r in results])
        avg_position_sim = np.mean([r['position_similarity'] for r in results])
        
        print(f"\n{algorithm}:")
        print(f"  一致性: {consistent_count}/{total_count} seeds通过")
        print(f"  平均奖励差异: {avg_reward_diff:.4f}")
        print(f"  平均覆盖率差异: {avg_coverage_diff:.2%}")
        print(f"  平均位置偏差: {avg_position_sim:.2f}")
        
        if consistent_count == total_count:
            print(f"  ✅ 完全一致")
        elif consistent_count > 0:
            print(f"  ⚠️ 部分一致")
        else:
            print(f"  ❌ 不一致")
    
    # 最终判断
    print("\n" + "=" * 80)
    all_consistent = all(
        sum(1 for r in results if r['consistent']) == len(results)
        for results in all_results.values()
    )
    
    if all_consistent:
        print("🎉 所有算法在所有seeds下都保持一致！")
        print("rules_new1成功保持了与rules_new的功能一致性，同时改进了架构设计。")
        return 0
    else:
        print("⚠️ 部分算法存在差异，需要进一步调试")
        return 1


if __name__ == "__main__":
    sys.exit(main())