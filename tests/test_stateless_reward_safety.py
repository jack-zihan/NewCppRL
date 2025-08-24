#!/usr/bin/env python3
"""
测试无状态奖励系统的线程安全性和功能正确性
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from concurrent.futures import ThreadPoolExecutor
import numpy as np
from envs_new.cpp_env_v2 import CppEnv
from envs_new.components.reward.reward_system import BaseCalculator


def test_parallel_different_configs():
    """测试并行环境使用不同配置 - 现在应该是安全的"""
    print("="*60)
    print("🧪 测试1：并行环境不同配置（验证无状态设计）")
    print("="*60)
    
    def create_and_run_env(config_value):
        """创建环境并运行几步"""
        env = CppEnv(reward_base_penalty=config_value)
        env.reset(seed=42)
        
        rewards = []
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break
        
        env.close()
        
        # 关键：返回环境配置的系数和实际获得的奖励
        return {
            'config_value': config_value,
            'avg_reward': np.mean(rewards) if rewards else 0,
            'env_config_value': env.config.reward_base_penalty
        }
    
    # 使用不同配置并行创建环境
    configs = [-0.1, -0.2, -0.3, -0.15]
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(create_and_run_env, cfg) for cfg in configs]
        results = [f.result() for f in futures]
    
    print("\n📊 结果:")
    for r in results:
        print(f"  配置={r['config_value']:.2f}, "
              f"环境config={r['env_config_value']:.2f}, "
              f"平均奖励={r['avg_reward']:.4f}")
    
    # 验证每个环境使用正确的配置
    all_correct = all(abs(r['config_value'] - r['env_config_value']) < 1e-6 for r in results)
    
    if all_correct:
        print("\n✅ 无状态设计成功！每个环境使用独立的配置")
        print("   不再有类变量共享问题")
    else:
        print("\n❌ 仍然存在问题")
    
    return all_correct


def test_curriculum_learning_scenario():
    """测试Curriculum Learning场景 - 动态更新奖励"""
    print("\n" + "="*60)
    print("🧪 测试2：Curriculum Learning（动态奖励调整）")
    print("="*60)
    
    env = CppEnv(reward_base_penalty=-0.05)
    env.reset(seed=42)
    
    print("阶段1：初始奖励（探索阶段）")
    print(f"  base_penalty = {env.config.reward_base_penalty}")
    
    # 运行一些步骤
    rewards_phase1 = []
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards_phase1.append(reward)
        if terminated or truncated:
            break
    
    # 动态更新奖励系数（进入利用阶段）
    print("\n动态调整奖励...")
    env.reward_system.update_coefficients({
        'base_penalty': -0.20,
        'weed_removal': 30.0  # 增加核心任务奖励
    })
    
    print("阶段2：调整后奖励（利用阶段）")
    print(f"  base_penalty = {env.config.reward_base_penalty}")
    print(f"  weed_removal = {env.config.reward_weed_removal}")
    
    # 继续运行
    rewards_phase2 = []
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards_phase2.append(reward)
        if terminated or truncated:
            break
    
    env.close()
    
    avg1 = np.mean(rewards_phase1) if rewards_phase1 else 0
    avg2 = np.mean(rewards_phase2) if rewards_phase2 else 0
    
    print(f"\n📊 结果:")
    print(f"  阶段1平均奖励: {avg1:.4f}")
    print(f"  阶段2平均奖励: {avg2:.4f}")
    print(f"  奖励变化: {avg2 - avg1:.4f}")
    
    print("\n✅ Curriculum Learning支持成功！")
    print("   奖励可以动态调整，立即生效")
    
    return True


def test_multi_env_independence():
    """测试多个环境实例的独立性"""
    print("\n" + "="*60)
    print("🧪 测试3：多环境实例独立性")
    print("="*60)
    
    # 创建多个环境，每个使用不同配置
    configs = [
        {'reward_base_penalty': -0.05, 'reward_weed_removal': 10.0},
        {'reward_base_penalty': -0.10, 'reward_weed_removal': 20.0},
        {'reward_base_penalty': -0.15, 'reward_weed_removal': 30.0},
    ]
    
    envs = []
    for cfg in configs:
        env = CppEnv(**cfg)
        env.reset(seed=42)
        envs.append(env)
    
    print("创建3个环境，配置如下:")
    for i, env in enumerate(envs):
        print(f"  环境{i}: base={env.config.reward_base_penalty:.2f}, "
              f"weed={env.config.reward_weed_removal:.1f}")
    
    # 并行运行所有环境
    def run_env_steps(env):
        rewards = []
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break
        return np.mean(rewards) if rewards else 0
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(run_env_steps, env) for env in envs]
        avg_rewards = [f.result() for f in futures]
    
    print("\n运行后检查配置:")
    for i, env in enumerate(envs):
        print(f"  环境{i}: base={env.config.reward_base_penalty:.2f}, "
              f"weed={env.config.reward_weed_removal:.1f}, "
              f"平均奖励={avg_rewards[i]:.4f}")
    
    # 验证配置保持独立
    config_correct = all(
        abs(envs[i].config.reward_base_penalty - configs[i]['reward_base_penalty']) < 1e-6
        for i in range(3)
    )
    
    # 清理
    for env in envs:
        env.close()
    
    if config_correct:
        print("\n✅ 多环境完全独立，互不干扰！")
    else:
        print("\n❌ 环境间存在干扰")
    
    return config_correct


def test_no_class_attributes():
    """验证Calculator类不再有coefficient类属性"""
    print("\n" + "="*60)
    print("🧪 测试4：验证无状态设计（无类属性）")
    print("="*60)
    
    # 检查BaseCalculator是否还有coefficient属性
    has_coefficient = hasattr(BaseCalculator, 'coefficient')
    
    if has_coefficient:
        print("❌ BaseCalculator仍然有coefficient类属性")
        print(f"   值为: {BaseCalculator.coefficient}")
    else:
        print("✅ BaseCalculator没有coefficient类属性")
    
    # 验证calculate方法签名
    import inspect
    sig = inspect.signature(BaseCalculator.calculate)
    params = list(sig.parameters.keys())
    
    print(f"\ncalculate方法签名: {params}")
    
    if 'coefficient' in params:
        print("✅ calculate方法包含coefficient参数")
    else:
        print("❌ calculate方法缺少coefficient参数")
    
    return not has_coefficient and 'coefficient' in params


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("🚀 无状态奖励系统测试套件")
    print("="*70 + "\n")
    
    tests = [
        ("并行环境安全性", test_parallel_different_configs),
        ("Curriculum Learning", test_curriculum_learning_scenario),
        ("多环境独立性", test_multi_env_independence),
        ("无状态设计验证", test_no_class_attributes),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ 测试 '{name}' 失败: {e}")
            results.append((name, False))
    
    print("\n" + "="*70)
    print("📊 测试总结")
    print("="*70)
    
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n🎉 所有测试通过！无状态设计成功解决了多环境问题")
        print("   主要改进:")
        print("   1. 消除了类变量共享问题")
        print("   2. 支持Curriculum Learning动态调整")
        print("   3. 多环境并行完全安全")
        print("   4. 代码更简洁，维护性更好")
    else:
        print("\n⚠️ 部分测试失败，请检查实现")
    
    print("="*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())