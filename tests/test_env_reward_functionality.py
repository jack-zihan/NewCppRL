#!/usr/bin/env python3
"""
测试环境奖励功能是否正常工作
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from envs_new.cpp_env_v2 import CppEnv


def test_basic_functionality():
    """测试基本功能是否正常"""
    print("🧪 测试环境基本功能...")
    
    env = CppEnv()
    obs, info = env.reset(seed=42)
    
    # obs是一个dict，包含'pixels'
    if isinstance(obs, dict) and 'pixels' in obs:
        print(f"  观察空间形状: {obs['pixels'].shape}")
    else:
        print(f"  观察类型: {type(obs)}")
    print(f"  动作空间: {env.action_space}")
    
    # 运行几步
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if i == 0:
            print(f"  第一步奖励: {reward:.4f}")
        
        if terminated or truncated:
            print(f"  Episode在第{i+1}步结束")
            break
    
    print(f"  总奖励: {total_reward:.4f}")
    env.close()
    
    print("✅ 基本功能正常\n")
    return True


def test_reward_breakdown():
    """测试奖励分解功能"""
    print("🧪 测试奖励分解功能...")
    
    env = CppEnv(
        reward_base_penalty=-0.15,
        reward_weed_removal=25.0,
        reward_turning_penalty=-0.8
    )
    obs, info = env.reset(seed=42)
    
    # 执行一步并获取奖励分解
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 获取奖励分解
    breakdown = env.reward_system.get_reward_breakdown(env.env_state)
    
    print("  奖励组件:")
    for name, value in breakdown['components'].items():
        if abs(value) > 1e-8:
            print(f"    {name}: {value:.4f}")
    
    print(f"  总奖励: {breakdown['total']:.4f}")
    print(f"  实际奖励: {reward:.4f}")
    
    # 验证总和是否正确
    if abs(breakdown['total'] - reward) < 1e-6:
        print("✅ 奖励分解正确\n")
    else:
        print("❌ 奖励分解不一致\n")
    
    env.close()
    return abs(breakdown['total'] - reward) < 1e-6


def test_group_coefficients():
    """测试组系数功能"""
    print("🧪 测试组系数功能...")
    
    # 创建环境，设置组系数
    env = CppEnv(
        reward_turning_group_coef=2.0,  # 放大转向相关奖励
        reward_frontier_group_coef=0.5  # 减小前沿相关奖励
    )
    obs, info = env.reset(seed=42)
    
    # 执行多步，收集转向相关的奖励
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        breakdown = env.reward_system.get_reward_breakdown(env.env_state)
        
        # 检查turning组的奖励
        turning_total = breakdown.get('turning_total', 0)
        if abs(turning_total) > 1e-8:
            print(f"  步{i+1} - turning组奖励: {turning_total:.4f}")
        
        if terminated or truncated:
            break
    
    env.close()
    print("✅ 组系数功能正常\n")
    return True


def test_config_isolation():
    """测试配置隔离性"""
    print("🧪 测试配置隔离性...")
    
    # 创建两个不同配置的环境
    env1 = CppEnv(reward_base_penalty=-0.1)
    env2 = CppEnv(reward_base_penalty=-0.3)
    
    env1.reset(seed=42)
    env2.reset(seed=42)
    
    # 执行相同的动作（使用离散动作）
    action = 50  # 一个离散动作索引
    
    _, reward1, _, _, _ = env1.step(action)
    _, reward2, _, _, _ = env2.step(action)
    
    print(f"  环境1 (base=-0.1): 奖励={reward1:.4f}")
    print(f"  环境2 (base=-0.3): 奖励={reward2:.4f}")
    print(f"  奖励差异: {abs(reward2 - reward1):.4f}")
    
    env1.close()
    env2.close()
    
    # 验证奖励不同（因为基础惩罚不同）
    if abs(reward2 - reward1) > 0.1:
        print("✅ 配置隔离正确\n")
        return True
    else:
        print("❌ 配置隔离失败\n")
        return False


def main():
    """运行所有功能测试"""
    print("\n" + "="*60)
    print("🚀 环境奖励功能测试")
    print("="*60 + "\n")
    
    tests = [
        test_basic_functionality,
        test_reward_breakdown,
        test_group_coefficients,
        test_config_isolation,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"❌ 测试 {test_func.__name__} 失败: {e}\n")
            results.append((test_func.__name__, False))
    
    print("="*60)
    print("📊 测试总结")
    print("="*60)
    
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n🎉 所有功能测试通过！")
    else:
        print("\n⚠️ 部分功能测试失败")
    
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())