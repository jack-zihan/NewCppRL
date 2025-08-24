#!/usr/bin/env python3
"""
Episode级别的奖励一致性测试
通过运行完整的episode，使用相同的动作序列，对比新旧环境的奖励序列
"""

import sys
import numpy as np
sys.path.append('/home/lzh/NewCppRL')

from envs.cpp_env_v2 import CppEnv as OldCppEnvV2
from envs_new.cpp_env_v2 import CppEnv as NewCppEnvV2


def run_episode_comparison(seed=42, num_steps=100):
    """运行episode对比测试"""
    print(f"🎯 运行Episode奖励一致性测试")
    print(f"   Seed: {seed}, Steps: {num_steps}")
    print("="*80)
    
    # 创建环境
    old_env = OldCppEnvV2(render_mode=None)
    new_env = NewCppEnvV2(render_mode=None)
    
    # 使用相同的种子重置
    old_obs, old_info = old_env.reset(seed=seed)
    new_obs, new_info = new_env.reset(seed=seed)
    
    # 设置相同的动作种子
    old_env.action_space.seed(seed)
    new_env.action_space.seed(seed)
    
    # 收集奖励序列
    old_rewards = []
    new_rewards = []
    differences = []
    
    print("\n步骤检查:")
    print(f"{'Step':<6} {'旧版本':>10} {'新版本':>10} {'差异':>10} {'状态':<10}")
    print("-"*50)
    
    for step in range(num_steps):
        # 生成相同的动作
        action = old_env.action_space.sample()
        
        # 执行动作
        old_obs, old_reward, old_done, old_truncated, old_info = old_env.step(action)
        new_obs, new_reward, new_done, new_truncated, new_info = new_env.step(action)
        
        # 记录奖励
        old_rewards.append(old_reward)
        new_rewards.append(new_reward)
        
        # 计算差异
        diff = float(new_reward) - float(old_reward)
        differences.append(diff)
        
        # 打印前10步和有差异的步骤
        if step < 10 or abs(diff) > 1e-6:
            status = "✅" if abs(diff) < 1e-6 else "❌"
            print(f"{step+1:<6} {old_reward:>10.4f} {new_reward:>10.4f} {diff:>10.6f} {status}")
        
        # 检查终止状态一致性
        if old_done != new_done or old_truncated != new_truncated:
            print(f"\n⚠️ 终止状态不一致！")
            print(f"   旧版本: done={old_done}, truncated={old_truncated}")
            print(f"   新版本: done={new_done}, truncated={new_truncated}")
            break
        
        if old_done or old_truncated:
            print(f"\n✓ Episode在第{step+1}步正常结束")
            break
    
    # 统计分析
    print("\n" + "="*80)
    print("📊 统计分析:")
    print(f"   总步数: {len(old_rewards)}")
    print(f"   旧版本总奖励: {sum(old_rewards):.4f}")
    print(f"   新版本总奖励: {sum(new_rewards):.4f}")
    print(f"   总奖励差异: {sum(new_rewards) - sum(old_rewards):.6f}")
    print(f"   平均每步差异: {np.mean(np.abs(differences)):.6f}")
    print(f"   最大差异: {np.max(np.abs(differences)):.6f}")
    print(f"   差异标准差: {np.std(differences):.6f}")
    
    # 判断是否完全一致
    max_diff = np.max(np.abs(differences))
    if max_diff < 1e-6:
        print("\n✅ 测试通过！新旧版本奖励计算完全一致！")
        return True
    else:
        print(f"\n❌ 测试失败！存在奖励差异，最大差异: {max_diff:.6f}")
        # 找出差异最大的步骤
        max_diff_idx = np.argmax(np.abs(differences))
        print(f"   最大差异出现在第{max_diff_idx+1}步:")
        print(f"   旧版本: {old_rewards[max_diff_idx]:.6f}")
        print(f"   新版本: {new_rewards[max_diff_idx]:.6f}")
        return False
    
    # 关闭环境
    old_env.close()
    new_env.close()


def run_multiple_episodes(num_episodes=5):
    """运行多个episode测试"""
    print("🔄 运行多Episode一致性测试")
    print("="*80)
    
    results = []
    
    for i in range(num_episodes):
        print(f"\n### Episode {i+1}/{num_episodes} ###")
        seed = 42 + i * 10
        success = run_episode_comparison(seed=seed, num_steps=200)
        results.append(success)
        print("\n" + "-"*80)
    
    # 总结
    print("\n" + "="*80)
    print("🏁 多Episode测试总结:")
    print(f"   测试Episodes: {num_episodes}")
    print(f"   通过: {sum(results)}/{num_episodes}")
    print(f"   通过率: {sum(results)/num_episodes*100:.1f}%")
    
    if all(results):
        print("\n🎉 所有Episode测试都通过！奖励系统完全一致！")
    else:
        print("\n⚠️ 部分Episode存在差异，需要进一步调查")


if __name__ == "__main__":
    # 运行单个详细的episode测试
    print("="*80)
    print("🧪 Episode级别奖励一致性测试")
    print("="*80)
    
    # 单个episode详细测试
    run_episode_comparison(seed=42, num_steps=200)
    
    print("\n\n")
    
    # 多个episode批量测试
    run_multiple_episodes(num_episodes=3)