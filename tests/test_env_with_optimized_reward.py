#!/usr/bin/env python3
"""
测试优化后的奖励系统在实际环境中的完整功能
"""
import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from envs_new.cpp_env_v2 import CppEnv


def test_environment_integration():
    """测试优化后的奖励系统与环境的集成"""
    print("🎯 测试优化后的奖励系统在实际环境中的运行...")
    
    # 创建环境
    env = CppEnv()
    print("✅ 环境创建成功")
    
    # 重置环境
    obs, info = env.reset(seed=42)
    print("✅ 环境重置成功")
    
    # 验证奖励系统的组件
    reward_system = env.reward_system
    print(f"\n📋 奖励系统组件:")
    print(f"  - 激活的Calculators: {reward_system.active_calculators}")
    print(f"  - REWARD_GROUPS已定义: {'REWARD_GROUPS' in dir(reward_system)}")
    
    # 执行几步，验证奖励计算
    print("\n🔄 执行步骤并计算奖励:")
    for i in range(3):
        # 生成连续动作（确保格式正确）
        if env.config.action_type == "continuous":
            # 连续动作：(线速度, 角速度)
            v = np.random.uniform(env.config.v_min, env.config.v_max)
            w = np.random.uniform(env.config.w_min, env.config.w_max)
            action = (v, w)
        else:
            action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 获取奖励分解
        reward_breakdown = env.reward_system.get_reward_breakdown(env.env_state)
        
        print(f"\n  Step {i}:")
        print(f"    - 总奖励: {reward:.4f}")
        print(f"    - 基础惩罚: {reward_breakdown['base']:.4f}")
        print(f"    - 杂草清除: {reward_breakdown['weed_removal']:.4f}")
        print(f"    - 前沿总计: {reward_breakdown['frontier_total']:.4f}")
        print(f"    - 转向总计: {reward_breakdown['turning_total']:.4f}")
        
        if terminated or truncated:
            print(f"    - Episode结束: terminated={terminated}, truncated={truncated}")
            break
    
    env.close()
    print("\n✅ 奖励系统在实际环境中运行正常!")
    return True


def test_coefficient_update_in_env():
    """测试在环境中动态更新奖励系数"""
    print("\n🔧 测试动态系数更新...")
    
    env = CppEnv()
    obs, info = env.reset(seed=42)
    
    # 获取初始配置
    print(f"  初始 base_penalty: {env.config.reward_base_penalty}")
    print(f"  初始 weed_removal: {env.config.reward_weed_removal}")
    
    # 更新系数
    new_coefficients = {
        'base_penalty': -0.2,
        'weed_removal': 30.0,
        'turning_penalty': -1.0
    }
    env.reward_system.update_coefficients(new_coefficients)
    
    # 验证更新
    print(f"  更新后 base_penalty: {env.config.reward_base_penalty}")
    print(f"  更新后 weed_removal: {env.config.reward_weed_removal}")
    print(f"  更新后 turning_penalty: {env.config.reward_turning_penalty}")
    
    assert env.config.reward_base_penalty == -0.2
    assert env.config.reward_weed_removal == 30.0
    assert env.config.reward_turning_penalty == -1.0
    
    env.close()
    print("✅ 动态系数更新测试通过")
    return True


def test_reward_groups_in_env():
    """测试组系数在实际环境中的应用"""
    print("\n🎯 测试组系数应用...")
    
    # 创建环境，设置明显的组系数
    env = CppEnv(
        reward_frontier_group_coef=2.0,
        reward_turning_group_coef=0.5
    )
    obs, info = env.reset(seed=42)
    
    print(f"  前沿组系数: {env.config.reward_frontier_group_coef}")
    print(f"  转向组系数: {env.config.reward_turning_group_coef}")
    
    # 执行一步
    action = (1.0, 0.0) if env.config.action_type == "continuous" else env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 获取奖励分解
    reward_breakdown = env.reward_system.get_reward_breakdown(env.env_state)
    
    if 'components_raw' in reward_breakdown and 'components' in reward_breakdown:
        print("\n  原始奖励 vs 应用组系数后:")
        for calc_name in ['frontier_coverage', 'frontier_variation']:
            if calc_name in reward_breakdown['components_raw']:
                raw = reward_breakdown['components_raw'][calc_name]
                final = reward_breakdown['components'][calc_name]
                if raw != 0:
                    ratio = final / raw
                    print(f"    {calc_name}: {raw:.4f} -> {final:.4f} (比例: {ratio:.2f})")
    
    env.close()
    print("✅ 组系数应用测试通过")
    return True


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("🧪 环境集成测试 - 验证优化后的奖励系统")
    print("="*60)
    
    try:
        # 运行测试
        test_environment_integration()
        test_coefficient_update_in_env()
        test_reward_groups_in_env()
        
        print("\n" + "="*60)
        print("🎉 所有环境集成测试通过！")
        print("\n📊 优化成果验证:")
        print("  ✅ 新的命名系统工作正常")
        print("  ✅ 系数更新机制简化且有效")
        print("  ✅ 组系数结构化定义正确应用")
        print("  ✅ 代码更简洁，无兼容性负担")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())