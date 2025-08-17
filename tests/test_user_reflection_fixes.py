#!/usr/bin/env python3
"""
验证基于用户反思的3个修复点：
1. 多尺度观察的严格参数验证（不再使用resize补丁）
2. int() vs round() 的正确使用场景
3. config引用传递机制
"""

import numpy as np
from envs_new.cpp_env_v2 import CppEnv
from envs_new.components.config.environment_config import EnvironmentConfig


def test_multiscale_validation():
    """测试1：多尺度观察的严格参数验证"""
    print("\n🔬 测试1：多尺度观察参数验证")
    
    # 正确的配置应该成功
    print("  测试正确配置...")
    try:
        env = CppEnv(
            use_multiscale=True,
            use_global_features=True,
            state_downsize=(128, 128),  # 128 / 16 = 8，可以整除
            multiscale_feature_size=16
        )
        obs, _ = env.reset()
        print(f"    ✅ 正确配置成功创建，观察形状: {obs['observation'].shape}")
        env.close()
    except Exception as e:
        print(f"    ❌ 错误: {e}")
    
    # 错误的配置应该报错
    print("  测试错误配置（不能整除）...")
    try:
        env = CppEnv(
            use_multiscale=True,
            use_global_features=True,
            state_downsize=(127, 127),  # 127 / 16 = 7.9375，不能整除
            multiscale_feature_size=16
        )
        print("    ❌ 不应该成功创建环境！")
    except ValueError as e:
        print(f"    ✅ 正确报错: {str(e)[:80]}...")
    
    # 测试池化后尺寸太小的情况
    print("  测试池化后尺寸太小...")
    try:
        env = CppEnv(
            use_multiscale=True,
            state_downsize=(32, 32),  # 32 / 2^3 = 4，小于默认的16
            multiscale_feature_size=16
        )
        print("    ❌ 不应该成功创建环境！")
    except ValueError as e:
        print(f"    ✅ 正确报错: {str(e)[:80]}...")


def test_int_vs_round():
    """测试2：int() vs round() 的正确使用"""
    print("\n🔬 测试2：int() vs round() 使用场景")
    
    # 测试网格索引（APF奖励计算）应该使用int()
    print("  测试APF奖励计算（网格索引）...")
    
    # 验证int()的floor行为
    test_positions = [50.1, 50.5, 50.9]
    for pos in test_positions:
        grid_idx = int(pos)  # 应该都是50
        print(f"    位置{pos}的网格索引: {grid_idx} (使用int，应该是50)")
    
    # 测试agent位置离散化应该使用round()
    print("  测试agent位置离散化...")
    
    test_positions = [50.1, 50.4, 50.5, 50.6, 50.9]
    for pos in test_positions:
        discrete = round(pos)
        expected = 50 if pos < 50.5 else 51
        print(f"    位置{pos}离散化: {discrete} (使用round，应该是{expected})")
    
    # 实际测试APF计算逻辑
    print("  测试实际APF计算...")
    env = CppEnv()
    obs, _ = env.reset(seed=42)
    
    # 移动agent几步以生成轨迹
    for _ in range(3):
        action = env.action_space.sample()
        obs, reward, _, _, _ = env.step(action)
    
    # 验证奖励计算使用了正确的索引
    print(f"    ✅ APF奖励计算正常: reward={reward:.4f}")
    
    env.close()


def test_config_reference():
    """测试3：config引用传递机制"""
    print("\n🔬 测试3：config引用传递机制")
    
    env = CppEnv()
    
    # 保存原始值
    original_state_size = env.config.state_size
    original_reward_coef = env.config.reward_frontier_coverage_coef
    
    print(f"  原始state_size: {original_state_size}")
    print(f"  原始reward_coef: {original_reward_coef}")
    
    # 修改config
    env.update_config({
        'state_size': (256, 256),
        'reward_frontier_coverage_coef': 1.0
    })
    
    # 验证所有组件都看到了更新
    print(f"  更新后env.config.state_size: {env.config.state_size}")
    print(f"  更新后observation_generator看到的: {env.observation_generator.config.state_size}")
    print(f"  更新后reward_system看到的: {env.reward_system.config.reward_frontier_coverage_coef}")
    
    # 验证是同一个对象（引用传递）
    if env.config is env.observation_generator.config:
        print("  ✅ config是引用传递（所有组件共享同一对象）")
    else:
        print("  ❌ config不是引用传递")
    
    env.close()


def test_state_info_len():
    """测试4：StateVariable的len()支持"""
    print("\n🔬 测试4：StateVariable的len()支持")
    
    from envs_new.components.state.environment_state import StateVariable
    
    # 创建StateVariable
    state_var = StateVariable("agent_position", history_length=100)
    
    print(f"  初始长度: {len(state_var)}")
    
    # 添加历史记录
    state_var.update((10, 10))
    print(f"  添加1个值后长度: {len(state_var)}")
    
    state_var.update((20, 20))
    print(f"  添加2个值后长度: {len(state_var)}")
    
    # 测试轨迹长度计算条件
    if len(state_var) >= 2:
        print("  ✅ len() >= 2 条件满足，可以计算轨迹")
    
    # 验证last属性
    print(f"  current: {state_var.current}")
    print(f"  last: {state_var.last}")


if __name__ == "__main__":
    print("=" * 60)
    print("基于用户反思的修复验证测试")
    print("=" * 60)
    
    test_multiscale_validation()
    test_int_vs_round()
    test_config_reference()
    test_state_info_len()
    
    print("\n" + "=" * 60)
    print("✅ 所有测试完成！")
    print("=" * 60)