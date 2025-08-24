#!/usr/bin/env python3
"""
Debug FieldCoverageUpdater to see why field isn't being covered
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_field_coverage_debug():
    """详细调试FieldCoverageUpdater"""
    print("\n=== 调试FieldCoverageUpdater ===")
    
    from envs_new.cpp_env_v4 import CppEnv as CppEnvV4
    env = CppEnvV4()
    
    # 确认使用了正确的updater
    updater = env.env_dynamics._updaters.get('field')
    print(f"Field updater类型: {updater.__class__.__name__}")
    
    obs, _ = env.reset(seed=42)
    
    # 获取初始状态
    initial_field = env.maps_dict['field'].copy()
    initial_sum = initial_field.sum()
    print(f"初始field总和: {initial_sum}")
    print(f"初始agent位置: {env.agent.position}")
    print(f"初始agent凸包: {env.agent.convex_hull}")
    
    # 执行多步，使用固定动作确保机器人移动
    for i in range(10):
        # 使用前进动作（假设动作空间中间值是前进）
        if env.config.action_type == "discrete":
            # 对于离散动作，选择一个前进的动作
            action = env.action_space.n // 2  # 中间动作通常是直行
        else:
            action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        current_field = env.maps_dict['field']
        current_sum = current_field.sum()
        
        print(f"\n步骤 {i+1}:")
        print(f"  Agent位置: {env.agent.position}")
        print(f"  Agent速度: {env.agent.speed}")
        print(f"  Field总和: {current_sum} (变化: {initial_sum - current_sum})")
        print(f"  Field覆盖率: {info.get('field_ratio', 'N/A')}")
        
        if current_sum < initial_sum:
            print(f"  ✅ Field被覆盖了！")
            break
    
    # 最终状态
    final_field = env.maps_dict['field']
    final_sum = final_field.sum()
    
    print(f"\n最终结果:")
    print(f"  初始field总和: {initial_sum}")
    print(f"  最终field总和: {final_sum}")
    print(f"  覆盖的像素数: {initial_sum - final_sum}")
    
    # 检查凸包是否正确
    print(f"\n凸包调试:")
    print(f"  凸包形状: {env.agent.convex_hull.shape}")
    print(f"  凸包: {env.agent.convex_hull}")
    
    env.close()
    
    return final_sum < initial_sum


def test_updater_directly():
    """直接测试FieldCoverageUpdater"""
    print("\n=== 直接测试FieldCoverageUpdater ===")
    
    from envs_new.components.dynamics.environment_dynamics import FieldCoverageUpdater
    from envs_new.cpp_env_v4 import CppEnv as CppEnvV4
    
    env = CppEnvV4()
    obs, _ = env.reset(seed=42)
    
    # 手动创建updater
    updater = FieldCoverageUpdater()
    
    # 准备state
    state = {
        'maps_dict': env.maps_dict,
        'agent': env.agent,
        'env_state': env.env_state
    }
    
    # 初始化
    updater.setup_state(state, 2)
    
    initial_field = env.maps_dict['field'].copy()
    initial_sum = initial_field.sum()
    print(f"初始field总和: {initial_sum}")
    
    # 移动agent到一个新位置
    env.agent.position = np.array([128, 128])
    env.agent.update_convex_hull()
    
    print(f"Agent新位置: {env.agent.position}")
    print(f"Agent凸包: {env.agent.convex_hull}")
    
    # 执行更新
    updater.update(state)
    
    final_sum = env.maps_dict['field'].sum()
    print(f"更新后field总和: {final_sum}")
    print(f"覆盖像素数: {initial_sum - final_sum}")
    
    env.close()
    
    return final_sum < initial_sum


if __name__ == "__main__":
    print("=" * 60)
    print("调试FieldCoverageUpdater")
    print("=" * 60)
    
    # 测试1：通过环境测试
    result1 = test_field_coverage_debug()
    
    # 测试2：直接测试updater
    result2 = test_updater_directly()
    
    if result1 and result2:
        print("\n✅ FieldCoverageUpdater工作正常")
    else:
        print("\n❌ FieldCoverageUpdater存在问题")
        if not result1:
            print("  - 通过环境测试失败")
        if not result2:
            print("  - 直接测试失败")