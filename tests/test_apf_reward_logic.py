#!/usr/bin/env python3
"""
测试APF奖励计算逻辑的正确性
验证障碍物奖励的正负号问题
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs_new.cpp_env_v2 import CppEnv


def test_apf_reward_logic():
    """测试APF奖励计算的逻辑"""
    print("=" * 60)
    print("APF奖励逻辑分析")
    print("=" * 60)
    
    # 创建环境
    env = CppEnv()
    obs, info = env.reset(seed=42)
    
    # 模拟APF场值
    print("\n1. APF势场值含义：")
    print("   - 障碍物处：APF = 1.0（最高值）")
    print("   - 远离障碍物：APF ≈ 0.0（最低值）")
    
    # 模拟靠近障碍物的场景
    print("\n2. 场景分析：")
    
    # 场景1：靠近障碍物
    apf_prev = 0.2  # 之前远离障碍物
    apf_curr = 0.8  # 现在靠近障碍物
    delta = apf_curr - apf_prev  # = 0.6 > 0
    
    print(f"\n场景1 - 靠近障碍物：")
    print(f"   之前APF = {apf_prev}, 现在APF = {apf_curr}")
    print(f"   变化量 = {delta:.2f} (正值)")
    
    # 当前代码的计算
    reward_obstacle = 0.3 * delta
    reward_obstacle_clipped = min(0., reward_obstacle)
    print(f"   当前代码：reward = 0.3 * {delta:.2f} = {reward_obstacle:.2f}")
    print(f"   经过min(0, reward) = {reward_obstacle_clipped:.2f}")
    print(f"   ❌ 问题：靠近障碍物没有惩罚！")
    
    # 正确的计算方式
    correct_reward = -0.3 * delta  # 或 0.3 * (apf_prev - apf_curr)
    correct_clipped = min(0., correct_reward)
    print(f"   正确方式：reward = -0.3 * {delta:.2f} = {correct_reward:.2f}")
    print(f"   经过min(0, reward) = {correct_clipped:.2f}")
    print(f"   ✅ 正确：靠近障碍物得到负奖励（惩罚）")
    
    # 场景2：远离障碍物
    apf_prev2 = 0.8  # 之前靠近障碍物
    apf_curr2 = 0.2  # 现在远离障碍物
    delta2 = apf_curr2 - apf_prev2  # = -0.6 < 0
    
    print(f"\n场景2 - 远离障碍物：")
    print(f"   之前APF = {apf_prev2}, 现在APF = {apf_curr2}")
    print(f"   变化量 = {delta2:.2f} (负值)")
    
    # 当前代码的计算
    reward_obstacle2 = 0.3 * delta2
    reward_obstacle_clipped2 = min(0., reward_obstacle2)
    print(f"   当前代码：reward = 0.3 * {delta2:.2f} = {reward_obstacle2:.2f}")
    print(f"   经过min(0, reward) = {reward_obstacle_clipped2:.2f}")
    print(f"   ❌ 问题：远离障碍物反而有惩罚！")
    
    # 正确的计算方式
    correct_reward2 = -0.3 * delta2  # 或 0.3 * (apf_prev2 - apf_curr2)
    correct_clipped2 = min(0., correct_reward2)
    print(f"   正确方式：reward = -0.3 * {delta2:.2f} = {correct_reward2:.2f}")
    print(f"   经过min(0, reward) = {correct_clipped2:.2f}")
    print(f"   ✅ 正确：远离障碍物没有惩罚")
    
    print("\n" + "=" * 60)
    print("结论：")
    print("=" * 60)
    print("当前代码存在逻辑错误！")
    print("障碍物奖励应该使用：")
    print("  reward_obstacle = -0.3 * (当前APF - 之前APF)")
    print("  或")
    print("  reward_obstacle = 0.3 * (之前APF - 当前APF)")
    print("\n这样才能实现：")
    print("  - 靠近障碍物 → 负奖励（惩罚）")
    print("  - 远离障碍物 → 0奖励（不惩罚）")
    
    # 测试weed的逻辑
    print("\n" + "=" * 60)
    print("杂草奖励逻辑分析")
    print("=" * 60)
    
    print("\n杂草APF场：")
    print("  - 有杂草处：APF = 1.0")
    print("  - 无杂草处：APF = 0.0")
    
    print("\n当agent清除杂草时：")
    print("  - 之前位置有杂草：APF = 1.0")
    print("  - 现在位置无杂草：APF = 0.0")
    print("  - 变化量 = 0.0 - 1.0 = -1.0")
    print("  - reward = 5.0 * (-1.0) = -5.0")
    print("  ❌ 问题：清除杂草反而得到负奖励！")
    
    print("\n正确应该是：")
    print("  - 当进入有杂草的区域并清除时，应该得到正奖励")
    print("  - 可能需要检查地图更新逻辑，或者调整奖励计算")
    
    env.close()


if __name__ == "__main__":
    test_apf_reward_logic()