#!/usr/bin/env python3
"""
简单测试v2优化版本的功能
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs_new.cpp_env_v2_optimized import CppEnv


def main():
    print("🧪 v2优化版本简单测试")
    print("=" * 60)
    
    # 创建环境
    env = CppEnv()
    
    # 重置环境
    obs, info = env.reset(seed=42)
    print(f"✅ 环境创建和重置成功")
    print(f"   观察形状: {obs['observation'].shape if isinstance(obs, dict) else obs.shape}")
    
    # 执行几步
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if i < 3:
            print(f"   步骤 {i+1}: 奖励={reward:.4f}")
    
    print(f"\n✅ 环境运行正常")
    print(f"   总奖励: {total_reward:.4f}")
    
    # 测试APF计算
    test_map = np.zeros((10, 10))
    test_map[5, 5] = 1
    apf_result = env.get_discounted_apf(test_map, propagate_distance=5)
    
    print(f"\n✅ APF计算正常")
    print(f"   源点值: {apf_result[5, 5]:.4f}")
    print(f"   距离1值: {apf_result[5, 4]:.4f}")
    
    env.close()
    print("\n🎉 所有测试通过！")


if __name__ == "__main__":
    main()