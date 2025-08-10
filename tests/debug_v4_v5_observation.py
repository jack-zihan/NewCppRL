#!/usr/bin/env python3
"""
调试V4和V5的观测空间问题
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import gymnasium as gym
import envs  # 注册环境

def debug_environment(env_id, **kwargs):
    """调试单个环境的观测设置"""
    print(f"\n{'='*50}")
    print(f"调试 {env_id}")
    print(f"创建参数: {kwargs}")
    print('='*50)
    
    # 创建环境
    env = gym.make(env_id, **kwargs)
    
    # 检查关键属性
    print(f"环境属性:")
    print(f"  - use_sgcnn: {env.use_sgcnn}")
    print(f"  - use_global_obs: {env.use_global_obs}")
    print(f"  - state_downsize: {env.state_downsize}")
    
    # 检查observation_space
    obs_space = env.observation_space
    print(f"\nObservation space:")
    print(f"  - observation: {obs_space['observation'].shape}")
    print(f"  - vector: {obs_space['vector'].shape}")
    
    # 实际reset并检查
    obs, info = env.reset(seed=42)
    print(f"\n实际观测:")
    print(f"  - observation: {obs['observation'].shape}")
    print(f"  - vector: {obs['vector'].shape} = {obs['vector']}")
    
    # 检查observation方法的执行路径
    print(f"\n执行路径分析:")
    if env.use_sgcnn:
        print(f"  ✓ 使用SGCNN多尺度")
        if env.use_global_obs:
            print(f"    ✓ 使用全局观测")
        else:
            print(f"    ✗ 不使用全局观测")
    else:
        print(f"  ✗ 不使用SGCNN（普通观测）")
    
    env.close()
    return obs['observation'].shape

def main():
    """运行调试"""
    print("V4和V5观测空间调试")
    
    # 测试V4：不应该使用SGCNN
    print("\n" + "="*60)
    print("测试1: V4默认创建（应该强制use_sgcnn=False）")
    shape1 = debug_environment("Pasture-v4", state_pixels=False)
    
    print("\n" + "="*60)
    print("测试2: V4显式传入use_sgcnn=True（应该被覆盖为False）")
    shape2 = debug_environment("Pasture-v4", state_pixels=False, use_sgcnn=True, use_global_obs=True)
    
    # 测试V5：应该使用SGCNN
    print("\n" + "="*60)
    print("测试3: V5默认创建（应该强制use_sgcnn=True）")
    shape3 = debug_environment("Pasture-v5", state_pixels=False)
    
    print("\n" + "="*60)
    print("测试4: V5显式传入use_sgcnn=False（应该被覆盖为True）")
    shape4 = debug_environment("Pasture-v5", state_pixels=False, use_sgcnn=False, use_global_obs=False)
    
    # 汇总
    print("\n" + "="*60)
    print("结果汇总:")
    print(f"V4默认: {shape1} - {'✓正确' if shape1 == (4, 128, 128) else '✗错误'}")
    print(f"V4强制: {shape2} - {'✓正确' if shape2 == (4, 128, 128) else '✗错误'}")
    print(f"V5默认: {shape3} - {'✓正确' if shape3 == (20, 16, 16) else '✗错误'}")
    print(f"V5强制: {shape4} - {'✓正确' if shape4 == (20, 16, 16) else '✗错误'}")

if __name__ == "__main__":
    main()