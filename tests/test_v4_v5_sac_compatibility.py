#!/usr/bin/env python3
"""
测试CppEnvV4和CppEnvV5与SAC训练的兼容性
"""

import sys
import torch
import gymnasium as gym
import numpy as np
from pathlib import Path
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs import TransformedEnv, RewardSum, StepCounter
from tensordict import TensorDict

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

import envs  # 注册环境

def test_environment_creation():
    """测试环境创建"""
    print("\n=== 测试环境创建 ===")
    
    for env_id in ["Pasture-v4", "Pasture-v5"]:
        print(f"\n测试 {env_id}:")
        try:
            env = gym.make(env_id, state_pixels=False)
            print(f"✓ {env_id} 创建成功")
            
            # 检查observation space
            obs_space = env.observation_space
            print(f"  Observation space:")
            print(f"    - observation: {obs_space['observation'].shape}")
            print(f"    - vector: {obs_space['vector'].shape}")
            print(f"    - weed_ratio: {obs_space['weed_ratio'].shape}")
            
            env.close()
        except Exception as e:
            print(f"✗ {env_id} 创建失败: {e}")
            return False
    
    return True

def test_observation_format():
    """测试观测格式"""
    print("\n=== 测试观测格式 ===")
    
    for env_id in ["Pasture-v4", "Pasture-v5"]:
        print(f"\n测试 {env_id}:")
        try:
            env = gym.make(env_id, state_pixels=False)
            obs, info = env.reset(seed=42)
            
            # 检查observation
            if 'observation' in obs:
                obs_shape = obs['observation'].shape
                expected_shape = env.observation_space['observation'].shape
                assert obs_shape == expected_shape, f"Observation shape不匹配: {obs_shape} vs {expected_shape}"
                print(f"✓ observation shape正确: {obs_shape}")
            
            # 检查vector (关键测试)
            if 'vector' in obs:
                vector = obs['vector']
                assert isinstance(vector, np.ndarray), f"Vector应该是ndarray，但是{type(vector)}"
                assert vector.shape == (1,), f"Vector shape应该是(1,)，但是{vector.shape}"
                print(f"✓ vector格式正确: type={type(vector)}, shape={vector.shape}, value={vector}")
            
            # 检查weed_ratio
            if 'weed_ratio' in obs:
                weed_ratio = obs['weed_ratio']
                print(f"  weed_ratio: type={type(weed_ratio)}, value={weed_ratio}")
            
            env.close()
        except Exception as e:
            print(f"✗ {env_id} 观测格式测试失败: {e}")
            return False
    
    return True

def test_torchrl_wrapper():
    """测试TorchRL包装器兼容性"""
    print("\n=== 测试TorchRL包装器 ===")
    
    for env_id in ["Pasture-v4", "Pasture-v5"]:
        print(f"\n测试 {env_id}:")
        try:
            # 创建并包装环境
            env = gym.make(env_id, state_pixels=False)
            env = GymWrapper(env, device="cpu")
            env = TransformedEnv(env)
            env.append_transform(RewardSum())
            env.append_transform(StepCounter())
            
            print(f"✓ {env_id} TorchRL包装成功")
            
            # 测试reset
            td = env.reset(seed=42)
            print(f"  Reset返回的TensorDict keys: {list(td.keys())}")
            
            # 检查关键字段
            if 'observation' in td:
                print(f"    - observation shape: {td['observation'].shape}")
            if 'vector' in td:
                print(f"    - vector shape: {td['vector'].shape}")
                assert td['vector'].shape[-1] == 1, f"Vector维度错误: {td['vector'].shape}"
            
            # 测试step
            action = env.action_space.sample()
            td = env.step(TensorDict({'action': torch.tensor(action)}, batch_size=[]))
            print(f"  Step执行成功")
            
            env.close()
        except Exception as e:
            print(f"✗ {env_id} TorchRL包装测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

def test_sac_network_compatibility():
    """测试与SAC网络的兼容性"""
    print("\n=== 测试SAC网络兼容性 ===")
    
    # 简化的网络输入测试
    for env_id in ["Pasture-v4", "Pasture-v5"]:
        print(f"\n测试 {env_id}:")
        try:
            env = gym.make(env_id, state_pixels=False)
            env = GymWrapper(env, device="cpu")
            
            # 获取一个观测
            td = env.reset(seed=42)
            
            # 模拟网络处理
            obs = td['observation']
            vector = td['vector']
            
            # 检查维度
            if env_id == "Pasture-v4":
                assert obs.shape[-3] == 4, f"V4 observation通道数应该是4，但是{obs.shape[-3]}"
            elif env_id == "Pasture-v5":
                assert obs.shape[-3] == 20, f"V5 observation通道数应该是20，但是{obs.shape[-3]}"
            
            assert vector.shape[-1] == 1, f"Vector维度应该是1，但是{vector.shape[-1]}"
            
            print(f"✓ {env_id} 与SAC网络兼容")
            print(f"    - observation: {obs.shape}")
            print(f"    - vector: {vector.shape}")
            
            env.close()
        except Exception as e:
            print(f"✗ {env_id} SAC网络兼容性测试失败: {e}")
            return False
    
    return True

def main():
    """运行所有测试"""
    print("=" * 60)
    print("CppEnvV4 和 CppEnvV5 SAC兼容性测试")
    print("=" * 60)
    
    tests = [
        ("环境创建", test_environment_creation),
        ("观测格式", test_observation_format),
        ("TorchRL包装器", test_torchrl_wrapper),
        ("SAC网络兼容性", test_sac_network_compatibility),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        result = test_func()
        results.append((test_name, result))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总:")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有测试通过！V4和V5环境与SAC训练兼容。")
    else:
        print("\n⚠️ 部分测试失败，需要进一步修复。")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)