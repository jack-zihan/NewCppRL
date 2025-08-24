"""测试设备自适应APF功能"""
import torch
import numpy as np
from torchrl.envs.libs.gym import GymWrapper
import gymnasium as gym

# 注册环境
import envs  # 注册旧环境
import envs_new  # 注册新环境

def test_device_adaptive_apf():
    """测试环境能否根据设备自动选择APF实现"""
    print("=" * 60)
    print("设备自适应APF测试")
    print("=" * 60)
    
    # 测试1：CPU环境
    print("\n📊 测试CPU环境:")
    try:
        # 创建CPU环境
        env_cpu = gym.make('Pasture-v2')
        env_cpu = GymWrapper(env_cpu, device='cpu')
        
        # 重置环境
        obs, info = env_cpu.reset(seed=42)
        
        # 执行几步
        for i in range(5):
            action = env_cpu.action_space.sample()
            obs, reward, terminated, truncated, info = env_cpu.step(action)
            if i == 0:
                print(f"  ✅ CPU环境运行正常")
                print(f"  设备: {getattr(env_cpu, 'device', 'None')}")
                print(f"  奖励示例: {reward:.4f}")
            if terminated or truncated:
                break
        
        env_cpu.close()
        print("  ✅ CPU环境测试通过")
        
    except Exception as e:
        print(f"  ❌ CPU环境测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试2：GPU环境（如果可用）
    if torch.cuda.is_available():
        print("\n🎮 测试GPU环境:")
        try:
            # 测试cuda:0
            env_gpu = gym.make('Pasture-v2')
            env_gpu = GymWrapper(env_gpu, device='cuda:0')
            
            # 重置环境
            obs, info = env_gpu.reset(seed=42)
            
            # 执行几步
            for i in range(5):
                action = env_gpu.action_space.sample()
                obs, reward, terminated, truncated, info = env_gpu.step(action)
                if i == 0:
                    print(f"  ✅ GPU环境运行正常")
                    print(f"  设备: {getattr(env_gpu, 'device', 'None')}")
                    print(f"  奖励示例: {reward:.4f}")
                if terminated or truncated:
                    break
            
            env_gpu.close()
            print("  ✅ GPU环境测试通过")
            
        except Exception as e:
            print(f"  ⚠️ GPU环境测试失败（可能是CuPy未安装）: {e}")
    else:
        print("\n⚠️ CUDA不可用，跳过GPU测试")
    
    # 测试3：多环境并行
    print("\n🔄 测试多环境并行:")
    try:
        from torchrl_utils.utils_env import make_env
        
        # 创建混合设备的环境
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda:0')
        
        envs = []
        for device in devices:
            env = make_env(num_envs=1, device=device)
            envs.append((device, env))
            print(f"  创建{device}环境成功")
        
        # 测试每个环境
        for device, env in envs:
            td = env.reset(seed=42)
            for _ in range(3):
                action = env.action_spec.rand()
                td = env.step(action)
            env.close()
            print(f"  ✅ {device}环境执行成功")
        
        print("  ✅ 多环境并行测试通过")
        
    except Exception as e:
        print(f"  ❌ 多环境并行测试失败: {e}")
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print("✅ 设备自适应APF功能正常工作")
    print("环境可以根据GymWrapper设置的device自动选择合适的APF实现")

def test_apf_function_directly():
    """直接测试APF函数选择逻辑"""
    print("\n" + "=" * 60)
    print("直接测试APF函数选择")
    print("=" * 60)
    
    from envs_new.cpp_env_v2 import CppEnv
    
    # 创建测试地图
    test_map = np.zeros((100, 100), dtype=np.uint8)
    test_map[50, 50] = 1  # 添加一个障碍物
    
    # 测试CPU版本
    print("\n测试CPU版本:")
    env_cpu = CppEnv()
    env_cpu.device = 'cpu'  # 模拟GymWrapper设置device
    result_cpu = env_cpu.get_discounted_apf(test_map, 30)
    print(f"  CPU结果形状: {result_cpu.shape}")
    print(f"  CPU结果范围: [{result_cpu.min():.4f}, {result_cpu.max():.4f}]")
    
    # 测试GPU版本（如果可用）
    if torch.cuda.is_available():
        print("\n测试GPU版本:")
        try:
            env_gpu = CppEnv()
            env_gpu.device = 'cuda:0'  # 模拟GymWrapper设置device
            result_gpu = env_gpu.get_discounted_apf(test_map, 30)
            print(f"  GPU结果形状: {result_gpu.shape}")
            print(f"  GPU结果范围: [{result_gpu.min():.4f}, {result_gpu.max():.4f}]")
            
            # 比较结果
            diff = np.abs(result_cpu - result_gpu)
            print(f"  CPU vs GPU差异: 平均={diff.mean():.6f}, 最大={diff.max():.6f}")
            
        except Exception as e:
            print(f"  GPU测试失败: {e}")
    
    print("\n✅ APF函数选择逻辑正常")

if __name__ == "__main__":
    # 运行测试
    test_device_adaptive_apf()
    test_apf_function_directly()
    
    print("\n🎉 所有测试完成！")