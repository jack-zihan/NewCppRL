"""
测试pixels键缺失的bug
验证TorchRL的GymWrapper是否正确处理Dict观察空间+from_pixels组合
"""
import gymnasium as gym
import torch
import numpy as np
from torchrl.envs import GymWrapper
import sys
sys.path.append('/home/lzh/NewCppRL')

def test_pixels_in_observation_space():
    """测试1: 验证AddRenderObservation是否正确添加pixels到观察空间"""
    print("\n=== 测试1: 观察空间检查 ===")
    
    # 创建环境，设置render_mode
    env_base = gym.make("NewPasture-v5", render_mode='rgb_array')
    print(f"原始观察空间键: {list(env_base.observation_space.spaces.keys())}")
    
    # 应用AddRenderObservation wrapper
    from gymnasium.wrappers import AddRenderObservation
    env_wrapped = AddRenderObservation(env_base, render_only=False)
    print(f"包装后观察空间键: {list(env_wrapped.observation_space.spaces.keys())}")
    
    # 检查pixels是否存在
    has_pixels = 'pixels' in env_wrapped.observation_space.spaces
    print(f"pixels键存在于包装后的环境: {has_pixels}")
    
    return env_wrapped, has_pixels


def test_gymwrapper_spec_conversion():
    """测试2: 验证GymWrapper的spec转换是否保留pixels键"""
    print("\n=== 测试2: GymWrapper spec转换 ===")
    
    # 创建包装后的环境
    env_base = gym.make("NewPasture-v5", render_mode='rgb_array')
    from gymnasium.wrappers import AddRenderObservation
    env_wrapped = AddRenderObservation(env_base, render_only=False)
    
    # 使用GymWrapper
    torchrl_env = GymWrapper(env_wrapped, from_pixels=True, pixels_only=False)
    
    # 检查observation_spec
    print(f"TorchRL observation_spec类型: {type(torchrl_env.observation_spec)}")
    print(f"TorchRL observation_spec键: {list(torchrl_env.observation_spec.keys())}")
    
    # 检查pixels是否在spec中
    has_pixels_in_spec = 'pixels' in torchrl_env.observation_spec.keys()
    print(f"pixels键存在于TorchRL spec: {has_pixels_in_spec}")
    
    return torchrl_env, has_pixels_in_spec


def test_actual_observation_output():
    """测试3: 验证实际reset和step输出是否包含pixels"""
    print("\n=== 测试3: 实际观察输出 ===")
    
    # 方式1: 直接使用from_pixels
    env1 = gym.make("NewPasture-v5", render_mode='rgb_array')
    torchrl_env1 = GymWrapper(env1, from_pixels=True, pixels_only=False)
    
    # Reset并检查输出
    td1 = torchrl_env1.reset()
    print(f"\n方式1 (from_pixels=True):")
    print(f"Reset输出键: {list(td1.keys())}")
    has_pixels_method1 = 'pixels' in td1.keys()
    print(f"pixels在输出中: {has_pixels_method1}")
    
    # 方式2: 手动添加wrapper
    env2 = gym.make("NewPasture-v5", render_mode='rgb_array')
    from gymnasium.wrappers import AddRenderObservation
    env2_wrapped = AddRenderObservation(env2, render_only=False)
    torchrl_env2 = GymWrapper(env2_wrapped, from_pixels=False, pixels_only=False)
    
    td2 = torchrl_env2.reset()
    print(f"\n方式2 (手动wrapper):")
    print(f"Reset输出键: {list(td2.keys())}")
    has_pixels_method2 = 'pixels' in td2.keys()
    print(f"pixels在输出中: {has_pixels_method2}")
    
    return has_pixels_method1, has_pixels_method2


def test_custom_env_creation():
    """测试4: 模拟env_utils.py的创建过程"""
    print("\n=== 测试4: 模拟实际创建过程 ===")
    
    # 模拟make_single_environment
    from_pixels = True
    env = gym.make("NewPasture-v5", render_mode='rgb_array' if from_pixels else None)
    print(f"环境render_mode: {env.render_mode}")
    
    torchrl_env = GymWrapper(env, device="cpu", from_pixels=from_pixels, pixels_only=False)
    
    # 检查spec
    print(f"observation_spec键: {list(torchrl_env.observation_spec.keys())}")
    
    # Reset并检查
    td = torchrl_env.reset()
    print(f"Reset tensordict键: {list(td.keys())}")
    
    # 检查pixels
    has_pixels = 'pixels' in td.keys()
    print(f"pixels存在: {has_pixels}")
    
    # 如果pixels不存在，检查是否能直接调用render
    if not has_pixels:
        print("\npixels不存在，尝试直接调用render():")
        try:
            image = torchrl_env.render()
            if image is not None:
                print(f"  render()成功，返回shape: {image.shape if hasattr(image, 'shape') else type(image)}")
            else:
                print(f"  render()返回None")
        except Exception as e:
            print(f"  render()失败: {e}")
    
    return torchrl_env, has_pixels


def main():
    print("=" * 60)
    print("测试TorchRL GymWrapper的pixels处理bug")
    print("=" * 60)
    
    # 运行各项测试
    try:
        # 测试1: 检查AddRenderObservation是否工作
        env_wrapped, has_pixels_wrapped = test_pixels_in_observation_space()
        
        # 测试2: 检查GymWrapper的spec转换
        torchrl_env, has_pixels_spec = test_gymwrapper_spec_conversion()
        
        # 测试3: 检查实际输出
        has_pixels1, has_pixels2 = test_actual_observation_output()
        
        # 测试4: 模拟实际创建
        final_env, has_pixels_final = test_custom_env_creation()
        
        # 总结
        print("\n" + "=" * 60)
        print("测试总结:")
        print("=" * 60)
        print(f"1. AddRenderObservation添加pixels到观察空间: {'✓' if has_pixels_wrapped else '✗'}")
        print(f"2. GymWrapper spec包含pixels: {'✓' if has_pixels_spec else '✗'}")
        print(f"3. from_pixels=True时输出包含pixels: {'✓' if has_pixels1 else '✗'}")
        print(f"4. 手动wrapper时输出包含pixels: {'✓' if has_pixels2 else '✗'}")
        print(f"5. 实际创建流程输出包含pixels: {'✓' if has_pixels_final else '✗'}")
        
        if has_pixels_wrapped and not has_pixels_final:
            print("\n🔴 BUG确认: AddRenderObservation正确添加了pixels，")
            print("   但GymWrapper在处理Dict观察空间时丢失了pixels键!")
        
    except Exception as e:
        print(f"\n测试过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()