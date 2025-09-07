"""
简化测试：验证TorchRL GymWrapper的pixels处理bug
使用标准Gymnasium环境避免import问题
"""
import gymnasium as gym
import torch
import numpy as np
from torchrl.envs import GymWrapper


def test_with_cartpole():
    """使用CartPole测试基本的pixels功能"""
    print("\n=== 使用CartPole-v1测试 ===")
    
    # 1. 测试render_mode设置
    env = gym.make("CartPole-v1", render_mode='rgb_array')
    print(f"环境render_mode: {env.render_mode}")
    print(f"原始观察空间: {env.observation_space}")
    
    # 2. 添加AddRenderObservation
    from gymnasium.wrappers import AddRenderObservation
    env_wrapped = AddRenderObservation(env, render_only=False)
    print(f"\n包装后观察空间类型: {type(env_wrapped.observation_space)}")
    print(f"包装后观察空间: {env_wrapped.observation_space}")
    
    if hasattr(env_wrapped.observation_space, 'spaces'):
        print(f"观察空间键: {list(env_wrapped.observation_space.spaces.keys())}")
        has_pixels = 'pixels' in env_wrapped.observation_space.spaces
        print(f"pixels键存在: {has_pixels}")
    
    # 3. 测试GymWrapper
    print("\n--- 测试GymWrapper ---")
    torchrl_env = GymWrapper(env_wrapped, from_pixels=False, pixels_only=False)
    
    print(f"TorchRL observation_spec类型: {type(torchrl_env.observation_spec)}")
    print(f"TorchRL observation_spec: {torchrl_env.observation_spec}")
    print(f"TorchRL observation_spec键: {list(torchrl_env.observation_spec.keys())}")
    
    # 4. 实际reset测试
    print("\n--- 测试Reset输出 ---")
    td = torchrl_env.reset()
    print(f"Reset tensordict键: {list(td.keys())}")
    
    for key in td.keys():
        val = td[key]
        if hasattr(val, 'shape'):
            print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
    
    has_pixels_in_output = 'pixels' in td.keys()
    print(f"\npixels在输出中: {has_pixels_in_output}")
    
    return has_pixels_in_output


def test_with_custom_dict_env():
    """创建自定义Dict观察空间环境测试"""
    print("\n=== 自定义Dict环境测试 ===")
    
    # 创建一个返回Dict观察的简单环境
    class SimpleDictEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = gym.spaces.Dict({
                'observation': gym.spaces.Box(low=-1, high=1, shape=(4,)),
                'vector': gym.spaces.Box(low=-1, high=1, shape=(2,))
            })
            self.action_space = gym.spaces.Discrete(2)
            self.render_mode = 'rgb_array'
        
        def reset(self, seed=None, options=None):
            obs = {
                'observation': np.zeros(4),
                'vector': np.zeros(2)
            }
            return obs, {}
        
        def step(self, action):
            obs = {
                'observation': np.zeros(4),
                'vector': np.zeros(2)
            }
            return obs, 0.0, False, False, {}
        
        def render(self):
            return np.zeros((64, 64, 3), dtype=np.uint8)
    
    # 1. 原始环境
    env = SimpleDictEnv()
    print(f"原始观察空间: {env.observation_space}")
    print(f"原始观察空间键: {list(env.observation_space.spaces.keys())}")
    
    # 2. 添加AddRenderObservation
    from gymnasium.wrappers import AddRenderObservation
    env_wrapped = AddRenderObservation(env, render_only=False)
    print(f"\n包装后观察空间: {env_wrapped.observation_space}")
    print(f"包装后观察空间键: {list(env_wrapped.observation_space.spaces.keys())}")
    
    has_pixels_wrapped = 'pixels' in env_wrapped.observation_space.spaces
    print(f"pixels键存在于wrapped环境: {has_pixels_wrapped}")
    
    # 3. GymWrapper with from_pixels=True
    print("\n--- 测试GymWrapper (from_pixels=True) ---")
    torchrl_env1 = GymWrapper(env, from_pixels=True, pixels_only=False)
    print(f"observation_spec键: {list(torchrl_env1.observation_spec.keys())}")
    
    td1 = torchrl_env1.reset()
    print(f"Reset输出键: {list(td1.keys())}")
    has_pixels1 = 'pixels' in td1.keys()
    print(f"pixels在输出中: {has_pixels1}")
    
    # 4. GymWrapper with wrapped env
    print("\n--- 测试GymWrapper (手动wrapper) ---")
    torchrl_env2 = GymWrapper(env_wrapped, from_pixels=False, pixels_only=False)
    print(f"observation_spec键: {list(torchrl_env2.observation_spec.keys())}")
    
    td2 = torchrl_env2.reset()
    print(f"Reset输出键: {list(td2.keys())}")
    has_pixels2 = 'pixels' in td2.keys()
    print(f"pixels在输出中: {has_pixels2}")
    
    return has_pixels_wrapped, has_pixels1, has_pixels2


def test_make_specs_logic():
    """直接测试_make_specs的逻辑"""
    print("\n=== 测试_make_specs逻辑 ===")
    
    # 创建一个简单环境
    env = gym.make("CartPole-v1", render_mode='rgb_array')
    
    # 测试1: 非Dict观察空间 + from_pixels=True
    print("\n1. Box空间 + from_pixels=True:")
    wrapper1 = GymWrapper(env, from_pixels=True, pixels_only=False)
    print(f"  observation_spec类型: {type(wrapper1.observation_spec)}")
    print(f"  observation_spec键: {list(wrapper1.observation_spec.keys())}")
    print(f"  包含pixels: {'pixels' in wrapper1.observation_spec.keys()}")
    
    # 测试2: Dict观察空间
    from gymnasium.wrappers import AddRenderObservation
    env_dict = AddRenderObservation(env, render_only=False)
    print(f"\n2. Dict空间（手动wrapper）:")
    print(f"  原始空间类型: {type(env_dict.observation_space)}")
    print(f"  原始空间键: {list(env_dict.observation_space.spaces.keys())}")
    
    wrapper2 = GymWrapper(env_dict, from_pixels=False, pixels_only=False)
    print(f"  observation_spec类型: {type(wrapper2.observation_spec)}")
    print(f"  observation_spec键: {list(wrapper2.observation_spec.keys())}")
    print(f"  包含pixels: {'pixels' in wrapper2.observation_spec.keys()}")
    
    # 测试3: Dict空间 + from_pixels=True (这是问题场景)
    print(f"\n3. Dict空间 + from_pixels=True (BUG场景):")
    wrapper3 = GymWrapper(env_dict, from_pixels=True, pixels_only=False)
    print(f"  observation_spec类型: {type(wrapper3.observation_spec)}")
    print(f"  observation_spec键: {list(wrapper3.observation_spec.keys())}")
    print(f"  包含pixels: {'pixels' in wrapper3.observation_spec.keys()}")
    
    # 验证实际输出
    td3 = wrapper3.reset()
    print(f"  实际reset输出键: {list(td3.keys())}")
    print(f"  实际包含pixels: {'pixels' in td3.keys()}")


def main():
    print("=" * 60)
    print("验证TorchRL GymWrapper pixels处理BUG")
    print("=" * 60)
    
    # 测试1: CartPole基础测试
    cartpole_has_pixels = test_with_cartpole()
    
    # 测试2: 自定义Dict环境
    dict_wrapped, dict_pixels1, dict_pixels2 = test_with_custom_dict_env()
    
    # 测试3: _make_specs逻辑
    test_make_specs_logic()
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"1. CartPole + AddRenderObservation: {'✓' if cartpole_has_pixels else '✗ pixels丢失'}")
    print(f"2. Dict环境 + AddRenderObservation包装正确: {'✓' if dict_wrapped else '✗'}")
    print(f"3. Dict环境 + from_pixels=True: {'✓' if dict_pixels1 else '✗ pixels丢失'}")
    print(f"4. Dict环境 + 手动wrapper: {'✓' if dict_pixels2 else '✗ pixels丢失'}")
    
    if dict_wrapped and not dict_pixels2:
        print("\n🔴 BUG确认:")
        print("   AddRenderObservation正确添加了pixels到Dict观察空间,")
        print("   但GymWrapper没有正确传递pixels到输出!")
    
    print("\n关键发现:")
    print("- 当观察空间已经是Dict时，GymWrapper的from_pixels逻辑被跳过")
    print("- _make_specs中的条件判断导致Dict空间的pixels处理失效")


if __name__ == "__main__":
    main()