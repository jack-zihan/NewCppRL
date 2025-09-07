"""
修正版测试：验证TorchRL GymWrapper的pixels处理
使用正确的PixelObservationWrapper
"""
import gymnasium as gym
import torch
import numpy as np
from torchrl.envs import GymWrapper


def test_pixelwrapper_behavior():
    """测试PixelObservationWrapper的行为"""
    print("\n=== 测试PixelObservationWrapper ===")
    
    # 1. 创建基础环境
    env = gym.make("CartPole-v1", render_mode='rgb_array')
    print(f"原始观察空间: {env.observation_space}")
    
    # 2. 添加PixelObservationWrapper
    from gymnasium.wrappers import PixelObservationWrapper
    
    # pixels_only=False时应该保留原始观察
    env_wrapped = PixelObservationWrapper(env, pixels_only=False)
    print(f"\n包装后观察空间类型: {type(env_wrapped.observation_space)}")
    print(f"包装后观察空间: {env_wrapped.observation_space}")
    
    # 检查观察空间结构
    if hasattr(env_wrapped.observation_space, 'spaces'):
        print(f"观察空间键: {list(env_wrapped.observation_space.spaces.keys())}")
        has_pixels = 'pixels' in env_wrapped.observation_space.spaces
        print(f"pixels键存在: {has_pixels}")
    else:
        print(f"观察空间不是Dict，类型是: {type(env_wrapped.observation_space)}")
        has_pixels = False
    
    # 3. 测试reset输出
    obs, info = env_wrapped.reset()
    print(f"\nreset返回类型: {type(obs)}")
    if isinstance(obs, dict):
        print(f"reset返回键: {list(obs.keys())}")
    else:
        print(f"reset返回shape: {obs.shape if hasattr(obs, 'shape') else 'N/A'}")
    
    return has_pixels


def test_gymwrapper_with_dict_env():
    """测试GymWrapper处理Dict观察空间"""
    print("\n=== 测试GymWrapper + Dict空间 ===")
    
    # 创建自定义Dict环境
    class DictEnv(gym.Env):
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
                'observation': np.zeros(4, dtype=np.float32),
                'vector': np.zeros(2, dtype=np.float32)
            }
            return obs, {}
        
        def step(self, action):
            obs = {
                'observation': np.zeros(4, dtype=np.float32),
                'vector': np.zeros(2, dtype=np.float32)
            }
            return obs, 0.0, False, False, {}
        
        def render(self):
            return np.zeros((64, 64, 3), dtype=np.uint8)
    
    # 测试1: Dict环境 + from_pixels=True
    print("\n1. Dict环境 + from_pixels=True:")
    env1 = DictEnv()
    wrapper1 = GymWrapper(env1, from_pixels=True, pixels_only=False)
    
    print(f"  observation_spec类型: {type(wrapper1.observation_spec)}")
    print(f"  observation_spec键: {list(wrapper1.observation_spec.keys())}")
    
    td1 = wrapper1.reset()
    print(f"  reset输出键: {list(td1.keys())}")
    has_pixels1 = 'pixels' in td1.keys()
    print(f"  包含pixels: {has_pixels1}")
    
    # 测试2: Dict环境 + PixelObservationWrapper
    print("\n2. Dict环境 + PixelObservationWrapper:")
    env2 = DictEnv()
    from gymnasium.wrappers import PixelObservationWrapper
    env2_wrapped = PixelObservationWrapper(env2, pixels_only=False)
    print(f"  包装后空间类型: {type(env2_wrapped.observation_space)}")
    
    if hasattr(env2_wrapped.observation_space, 'spaces'):
        print(f"  包装后空间键: {list(env2_wrapped.observation_space.spaces.keys())}")
    
    wrapper2 = GymWrapper(env2_wrapped, from_pixels=False, pixels_only=False)
    print(f"  observation_spec键: {list(wrapper2.observation_spec.keys())}")
    
    td2 = wrapper2.reset()
    print(f"  reset输出键: {list(td2.keys())}")
    has_pixels2 = 'pixels' in td2.keys()
    print(f"  包含pixels: {has_pixels2}")
    
    return has_pixels1, has_pixels2


def test_gymwrapper_internal_logic():
    """深入测试GymWrapper的内部逻辑"""
    print("\n=== 测试GymWrapper内部逻辑 ===")
    
    # 创建环境
    env = gym.make("CartPole-v1", render_mode='rgb_array')
    
    # 查看from_pixels的处理
    print("\n1. from_pixels=True的处理:")
    wrapper = GymWrapper(env, from_pixels=True, pixels_only=False)
    
    print(f"  wrapper.from_pixels: {wrapper.from_pixels}")
    print(f"  wrapper.pixels_only: {wrapper.pixels_only}")
    print(f"  observation_spec类型: {type(wrapper.observation_spec)}")
    print(f"  observation_spec键: {list(wrapper.observation_spec.keys())}")
    
    # 检查_env属性
    print(f"  wrapper._env类型: {type(wrapper._env)}")
    
    # 测试reset
    td = wrapper.reset()
    print(f"  reset输出键: {list(td.keys())}")
    
    # 测试render
    print("\n2. 测试render()调用:")
    try:
        img = wrapper.render()
        if img is not None:
            print(f"  render()返回shape: {img.shape}")
        else:
            print(f"  render()返回None")
    except Exception as e:
        print(f"  render()失败: {e}")
    
    # 对比: from_pixels=False
    print("\n3. from_pixels=False的处理:")
    wrapper2 = GymWrapper(env, from_pixels=False, pixels_only=False)
    print(f"  observation_spec键: {list(wrapper2.observation_spec.keys())}")
    td2 = wrapper2.reset()
    print(f"  reset输出键: {list(td2.keys())}")


def test_actual_scenario():
    """模拟实际使用场景"""
    print("\n=== 模拟实际场景 ===")
    
    # 模拟env_utils.py中的make_single_environment
    def make_single_environment_mock(from_pixels):
        env_id = "CartPole-v1"
        env = gym.make(env_id, render_mode='rgb_array' if from_pixels else None)
        return GymWrapper(env, device="cpu", from_pixels=from_pixels, pixels_only=False)
    
    # 测试评估场景
    print("\n测试评估场景 (from_pixels=True):")
    eval_env = make_single_environment_mock(from_pixels=True)
    
    print(f"observation_spec键: {list(eval_env.observation_spec.keys())}")
    
    # Reset并检查
    td = eval_env.reset()
    print(f"Reset输出键: {list(td.keys())}")
    
    # 检查pixels
    if 'pixels' in td.keys():
        print(f"✓ pixels存在，shape: {td['pixels'].shape}")
    else:
        print(f"✗ pixels不存在!")
        
        # 尝试直接render
        print("\n尝试直接调用render():")
        try:
            img = eval_env.render()
            if img is not None:
                print(f"  ✓ render()成功，shape: {img.shape}")
            else:
                print(f"  ✗ render()返回None")
        except Exception as e:
            print(f"  ✗ render()失败: {e}")


def main():
    print("=" * 60)
    print("TorchRL GymWrapper pixels处理测试")
    print("=" * 60)
    
    # 测试1: PixelObservationWrapper行为
    has_pixels_wrapped = test_pixelwrapper_behavior()
    
    # 测试2: Dict环境场景
    dict_pixels1, dict_pixels2 = test_gymwrapper_with_dict_env()
    
    # 测试3: 内部逻辑
    test_gymwrapper_internal_logic()
    
    # 测试4: 实际场景
    test_actual_scenario()
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    print("\n关键发现:")
    print("1. PixelObservationWrapper会改变观察空间结构")
    print("2. GymWrapper的from_pixels参数会触发PixelObservationWrapper")
    print("3. 但对于Dict观察空间，处理逻辑可能有问题")
    print("4. 当观察空间已经是Dict/Composite时，from_pixels逻辑可能被跳过")
    
    print("\n结论:")
    if not dict_pixels2:
        print("🔴 确认BUG: Dict观察空间 + from_pixels 处理存在问题")
        print("   解决方案: 直接调用render()是正确的workaround")
    else:
        print("✓ 未发现明显问题，可能是配置或版本问题")


if __name__ == "__main__":
    main()