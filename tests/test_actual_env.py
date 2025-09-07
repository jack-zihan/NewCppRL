"""
测试实际的NewPasture-v5环境
"""
import sys
sys.path.append('/home/lzh/NewCppRL')

# 先导入envs_new以注册环境
import envs_new

import gymnasium as gym
import torch
from torchrl.envs import GymWrapper


def test_newpasture_pixels():
    """测试NewPasture-v5的pixels处理"""
    print("\n=== 测试NewPasture-v5 ===")
    
    # 1. 创建环境
    print("\n1. 创建环境 (render_mode='rgb_array'):")
    env = gym.make("NewPasture-v5", render_mode='rgb_array')
    print(f"  环境类型: {type(env)}")
    print(f"  render_mode: {env.render_mode}")
    print(f"  观察空间类型: {type(env.observation_space)}")
    print(f"  观察空间: {env.observation_space}")
    
    if hasattr(env.observation_space, 'spaces'):
        print(f"  观察空间键: {list(env.observation_space.spaces.keys())}")
    
    # 2. 测试GymWrapper with from_pixels=True
    print("\n2. GymWrapper (from_pixels=True):")
    wrapper = GymWrapper(env, device="cpu", from_pixels=True, pixels_only=False)
    
    print(f"  wrapper.from_pixels: {wrapper.from_pixels}")
    print(f"  wrapper._env类型: {type(wrapper._env)}")
    print(f"  observation_spec类型: {type(wrapper.observation_spec)}")
    print(f"  observation_spec键: {list(wrapper.observation_spec.keys())}")
    
    # 3. 测试reset
    print("\n3. Reset测试:")
    td = wrapper.reset()
    print(f"  输出键: {list(td.keys())}")
    
    # 详细检查每个键
    for key in td.keys():
        if hasattr(td[key], 'shape'):
            print(f"    {key}: shape={td[key].shape}, dtype={td[key].dtype}")
    
    # 4. 检查pixels
    has_pixels = 'pixels' in td.keys()
    print(f"\n  pixels存在: {has_pixels}")
    
    if not has_pixels:
        # 尝试直接render
        print("\n  pixels不存在，尝试直接render():")
        try:
            img = wrapper.render()
            if img is not None:
                print(f"    render()成功: shape={img.shape}")
            else:
                print(f"    render()返回None")
        except Exception as e:
            print(f"    render()失败: {e}")
    
    # 5. 测试step
    print("\n4. Step测试:")
    action = wrapper.action_space.sample()
    td_next = wrapper.step(td.set("action", action))
    print(f"  Step后输出键: {list(td_next.keys())}")
    
    # 查看next字典
    if 'next' in td_next.keys():
        print(f"  Next子字典键: {list(td_next['next'].keys())}")
        has_pixels_next = 'pixels' in td_next['next'].keys()
        print(f"  Next中pixels存在: {has_pixels_next}")
    
    return has_pixels


def test_with_env_utils():
    """使用env_utils的方式创建环境"""
    print("\n=== 使用env_utils方式 ===")
    
    from rl_new.sac_cont_sy.env_utils import make_single_environment
    from omegaconf import DictConfig
    
    # 模拟配置
    cfg = DictConfig({
        'env': {
            'env_id': 'NewPasture-v5',
            'env_kwargs': {}
        }
    })
    
    # 创建环境
    print("\n使用make_single_environment:")
    env = make_single_environment(cfg, device="cpu", seed=0, from_pixels=True)
    
    print(f"环境类型: {type(env)}")
    print(f"observation_spec键: {list(env.observation_spec.keys())}")
    
    # Reset
    td = env.reset()
    print(f"Reset输出键: {list(td.keys())}")
    
    has_pixels = 'pixels' in td.keys()
    print(f"pixels存在: {has_pixels}")
    
    return has_pixels


def main():
    print("=" * 60)
    print("NewPasture-v5 Pixels测试")
    print("=" * 60)
    
    # 测试1: 直接测试
    has_pixels1 = test_newpasture_pixels()
    
    # 测试2: 使用env_utils
    try:
        has_pixels2 = test_with_env_utils()
    except Exception as e:
        print(f"\nenv_utils测试失败: {e}")
        has_pixels2 = False
    
    # 总结
    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)
    print(f"1. 直接GymWrapper测试: {'✓ pixels存在' if has_pixels1 else '✗ pixels缺失'}")
    print(f"2. env_utils方式测试: {'✓ pixels存在' if has_pixels2 else '✗ pixels缺失'}")
    
    if not has_pixels1 or not has_pixels2:
        print("\n🔴 发现问题:")
        print("   NewPasture-v5的Dict观察空间与from_pixels组合存在问题")
        print("   这证实了你的猜想是正确的!")
    else:
        print("\n✓ pixels正常工作")


if __name__ == "__main__":
    main()