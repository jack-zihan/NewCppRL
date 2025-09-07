"""
深入测试SAC Actor的行为，验证TorchRL的设计理念
"""
import sys
sys.path.append('/home/lzh/NewCppRL')

# 先导入envs_new以注册环境
import envs_new

import torch
from torchrl.envs import GymWrapper
from tensordict import TensorDict
import gymnasium as gym

# 导入真实的SAC模型创建函数
from rl_new.sac_cont_sy.model_utils import make_sac_models
from rl_new.sac_cont_sy.env_utils import make_single_environment
from omegaconf import OmegaConf


def test_real_actor_behavior():
    """测试真实SAC actor对tensordict的处理"""
    print("\n" + "="*80)
    print("测试真实SAC Actor对TensorDict的处理")
    print("="*80)
    
    # 1. 创建环境
    print("\n1. 创建环境（from_pixels=True）:")
    env = gym.make("NewPasture-v5", render_mode='rgb_array')
    wrapped_env = GymWrapper(env, device="cpu", from_pixels=True, pixels_only=False)
    print(f"   环境创建成功")
    
    # 2. 创建真实的SAC actor
    print("\n2. 创建真实的SAC actor:")
    actor_critic = make_sac_models(wrapped_env, device="cpu")
    actor = actor_critic[0]  # 获取actor部分
    print(f"   Actor类型: {type(actor).__name__}")
    
    # 3. Reset环境获取初始tensordict
    print("\n3. Reset环境并测试actor行为:")
    td = wrapped_env.reset()
    print(f"   Reset后的键: {list(td.keys())}")
    print(f"   包含pixels: {'pixels' in td}")
    if 'pixels' in td:
        print(f"   pixels shape: {td['pixels'].shape}")
    
    # 4. 测试actor对tensordict的处理
    print("\n4. 通过actor处理tensordict:")
    print(f"   输入td的键: {list(td.keys())}")
    
    # 调用actor
    output_td = actor(td)
    
    print(f"   输出td的键: {list(output_td.keys())}")
    print(f"   输出包含pixels: {'pixels' in output_td}")
    print(f"   输出包含action: {'action' in output_td}")
    
    # 5. 验证是否保留了所有原始键
    print("\n5. 验证键的保留情况:")
    original_keys = set(td.keys())
    output_keys = set(output_td.keys())
    
    preserved_keys = original_keys.intersection(output_keys)
    lost_keys = original_keys - output_keys
    new_keys = output_keys - original_keys
    
    print(f"   保留的键: {preserved_keys}")
    print(f"   丢失的键: {lost_keys}")
    print(f"   新增的键: {new_keys}")
    
    # 6. 测试step操作
    print("\n6. 测试step操作:")
    if 'action' in output_td:
        next_td = wrapped_env.step(output_td)
        print(f"   Step后的键: {list(next_td.keys())}")
        print(f"   Step后包含pixels: {'pixels' in next_td}")
        if 'next' in next_td:
            print(f"   Next子字典的键: {list(next_td['next'].keys())}")
            print(f"   Next中包含pixels: {'pixels' in next_td['next']}")
    
    return output_td, td


def test_evaluation_flow_simulation():
    """精确模拟评估代码流程，找出pixels丢失的位置"""
    print("\n" + "="*80)
    print("精确模拟评估流程，定位pixels丢失位置")
    print("="*80)
    
    # 创建配置（模拟实际配置）
    cfg = OmegaConf.create({
        'seed': 42,
        'env': {
            'env_id': 'NewPasture-v5',
            'env_kwargs': {}
        },
        'collector': {
            'env_per_collector': 1
        },
        'logger': {
            'eval_video': True,
            'eval_episodes': 2,
            'eval_max_steps': 100,
            'eval_video_skip': 1
        }
    })
    
    # 1. 创建环境（模拟评估环境创建）
    print("\n1. 创建评估环境:")
    envs = []
    for i in range(2):
        env = make_single_environment(cfg, device="cpu", seed=cfg.seed + i, from_pixels=True)
        envs.append(env)
    print(f"   创建了{len(envs)}个环境")
    
    # 2. 创建actor
    print("\n2. 创建actor:")
    dummy_env = make_single_environment(cfg, device="cpu", from_pixels=False)
    actor_critic = make_sac_models(dummy_env, device="cpu")
    actor = actor_critic[0]
    dummy_env.close()
    print(f"   Actor创建成功")
    
    # 3. Reset所有环境
    print("\n3. Reset所有环境:")
    tds = []
    for i, env in enumerate(envs):
        td = env.reset()
        tds.append(td)
        print(f"   环境{i} - 键: {list(td.keys())}, 包含pixels: {'pixels' in td}")
    
    # 4. 模拟评估循环的第一步
    print("\n4. 模拟评估循环:")
    
    # 收集active tensordict
    active_tds = [tds[0], tds[1]]
    active_indices = [0, 1]
    
    # 批处理
    print("\n   4.1 批处理tensordict:")
    batch_td = torch.stack(active_tds)
    print(f"       批处理后的键: {list(batch_td.keys())}")
    print(f"       批处理包含pixels: {'pixels' in batch_td}")
    
    # 模拟GPU推理（第409-411行）
    print("\n   4.2 Actor推理:")
    print(f"       推理前batch_td的键: {list(batch_td.keys())}")
    
    # 这是实际代码第411行的操作
    batch_td = actor(batch_td)  # 注意：这里覆盖了batch_td！
    
    print(f"       推理后batch_td的键: {list(batch_td.keys())}")
    print(f"       推理后包含pixels: {'pixels' in batch_td}")
    print(f"       推理后包含action: {'action' in batch_td}")
    
    # 模拟unbind和step（第414-417行）
    print("\n   4.3 Unbind和step:")
    for i, td in enumerate(batch_td.unbind(0)):
        print(f"\n       环境{i}:")
        print(f"         Unbind后td的键: {list(td.keys())}")
        print(f"         包含pixels: {'pixels' in td}")
        
        # 执行step
        next_td = envs[i].step(td)
        tds[i] = next_td
        
        print(f"         Step后的键: {list(next_td.keys())}")
        print(f"         Step后包含pixels: {'pixels' in next_td}")
    
    # 5. 检查视频录制时的pixels
    print("\n5. 视频录制检查:")
    for idx in range(2):
        has_pixels = "pixels" in tds[idx]
        print(f"   tds[{idx}]包含pixels: {has_pixels}")
        if not has_pixels:
            print(f"   ⚠️ 这就是视频录制失败的原因!")
    
    return tds


def test_actor_preserves_keys():
    """详细测试actor是否真的保留所有键"""
    print("\n" + "="*80)
    print("详细测试Actor是否保留所有键")
    print("="*80)
    
    # 创建环境和actor
    env = gym.make("NewPasture-v5", render_mode='rgb_array')
    wrapped_env = GymWrapper(env, device="cpu", from_pixels=True, pixels_only=False)
    actor_critic = make_sac_models(wrapped_env, device="cpu")
    actor = actor_critic[0]
    
    # 获取初始tensordict
    td = wrapped_env.reset()
    
    print("\n测试1: Actor是否in-place修改:")
    original_id = id(td)
    output = actor(td)
    output_id = id(output)
    
    print(f"  输入td的id: {original_id}")
    print(f"  输出td的id: {output_id}")
    print(f"  是同一个对象: {original_id == output_id}")
    
    print("\n测试2: 使用clone()后的行为:")
    td_clone = td.clone()
    print(f"  Clone前的键: {list(td_clone.keys())}")
    output_clone = actor(td_clone)
    print(f"  Clone后actor输出的键: {list(output_clone.keys())}")
    print(f"  是同一个对象: {id(td_clone) == id(output_clone)}")
    
    print("\n测试3: 检查TorchRL的Actor设计:")
    print(f"  Actor类型: {type(actor).__name__}")
    print(f"  Actor的模块: {actor.__class__.__module__}")
    
    # 检查actor的行为模式
    if hasattr(actor, 'forward'):
        import inspect
        print(f"  Forward方法签名: {inspect.signature(actor.forward)}")
    
    return td, output


if __name__ == "__main__":
    print("开始深入测试SAC Actor行为...")
    print("="*80)
    
    # 测试1: 真实actor的行为
    output_td, input_td = test_real_actor_behavior()
    
    # 测试2: 模拟评估流程
    tds = test_evaluation_flow_simulation()
    
    # 测试3: 详细测试actor行为
    test_actor_preserves_keys()
    
    print("\n" + "="*80)
    print("测试完成 - 总结")
    print("="*80)
    print("\n关键发现：")
    print("1. SAC Actor是否保留所有键？")
    print("2. 评估代码第411行是否覆盖了batch_td？")
    print("3. pixels在哪一步丢失？")