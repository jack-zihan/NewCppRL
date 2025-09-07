#!/usr/bin/env python
"""
测试TensorDict的实际结构，确认completion_ratio的位置
"""

import sys
import os
import torch
import yaml
from omegaconf import OmegaConf
from tensordict import TensorDict

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_new.sac_cont_sy.env_utils import make_single_environment

def inspect_tensordict_structure():
    """深入检查TensorDict的结构"""
    print("=" * 60)
    print("TensorDict结构深度分析")
    print("=" * 60)
    
    # 加载配置
    config_path = "/home/lzh/NewCppRL/rl_new/sac_cont_sy/config-async.yaml"
    with open(config_path, 'r') as f:
        cfg = OmegaConf.create(yaml.safe_load(f))
    
    # 创建单个环境
    print("\n创建测试环境...")
    env = make_single_environment(cfg, device="cpu", seed=42)
    
    # 重置环境
    print("\n重置环境...")
    td = env.reset()
    
    print("\n初始TensorDict的键:")
    print(f"  根级键: {list(td.keys())}")
    
    # 检查observation是否包含completion_ratio
    if "observation" in td:
        print(f"  observation类型: {type(td['observation'])}")
        if hasattr(td['observation'], 'keys'):
            print(f"  observation子键: {list(td['observation'].keys())}")
    
    if "completion_ratio" in td:
        print(f"  根级completion_ratio存在: shape={td['completion_ratio'].shape}")
        print(f"    值: {td['completion_ratio'].item():.4f}")
    
    # 执行一步
    print("\n执行一步动作...")
    action = env.action_spec.rand()
    # 将action包装成TensorDict
    td["action"] = action
    next_td = env.step(td)
    
    print("\nstep后TensorDict的键:")
    print(f"  根级键: {list(next_td.keys())}")
    
    # 详细检查每个键
    for key in next_td.keys():
        value = next_td[key]
        if isinstance(value, TensorDict):
            print(f"  '{key}' 是TensorDict，包含子键: {list(value.keys())}")
            # 检查子TensorDict中的completion_ratio
            if "completion_ratio" in value:
                print(f"    -> '{key}/completion_ratio' 存在: shape={value['completion_ratio'].shape}")
                print(f"       值: {value['completion_ratio'].item():.4f}")
        elif isinstance(value, torch.Tensor):
            print(f"  '{key}' 是Tensor: shape={value.shape}")
    
    # 检查next子字典（如果存在）
    if "next" in next_td and isinstance(next_td["next"], TensorDict):
        print("\n'next'子TensorDict的详细信息:")
        next_sub = next_td["next"]
        for key in next_sub.keys():
            value = next_sub[key]
            if isinstance(value, torch.Tensor):
                print(f"  next/{key}: shape={value.shape}, dtype={value.dtype}")
                if key == "completion_ratio":
                    print(f"    -> 值: {value.item():.4f}")
    
    # 测试不同的访问方式
    print("\n测试不同的访问路径:")
    
    # 方式1：根级completion_ratio
    try:
        val = next_td["completion_ratio"]
        print(f"✅ next_td['completion_ratio']: {val.item():.4f}")
    except (KeyError, AttributeError) as e:
        print(f"❌ next_td['completion_ratio']: {e}")
    
    # 方式2：next中的completion_ratio
    try:
        val = next_td["next"]["completion_ratio"]
        print(f"✅ next_td['next']['completion_ratio']: {val.item():.4f}")
    except (KeyError, AttributeError) as e:
        print(f"❌ next_td['next']['completion_ratio']: {e}")
    
    # 方式3：observation中的completion_ratio（如果observation是dict）
    try:
        if "observation" in next_td and hasattr(next_td["observation"], "__getitem__"):
            val = next_td["observation"]["completion_ratio"]
            print(f"✅ next_td['observation']['completion_ratio']: {val.item():.4f}")
    except (KeyError, TypeError, AttributeError) as e:
        print(f"❌ next_td['observation']['completion_ratio']: 不是dict或不存在")
    
    # 检查episode_reward（由RewardSum transform添加）
    if "episode_reward" in next_td["next"]:
        print(f"\nepisode_reward存在于next中: {next_td['next']['episode_reward'].item():.4f}")
    
    env.close()
    return True

def test_parallel_env_structure():
    """测试并行环境的TensorDict结构"""
    print("\n" + "=" * 60)
    print("并行环境TensorDict结构分析")
    print("=" * 60)
    
    # 加载配置
    config_path = "/home/lzh/NewCppRL/rl_new/sac_cont_sy/config-async.yaml"
    with open(config_path, 'r') as f:
        cfg = OmegaConf.create(yaml.safe_load(f))
    
    # 临时减少环境数量
    cfg.collector.env_per_collector = 2
    
    from rl_new.sac_cont_sy.env_utils import make_train_environment
    
    print("\n创建并行环境...")
    env = make_train_environment(cfg, device="cpu")
    
    # 重置环境
    td = env.reset()
    print(f"\n并行环境初始TensorDict:")
    print(f"  批次大小: {td.batch_size}")
    print(f"  根级键: {list(td.keys())}")
    
    # 执行一步
    action = env.action_spec.rand()
    td["action"] = action
    next_td = env.step(td)
    
    print(f"\n并行环境step后TensorDict:")
    print(f"  批次大小: {next_td.batch_size}")
    print(f"  根级键: {list(next_td.keys())}")
    
    if "next" in next_td:
        print(f"  'next'子键: {list(next_td['next'].keys())}")
    
    # 检查completion_ratio位置
    print("\n并行环境中completion_ratio的位置:")
    
    # 检查根级
    if "completion_ratio" in next_td:
        print(f"  ✅ 根级completion_ratio: shape={next_td['completion_ratio'].shape}")
    else:
        print(f"  ❌ 根级没有completion_ratio")
    
    # 检查next中
    if "next" in next_td and "completion_ratio" in next_td["next"]:
        print(f"  ✅ next/completion_ratio: shape={next_td['next']['completion_ratio'].shape}")
    else:
        print(f"  ❌ next中没有completion_ratio")
    
    env.close()
    return True

if __name__ == "__main__":
    success1 = inspect_tensordict_structure()
    success2 = test_parallel_env_structure()
    
    print("\n" + "=" * 60)
    print("测试结论")
    print("=" * 60)
    
    if success1 and success2:
        print("✅ 结构分析完成，请根据输出调整代码")
    else:
        print("❌ 分析过程中出现错误")