#!/usr/bin/env python3
"""
诊断为什么agent不移动的问题
检查随机动作的值和环境响应
"""
import sys
sys.path.append('/home/lzh/NewCppRL')

import torch
from omegaconf import OmegaConf
import numpy as np

# 导入环境创建函数
from rl_new.sac_cont_sy.env_utils import make_single_environment

def diagnose_action_issue():
    """诊断动作和环境响应问题"""
    print("\n" + "="*80)
    print("动作诊断：检查随机动作和环境响应")
    print("="*80)
    
    # 1. 加载配置
    print("\n1. 加载配置:")
    config = OmegaConf.load('/home/lzh/NewCppRL/rl_new/sac_cont_sy/config-async-server.yaml')
    config.logger.eval_episodes = 1  # 只测试一个环境
    
    # 2. 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"   使用设备: {device}")
    
    # 3. 创建环境
    print("\n2. 创建测试环境:")
    env = make_single_environment(
        cfg=config,
        from_pixels=True,
        device=device,
        seed=42
    )
    print(f"   ✅ 环境创建成功")
    
    # 4. 检查action_spec
    print("\n3. 检查action_spec:")
    action_spec = env.action_spec
    print(f"   Action spec shape: {action_spec.shape}")
    print(f"   Action spec dtype: {action_spec.dtype}")
    print(f"   Action spec device: {action_spec.device}")
    
    # 打印action_spec的详细信息
    if hasattr(action_spec, 'space'):
        space = action_spec.space
        if hasattr(space, 'minimum') and hasattr(space, 'maximum'):
            print(f"   Action range: [{space.minimum}, {space.maximum}]")
    
    # 5. Reset环境并获取初始状态
    print("\n4. Reset环境:")
    td = env.reset()
    print(f"   初始位置: {td.get('position', 'N/A')}")
    print(f"   初始方向: {td.get('direction', 'N/A')}")
    
    # 获取初始pixels
    initial_pixels = td.get("pixels")
    if initial_pixels is not None:
        print(f"   初始pixels shape: {initial_pixels.shape}")
        print(f"   初始pixels范围: [{initial_pixels.min()}, {initial_pixels.max()}]")
    
    # 6. 测试随机动作
    print("\n5. 测试10个随机动作:")
    print("-" * 60)
    
    for i in range(10):
        # 生成随机动作
        action = env.action_spec.rand()
        print(f"\n步骤 {i+1}:")
        print(f"   生成的动作: {action}")
        
        # 检查动作的具体值
        if hasattr(action, 'cpu'):
            action_values = action.cpu().numpy()
            print(f"   动作值: {action_values}")
            print(f"   是否全为0: {np.allclose(action_values, 0)}")
        
        # 将动作应用到环境
        td["action"] = action
        next_td = env.step(td)
        
        # 检查环境响应
        print(f"   Step后位置: {next_td.get('next', {}).get('position', 'N/A')}")
        print(f"   Step后方向: {next_td.get('next', {}).get('direction', 'N/A')}")
        
        # 检查奖励
        reward = next_td.get("next", {}).get("reward", None)
        if reward is not None:
            if hasattr(reward, 'item'):
                reward = reward.item()
            print(f"   奖励: {reward}")
        
        # 检查pixels是否变化
        current_pixels = next_td.get("next", {}).get("pixels", None)
        if current_pixels is not None and initial_pixels is not None:
            pixel_diff = torch.abs(current_pixels - initial_pixels).sum()
            print(f"   Pixels变化量: {pixel_diff.item()}")
            if pixel_diff > 0:
                print(f"   ✅ Pixels有变化!")
            else:
                print(f"   ❌ Pixels没有变化!")
        
        # 更新td为下一个状态
        td = next_td["next"]
        
        # 检查是否done
        done = td.get("done", False)
        if hasattr(done, 'item'):
            done = done.item()
        if done:
            print(f"   环境结束!")
            break
    
    # 7. 测试特定的非零动作
    print("\n6. 测试特定的非零动作:")
    print("-" * 60)
    
    # 重置环境
    td = env.reset()
    print(f"   重置后位置: {td.get('position', 'N/A')}")
    
    # 创建明确的非零动作
    # 假设动作是[linear_velocity, angular_velocity]
    test_actions = [
        torch.tensor([1.0, 0.0], device=device),  # 前进
        torch.tensor([0.0, 10.0], device=device),  # 旋转
        torch.tensor([2.0, 5.0], device=device),   # 前进+旋转
    ]
    
    for i, action in enumerate(test_actions):
        print(f"\n测试动作 {i+1}: {action}")
        td["action"] = action
        next_td = env.step(td)
        
        print(f"   Step后位置: {next_td.get('next', {}).get('position', 'N/A')}")
        print(f"   Step后方向: {next_td.get('next', {}).get('direction', 'N/A')}")
        
        td = next_td["next"]
    
    # 8. 检查环境的action space
    print("\n7. 详细检查环境的action space:")
    if hasattr(env, 'action_space'):
        action_space = env.action_space
        print(f"   Action space类型: {type(action_space)}")
        if hasattr(action_space, 'shape'):
            print(f"   Action space shape: {action_space.shape}")
        if hasattr(action_space, 'low') and hasattr(action_space, 'high'):
            print(f"   Action space范围: [{action_space.low}, {action_space.high}]")
    
    # 清理
    env.close()
    
    print("\n" + "="*80)
    print("诊断完成!")
    print("="*80)


if __name__ == "__main__":
    diagnose_action_issue()