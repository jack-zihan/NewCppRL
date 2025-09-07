#!/usr/bin/env python3
"""
测试pixels在TensorDict中的位置
理解step后的数据结构
"""
import sys
sys.path.append('/home/lzh/NewCppRL')

import torch
from omegaconf import OmegaConf

# 导入环境创建函数
from rl_new.sac_cont_sy.env_utils import make_single_environment

def test_pixels_location():
    """测试pixels在tensordict中的位置"""
    print("\n" + "="*80)
    print("测试Pixels在TensorDict中的位置")
    print("="*80)
    
    # 1. 加载配置
    config = OmegaConf.load('/home/lzh/NewCppRL/rl_new/sac_cont_sy/config-async-server.yaml')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 2. 创建环境
    print("\n1. 创建环境并reset:")
    env = make_single_environment(
        cfg=config,
        from_pixels=True,
        device=device,
        seed=42
    )
    
    # 3. Reset环境
    td = env.reset()
    print(f"   Reset后的TensorDict keys: {list(td.keys())}")
    print(f"   'pixels' in td: {'pixels' in td}")
    if "pixels" in td:
        print(f"   td['pixels'].shape: {td['pixels'].shape}")
        initial_pixels = td["pixels"].clone()
    
    # 4. 执行一步
    print("\n2. 执行一步动作:")
    action = env.action_spec.rand()
    print(f"   动作: {action}")
    td["action"] = action
    
    next_td = env.step(td)
    print(f"\n   Step后的next_td keys: {list(next_td.keys())}")
    
    # 检查next_td的结构
    if "next" in next_td:
        print(f"   'next' in next_td: True")
        print(f"   next_td['next'] keys: {list(next_td['next'].keys())}")
        
        # 检查pixels在哪里
        print(f"\n3. 查找pixels:")
        print(f"   'pixels' in next_td: {'pixels' in next_td}")
        print(f"   'pixels' in next_td['next']: {'pixels' in next_td['next']}")
        
        if "pixels" in next_td:
            print(f"   next_td['pixels'].shape: {next_td['pixels'].shape}")
            print(f"   这是旧的pixels（与initial_pixels相同）: {torch.equal(next_td['pixels'], initial_pixels)}")
        
        if "pixels" in next_td["next"]:
            print(f"   next_td['next']['pixels'].shape: {next_td['next']['pixels'].shape}")
            new_pixels = next_td["next"]["pixels"]
            print(f"   这是新的pixels（与initial_pixels不同）: {not torch.equal(new_pixels, initial_pixels)}")
            diff = torch.abs(new_pixels - initial_pixels).sum()
            print(f"   像素变化量: {diff.item()}")
    
    # 5. 模拟evaluate_policy的做法
    print("\n4. 模拟evaluate_policy的更新方式:")
    print("   执行: tds[idx] = next_td")
    tds = [next_td]  # 模拟列表
    
    print(f"   现在tds[0] keys: {list(tds[0].keys())}")
    print(f"   'pixels' in tds[0]: {'pixels' in tds[0]}")
    print(f"   'pixels' in tds[0]['next']: {'pixels' in tds[0]['next']}")
    
    if "pixels" in tds[0]:
        print(f"   tds[0]['pixels']是旧的: {torch.equal(tds[0]['pixels'], initial_pixels)}")
    if "pixels" in tds[0]["next"]:
        print(f"   tds[0]['next']['pixels']是新的: {not torch.equal(tds[0]['next']['pixels'], initial_pixels)}")
    
    print("\n" + "="*80)
    print("结论：")
    print("1. reset后，pixels在td['pixels']中")
    print("2. step后，旧pixels在next_td['pixels']，新pixels在next_td['next']['pixels']")
    print("3. 如果tds[idx] = next_td，要获取最新pixels应该用tds[idx]['next']['pixels']")
    print("4. 但第一帧（reset后）pixels在tds[idx]['pixels']")
    print("="*80)
    
    env.close()


if __name__ == "__main__":
    test_pixels_location()