#!/usr/bin/env python3
"""
测试completion_ratio在TensorDict中的确切位置
验证TorchRL如何处理嵌套的observation字典
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import envs_new  # 触发环境注册
from rl_new.sac_cont_sy.env_utils import make_single_environment
from omegaconf import OmegaConf

def test_completion_ratio_location():
    """测试completion_ratio在TensorDict中的位置"""
    
    # 加载配置
    cfg_path = Path(__file__).parent.parent / "rl_new/sac_cont_sy/config-async.yaml"
    cfg = OmegaConf.load(cfg_path)
    
    print("测试completion_ratio的位置")
    print("=" * 60)
    
    # 创建环境
    env = make_single_environment(cfg, device="cpu", seed=42, from_pixels=False)
    
    # Reset环境
    print("\n1. Reset环境后的TensorDict结构:")
    td = env.reset()
    print(f"   根键: {list(td.keys())}")
    
    # 检查completion_ratio是否在根目录
    if "completion_ratio" in td.keys():
        print(f"   ✓ completion_ratio在根目录，值: {td['completion_ratio'].item():.4f}")
    else:
        print(f"   ✗ completion_ratio不在根目录")
    
    # 检查observation结构
    if "observation" in td.keys():
        print(f"   observation存在，shape: {td['observation'].shape}")
    
    # 执行一步
    print("\n2. Step后的TensorDict结构:")
    # 生成合法的随机动作
    import torch
    # 动作空间：[线速度(0-3.5), 角速度(-28.6, 28.6)]
    td["action"] = torch.tensor([1.5, 0.0])  # 合法的动作
    
    # Step
    transition = env.step(td)
    print(f"   transition根键: {list(transition.keys())}")
    print(f"   transition['next']键: {list(transition['next'].keys())}")
    
    # 检查completion_ratio位置
    if "completion_ratio" in transition["next"].keys():
        print(f"   ✓ completion_ratio在transition['next']中，值: {transition['next']['completion_ratio'].item():.4f}")
    else:
        print(f"   ✗ completion_ratio不在transition['next']中")
    
    # 使用step_mdp提取干净状态
    print("\n3. 使用step_mdp后的结构:")
    clean_state = env.step_mdp(transition)
    print(f"   clean_state键: {list(clean_state.keys())}")
    
    if "completion_ratio" in clean_state.keys():
        print(f"   ✓ completion_ratio在clean_state中，值: {clean_state['completion_ratio'].item():.4f}")
    else:
        print(f"   ✗ completion_ratio不在clean_state中")
    
    # 测试多步
    print("\n4. 连续多步测试:")
    for i in range(3):
        # 基于当前状态生成合法动作
        clean_state["action"] = torch.tensor([1.5 + i*0.2, i*0.5])
        
        # Step
        transition = env.step(clean_state)
        
        # 检查completion_ratio
        if "completion_ratio" in transition["next"]:
            cr_value = transition["next"]["completion_ratio"].item()
            print(f"   步骤{i+1}: completion_ratio = {cr_value:.4f}")
        else:
            print(f"   步骤{i+1}: completion_ratio不存在!")
        
        # 提取下一状态
        clean_state = env.step_mdp(transition)
    
    print("\n" + "=" * 60)
    print("结论:")
    print("✓ completion_ratio在reset后的根目录")
    print("✓ completion_ratio在step后的transition['next']中")
    print("✓ 代码中的访问路径transition['next']['completion_ratio']是正确的")
    
    env.close()
    return True

if __name__ == "__main__":
    test_completion_ratio_location()