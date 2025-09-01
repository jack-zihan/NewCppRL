#!/usr/bin/env python3
"""
测试v2、v4、v5环境的兼容性
检查观察空间、动作空间是否正确
"""

import torch
import numpy as np
from omegaconf import DictConfig

# 导入环境创建函数
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_new.sac_cont_sy.env_utils import make_train_environment, make_single_environment


def test_environment_compatibility():
    """测试三个环境的兼容性"""
    
    env_configs = {
        "v2": {
            "env_id": "NewPasture-v2",
            "description": "APF增强环境",
            "expected_channels": 4  # field, weed, obstacle, apf 
        },
        "v4": {
            "env_id": "NewPasture-v4", 
            "description": "田地覆盖环境（无杂草）",
            "expected_channels": 2  # field, obstacle (无weed)
        },
        "v5": {
            "env_id": "NewPasture-v5",
            "description": "HIF方向引导环境",
            "expected_channels": 3  # field, obstacle, hif (无weed)
        }
    }
    
    print("=" * 80)
    print("环境兼容性测试")
    print("=" * 80)
    
    for version, env_cfg in env_configs.items():
        print(f"\n测试 {version} 环境: {env_cfg['description']}")
        print("-" * 60)
        
        # 创建配置
        cfg = DictConfig({
            "env": {
                "env_id": env_cfg["env_id"],
                "seed": 42,
                "env_kwargs": {}
            },
            "collector": {
                "env_per_collector": 1
            }
        })
        
        try:
            # 创建单个环境进行测试
            env = make_single_environment(
                cfg,
                device="cpu",
                seed=42,
                from_pixels=False
            )
            
            # 重置环境
            td = env.reset()
            
            # 检查观察空间
            obs = td["observation"]
            print(f"  ✅ 观察空间形状: {obs.shape}")
            print(f"  ✅ 观察通道数: {obs.shape[1]}")
            
            # 验证通道数
            if obs.shape[1] != env_cfg["expected_channels"]:
                print(f"  ⚠️ 警告: 期望 {env_cfg['expected_channels']} 通道，实际 {obs.shape[1]} 通道")
            
            # 检查动作空间
            action_spec = env.action_spec
            print(f"  ✅ 动作空间: {action_spec}")
            
            # 检查是否为连续动作空间
            if hasattr(action_spec.space, 'low') and hasattr(action_spec.space, 'high'):
                print(f"  ✅ 连续动作空间范围: [{action_spec.space.low.tolist()}, {action_spec.space.high.tolist()}]")
            else:
                print(f"  ❌ 错误: 非连续动作空间！")
            
            # 执行一个随机动作
            random_action = torch.randn(2) * 0.1  # 小幅随机动作
            td["action"] = random_action
            next_td = env.step(td)
            
            # 检查奖励
            reward = next_td.get("next_reward", next_td.get("reward", None))
            if reward is not None:
                print(f"  ✅ 奖励值: {reward.item():.4f}")
            
            # 检查completion_ratio（如果存在）
            if "completion_ratio" in td:
                print(f"  ✅ Completion ratio: {td['completion_ratio'].item():.4f}")
            elif "completion_ratio" in next_td:
                print(f"  ✅ Completion ratio: {next_td['completion_ratio'].item():.4f}")
            
            # 检查额外的键
            extra_keys = set(td.keys()) - {"observation", "action", "done", "terminated", "truncated"}
            if extra_keys:
                print(f"  ℹ️ 额外的观察键: {extra_keys}")
            
            env.close()
            print(f"  ✅ {version} 环境测试通过")
            
        except Exception as e:
            print(f"  ❌ {version} 环境测试失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("测试模型兼容性")
    print("=" * 80)
    
    # 测试模型创建
    from rl_new.sac_cont_sy.model_utils import make_sac_models
    
    for version, env_cfg in env_configs.items():
        print(f"\n测试 {version} 环境的模型创建")
        try:
            # 创建环境
            cfg = DictConfig({
                "env": {
                    "env_id": env_cfg["env_id"],
                    "seed": 42,
                    "env_kwargs": {}
                },
                "collector": {
                    "env_per_collector": 1
                }
            })
            
            env = make_single_environment(cfg, device="cpu", seed=42)
            
            # 创建模型
            from rl_new.sac_cont_sy.model_utils import make_sac_modules
            policy_module, qvalue_module = make_sac_modules(env)
            
            # 测试前向传播
            td = env.reset()
            
            # 测试policy
            with torch.no_grad():
                policy_td = policy_module(td)
                if "action" in policy_td:
                    print(f"  ✅ Policy输出动作形状: {policy_td['action'].shape}")
                
                # 测试Q网络
                td["action"] = policy_td["action"]
                q_td = qvalue_module(td)
                if "state_action_value" in q_td:
                    print(f"  ✅ Q值输出形状: {q_td['state_action_value'].shape}")
            
            env.close()
            print(f"  ✅ {version} 模型测试通过")
            
        except Exception as e:
            print(f"  ❌ {version} 模型测试失败: {str(e)}")
    
    print("\n" + "=" * 80)
    print("兼容性测试完成")
    print("=" * 80)


if __name__ == "__main__":
    test_environment_compatibility()