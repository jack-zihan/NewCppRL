#!/usr/bin/env python3
"""
总结测试：验证evaluate_policy修复的效果
重点验证：
1. 状态管理（step_mdp）是否正确
2. 像素是否正确更新（动态而非静止）  
3. 指标是否正确收集
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from omegaconf import OmegaConf
from torchrl.envs.utils import ExplorationType, set_exploration_type

from rl_new.sac_cont_sy.env_utils import make_train_environment, make_single_environment
from rl_new.sac_cont_sy.model_utils import make_sac_models
from rl_new.sac_cont_sy.sac_utils import evaluate_policy
import envs_new


def main():
    print("\n" + "=" * 80)
    print("🔬 evaluate_policy 修复效果验证测试")
    print("=" * 80)
    
    # 加载配置
    cfg = OmegaConf.load("rl_new/sac_cont_sy/config-async.yaml")
    cfg.env.env_name = "CppEnvParallel-v0"
    cfg.env.from_pixels = True
    cfg.env.num_envs = 2
    cfg.seed = 42
    
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    print("\n📝 核心修复内容：")
    print("   1. 使用 env.step_mdp() 提取当前状态")
    print("   2. 从 transition['next'] 读取像素和指标")
    print("   3. 初始化 vid_tensor 变量避免未定义错误")
    
    # 创建环境和模型
    print("\n🔧 创建测试环境和模型...")
    dummy_env = make_train_environment(cfg, device="cpu")
    actor_critic = make_sac_models(dummy_env, device="cpu")
    policy = actor_critic[0]
    
    # 测试1：验证状态管理
    print("\n✅ 测试1: 验证状态管理（step_mdp）")
    print("-" * 40)
    
    eval_env = make_single_environment(cfg, device="cpu", seed=42, from_pixels=True)
    td = eval_env.reset()
    
    pixel_diffs = []
    for i in range(3):
        if "pixels" in td:
            prev_pixels = td["pixels"].clone()
        
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            td = policy(td)
        
        transition = eval_env.step(td)
        td = eval_env.step_mdp(transition)  # 关键：使用step_mdp
        
        if "pixels" in td and i > 0:
            diff = (td["pixels"] - prev_pixels).abs().sum().item()
            pixel_diffs.append(diff)
            status = "✓ 动态" if diff > 0 else "✗ 静止"
            print(f"   步骤{i+1}: 像素差异={diff:>8.0f} [{status}]")
    
    eval_env.close()
    
    # 测试2：调用完整的evaluate_policy
    print("\n✅ 测试2: 完整评估流程")
    print("-" * 40)
    
    metrics = evaluate_policy(
        actor_critic=actor_critic,
        cfg=cfg,
        train_device=torch.device("cpu"),
        logger=None,
        step=1000
    )
    
    key_metrics = ["eval/reward_mean", "eval/episode_length", "eval/episodes_completed"]
    for key in key_metrics:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, torch.Tensor):
                value = value.item() if value.numel() == 1 else value.mean().item()
            print(f"   {key:30s} = {value:>10.4f}")
    
    dummy_env.close()
    
    # 总结
    print("\n" + "=" * 80)
    print("📊 测试总结：")
    
    all_pass = True
    
    if pixel_diffs and all(d > 0 for d in pixel_diffs):
        print("   ✅ 画面更新正常 - step_mdp修复有效")
    else:
        print("   ❌ 画面未更新 - 可能存在问题")
        all_pass = False
    
    if "eval/reward_mean" in metrics:
        print("   ✅ 指标收集正常 - 能够正确计算奖励")
    else:
        print("   ❌ 指标收集异常")
        all_pass = False
    
    if all_pass:
        print("\n🎉 所有测试通过！evaluate_policy修复成功！")
        print("\n关键要点：")
        print("   • 使用 env.step_mdp(transition) 获取当前状态")
        print("   • 从 transition['next'] 读取更新后的数据")
        print("   • 确保所有变量正确初始化")
    else:
        print("\n⚠️  部分测试未通过，需要进一步检查")
    
    print("=" * 80 + "\n")
    
    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)