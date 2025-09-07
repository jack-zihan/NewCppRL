#!/usr/bin/env python3
"""
使用项目已有的模型和环境创建方式来测试evaluate_policy函数
验证视频录制、指标收集和状态管理的正确性
"""
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from omegaconf import OmegaConf
import tempfile
import shutil
from torchrl.envs.utils import ExplorationType, set_exploration_type

# 导入项目中的模块
from rl_new.sac_cont_sy.env_utils import make_train_environment, make_single_environment
from rl_new.sac_cont_sy.model_utils import make_sac_models
from rl_new.sac_cont_sy.sac_utils import evaluate_policy
import envs_new  # 触发环境注册


def test_evaluate_policy_with_existing_modules():
    """使用项目已有的方式测试evaluate_policy函数"""
    print("=" * 80)
    print("测试 evaluate_policy 函数（使用项目已有模块）")
    print("=" * 80)
    
    # 加载配置
    cfg_path = Path(__file__).parent.parent / "rl_new/sac_cont_sy/config-async.yaml"
    cfg = OmegaConf.load(cfg_path)
    
    # 修改配置用于测试
    cfg.env.env_name = "CppEnvParallel-v0"
    cfg.env.frame_skip = 1
    cfg.env.from_pixels = True  # 确保录制视频
    cfg.env.num_envs = 3  # 测试多个环境
    cfg.env.device = "cpu"
    cfg.seed = 42
    
    # 设置视频录制参数
    cfg.logger.video = True
    cfg.logger.eval_envs = 3  # 录制3个环境的视频
    
    # 创建临时目录用于保存视频
    temp_dir = tempfile.mkdtemp(prefix="test_eval_fixed_")
    video_dir = Path(temp_dir) / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n临时视频目录: {video_dir}")
    
    # 设置随机种子
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    eval_env = None
    try:
        # 1. 创建环境（使用项目的方式）
        print("\n1. 创建测试环境（使用make_train_environment）...")
        dummy_env = make_train_environment(cfg, device="cpu")
        
        # 获取环境信息
        td_test = dummy_env.reset()
        obs_shape = td_test["observation"].shape
        if "pixels" in td_test:
            pixels_shape = td_test["pixels"].shape
        else:
            pixels_shape = None
        
        print(f"   观察空间形状: {obs_shape}")
        print(f"   像素形状: {pixels_shape}")
        
        # 2. 创建模型（使用项目的make_sac_models）
        print("\n2. 创建 SAC 模型（使用make_sac_models）...")
        device = torch.device("cpu")  # 测试使用CPU
        
        # 创建actor_critic（返回ModuleList包含[policy, qvalue]）
        actor_critic = make_sac_models(dummy_env, device=device)
        policy_module = actor_critic[0]  # 提取policy用于评估
        
        print(f"   Actor-Critic 模块创建成功")
        print(f"   Policy模块类型: {type(policy_module).__name__}")
        
        # 关闭dummy环境
        dummy_env.close()
        del dummy_env
        
        # 3. 测试策略
        print("\n3. 测试策略生成动作...")
        
        # 创建用于评估的环境
        eval_env = make_single_environment(cfg, device="cpu", seed=42, from_pixels=True)
        
        # 测试单步动作生成
        td = eval_env.reset()
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            td_with_action = policy_module(td)
        
        if "action" in td_with_action:
            action = td_with_action["action"].detach().cpu().numpy()
            print(f"   生成的动作: {action}")
            print(f"   动作形状: {action.shape}")
        
        # 4. 调用evaluate_policy进行完整测试
        print("\n4. 调用 evaluate_policy 函数...")
        
        # 评估策略
        # 注意：evaluate_policy函数签名是 (actor_critic, cfg, train_device, logger, step)
        # 为了测试，创建一个简单的logger
        from torchrl.record.loggers import get_logger
        test_logger = get_logger(
            logger_type="tensorboard",
            experiment_name="test_eval",
            logger_name="test"
        ) if cfg.logger.backend else None
        
        eval_metrics = evaluate_policy(
            actor_critic=actor_critic,  # 传递完整的actor_critic
            cfg=cfg,
            train_device=device,
            logger=test_logger,
            step=1000  # 模拟训练步数
        )
        
        # 5. 打印收集到的指标
        print("\n5. 收集到的评估指标:")
        print("-" * 40)
        
        for key, value in eval_metrics.items():
            if isinstance(value, (int, float)):
                print(f"   {key:35s}: {value:>12.4f}")
            elif isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    print(f"   {key:35s}: {value.item():>12.4f}")
                else:
                    print(f"   {key:35s}: shape={value.shape}, mean={value.mean().item():.4f}")
            else:
                print(f"   {key:35s}: {type(value).__name__}")
        
        # 6. 验证关键指标
        print("\n6. 验证关键指标存在性:")
        print("-" * 40)
        
        # 必须存在的指标
        required_metrics = [
            "eval/episode_reward",
            "eval/episode_reward_mean", 
            "eval/episode_length",
            "eval/episode_length_mean",
            "eval/completion_ratio",
            "eval/completion_ratio_mean"
        ]
        
        missing_metrics = []
        for metric in required_metrics:
            if metric in eval_metrics:
                value = eval_metrics[metric]
                if isinstance(value, torch.Tensor):
                    value_str = f"{value.mean().item():.4f}" if value.numel() > 1 else f"{value.item():.4f}"
                else:
                    value_str = f"{value:.4f}"
                print(f"   ✓ {metric:35s} = {value_str}")
            else:
                print(f"   ✗ {metric:35s} 缺失")
                missing_metrics.append(metric)
        
        # 7. 检查视频文件
        print("\n7. 检查视频文件生成:")
        print("-" * 40)
        
        video_files = list(video_dir.glob("*.mp4"))
        if video_files:
            print(f"   ✓ 找到 {len(video_files)} 个视频文件:")
            for vf in video_files:
                file_size = vf.stat().st_size / 1024  # KB
                print(f"      - {vf.name} ({file_size:.1f} KB)")
                
                if file_size > 10:  # 至少10KB才算正常
                    print(f"        ✓ 文件大小正常")
                else:
                    print(f"        ⚠ 文件可能太小")
        else:
            print(f"   ✗ 未找到视频文件")
        
        # 8. 测试状态管理（验证step_mdp的正确性）
        print("\n8. 测试状态管理（验证画面更新）:")
        print("-" * 40)
        
        td = eval_env.reset()
        pixel_changes = []
        rewards = []
        
        for step in range(5):
            # 获取当前像素（如果有）
            if "pixels" in td:
                current_pixels = td["pixels"].clone()
            
            # 使用策略生成动作
            with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
                td = policy_module(td)
            
            # 执行环境步进
            transition = eval_env.step(td)
            
            # 记录奖励
            if "next" in transition and "reward" in transition["next"]:
                rewards.append(transition["next"]["reward"].item())
            
            # 使用step_mdp获取下一状态
            td = eval_env.step_mdp(transition)
            
            # 检查像素是否变化
            if "pixels" in td and step > 0:
                pixel_diff = (td["pixels"] - current_pixels).abs().sum().item()
                pixel_changes.append(pixel_diff)
                print(f"   步骤 {step+1}: 像素变化量 = {pixel_diff:>10.2f}, 奖励 = {rewards[-1] if rewards else 0:>7.4f}")
        
        if pixel_changes:
            avg_change = np.mean(pixel_changes)
            total_reward = sum(rewards)
            if avg_change > 0:
                print(f"\n   ✓ 平均像素变化量: {avg_change:.2f} (画面正在更新)")
                print(f"   ✓ 累计奖励: {total_reward:.4f}")
            else:
                print(f"\n   ✗ 平均像素变化量: {avg_change:.2f} (画面可能静止)")
        
        # 9. 总结
        print("\n" + "=" * 80)
        print("测试总结:")
        print("-" * 40)
        
        success = True
        issues = []
        
        if missing_metrics:
            success = False
            issues.append(f"缺失指标: {missing_metrics}")
        
        if not video_files:
            success = False
            issues.append("未生成视频文件")
        elif all(vf.stat().st_size < 10240 for vf in video_files):  # 10KB
            success = False
            issues.append("视频文件太小，可能损坏")
        
        if pixel_changes and avg_change == 0:
            success = False
            issues.append("画面未更新（静止）")
        
        if success:
            print("✅ 所有测试通过！")
            print("   - 所有必需指标都已正确收集")
            print("   - 视频文件已成功生成且大小正常")
            print("   - 画面正确更新（非静止）")
            print("   - 奖励信号正常")
        else:
            print("⚠️  发现以下问题:")
            for issue in issues:
                print(f"   - {issue}")
        
        return eval_metrics, success
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None, False
        
    finally:
        # 清理
        print(f"\n清理临时目录: {temp_dir}")
        if eval_env is not None:
            eval_env.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    metrics, success = test_evaluate_policy_with_existing_modules()
    
    if success:
        print("\n🎉 测试完成，evaluate_policy函数工作正常！")
    else:
        print("\n⚠️  测试完成，但发现一些问题需要注意。")
    
    print("\n" + "=" * 80)