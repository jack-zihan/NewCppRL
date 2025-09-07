#!/usr/bin/env python3
"""
测试evaluate_policy函数的WandB视频上传功能
验证视频是否正确录制并上传到WandB，同时返回所有指标值
"""
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from omegaconf import OmegaConf
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record.loggers import get_logger
import time

# 导入项目模块
from rl_new.sac_cont_sy.env_utils import make_train_environment
from rl_new.sac_cont_sy.model_utils import make_sac_models
from rl_new.sac_cont_sy.sac_utils import evaluate_policy, generate_exp_name
import envs_new  # 触发环境注册


def test_evaluate_with_wandb():
    """测试evaluate_policy函数的WandB集成"""
    print("\n" + "=" * 80)
    print("🎥 测试 evaluate_policy 的 WandB 视频上传")
    print("=" * 80)
    
    # 加载配置
    cfg_path = Path(__file__).parent.parent / "rl_new/sac_cont_sy/config-async.yaml"
    cfg = OmegaConf.load(cfg_path)
    
    # 配置环境和日志参数
    cfg.env.env_name = "CppEnvParallel-v0"
    cfg.env.env_id = "CppEnvParallel-v0"
    cfg.env.frame_skip = 1
    cfg.env.from_pixels = True  # 必须为True才能录制视频
    cfg.env.num_envs = 4  # 评估4个环境
    cfg.env.device = "cpu"
    cfg.seed = 42
    
    # 配置日志和视频录制
    cfg.logger.backend = "wandb"
    cfg.logger.mode = "online"  # 确保上传到云端
    cfg.logger.video = True
    cfg.logger.eval_envs = 4  # 录制4个环境的视频
    cfg.logger.project_name = "test-evaluate-video"
    cfg.logger.group_name = "debug"
    cfg.logger.model_name = "sac"
    cfg.logger.exp_name = f"test_{int(time.time())}"
    
    # 设置随机种子
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    # 生成实验名称
    exp_name = generate_exp_name(cfg.logger.model_name, cfg.logger.exp_name)
    print(f"\n📝 实验名称: {exp_name}")
    
    # 创建WandB logger
    print("\n🌐 初始化WandB Logger...")
    logger = get_logger(
        logger_type="wandb",
        experiment_name=exp_name,
        logger_name=exp_name,
        wandb_kwargs={
            "mode": cfg.logger.mode,
            "config": dict(cfg),
            "project": cfg.logger.project_name,
            "group": cfg.logger.group_name,
            "name": exp_name,
            "tags": ["test", "evaluate_policy", "video"]
        }
    )
    
    # 获取WandB运行URL
    if hasattr(logger.experiment, 'url'):
        wandb_url = logger.experiment.url
        print(f"\n🔗 WandB运行地址: {wandb_url}")
        print(f"   请在浏览器中打开查看视频上传结果")
    else:
        print(f"\n⚠️  无法获取WandB URL，请检查: https://wandb.ai/{cfg.logger.project_name}")
    
    print("\n🔧 创建环境和模型...")
    
    # 创建环境用于模型初始化
    dummy_env = make_train_environment(cfg, device="cpu")
    
    # 创建SAC模型
    device = torch.device("cpu")
    actor_critic = make_sac_models(dummy_env, device=device)
    
    print(f"   ✓ 环境创建成功")
    print(f"   ✓ SAC模型创建成功")
    
    # 关闭dummy环境
    dummy_env.close()
    
    # 测试evaluate_policy函数
    print("\n📊 调用 evaluate_policy 函数...")
    print("-" * 40)
    
    # 调用评估函数（会自动创建环境、录制视频、上传到WandB）
    eval_metrics = evaluate_policy(
        actor_critic=actor_critic,
        cfg=cfg,
        train_device=device,
        logger=logger,  # 传递WandB logger
        step=1000  # 模拟训练步数
    )
    
    # 打印所有返回的指标
    print("\n📈 evaluate_policy 返回的所有指标:")
    print("=" * 60)
    
    # 按字母顺序排序打印
    sorted_metrics = sorted(eval_metrics.items())
    
    for key, value in sorted_metrics:
        # 格式化输出
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                value_str = f"{value.item():.4f}"
            else:
                # 如果是多值张量，显示形状和统计信息
                value_str = f"shape={value.shape}, mean={value.mean().item():.4f}, std={value.std().item():.4f}"
        elif isinstance(value, (int, float)):
            value_str = f"{value:.4f}"
        else:
            value_str = str(value)
        
        print(f"  {key:35s} : {value_str}")
    
    # 检查关键指标
    print("\n🔍 关键指标检查:")
    print("-" * 40)
    
    key_metrics = {
        "eval/reward_mean": "平均奖励",
        "eval/reward_std": "奖励标准差",
        "eval/episode_length": "平均episode长度",
        "eval/episodes_completed": "完成的episode数量"
    }
    
    for metric_key, description in key_metrics.items():
        if metric_key in eval_metrics:
            value = eval_metrics[metric_key]
            if isinstance(value, torch.Tensor):
                value = value.item() if value.numel() == 1 else value.mean().item()
            print(f"  ✓ {description:20s} : {value:.4f}")
        else:
            print(f"  ✗ {description:20s} : 未找到")
    
    # 检查视频是否已上传
    print("\n🎬 视频上传状态:")
    print("-" * 40)
    
    # 检查logger是否记录了视频
    if hasattr(logger, '_videos_written'):
        print(f"  ✓ 已记录 {logger._videos_written} 个视频")
    else:
        print(f"  ℹ️  视频已提交到WandB队列")
    
    # 最终总结
    print("\n" + "=" * 80)
    print("📊 测试总结:")
    print("-" * 40)
    
    if "eval/reward_mean" in eval_metrics:
        reward = eval_metrics["eval/reward_mean"]
        if isinstance(reward, torch.Tensor):
            reward = reward.item()
        print(f"  ✅ 评估成功完成，平均奖励: {reward:.2f}")
    else:
        print(f"  ⚠️  评估完成但缺少奖励指标")
    
    if hasattr(logger.experiment, 'url'):
        print(f"\n  🔗 查看完整结果（包括视频）:")
        print(f"     {wandb_url}")
    
    print("\n  💡 提示:")
    print("     1. 视频可能需要几秒钟才能在WandB界面显示")
    print("     2. 在WandB界面点击'Media'标签查看录制的视频")
    print("     3. 视频名称为'eval/video'")
    
    print("=" * 80 + "\n")
    
    # 关闭logger
    logger.close()
    
    return eval_metrics


if __name__ == "__main__":
    print("\n🚀 开始测试 evaluate_policy 的 WandB 集成...")
    
    try:
        metrics = test_evaluate_with_wandb()
        
        # 保存指标到文件（可选）
        import json
        metrics_dict = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                metrics_dict[k] = v.tolist() if v.numel() > 1 else v.item()
            else:
                metrics_dict[k] = v
        
        with open("test_evaluate_metrics.json", "w") as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"\n📄 指标已保存到 test_evaluate_metrics.json")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    print("\n✅ 测试完成！")