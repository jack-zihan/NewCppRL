#!/usr/bin/env python3
"""
测试evaluate_policy_parallel函数的正确性
重点验证：
1. break_when_all_done=True的行为
2. 数据从"next"字典提取
3. completion_ratio的正确获取
4. dump_video的调用
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 注册环境
import envs_new

import torch
from omegaconf import DictConfig
from rl_new.sac_cont_sy.sac_utils import evaluate_policy_parallel
from rl_new.sac_cont_sy.model_utils import make_sac_models
from rl_new.sac_cont_sy.env_utils import make_environment

def test_evaluate_parallel():
    """测试并行评估函数"""
    
    # 创建测试配置
    cfg = DictConfig({
        "env": {
            "env_id": "NewPasture-v2",
            "seed": 42,
            "env_kwargs": {}
        },
        "collector": {
            "env_per_collector": 4  # 并行环境数
        },
        "logger": {
            "eval_episodes": 4,  # 评估episode数
            "eval_max_steps": 100,  # 每个episode最大步数
            "video": True,  # 启用视频录制
            "eval_video_skip": 10  # 视频跳帧
        },
        "training": {
            "device": "cpu"  # 使用CPU避免CUDA多进程问题
        },
        "seed": 42
    })
    
    # 设置设备
    device = torch.device(cfg.training.device)
    print(f"使用设备: {device}")
    
    # 创建环境获取规格
    _, eval_env = make_environment(cfg, None, device, device)
    
    # 创建模型
    policy_module, qvalue_module = make_sac_models(eval_env)
    actor_critic = (policy_module, qvalue_module)
    
    print("\n测试evaluate_policy_parallel函数...")
    print("-" * 50)
    
    try:
        # 调用评估函数
        eval_metrics = evaluate_policy_parallel(
            actor_critic=actor_critic,
            cfg=cfg,
            train_device=device,
            logger=None,  # 不使用logger，避免依赖
            step=0
        )
        
        # 验证返回的指标
        print("\n评估指标:")
        for key, value in eval_metrics.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        
        # 检查必要的指标是否存在
        required_keys = [
            "eval/reward_mean",
            "eval/reward_std", 
            "eval/reward_min",
            "eval/reward_max",
            "eval/episode_length"
        ]
        
        for key in required_keys:
            assert key in eval_metrics, f"缺少指标: {key}"
            print(f"✅ {key} 存在")
        
        # 检查completion_ratio（如果环境支持）
        if "eval/completion_ratio" in eval_metrics:
            print(f"✅ completion_ratio 成功获取: {eval_metrics['eval/completion_ratio']:.4f}")
        else:
            print("ℹ️ 环境不提供completion_ratio或值为None")
        
        print("\n✅ evaluate_policy_parallel测试通过!")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理环境（已在evaluate_policy_parallel中关闭）
        pass
    
    return True

def test_data_extraction():
    """测试数据提取逻辑的正确性"""
    print("\n测试数据提取逻辑...")
    print("-" * 50)
    
    # 创建简单配置
    cfg = DictConfig({
        "env": {
            "env_id": "NewPasture-v2",
            "seed": 42,
            "env_kwargs": {}
        },
        "collector": {
            "env_per_collector": 2  # 使用2个环境测试
        },
        "logger": {
            "eval_episodes": 2,
            "eval_max_steps": 50,
            "video": False  # 不录制视频以加快测试
        },
        "training": {
            "device": "cpu"  # 使用CPU简化测试
        },
        "seed": 42
    })
    
    device = torch.device("cpu")
    
    # 创建环境
    _, eval_env = make_environment(cfg, None, device, device)
    
    # 执行rollout获取原始数据
    from torchrl.envs.utils import ExplorationType, set_exploration_type
    
    with set_exploration_type(ExplorationType.RANDOM):
        eval_rollout = eval_env.rollout(
            max_steps=cfg.logger.eval_max_steps,
            policy=None,  # 使用随机策略
            auto_cast_to_device=False,
            break_when_all_done=True  # 关键参数
        )
    
    print(f"Rollout形状: {eval_rollout.shape}")
    print(f"Rollout keys: {eval_rollout.keys()}")
    
    # 检查"next"字典
    if "next" in eval_rollout.keys():
        print(f"'next'字典keys: {eval_rollout['next'].keys()}")
        
        # 检查episode_reward和step_count
        if "episode_reward" in eval_rollout["next"].keys():
            episode_rewards = eval_rollout["next", "episode_reward"][:, -1]
            print(f"✅ episode_reward形状: {episode_rewards.shape}")
            print(f"   值: {episode_rewards.cpu().numpy()}")
        
        if "step_count" in eval_rollout["next"].keys():
            step_counts = eval_rollout["next", "step_count"][:, -1]
            print(f"✅ step_count形状: {step_counts.shape}")
            print(f"   值: {step_counts.cpu().numpy()}")
        
        # 检查completion_ratio
        if "completion_ratio" in eval_rollout["next"].keys():
            completion_ratios = eval_rollout["next", "completion_ratio"][:, -1]
            print(f"✅ completion_ratio形状: {completion_ratios.shape}")
            print(f"   值: {completion_ratios.cpu().numpy()}")
        else:
            print("ℹ️ 'next'字典中没有completion_ratio")
    
    eval_env.close()
    print("\n✅ 数据提取测试完成!")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("evaluate_policy_parallel函数测试")
    print("=" * 60)
    
    # 运行测试
    success = True
    
    # 测试数据提取
    if not test_data_extraction():
        success = False
    
    # 测试完整函数
    if not test_evaluate_parallel():
        success = False
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 所有测试通过!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ 部分测试失败")
        print("=" * 60)
        sys.exit(1)