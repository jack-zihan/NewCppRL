#!/usr/bin/env python3
"""
使用真实的evaluate_policy函数测试视频上传
直接调用evaluate_policy，随机初始化actor_critic
"""
import sys
sys.path.append('/home/lzh/NewCppRL')

import torch
from omegaconf import OmegaConf
from torchrl.record.loggers import get_logger

# 导入实际的环境创建和模型创建函数
from rl_new.sac_cont_sy.env_utils import make_single_environment
from rl_new.sac_cont_sy.model_utils import make_sac_models
from rl_new.sac_cont_sy.sac_utils import evaluate_policy


def test_evaluate_policy_with_video():
    """测试evaluate_policy生成和上传视频"""
    print("\n" + "="*80)
    print("测试evaluate_policy视频上传功能")
    print("="*80)
    
    # 1. 加载配置并修改
    print("\n1. 加载并修改配置:")
    config = OmegaConf.load('/home/lzh/NewCppRL/rl_new/sac_cont_sy/config-async-server.yaml')
    
    # 修改配置以便快速测试
    config.logger.eval_max_steps = 1000  # 测试1000帧
    config.logger.eval_video_skip = 10   # 每10帧记录一次，确保能录制到视频
    config.logger.eval_episodes = 4      # 测试4个环境
    config.logger.eval_video = True      # 确保录制视频
    config.logger.backend = 'wandb'      # 使用wandb
    config.logger.mode = 'online'        # 在线模式
    
    print(f"   eval_max_steps = {config.logger.eval_max_steps}")
    print(f"   eval_video_skip = {config.logger.eval_video_skip}")
    print(f"   eval_episodes = {config.logger.eval_episodes}")
    print(f"   eval_video = {config.logger.eval_video}")
    
    # 2. 创建logger
    print("\n2. 创建wandb logger:")
    exp_name = "test_evaluate_policy_real"
    logger = get_logger(
        logger_type=config.logger.backend,
        experiment_name=exp_name,
        logger_name=exp_name,
        wandb_kwargs={
            "mode": config.logger.mode,
            "config": dict(config),
            "project": config.logger.project_name,
            "group": "test_evaluation",
            "name": exp_name
        }
    )
    print(f"   ✅ Logger创建成功")
    
    # 3. 设置设备
    print("\n3. 设置设备:")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"   使用设备: {device}")
    
    # 4. 创建actor_critic模型
    print("\n4. 创建actor_critic模型:")
    # 先创建一个环境来获取规格
    test_env = make_single_environment(
        cfg=config,
        from_pixels=True,
        device=device
    )
    
    # 创建SAC模型
    actor_critic = make_sac_models(
        env=test_env,
        device=device
    )
    print(f"   ✅ Actor-critic模型创建成功")
    print("   使用随机初始化的权重")
    
    # 关闭测试环境
    test_env.close()
    
    # 5. 调用evaluate_policy
    print("\n5. 调用evaluate_policy函数:")
    print("   开始评估...")
    
    try:
        # 调用实际的evaluate_policy函数
        evaluate_policy(
            actor_critic=actor_critic,
            cfg=config,
            train_device=device,
            logger=logger,
            step=1000  # 当前训练步数（用于记录）
        )
        print("   ✅ evaluate_policy执行成功")
        
    except Exception as e:
        print(f"   ❌ evaluate_policy执行失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. 清理
    print("\n6. 清理资源:")
    try:
        logger.close()
        print("   ✅ 资源清理完成")
    except:
        pass
    
    print("\n" + "="*80)
    print("测试完成！")
    print("请检查wandb项目中是否有新的视频")
    print("项目: SAC_2025, 实验: test_evaluate_policy_real")
    print("="*80)


if __name__ == "__main__":
    test_evaluate_policy_with_video()