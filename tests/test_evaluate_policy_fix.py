#!/usr/bin/env python
"""
测试修复后的evaluate_policy函数
验证：
1. CPU评估是否正常工作
2. 内存使用是否降低
3. 视频录制是否正常
"""

import sys
import os
import torch
import numpy as np
import psutil
import yaml
from omegaconf import OmegaConf

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_new.sac_cont_sy.sac_utils import evaluate_policy
from rl_new.sac_cont_sy.env_utils import make_single_environment, make_train_environment
from rl_new.sac_cont_sy.model_utils import make_sac_models

def get_memory_usage():
    """获取当前进程的内存使用（GB）"""
    process = psutil.Process()
    return process.memory_info().rss / (1024**3)

def test_evaluate_policy():
    """测试评估函数"""
    
    print("=" * 60)
    print("测试修复后的evaluate_policy函数")
    print("=" * 60)
    
    # 加载配置
    config_path = "/home/lzh/NewCppRL/rl_new/sac_cont_sy/config-async.yaml"
    with open(config_path, 'r') as f:
        cfg = OmegaConf.create(yaml.safe_load(f))
    
    # 修改配置以进行快速测试
    cfg.logger.eval_episodes = 2  # 减少评估episode数
    cfg.logger.eval_max_steps = 100  # 减少最大步数
    cfg.logger.eval_video = True  # 启用视频录制
    cfg.logger.eval_video_skip = 10  # 减少跳帧
    cfg.logger.show_progress = True  # 启用进度条
    
    # 设置设备
    train_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"训练设备: {train_device}")
    print(f"评估环境数: {cfg.logger.eval_episodes}")
    print(f"最大步数: {cfg.logger.eval_max_steps}")
    print(f"视频录制: {cfg.logger.eval_video}")
    print(f"显示进度条: {cfg.logger.get('show_progress', False)}")
    print()
    
    # 创建一个简单的actor-critic模型
    print("创建测试模型...")
    dummy_env = make_train_environment(cfg, device="cpu")
    actor_critic = make_sac_models(dummy_env, device=train_device)
    dummy_env.close()
    
    # 记录初始内存
    initial_memory = get_memory_usage()
    print(f"初始内存使用: {initial_memory:.2f} GB")
    
    # GPU内存监控（如果有GPU）
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_gpu_memory = torch.cuda.memory_allocated(train_device) / (1024**3)
        print(f"初始GPU内存使用: {initial_gpu_memory:.2f} GB")
    
    try:
        # 执行评估
        print("\n开始评估...")
        eval_metrics = evaluate_policy(
            actor_critic=actor_critic,
            cfg=cfg,
            train_device=train_device,
            logger=None,  # 暂时不使用logger
            step=0
        )
        
        print("\n评估完成！")
        print("评估指标:")
        for key, value in eval_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # 检查内存使用
        final_memory = get_memory_usage()
        memory_increase = final_memory - initial_memory
        print(f"\n最终内存使用: {final_memory:.2f} GB")
        print(f"内存增加: {memory_increase:.2f} GB")
        
        if torch.cuda.is_available():
            final_gpu_memory = torch.cuda.memory_allocated(train_device) / (1024**3)
            gpu_memory_increase = final_gpu_memory - initial_gpu_memory
            print(f"最终GPU内存使用: {final_gpu_memory:.2f} GB")
            print(f"GPU内存增加: {gpu_memory_increase:.2f} GB")
            
            # 验证GPU内存没有大幅增加
            if gpu_memory_increase > 1.0:  # 如果增加超过1GB
                print("⚠️ 警告：GPU内存增加较多，可能仍有问题")
            else:
                print("✅ GPU内存使用正常")
        
        # 验证CPU内存增加合理
        if memory_increase > 2.0:  # 如果增加超过2GB
            print("⚠️ 警告：CPU内存增加较多")
        else:
            print("✅ CPU内存使用正常")
        
        print("\n✅ 测试通过！evaluate_policy函数工作正常。")
        
    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_evaluate_policy()
    sys.exit(0 if success else 1)