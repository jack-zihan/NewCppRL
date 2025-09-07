#!/usr/bin/env python3
"""
直接调用带有debug日志的evaluate_policy函数，查看实际执行情况
"""
import sys
sys.path.append('/home/lzh/NewCppRL')

import torch
from omegaconf import OmegaConf
from rl_new.sac_cont_sy.sac_utils import evaluate_policy
from rl_new.sac_cont_sy.env_utils import make_single_environment
from rl_new.sac_cont_sy.model_utils import make_sac_models
from torchrl.record.loggers import get_logger


def test_actual_eval_with_debug():
    """测试实际的评估函数并查看debug输出"""
    print("\n" + "="*80)
    print("测试实际评估函数 with DEBUG输出")
    print("="*80)
    
    # 加载配置
    cfg = OmegaConf.create({
        'seed': 42,
        'env': {
            'env_id': 'NewPasture-v5',
            'env_kwargs': {}
        },
        'collector': {
            'env_per_collector': 1
        },
        'logger': {
            'eval_episodes': 4,
            'eval_video': True,
            'eval_video_skip': 5,  # 每5步录制一帧
            'eval_max_steps': 50,   # 评估50步
            'show_progress': False,
            'wandb': {
                'project': 'test_debug_eval',
                'entity': None,
                'tags': ['debug']
            }
        }
    })
    
    # 创建logger
    print("\n1. 创建wandb logger:")
    logger = get_logger(
        logger_type="wandb",
        logger_name="test_eval_debug",
        experiment_name="debug_eval_test",
        wandb_kwargs={
            'project': 'test_debug_eval',
            'tags': ['debug', 'test'],
            'mode': 'online'  # 确保上传
        }
    )
    print(f"   Logger创建成功: {logger}")
    
    # 创建模型
    print("\n2. 创建SAC模型:")
    dummy_env = make_single_environment(cfg, device="cpu", from_pixels=False)
    actor_critic = make_sac_models(dummy_env, device="cuda" if torch.cuda.is_available() else "cpu")
    dummy_env.close()
    print(f"   模型创建成功")
    
    # 运行评估
    print("\n3. 运行evaluate_policy with DEBUG:")
    print("   查看下面的DEBUG输出来了解问题所在...")
    print("-"*80)
    
    try:
        metrics = evaluate_policy(
            actor_critic=actor_critic,
            cfg=cfg,
            train_device="cuda" if torch.cuda.is_available() else "cpu",
            logger=logger,
            step=1000
        )
        
        print("-"*80)
        print("\n4. 评估指标:")
        for key, value in metrics.items():
            print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"\n❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 关闭logger
    logger.close()
    
    print("\n" + "="*80)
    print("分析DEBUG输出:")
    print("1. 检查from_pixels是否为True")
    print("2. 检查每个环境reset后是否有pixels")
    print("3. 检查recorder.idx的变化")
    print("4. 检查dump是否返回None")
    print("5. 检查wandb上传是否成功")
    print("="*80)


if __name__ == "__main__":
    test_actual_eval_with_debug()