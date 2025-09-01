#!/usr/bin/env python3
"""
测试evaluate_policy_parallel函数的正确性
比较新旧两个评估函数的结果和性能
"""

import sys
import time
import torch
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

# 添加项目路径
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl_new.sac_cont_sy.sac_utils import evaluate_policy, evaluate_policy_parallel
from rl_new.sac_cont_sy.model_utils import make_sac_models
from rl_new.sac_cont_sy.env_utils import make_single_environment


def test_evaluate_functions():
    """测试并比较两个评估函数"""
    
    # 创建简单的配置
    cfg_dict = {
        'env': {
            'env_id': 'NewPasture-v2',
            'seed': 42,
            'env_kwargs': {}
        },
        'logger': {
            'eval_episodes': 4,
            'eval_max_steps': 1000,
            'eval_video': False,  # 关闭视频以加快测试
            'video': False
        },
        'collector': {
            'env_per_collector': 16
        },
        'seed': 42
    }
    cfg = OmegaConf.create(cfg_dict)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建环境样例获取规格
    test_env = make_single_environment(cfg, device=device)
    
    # 创建模型
    print("创建SAC模型...")
    actor_critic = make_sac_models(env=test_env)
    actor_critic[0].to(device)  # actor
    actor_critic[1].to(device)  # critic
    
    # 关闭测试环境
    test_env.close()
    
    print("\n" + "="*60)
    print("测试原版evaluate_policy...")
    print("="*60)
    
    # 测试原版函数
    start_time = time.time()
    metrics_original = evaluate_policy(
        actor_critic=actor_critic,
        cfg=cfg,
        train_device=device,
        logger=None,  # 不使用logger
        step=0
    )
    time_original = time.time() - start_time
    
    print(f"原版执行时间: {time_original:.2f}秒")
    print("原版指标:")
    for key, value in metrics_original.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n" + "="*60)
    print("测试并行版evaluate_policy_parallel...")
    print("="*60)
    
    # 测试并行版函数
    start_time = time.time()
    metrics_parallel = evaluate_policy_parallel(
        actor_critic=actor_critic,
        cfg=cfg,
        train_device=device,
        logger=None,  # 不使用logger
        step=0
    )
    time_parallel = time.time() - start_time
    
    print(f"并行版执行时间: {time_parallel:.2f}秒")
    print("并行版指标:")
    for key, value in metrics_parallel.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n" + "="*60)
    print("性能对比:")
    print("="*60)
    
    speedup = time_original / time_parallel
    print(f"速度提升: {speedup:.2f}x")
    print(f"时间节省: {time_original - time_parallel:.2f}秒")
    
    print("\n" + "="*60)
    print("结果一致性检查:")
    print("="*60)
    
    # 比较关键指标
    tolerance = 0.1  # 允许10%的差异（由于随机性）
    
    for key in ['eval/reward_mean', 'eval/episode_length']:
        if key in metrics_original and key in metrics_parallel:
            diff = abs(metrics_original[key] - metrics_parallel[key])
            rel_diff = diff / (abs(metrics_original[key]) + 1e-8)
            status = "✅" if rel_diff < tolerance else "⚠️"
            print(f"{status} {key}: 差异 {rel_diff*100:.2f}%")
    
    print("\n测试完成！")
    
    # 返回是否通过测试
    return all(
        abs(metrics_original.get(key, 0) - metrics_parallel.get(key, 0)) / (abs(metrics_original.get(key, 0)) + 1e-8) < tolerance
        for key in ['eval/reward_mean', 'eval/episode_length']
        if key in metrics_original
    )


if __name__ == "__main__":
    print("开始测试evaluate_policy_parallel函数...")
    success = test_evaluate_functions()
    
    if success:
        print("\n✅ 所有测试通过！并行版本功能正常。")
    else:
        print("\n⚠️ 部分指标差异较大，请检查实现。")
        print("注：由于环境的随机性，小幅差异是正常的。")