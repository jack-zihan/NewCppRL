"""
快速测试V5评估脚本
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.sac_cont.area_coverage_v5_sac_cont_eval import AreaCoverageV5SacEvaluator

# 创建评估器并运行一小段
evaluator = AreaCoverageV5SacEvaluator(
    episodes=1,
    max_frames=1,
    max_step=5,  # 只运行5步
    skip_frames=1,
    video=False,
    device='cuda',
    start_idx=0,
    ckpt_path='/home/lzh/NewCppRL/ckpt/area_coverage_v5_sac_cont/2025-08-11_05-56-26_area_coverage_v5_with_direction_field',
)

print("开始运行修复后的V5评估脚本...")
try:
    # 直接调用eval_actor而不是run，避免等待新模型
    import gymnasium as gym
    import torch
    
    # 创建环境
    envs = []
    for _ in range(evaluator.episodes):
        env = gym.make(render_mode=None, **evaluator.env_cfg.env.params)
        envs.append(env)
    
    # 加载模型
    pt_path = '/home/lzh/NewCppRL/ckpt/area_coverage_v5_sac_cont/2025-08-11_05-56-26_area_coverage_v5_with_direction_field/t[00042].pt'
    actor = evaluator.get_actor(pt_path)
    
    # 评估
    rewards_mean, rewards_min, rewards_max = evaluator.eval_actor(
        envs, actor, None, 42000, None
    )
    
    print(f"✅ 评估成功完成!")
    print(f"   平均奖励: {rewards_mean:.4f}")
    print(f"   最小奖励: {rewards_min:.4f}")
    print(f"   最大奖励: {rewards_max:.4f}")
    
except Exception as e:
    print(f"✗ 评估失败: {e}")
    import traceback
    traceback.print_exc()