"""
完全精确地模拟sac_utils.py的评估流程
找出pixels到底在哪里丢失
"""
import sys
sys.path.append('/home/lzh/NewCppRL')

import envs_new
import torch
from rl_new.sac_cont_sy.model_utils import make_sac_models
from rl_new.sac_cont_sy.env_utils import make_single_environment
from omegaconf import OmegaConf
import numpy as np


def exact_evaluate_simulation():
    """完全按照sac_utils.py的逻辑模拟评估"""
    print("\n" + "="*80)
    print("完全精确模拟sac_utils.py的评估流程")
    print("="*80)
    
    # 配置
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
            'eval_video': True,
            'eval_episodes': 2,
            'eval_max_steps': 5,  # 少一点方便调试
            'eval_video_skip': 1
        }
    })
    
    # 设备配置（模拟服务器环境）
    train_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    # 准备
    eval_episodes = cfg.logger.eval_episodes
    eval_envs = []
    seeds = [cfg.seed + i for i in range(eval_episodes)]
    
    # 创建环境（第357-359行）
    for seed in seeds:
        env = make_single_environment(cfg, device="cpu", seed=seed, from_pixels=True)
        eval_envs.append(env)
    
    # 创建actor（模拟已训练的模型）
    dummy_env = make_single_environment(cfg, device="cpu", from_pixels=False)
    actor_critic = make_sac_models(dummy_env, device=train_device)
    dummy_env.close()
    
    print(f"设备: train_device={train_device}")
    print(f"环境数: {eval_episodes}")
    print(f"最大步数: {cfg.logger.eval_max_steps}")
    
    # 初始化统计（第373-376行）
    episode_rewards = [0.0] * eval_episodes
    episode_lengths = [0] * eval_episodes
    dones = [False] * eval_episodes
    
    # Reset所有环境（第378-382行）
    tds = []
    for env in eval_envs:
        td = env.reset()
        tds.append(td)
    
    print("\n初始状态（t=0，reset后）:")
    for idx in range(eval_episodes):
        has_pixels = "pixels" in tds[idx]
        print(f"  tds[{idx}]: 键={list(tds[idx].keys())[:3]}..., 包含pixels={has_pixels}")
    
    # 运行评估循环（第395行开始）
    max_steps = cfg.logger.eval_max_steps
    
    for t in range(max_steps):
        print(f"\n时间步 t={t}:")
        
        # 收集未结束的环境的观察（第396-402行）
        active_tds = []
        active_indices = []
        for idx, (td, done) in enumerate(zip(tds, dones)):
            if not done:
                active_tds.append(td)
                active_indices.append(idx)
        
        if not active_tds:
            print("  所有环境已结束")
            break
        
        print(f"  活跃环境: {active_indices}")
        
        # 批处理（第407行）
        batch_td = torch.stack(active_tds)
        print(f"  批处理后: 键={list(batch_td.keys())[:3]}..., pixels={'pixels' in batch_td}")
        
        # GPU推理（第409-411行）
        batch_td = batch_td.to(train_device)
        print(f"  移到{train_device}后: pixels={'pixels' in batch_td}")
        
        with torch.no_grad():
            batch_td = actor_critic[0](batch_td).to("cpu")
        
        print(f"  Actor推理并回CPU后: pixels={'pixels' in batch_td}")
        
        # 执行动作并更新（第414-424行）
        for i, (td, idx) in enumerate(zip(batch_td.unbind(0), active_indices)):
            print(f"\n    处理环境{idx}:")
            print(f"      unbind后td: 键={list(td.keys())[:3]}..., pixels={'pixels' in td}")
            
            # 步进环境（第416行）
            next_td = eval_envs[idx].step(td)
            tds[idx] = next_td  # 关键：更新tds！
            
            print(f"      step后next_td: 键={list(next_td.keys())[:3]}..., pixels={'pixels' in next_td}")
            print(f"      更新tds[{idx}]后: pixels={'pixels' in tds[idx]}")
            
            # 更新统计
            reward = next_td["next"]["reward"]
            if hasattr(reward, 'item'):
                reward = reward.item()
            episode_rewards[idx] += reward
            episode_lengths[idx] += 1
            
            done = next_td["next"]["done"]
            if hasattr(done, 'item'):
                done = done.item()
            dones[idx] = done
        
        # 视频录制检查（第453-460行）
        print(f"\n  视频录制检查（t={t}）:")
        pixels = []
        for idx in range(min(4, eval_episodes)):
            if not dones[idx]:
                has_pixels = "pixels" in tds[idx]
                print(f"    tds[{idx}]包含pixels: {has_pixels}")
                if has_pixels:
                    pixels.append(tds[idx]["pixels"])
                else:
                    print(f"    ⚠️ 缺少pixels！这就是问题所在！")
        
        if pixels:
            print(f"    成功收集{len(pixels)}个pixels")
        else:
            print(f"    ❌ 没有收集到任何pixels！")
    
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    
    # 最终检查
    print("\n最终状态:")
    for idx in range(eval_episodes):
        has_pixels = "pixels" in tds[idx]
        print(f"  tds[{idx}]: 包含pixels={has_pixels}, done={dones[idx]}")
    
    return tds


def test_simple_loop():
    """简化版测试，只关注pixels的流向"""
    print("\n" + "="*80)
    print("简化测试：追踪pixels的流向")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("使用CPU测试")
        train_device = torch.device("cpu")
    else:
        print("使用GPU测试")
        train_device = torch.device("cuda:0")
    
    # 创建单个环境
    import gymnasium as gym
    from torchrl.envs import GymWrapper
    
    env = gym.make("NewPasture-v5", render_mode='rgb_array')
    wrapped_env = GymWrapper(env, device="cpu", from_pixels=True, pixels_only=False)
    
    # 创建actor
    actor_critic = make_sac_models(wrapped_env, device=train_device)
    actor = actor_critic[0]
    
    # 初始reset
    td = wrapped_env.reset()
    print(f"\n初始reset: pixels={'pixels' in td}")
    
    # 模拟3个时间步
    for t in range(3):
        print(f"\n时间步 {t}:")
        
        # 模拟批处理和GPU推理
        batch_td = td.unsqueeze(0)  # 添加批次维度
        print(f"  批处理前: pixels={'pixels' in td}")
        
        batch_td = batch_td.to(train_device)
        print(f"  移到{train_device}: pixels={'pixels' in batch_td}")
        
        with torch.no_grad():
            batch_td = actor(batch_td).to("cpu")
        print(f"  Actor后回CPU: pixels={'pixels' in batch_td}")
        
        # unbind
        td = batch_td.squeeze(0)  # 移除批次维度
        print(f"  unbind后: pixels={'pixels' in td}")
        
        # step
        next_td = wrapped_env.step(td)
        print(f"  step后: pixels={'pixels' in next_td}")
        
        # 关键：更新td为next_td
        td = next_td
        print(f"  更新td: pixels={'pixels' in td}")


if __name__ == "__main__":
    print("开始精确模拟评估流程...")
    print("="*80)
    
    # 测试1: 完全精确的模拟
    tds = exact_evaluate_simulation()
    
    # 测试2: 简化的追踪
    test_simple_loop()
    
    print("\n" + "="*80)
    print("测试完成")
    print("="*80)