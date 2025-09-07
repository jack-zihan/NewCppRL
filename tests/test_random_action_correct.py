#!/usr/bin/env python3
"""
正确的评估逻辑测试：每个环境只运行一个episode
使用随机动作，录制完整episode的视频
"""
import sys
sys.path.append('/home/lzh/NewCppRL')

import torch
from omegaconf import OmegaConf
from torchrl.record.loggers import get_logger
from tensordict import TensorDict
import numpy as np
from tqdm import tqdm

# 导入环境创建函数
from rl_new.sac_cont_sy.env_utils import make_single_environment
# 导入视频录制器
from torchrl_utils.local_video_recorder import LocalVideoRecorder


def test_random_action_correct():
    """使用随机动作测试正确的评估逻辑"""
    print("\n" + "="*80)
    print("测试正确的评估逻辑（每个环境一个episode）")
    print("="*80)
    
    # 1. 加载配置并修改
    print("\n1. 加载并修改配置:")
    config = OmegaConf.load('/home/lzh/NewCppRL/rl_new/sac_cont_sy/config-async-server.yaml')
    
    # 修改配置
    config.logger.eval_max_steps = 3000  # 最大步数
    config.logger.eval_video_skip = 3    # 每3帧记录一次
    config.logger.eval_episodes = 4      # 测试4个环境
    config.logger.eval_video = True      # 确保录制视频
    config.logger.backend = 'wandb'      # 使用wandb
    config.logger.mode = 'online'        # 在线模式
    
    print(f"   eval_max_steps = {config.logger.eval_max_steps}")
    print(f"   eval_video_skip = {config.logger.eval_video_skip}")
    print(f"   eval_episodes = {config.logger.eval_episodes}")
    
    # 2. 创建logger
    print("\n2. 创建wandb logger:")
    exp_name = "test_eval_correct_logic"
    logger = get_logger(
        logger_type=config.logger.backend,
        experiment_name=exp_name,
        logger_name=exp_name,
        wandb_kwargs={
            "mode": config.logger.mode,
            "config": dict(config),
            "project": config.logger.project_name,
            "group": "test_correct_eval",
            "name": exp_name
        }
    )
    print(f"   ✅ Logger创建成功")
    
    # 3. 设置设备
    print("\n3. 设置设备:")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"   使用设备: {device}")
    
    # 4. 创建评估环境（4个独立环境）
    print("\n4. 创建4个评估环境:")
    eval_envs = []
    for i in range(config.logger.eval_episodes):
        env = make_single_environment(
            cfg=config,
            from_pixels=True,  # 重要：需要像素用于视频录制
            device=device,
            seed=42 + i
        )
        eval_envs.append(env)
    print(f"   ✅ 创建了{len(eval_envs)}个环境")
    
    # 5. 创建视频录制器
    print("\n5. 创建视频录制器:")
    recorder = LocalVideoRecorder(
        device=device,
        max_len=config.logger.eval_max_steps // config.logger.eval_video_skip + 10,
        use_memmap=True,
        make_grid=True,
        nrow=2,
        skip=1,
        fps=30
    )
    print(f"   ✅ Recorder创建成功")
    
    # 6. 运行评估循环（与evaluate_policy逻辑一致）
    print("\n6. 开始评估循环:")
    
    # 初始化环境
    tds = []
    for idx, env in enumerate(eval_envs):
        td = env.reset()
        tds.append(td)
        print(f"   环境{idx}: reset完成, 包含pixels: {'pixels' in td}")
    
    # 统计变量
    episode_rewards = [0.0] * len(eval_envs)
    episode_lengths = [0] * len(eval_envs)
    dones = [False] * len(eval_envs)
    
    # 进度条
    pbar = tqdm(range(config.logger.eval_max_steps), desc="评估中")
    
    for t in pbar:
        # 收集未结束的环境
        active_indices = []
        for idx, done in enumerate(dones):
            if not done:
                active_indices.append(idx)
        
        # 如果所有环境都结束了，停止评估（与evaluate_policy一致）
        if not active_indices:
            print(f"\n   所有环境都完成了，在第{t}步结束评估")
            break
        
        # 为活跃的环境生成随机动作并执行
        for idx in active_indices:
            # 使用环境的action_space生成随机动作
            action = eval_envs[idx].action_spec.rand()
            tds[idx]["action"] = action
            
            # 执行动作
            next_td = eval_envs[idx].step(tds[idx])
            tds[idx] = next_td
            
            # 更新统计
            reward = next_td["next"]["reward"]
            if hasattr(reward, 'item'):
                reward = reward.item()
            episode_rewards[idx] += reward
            episode_lengths[idx] += 1
            
            # 检查是否结束
            done = next_td["next"]["done"]
            if hasattr(done, 'item'):
                done = done.item()
            dones[idx] = done
            
            if done:
                print(f"\n   环境{idx}完成: 步数={episode_lengths[idx]}, 总奖励={episode_rewards[idx]:.2f}")
        
        # 录制视频帧（每eval_video_skip步录制一次）
        if (t + 1) % config.logger.eval_video_skip == 0:
            # 收集所有环境的pixels（包括已结束的，使用最后一帧）
            pixels = []
            for i in range(len(eval_envs)):
                if "pixels" in tds[i]:
                    pixels.append(tds[i]["pixels"])
                else:
                    # 如果没有pixels，创建黑色占位图
                    pixels.append(torch.zeros(400, 400, 3, dtype=torch.uint8, device=device))
            
            # Stack并应用到recorder
            stacked = torch.stack(pixels[:4], 0)  # 确保只取前4个
            recorder.apply(stacked)
            
            # 更新进度条信息
            if t % 30 == 0:  # 每30步更新一次
                active_count = len(active_indices)
                completed_count = sum(dones)
                pbar.set_postfix({
                    'active': f'{active_count}/{config.logger.eval_episodes}',
                    'done': completed_count,
                    'frames': recorder.idx
                })
    
    pbar.close()
    
    # 7. 统计最终结果
    print(f"\n7. 评估结果统计:")
    print(f"   录制的总帧数: {recorder.idx}")
    print(f"   各环境完成状态: {dones}")
    print(f"   各环境episode长度: {episode_lengths}")
    print(f"   各环境总奖励: {[f'{r:.2f}' for r in episode_rewards]}")
    print(f"   平均奖励: {np.mean(episode_rewards):.2f}")
    print(f"   平均步数: {np.mean(episode_lengths):.2f}")
    
    # 8. 上传视频到wandb
    print(f"\n8. 准备上传视频:")
    vid_tensor = recorder.dump()
    
    if vid_tensor is not None:
        print(f"   ✅ Dump成功, 视频shape: {vid_tensor.shape}")
        print(f"   视频长度: {vid_tensor.shape[1]}帧, 约{vid_tensor.shape[1]/30:.1f}秒（30fps）")
        print(f"   正在上传到wandb...")
        
        try:
            logger.log_video('eval/video', vid_tensor, step=1000)
            print(f"   ✅ 视频上传成功!")
        except Exception as e:
            print(f"   ❌ 上传失败: {e}")
    else:
        print(f"   ❌ Dump返回None，没有视频数据")
    
    # 9. 清理资源
    print("\n9. 清理资源:")
    for env in eval_envs:
        try:
            env.close()
        except:
            pass
    
    try:
        logger.close()
        print("   ✅ 资源清理完成")
    except:
        pass
    
    print("\n" + "="*80)
    print("测试完成！")
    print("这次的评估逻辑与实际evaluate_policy一致：")
    print("- 每个环境只运行一个episode")
    print("- 当所有环境都完成后结束评估")
    print("- 录制所有环境的完整episode视频")
    print(f"项目: SAC_2025, 实验: {exp_name}")
    print("="*80)


if __name__ == "__main__":
    test_random_action_correct()