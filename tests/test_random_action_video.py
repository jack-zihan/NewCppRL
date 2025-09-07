#!/usr/bin/env python3
"""
使用随机动作测试更长的视频上传
测试3000步，每3步录制一帧，应该有约1000帧视频
"""
import sys
sys.path.append('/home/lzh/NewCppRL')

import torch
from omegaconf import OmegaConf
from torchrl.record.loggers import get_logger
from torchrl.envs import EnvCreator, ParallelEnv
from tensordict import TensorDict
import numpy as np
from tqdm import tqdm

# 导入环境创建函数
from rl_new.sac_cont_sy.env_utils import make_single_environment
# 导入视频录制器
from torchrl_utils.local_video_recorder import LocalVideoRecorder


def test_random_action_long_video():
    """使用随机动作测试长视频上传"""
    print("\n" + "="*80)
    print("测试随机动作长视频上传功能")
    print("="*80)
    
    # 1. 加载配置并修改
    print("\n1. 加载并修改配置:")
    config = OmegaConf.load('/home/lzh/NewCppRL/rl_new/sac_cont_sy/config-async-server.yaml')
    
    # 修改配置以录制长视频
    config.logger.eval_max_steps = 3000  # 测试3000帧
    config.logger.eval_video_skip = 3    # 每3帧记录一次，约1000帧视频
    config.logger.eval_episodes = 4      # 测试4个环境
    config.logger.eval_video = True      # 确保录制视频
    config.logger.backend = 'wandb'      # 使用wandb
    config.logger.mode = 'online'        # 在线模式
    
    print(f"   eval_max_steps = {config.logger.eval_max_steps}")
    print(f"   eval_video_skip = {config.logger.eval_video_skip}")
    print(f"   预计录制帧数 = {config.logger.eval_max_steps // config.logger.eval_video_skip}")
    print(f"   eval_episodes = {config.logger.eval_episodes}")
    
    # 2. 创建logger
    print("\n2. 创建wandb logger:")
    exp_name = "test_random_action_long_video"
    logger = get_logger(
        logger_type=config.logger.backend,
        experiment_name=exp_name,
        logger_name=exp_name,
        wandb_kwargs={
            "mode": config.logger.mode,
            "config": dict(config),
            "project": config.logger.project_name,
            "group": "test_long_video",
            "name": exp_name
        }
    )
    print(f"   ✅ Logger创建成功")
    
    # 3. 设置设备
    print("\n3. 设置设备:")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"   使用设备: {device}")
    
    # 4. 创建评估环境（4个并行环境）
    print("\n4. 创建4个并行评估环境:")
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
        max_len=config.logger.eval_max_steps // config.logger.eval_video_skip + 10,  # 留出余量
        use_memmap=True,
        make_grid=True,
        nrow=2,
        skip=1,
        fps=30  # 提高fps让视频更流畅
    )
    print(f"   ✅ Recorder创建成功，最大长度: {recorder.max_len}")
    
    # 6. 运行评估循环
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
    episode_count = [0] * len(eval_envs)  # 记录每个环境完成的episode数
    
    # 进度条
    pbar = tqdm(range(config.logger.eval_max_steps), desc="评估中")
    
    for t in pbar:
        # 为每个未结束的环境生成随机动作
        for idx, (td, done) in enumerate(zip(tds, dones)):
            if not done:
                # 使用环境的action_space生成随机动作
                action = eval_envs[idx].action_spec.rand()
                td["action"] = action
                
                # 执行动作
                next_td = eval_envs[idx].step(td)
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
                
                if done:
                    # 记录完成的episode
                    episode_count[idx] += 1
                    # 重置环境继续运行（让视频更长）
                    td_reset = eval_envs[idx].reset()
                    tds[idx] = td_reset
                    episode_rewards[idx] = 0
                    episode_lengths[idx] = 0
                    dones[idx] = False  # 重置done标志，继续运行
        
        # 录制视频帧
        if (t + 1) % config.logger.eval_video_skip == 0:
            # 收集4个环境的pixels
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
                avg_reward = np.mean([r for r in episode_rewards if r > 0]) if any(episode_rewards) else 0
                pbar.set_postfix({
                    'frames': recorder.idx,
                    'episodes': sum(episode_count),
                    'avg_reward': f'{avg_reward:.2f}'
                })
    
    pbar.close()
    
    # 7. 上传视频到wandb
    print(f"\n7. 准备上传视频:")
    print(f"   录制的总帧数: {recorder.idx}")
    print(f"   各环境完成的episode数: {episode_count}")
    
    vid_tensor = recorder.dump()
    
    if vid_tensor is not None:
        print(f"   ✅ Dump成功, 视频shape: {vid_tensor.shape}")
        print(f"   正在上传到wandb...")
        
        try:
            logger.log_video('eval/video', vid_tensor, step=1000)
            print(f"   ✅ 视频上传成功!")
        except Exception as e:
            print(f"   ❌ 上传失败: {e}")
    else:
        print(f"   ❌ Dump返回None，没有视频数据")
    
    # 8. 清理资源
    print("\n8. 清理资源:")
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
    print(f"录制了{recorder.idx}帧视频，约{recorder.idx/30:.1f}秒（30fps）")
    print("请检查wandb项目中的长视频")
    print("项目: SAC_2025, 实验: test_random_action_long_video")
    print("="*80)


if __name__ == "__main__":
    test_random_action_long_video()