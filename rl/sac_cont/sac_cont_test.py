from pathlib import Path

import gymnasium as gym
import torch
import yaml
import numpy as np
import pygame
from gymnasium.wrappers import HumanRendering
from omegaconf import DictConfig
from torchrl.envs.utils import ExplorationType, set_exploration_type

import envs  # noqa
from envs.wrapper.reward_tracker import RewardTracker

# 初始化pygame以处理键盘事件
pygame.init()

base_dir = Path(__file__).parent.parent.parent
cfg = DictConfig(yaml.load(open(f'{base_dir}/configs/env_config.yaml'), Loader=yaml.FullLoader))
episodes = 1
render = True
log_reward = False
act_randomly = True
# act_randomly = False

device = 'cpu'
pt_path = f'/home/lzh/NewCppRL/ckpt/sac_cont/0909/sac_our_model_con3_t[02350]_r[2703.08=2662.85~2782.18].pt'
actor_critic = torch.load(pt_path).to(device)
actor = actor_critic[0].to(device)

# cfg.env.params.num_obstacles_range = [0, 0]
env = gym.make(
    render_mode='rgb_array' if render else None,
    **cfg.env.params,
)

if log_reward:
    env = RewardTracker(env)

if render:
    env = HumanRendering(env)

exploration_type = ExplorationType.RANDOM if act_randomly else ExplorationType.DETERMINISTIC

episode_returns = []

with set_exploration_type(exploration_type), torch.no_grad():
    for i in range(episodes):
        obs, info = env.reset(seed=88,  # 120
                              options={
                                  'weed_dist': 'uniform',
                                  # 'map_id': 66, #100
                                  "weed_num": 10,

                                  # "specific_scenario_dir": real_map_dir
                              })
        done = False
        ret = 0.
        t = 0

        print(f"\n=== Episode {i + 1}/{episodes} ===")

        paused = False
        while not done:
            # 检查键盘事件
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                        print("模拟已" + ("暂停" if paused else "恢复"))

            # 如果暂停状态，持续检测键盘输入，直到按下空格键恢复
            if paused:
                if render:
                    env.render()  # 在暂停状态也持续渲染环境
                pygame.time.wait(100)  # 等待100毫秒，减少CPU占用
                continue

            if isinstance(obs, dict):
                observation = obs['observation']
                vector = obs['vector']
            observation = torch.from_numpy(observation).float().to(device).unsqueeze(0)
            vector = torch.tensor([vector]).float().to(device).unsqueeze(0)
            # Get Output
            logits = actor(observation=observation, vector=vector)
            action = logits[2][0].tolist()
            obs, reward, done, _, info = env.step(action)
            t += 1
            ret += reward
            if 'reward_details' in info:
                details = info['reward_details']
                print(f'{t:04d} | Total: {reward:7.3f} (Σ={ret:7.3f}) | '
                      f'Const: {details["const"]:6.3f} | '
                      f'Turn: {details["turn"]:6.3f} | '
                      f'Frontier: {details["frontier"]:6.3f} | '
                      f'Weed: {details["weed"]:6.3f} | '
                      f'Extra: {details["extra"]:6.3f}')
            else:
                print('no_reward_details')
                print(f'{t:04d} | {reward:.3f}, {ret:.3f}')

            if render:
                env.render()

        episode_returns.append(ret)

        if log_reward:
            summary = env.get_episode_summary(-1)
            if summary:
                print(f"\nEpisode {i + 1} Summary:")
                print(f"  Total return: {ret:.2f}")
                for key in ['const', 'turn', 'frontier', 'weed', 'extra']:
                    sum_key = f'{key}_sum'
                    mean_key = f'{key}_mean'
                    std_key = f'{key}_std'
                    print(f"  {key.capitalize():8s}: Sum={summary[sum_key]:7.2f}, "
                          f"Mean={summary[mean_key]:6.3f}, "
                          f"Std={summary[std_key]:6.3f}")
if log_reward:
    output_dir = Path(f'{base_dir}/logs/reward_analysis')
    output_dir.mkdir(exist_ok=True)
    env.save_rewards(str(output_dir / 'reward_details.csv'))
    env.plot_rewards(-1, save_path=str(output_dir / 'episode_rewards.png'))

    # Print overall statistics
    print(f"\n=== Overall Statistics ===")
    print(f"Episodes completed: {episodes}")
    print(f"Average return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
    print(f"Min return: {np.min(episode_returns):.2f}")
    print(f"Max return: {np.max(episode_returns):.2f}")

env.close()
pygame.quit()

