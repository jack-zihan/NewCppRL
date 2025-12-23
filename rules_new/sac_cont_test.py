from pathlib import Path
import sys

# 设置基础路径（项目根目录）
BASE_DIR = Path(__file__).parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

def to_absolute_path(path):
    """将路径转换为绝对路径"""
    p = Path(path)
    return str(p if p.is_absolute() else BASE_DIR / p)

import gymnasium as gym
import torch
import yaml
from gymnasium.wrappers import HumanRendering
from omegaconf import DictConfig
from torchrl.envs.utils import ExplorationType, set_exploration_type
from rules.env_make import get_env
import os
import time
import csv
import numpy as np
import envs  # noqa

episodes = 10
render = True
act_randomly = True
# act_randomly = False

device = 'cpu'

noise_set = [0, 0, 0]


# 使用 BASE_DIR 构建路径
LOG_DIR = BASE_DIR / 'rules' / 'logs'

noise_set = [0, 0, 0]
difficulty = "easy"
rl_model = "t[02600]_r[2731.41=2717.75~2750.74].pt"
save_path = LOG_DIR / f"{rl_model}_{difficulty}.csv"

episodes = 10


device = 'cpu'
pt_path = 'rules/ckpt/t[02600]_r[2731.41=2717.75~2750.74].pt'

model = torch.load(to_absolute_path(pt_path), map_location=torch.device('cpu')).to(device)

actor_critic = torch.load(to_absolute_path(pt_path), map_location=torch.device('cpu')).to(device)
actor = actor_critic[0].to(device)

env, obs = get_env()

costs = []

cover_90,cover_95, cover_98, cover, dist_list = -1, -1, -1, [], []
init_weed = env.map_weed.sum()
weed_dist = "gaussian"
random_seed = 25
map_id = 2
collapse = -1
overall_length = 0

exploration_type = ExplorationType.RANDOM if act_randomly else ExplorationType.DETERMINISTIC

def save_data_to_csv(file_path, weed_dist, random_seed, map_id, noise_set,  collapse, cover_90, cover_95, cover_98,cover, dist_list):
    file_path = to_absolute_path(file_path)  # 确保路径是绝对路径
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["weed_dist","random_seed", "map_id", "noise_set", "collapse", "cover_90", "cover_95", "cover_98", "cover", "dist_list"])
        cover_str = ",".join(map(str, cover))
        dist_str = ",".join(map(str, dist_list))
        writer.writerow([weed_dist, random_seed, map_id, noise_set,  collapse, cover_90, cover_95, cover_98,cover_str, dist_str])

with set_exploration_type(exploration_type), torch.no_grad():
    for i in range(episodes):
        done = False
        ret = 0.
        t = 0
        start_time = time.time()
        while not done:
            if time.time() - start_time > 300: 
                save_data_to_csv(save_path, weed_dist, random_seed, map_id, noise_set, 0, cover_90, cover_95, cover_98, cover,dist_list)
                print("运行时间超过5分钟，程序已退出。")
                sys.exit()
            if isinstance(obs, dict):
                observation = obs['observation']
                vector = obs['vector']
            # observation: [C,H,W] -> [B=1,C,H,W]
            observation = torch.from_numpy(observation).float().to(device).unsqueeze(0)
            vector = torch.tensor(vector).float().to(device).unsqueeze(0)
            # Get Output
            logits = actor(observation=observation, vector=vector)
            action = logits[2][0].tolist()
            past_position = env.agent.position
            obs, reward, done, _, info = env.step(action)
            now_position = env.agent.position
            distance = np.linalg.norm(np.array(now_position) - np.array(past_position))
            overall_length += distance
            
            cover_rate = (init_weed - env.map_weed.sum()) / init_weed
            if cover_rate >= 0.98:
                cover_98 = overall_length
            elif cover_rate >= 0.95:
                cover_95 = overall_length
            elif cover_rate >= 0.90:
                cover_90 = overall_length
            cover.append(cover_rate)
            dist_list.append(overall_length)
            if done:
                if env.check_collision():
                    save_data_to_csv(save_path, weed_dist, random_seed, map_id, noise_set, 1, cover_90, cover_95, cover_98, cover,dist_list)
                    exit()
                else:
                    save_data_to_csv(save_path, weed_dist, random_seed, map_id, noise_set, 0, cover_90, cover_95, cover_98, cover,dist_list)
                    exit()
            
            
            t += 1
            ret += reward
            # print(f'{t:04d} | {reward:.3f}, {ret:.3f}')
            if render:
                env.render()
env.close()
