from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gymnasium as gym
import numpy as np
import torch
import yaml
from gymnasium.wrappers import HumanRendering
from omegaconf import DictConfig
from torchrl.envs.utils import ExplorationType, set_exploration_type
import csv
import os
import time
from env_make import get_env


LOG_DIR = '/Users/chuyuliu/CppRL-main-chuyu/logs'

difficulty = "hard"
rl_model = "dqn_model_2"
save_path = os.path.join(LOG_DIR, f"{rl_model}_{difficulty}.csv")

base_dir = Path(__file__).parent.parent.parent
cfg = DictConfig(yaml.load(open(f'{base_dir}/configs/env_config.yaml'), Loader=yaml.FullLoader))
episodes = 10


device = 'cpu'
pt_path = '/Users/chuyuliu/CppRL-main-chuyu/ckpt/dqn_model_3_0907.pt'
model = torch.load(pt_path, map_location=torch.device('cpu')).to(device)
actor = model[0]
noise_set = [0, 0, 0]

# cfg.env.params.num_obstacles_range = [0, 0]

env, obs = get_env()

costs = []

cover_90,cover_95, cover_98, cover, dist_list = -1, -1, -1, [], []
init_weed = env.map_weed.sum()
weed_dist = "uniform"
random_seed = 96
map_id = 22
collapse = -1
overall_length = 0


def save_data_to_csv(file_path, weed_dist, random_seed, map_id, noise_set,  collapse, cover_90, cover_95, cover_98,cover, dist_list):
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["weed_dist","random_seed", "map_id", "noise_set", "collapse", "cover_90", "cover_95", "cover_98", "cover", "dist_list"])
        cover_str = ",".join(map(str, cover))
        dist_str = ",".join(map(str, dist_list))
        writer.writerow([weed_dist, random_seed, map_id, noise_set,  collapse, cover_90, cover_95, cover_98,cover_str, dist_str])

with set_exploration_type(ExplorationType.MODE), torch.no_grad():
    i = 0
    failed_count = 0
    while i < episodes:
        done = False
        ret = 0.
        max_r = -100.
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
            observation = torch.from_numpy(observation).float().to(device).unsqueeze(0)
            vector = torch.tensor([vector]).float().to(device).unsqueeze(0)
            # print(obs)
            # print(observation[0, 5:8])
            # Get Output
            action = actor(observation=observation, vector=vector)
            # print(action[0])
            action = action[0].argmax().item()
            # action = int(action)
            # print(action)
            
            past_position = env.agent.position
            obs, reward, done, time_out, _ = env.step(action)
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
            
            
            max_r = max(max_r, reward)
            t += 1
            ret += 0.99 ** t * reward
            # print(f'{t:04d} | {reward:.3f}, {ret:.3f}')
        print(f'Max r: {max_r}')
env.close()
costs = np.array(costs)
print(f'{costs.mean()} +- {costs.std()}')
print(f'{failed_count} / {i + failed_count} = {failed_count / (i + failed_count)}')
