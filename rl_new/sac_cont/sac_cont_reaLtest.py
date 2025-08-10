import os
import torch
import yaml
import json
import gymnasium as gym
from pathlib import Path
from omegaconf import DictConfig
from gymnasium.wrappers import HumanRendering, RecordVideo
from torchrl.envs.utils import ExplorationType, set_exploration_type

import envs  # noqa
from envs.maps.real_map_convertion_2 import restore_coordinates

base_dir = Path(__file__).parent.parent.parent
# real_map_dir = f'{base_dir}/envs/maps/real'
# real_map_dir = f'{base_dir}/envs/maps/real_map_test'
# real_map_dir = f'{base_dir}/envs/maps/real_test_1'
real_map_dir = f'{base_dir}/envs/maps/real_true'
cfg = DictConfig(yaml.load(open(f'{base_dir}/configs/env_config.yaml'), Loader=yaml.FullLoader))

render_human = True
record_video = False
# record_video = False
# rocord_state = True
rocord_state = False
# act_randomly = True
act_randomly = False
seed =114# 44
name_prefix="hard4"

device = 'cpu'
video_save_path = '/home/lzh/NewCppRL/ckpt/video'
# pt_path = f'/home/lzh/NewCppRL/ckpt/sac_cont/0909/sac_our_model_con3_t[02350]_r[2703.08=2662.85~2782.18].pt'
# pt_path = f'/home/lzh/NewCppRL/ckpt/sac_cont/finetune/t[02350]_r[2782.06=2666.52~2872.77].pt'
pt_path = f'/home/lzh/NewCppRL/ckpt/sac_cont/finetune/t[01700]_r[2741.21=2632.53~2847.21].pt'
actor_critic = torch.load(pt_path, map_location=device)
actor = actor_critic[0].to(device)

# cfg.env.params.num_obstacles_range = [0, 0]
env = gym.make(
    render_mode='rgb_array' if (render_human or record_video) else None,
    **cfg.env.params,
)

if record_video:
    render_human = True
    env = RecordVideo(env, video_folder=video_save_path, name_prefix=name_prefix,episode_trigger=lambda x: True)
    env.metadata['render_fps'] = 2

if render_human:
    env = HumanRendering(env)
    env.metadata['render_fps'] = 30

exploration_type = ExplorationType.RANDOM if act_randomly else ExplorationType.DETERMINISTIC

with set_exploration_type(exploration_type), torch.no_grad():
    # obs, info = env.reset(seed=120, options={
    #     'weed_dist': 'gaussian',
    #     # 'weed_dist': 'uniform',
    #     # 'map_id': 80,
    #     "weed_num": 100,
    #     # "specific_scenario_dir": real_map_dir
    # })
    env.set_obstacle_range([5,8])
    obs, info = env.reset(
        # seed=120,
        seed=seed,
        options={
            'weed_dist': 'gaussian',
            # 'weed_dist': 'uniform',
            # 'map_id': 80,
            "weed_num": 200,
            # "weed_num": 50,
            "specific_scenario_dir": real_map_dir,
            # 'initial_position': (200, 200),  # X 和 Y 坐标
            # 'initial_direction': 360.0,
        }
    )
    env.action_space.seed(66)
    done = False
    ret = 0.
    t = 0

    positions = []
    directions = []

    while not done:
        if isinstance(obs, dict):
            observation = torch.from_numpy(obs['observation']).float().to(device).unsqueeze(0)
            vector = torch.tensor([obs['vector']]).float().to(device).unsqueeze(0)
        # observation = torch.from_numpy(observation).float().to(device).unsqueeze(0)
        # vector = torch.tensor([vector]).float().to(device).unsqueeze(0)
        # Get Output
        logits = actor(observation=observation, vector=vector)
        action = logits[2][0].tolist()
        # print(action)
        # print(env.agent.position, env.agent.direction)
        positions.append(env.agent.position)
        directions.append(env.agent.direction)

        obs, reward, done, _, info = env.step(action)
        t += 1
        ret += reward
        if record_video:
            print(f'{t:04d} | {reward:.3f}, {ret:.3f}')
        if render_human or record_video:
            env.render()
env.close()

if rocord_state:
    positions = restore_coordinates(positions, f'{real_map_dir}/transformation.json')
    data = {
        "Positions": [  # List of position dictionaries
            {
                "x": pos[0],
                "y": pos[1]
            } for pos in positions
        ],
        "Directions": directions  # List of directions
    }
    # Save data to JSON file
    json_path = os.path.join(real_map_dir, 'agent_path.json')
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Agent's path saved to {json_path}")

    if hasattr(env, 'close'):
        env.close()
