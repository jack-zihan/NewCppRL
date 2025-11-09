from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.wrappers import HumanRendering
from torchrl.envs.utils import ExplorationType, set_exploration_type

import envs_new  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description="SAC continuous action test script")
    parser.add_argument("--env_id", type=str, default="NewPasture-v2")
    parser.add_argument("--ckpt", type=str,
                       default="/home/lzh/NewCppRL/ckpt/sac_cont/202400915-finetune/t[02350]_r[2782.06=2666.52~2872.77].pt")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    env = gym.make(args.env_id, render_mode="rgb_array" if args.render else None)
    if args.render:
        env = HumanRendering(env)

    try:
        actor_critic = torch.load(str(Path(args.ckpt)), map_location=device, weights_only=False)
    except ValueError as e:
        if "InteractionType" not in str(e):
            raise
        # Create compatible InteractionType enum for old checkpoints
        from enum import IntEnum
        import tensordict.nn.probabilistic
        class InteractionType(IntEnum):
            MODE = 0
            MEAN = 1
            RANDOM = 2
            DETERMINISTIC = 3
            _LEGACY_4 = 4  # Old checkpoint value
        tensordict.nn.probabilistic.InteractionType = InteractionType
        actor_critic = torch.load(str(Path(args.ckpt)), map_location=device, weights_only=False)

    actor = actor_critic[0] if isinstance(actor_critic, torch.nn.ModuleList) else actor_critic
    actor.eval()

    exploration_mode = ExplorationType.DETERMINISTIC if args.deterministic else ExplorationType.RANDOM

    with set_exploration_type(exploration_mode), torch.no_grad():
        for ep in range(args.episodes):
            obs, _ = env.reset()
            done = False
            episode_return = 0.0
            step = 0

            while not done and step < args.max_steps:
                observation = obs["observation"] if isinstance(obs, dict) else obs
                vector = obs["vector"] if isinstance(obs, dict) else np.zeros(1, dtype=np.float32)

                obs_t = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                vec_t = torch.as_tensor(vector, dtype=torch.float32, device=device).unsqueeze(0)

                logits = actor(observation=obs_t, vector=vec_t)
                action = logits[2][0].tolist()

                obs, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)

                step += 1
                episode_return += float(reward)

                if step % 100 == 0 or done:
                    print(f"{step:4d} | {float(reward):7.3f}, {episode_return:7.3f}")

                if args.render:
                    env.render()

            print(f"Episode {ep + 1}: {step} steps, return = {episode_return:.3f}")

    env.close()


if __name__ == "__main__":
    main()
