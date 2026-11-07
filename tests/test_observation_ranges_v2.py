import numpy as np
import gymnasium as gym
import envs_new


def test_v2_observation_nonnegative():
    env = gym.make('NewPasture-v2')
    try:
        for seed in [0, 1, 2, 3, 4]:
            obs, info = env.reset(seed=seed)
            mn = float(obs['observation'].min()); mx = float(obs['observation'].max())
            assert mn >= 0.0 - 1e-12 and mx <= 1.0 + 1e-6, (
                f"v2 observation out of [0,1]: min={mn}, max={mx}, seed={seed}"
            )
            assert env.observation_space.contains(obs)
    finally:
        env.close()

