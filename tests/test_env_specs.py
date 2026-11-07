import gymnasium as gym
import numpy as np
import pytest

import envs_new  # noqa: F401 - ensures Gymnasium registrations are executed
from torchrl.envs import GymWrapper
from torchrl.envs.gym_like import default_info_dict_reader


def sample_observations(env_id: str, num_steps: int = 2):
    """Collect reset + step observations for reproducibility."""
    env = gym.make(env_id)
    try:
        obs, _ = env.reset(seed=0)
        yield obs
        action = env.action_space.sample()
        for _ in range(num_steps):
            obs, *_ = env.step(action)
            yield obs
    finally:
        env.close()


def test_newpasture_v6_observation_maps_stay_in_declared_range():
    """V6声明 observation ∈ [0,1]，但实际观测含有负值，触发 PassiveEnvChecker."""
    mins = []
    for obs in sample_observations("NewPasture-v6"):
        mins.append(float(np.min(obs["observation"])))
    min_value = min(mins)
    assert min_value >= 0.0, f"observation entries dropped below 0 (min={min_value})"


def test_newpasture_v6_auto_register_info_dict_breaks_due_to_spec_mismatch():
    """TorchRL 自动注册 info 时会因 observation_spec 校验失败而抛出 AssertionError。"""
    env = gym.make("NewPasture-v6")
    try:
        wrapper = GymWrapper(env)
        reader = default_info_dict_reader(keys=["overlap_count"])
        with pytest.raises(AssertionError):
            wrapper.auto_register_info_dict(reader)
    finally:
        env.close()
