"""
Learning policy loader for evaluation.

Supports:
- ResNet dual-head actor (architecture="resnet")
- Vanilla SAC actor (architecture="vanilla")
"""
from functools import lru_cache
from typing import Callable, Dict, Any

import gymnasium as gym
import torch
from torchrl.envs.utils import set_exploration_type, ExplorationType

import envs_new  # noqa: F401  # ensure envs are registered
from rl_new.sac_cont_sy.model_utils import make_sac_models, make_sac_resnet_dual_models


@lru_cache(maxsize=8)
def _load_actor(env_id: str, ckpt_path: str, architecture: str, device: str = "cpu"):
    env = gym.make(env_id)
    if architecture == "resnet":
        actor, _ = make_sac_resnet_dual_models(env=env, device=device, enable_hif=False)
    else:
        actor, _ = make_sac_models(env=env, device=device)

    state = torch.load(ckpt_path, map_location=device)
    # ckpt 可能保存 {'actor': state_dict, 'critic': ...} 或直接 ModuleList
    if isinstance(state, dict) and "actor" in state:
        actor.load_state_dict(state["actor"])
    else:
        actor.load_state_dict(state[0] if isinstance(state, (list, tuple)) else state)

    actor.to(device)
    actor.eval()
    env.close()
    return actor


def build_learning_policy(method_cfg: Dict[str, Any],
                          env_id: str,
                          device: str = "cpu") -> Callable:
    """
    Return a policy_fn(obs_dict, env) -> np.ndarray (action)
    """
    actor = _load_actor(env_id, method_cfg["ckpt"], method_cfg.get("architecture", "resnet"), device)

    def policy_fn(obs, env):
        # obs is dict with 'observation', 'vector'
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            td = {
                "observation": torch.as_tensor(obs["observation"]).unsqueeze(0).to(device),
                "vector": torch.as_tensor(obs["vector"]).unsqueeze(0).to(device),
            }
            out = actor(td)
            action = out["action"].squeeze(0).cpu().numpy()
        return action

    return policy_fn
