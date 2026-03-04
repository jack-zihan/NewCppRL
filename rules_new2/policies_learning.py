from __future__ import annotations

"""
Learning policy loader for rules_new2 evaluation.

This module keeps only the essentials:
- build actor from the same model factories used in training;
- load checkpoint actor weights;
- expose policy_fn(obs, env) -> action for collect_episode.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict

import torch
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

import envs_new  # noqa: F401  # ensure gym ids are registered
from rl_new.sac_cont_sy.env_utils import make_env_lambda
from rl_new.sac_cont_sy.model_utils import (
    make_sac_cnn_dual_models,
    make_sac_models,
    make_sac_resnet_dual_models,
)


def _resolve_device(device: str) -> str:
    dev = str(device).strip().lower()
    if dev == "cuda":
        dev = "cuda:0"
    if dev.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device '{device}' but CUDA is not available")
    return dev


def _extract_actor_state_dict(checkpoint: Any) -> Dict[str, Any]:
    if isinstance(checkpoint, dict):
        if "actor" in checkpoint:
            return checkpoint["actor"]
        if checkpoint and all(hasattr(v, "shape") for v in checkpoint.values()):
            return checkpoint
    if isinstance(checkpoint, (list, tuple)) and len(checkpoint) > 0:
        first = checkpoint[0]
        if isinstance(first, dict):
            return first
        if hasattr(first, "state_dict"):
            return first.state_dict()
    if hasattr(checkpoint, "state_dict"):
        return checkpoint.state_dict()
    raise RuntimeError("Unsupported checkpoint format: cannot find actor state_dict")


@lru_cache(maxsize=16)
def _load_actor(
    env_id: str,
    env_kwargs_json: str,
    ckpt_path: str,
    architecture: str,
    device: str,
    enable_hif: bool,
    backbone: str,
    hif_decoder_type: str,
):
    env_kwargs = json.loads(env_kwargs_json)
    model_env = make_env_lambda(
        env_id=env_id,
        device="cpu",
        from_pixels=False,
        **env_kwargs,
    )

    if architecture == "resnet":
        actor, _ = make_sac_resnet_dual_models(
            env=model_env,
            device=device,
            enable_hif=enable_hif,
            hif_decoder_type=hif_decoder_type,
            backbone_type=backbone,
        )
    elif architecture == "cnn":
        actor, _ = make_sac_cnn_dual_models(
            env=model_env,
            device=device,
            enable_hif=enable_hif,
        )
    elif architecture in {"vanilla", "mlp"}:
        actor, _ = make_sac_models(env=model_env, device=device)
    else:
        raise ValueError(f"Unsupported architecture='{architecture}', expected cnn|resnet|vanilla")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor_state = _extract_actor_state_dict(checkpoint)
    try:
        actor.load_state_dict(actor_state)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load actor state_dict. This usually means checkpoint architecture or "
            "evaluation env config (observation/action spec) does not match training. "
            f"arch={architecture}, env_id={env_id}, env_kwargs={env_kwargs}. "
            f"Original error: {exc}"
        ) from exc
    actor.eval()
    actor.to(device)
    return actor


def build_learning_policy(
    method_cfg: Dict[str, Any],
    env_id: str,
    env_kwargs: Dict[str, Any],
) -> Callable:
    ckpt = str(method_cfg["ckpt"])
    if not Path(ckpt).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    architecture = str(method_cfg.get("architecture", "resnet")).lower()
    device = _resolve_device(str(method_cfg.get("device", "cpu")))
    enable_hif = bool(method_cfg.get("enable_hif", False))
    backbone = str(method_cfg.get("backbone", "resnet18"))
    hif_decoder_type = str(method_cfg.get("hif_decoder_type", "two_stage"))

    actor = _load_actor(
        env_id=env_id,
        env_kwargs_json=json.dumps(env_kwargs, sort_keys=True),
        ckpt_path=ckpt,
        architecture=architecture,
        device=device,
        enable_hif=enable_hif,
        backbone=backbone,
        hif_decoder_type=hif_decoder_type,
    )

    def policy_fn(obs, _env):
        obs_tensor = torch.as_tensor(obs["observation"], dtype=torch.float32, device=device)
        vec_tensor = torch.as_tensor(obs["vector"], dtype=torch.float32, device=device)

        if obs_tensor.ndim == 3:
            obs_tensor = obs_tensor.unsqueeze(0)
        if vec_tensor.ndim == 0:
            vec_tensor = vec_tensor.view(1, 1)
        elif vec_tensor.ndim == 1:
            vec_tensor = vec_tensor.unsqueeze(0)

        td = TensorDict(
            {"observation": obs_tensor, "vector": vec_tensor},
            batch_size=[obs_tensor.shape[0]],
        )
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            out = actor(td)
        action = out["action"].detach().cpu().numpy()
        if action.shape[0] == 1:
            return action[0]
        return action

    return policy_fn
