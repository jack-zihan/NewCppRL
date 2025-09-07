#!/usr/bin/env python3
"""
Focused test for rl_new/sac_cont_sy/sac_utils.evaluate_policy
- Uses 4 eval environments
- Initializes an untrained actor_critic
- Logs video to Weights & Biases (wandb)
- Captures and prints the wandb run URL and eval metrics

This test does not modify any existing code; all config tweaks are done here.
"""
import os
import sys

# Ensure repo root is importable
REPO_ROOT = "/home/lzh/NewCppRL"
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

import torch
import wandb
import types
import numpy as np
from omegaconf import OmegaConf

# --- Minimal stubs to avoid heavy GPU deps in envs_new import path (cupy) ---
def _install_cupy_stubs():
    if "cupy" in sys.modules:
        return
    cupy = types.ModuleType("cupy")
    cupy.float32 = np.float32

    class _CudaDevice:
        def __init__(self, device_id):
            self.device_id = device_id
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False

    class _Cuda:
        Device = _CudaDevice

    def asarray(x):
        return np.asarray(x)

    def any(x):
        return np.any(x)

    cupy.asarray = asarray
    cupy.any = any
    cupy.cuda = _Cuda()

    # cupyx.scipy.ndimage stub
    cupyx = types.ModuleType("cupyx")
    scipy = types.ModuleType("scipy")
    ndimage = types.ModuleType("ndimage")

    def distance_transform_edt(arr):
        # Return a dummy object with .astype and .get
        class _Dummy:
            def __init__(self, base):
                self._base = np.asarray(base)
            def astype(self, dtype):
                return self
            def get(self):
                # Return zeros with same shape; function won't be used in this test path
                return np.zeros_like(self._base, dtype=np.float32)
        return _Dummy(arr)

    ndimage.distance_transform_edt = distance_transform_edt
    scipy.ndimage = ndimage
    cupyx.scipy = scipy

    sys.modules["cupy"] = cupy
    sys.modules["cupyx"] = cupyx
    sys.modules["cupyx.scipy"] = scipy
    sys.modules["cupyx.scipy.ndimage"] = ndimage


_install_cupy_stubs()

from rl_new.sac_cont_sy.env_utils import make_single_environment
from rl_new.sac_cont_sy.model_utils import make_sac_models
from rl_new.sac_cont_sy.sac_utils import evaluate_policy


class WandbLoggerShim:
    """Minimal logger shim matching the interface expected by evaluate_policy.

    - Initializes a wandb run in __init__
    - log_scalar(tag, value, step)
    - log_video(tag, vid_tensor [1, T, C, H, W] uint8, step)
    """

    def __init__(self, project: str, group: str, name: str, config: dict, mode: str = "online"):
        # Respect requested mode via environment, compatible across wandb versions
        if mode:
            os.environ["WANDB_MODE"] = mode
        # Start run
        self._run = wandb.init(project=project, group=group, name=name, config=config)

    def log_scalar(self, tag: str, value, step: int):
        try:
            wandb.log({tag: float(value) if hasattr(value, "__float__") else value, "_step": step}, step=step)
        except Exception:
            # Best-effort fallback
            wandb.log({tag: value}, step=step)

    def log_video(self, tag: str, vid_tensor: torch.Tensor, step: int):
        if vid_tensor is None:
            return
        # Expect [1, T, C, H, W] uint8 on CPU; adapt robustly
        video = vid_tensor
        if video.ndim == 5 and video.shape[0] == 1:
            video = video[0]
        # Ensure CPU uint8 numpy array in [T, C, H, W]
        if isinstance(video, torch.Tensor):
            video = video.detach().cpu()
        if video.dtype != torch.uint8:
            video = video.to(torch.uint8)
        video_np = video.numpy()
        wb_video = wandb.Video(video_np, fps=6, format="mp4")
        wandb.log({tag: wb_video}, step=step)

    def close(self):
        try:
            wandb.finish()
        except Exception:
            pass


def test_evaluate_policy_wandb_4envs():
    # 1) Load base config and apply minimal, test-scoped overrides
    cfg = OmegaConf.load(os.path.join(REPO_ROOT, "rl_new/sac_cont_sy/config-async-server.yaml"))
    # Keep seeds deterministic and use 4 eval envs as requested
    cfg.seed = 42
    cfg.logger.backend = "wandb"
    cfg.logger.mode = "online"  # user wants actual upload
    cfg.logger.eval_episodes = 4
    cfg.logger.eval_video = True
    # Keep evaluation short yet with enough frames for a visible video
    cfg.logger.eval_max_steps = 300
    cfg.logger.eval_video_skip = 10

    # 2) Create W&B logger (project/group/name sourced from cfg) without torchrl dependency
    exp_name = "evaluate_policy_wandb_4envs"
    logger = WandbLoggerShim(
        project=cfg.logger.project_name,
        group=cfg.logger.group_name,
        name=exp_name,
        config=dict(cfg),
        mode=cfg.logger.mode,
    )

    # 3) Device and model creation (untrained actor_critic)
    train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Build a single non-pixel env to construct the model specs
    dummy_env = make_single_environment(cfg, device=train_device, from_pixels=False)
    actor_critic = make_sac_models(env=dummy_env, device=train_device)
    dummy_env.close()

    # 4) Run evaluation; returns metrics dict
    step = 0
    eval_metrics = evaluate_policy(actor_critic=actor_critic, cfg=cfg, train_device=train_device, logger=logger, step=step)

    # 5) Gather W&B run info for reporting
    run_url = None
    run_id = None
    try:
        if wandb.run is not None:
            run_url = getattr(wandb.run, "url", None)
            run_id = getattr(wandb.run, "id", None)
    except Exception:
        # Best-effort; do not fail the test on URL fetch issues
        pass

    # 6) Print concise summary for humans; assertions ensure function correctness
    print("\n=== evaluate_policy summary ===")
    print(f"wandb_run_url: {run_url}")
    print(f"wandb_run_id: {run_id}")
    print("metrics:")
    for k, v in eval_metrics.items():
        print(f"  {k}: {v}")

    # Basic sanity checks on returned metrics keys
    assert "eval/reward_mean" in eval_metrics
    assert "eval/reward_min" in eval_metrics
    assert "eval/reward_max" in eval_metrics
    assert "eval/episode_length" in eval_metrics

    # Ensure at least one episode completed or progressed
    assert eval_metrics["eval/episode_length"] > 0

    # 7) Finish and flush W&B to ensure upload completes
    logger.close()

    # Optionally return for ad-hoc script usage
    return {"wandb_url": run_url, "metrics": eval_metrics}


if __name__ == "__main__":
    # Allow running directly for quick smoke
    out = test_evaluate_policy_wandb_4envs()
    if out is not None:
        print("\n=== Direct Run Output ===")
        print(f"wandb_url: {out['wandb_url']}")
        for k, v in out["metrics"].items():
            print(f"{k}: {v}")
