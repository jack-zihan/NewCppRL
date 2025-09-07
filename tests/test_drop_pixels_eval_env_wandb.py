#!/usr/bin/env python3
"""
Test make_drop_pixels_eval_environment: ensure video is uploaded to W&B and metrics are computed
without GPU memory growth (rollout tensordict stays on CPU).

This does not modify existing code. It:
- Creates W&B run (online)
- Builds eval env via make_drop_pixels_eval_environment (CPU), with VideoRecorder + DropPixels
- Builds an untrained actor_critic on GPU (if available)
- Runs rollout with auto_cast_to_device=True and break_when_all_done=True
- Uploads the video via env.apply(dump_video)
- Prints W&B URL and metrics
"""
import os
import sys
import types
import numpy as np

# Repo root
REPO_ROOT = "/home/lzh/NewCppRL"
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

import torch
import wandb
from omegaconf import OmegaConf

from torchrl.envs.utils import ExplorationType, set_exploration_type

from rl_new.sac_cont_sy.env_utils import (
    make_drop_pixels_eval_environment,
    make_single_environment,
)
from rl_new.sac_cont_sy.model_utils import make_sac_models


# --- Minimal cupy stubs to avoid GPU-APF hard dep in tests ---
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
        class _Dummy:
            def __init__(self, base):
                self._base = np.asarray(base)
            def astype(self, dtype):
                return self
            def get(self):
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


class WandbLoggerShim:
    """Logger shim compatible with torchrl.record.VideoRecorder.log_video signature.

    Supports both styles:
    - log_video(name=..., video=..., step=...)
    - log_video(tag, vid, step)
    """

    def __init__(self, project: str, group: str, name: str, config: dict, mode: str = "online"):
        if mode:
            os.environ["WANDB_MODE"] = mode
        self._run = wandb.init(project=project, group=group, name=name, config=config)

    def log_scalar(self, tag: str, value, step: int):
        try:
            wandb.log({tag: float(value) if hasattr(value, "__float__") else value}, step=step)
        except Exception:
            wandb.log({tag: value}, step=step)

    def log_video(self, *args, **kwargs):
        # Accept both positional and keyword styles
        if "video" in kwargs and "name" in kwargs:
            tag = kwargs.get("name")
            vid = kwargs.get("video")
            step = kwargs.get("step", None)
        else:
            # Expect (tag, vid[, step])
            tag = args[0]
            vid = args[1]
            step = args[2] if len(args) > 2 else kwargs.get("step", None)

        if isinstance(vid, torch.Tensor):
            v = vid.detach().cpu()
        else:
            v = torch.as_tensor(vid).cpu()
        if v.ndim == 5 and v.shape[0] == 1:
            v = v[0]
        if v.dtype != torch.uint8:
            v = v.to(torch.uint8)
        wb_video = wandb.Video(v.numpy(), fps=6, format="mp4")
        log_dict = {tag: wb_video}
        if step is not None:
            wandb.log(log_dict, step=step)
        else:
            wandb.log(log_dict)

    def close(self):
        try:
            wandb.finish()
        except Exception:
            pass


def run_test():
    print("\n=== DropPixels eval env W&B test ===")
    cfg = OmegaConf.load(os.path.join(REPO_ROOT, "rl_new/sac_cont_sy/config-async-server.yaml"))
    # Short, deterministic eval
    cfg.seed = 42
    cfg.logger.eval_episodes = 9  # use 3x3 grid
    cfg.logger.eval_max_steps = 300
    cfg.logger.eval_video = True
    cfg.logger.eval_video_skip = 10

    # Logger
    exp_name = "drop_pixels_eval_env_test"
    logger = WandbLoggerShim(
        project=cfg.logger.project_name,
        group=cfg.logger.group_name,
        name=exp_name,
        config=dict(cfg),
        mode="online",
    )

    # Policy device (GPU if available), env device CPU
    policy_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    eval_device = torch.device("cpu")

    # Build actor_critic from a non-pixel single env to get specs
    dummy_env = make_single_environment(cfg, device=policy_device, from_pixels=False)
    actor_critic = make_sac_models(env=dummy_env, device=policy_device)
    dummy_env.close()

    # Build eval env with VideoRecorder + DropPixels on CPU
    _, eval_env = make_drop_pixels_eval_environment(
        cfg=cfg, logger=logger, train_device=eval_device, eval_device=eval_device
    )
    eval_env.set_seed(cfg.seed)

    # Rollout with deterministic policy and break_when_all_done
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        td = eval_env.rollout(
            max_steps=cfg.logger.eval_max_steps,
            policy=actor_critic[0],
            auto_cast_to_device=True,
            break_when_all_done=True,
        )

    # Trigger video upload from VideoRecorder
    from rl_new.sac_cont_sy.sac_utils import dump_video
    eval_env.apply(dump_video)

    # Compute metrics (mirror evaluate_policy_parallel logic)
    rewards = td.get(("next", "episode_reward"))[:, -1].cpu().numpy()
    lengths = td.get(("next", "step_count"))[:, -1].cpu().numpy()
    metrics = {
        "eval/reward_mean": float(np.mean(rewards)),
        "eval/reward_std": float(np.std(rewards)),
        "eval/reward_min": float(np.min(rewards)),
        "eval/reward_max": float(np.max(rewards)),
        "eval/episode_length": float(np.mean(lengths)),
        "eval/episodes_completed": int(rewards.shape[0]),
    }
    if ("completion_ratio") in td.get("next").keys():
        cr = td.get(("next", "completion_ratio"))[:, -1].cpu().numpy()
        metrics["eval/completion_ratio"] = float(np.mean(cr))
        metrics["eval/completion_ratio_max"] = float(np.max(cr))

    # Log metrics to W&B
    for k, v in metrics.items():
        logger.log_scalar(k, v, step=0)

    # Print run URL and metrics
    run_url = getattr(wandb.run, "url", None)
    run_id = getattr(wandb.run, "id", None)
    print(f"wandb_run_url: {run_url}")
    print(f"wandb_run_id: {run_id}")
    print("metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Cleanup
    eval_env.close()
    logger.close()
    return {"wandb_url": run_url, "metrics": metrics}


if __name__ == "__main__":
    out = run_test()
    if out:
        print("\n=== Done ===")
