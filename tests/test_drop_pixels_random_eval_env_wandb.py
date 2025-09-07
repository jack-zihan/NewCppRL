#!/usr/bin/env python3
"""
Random-action evaluation with make_drop_pixels_eval_environment.
- 3x3 (9 envs) grid video to W&B
- No actor_critic; env.rollout uses env.rand_action
- Computes and logs eval metrics
"""
import os
import sys
import types
import numpy as np

REPO_ROOT = "/home/lzh/NewCppRL"
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

import torch
import wandb
from omegaconf import OmegaConf
from torchrl.envs.utils import ExplorationType, set_exploration_type

from rl_new.sac_cont_sy.env_utils import make_drop_pixels_eval_environment
from rl_new.sac_cont_sy.sac_utils import dump_video


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
    def __init__(self, project: str, group: str, name: str, config: dict, mode: str = "online"):
        if mode:
            os.environ["WANDB_MODE"] = mode
        self._run = wandb.init(project=project, group=group, name=name, config=config)

    def log_scalar(self, tag: str, value, step: int):
        wandb.log({tag: float(value) if hasattr(value, "__float__") else value}, step=step)

    def log_video(self, *args, **kwargs):
        if "video" in kwargs and "name" in kwargs:
            tag = kwargs.get("name")
            vid = kwargs.get("video")
            step = kwargs.get("step", None)
        else:
            tag = args[0]
            vid = args[1]
            step = args[2] if len(args) > 2 else kwargs.get("step", None)
        v = vid.detach().cpu() if isinstance(vid, torch.Tensor) else torch.as_tensor(vid).cpu()
        if v.ndim == 5 and v.shape[0] == 1:
            v = v[0]
        if v.dtype != torch.uint8:
            v = v.to(torch.uint8)
        wb_video = wandb.Video(v.numpy(), fps=6, format="mp4")
        wandb.log({tag: wb_video}, step=step)

    def close(self):
        try:
            wandb.finish()
        except Exception:
            pass


def run_test_random():
    print("\n=== DropPixels eval env W&B test (random policy, 3x3) ===")
    cfg = OmegaConf.load(os.path.join(REPO_ROOT, "rl_new/sac_cont_sy/config-async-server.yaml"))
    cfg.seed = 42
    cfg.logger.eval_episodes = 9
    cfg.logger.eval_max_steps = 300
    cfg.logger.eval_video = True
    cfg.logger.eval_video_skip = 10

    logger = WandbLoggerShim(
        project=cfg.logger.project_name,
        group=cfg.logger.group_name,
        name="drop_pixels_eval_env_random",
        config=dict(cfg),
        mode="online",
    )

    eval_device = torch.device("cpu")
    _, eval_env = make_drop_pixels_eval_environment(cfg=cfg, logger=logger, train_device=eval_device, eval_device=eval_device)
    eval_env.set_seed(cfg.seed)

    # Random actions: policy=None -> env.rand_action
    with set_exploration_type(ExplorationType.RANDOM):
        td = eval_env.rollout(
            max_steps=cfg.logger.eval_max_steps,
            policy=None,
            auto_cast_to_device=True,
            break_when_all_done=True,
        )

    # Upload recorded video
    eval_env.apply(dump_video)

    # Metrics
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

    for k, v in metrics.items():
        logger.log_scalar(k, v, step=0)

    run_url = getattr(wandb.run, "url", None)
    run_id = getattr(wandb.run, "id", None)
    print(f"wandb_run_url: {run_url}")
    print(f"wandb_run_id: {run_id}")
    print("metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    eval_env.close()
    logger.close()
    return {"wandb_url": run_url, "metrics": metrics}


if __name__ == "__main__":
    out = run_test_random()
    if out:
        print("\n=== Done ===")

