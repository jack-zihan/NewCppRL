"""
Environment utilities for TorchRL training with new environment system.
This version uses envs_new environments without YAML dependency.
"""
# Import to trigger Gymnasium registration
import envs_new  # noqa

import functools
import torch
import gymnasium as gym
from torchrl.envs import (Compose, DoubleToFloat, EnvCreator, GymWrapper, ParallelEnv, TransformedEnv)
from torchrl.envs.transforms import InitTracker, RewardSum, StepCounter, Transform
from torchrl.record import VideoRecorder


# ====================================================================
# Environment utils - New implementation without YAML
# --------------------------------------------------------------------

def make_env_lambda(env_id="NewPasture-v2", device="cpu", from_pixels=False, **env_kwargs):
    """
    Create a single environment instance.

    Args:
        env_id: Gymnasium environment ID (NewPasture-v1/v2/v3/v4/v5)
        device: Device to run the environment on
        from_pixels: Whether to use pixel observations
        **env_kwargs: Additional keyword arguments passed to the environment
            Examples:
            - use_apf: bool (for v2)
            - use_multiscale: bool (for v3)
            - map_dir: str (custom map directory)
            - num_obstacles_range: tuple (obstacle configuration)
            - reward_*: float (reward coefficients)

    Returns:
        GymWrapper: Wrapped environment ready for TorchRL
    """
    # Create environment with flexible configuration
    env = gym.make(env_id, render_mode='rgb_array' if from_pixels else None, **env_kwargs)
    return GymWrapper(env, device=device, from_pixels=from_pixels, pixels_only=False)  # Wrap for TorchRL compatibility


def apply_env_transforms(env):
    """Apply transforms. No max_episode_steps since env handles it."""
    return TransformedEnv(env, Compose(InitTracker(), StepCounter(), DoubleToFloat(), RewardSum(), ))


def make_environment(cfg, logger=None, train_device="cpu", eval_device="cpu"):
    """Make train and eval environments. Super simple - just pass cfg.env directly."""

    # 训练环境提前参数绑定
    partial = functools.partial(make_env_lambda, env_id=cfg.env.env_id, device=train_device,
                                from_pixels=False, **(cfg.env.get('env_kwargs') or {}))
    parallel_env = ParallelEnv(cfg.collector.env_per_collector, EnvCreator(partial), serial_for_single=True)  # 并行训练环境
    parallel_env.set_seed(cfg.seed)
    train_env = apply_env_transforms(parallel_env)

    # 验证环境提前绑定参数
    partial_eval = functools.partial(make_env_lambda, env_id=cfg.env.env_id, device=eval_device,
                                     from_pixels=cfg.logger.eval_video,
                                     **(cfg.env.get('env_kwargs') or {}))  # 与训练环境不同的是from_pixels
    trsf_clone = train_env.transform.clone()
    if cfg.logger.eval_video: trsf_clone.insert(0, VideoRecorder(logger, tag="rendering/test", in_keys=["pixels"],
                                                                 make_grid=True, skip=cfg.logger.eval_video_skip))
    eval_env = TransformedEnv(
        ParallelEnv(cfg.logger.eval_episodes, EnvCreator(partial_eval), serial_for_single=True),
        trsf_clone)
    return train_env, eval_env


def make_train_environment(cfg, device="cpu"):
    """Make training environment only."""
    partial = functools.partial(make_env_lambda, env_id=cfg.env.env_id, device=device, from_pixels=False,
                                **(cfg.env.get('env_kwargs') or {}))
    parallel_env = ParallelEnv(cfg.collector.env_per_collector, EnvCreator(partial), serial_for_single=True)
    parallel_env.set_seed(cfg.seed)
    return apply_env_transforms(parallel_env)


def make_single_environment(cfg, device="cpu", seed=None, from_pixels=False):
    # 创建基础环境
    env = make_env_lambda(env_id=cfg.env.env_id, device=device, from_pixels=from_pixels,
                          **(cfg.env.get('env_kwargs') or {}))  # 传递环境特定参数
    env = apply_env_transforms(env)  # 应用transforms（与训练环境保持一致）
    if seed is not None: env.set_seed(seed)  # 设置种子
    return env


# ====================================================================
# Evaluation-only env with VideoRecorder + DropPixels
# --------------------------------------------------------------------
def make_drop_pixels_eval_environment(cfg, logger=None, eval_device="cpu"):
    """
    构建仅用于评估的环境：
    - 环境放在 eval_device（建议 CPU），以便 rollouts 的 TensorDict 堆叠在内存而非显存。
    - 在 Transform 链中先接入 VideoRecorder（录制 2x2 网格视频），随后接入 DropPixels（剔除 pixels），这样既保留视频上传，又避免像素随时间堆叠带来的内存压力。
    接口与 make_environment 保持一致，返回 (None, eval_env)。
    """

    class DropPixels(Transform):
        """在每个 step/reset 之后剔除 pixels 相关键，减少内存占用。

        删除的键包括：'pixels' 与 ('next','pixels')。
        """

        def _call(self, tensordict):
            tensordict.pop("pixels", None)  # 当前步像素（若存在）
            try:
                tensordict["next"].pop("pixels", None)
            except Exception:
                pass
            return tensordict

        def _reset(self, tensordict, tensordict_reset):
            tensordict_reset.pop("pixels", None)  # reset 返回的起始观测也剔除像素
            return tensordict_reset

    class KeepLastPixels(Transform):
        """在 VideoRecorder 之前缓存每个并行环境的最后有效像素，并在 done/黑帧时复用上一帧，避免黑屏, 仅依赖 'pixels' 与可选的 'done' 键；不改变 spec，仅替换当步返回的像素内容。"""

        def __init__(self):
            super().__init__(in_keys=[])
            self._last = None

        def _maybe_init(self, pix: torch.Tensor):
            if self._last is None or self._last.shape != pix.shape or self._last.device != pix.device:
                self._last = torch.zeros_like(pix)

        def _replace_mask(self, tensordict, pix: torch.Tensor) -> torch.Tensor:
            b = pix.shape[0] if pix.ndim > 0 else 1
            flat = pix.reshape(b, -1)
            if pix.dtype == torch.uint8:
                sums = flat.to(torch.int64).sum(dim=1)
            else:
                sums = flat.abs().sum(dim=1)
            zero_mask = sums == 0
            done = tensordict.get("done", None)
            if isinstance(done, torch.Tensor):
                dflat = done.reshape(b, -1)
                done_mask = dflat.any(dim=1)
            else:
                done_mask = torch.zeros(b, dtype=torch.bool, device=pix.device)
            return zero_mask | done_mask

        def _call(self, tensordict):
            pix = tensordict.get("pixels", None)
            if not isinstance(pix, torch.Tensor):
                return tensordict
            self._maybe_init(pix)
            mask = self._replace_mask(tensordict, pix)
            if mask.any():
                new_pix = pix.clone()
                new_pix[mask] = self._last[mask]
                tensordict.set("pixels", new_pix)
                pix = new_pix
            valid = ~mask if mask.ndim > 0 else ~mask
            if valid.any():
                self._last[valid] = pix[valid]
            return tensordict

        def _reset(self, tensordict, tensordict_reset):
            pix = tensordict_reset.get("pixels", None)
            if isinstance(pix, torch.Tensor):
                self._maybe_init(pix)
                self._last.copy_(pix)
            else:
                self._last = None
            return tensordict_reset

    # 1) 构建并行评估环境（仅评估，不构建训练环境）
    partial_eval = functools.partial(make_env_lambda, env_id=cfg.env.env_id, device=eval_device,
                                     from_pixels=cfg.logger.eval_video, **(cfg.env.get("env_kwargs") or {}))
    eval_parallel = ParallelEnv(cfg.logger.eval_episodes, EnvCreator(partial_eval), serial_for_single=True)

    # 2) 组装 Transform 链
    trsf = Compose(InitTracker(), StepCounter(), DoubleToFloat(), RewardSum())

    if cfg.logger.eval_video and logger is not None: # KeepLastPixels -> VideoRecorder -> DropPixels
        trsf.insert(0, KeepLastPixels())
        trsf.insert(1, VideoRecorder(logger=logger, tag="eval/video", in_keys=["pixels"],
                                     make_grid=True, skip=cfg.logger.eval_video_skip))
        trsf.insert(2, DropPixels())
    else:
        # 未开启视频录制也插入 KeepLastPixels + DropPixels（防御性）
        trsf.insert(0, KeepLastPixels())
        trsf.insert(1, DropPixels())

    # 3) 构建最终评估环境
    eval_env = TransformedEnv(eval_parallel, trsf)
    return None, eval_env

