"""
Environment utilities for TorchRL training with new environment system.
This version uses envs_new environments without YAML dependency.
"""
# Import to trigger Gymnasium registration
import envs_new  # noqa

import functools
import gymnasium as gym
from torchrl.envs import (Compose, DoubleToFloat, EnvCreator, GymWrapper, ParallelEnv, TransformedEnv)
from torchrl.envs.transforms import InitTracker, RewardSum, StepCounter
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
