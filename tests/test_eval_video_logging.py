import types
from dataclasses import dataclass, field
import sys

import torch
import pytest
from tensordict import TensorDict


class DummyEnv:
    def __init__(self, max_steps=5, h=32, w=32):
        self.t = 0
        self.max_steps = max_steps
        self.h = h
        self.w = w

    def reset(self):
        self.t = 0
        pixels = torch.randint(0, 255, (self.h, self.w, 3), dtype=torch.uint8)
        return TensorDict({"pixels": pixels}, batch_size=[])

    def step(self, td):
        self.t += 1
        done = self.t >= self.max_steps
        reward = torch.tensor(1.0)
        pixels = torch.randint(0, 255, (self.h, self.w, 3), dtype=torch.uint8)
        next_td = TensorDict(
            {
                "pixels": pixels,
                "next": {
                    "reward": reward,
                    "done": torch.tensor(done),
                    "completion_ratio": torch.tensor(min(1.0, self.t / self.max_steps)),
                },
            },
            batch_size=[],
        )
        return next_td

    def close(self):
        pass


class DummyActor:
    def __call__(self, td: TensorDict) -> TensorDict:
        # No-op actor: just attach a dummy action
        batch = td.batch_size
        td = td.clone()
        td["action"] = torch.zeros(*batch, 1)
        return td


class TestLogger:
    def __init__(self):
        self.scalars = []
        self.videos = []

    def log_scalar(self, key, value, step=None):
        self.scalars.append((key, float(value), step))

    def log_video(self, key, video_tensor, step=None):
        self.videos.append((key, video_tensor, step))


class FakeLocalVideoRecorder:
    def __init__(self, device=None, max_len=64, use_memmap=True, nrow=2, skip=1, fps=6, make_grid=True, center_crop=None):
        self.frames = []
        self.skip = max(1, int(skip))
        self.count = 0
        self.fps = fps
        self.idx = 0
        self.obs = None

    def apply(self, observation: torch.Tensor) -> torch.Tensor:
        # observation shape: [B, H, W, 3] or [B, 3, H, W]
        self.count += 1
        if self.count % self.skip != 0:
            return observation
        x = observation
        if x.ndim == 3:  # [H,W,3]
            x = x.unsqueeze(0)
        if x.shape[-1] == 3:  # [B,H,W,3] -> [B,3,H,W]
            x = x.permute(0, 3, 1, 2)
        x = x.to(torch.uint8)
        self.frames.append(x)
        self.idx += x.shape[0]
        # maintain a lightweight obs tensor for debug shape prints
        try:
            self.obs = torch.cat(self.frames, dim=0).unsqueeze(0)
        except Exception:
            self.obs = None
        return observation

    def dump(self, filepath: str | None = None):
        if not self.frames:
            return None
        vid = torch.cat(self.frames, dim=0)  # [T, C, H, W]
        vid = vid.unsqueeze(0)  # [1, T, C, H, W]
        # reset
        self.frames.clear()
        self.count = 0
        self.idx = 0
        return vid


@dataclass
class _LoggerCfg:
    eval_video: bool = True
    eval_episodes: int = 2
    eval_max_steps: int = 4
    eval_video_skip: int = 1
    show_progress: bool = False

    def __getitem__(self, k):
        return getattr(self, k)


@dataclass
class _Cfg:
    seed: int = 0
    logger: _LoggerCfg = field(default_factory=_LoggerCfg)


def test_eval_video_logging_monkeypatch(monkeypatch):
    # Stub heavy deps before importing sac_utils
    sys.modules.setdefault('envs_new', types.ModuleType('envs_new'))
    # Stub torchrl minimal surface used by sac_utils
    torchrl_pkg = types.ModuleType('torchrl')
    torchrl_utils = types.ModuleType('torchrl._utils')
    class _Logger:
        def info(self, *a, **k):
            pass
        def warning(self, *a, **k):
            pass
        def error(self, *a, **k):
            pass
    torchrl_utils.compile_with_warmup = lambda *a, **k: None
    torchrl_utils.logger = _Logger()
    torchrl_envs = types.ModuleType('torchrl.envs')
    torchrl_envs_utils = types.ModuleType('torchrl.envs.utils')
    class _ExplorationType:
        DETERMINISTIC = 0
    class _SetExploration:
        def __init__(self, *a, **k):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False
    torchrl_envs_utils.ExplorationType = _ExplorationType
    torchrl_envs_utils.set_exploration_type = lambda *a, **k: _SetExploration()
    torchrl_record = types.ModuleType('torchrl.record')
    class _VideoRecorder:
        def __init__(self, *a, **k):
            pass
        def dump(self):
            return None
    torchrl_record.VideoRecorder = _VideoRecorder
    tensordict_nn = types.ModuleType('tensordict.nn')
    class _CudaGraphModule:
        pass
    tensordict_nn.CudaGraphModule = _CudaGraphModule
    sys.modules['torchrl'] = torchrl_pkg
    sys.modules['torchrl._utils'] = torchrl_utils
    sys.modules['torchrl.envs'] = torchrl_envs
    sys.modules['torchrl.envs.utils'] = torchrl_envs_utils
    sys.modules['torchrl.record'] = torchrl_record
    sys.modules['tensordict.nn'] = tensordict_nn

    # Stub torchrl_utils_new.local_video_recorder before sac_utils import
    tru_pkg = types.ModuleType('torchrl_utils_new')
    tru_pkg.__path__ = []  # mark as package
    tru_lvr = types.ModuleType('torchrl_utils_new.local_video_recorder')
    class _StubLVR:
        def __init__(self, *a, **k):
            pass
        def apply(self, x):
            return x
        def dump(self, *a, **k):
            return None
    tru_lvr.LocalVideoRecorder = _StubLVR
    sys.modules['torchrl_utils_new'] = tru_pkg
    sys.modules['torchrl_utils_new.local_video_recorder'] = tru_lvr

    # Stub env_utils to return our DummyEnv without importing real module
    fake_env_utils = types.ModuleType('rl_new.sac_cont_sy.env_utils')
    fake_env_utils.make_single_environment = lambda cfg, device="cpu", seed=None, from_pixels=False: DummyEnv()
    fake_env_utils.make_environment = lambda *a, **k: (DummyEnv(), DummyEnv())
    sys.modules['rl_new.sac_cont_sy.env_utils'] = fake_env_utils

    import rl_new.sac_cont_sy.sac_utils as sac_utils
    monkeypatch.setattr(sac_utils, "LocalVideoRecorder", FakeLocalVideoRecorder)

    logger = TestLogger()
    cfg = _Cfg()
    actor_critic = [DummyActor(), None]

    step_i = 7
    metrics = sac_utils.evaluate_policy(actor_critic=actor_critic, cfg=cfg, train_device=torch.device("cpu"), logger=logger, step=step_i)

    # Metrics sanity
    assert "eval/reward_mean" in metrics
    assert "eval/episode_length" in metrics

    # Video logging assertions
    assert len(logger.videos) == 1
    key, vid, step = logger.videos[0]
    assert key == "eval/video"
    assert step == step_i
    assert isinstance(vid, torch.Tensor)
    assert vid.ndim == 5  # [1, T, C, H, W]
    assert vid.dtype == torch.uint8
