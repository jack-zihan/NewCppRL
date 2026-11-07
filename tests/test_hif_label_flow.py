import types
from types import SimpleNamespace

import torch
import pytest

from rl_new.sac_cont_sy.env_utils import make_env_lambda, HIFBidirectionalEnv


class _EnvConf(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def _minimal_cfg(env_id="NewPasture-v6"):
    # Build a minimal cfg-like namespace with only the fields env_utils needs
    return SimpleNamespace(
        env=_EnvConf(env_id=env_id, env_kwargs={}),
        hif=SimpleNamespace(enabled=True),
        collector=SimpleNamespace(env_per_collector=1),
        logger=SimpleNamespace(eval_episodes=1, eval_video=False, eval_video_skip=1, eval_max_steps=128),
        seed=0,
    )


@pytest.mark.parametrize("env_id", ["NewPasture-v6"])  # v6 carries HIF labels
def test_label_ego_hif_present_in_reset_and_next(env_id):
    cfg = _minimal_cfg(env_id)
    # Build a single GymWrapper env and wrap with our label injector
    base = make_env_lambda(env_id=cfg.env.env_id, device="cpu", from_pixels=False, **cfg.env.env_kwargs)
    env = HIFBidirectionalEnv(base)
    try:
        td = env.reset()
        # root (t=0) should contain label_ego_hif from _reset
        root_keys = set(td.keys())
        assert "label_ego_hif" in root_keys, f"reset root missing label_ego_hif; keys={root_keys}"

        # step once with a random action
        action = env.action_spec.rand()
        td.set("action", action)
        # Call the inner _step directly to bypass EnvBase bookkeeping
        next_td = env._step(td)

        next_keys = set(next_td.keys())
        assert (
            "label_ego_hif" in next_keys
        ), f"_step() return missing label_ego_hif; keys={next_keys}"

        # Basic sanity on shapes and dtype
        lab0 = td["label_ego_hif"]
        lab1 = next_td["label_ego_hif"]
        assert lab0.ndim in (3, 4) and lab0.shape[-3] == 3, f"unexpected label_0 shape: {tuple(lab0.shape)}"
        assert lab1.ndim in (3, 4) and lab1.shape[-3] == 3, f"unexpected label_1 shape: {tuple(lab1.shape)}"
        assert lab0.dtype == torch.float32 and lab1.dtype == torch.float32
    finally:
        env.close()
