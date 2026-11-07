import pytest
import torch
from types import SimpleNamespace

from rl_new.sac_cont_sy.env_utils import make_single_environment
from rl_new.sac_cont_sy.model_utils import make_sac_resnet_dual_models


class _EnvConf(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def _minimal_cfg(env_id="NewPasture-v6"):
    return SimpleNamespace(
        env=_EnvConf(env_id=env_id, env_kwargs={}),
        collector=SimpleNamespace(env_per_collector=1),
        logger=SimpleNamespace(eval_episodes=1, eval_video=False, eval_video_skip=1, eval_max_steps=64),
        seed=0,
    )


@pytest.mark.parametrize("env_id", ["NewPasture-v6"])
def test_actor_produces_pred_hif(env_id):
    # Use a single env (no HIF wrapper needed for shapes) to build models
    cfg = _minimal_cfg(env_id)
    env = make_single_environment(cfg, device="cpu")
    try:
        actor, critic = make_sac_resnet_dual_models(env=make_single_environment(cfg, device="cpu"), device="cpu", hif_decoder_type="two_stage")

        td = env.reset()
        # Ensure batch dim present for actor
        if td["observation"].ndim == 3:
            td["observation"] = td["observation"].unsqueeze(0)
        if td["vector"].ndim == 1:
            td["vector"] = td["vector"].unsqueeze(0)

        out = actor(td)
        assert "pred_ego_hif" in out.keys(), f"actor did not write 'pred_ego_hif'; keys={set(out.keys())}"

        pred = out["pred_ego_hif"]
        assert pred.ndim == 4 and pred.shape[1] == 2, f"unexpected pred_ego_hif shape: {tuple(pred.shape)}"
        assert pred.dtype == torch.float32
    finally:
        env.close()
