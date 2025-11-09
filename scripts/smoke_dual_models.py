#!/usr/bin/env python
"""
Smoke test for ResNet-FPN dual-head models on v4/v5/v6 environments.

Runs a minimal rollout loop (CPU) to verify:
- make_single_environment() creates a working TorchRL env
- make_sac_resnet_dual_models() returns [actor, critic, recon]
- actor/critic/recon forward pass succeed with batched inputs
- env.step() accepts the action produced by actor

Usage:
  python scripts/smoke_dual_models.py
"""

import traceback
import torch
from tensordict import TensorDict

from rl_new.sac_cont_sy.env_utils import make_single_environment
from rl_new.sac_cont_sy.model_utils import make_sac_resnet_dual_models
from torchrl.envs.utils import ExplorationType, set_exploration_type


class EnvObj:
    def __init__(self, env_id):
        self.env_id = env_id
        self.env_kwargs = {}

    def get(self, k, default=None):
        return getattr(self, k, default)


class Cfg:
    pass


def make_min_cfg(env_id: str) -> Cfg:
    cfg = Cfg()
    cfg.env = EnvObj(env_id)
    cfg.seed = 0
    return cfg


def run_one(env_id: str, steps: int = 3) -> bool:
    print(f"\n=== Smoke: {env_id} ===")
    try:
        cfg = make_min_cfg(env_id)
        env = make_single_environment(cfg, device="cpu")

        # Build models (actor, critic, recon) on CPU
        modules = make_sac_resnet_dual_models(env=make_single_environment(cfg, device="cpu"), device="cpu")
        actor, critic, recon = modules
        for m in [actor, critic, recon]:
            m.eval()

        with set_exploration_type(ExplorationType.RANDOM):
            td = env.reset()
            for _ in range(steps):
                # Batched copy for model forward
                td_b = TensorDict(
                    {
                        "observation": td["observation"].unsqueeze(0),
                        "vector": td["vector"].unsqueeze(0),
                    },
                    batch_size=[1],
                )

                # Actor forward
                td_b = actor(td_b)
                assert "action" in td_b.keys(), "actor should write 'action'"

                # Critic forward (ensure no crash)
                _ = critic(td_b)

                # Recon forward
                td_in = TensorDict({"observation": td_b["observation"]}, batch_size=td_b.batch_size)
                td_out = recon(td_in)
                hp = td_out["hif_pred"]
                assert hp.ndim == 4 and hp.shape[1] == 2, "hif_pred must be [B,2,H,W]"

                # Step environment with unbatched action
                td.set("action", td_b["action"].squeeze(0))
                td = env.step(td)
                if "next" in td.keys():
                    td = td["next"]

        env.close()
        print(f"PASS: {env_id}")
        return True
    except Exception as e:
        print(f"FAIL: {env_id}: {e}")
        traceback.print_exc()
        return False


def main():
    ok = True
    for env_id in ["NewPasture-v4", "NewPasture-v5", "NewPasture-v6"]:
        ok &= run_one(env_id, steps=3)
    print("\nALL OK" if ok else "\nSOME TESTS FAILED")


if __name__ == "__main__":
    main()

