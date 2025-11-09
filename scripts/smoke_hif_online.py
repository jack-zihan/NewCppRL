#!/usr/bin/env python
"""
Smoke test for online HIF auxiliary flow on v6 (no optimizer step).

What it checks:
- Build v6 env with HIF Transform enabled (labels attached on step)
- Build ResNet-FPN dual models
- Step env once with a dummy action to produce HIF labels in tensordict
- Run recon forward (batched), compute HIF loss, do backward (no step)
- Verify encoder has non-zero gradients from HIF backward

Usage:
  python scripts/smoke_hif_online.py
"""

import traceback
import torch
from tensordict import TensorDict

from rl_new.sac_cont_sy.env_utils import make_single_environment
from rl_new.sac_cont_sy.model_utils import make_sac_resnet_dual_models
from torchrl_utils.model.resnet_fpn_dual import HIFReconstructionLoss


class EnvObj:
    def __init__(self, env_id):
        self.env_id = env_id
        self.env_kwargs = {}

    def get(self, k, default=None):
        return getattr(self, k, default)


class HIFCfg:
    def __init__(self):
        self.enabled = True
        self.online_auxiliary = True


class Cfg:
    pass


def make_cfg_v6():
    cfg = Cfg()
    cfg.env = EnvObj("NewPasture-v6")
    cfg.hif = HIFCfg()
    cfg.seed = 0
    return cfg


def main():
    try:
        device = torch.device("cpu")
        cfg = make_cfg_v6()

        # Build env with HIF Transform enabled
        env = make_single_environment(cfg, device=device)

        # Build models (actor, critic, recon)
        modules = make_sac_resnet_dual_models(env=make_single_environment(cfg, device=device), device=str(device))
        actor, critic, recon = modules
        for m in [actor, critic, recon]:
            m.eval()

        # Reset + up to two steps to attach HIF labels via Transform
        td = env.reset()
        for _ in range(2):
            # Build a dummy action within range (zeros)
            act_shape = env.action_spec.shape
            action = torch.zeros(*act_shape, dtype=torch.float32)
            td.set("action", action)
            td = env.step(td)

            # look for labels either in root or next
            has_root = all(k in td.keys() for k in ("hif_label_cos2","hif_label_sin2","hif_label_conf"))
            has_next = "next" in td.keys() and all(k in td["next"].keys() for k in ("hif_label_cos2","hif_label_sin2","hif_label_conf"))
            if has_root or has_next:
                break

        # Prefer next obs for recon
        if "next" in td.keys():
            td_obs = td["next"]
        else:
            td_obs = td

        # Extract labels from root if present, else from next; if still not found, try maps_dict fallback
        def _stack(c2, s2, cf):
            return torch.stack([s2, c2], dim=1), cf

        cos2_t = td.get("hif_label_cos2", None)
        sin2_t = td.get("hif_label_sin2", None)
        conf_t = td.get("hif_label_conf", None)
        if cos2_t is not None and sin2_t is not None and conf_t is not None:
            target_hif, conf = _stack(cos2_t.unsqueeze(0), sin2_t.unsqueeze(0), conf_t.unsqueeze(0))
        elif "next" in td.keys():
            cos2_t = td["next"].get("hif_label_cos2", None)
            sin2_t = td["next"].get("hif_label_sin2", None)
            conf_t = td["next"].get("hif_label_conf", None)
            if cos2_t is not None and sin2_t is not None and conf_t is not None:
                target_hif, conf = _stack(cos2_t.unsqueeze(0), sin2_t.unsqueeze(0), conf_t.unsqueeze(0))
            else:
                # Fallback: dig maps_dict from underlying gym env
                cur = env
                max_hops = 6
                maps = None
                while max_hops > 0 and cur is not None:
                    if hasattr(cur, 'maps_dict'):
                        maps = getattr(cur, 'maps_dict')
                        break
                    cur = getattr(cur, 'env', None)
                    max_hops -= 1
                assert maps is not None, "Cannot locate maps_dict for fallback"
                cos2 = torch.from_numpy(maps['ego_hif_cos2']).unsqueeze(0)
                sin2 = torch.from_numpy(maps['ego_hif_sin2']).unsqueeze(0)
                conf = torch.from_numpy(maps['ego_hif_conf']).unsqueeze(0)
                target_hif, conf = _stack(cos2, sin2, conf)

        cos2 = td["hif_label_cos2"].unsqueeze(0)
        sin2 = td["hif_label_sin2"].unsqueeze(0)
        conf = td["hif_label_conf"].unsqueeze(0)
        target_hif = torch.stack([sin2, cos2], dim=1)  # [B,2,H,W]

        # Recon forward (batched)
        td_in = TensorDict({"observation": td_obs["observation"].unsqueeze(0)}, batch_size=[1])
        td_out = recon(td_in)
        pred = td_out["hif_pred"]
        assert pred.ndim == 4 and pred.shape[1] == 2

        # Prepare encoder params for grad check
        recon_core = recon.module  # HIFReconModule
        encoder = recon_core.encoder
        for p in encoder.parameters():
            if p.grad is not None:
                p.grad = None

        # HIF loss backward (no step)
        criterion = HIFReconstructionLoss(lambda_cosine=1.0, lambda_tv=1e-5).to(device)
        loss, parts = criterion(pred, target_hif, conf)
        loss.backward()

        # Check encoder grads present and non-zero
        has_grad = False
        total_norm = 0.0
        for p in encoder.parameters():
            if p.grad is not None:
                g = p.grad.data
                total_norm += float(g.norm().cpu())
                has_grad = has_grad or (g.abs().sum() > 0)

        assert has_grad and total_norm > 0, "Encoder did not receive HIF gradients"

        print("\nHIF online smoke: PASS\n  loss=%.6f  cosine=%.6f  tv=%.6f  enc_grad_norm=%.6f" % (
            float(loss), parts.get("hif_cosine_loss", 0.0), parts.get("hif_tv_loss", 0.0), total_norm))

    except Exception as e:
        print("\nHIF online smoke: FAIL:", e)
        traceback.print_exc()


if __name__ == "__main__":
    main()
