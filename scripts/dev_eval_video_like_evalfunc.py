from pathlib import Path
from functools import partial

import torch
from torchrl.envs import Compose, TransformedEnv
from torchrl.envs.transforms import InitTracker, StepCounter, DoubleToFloat, Transform
from torchrl.record import VideoRecorder
from torchrl.record.loggers import CSVLogger
from torchrl.envs.utils import ExplorationType, set_exploration_type

from rl_new.sac_cont_sy.env_utils import (
    make_env_lambda,
    PredEgoHIFInjectionEnv,
    Steps95ToDoneCounter,
)
from rl_new.sac_cont_sy.model_utils import make_sac_resnet_dual_models


def main():
    device = torch.device("cpu")
    # 1) build actor (policy only)
    bootstrap_env = make_env_lambda(env_id="NewPasture-v6", device=device, from_pixels=False,
                                    render_hif_lines=True, use_multiscale=True, use_global_features=False)
    modules = make_sac_resnet_dual_models(bootstrap_env, device=device)
    actor = modules[0]

    # 2) single env with pred-injection wrapper (no ParallelEnv)
    base_env = make_env_lambda(env_id="NewPasture-v6", device=device, from_pixels=True,
                               render_hif_lines=True, use_multiscale=True, use_global_features=False)
    env = PredEgoHIFInjectionEnv(base_env)

    # 3) transforms: KeepLastPixels -> VideoRecorder -> DropPixels, then stats
    class DropPixels(Transform):
        def _call(self, tensordict):
            tensordict.pop("pixels", None)
            if "next" in tensordict.keys():
                tensordict["next"].pop("pixels", None)
            return tensordict
        def _reset(self, tensordict, tensordict_reset):
            tensordict_reset.pop("pixels", None)
            return tensordict_reset

    class KeepLastPixels(Transform):
        def __init__(self):
            super().__init__(in_keys=[])
            self._last = None
        def _maybe_init(self, pix):
            if self._last is None or self._last.shape != pix.shape or self._last.device != pix.device:
                self._last = torch.zeros_like(pix)
        def _replace_mask(self, tensordict, pix):
            b = pix.shape[0] if pix.ndim > 0 else 1
            flat = pix.reshape(b, -1)
            sums = flat.to(torch.int64).sum(dim=1) if pix.dtype == torch.uint8 else flat.abs().sum(dim=1)
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

    out_dir = Path("outputs/dev_eval_video_like_evalfunc")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_logger = CSVLogger(exp_name="exp", log_dir=str(out_dir), video_format="mp4", video_fps=6)

    tr = Compose(
        KeepLastPixels(),
        VideoRecorder(logger=csv_logger, tag="eval/video", in_keys=["pixels"], make_grid=True, skip=1),
        DropPixels(),
        InitTracker(),
        StepCounter(max_steps=20),
        Steps95ToDoneCounter(),
        DoubleToFloat(),
    )
    eval_env = TransformedEnv(env, tr)

    # 4) rollout with actor; pred_ego_hif is injected by wrapper in _step
    from rl_new.sac_cont_sy.sac_utils import dump_video
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        td = eval_env.rollout(max_steps=20, policy=actor, auto_cast_to_device=True, break_when_all_done=True)
    eval_env.apply(partial(dump_video, step=0))

    # locate mp4
    vp = out_dir / "exp" / "videos" / "eval" / "video_0.mp4"
    print(f"Saved video: {vp}")


if __name__ == "__main__":
    import torch
    main()

