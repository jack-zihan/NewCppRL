import os
from pathlib import Path

import torch
import torch.multiprocessing as mp

from rl_new.sac_cont_sy.env_utils import make_env_lambda
from rl_new.sac_cont_sy import env_utils as EU
from rl_new.sac_cont_sy.model_utils import make_sac_resnet_dual_models
from rl_new.sac_cont_sy.sac_utils import evaluate_policy_parallel
from torchrl.record.loggers import CSVLogger


class _LoggerCfg:
    def __init__(self):
        self.eval_episodes = 1
        self.eval_max_steps = 3
        self.eval_video = True
        self.eval_video_skip = 1
        self.show_progress = False
        self.mode = "offline"
        self.project = "dev-smoke"
        self.group = "dev"
        self.exp_name = "dev_eval_video_smoke"


class _EnvCfg:
    def __init__(self):
        self.env_id = "NewPasture-v6"
        self.env_kwargs = {
            "render_hif_lines": True,
            "use_multiscale": True,
            "use_global_features": False,
        }
    # provide dict-like get for env_utils compatibility
    def get(self, key, default=None):
        return getattr(self, key, default)


class _Cfg:
    def __init__(self):
        self.logger = _LoggerCfg()
        self.env = _EnvCfg()


def main():
    device = torch.device("cpu")
    # Avoid shared memory in sandbox
    try:
        mp.set_sharing_strategy("file_system")
    except Exception:
        pass
    cfg = _Cfg()

    # Build a temporary logger directory
    out_dir = Path("outputs/dev_eval_video_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(exp_name=cfg.logger.exp_name, log_dir=str(out_dir), video_format="mp4", video_fps=6)

    # Create a bootstrap env for spec, then build models
    bootstrap_env = make_env_lambda(env_id=cfg.env.env_id, device=device, from_pixels=False, **cfg.env.env_kwargs)
    modules = make_sac_resnet_dual_models(bootstrap_env, device=device)

    # Monkeypatch: use a testing wrapper that tolerates EnvCreator's rand_step without pred_ego_hif
    class PredEgoHIFInjectionEnvForTest(EU.PredEgoHIFInjectionEnv):
        def rand_step(self, tensordict):
            # 直接委托给底层环境的 rand_step（用于 EnvCreator 初始化）
            return self._env.rand_step(tensordict)

        def _step(self, tensordict):
            # rollout 阶段严格要求 pred_ego_hif；初始化阶段不会触发此逻辑
            return super()._step(tensordict)

    EU.PredEgoHIFInjectionEnv = PredEgoHIFInjectionEnvForTest

    # Run evaluation (this will record video via VideoRecorder and dump it)
    metrics = evaluate_policy_parallel(modules, cfg, logger, step=0, position=1)
    print("Eval metrics:", metrics)

    # Locate saved videos under logger directory
    # Expected location: log_dir / exp_name / videos / eval / video_0.mp4
    mp4s = list((out_dir / cfg.logger.exp_name).rglob("*.mp4"))
    if not mp4s:
        print("No videos found under:", out_dir)
        return
    print("Saved videos:")
    for p in mp4s:
        print(" -", p)

    print("NOTE: With eval_episodes=1 and from_pixels=True, each frame should be 800x1600 (GT|Pred).")


if __name__ == "__main__":
    main()
