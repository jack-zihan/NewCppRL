import os
from pathlib import Path

import cv2
import numpy as np
import torch
from tensordict import TensorDict

import gymnasium as gym
from rl_new.sac_cont_sy.env_utils import make_env_lambda
from rl_new.sac_cont_sy.model_utils import make_sac_resnet_dual_models


def main():
    device = torch.device("cpu")

    # 1) Build single v6 env with pixels enabled so env.render() works
    env = gym.make(
        "NewPasture-v6",
        render_mode="rgb_array",
        render_hif_lines=True,
        use_multiscale=True,
        use_global_features=False,
    )

    # 2) Build models (actor includes HIF recon head and writes pred_ego_hif)
    modules = make_sac_resnet_dual_models(
        make_env_lambda(env_id="NewPasture-v6", device=device, from_pixels=False,
                         render_hif_lines=True, use_multiscale=True, use_global_features=False),
        device=device,
    )
    actor = modules[0]

    # 3) Reset env; video writer will be created on first frame after pred injection
    obs, info = env.reset()

    out_dir = Path("outputs/dev_eval_video_manual")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "video_manual.mp4"
    writer = None

    # 4) Roll a short episode manually (no multiprocessing, no transforms)
    max_steps = 20
    steps = 0
    done = False

    while not done and steps < max_steps:
        # build batched TD for actor
        obs_t = torch.from_numpy(obs["observation"]).unsqueeze(0)
        vec_t = torch.from_numpy(obs["vector"]).unsqueeze(0)
        in_td = TensorDict({"observation": obs_t, "vector": vec_t}, batch_size=[1])

        with torch.no_grad():
            out_td = actor(in_td)

        # inject predicted HIF to env for this frame
        pred = out_td["pred_ego_hif"]  # [1,2,H,W]
        sin2 = pred[0, 0].detach().cpu().numpy()
        cos2 = pred[0, 1].detach().cpu().numpy()
        conf = (np.sqrt(sin2 ** 2 + cos2 ** 2) > 0.5).astype(np.float32)

        # unwrap and set
        env.unwrapped.set_pred_hif({"cosine2": cos2, "sine2": sin2, "confidence": conf})

        # render and write video frame BEFORE stepping (ensures pred present for this frame)
        frame = env.render()
        if writer is None:
            h, w, c = frame.shape
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 6, (w, h))
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # step the env with action from actor (squeeze batch)
        action = out_td["action"][0].detach().cpu().numpy()
        obs, reward, terminated, truncated, info = env.step(action)

        done = bool(terminated) or bool(truncated)
        steps += 1

    if writer is not None:
        writer.release()
        print(f"Saved video: {out_path}")
    else:
        print("No frames were written.")


if __name__ == "__main__":
    main()
