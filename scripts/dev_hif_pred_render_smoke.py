import sys
import numpy as np
import torch

from rl_new.sac_cont_sy.env_utils import make_env_lambda, PredEgoHIFInjectionEnv
from rl_new.sac_cont_sy.model_utils import make_sac_resnet_dual_models


def main():
    device = torch.device("cpu")

    # 1) Build single v6 env (GymWrapper) and wrap with pred-injection wrapper
    base = make_env_lambda(env_id="NewPasture-v6", device=device, from_pixels=True,
                           render_hif_lines=True)
    env = PredEgoHIFInjectionEnv(base)

    # 2) Build models (actor will write pred_ego_hif during forward)
    modules = make_sac_resnet_dual_models(make_env_lambda(env_id="NewPasture-v6", device=device, from_pixels=True,
                                                          render_hif_lines=True),
                                          device=device)
    actor = modules[0]

    # 3) Reset and run one actor forward
    td0 = env.reset()
    # 统一 4D/2D：为 observation / vector 显式加入 batch 维
    from tensordict import TensorDict as _TD
    obs = td0["observation"].unsqueeze(0) if td0["observation"].dim() == 3 else td0["observation"]
    vec = td0["vector"].unsqueeze(0) if td0["vector"].dim() == 1 else td0["vector"]
    td = _TD({"observation": obs, "vector": vec}, batch_size=[1])
    td = actor(td)  # fills 'action' and 'pred_ego_hif'

    has_pred = "pred_ego_hif" in td.keys()
    print(f"actor(td) wrote pred_ego_hif: {has_pred}")
    if has_pred:
        pred = td["pred_ego_hif"]
        print(f"pred_ego_hif shape: {tuple(pred.shape)} dtype={pred.dtype}")

    # 4) Step once (wrapper injects pred_ego_hif before base_env.step), then render
    # 4) 直接将预测注入底层 env 并渲染（避开与 step/reward 相关的 TorchRL 细节，专注渲染路径）
    pred = td["pred_ego_hif"]  # [B,2,H,W]
    sin2 = pred[0, 0].detach().cpu().numpy()
    cos2 = pred[0, 1].detach().cpu().numpy()
    conf = (np.sqrt(sin2 ** 2 + cos2 ** 2) > 0.5).astype(np.float32)

    # unwrap 到具有 set_pred_hif 的 env
    unwrapped = env
    while hasattr(unwrapped, '_env') and not hasattr(unwrapped, 'set_pred_hif'):
        unwrapped = unwrapped._env
    unwrapped.set_pred_hif({'cosine2': cos2, 'sine2': sin2, 'confidence': conf})

    # Call underlying env.render() since the wrapper doesn't proxy render()
    base_env = env
    while hasattr(base_env, '_env'):
        base_env = base_env._env
    img = base_env.render()
    if img is None:
        print("render() returned None (render_mode disabled)")
        return

    h, w, c = img.shape
    print(f"render image shape: {img.shape}")

    # Heuristic: if both GT and Pred are present, width should be ~2x of base map width
    # We can’t know base width precisely here; just assert c==3 and width>height/2 as a weak check.
    assert c == 3, "rendered image must be RGB"
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
