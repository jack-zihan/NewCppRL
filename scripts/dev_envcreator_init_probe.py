import traceback
import torch
from tensordict import TensorDict

from rl_new.sac_cont_sy.env_utils import make_env_lambda, PredEgoHIFInjectionEnv


def main():
    # Build a single wrapped env exactly like make_drop_pixels_eval_environment does for v6
    base = make_env_lambda(env_id="NewPasture-v6", device="cpu", from_pixels=False)
    env = PredEgoHIFInjectionEnv(base)

    # Simulate EnvCreator.init_ calling rand_step/step BEFORE any policy forward
    # i.e., tensordict does not contain 'pred_ego_hif' or even 'action'
    td = TensorDict({}, batch_size=[])
    try:
        # TorchRL would call rand_step(td) -> step(td) -> our _step(td)
        env.rand_step(td)
    except Exception as e:
        print("Caught exception during rand_step without pred_ego_hif:")
        traceback.print_exc()
        return

    print("No exception raised (unexpected) – wrapper did not fail-fast as designed.")


if __name__ == "__main__":
    main()

