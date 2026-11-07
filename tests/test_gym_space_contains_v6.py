import numpy as np
import gymnasium as gym
import envs_new  # ensure envs are registered


def _space_key_contains(space, obs_dict):
    """Return list of (key, ok, min, max, dtype, shape) for diagnostics."""
    results = []
    assert set(space.spaces.keys()) == set(obs_dict.keys())
    for k, subspace in space.spaces.items():
        x = obs_dict[k]
        ok = subspace.contains(x)
        mn = float(np.min(x)) if hasattr(x, 'min') else None
        mx = float(np.max(x)) if hasattr(x, 'max') else None
        results.append((k, ok, mn, mx, getattr(x, 'dtype', None), getattr(x, 'shape', None)))
    return results


def test_v6_space_contains_precise():
    env = gym.make('NewPasture-v6')
    try:
        for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            obs, info = env.reset(seed=seed)
            ok = env.observation_space.contains(obs)
            if not ok:
                diags = _space_key_contains(env.observation_space, obs)
                msg = "; ".join(
                    f"{k}: ok={ok_key}, min={mn:.6g}, max={mx:.6g}, dtype={dt}, shape={sh}"
                    for (k, ok_key, mn, mx, dt, sh) in diags
                )
                raise AssertionError(f"reset obs not in space for seed={seed}; {msg}")

            # also check one step
            a = env.action_space.sample()
            obs2, r, d, t, inf = env.step(a)
            ok2 = env.observation_space.contains(obs2)
            if not ok2:
                diags = _space_key_contains(env.observation_space, obs2)
                msg = "; ".join(
                    f"{k}: ok={ok_key}, min={mn:.6g}, max={mx:.6g}, dtype={dt}, shape={sh}"
                    for (k, ok_key, mn, mx, dt, sh) in diags
                )
                raise AssertionError(f"step obs not in space for seed={seed}; {msg}")
    finally:
        env.close()

