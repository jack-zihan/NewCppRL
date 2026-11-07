import numpy as np
import gymnasium as gym
import envs_new  # trigger Gym registration for NewPasture-v* envs
from envs_new.utils.image_utils import extract_ego_patch
import cv2


def _slice_vector_segments(vec: np.ndarray, L: int):
    """
    Return segments within the vector observation for v4/v6 layout:
    [coverage(1), time_prog(1), steer(L), speed(L), dx_ego(L), dy_ego(L), heading_sin(L), heading_cos(L)]
    """
    assert vec.ndim == 1, "vector must be 1D"
    idx = 0
    coverage = vec[idx]; idx += 1
    time_prog = vec[idx]; idx += 1
    steer = vec[idx: idx+L]; idx += L
    speed = vec[idx: idx+L]; idx += L
    dx = vec[idx: idx+L]; idx += L
    dy = vec[idx: idx+L]; idx += L
    hsin = vec[idx: idx+L]; idx += L
    hcos = vec[idx: idx+L]
    return {
        'coverage': np.array([coverage]),
        'time_prog': np.array([time_prog]),
        'steer': steer,
        'speed': speed,
        'dx_ego': dx,
        'dy_ego': dy,
        'heading_sin': hsin,
        'heading_cos': hcos,
    }


def _per_map_channel_ranges(env, obs_tensor: np.ndarray):
    """
    Compute min/max per logical map across all scales for multiscale observation.
    Assumes ObservationGenerator concatenates scales sequentially (scale0, scale1, scale2, scale3).
    """
    # Pull the exact obs_maps the base env would use (order matters)
    base = env.unwrapped
    obs_maps = base._get_observation_maps()
    map_names = list(obs_maps.keys())
    num_maps = len(map_names)

    C = obs_tensor.shape[0]
    assert C % num_maps == 0, f"channels {C} not divisible by num_maps {num_maps}"
    num_scales = C // num_maps

    results = []
    for i, name in enumerate(map_names):
        # gather all channels belonging to map i across scales
        ch_idxs = [i + s * num_maps for s in range(num_scales)]
        sub = obs_tensor[ch_idxs]
        results.append((name, float(np.nanmin(sub)), float(np.nanmax(sub)), bool(np.any(~np.isfinite(sub)))))
    return results


def _assert_in_range(name: str, arr: np.ndarray, lo: float, hi: float):
    assert np.all(np.isfinite(arr)), f"{name} contains non-finite values"
    mn = float(arr.min())
    mx = float(arr.max())
    assert mn >= lo - 1e-6 and mx <= hi + 1e-6, (
        f"{name} out of range: min={mn:.6f}, max={mx:.6f}, expected within [{lo},{hi}]"
    )


def test_v6_observation_ranges_smoke():
    env = gym.make('NewPasture-v6')
    try:
        # run multiple seeds to expose sporadic issues
        for seed in [7, 11, 23, 42, 99]:
            obs, info = env.reset(seed=seed)

            # 1) completion ratio
            _assert_in_range('completion_ratio(reset)', obs['completion_ratio'], 0.0, 1.0)

            # 2) vector ranges and segment diagnostics
            L = env.unwrapped.config.state_history_length
            segments = _slice_vector_segments(obs['vector'], L)
            # hard constraints expected by space
            _assert_in_range('vector(reset)', obs['vector'], -1.0, 1.0)

            # provide precise diagnostics for dx/dy which are most suspect
            dx_max = float(np.abs(segments['dx_ego']).max())
            dy_max = float(np.abs(segments['dy_ego']).max())
            # if these exceed bounds, raise with detailed message
            assert dx_max <= 1.0 + 1e-6 and dy_max <= 1.0 + 1e-6, (
                f"ego position normalized exceeded bounds: |dx|_max={dx_max:.6f}, |dy|_max={dy_max:.6f}; "
                f"map_size={env.unwrapped.env_state.dimensions}, L={L}"
            )

            # 3) observation multiscale per-map ranges
            per_map = _per_map_channel_ranges(env, obs['observation'])
            for name, mn, mx, has_nonfinite in per_map:
                assert not has_nonfinite, f"{name} contains NaN/Inf"
                assert mn >= -1e-6 and mx <= 1.0 + 1e-6, (
                    f"{name} out of [0,1]: min={mn:.6f}, max={mx:.6f} (seed={seed})"
                )

            # Step once to also check step() outputs
            a = env.action_space.sample()
            obs2, rew, done, trunc, info2 = env.step(a)
            _assert_in_range('completion_ratio(step)', obs2['completion_ratio'], 0.0, 1.0)
            segments2 = _slice_vector_segments(obs2['vector'], L)
            _assert_in_range('vector(step)', obs2['vector'], -1.0, 1.0)
            dx2_max = float(np.abs(segments2['dx_ego']).max())
            dy2_max = float(np.abs(segments2['dy_ego']).max())
            assert dx2_max <= 1.0 + 1e-6 and dy2_max <= 1.0 + 1e-6, (
                f"(step) ego position normalized exceeded bounds: |dx|_max={dx2_max:.6f}, |dy|_max={dy2_max:.6f}; "
                f"map_size={env.unwrapped.env_state.dimensions}, L={L}"
            )

            per_map2 = _per_map_channel_ranges(env, obs2['observation'])
            for name, mn, mx, has_nonfinite in per_map2:
                assert not has_nonfinite, f"(step){name} contains NaN/Inf"
                assert mn >= -1e-6 and mx <= 1.0 + 1e-6, (
                    f"(step){name} out of [0,1]: min={mn:.6f}, max={mx:.6f} (seed={seed})"
                )
    finally:
        env.close()


def test_v6_time_series_divisor_positive():
    """Fail-fast: total_field_area used for time_series normalization must be > 0."""
    env = gym.make('NewPasture-v6')
    try:
        obs, info = env.reset(seed=123)
        total_area = int(env.unwrapped.env_state.total_field_area)
        assert total_area > 0, f"total_field_area must be > 0, got {total_area}"
    finally:
        env.close()


def test_v6_trajectory_weights_rotation_is_source():
    """Pinpoint whether rotation/cropping introduces negatives for trajectory_weights."""
    env = gym.make('NewPasture-v6')
    try:
        obs, info = env.reset(seed=0)
        base = env.unwrapped
        w = base.maps_dict['trajectory_weights'].astype(np.float32)
        # pre-rotation min/max
        mn0, mx0 = float(w.min()), float(w.max())
        assert mn0 >= 0.0 - 1e-12, f"pre-rotation min < 0: {mn0}"
        assert mx0 <= 1.0 + 1e-6, f"pre-rotation max > 1: {mx0}"

        # build 1-channel stack and rotate via the same helper
        stacked = np.stack([w], axis=-1)
        cy = base.env_state.get_info('agent_position').current[1]
        cx = base.env_state.get_info('agent_position').current[0]
        direction = base.env_state.get_info('agent_direction').current
        patch = extract_ego_patch(
            maps=stacked,
            pad_values=[0.0],
            center_y=cy,
            center_x=cx,
            direction_deg=direction,
            patch_size=base.config.state_size,
        )
        # to downsize if needed (ObservationGenerator will downsize later); here we just check raw
        mn1, mx1 = float(patch.min()), float(patch.max())
        assert mx1 <= 1.0 + 1e-6
        # The failure we're tracking is extremely small negative after rotation
        assert mn1 >= -1e-12, f"rotation introduced negatives: min={mn1}"
    finally:
        env.close()


def test_v6_weight_map_negative_originates_from_filter():
    """Show that negatives are introduced by Gaussian filtering of positive sparse weights."""
    env = gym.make('NewPasture-v6')
    try:
        obs, info = env.reset(seed=0)
        base = env.unwrapped
        traj = base._get_trajectory_history()
        h, w = base.maps_dict['field'].shape
        # Step 1: build sparse time-diffused maps (pre-filter)
        td = base._generate_time_diffusion_maps(traj, (h, w))
        wm0 = td['trajectory_weights'].astype(np.float32)
        assert float(wm0.min()) >= 0.0, f"pre-filter weight_map min < 0: {wm0.min()}"
        # Step 2: apply the same Gaussian filter as env does
        kernel = base._create_gaussian_kernel(base.config.spatial_spread_sigma)
        wm1 = cv2.filter2D(wm0, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        mn1 = float(wm1.min()); mx1 = float(wm1.max())
        # Expect extremely small negative due to floating arithmetic
        assert mn1 >= -1e-12, f"filter introduced too negative values: {mn1}"
        assert mx1 >= 0.0
    finally:
        env.close()
