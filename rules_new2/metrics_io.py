from __future__ import annotations

"""
rules_new2.metrics_io
---------------------

Minimal evaluation utilities for rules_new2 planners.

Design goals (Less is More):
- Reuse the proven rules_new evaluation shape.
- Add only what is necessary for rules_new2:
  * Two-stage reset with initial_position/direction.
  * Planner-idle termination support.
  * Incremental re-evaluation checks via meta/L-threshold completeness.
  * Summary split by weed distribution + overall.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import imageio
import numpy as np

from rules_new2.coverage_planners_v2 import RectifiedFrame


PolicyFn = Callable[[Dict[str, np.ndarray], Any], Tuple[float, float]]
PolicyBuilder = Callable[[Any], Tuple[PolicyFn, Optional[Any]]]


def compute_L_metrics(
    completion: List[float],
    path_length: List[float],
    thresholds: List[float],
) -> Dict[str, float]:
    """Compute Lxx (first path length when completion >= threshold)."""
    out: Dict[str, float] = {}
    for tau in thresholds:
        key = f"L{int(tau * 100)}"
        idx = next((i for i, c in enumerate(completion) if c >= tau), None)
        out[key] = float(path_length[idx]) if idx is not None else -1.0
    return out


def save_run_npy(run: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, run, allow_pickle=True)


def save_last_frame_png(env, path: Path) -> None:
    frame = env.render()
    if frame is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, frame)


def canonical_start_pose(env) -> Tuple[float, float, float]:
    """Compute canonical start pose (world x,y,theta_deg) from a stage-1 env."""
    base = getattr(env, "unwrapped", env)
    frame = RectifiedFrame.from_env(base)
    safe_x_min = frame.x_min + frame.margin_x
    first_y_p = frame.y_min + frame.B / 2.0
    wx0, wy0 = frame.rect_to_world(safe_x_min, first_y_p)
    theta0 = frame.rect_heading_to_world(0.0)
    return wx0, wy0, theta0


def is_planner_idle(planner: Any) -> bool:
    """Lightweight idle detection for rules_new2 planners."""
    for attr in ("_mode", "_phase"):
        if getattr(planner, attr, None) == "idle":
            return True
    return False


def collect_episode(
    env,
    policy_builder: PolicyBuilder,
    meta: Dict[str, Any],
    max_steps: int,
    thresholds: List[float],
    render_last_frame: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run one episode with two-stage reset; return (run_dict, extra)."""
    step_limit = int(max_steps)
    use_step_cap = step_limit >= 0
    seed = int(meta["seed"])
    scene_opts = {
        "map_id": meta["map_id"],
        "weed_distribution": meta["weed_dist"],
        "weed_count": meta["weed_num"],
    }

    # Stage-1 reset to read bounding box.
    env.reset(seed=seed, options=scene_opts)
    wx0, wy0, theta0 = canonical_start_pose(env)

    # Stage-2 reset: identical seed/options, but canonical initial pose.
    obs, _ = env.reset(
        seed=seed,
        options={**scene_opts, "initial_position": (wx0, wy0), "initial_direction": theta0},
    )

    # Keep wrappers for reset/step/render, but use the unwrapped env for internal
    # state access to avoid Gymnasium wrapper-attr deprecation warnings.
    base_env = getattr(env, "unwrapped", env)

    policy_fn, planner = policy_builder(env)

    rewards: List[float] = []
    completion: List[float] = []
    overlap: List[float] = []
    path_length: List[float] = []
    positions: List[Tuple[float, float]] = []

    terminated = False
    truncated = False
    planner_idle = False
    hit_step_cap = False
    step_count = 0

    while True:
        if use_step_cap and step_count >= step_limit:
            hit_step_cap = True
            break

        action = policy_fn(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

        rewards.append(float(reward))
        completion.append(float(obs["completion_ratio"][0]))
        path_length.append(float(base_env.env_state.trajectory_length))

        if "overlap_count" in base_env.env_state._state_infos:
            overlap.append(float(base_env.env_state.overlap_count))
        elif "overlap_count" in info:
            overlap.append(float(info["overlap_count"]))

        pos_info = base_env.env_state.get_info("agent_position")
        if pos_info and pos_info.current is not None:
            positions.append(tuple(pos_info.current))

        timeout_now = bool(base_env.env_state.timeout)
        if terminated:
            break
        # max_steps < 0 时忽略环境 timeout 截断（由成功/碰撞/规划器结束）。
        if truncated and not (not use_step_cap and timeout_now):
            break
        if planner is not None and is_planner_idle(planner):
            planner_idle = True
            break

    crashed = bool(base_env.env_state.crashed)
    finished = bool(base_env.env_state.finished)
    timeout_flag = bool(base_env.env_state.timeout)

    if crashed:
        done_type = "collision"
        done_trigger = "env"
    elif finished:
        done_type = "env_success"
        done_trigger = "env"
    elif planner_idle:
        done_type = "planner_idle"
        done_trigger = "planner"
    elif timeout_flag or hit_step_cap:
        done_type = "timeout"
        done_trigger = "env" if timeout_flag else "max_steps"
    else:
        done_type = "unknown"
        done_trigger = "env"

    L = compute_L_metrics(completion, path_length, thresholds)
    idx95 = next((i for i, c in enumerate(completion) if c >= 0.95), None)
    steps95 = (len(rewards) - idx95) if idx95 is not None else None

    # 90%/95%->done 的“路径长度与比例”版本（只对正常 finished 的回合有意义）
    L90_val = float(L.get("L90", -1.0))
    L95_val = float(L.get("L95", -1.0))
    if done_type == "env_success" and path_length:
        total_len = float(path_length[-1])
        length_90_to_done = float(total_len - L90_val) if L90_val >= 0 else np.nan
        length_95_to_done = float(total_len - L95_val) if L95_val >= 0 else np.nan
        if total_len > 0:
            path_len_ratio_90_to_done = (
                float(length_90_to_done / total_len) if np.isfinite(length_90_to_done) else np.nan
            )
            path_len_ratio_95_to_done = (
                float(length_95_to_done / total_len) if np.isfinite(length_95_to_done) else np.nan
            )
        else:
            path_len_ratio_90_to_done = np.nan
            path_len_ratio_95_to_done = np.nan
    else:
        length_90_to_done = np.nan
        length_95_to_done = np.nan
        path_len_ratio_90_to_done = np.nan
        path_len_ratio_95_to_done = np.nan

    run: Dict[str, Any] = {
        "meta": meta,
        "episode_reward": float(sum(rewards)),
        "episode_length": len(rewards),
        "completion_final": completion[-1] if completion else 0.0,
        "overlap_count_final": overlap[-1] if overlap else None,
        "done_type": done_type,
        "done_trigger": done_trigger,
        "steps_95_to_done": steps95,  # 保留 raw step 版本，summary 不再使用
        "length_90_to_done": length_90_to_done,
        "length_95_to_done": length_95_to_done,
        "path_len_ratio_90_to_done": path_len_ratio_90_to_done,
        "path_len_ratio_95_to_done": path_len_ratio_95_to_done,
        "rewards": rewards,
        "completion": completion,
        "overlap": overlap,
        "path_length": path_length,
        "positions": positions,
    }
    run.update(L)

    extra: Dict[str, Any] = {}
    if render_last_frame:
        extra["render_png"] = True
    return run, extra


def load_runs(root: Path):
    """Load all runs under root into a pandas DataFrame."""
    import pandas as pd

    rows = []
    for npy_path in root.glob("*/level-*/*.npy"):
        # Ignore legacy folders created before removing the "alg-" prefix.
        method_dir = npy_path.parents[1].name
        if method_dir.startswith("alg-"):
            continue
        data = np.load(npy_path, allow_pickle=True).item()
        meta = data["meta"]
        row: Dict[str, Any] = {
            "method": meta["method"],
            "method_type": meta.get("method_type", "rules"),
            "level": meta["level"],
            "dist": meta["weed_dist"],
            "episode_reward": data["episode_reward"],
            "completion_final": data["completion_final"],
            "collision": 1.0 if data["done_type"] == "collision" else 0.0,
            "timeout": 1.0 if data["done_type"] == "timeout" else 0.0,
            "planner_idle": 1.0 if data["done_type"] == "planner_idle" else 0.0,
            "success": 1.0 if data["done_type"] == "env_success" else 0.0,
            "overlap_count_final": data.get("overlap_count_final", data.get("overlap_final", np.nan)),
            "path_len_ratio_90_to_done": data.get("path_len_ratio_90_to_done", np.nan),
            "path_len_ratio_95_to_done": data.get("path_len_ratio_95_to_done", np.nan),
        }

        # Dynamically include all Lxx metrics present in the run file.
        for k, v in data.items():
            if isinstance(k, str) and k.startswith("L") and k[1:].isdigit():
                row[k] = float(v)

        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_runs_to_summary(root: Path):
    """Return (overall_summary_df, by_dist_summary_df)."""
    import pandas as pd

    df = load_runs(root)
    if df.empty:
        return None, None

    # Identify Lxx columns and ignore runs that never reached the threshold.
    L_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("L") and c[1:].isdigit()]
    L_cols.sort(key=lambda s: int(s[1:]))
    for col in L_cols:
        df.loc[df[col] < 0, col] = np.nan

    agg_spec: Dict[str, Tuple[str, str]] = {
        "n_runs": ("episode_reward", "count"),
        "reward_mean": ("episode_reward", "mean"),
        "completion_mean": ("completion_final", "mean"),
        "collision_rate": ("collision", "mean"),
        "timeout_rate": ("timeout", "mean"),
        "planner_idle_rate": ("planner_idle", "mean"),
        "success_rate": ("success", "mean"),
        "overlap_count_mean": ("overlap_count_final", "mean"),
        "path_len_ratio_90_to_done_mean": ("path_len_ratio_90_to_done", "mean"),
        "path_len_ratio_95_to_done_mean": ("path_len_ratio_95_to_done", "mean"),
    }
    for col in L_cols:
        agg_spec[f"{col}_mean"] = (col, "mean")

    agg_by = df.groupby(["method", "level", "dist"]).agg(**agg_spec).reset_index()
    agg_all = df.groupby(["method", "level"]).agg(**agg_spec).reset_index()
    agg_all.insert(2, "dist", "all")

    # Reorder columns for readability and stable CSV output.
    ordered_L_means = [f"{c}_mean" for c in L_cols]
    base_cols = (
        ["method", "level", "dist", "n_runs", "reward_mean", "completion_mean"]
        + ordered_L_means
        + [
            "path_len_ratio_90_to_done_mean",
            "path_len_ratio_95_to_done_mean",
            "success_rate",
            "collision_rate",
            "timeout_rate",
            "planner_idle_rate",
            "overlap_count_mean",
        ]
    )
    agg_all = agg_all.reindex(columns=base_cols)
    agg_by = agg_by.reindex(columns=base_cols)

    return agg_all, agg_by


def save_config_snapshot(cfg: Dict[str, Any], output_root: Path) -> None:
    """Save a YAML snapshot of the config used for this eval.

    Notes:
    - `config_used.yaml` is the single source of truth for reproducibility.
    - If a legacy `config_used.json` exists from older runs, it will be removed
      to keep the output directory clean.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    yaml_path = output_root / "config_used.yaml"
    import yaml

    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    legacy_json = output_root / "config_used.json"
    try:
        legacy_json.unlink()
    except FileNotFoundError:
        pass
