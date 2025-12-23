import os
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple

import imageio
import numpy as np


def compute_L_metrics(completion: List[float], path_length: List[float],
                      thresholds: List[float]) -> Dict[str, float]:
    """Compute Lxx (first path length when completion >= threshold)."""
    L = {}
    for tau in thresholds:
        key = f"L{int(tau * 100)}"
        idx = next((i for i, c in enumerate(completion) if c >= tau), None)
        L[key] = float(path_length[idx]) if idx is not None else -1.0
    return L


def save_run_npy(run: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, run, allow_pickle=True)


def save_last_frame_png(env, path: Path) -> None:
    frame = env.render()
    if frame is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, frame)


def collect_episode(env,
                    policy_fn,
                    meta: Dict[str, Any],
                    max_steps: int,
                    thresholds: List[float],
                    render_last_frame: bool) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run one episode, return (run_dict, extra_paths)."""
    obs, _ = env.reset(seed=meta["seed"], options={
        "map_id": meta["map_id"],
        "weed_distribution": meta["weed_dist"],
        "weed_count": meta["weed_num"],
    })

    rewards, completion, overlap, path_length = [], [], [], []
    positions = []

    terminated = False
    truncated = False

    for step in range(max_steps):
        action = policy_fn(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(float(reward))
        completion.append(float(obs["completion_ratio"][0]))
        path_length.append(float(env.env_state.trajectory_length))

        if "overlap_count" in env.env_state._state_infos:
            overlap.append(float(env.env_state.overlap_count))
        elif "overlap_count" in info:
            overlap.append(float(info["overlap_count"]))

        pos_info = env.env_state.get_info("agent_position")
        if pos_info and pos_info.current is not None:
            positions.append(tuple(pos_info.current))

        if terminated or truncated:
            break

    # done type
    crashed = bool(env.env_state.crashed)
    finished = bool(env.env_state.finished)
    timeout_flag = bool(env.env_state.timeout)
    if crashed:
        done_type = "collision"
    elif timeout_flag and not finished:
        done_type = "timeout"
    elif finished:
        done_type = "success"
    else:
        # 没有触发终止且达到max_steps，则视为超时
        done_type = "timeout" if not (terminated or truncated) else "unknown"

    L = compute_L_metrics(completion, path_length, thresholds)

    idx95 = next((i for i, c in enumerate(completion) if c >= 0.95), None)
    steps95 = (len(rewards) - idx95) if idx95 is not None else None

    run = {
        "meta": meta,
        "episode_reward": float(sum(rewards)),
        "episode_length": len(rewards),
        "completion_final": completion[-1] if completion else 0.0,
        "overlap_final": overlap[-1] if overlap else None,
        "done_type": done_type,
        "steps_95_to_done": steps95,
        "rewards": rewards,
        "completion": completion,
        "overlap": overlap,
        "path_length": path_length,
        "positions": positions,
    }
    run.update(L)

    extra = {}
    if render_last_frame:
        extra["render_png"] = True
    return run, extra


def aggregate_runs_to_summary(root: Path) -> np.ndarray:
    """
    Scan all npy files under root and produce a summary table (numpy structured array).
    """
    rows = []
    for npy_path in root.glob("alg-*/level-*/*.npy"):
        data = np.load(npy_path, allow_pickle=True).item()
        meta = data["meta"]
        method = meta["method"]
        level = meta["level"]
        dist = meta["weed_dist"]
        row = {
            "method": method,
            "level": level,
            "dist": dist,
            "episode_reward": data["episode_reward"],
            "completion_final": data["completion_final"],
            "L90": data.get("L90", -1.0),
            "L95": data.get("L95", -1.0),
            "L98": data.get("L98", -1.0),
            "collision": 1.0 if data["done_type"] == "collision" else 0.0,
            "timeout": 1.0 if data["done_type"] == "timeout" else 0.0,
            "overlap_final": data.get("overlap_final", np.nan),
            "steps_95_to_done": data.get("steps_95_to_done", np.nan),
        }
        rows.append(row)

    if not rows:
        return np.array([], dtype=[])

    # Group by (method, level, dist)
    import pandas as pd
    df = pd.DataFrame(rows)
    agg = df.groupby(["method", "level", "dist"]).agg(
        n_runs=("episode_reward", "count"),
        reward_mean=("episode_reward", "mean"),
        completion_mean=("completion_final", "mean"),
        L90_mean=("L90", "mean"),
        L95_mean=("L95", "mean"),
        L98_mean=("L98", "mean"),
        collision_rate=("collision", "mean"),
        timeout_rate=("timeout", "mean"),
        overlap_mean=("overlap_final", "mean"),
        steps95_mean=("steps_95_to_done", "mean"),
    ).reset_index()
    return agg
