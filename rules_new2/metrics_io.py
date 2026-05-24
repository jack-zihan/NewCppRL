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
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import imageio
import numpy as np

from rules_new2.coverage_planners_v2 import RectifiedFrame


PolicyFn = Callable[[Dict[str, np.ndarray], Any], Tuple[float, float]]
PolicyBuilder = Callable[[Any], Tuple[PolicyFn, Optional[Any]]]


@lru_cache(maxsize=1)
def _field_map_files() -> Tuple[Path, ...]:
    field_dir = Path(__file__).resolve().parents[1] / "envs_new" / "maps" / "weed_coverage" / "field"
    return tuple(sorted(p for p in field_dir.iterdir() if p.suffix.lower() == ".png"))


@lru_cache(maxsize=None)
def _field_area_from_map_id(map_id: int) -> float:
    map_files = _field_map_files()
    if map_id < 0 or map_id >= len(map_files):
        raise IndexError(f"map_id={map_id} out of range for field map list of size {len(map_files)}")
    img = cv2.imread(str(map_files[map_id]), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read field map: {map_files[map_id]}")
    field = (img.sum(axis=-1) > 0).astype(np.uint8)
    return float(field.sum())


def _field_area_ref_from_config(root: Path) -> Optional[float]:
    cfg_path = root / "config_used.yaml"
    if not cfg_path.exists():
        return None

    import yaml

    cfg = yaml.safe_load(cfg_path.read_text())
    levels = cfg.get("scenes", {}).get("levels", [])
    map_ids = sorted({int(map_id) for level in levels for map_id in level.get("map_ids", [])})
    if not map_ids:
        return None
    return float(np.mean([_field_area_from_map_id(map_id) for map_id in map_ids]))


def _summary_orders_from_config(root: Path) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """从 config_used.yaml 读取 method / level / dist 的展示顺序。"""
    cfg_path = root / "config_used.yaml"
    if not cfg_path.exists():
        return {}, {}, {}

    import yaml

    cfg = yaml.safe_load(cfg_path.read_text())
    levels = cfg.get("scenes", {}).get("levels", [])
    weed_dists = cfg.get("scenes", {}).get("weed_dists", [])
    learning = cfg.get("methods", {}).get("learning", [])
    rules = cfg.get("methods", {}).get("rules", [])

    level_order = {str(level["name"]): i for i, level in enumerate(levels)}
    method_order = {
        str(method["name"]): i
        for i, method in enumerate(list(learning) + list(rules))
    }
    dist_order = {"all": 0}
    for i, dist in enumerate(weed_dists, start=1):
        dist_order[str(dist)] = i
    return method_order, level_order, dist_order


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


def build_success_contact_sheets(root: Path, grid: Tuple[int, int] = (8, 8)) -> None:
    """为每个 method/level 生成成功案例聚合图。

    输入：
    - root: 评估输出根目录，例如 rules_new2/full_opcpp_eval
    - grid: 每页网格大小，默认 8x8

    规则：
    - 仅收集 `done_type == "env_success"` 且对应 png 存在的 run；
    - 每个 level 目录下生成 `success_sheet_XX.png`；
    - 排序遵循：map_id -> weed_dist(gaussian/uniform) -> seed。
    """
    method_order, level_order, dist_order = _summary_orders_from_config(root)
    grid_rows, grid_cols = grid
    per_page = max(1, grid_rows * grid_cols)

    thumb_w = 170
    thumb_h = 170
    text_h = 42
    pad = 10
    title_h = 44
    cell_w = thumb_w + 2 * pad
    cell_h = thumb_h + text_h + 2 * pad
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    line1_y = thumb_h + pad + 16
    line2_y = thumb_h + pad + 34

    level_dirs = sorted(
        [p for p in root.glob("*/level-*") if p.is_dir()],
        key=lambda p: (
            method_order.get(p.parent.name, 999),
            level_order.get(p.name.replace("level-", ""), 999),
            p.parent.name,
            p.name,
        ),
    )

    for level_dir in level_dirs:
        for old in level_dir.glob("success_sheet_*.png"):
            old.unlink()

        runs = []
        for npy_path in sorted(level_dir.glob("*.npy")):
            stem = npy_path.stem
            png_path = level_dir / f"{stem}.png"
            if not png_path.exists():
                continue
            data = np.load(npy_path, allow_pickle=True).item()
            if data.get("done_type") != "env_success":
                continue
            meta = data.get("meta", {})
            dist = str(meta.get("weed_dist", ""))
            runs.append(
                {
                    "npy": npy_path,
                    "png": png_path,
                    "map_id": int(meta.get("map_id", -1)),
                    "seed": int(meta.get("seed", -1)),
                    "dist": dist,
                    "dist_tag": "G" if dist == "gaussian" else ("U" if dist == "uniform" else dist[:1].upper()),
                    "completion": float(data.get("completion_final", 0.0)),
                }
            )

        runs.sort(
            key=lambda x: (
                x["map_id"],
                dist_order.get(x["dist"], 999),
                x["seed"],
                x["png"].name,
            )
        )

        if not runs:
            continue

        n_pages = int(np.ceil(len(runs) / per_page))
        for page_idx in range(n_pages):
            page_runs = runs[page_idx * per_page : (page_idx + 1) * per_page]
            canvas_h = title_h + grid_rows * cell_h
            canvas_w = grid_cols * cell_w
            canvas = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)

            level_name = level_dir.name.replace("level-", "")
            title = (
                f"{level_dir.parent.name} | {level_name} | "
                f"success {len(runs)} | page {page_idx + 1}/{n_pages}"
            )
            cv2.putText(canvas, title, (12, 28), font, 0.7, (20, 20, 20), 2, cv2.LINE_AA)

            for i, item in enumerate(page_runs):
                row = i // grid_cols
                col = i % grid_cols
                x0 = col * cell_w
                y0 = title_h + row * cell_h

                img = cv2.imread(str(item["png"]), cv2.IMREAD_COLOR)
                if img is None:
                    continue

                h, w = img.shape[:2]
                scale = min(thumb_w / max(w, 1), thumb_h / max(h, 1))
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

                ix = x0 + pad + (thumb_w - new_w) // 2
                iy = y0 + pad + (thumb_h - new_h) // 2
                canvas[iy : iy + new_h, ix : ix + new_w] = img
                cv2.rectangle(canvas, (x0 + pad, y0 + pad), (x0 + pad + thumb_w, y0 + pad + thumb_h), (180, 180, 180), 1)

                line1 = f"map{item['map_id']} {item['dist_tag']} s{item['seed']}"
                line2 = f"cov={item['completion']:.3f}"
                cv2.putText(canvas, line1, (x0 + pad, y0 + line1_y), font, font_scale, (30, 30, 30), 1, cv2.LINE_AA)
                cv2.putText(canvas, line2, (x0 + pad, y0 + line2_y), font, font_scale, (30, 30, 30), 1, cv2.LINE_AA)

            out_path = level_dir / f"success_sheet_{page_idx + 1:02d}.png"
            cv2.imwrite(str(out_path), canvas)


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
    field_area = float(base_env.env_state.get_static_info("total_field_area") or 0.0)

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
        "field_area": field_area,
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
        map_id = int(meta["map_id"])
        row: Dict[str, Any] = {
            "method": meta["method"],
            "method_type": meta.get("method_type", "rules"),
            "level": meta["level"],
            "dist": meta["weed_dist"],
            "map_id": map_id,
            "episode_reward": data["episode_reward"],
            "completion_final": data["completion_final"],
            # 这里始终按当前 map_id 对应的 field 掩码重算面积，而不是信任
            # 历史 npy 中保存的 field_area。原因是地图掩码本身可能在评估后
            # 被修正/替换；若继续沿用旧值，会让 Lxx_area_norm 与当前场景
            # 几何定义不一致。
            "field_area": float(_field_area_from_map_id(map_id)),
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

    field_area_ref = _field_area_ref_from_config(root)
    if field_area_ref is None or field_area_ref <= 0:
        field_area_ref = float(
            df[["map_id", "field_area"]].drop_duplicates(subset=["map_id"])["field_area"].mean()
        )

    valid_area = df["field_area"] > 0
    area_scale = pd.Series(np.nan, index=df.index, dtype=float)
    area_scale.loc[valid_area] = field_area_ref / df.loc[valid_area, "field_area"]

    norm_cols: List[str] = []
    for col in L_cols:
        norm_col = f"{col}_area_norm"
        df[norm_col] = df[col] * area_scale
        norm_cols.append(norm_col)

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
        agg_spec[f"{col}_n"] = (col, lambda s: int(s.notna().sum()))
        agg_spec[f"{col}_mean"] = (col, "mean")
    for col in norm_cols:
        agg_spec[f"{col}_mean"] = (col, "mean")

    agg_by = df.groupby(["method", "level", "dist"]).agg(**agg_spec).reset_index()
    agg_all = df.groupby(["method", "level"]).agg(**agg_spec).reset_index()
    agg_all.insert(2, "dist", "all")
    agg_all.insert(3, "field_area_ref", field_area_ref)
    agg_by.insert(3, "field_area_ref", field_area_ref)

    method_order, level_order, dist_order = _summary_orders_from_config(root)
    for out_df in (agg_all, agg_by):
        out_df["_method_order"] = out_df["method"].map(method_order).fillna(999).astype(int)
        out_df["_level_order"] = out_df["level"].map(level_order).fillna(999).astype(int)
        out_df["_dist_order"] = out_df["dist"].map(dist_order).fillna(999).astype(int)

    agg_all = agg_all.sort_values(
        by=["_method_order", "_level_order", "_dist_order", "method", "level", "dist"],
        kind="stable",
    ).reset_index(drop=True)
    agg_by = agg_by.sort_values(
        by=["_method_order", "_level_order", "_dist_order", "method", "level", "dist"],
        kind="stable",
    ).reset_index(drop=True)

    # Reorder columns for readability and stable CSV output.
    ordered_L_stats: List[str] = []
    for c in L_cols:
        ordered_L_stats.append(f"{c}_n")
        ordered_L_stats.append(f"{c}_mean")
    ordered_norm_L_means = [f"{c}_mean" for c in norm_cols]
    base_cols = (
        ["method", "level", "dist", "field_area_ref", "n_runs", "reward_mean", "completion_mean"]
        + ordered_L_stats
        + ordered_norm_L_means
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
