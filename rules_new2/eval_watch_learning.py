from __future__ import annotations

"""
训练模型 checkpoint 的监控评估循环脚本。 python -m rules_new2.eval_watch_learning --config rules_new2/configs/eval_watch_learning.yaml

核心流程（Less is More）：
1) 扫描 checkpoint 目录，识别未评估的 step；
2) 仅对新增 step 复用 rules_new2 的现有评估管线执行评估；
3) 每完成一个新 step，立即重建全局汇总表与趋势图。

"""

import argparse
import copy
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from rules_new2.eval_runner import build_tasks, run_task_batch, should_run_task
from rules_new2.metrics_io import aggregate_runs_to_summary, save_config_snapshot

STEP_PATTERN = re.compile(r"model_step(\d+).*\.pt$")
TIMEOUT_DISABLED_MAX_STEPS = 2_147_483_647


def load_config(path: Path) -> Dict[str, Any]:
    """加载并校验监控评估配置（最小必需字段）。"""
    cfg = yaml.safe_load(path.read_text())
    required = ["watch", "env", "scenes", "method", "eval"]
    for key in required:
        if key not in cfg:
            raise ValueError(f"Missing top-level key: {key}")
    return cfg


def apply_eval_timeout_override(cfg: Dict[str, Any]) -> None:
    """将 eval.max_steps 作为唯一超时来源，覆盖环境 max_episode_steps。"""
    eval_max_steps = int(cfg["eval"]["max_steps"])
    env_cfg = cfg.setdefault("env", {})
    base_kwargs = env_cfg.setdefault("base_kwargs", {})
    if eval_max_steps < 0:
        if (
            "max_episode_steps" in base_kwargs
            and int(base_kwargs["max_episode_steps"]) != TIMEOUT_DISABLED_MAX_STEPS
        ):
            print(
                "[Watch] override env.base_kwargs.max_episode_steps="
                f"{base_kwargs['max_episode_steps']} -> {TIMEOUT_DISABLED_MAX_STEPS}"
            )
        base_kwargs["max_episode_steps"] = TIMEOUT_DISABLED_MAX_STEPS
        print(
            "[Watch] eval.max_steps < 0: disable step-cap timeout "
            f"(env.max_episode_steps={TIMEOUT_DISABLED_MAX_STEPS})"
        )
    else:
        if (
            "max_episode_steps" in base_kwargs
            and int(base_kwargs["max_episode_steps"]) != eval_max_steps
        ):
            print(
                "[Watch] override env.base_kwargs.max_episode_steps="
                f"{base_kwargs['max_episode_steps']} -> eval.max_steps={eval_max_steps}"
            )
        base_kwargs["max_episode_steps"] = eval_max_steps


def resolve_output_root(cfg: Dict[str, Any]) -> Path:
    """解析输出目录：当 output_subdir 是相对路径时，挂在 ckpt_dir 下。"""
    ckpt_dir = Path(cfg["watch"]["ckpt_dir"]).expanduser().resolve()
    out_subdir = str(cfg["watch"].get("output_subdir", "external_eval")).strip()
    out_root = Path(out_subdir).expanduser()
    if not out_root.is_absolute():
        out_root = ckpt_dir / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    return out_root


def scan_checkpoint_steps(ckpt_dir: Path, settle_seconds: int) -> Dict[int, Path]:
    """返回 step->ckpt 的映射。

    规则：
    - 从文件名解析 step，并按 step 分组；
    - 忽略“过新文件”，避免读到尚未写完的 checkpoint；
    - 同一步同时存在 pending 和重命名文件时，优先使用非 pending 文件。
    """
    now = time.time()
    by_step: Dict[int, List[Path]] = {}
    for pt_path in sorted(ckpt_dir.glob("*.pt")):
        match = STEP_PATTERN.match(pt_path.name)
        if not match:
            continue
        if now - pt_path.stat().st_mtime < settle_seconds:
            continue
        step = int(match.group(1))
        by_step.setdefault(step, []).append(pt_path)

    chosen: Dict[int, Path] = {}
    for step, files in by_step.items():
        non_pending = [p for p in files if "eval_pending" not in p.name]
        pool = non_pending if non_pending else files
        chosen[step] = max(pool, key=lambda p: p.stat().st_mtime)
    return chosen


def step_output_dir(output_root: Path, step: int) -> Path:
    return output_root / "steps" / f"step_{step:08d}"


def step_meta_path(output_root: Path, step: int) -> Path:
    return step_output_dir(output_root, step) / "step_meta.json"


def is_step_processed(output_root: Path, step: int) -> bool:
    return step_meta_path(output_root, step).exists()


def build_step_eval_cfg(base_cfg: Dict[str, Any], ckpt_path: Path, step_out: Path) -> Dict[str, Any]:
    """构造单 step 的评估配置，复用标准 eval_runner 管线。"""
    cfg = copy.deepcopy(base_cfg)
    method_cfg = copy.deepcopy(cfg["method"])
    method_cfg["ckpt"] = str(ckpt_path.resolve())

    cfg["methods"] = {"learning": [method_cfg], "rules": []}
    cfg["eval"]["output_root"] = str(step_out)
    cfg["eval"]["mode"] = str(cfg["eval"].get("mode", "incremental"))
    return cfg


def run_one_step_eval(base_cfg: Dict[str, Any], step: int, ckpt_path: Path, output_root: Path) -> Dict[str, Any]:
    """执行单个 step 评估，并写入 step 级元信息。"""
    step_out = step_output_dir(output_root, step)
    step_out.mkdir(parents=True, exist_ok=True)
    step_cfg = build_step_eval_cfg(base_cfg, ckpt_path, step_out)

    tasks_all = build_tasks(step_cfg)
    eval_mode = str(step_cfg["eval"].get("mode", "incremental"))
    tasks = [task for task in tasks_all if should_run_task(task, eval_mode)]
    skipped = len(tasks_all) - len(tasks)

    workers_net = int(step_cfg["eval"].get("workers_net", step_cfg["eval"].get("workers_learning", 1)))
    print(
        f"[Watch] step={step} ckpt={ckpt_path.name} "
        f"tasks_total={len(tasks_all)} to_run={len(tasks)} skipped={skipped}"
    )
    done_counts, errors = run_task_batch(tasks, step_cfg, max_workers=max(workers_net, 1))

    meta = {
        "step": step,
        "ckpt": str(ckpt_path.resolve()),
        "ckpt_name": ckpt_path.name,
        "tasks_total": len(tasks_all),
        "tasks_ran": len(tasks),
        "tasks_skipped": skipped,
        "workers_net": workers_net,
        "done_stats": {
            "env_success": int(done_counts.get("env_success", 0)),
            "collision": int(done_counts.get("collision", 0)),
            "timeout": int(done_counts.get("timeout", 0)),
            "planner_idle": int(done_counts.get("planner_idle", 0)),
            "errors": int(errors),
        },
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    step_meta_path(output_root, step).write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[Watch] step={step} done_stats={meta['done_stats']}")
    return meta


def build_summary_all_steps(output_root: Path) -> pd.DataFrame:
    """将每个 step 的 summary 合并为跨 step 的全局 CSV。"""
    rows: List[pd.DataFrame] = []
    steps_root = output_root / "steps"
    if not steps_root.exists():
        return pd.DataFrame()

    for step_dir in sorted(steps_root.glob("step_*")):
        if not step_dir.is_dir():
            continue
        step_match = re.match(r"step_(\d+)$", step_dir.name)
        if step_match is None:
            continue
        step = int(step_match.group(1))

        agg_all, agg_by = aggregate_runs_to_summary(step_dir)
        if agg_all is None or agg_by is None:
            continue

        meta_file = step_dir / "step_meta.json"
        ckpt_name = ""
        if meta_file.exists():
            try:
                ckpt_name = json.loads(meta_file.read_text()).get("ckpt_name", "")
            except Exception:
                ckpt_name = ""

        merged = pd.concat([agg_all, agg_by], ignore_index=True)
        merged.insert(0, "step", step)
        merged.insert(1, "ckpt_name", ckpt_name)
        rows.append(merged)

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    df = df.sort_values(by=["step", "method", "level", "dist"]).reset_index(drop=True)
    summary_path = output_root / "summary_all_steps.csv"
    df.to_csv(summary_path, index=False)
    return df


def _plot_lines(ax, df: pd.DataFrame, levels: List[str], y_col: str, title: str, ylabel: str) -> bool:
    """按 level 画折线的公共辅助函数。"""
    if y_col not in df.columns:
        return False
    has_data = False
    for level in levels:
        d = df[df["level"] == level]
        if d.empty:
            continue
        ax.plot(d["step"], d[y_col], marker="o", linewidth=1.6, label=level)
        has_data = True
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    if has_data:
        ax.legend()
    return has_data


def draw_trend_plots(df_summary: pd.DataFrame, output_root: Path, levels: List[str]) -> None:
    """仅使用 dist='all' 的数据绘制指标-步数趋势图。"""
    if df_summary.empty:
        return
    df = df_summary[df_summary["dist"] == "all"].copy()
    if df.empty:
        return
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df = df.dropna(subset=["step"]).sort_values(by="step")

    trend_dir = output_root / "trend"
    plot_dir = trend_dir / "plots"
    trend_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(trend_dir / "trend_all_levels.csv", index=False)

    # 奖励趋势图
    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    if _plot_lines(ax, df, levels, "reward_mean", "Reward vs Step (dist=all)", "reward_mean"):
        fig.tight_layout()
        fig.savefig(plot_dir / "reward_curve.png")
    plt.close(fig)

    # 成功率与碰撞率趋势图
    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    has_data = False
    for level in levels:
        d = df[df["level"] == level]
        if d.empty:
            continue
        if "success_rate" in d.columns:
            ax.plot(d["step"], d["success_rate"], marker="o", linewidth=1.6, label=f"{level}-success")
            has_data = True
        if "collision_rate" in d.columns:
            ax.plot(d["step"], d["collision_rate"], marker="x", linestyle="--", linewidth=1.4, label=f"{level}-collision")
            has_data = True
    ax.set_title("Success & Collision Rate vs Step (dist=all)")
    ax.set_xlabel("step")
    ax.set_ylabel("rate")
    ax.grid(alpha=0.3)
    if has_data:
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(plot_dir / "success_collision_curve.png")
    plt.close(fig)

    # L90/L95/L98（按 level 分子图）
    l_metrics = [col for col in ["L90_mean", "L95_mean", "L98_mean"] if col in df.columns]
    if l_metrics:
        fig, axes = plt.subplots(1, len(levels), figsize=(5.2 * len(levels), 4.6), dpi=140, sharey=True)
        if len(levels) == 1:
            axes = [axes]
        for i, level in enumerate(levels):
            ax = axes[i]
            d = df[df["level"] == level]
            for metric in l_metrics:
                ax.plot(d["step"], d[metric], marker="o", linewidth=1.5, label=metric)
            ax.set_title(f"{level}")
            ax.set_xlabel("step")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_ylabel("path length")
            ax.legend()
        fig.suptitle("L90/L95/L98 vs Step (dist=all)")
        fig.tight_layout()
        fig.savefig(plot_dir / "L_curve.png")
        plt.close(fig)

    # 90/95 到终点的尾段路径比例（按 level 分子图）
    tail_metrics = [
        col
        for col in ["path_len_ratio_90_to_done_mean", "path_len_ratio_95_to_done_mean"]
        if col in df.columns
    ]
    if tail_metrics:
        fig, axes = plt.subplots(1, len(levels), figsize=(5.2 * len(levels), 4.6), dpi=140, sharey=True)
        if len(levels) == 1:
            axes = [axes]
        for i, level in enumerate(levels):
            ax = axes[i]
            d = df[df["level"] == level]
            for metric in tail_metrics:
                ax.plot(d["step"], d[metric], marker="o", linewidth=1.5, label=metric)
            ax.set_title(f"{level}")
            ax.set_xlabel("step")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_ylabel("ratio")
            ax.legend()
        fig.suptitle("Tail Path Ratio vs Step (dist=all)")
        fig.tight_layout()
        fig.savefig(plot_dir / "tail_ratio_curve.png")
        plt.close(fig)

    # overlap_count 趋势图
    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    if _plot_lines(
        ax,
        df,
        levels,
        "overlap_count_mean",
        "Overlap Count vs Step (dist=all)",
        "overlap_count_mean",
    ):
        fig.tight_layout()
        fig.savefig(plot_dir / "overlap_curve.png")
    plt.close(fig)


def update_global_outputs(output_root: Path, levels: List[str]) -> None:
    """刷新全局汇总表和全部趋势图。"""
    df = build_summary_all_steps(output_root)
    draw_trend_plots(df, output_root, levels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    apply_eval_timeout_override(cfg)

    ckpt_dir = Path(cfg["watch"]["ckpt_dir"]).expanduser().resolve()
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"ckpt_dir not found: {ckpt_dir}")
    output_root = resolve_output_root(cfg)
    save_config_snapshot(cfg, output_root)

    poll_seconds = int(cfg["watch"].get("poll_seconds", 60))
    settle_seconds = int(cfg["watch"].get("settle_seconds", 20))
    run_once = bool(cfg["watch"].get("run_once", False))
    levels = [str(level["name"]) for level in cfg["scenes"]["levels"]]

    print(f"[Watch] ckpt_dir={ckpt_dir}")
    print(f"[Watch] output_root={output_root}")
    print(f"[Watch] poll_seconds={poll_seconds}, settle_seconds={settle_seconds}, run_once={run_once}")

    while True:
        # 仅检测“未处理 step”，已处理以 step_meta.json 作为标记。
        step_to_ckpt = scan_checkpoint_steps(ckpt_dir, settle_seconds=settle_seconds)
        all_steps = sorted(step_to_ckpt.keys())
        new_steps = [step for step in all_steps if not is_step_processed(output_root, step)]

        if new_steps:
            print(f"[Watch] found new steps: {new_steps}")
            for step in new_steps:
                # 增量更新：评估一个 step 后，立刻刷新全局产物。
                run_one_step_eval(cfg, step, step_to_ckpt[step], output_root)
                update_global_outputs(output_root, levels=levels)
        else:
            print("[Watch] no new checkpoint step, sleep...")

        if run_once:
            break
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
