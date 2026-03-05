from __future__ import annotations

"""
Unified evaluation entrypoint for rules_new2.

Usage:
    python -m rules_new2.eval_runner --config rules_new2/configs/eval_example.yaml

Key features (minimal, research-friendly):
- YAML-driven task grid (levels × map_ids × seeds × dists × methods).
- Two-stage reset for canonical start pose.
- Planner-idle termination support.
- Incremental evaluation (skip tasks with complete cached npy).
- Summary split: overall + gaussian + uniform.
"""

import argparse
import csv
import json
import traceback
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import yaml

import envs_new  # noqa: F401  # register gym envs
from rules_new2.coverage_planners_v2 import (
    BCPPlannerV2,
    JumpPlannerV2,
    SnakePlannerV2,
    RestrictedSnakePlannerV2,
    ReactPlannerV2,
)
from rules_new2.metrics_io import (
    aggregate_runs_to_summary,
    collect_episode,
    save_config_snapshot,
    save_last_frame_png,
    save_run_npy,
)

TIMEOUT_DISABLED_MAX_STEPS = 2_147_483_647

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

# Keep progress output readable by silencing noisy Gymnasium wrapper warnings.
warnings.filterwarnings(
    "ignore",
    message=r".*env\.(env_state|config|agent|action_processor|maps_dict).*deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*The environment .* is out of date.*",
    category=DeprecationWarning,
)


@dataclass
class Task:
    method: str
    method_type: str  # learning | rules
    env_id: str
    env_kwargs: Dict[str, Any]
    level: str
    map_id: int
    weed_dist: str
    weed_num: int
    seed: int
    L_thresholds: List[float]
    render_last_frame: bool
    output_root: Path
    method_cfg: Dict[str, Any]

    @property
    def out_dir(self) -> Path:
        # Keep directory names clean: one folder per method, then per level.
        return self.output_root / f"{self.method}" / f"level-{self.level}"

    @property
    def run_stem(self) -> str:
        return f"{self.method}__{self.level}__map{self.map_id}__seed{self.seed}__{self.weed_dist}"


def load_config(path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    required_top = ["env", "scenes", "methods", "eval"]
    for k in required_top:
        if k not in cfg:
            raise ValueError(f"Missing top-level key: {k}")
    return cfg


def build_tasks(cfg: Dict[str, Any]) -> List[Task]:
    env_id = cfg["env"]["id"]
    base_kwargs = cfg["env"].get("base_kwargs", {})
    scenes = cfg["scenes"]
    seeds = scenes["seeds"]
    weed_dists = scenes["weed_dists"]
    levels = scenes["levels"]
    methods_learning = cfg["methods"].get("learning", [])
    methods_rules = cfg["methods"].get("rules", [])
    eval_cfg = cfg["eval"]

    tasks: List[Task] = []

    for level in levels:
        level_name = level["name"]
        num_obstacles_range = level["num_obstacles_range"]
        weed_num = level["weed_num"]
        map_ids = level["map_ids"]

        env_kwargs = dict(base_kwargs)
        env_kwargs["num_obstacles_range"] = tuple(num_obstacles_range)

        for map_id in map_ids:
            for dist in weed_dists:
                for seed in seeds:
                    meta_common = dict(
                        env_id=env_id,
                        env_kwargs=dict(env_kwargs),
                        level=level_name,
                        map_id=int(map_id),
                        weed_dist=dist,
                        weed_num=int(weed_num),
                        seed=int(seed),
                        L_thresholds=eval_cfg["L_thresholds"],
                        render_last_frame=bool(eval_cfg.get("render_last_frame", True)),
                        output_root=Path(eval_cfg["output_root"]),
                    )
                    for m in methods_learning:
                        tasks.append(
                            Task(
                                method=m["name"],
                                method_type="learning",
                                method_cfg=m,
                                **meta_common,
                            )
                        )
                    for m in methods_rules:
                        tasks.append(
                            Task(
                                method=m["name"],
                                method_type="rules",
                                method_cfg=m,
                                **meta_common,
                            )
                        )
    return tasks


def make_rules_planner(name: str, env, cfg: Dict[str, Any]):
    base_env = getattr(env, "unwrapped", env)
    mapping = {
        "BCP": BCPPlannerV2,
        "JUMP": JumpPlannerV2,
        "SNAKE": SnakePlannerV2,
        "R_SNAKE": RestrictedSnakePlannerV2,
        "REACT": ReactPlannerV2,
    }
    if name not in mapping:
        raise ValueError(f"Unknown rules_new2 method: {name}")
    cls = mapping[name]
    # Allow method-specific kwargs in YAML (e.g., length_limit_factor for REACT).
    kwargs = {k: v for k, v in cfg.items() if k not in {"name"}}
    planner = cls(base_env, **kwargs) if kwargs else cls(base_env)
    planner.reset(place_agent=False)
    return planner


def make_policy_builder(task: Task, cfg: Dict[str, Any]):
    if task.method_type == "learning":
        from rules_new2.policies_learning import build_learning_policy

        m_cfg = task.method_cfg

        def builder(env):
            policy_fn = build_learning_policy(
                method_cfg=m_cfg,
                env_id=task.env_id,
                env_kwargs=task.env_kwargs,
            )
            return policy_fn, None

        return builder

    m_cfg = task.method_cfg

    def builder(env):
        planner = make_rules_planner(task.method, env, m_cfg)

        def policy_fn(obs, _env):
            return planner.act()

        return policy_fn, planner

    return builder


def _normalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _meta_signature(task: Task) -> Dict[str, Any]:
    return {
        "method": task.method,
        "method_type": task.method_type,
        "level": task.level,
        "map_id": task.map_id,
        "weed_dist": task.weed_dist,
        "weed_num": task.weed_num,
        "seed": task.seed,
        "env_id": task.env_id,
        "env_kwargs": task.env_kwargs,
        "method_cfg": task.method_cfg,
    }


def should_run_task(task: Task, eval_mode: str) -> bool:
    npy_path = task.out_dir / f"{task.run_stem}.npy"
    if eval_mode == "overwrite":
        return True
    if not npy_path.exists():
        return True
    try:
        data = np.load(npy_path, allow_pickle=True).item()
    except Exception:
        return True
    saved_meta = data.get("meta", {})
    if json.dumps(_normalize(saved_meta), sort_keys=True) != json.dumps(
        _normalize(_meta_signature(task)), sort_keys=True
    ):
        return True
    for tau in task.L_thresholds:
        key = f"L{int(tau * 100)}"
        if key not in data:
            return True
    return False


def run_task(task: Task, cfg: Dict[str, Any]) -> str:
    env = None
    try:
        env = gym.make(
            task.env_id,
            render_mode="rgb_array" if task.render_last_frame else None,
            **task.env_kwargs,
        )
        meta = _meta_signature(task)

        policy_builder = make_policy_builder(task, cfg)
        run, extra = collect_episode(
            env,
            policy_builder,
            meta,
            max_steps=cfg["eval"]["max_steps"],
            thresholds=task.L_thresholds,
            render_last_frame=task.render_last_frame,
        )

        npy_path = task.out_dir / f"{task.run_stem}.npy"
        save_run_npy(run, npy_path)
        if task.render_last_frame and extra.get("render_png"):
            png_path = task.out_dir / f"{task.run_stem}.png"
            save_last_frame_png(env, png_path)

        return json.dumps(
            {
                "ok": True,
                "npy": str(npy_path),
                "done_type": run.get("done_type", "unknown"),
            },
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps(
            {
                "ok": False,
                "task": task.run_stem,
                "err": str(e),
                "traceback": traceback.format_exc(),
            },
            ensure_ascii=False,
        )
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _format_counts(done_counts: Counter[str], errors: int) -> Dict[str, int]:
    return {
        "err": int(errors),
        "suc": int(done_counts.get("env_success", 0)),
        "col": int(done_counts.get("collision", 0)),
        "tmo": int(done_counts.get("timeout", 0)),
        "idle": int(done_counts.get("planner_idle", 0)),
    }


def _shutdown_executor(executor, wait: bool, cancel_futures: bool) -> None:
    """Compatibility wrapper for executor.shutdown across Python versions."""
    try:
        executor.shutdown(wait=wait, cancel_futures=cancel_futures)
    except TypeError:
        executor.shutdown(wait=wait)


def _terminate_process_workers(executor) -> None:
    """Best-effort hard cleanup for ProcessPool workers after interruption."""
    processes = getattr(executor, "_processes", None)
    if not processes:
        return
    workers = list(processes.values())

    # First ask workers to terminate gracefully.
    for proc in workers:
        if proc is None:
            continue
        try:
            if proc.is_alive():
                proc.terminate()
        except Exception:
            pass

    # Give them a short time to exit.
    for proc in workers:
        if proc is None:
            continue
        try:
            proc.join(timeout=1.0)
        except Exception:
            pass

    # Force kill remaining workers.
    for proc in workers:
        if proc is None:
            continue
        try:
            if proc.is_alive():
                proc.kill()
        except Exception:
            pass


def save_summary_csv(agg_all, agg_by, summary_path: Path) -> None:
    """Write overall + gaussian + uniform sections into one CSV with blank lines."""
    if agg_all is None or agg_by is None:
        return

    header = list(agg_all.columns)
    gaussian = agg_by[agg_by["dist"] == "gaussian"]
    uniform = agg_by[agg_by["dist"] == "uniform"]

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(agg_all[header].values.tolist())
        writer.writerow([])
        writer.writerow([])
        if not gaussian.empty:
            writer.writerows(gaussian[header].values.tolist())
        writer.writerow([])
        writer.writerow([])
        if not uniform.empty:
            writer.writerows(uniform[header].values.tolist())


def run_task_batch(
    tasks: List[Task],
    cfg: Dict[str, Any],
    max_workers: int,
) -> Tuple[Counter[str], int]:
    if len(tasks) == 0:
        return Counter(), 0

    futures = []
    executor_cls = ProcessPoolExecutor
    try:
        ex_ctx = executor_cls(max_workers=max_workers)
    except PermissionError:
        # Some sandboxed environments disallow process semaphores.
        # Fall back to threads so the script remains usable.
        executor_cls = ThreadPoolExecutor
        ex_ctx = executor_cls(max_workers=max_workers)

    interrupted = False
    done_counts: Counter[str] = Counter()
    errors = 0
    try:
        future_to_task = {}
        for task in tasks:
            fut = ex_ctx.submit(run_task, task, cfg)
            futures.append(fut)
            future_to_task[fut] = task

        if tqdm is None:
            for fut in as_completed(futures):
                task = future_to_task[fut]
                raw = fut.result()
                try:
                    res = json.loads(raw)
                except Exception:
                    res = {"ok": False, "task": task.run_stem, "err": "non-json result", "traceback": raw}
                if not res.get("ok", False):
                    errors += 1
                    print(f"[Eval][ERROR] task={res.get('task', task.run_stem)} err={res.get('err')}")
                    tb = res.get("traceback")
                    if tb:
                        print(tb)
                else:
                    done_counts[str(res.get("done_type", "unknown"))] += 1
            return done_counts, errors

        type_order = {"rules": 0, "learning": 1}
        level_order = {lvl["name"]: i for i, lvl in enumerate(cfg["scenes"]["levels"])}
        group_totals = Counter((t.level, t.method_type) for t in tasks)
        group_keys = sorted(
            group_totals.keys(),
            key=lambda k: (
                level_order.get(k[0], 999),
                type_order.get(k[1], 999),
                k[0],
                k[1],
            ),
        )

        p_total = tqdm(total=len(tasks), desc="Total", position=0)
        group_bars = {}
        pos = 1
        for key in group_keys:
            total_k = int(group_totals[key])
            if total_k <= 0:
                continue
            group_bars[key] = tqdm(total=total_k, desc=f"{key[0]}/{key[1]}", position=pos, leave=False)
            pos += 1

        try:
            for fut in as_completed(futures):
                task = future_to_task[fut]
                raw = fut.result()
                try:
                    res = json.loads(raw)
                except Exception:
                    res = {"ok": False, "task": task.run_stem, "err": "non-json result", "traceback": raw}

                p_total.update(1)
                group_bars[(task.level, task.method_type)].update(1)

                if not res.get("ok", False):
                    errors += 1
                    tqdm.write(f"[Eval][ERROR] task={res.get('task', task.run_stem)} err={res.get('err')}")
                    tb = res.get("traceback")
                    if tb:
                        tqdm.write(tb)
                else:
                    done_counts[str(res.get("done_type", "unknown"))] += 1
                p_total.set_postfix(_format_counts(done_counts, errors))
        finally:
            p_total.close()
            for bar in group_bars.values():
                bar.close()
        return done_counts, errors
    except KeyboardInterrupt:
        interrupted = True
        print("\n[Eval] interrupted by user, stopping workers...")
        for fut in futures:
            try:
                fut.cancel()
            except Exception:
                pass
        _shutdown_executor(ex_ctx, wait=False, cancel_futures=True)
        if executor_cls is ProcessPoolExecutor:
            _terminate_process_workers(ex_ctx)
        raise
    finally:
        # Ensure no worker is leaked on any exit path.
        if interrupted:
            _shutdown_executor(ex_ctx, wait=False, cancel_futures=True)
            if executor_cls is ProcessPoolExecutor:
                _terminate_process_workers(ex_ctx)
        else:
            _shutdown_executor(ex_ctx, wait=True, cancel_futures=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    # Use eval.max_steps as the single timeout source of truth.
    # Special case: eval.max_steps < 0 means "no step-cap timeout".
    eval_max_steps = int(cfg["eval"]["max_steps"])
    env_cfg = cfg.setdefault("env", {})
    base_kwargs = env_cfg.setdefault("base_kwargs", {})
    if eval_max_steps < 0:
        if (
            "max_episode_steps" in base_kwargs
            and int(base_kwargs["max_episode_steps"]) != TIMEOUT_DISABLED_MAX_STEPS
        ):
            print(
                "[Eval] override env.base_kwargs.max_episode_steps="
                f"{base_kwargs['max_episode_steps']} -> {TIMEOUT_DISABLED_MAX_STEPS}"
            )
        base_kwargs["max_episode_steps"] = TIMEOUT_DISABLED_MAX_STEPS
        print(
            "[Eval] eval.max_steps < 0: disable step-cap timeout "
            f"(env.max_episode_steps={TIMEOUT_DISABLED_MAX_STEPS})"
        )
    else:
        if (
            "max_episode_steps" in base_kwargs
            and int(base_kwargs["max_episode_steps"]) != eval_max_steps
        ):
            print(
                "[Eval] override env.base_kwargs.max_episode_steps="
                f"{base_kwargs['max_episode_steps']} -> eval.max_steps={eval_max_steps}"
            )
        base_kwargs["max_episode_steps"] = eval_max_steps

    tasks_all = build_tasks(cfg)

    eval_cfg = cfg["eval"]
    eval_mode = str(eval_cfg.get("mode", "incremental"))
    if eval_mode not in {"incremental", "overwrite"}:
        raise ValueError(f"Invalid eval.mode={eval_mode}, expected incremental|overwrite")
    output_root = Path(eval_cfg["output_root"])
    save_config_snapshot(cfg, output_root)

    tasks = [t for t in tasks_all if should_run_task(t, eval_mode)]
    skipped = len(tasks_all) - len(tasks)
    print(f"[Eval] total tasks={len(tasks_all)}, to_run={len(tasks)}, skipped={skipped}")

    workers_rules = int(eval_cfg.get("workers_rules", 1))
    workers_net = int(eval_cfg.get("workers_net", eval_cfg.get("workers_learning", 1)))

    rules_tasks = [t for t in tasks if t.method_type == "rules"]
    learning_tasks = [t for t in tasks if t.method_type == "learning"]

    done_counts: Counter[str] = Counter()
    errors = 0

    if rules_tasks:
        print(f"[Eval] running rules tasks={len(rules_tasks)}, workers_rules={workers_rules}")
        c, e = run_task_batch(rules_tasks, cfg, max_workers=max(workers_rules, 1))
        done_counts.update(c)
        errors += e
    if learning_tasks:
        print(f"[Eval] running learning tasks={len(learning_tasks)}, workers_net={workers_net}")
        c, e = run_task_batch(learning_tasks, cfg, max_workers=max(workers_net, 1))
        done_counts.update(c)
        errors += e

    if len(tasks) > 0:
        print(f"[Eval] done stats: {_format_counts(done_counts, errors)}")

    agg_all, agg_by = aggregate_runs_to_summary(output_root)
    if agg_all is not None and len(agg_all) > 0:
        summary_path = output_root / "summary.csv"
        save_summary_csv(agg_all, agg_by, summary_path)
        meta_path = output_root / "summary_meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "eval_mode": eval_mode,
                    "L_thresholds": eval_cfg.get("L_thresholds", []),
                    "max_steps": eval_cfg.get("max_steps"),
                    "workers_rules": workers_rules,
                    "workers_net": workers_net,
                    "tasks_total": len(tasks_all),
                    "tasks_ran": len(tasks),
                    "tasks_skipped": skipped,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        print(f"[Eval] summary saved to {summary_path}")
    else:
        print("[Eval] no runs to summarize")


if __name__ == "__main__":
    main()
