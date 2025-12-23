"""
统一评估入口（envs_new）。

Usage:
    python -m rules_new.eval_runner --config rules_new/configs/eval_new_env.yaml

设计原则：
- learning / rules 共享同一条 rollout + 指标采集流水线
- 多进程并行，每个任务独立创建 env / policy，跑完即落盘
- fast-fail：缺关键字段直接报错
"""
from __future__ import annotations

import argparse
import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
import yaml

import envs_new  # noqa: F401  # register gym envs
from rules_new.metrics_io import collect_episode, save_run_npy, save_last_frame_png, aggregate_runs_to_summary
from rules_new.policies_rules import build_rule_policy


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

    @property
    def out_dir(self) -> Path:
        return self.output_root / f"alg-{self.method}" / f"level-{self.level}"

    @property
    def run_stem(self) -> str:
        return f"{self.method}__{self.level}__map{self.map_id}__seed{self.seed}__{self.weed_dist}"


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
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
                        env_kwargs=env_kwargs,
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
                        tasks.append(Task(method=m["name"], method_type="learning", **meta_common))
                    for m in methods_rules:
                        tasks.append(Task(method=m["name"], method_type="rules", **meta_common))
    return tasks


def make_policy(task: Task, cfg: Dict[str, Any]):
    if task.method_type == "learning":
        # 为避免在纯 rules 评估时引入不必要的 Torch 依赖，这里按需延迟导入。
        from rules_new.policies_learning import build_learning_policy

        m_cfg = next(m for m in cfg["methods"]["learning"] if m["name"] == task.method)
        return build_learning_policy(m_cfg, task.env_id)
    else:
        return build_rule_policy(task.method)


def run_task(task: Task, cfg: Dict[str, Any]) -> str:
    """
    单个任务的执行函数，在子进程内运行。
    返回值：任务完成的 npy 路径（或错误信息）。
    """
    try:
        policy_fn = make_policy(task, cfg)

        env = gym.make(task.env_id, render_mode="rgb_array" if task.render_last_frame else None, **task.env_kwargs)
        meta = {
            "method": task.method,
            "method_type": task.method_type,
            "level": task.level,
            "map_id": task.map_id,
            "weed_dist": task.weed_dist,
            "weed_num": task.weed_num,
            "seed": task.seed,
            "env_id": task.env_id,
            "env_kwargs": task.env_kwargs,
        }
        run, extra = collect_episode(env, policy_fn, meta, max_steps=cfg["eval"]["max_steps"],
                                     thresholds=task.L_thresholds,
                                     render_last_frame=task.render_last_frame)

        npy_path = task.out_dir / f"{task.run_stem}.npy"
        save_run_npy(run, npy_path)
        if task.render_last_frame and extra.get("render_png"):
            png_path = task.out_dir / f"{task.run_stem}.png"
            save_last_frame_png(env, png_path)

        env.close()
        return str(npy_path)
    except Exception as e:
        return f"ERROR: task={task.run_stem}, err={e}\n{traceback.format_exc()}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    tasks = build_tasks(cfg)
    print(f"[Eval] total tasks: {len(tasks)}")

    workers_learning = cfg["eval"].get("workers_learning", 1)
    workers_rules = cfg["eval"].get("workers_rules", 1)

    # 简化：统一放到一个进程池，大小 = max(workers_learning, workers_rules)
    pool_size = max(workers_learning, workers_rules, 1)

    futures = []
    with ProcessPoolExecutor(max_workers=pool_size) as ex:
        for t in tasks:
            futures.append(ex.submit(run_task, t, cfg))

        for f in as_completed(futures):
            res = f.result()
            print(res)

    summary = aggregate_runs_to_summary(Path(cfg["eval"]["output_root"]))
    if summary is not None and len(summary) > 0:
        summary_path = Path(cfg["eval"]["output_root"]) / "summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"[Eval] summary saved to {summary_path}")
    else:
        print("[Eval] no runs to summarize")


if __name__ == "__main__":
    main()
