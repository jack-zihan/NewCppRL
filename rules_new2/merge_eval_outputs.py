from __future__ import annotations

"""
合并多个服务器产出的 rules_new2 评估结果。

用途
----
当同一套评估任务被拆到多台服务器运行时，每台服务器通常只产出一部分结果，
或者会有少量重叠结果。这个脚本将这些子目录合并成一个“完整评估目录”，
使其在结构和统计上都等价于“单台服务器一次性跑完”的结果。

合并规则
--------
1. 输入目录通常形如：
   rules_new2/opcpp_eval_metric/1/full_opcpp_eval
   rules_new2/opcpp_eval_metric/2/full_opcpp_eval
   rules_new2/opcpp_eval_metric/3/full_opcpp_eval

2. 以相对路径作为 run 唯一键，例如：
   JUMP/level-medium/JUMP__medium__map21__seed25__gaussian.npy

3. 对重复 run：
   - 保留第一份；
   - 后续重复的 npy/png 直接忽略；
   - 最终 summary 基于合并后的唯一 run 全量重算。

4. `config_used.yaml` 不是简单复制某一份，而是取三份配置的并集：
   - levels/map_ids 取并集；
   - seeds/weed_dists 取并集且保持原顺序；
   - methods 取并集；
   - `eval.output_root` 改写为合并后的目标目录。

用法
----
默认用法（合并 1/2/3 到同级的 full_opcpp_eval）：

    python -m rules_new2.merge_eval_outputs

显式指定来源与目标：

    python -m rules_new2.merge_eval_outputs \
        --parent-root rules_new2/opcpp_eval_metric \
        --sources 1 2 3 \
        --child-root full_opcpp_eval \
        --dest-name full_opcpp_eval
"""

import argparse
import csv
import json
import os
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

from rules_new2.metrics_io import aggregate_runs_to_summary, build_success_contact_sheets


def load_yaml(path: Path) -> Dict[str, Any]:
    """读取 YAML 配置。"""
    return yaml.safe_load(path.read_text())


def save_yaml(path: Path, data: Dict[str, Any]) -> None:
    """保存 YAML 配置，保留键顺序。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def save_summary_csv(agg_all, agg_by, summary_path: Path) -> None:
    """写出和 eval_runner 相同格式的 summary.csv。"""
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


def unique_keep_order(items: Iterable[Any]) -> List[Any]:
    """去重但保持首次出现顺序。"""
    return list(OrderedDict((item, None) for item in items).keys())


def merge_configs(configs: List[Dict[str, Any]], output_root: str) -> Dict[str, Any]:
    """将多份 config_used.yaml 合并成一份完整配置。

    这里要求：
    - 同名 level 的 `num_obstacles_range` 与 `weed_num` 必须一致；
    - 同名 method 视为同一个方法；
    - seeds / weed_dists / map_ids 做有序并集。
    """
    if not configs:
        raise ValueError("No config files found to merge.")

    base = json.loads(json.dumps(configs[0]))

    weed_dists: List[str] = []
    seeds: List[int] = []
    levels_by_name: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    learning_by_name: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    rules_by_name: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    for cfg in configs:
        weed_dists.extend(cfg.get("scenes", {}).get("weed_dists", []))
        seeds.extend(cfg.get("scenes", {}).get("seeds", []))

        for level in cfg.get("scenes", {}).get("levels", []):
            name = str(level["name"])
            if name in levels_by_name:
                prev = levels_by_name[name]
                if (
                    list(prev.get("num_obstacles_range", [])) != list(level.get("num_obstacles_range", []))
                    or int(prev.get("weed_num")) != int(level.get("weed_num"))
                ):
                    raise ValueError(f"Conflicting level definition for {name}")
                prev["map_ids"] = unique_keep_order(list(prev.get("map_ids", [])) + list(level.get("map_ids", [])))
            else:
                levels_by_name[name] = json.loads(json.dumps(level))

        for method in cfg.get("methods", {}).get("learning", []):
            learning_by_name.setdefault(str(method["name"]), json.loads(json.dumps(method)))
        for method in cfg.get("methods", {}).get("rules", []):
            rules_by_name.setdefault(str(method["name"]), json.loads(json.dumps(method)))

    base["scenes"]["weed_dists"] = unique_keep_order(weed_dists)
    base["scenes"]["seeds"] = unique_keep_order(seeds)
    base["scenes"]["levels"] = list(levels_by_name.values())
    base["methods"]["learning"] = list(learning_by_name.values())
    base["methods"]["rules"] = list(rules_by_name.values())
    base["eval"]["output_root"] = output_root
    return base


def link_or_copy(src: Path, dst: Path) -> str:
    """优先硬链接，失败时再复制。

    这样既节省空间，也保持脚本足够简单。对使用者来说，结果目录的行为与
    普通文件目录一致，不需要额外理解复杂的数据组织方式。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "linked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def iter_run_files(root: Path) -> Iterable[Tuple[Path, Path]]:
    """枚举一个评估根目录下的所有 run 文件（仅 npy/png）。"""
    for path in sorted(root.glob("*/*/*")):
        if path.suffix.lower() not in {".npy", ".png"}:
            continue
        # 仅把真实 run 文件纳入合并；像 success_sheet_01.png 这样的聚合图
        # 是后处理产物，不应当视为单次 run 的可视化。
        if "__" not in path.stem:
            continue
        yield path, path.relative_to(root)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="合并多个 rules_new2 评估输出目录，并重算 summary。",
    )
    parser.add_argument(
        "--parent-root",
        type=Path,
        default=Path("rules_new2/opcpp_eval_metric"),
        help="Parent directory containing per-server folders like 1/2/3.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["1", "2", "3"],
        help="Source subdirectories under parent-root.",
    )
    parser.add_argument(
        "--child-root",
        type=str,
        default="full_opcpp_eval",
        help="Eval output directory name inside each source folder.",
    )
    parser.add_argument(
        "--dest-name",
        type=str,
        default="full_opcpp_eval",
        help="Merged output directory name under parent-root.",
    )
    args = parser.parse_args()

    # source_roots 是每台服务器各自产出的评估目录。
    source_roots = [args.parent_root / src / args.child_root for src in args.sources]
    for root in source_roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Missing source root: {root}")

    # 目标目录必须是全新的，避免把旧结果混进来。
    dest_root = args.parent_root / args.dest_name
    if dest_root.exists():
        raise FileExistsError(f"Destination already exists: {dest_root}")

    # 先合并配置，再拷贝 run 文件，这样后续 summary 的面积缩放等统计
    # 都会基于“完整地图并集”的 config_used.yaml 来重算。
    configs = [load_yaml(root / "config_used.yaml") for root in source_roots]
    merged_cfg = merge_configs(configs, str(dest_root))
    save_yaml(dest_root / "config_used.yaml", merged_cfg)

    seen: Dict[str, Path] = {}
    stats = {
        "sources": [str(root) for root in source_roots],
        "unique_npy": 0,
        "unique_png": 0,
        "duplicate_npy": 0,
        "duplicate_png": 0,
        "linked": 0,
        "copied": 0,
    }

    for root in source_roots:
        for src, rel in iter_run_files(root):
            rel_str = str(rel)
            dst = dest_root / rel
            if rel_str in seen:
                if src.suffix.lower() == ".npy":
                    stats["duplicate_npy"] += 1
                else:
                    stats["duplicate_png"] += 1
                continue

            mode = link_or_copy(src, dst)
            stats[mode] += 1
            seen[rel_str] = src
            if src.suffix.lower() == ".npy":
                stats["unique_npy"] += 1
            else:
                stats["unique_png"] += 1

    # summary 必须基于合并后的原始 npy 全量重算，不能拼接旧 summary。
    agg_all, agg_by = aggregate_runs_to_summary(dest_root)
    if agg_all is None or agg_by is None or len(agg_all) == 0:
        raise RuntimeError(f"No merged runs found under {dest_root}")

    save_summary_csv(agg_all, agg_by, dest_root / "summary.csv")
    build_success_contact_sheets(dest_root)

    total_tasks = 0
    weed_dists = merged_cfg.get("scenes", {}).get("weed_dists", [])
    seeds = merged_cfg.get("scenes", {}).get("seeds", [])
    levels = merged_cfg.get("scenes", {}).get("levels", [])
    n_methods = len(merged_cfg.get("methods", {}).get("learning", [])) + len(
        merged_cfg.get("methods", {}).get("rules", [])
    )
    for level in levels:
        total_tasks += len(level.get("map_ids", [])) * len(seeds) * len(weed_dists) * n_methods

    meta = {
        "merged_from": stats["sources"],
        "tasks_total": total_tasks,
        "tasks_ran": stats["unique_npy"],
        "tasks_skipped": 0,
        "duplicate_npy_ignored": stats["duplicate_npy"],
        "duplicate_png_ignored": stats["duplicate_png"],
        "file_mode_counts": {"linked": stats["linked"], "copied": stats["copied"]},
        "eval_mode": "merged_union",
        "L_thresholds": merged_cfg.get("eval", {}).get("L_thresholds", []),
        "max_steps": merged_cfg.get("eval", {}).get("max_steps"),
        "workers_rules": merged_cfg.get("eval", {}).get("workers_rules"),
        "workers_net": merged_cfg.get("eval", {}).get("workers_net", merged_cfg.get("eval", {}).get("workers_learning")),
    }
    (dest_root / "summary_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print(f"[Merge] dest={dest_root}")
    print(
        "[Merge] "
        f"unique_npy={stats['unique_npy']} unique_png={stats['unique_png']} "
        f"duplicate_npy={stats['duplicate_npy']} duplicate_png={stats['duplicate_png']}"
    )
    print(f"[Merge] summary saved to {dest_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
