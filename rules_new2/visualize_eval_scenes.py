from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

import envs_new  # noqa: F401
from rules_new2.metrics_io import canonical_start_pose


def load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text())


def fit_frame(frame: np.ndarray, width: int, height: int) -> Image.Image:
    img = Image.fromarray(frame)
    scale = min(width / img.width, height / img.height)
    new_w = max(1, int(round(img.width * scale)))
    new_h = max(1, int(round(img.height * scale)))
    img = img.resize((new_w, new_h), Image.Resampling.NEAREST)

    canvas = Image.new("RGB", (width, height), "#111111")
    ox = (width - new_w) // 2
    oy = (height - new_h) // 2
    canvas.paste(img, (ox, oy))
    return canvas


def render_initial_frame(
    env,
    map_id: int,
    seed: int,
    weed_dist: str,
    weed_num: int,
) -> np.ndarray:
    scene_opts = {
        "map_id": map_id,
        "weed_distribution": weed_dist,
        "weed_count": weed_num,
    }

    env.reset(seed=seed, options=scene_opts)
    wx0, wy0, theta0 = canonical_start_pose(env)
    env.reset(
        seed=seed,
        options={**scene_opts, "initial_position": (wx0, wy0), "initial_direction": theta0},
    )

    frame = env.render()
    if frame is None:
        raise RuntimeError("env.render() returned None")
    return frame


def build_level_overview(
    cfg: Dict[str, Any],
    level_cfg: Dict[str, Any],
    output_path: Path,
) -> None:
    env_id = cfg["env"]["id"]
    base_kwargs = dict(cfg["env"].get("base_kwargs", {}))
    base_kwargs["num_obstacles_range"] = tuple(level_cfg["num_obstacles_range"])

    seeds: List[int] = [int(seed) for seed in cfg["scenes"]["seeds"]]
    weed_dists: List[str] = [str(dist) for dist in cfg["scenes"]["weed_dists"]]
    map_ids: List[int] = [int(map_id) for map_id in level_cfg["map_ids"]]
    weed_num = int(level_cfg["weed_num"])

    if weed_dists != ["gaussian", "uniform"]:
        raise ValueError("当前脚本固定使用 gaussian/uniform 双视图，请保持配置中的 weed_dists 为这两个值")

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    sub_w = 150
    sub_h = 150
    cell_w = sub_w
    cell_h = sub_h * 2 + 34
    left_w = 118
    top_h = 56
    gap_x = 14
    gap_y = 18

    canvas_w = left_w + len(seeds) * (cell_w + gap_x) + gap_x
    canvas_h = top_h + len(map_ids) * (cell_h + gap_y) + gap_y
    canvas = Image.new("RGB", (canvas_w, canvas_h), "#f4f4f1")
    draw = ImageDraw.Draw(canvas)

    title = (
        f"{level_cfg['name']}  obstacles={tuple(level_cfg['num_obstacles_range'])}  "
        f"weed_num={weed_num}"
    )
    draw.text((gap_x, 12), title, fill="#111111", font=title_font)

    for col, seed in enumerate(seeds):
        x = left_w + gap_x + col * (cell_w + gap_x)
        draw.text((x, top_h - 28), f"seed={seed}", fill="#111111", font=font)

    env = gym.make(env_id, render_mode="rgb_array", **base_kwargs)
    try:
        for row, map_id in enumerate(map_ids):
            y = top_h + row * (cell_h + gap_y)
            draw.text((gap_x, y + 8), f"map_id={map_id}", fill="#111111", font=font)

            for col, seed in enumerate(seeds):
                x = left_w + gap_x + col * (cell_w + gap_x)
                draw.rectangle([x, y, x + cell_w, y + cell_h], outline="#333333", width=1)

                frame_g = render_initial_frame(env, map_id, seed, "gaussian", weed_num)
                frame_u = render_initial_frame(env, map_id, seed, "uniform", weed_num)

                tile_g = fit_frame(frame_g, sub_w - 2, sub_h - 2)
                tile_u = fit_frame(frame_u, sub_w - 2, sub_h - 2)

                canvas.paste(tile_g, (x + 1, y + 1))
                canvas.paste(tile_u, (x + 1, y + sub_h + 17))

                draw.text((x + 6, y + sub_h - 18), "G", fill="#ffffff", font=font)
                draw.text((x + 6, y + cell_h - 18), "U", fill="#ffffff", font=font)
    finally:
        env.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="rules_new2/configs/opcpp_eval.yaml")
    parser.add_argument("--output-dir", type=str, default="rules_new2/eval_scene_overviews")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    output_dir = Path(args.output_dir)

    for level_cfg in cfg["scenes"]["levels"]:
        level_name = str(level_cfg["name"])
        out_path = output_dir / f"{level_name}_scene_overview.png"
        build_level_overview(cfg, level_cfg, out_path)
        print(out_path)


if __name__ == "__main__":
    main()
