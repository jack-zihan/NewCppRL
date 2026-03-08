from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build gallery pages for weed_coverage field maps using map_id labels."
    )
    parser.add_argument(
        "--field-dir",
        type=Path,
        default=Path("envs_new/maps/weed_coverage/field"),
        help="Directory containing field_*.png",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for gallery pages; default is sibling 'field_visualization'",
    )
    parser.add_argument("--rows", type=int, default=8, help="Rows per page")
    parser.add_argument("--cols", type=int, default=8, help="Cols per page")
    parser.add_argument(
        "--thumb-size",
        type=int,
        default=96,
        help="Square size of each resized field map",
    )
    parser.add_argument(
        "--tile-header",
        type=int,
        default=24,
        help="Text area height above each thumbnail",
    )
    parser.add_argument("--gap", type=int, default=8, help="Gap between tiles")
    return parser.parse_args()


def load_binary_map(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return (image > 0).astype(np.uint8) * 255


def draw_tile(
    field_map: np.ndarray,
    map_id: int,
    thumb_size: int,
    tile_header: int,
) -> np.ndarray:
    tile_h = tile_header + thumb_size
    tile_w = thumb_size
    tile = np.full((tile_h, tile_w, 3), 36, dtype=np.uint8)

    thumb = cv2.resize(field_map, (thumb_size, thumb_size), interpolation=cv2.INTER_NEAREST)
    thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
    tile[tile_header:, :, :] = thumb_rgb

    cv2.putText(
        tile,
        f"map_id={map_id}",
        (4, tile_header - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 220, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def build_page(
    tiles: list[np.ndarray],
    rows: int,
    cols: int,
    gap: int,
    title: str,
) -> np.ndarray:
    tile_h, tile_w = tiles[0].shape[:2]
    title_h = 34
    page_h = title_h + gap + rows * tile_h + (rows + 1) * gap
    page_w = cols * tile_w + (cols + 1) * gap
    page = np.full((page_h, page_w, 3), 20, dtype=np.uint8)

    cv2.putText(
        page,
        title,
        (gap, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )

    start_y = title_h + gap
    for i, tile in enumerate(tiles):
        r = i // cols
        c = i % cols
        y = start_y + gap + r * (tile_h + gap)
        x = gap + c * (tile_w + gap)
        page[y : y + tile_h, x : x + tile_w] = tile
    return page


def main() -> None:
    args = parse_args()
    field_dir: Path = args.field_dir
    if not field_dir.exists():
        raise FileNotFoundError(f"field directory not found: {field_dir}")

    out_dir = args.out_dir or field_dir.parent / "field_visualization"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in field_dir.iterdir() if p.suffix.lower() == ".png"])
    if not files:
        raise RuntimeError(f"No png files found in {field_dir}")

    per_page = args.rows * args.cols
    num_pages = math.ceil(len(files) / per_page)

    print(f"[gallery] field_dir={field_dir}")
    print(f"[gallery] total_maps={len(files)}, per_page={per_page}, pages={num_pages}")
    print(f"[gallery] out_dir={out_dir}")

    for page_idx in range(num_pages):
        start = page_idx * per_page
        end = min((page_idx + 1) * per_page, len(files))
        page_files = files[start:end]

        tiles: list[np.ndarray] = []
        for map_id, path in enumerate(page_files, start=start):
            field_map = load_binary_map(path)
            tile = draw_tile(field_map, map_id, args.thumb_size, args.tile_header)
            tiles.append(tile)

        page_title = f"Field Gallery  map_id [{start}, {end - 1}]  count={len(page_files)}"
        page = build_page(
            tiles=tiles,
            rows=args.rows,
            cols=args.cols,
            gap=args.gap,
            title=page_title,
        )
        out_path = out_dir / f"field_gallery_{page_idx:03d}_mapid_{start:04d}_{end-1:04d}.png"
        cv2.imwrite(str(out_path), page)
        print(f"[gallery] saved: {out_path}")


if __name__ == "__main__":
    main()
