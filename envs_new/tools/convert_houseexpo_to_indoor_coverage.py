#!/usr/bin/env python3
"""
HouseExpo 数据集转换脚本 - 将 HouseExpo PNG 转换为 NewCppRL 格式

处理逻辑：
1. 等比缩放：长边→320，短边等比
2. 居中放置到 400×400 黑色画布（边界外为障碍物）
3. 生成配对的 field/obstacle PNG

使用方法：
    python envs_new/tools/convert_houseexpo_to_indoor_coverage.py
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def convert_houseexpo(src_dir, output_dir, max_edge=320, canvas_size=400):
    """
    转换 HouseExpo PNG 到 indoor_coverage 格式

    参数：
        src_dir: HouseExpo 目录
        output_dir: 输出目录（包含 field/ 和 obstacle/ 子目录）
        max_edge: 缩放后长边尺寸（默认320，留出边界空间）
        canvas_size: 画布尺寸（默认400）
    """
    field_dir = Path(output_dir) / 'field'
    obstacle_dir = Path(output_dir) / 'obstacle'
    field_dir.mkdir(parents=True, exist_ok=True)
    obstacle_dir.mkdir(parents=True, exist_ok=True)

    # 智能接续：找现有最大序号
    existing = list(field_dir.glob('field_*.png'))
    if existing:
        start_idx = max(int(f.stem.split('_')[1]) for f in existing) + 1
    else:
        start_idx = 0

    # 收集 PNG 文件（排除 Zone.Identifier）
    pngs = sorted([f for f in Path(src_dir).glob('*.png') if 'Zone' not in f.name])

    print(f"源目录: {src_dir}")
    print(f"输出目录: {output_dir}")
    print(f"找到 {len(pngs)} 个 PNG 文件")
    print(f"起始序号: {start_idx}")
    print(f"缩放参数: 长边→{max_edge}, 画布→{canvas_size}×{canvas_size}")
    print()

    for i, png_file in enumerate(tqdm(pngs, desc="转换中")):
        idx = start_idx + i

        # 读取图片
        img = np.array(Image.open(png_file).convert('L'))  # 确保灰度
        h, w = img.shape

        # 等比缩放：长边→max_edge
        scale = max_edge / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # 二值化（确保只有 0 和 255）
        resized = ((resized > 127) * 255).astype(np.uint8)

        # 创建黑色画布（全障碍物），居中放置
        field = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        y_off = (canvas_size - new_h) // 2
        x_off = (canvas_size - new_w) // 2
        field[y_off:y_off+new_h, x_off:x_off+new_w] = resized

        # obstacle = 反转
        obstacle = 255 - field

        # 保存
        cv2.imwrite(str(field_dir / f'field_{idx}.png'), field)
        cv2.imwrite(str(obstacle_dir / f'obstacle_{idx}.png'), obstacle)

    print(f"\n✅ 转换完成！共处理 {len(pngs)} 个场景")
    print(f"   序号范围: {start_idx} ~ {start_idx + len(pngs) - 1}")
    print(f"   Field地图: {field_dir}")
    print(f"   Obstacle地图: {obstacle_dir}")


if __name__ == "__main__":
    # 配置路径
    SRC_DIR = '/home/lzh/HouseExpo'
    OUTPUT_DIR = '/home/lzh/NewCppRL/envs_new/maps/indoor_coverage'
    MAX_EDGE = 320      # 长边缩放到 320，留出边界空间
    CANVAS_SIZE = 400   # 与默认场景尺寸一致

    convert_houseexpo(SRC_DIR, OUTPUT_DIR, MAX_EDGE, CANVAS_SIZE)
