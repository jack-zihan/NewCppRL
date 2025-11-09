#!/usr/bin/env python3
"""
PGM场景转换脚本 - 将Explore-Bench的PGM室内场景转换为NewCppRL格式

功能：
1. 读取PGM文件（250×250）
2. Resize到400×400（与默认场景尺寸一致）
3. 分离field和obstacle
4. 保存为配对的PNG文件

使用方法：
    python scripts/convert_pgm_to_indoor_coverage.py
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path


def convert_pgm_to_indoor_maps(pgm_dir, output_dir, target_size=(400, 400)):
    """
    转换PGM场景到indoor_coverage格式

    参数：
        pgm_dir: PGM文件目录
        output_dir: 输出目录
        target_size: 目标尺寸 (height, width)
    """
    pgm_files = sorted(Path(pgm_dir).glob('*.pgm'))

    # 创建输出目录
    field_dir = Path(output_dir) / 'field'
    obstacle_dir = Path(output_dir) / 'obstacle'
    field_dir.mkdir(parents=True, exist_ok=True)
    obstacle_dir.mkdir(parents=True, exist_ok=True)

    print(f"找到 {len(pgm_files)} 个PGM文件")
    print(f"目标尺寸: {target_size[0]}×{target_size[1]}")
    print(f"输出目录: {output_dir}\n")

    for idx, pgm_file in enumerate(pgm_files):
        print(f"[{idx+1}/{len(pgm_files)}] 处理: {pgm_file.name}")

        # 读取PGM文件
        pgm_array = np.array(Image.open(pgm_file))
        original_shape = pgm_array.shape
        print(f"  原始尺寸: {original_shape[0]}×{original_shape[1]}")

        # 统计PGM值分布
        unique, counts = np.unique(pgm_array, return_counts=True)
        value_dist = dict(zip(unique, counts))
        total_pixels = pgm_array.size
        print(f"  像素值分布:")
        for value, count in value_dist.items():
            percentage = (count / total_pixels) * 100
            label = "障碍物" if value == 0 else ("未知" if value == 205 else "自由空间")
            print(f"    {value:3d} ({label:6s}): {count:6d} pixels ({percentage:5.2f}%)")

        # Resize到目标尺寸（使用最近邻插值保持二值特性）
        pgm_resized = cv2.resize(pgm_array, (target_size[1], target_size[0]),
                                 interpolation=cv2.INTER_NEAREST)

        # 分离field和obstacle
        # Field: 自由空间（254） → 255
        # Obstacle: 障碍物（0）+ 未知（205） → 255
        field_map = (pgm_resized == 254).astype(np.uint8) * 255
        obstacle_map = (pgm_resized != 254).astype(np.uint8) * 255

        # 统计转换后的占比
        field_pixels = np.sum(field_map > 0)
        obstacle_pixels = np.sum(obstacle_map > 0)
        field_pct = (field_pixels / field_map.size) * 100
        obstacle_pct = (obstacle_pixels / obstacle_map.size) * 100

        print(f"  转换后: field={field_pixels} ({field_pct:.2f}%), "
              f"obstacle={obstacle_pixels} ({obstacle_pct:.2f}%)")

        # 保存配对文件
        field_path = field_dir / f'field_{idx}.png'
        obstacle_path = obstacle_dir / f'obstacle_{idx}.png'

        cv2.imwrite(str(field_path), field_map)
        cv2.imwrite(str(obstacle_path), obstacle_map)

        print(f"  ✓ 已保存: field_{idx}.png + obstacle_{idx}.png\n")

    print(f"✅ 转换完成！共处理 {len(pgm_files)} 个场景")
    print(f"   Field地图: {field_dir}")
    print(f"   Obstacle地图: {obstacle_dir}")


if __name__ == "__main__":
    # 配置路径
    PGM_DIR = '/home/lzh/Explore-Bench/onpolicy/onpolicy/envs/GridEnv/datasets'
    OUTPUT_DIR = '/home/lzh/NewCppRL/envs_new/maps/indoor_coverage'
    TARGET_SIZE = (400, 400)  # 与默认场景尺寸一致

    # 执行转换
    convert_pgm_to_indoor_maps(PGM_DIR, OUTPUT_DIR, TARGET_SIZE)
