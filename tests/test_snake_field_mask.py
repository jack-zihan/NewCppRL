#!/usr/bin/env python
"""
诊断测试：验证SNAKE的forward_line生成逻辑和field_mask检查
"""
import sys
sys.path.insert(0, '/home/lzh/NewCppRL')

import gymnasium as gym
import numpy as np
import math
import envs_new  # This registers the environments

def test_snake_field_mask():
    """测试SNAKE的field_mask和forward_line生成"""
    from rules_new.policies_rules import SnakeController

    # 创建环境 - v4 是纯覆盖环境，默认无障碍物
    env = gym.make('NewPasture-v4')
    obs, info = env.reset()

    # 创建控制器
    controller = SnakeController()
    controller.reset(env)

    # 打印field_mask信息
    print("=== Field Mask 信息 ===")
    print(f"field_mask shape: {controller._field_mask.shape}")
    print(f"field_mask dtype: {controller._field_mask.dtype}")
    print(f"field_mask非零像素数: {np.sum(controller._field_mask > 0)}")

    H, W = controller._field_mask.shape

    # 获取field的实际边界
    ys, xs = np.nonzero(controller._field_mask > 0)
    print(f"\nField边界: x=[{xs.min()}, {xs.max()}], y=[{ys.min()}, {ys.max()}]")
    print(f"Field中心: ({(xs.min()+xs.max())//2}, {(ys.min()+ys.max())//2})")

    # 控制器属性
    print(f"\n=== 控制器属性 ===")
    print(f"_u (主方向): {controller._u}")
    print(f"_v (垂直方向): {controller._v}")
    print(f"main_angle: {math.degrees(controller.main_angle):.1f}°")
    print(f"_pass_dir: {controller._pass_dir}")
    print(f"_diag_length: {controller._diag_length:.1f}")

    # 找一个真正在field内的点 - 使用质心而非bounding box中心
    print("\n=== 寻找真正在field内的点 ===")
    bbox_cx = (xs.min() + xs.max()) // 2
    bbox_cy = (ys.min() + ys.max()) // 2
    print(f"Bounding box中心: ({bbox_cx}, {bbox_cy})")
    print(f"Bounding box中心 field_mask[{bbox_cy}, {bbox_cx}] = {controller._field_mask[bbox_cy, bbox_cx]}")

    # 使用质心（所有field像素的平均位置）
    centroid_x = int(np.mean(xs))
    centroid_y = int(np.mean(ys))
    print(f"\nField质心: ({centroid_x}, {centroid_y})")
    print(f"Field质心 field_mask[{centroid_y}, {centroid_x}] = {controller._field_mask[centroid_y, centroid_x]}")

    # 使用质心作为起点
    cx, cy = centroid_x, centroid_y

    # 如果质心也不在field内，找一个确定在field内的点
    if controller._field_mask[cy, cx] == 0:
        print("质心也不在field内！使用第一个field像素")
        cx, cy = xs[0], ys[0]
        print(f"使用: ({cx}, {cy}), field_mask[{cy}, {cx}] = {controller._field_mask[cy, cx]}")

    # 测试forward_line生成
    print("\n=== 测试Forward Line生成 ===")

    # 验证起点在field内
    print(f"起点: ({cx}, {cy})")
    print(f"起点 field_mask[{cy}, {cx}] = {controller._field_mask[cy, cx]}")

    # 获取主方向
    rad_vec = controller._u if controller._pass_dir >= 0 else -controller._u
    rad_vec_norm = rad_vec / np.linalg.norm(rad_vec)
    print(f"方向向量: {rad_vec_norm}")

    # 生成forward_line（复制SNAKE的逻辑）
    forward_line = []
    max_len = controller._diag_length * 2.0
    start = np.array([cx, cy], dtype=float)

    for s in np.arange(0.0, max_len, 1.0):
        pt = start + s * rad_vec_norm
        ix, iy = int(round(pt[0])), int(round(pt[1]))

        in_bounds = 0 <= ix < W and 0 <= iy < H
        in_field = in_bounds and controller._field_mask[iy, ix] > 0

        if in_field:
            forward_line.append((float(pt[0]), float(pt[1])))
        else:
            print(f"\n在 s={s:.1f} 处停止")
            print(f"  坐标: ({ix}, {iy})")
            print(f"  在图像边界内: {in_bounds}")
            if in_bounds:
                print(f"  field_mask[{iy}, {ix}] = {controller._field_mask[iy, ix]}")
            break

    print(f"\nForward line长度: {len(forward_line)} 点")
    if len(forward_line) > 0:
        print(f"起始点: {forward_line[0]}")
        print(f"结束点: {forward_line[-1]}")

    # 验证所有forward_line点都在field内
    print("\n=== 验证Forward Line点 ===")
    outside_count = 0
    for i, (fx, fy) in enumerate(forward_line):
        ix, iy = int(round(fx)), int(round(fy))
        in_bounds = 0 <= ix < W and 0 <= iy < H
        in_field = in_bounds and controller._field_mask[iy, ix] > 0
        if not in_field:
            outside_count += 1
            if outside_count <= 5:  # 只打印前5个
                print(f"  [{i}] ({fx:.1f}, {fy:.1f}) 在field外! bounds={in_bounds}")

    print(f"\n共 {outside_count}/{len(forward_line)} 点在field外")

    # ===== 关键测试：模拟从边缘开始 =====
    print("\n" + "="*60)
    print("=== 关键测试：从田地边缘开始的Forward Line ===")
    print("="*60)

    # 找一个接近边缘但在field内的点
    # 沿着+u方向移动到接近终点
    edge_start = start + (len(forward_line) - 10) * rad_vec_norm
    edge_ix, edge_iy = int(round(edge_start[0])), int(round(edge_start[1]))

    print(f"边缘起点: ({edge_ix}, {edge_iy})")
    in_bounds_edge = 0 <= edge_ix < W and 0 <= edge_iy < H
    if in_bounds_edge:
        print(f"field_mask[{edge_iy}, {edge_ix}] = {controller._field_mask[edge_iy, edge_ix]}")

    # 从边缘点生成forward_line
    forward_line_edge = []
    for s in np.arange(0.0, max_len, 1.0):
        pt = edge_start + s * rad_vec_norm
        ix, iy = int(round(pt[0])), int(round(pt[1]))

        in_bounds = 0 <= ix < W and 0 <= iy < H
        in_field = in_bounds and controller._field_mask[iy, ix] > 0

        if in_field:
            forward_line_edge.append((float(pt[0]), float(pt[1])))
        else:
            print(f"边缘测试在 s={s:.1f} 处停止")
            break

    print(f"边缘Forward line长度: {len(forward_line_edge)} 点")

    # ===== 核心问题测试：反方向forward_line =====
    print("\n" + "="*60)
    print("=== 核心问题：反方向的Forward Line ===")
    print("="*60)

    # 当pass_dir翻转后，方向变成 -u
    rad_vec_reverse = -controller._u
    rad_vec_reverse_norm = rad_vec_reverse / np.linalg.norm(rad_vec_reverse)
    print(f"反方向向量: {rad_vec_reverse_norm}")

    # 从同一起点，反方向生成
    forward_line_rev = []
    for s in np.arange(0.0, max_len, 1.0):
        pt = start + s * rad_vec_reverse_norm
        ix, iy = int(round(pt[0])), int(round(pt[1]))

        in_bounds = 0 <= ix < W and 0 <= iy < H
        in_field = in_bounds and controller._field_mask[iy, ix] > 0

        if in_field:
            forward_line_rev.append((float(pt[0]), float(pt[1])))
        else:
            print(f"反方向在 s={s:.1f} 处停止")
            if in_bounds:
                print(f"  field_mask[{iy}, {ix}] = {controller._field_mask[iy, ix]}")
            break

    print(f"反方向Forward line长度: {len(forward_line_rev)} 点")

    # ===== 测试换行后的情况 =====
    print("\n" + "="*60)
    print("=== 测试换行后的Forward Line ===")
    print("="*60)

    # 模拟换行：移动到下一条pass (沿v方向偏移)
    strip_spacing = getattr(controller, '_strip_spacing', 10.0)
    next_pass_start = start + strip_spacing * controller._v
    next_ix, next_iy = int(round(next_pass_start[0])), int(round(next_pass_start[1]))

    print(f"换行后起点: ({next_ix}, {next_iy})")
    in_bounds_next = 0 <= next_ix < W and 0 <= next_iy < H
    if in_bounds_next:
        in_field_next = controller._field_mask[next_iy, next_ix] > 0
        print(f"field_mask[{next_iy}, {next_ix}] = {controller._field_mask[next_iy, next_ix]}")
        print(f"在field内: {in_field_next}")

    # 换行后用反方向生成
    forward_line_next = []
    for s in np.arange(0.0, max_len, 1.0):
        pt = next_pass_start + s * rad_vec_reverse_norm
        ix, iy = int(round(pt[0])), int(round(pt[1]))

        in_bounds = 0 <= ix < W and 0 <= iy < H
        in_field = in_bounds and controller._field_mask[iy, ix] > 0

        if in_field:
            forward_line_next.append((float(pt[0]), float(pt[1])))
        else:
            print(f"换行测试在 s={s:.1f} 处停止")
            break

    print(f"换行后Forward line长度: {len(forward_line_next)} 点")

    env.close()
    print("\n✅ 诊断测试完成")

if __name__ == "__main__":
    test_snake_field_mask()
