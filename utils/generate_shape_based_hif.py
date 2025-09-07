#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于形状的HIF方向场生成器
从农田掩码形状生成方向场，产生类似等高线的流动效果
"""

import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt, gaussian_filter
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse


def compute_orientation_field(mask, smooth_sigma=0):
    """
    基于形状计算方向场 - 使用距离变换和梯度正交化
    
    Args:
        mask: 二值掩码 (H, W)，1为内部，0为外部
        smooth_sigma: 可选的高斯平滑参数（默认0，不平滑）
    
    Returns:
        orientation: 方向场 (H, W)，范围[0, π)，无效值-1
    """
    # 1. 计算距离场 - 每个内部像素到边界的最短距离
    distance_field = distance_transform_edt(mask)
    
    # 2. 可选的平滑处理
    if smooth_sigma > 0:
        distance_field = gaussian_filter(distance_field, sigma=smooth_sigma)
    
    # 3. 计算梯度 - 指向边界的法线方向
    grad_y, grad_x = np.gradient(distance_field)
    
    # 4. 正交化得到切线方向 - 沿边界流动的方向
    # 旋转90度：切线 = (-grad_y, grad_x)
    tangent_x = -grad_y
    tangent_y = grad_x
    
    # 5. 计算方向角度
    orientation = np.arctan2(tangent_y, tangent_x)
    
    # 6. 归一化到[0, π)范围（无向场）
    orientation = np.mod(orientation, np.pi)
    
    # 7. 设置无效区域
    orientation[mask == 0] = -1
    
    # 8. 边界处理 - 梯度为0的地方（如中心点）设为无效
    magnitude = np.sqrt(tangent_x**2 + tangent_y**2)
    orientation[magnitude < 1e-6] = -1
    
    return orientation.astype(np.float32)


def smooth_orientation_field(orientation, mask, iterations=2, sigma=1.5):
    """
    使用circular mean平滑方向场，保持方向连续性
    
    Args:
        orientation: 方向场，范围[0, π)
        mask: 有效区域掩码
        iterations: 平滑迭代次数
        sigma: 高斯滤波的标准差（默认1.5）
    
    Returns:
        平滑后的方向场
    """
    valid = (orientation >= 0) & (mask > 0)
    
    for _ in range(iterations):
        # 转换为单位向量（处理180度周期性）
        cos_2theta = np.cos(2 * orientation)
        sin_2theta = np.sin(2 * orientation)
        
        # 应用高斯滤波
        cos_smooth = gaussian_filter(cos_2theta * valid, sigma=sigma)
        sin_smooth = gaussian_filter(sin_2theta * valid, sigma=sigma)
        valid_smooth = gaussian_filter(valid.astype(float), sigma=sigma)
        
        # 避免除零
        valid_smooth[valid_smooth < 0.1] = 0.1
        
        # 归一化
        cos_smooth /= valid_smooth
        sin_smooth /= valid_smooth
        
        # 重建角度
        orientation_smooth = np.arctan2(sin_smooth, cos_smooth) / 2.0
        orientation_smooth = np.mod(orientation_smooth, np.pi)
        
        # 只更新有效区域
        orientation[valid] = orientation_smooth[valid]
    
    return orientation


def visualize_orientation_hsv(orientation, mask):
    """
    HSV增强可视化 - 与HIFS系统标准一致
    
    Args:
        orientation: 方向场，范围[0, π)
        mask: 有效区域掩码
    
    Returns:
        RGB图像 (H, W, 3)
    """
    h, w = orientation.shape
    valid_mask = (orientation >= 0) & (mask > 0)
    
    # 创建HSV图像
    hsv_image = np.zeros((h, w, 3))
    hsv_image[:, :, 0] = np.where(valid_mask, orientation / np.pi, 0)  # Hue: [0,1]
    hsv_image[:, :, 1] = np.where(valid_mask, 1.0, 0)  # Saturation: 最大饱和度
    hsv_image[:, :, 2] = np.where(valid_mask, 1.0, 0)  # Value: 最大亮度
    
    # 转换为RGB
    rgb_image = hsv_to_rgb(hsv_image)
    
    # 背景设为白色
    rgb_image[~valid_mask] = [1, 1, 1]
    
    return (rgb_image * 255).astype(np.uint8)


def visualize_orientation_vector(orientation, mask, density=20, scale=1.0):
    """
    矢量场可视化 - 箭头表示方向
    
    Args:
        orientation: 方向场，范围[0, π)
        mask: 有效区域掩码
        density: 箭头密度
        scale: 箭头缩放
    
    Returns:
        matplotlib figure
    """
    h, w = orientation.shape
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # 显示背景（掩码）
    ax.imshow(mask, cmap='gray', alpha=0.3, origin='upper')
    
    # 创建矢量场网格
    step = max(h, w) // density
    if step < 1:
        step = 1
    
    Y, X = np.mgrid[step//2:h:step, step//2:w:step]
    
    # 收集有效点
    valid_points = []
    U = []
    V = []
    C = []  # 颜色
    
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            y, x = int(Y[i, j]), int(X[i, j])
            if y < h and x < w and orientation[y, x] >= 0:
                angle = orientation[y, x]
                u = np.cos(angle)
                v = np.sin(angle)
                valid_points.append([x, y])
                U.append(u)
                V.append(v)
                # HSV颜色编码
                C.append(hsv_to_rgb([angle/np.pi, 1, 1]))
    
    if valid_points:
        valid_points = np.array(valid_points)
        U = np.array(U)
        V = np.array(V)
        
        # 绘制矢量场
        arrow_scale = step * 2 * scale
        ax.quiver(valid_points[:, 0], valid_points[:, 1], 
                 U, V, color=C, 
                 scale=1/arrow_scale, scale_units='xy', 
                 width=0.003, headwidth=3, headlength=1,
                 alpha=0.8)
    
    # 添加颜色条说明
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection
    
    # 创建颜色条
    n_colors = 180
    patches = []
    colors = []
    for i in range(n_colors):
        angle = i * np.pi / n_colors
        rect = Rectangle((w + 10, h - i * h / n_colors), 20, h / n_colors)
        patches.append(rect)
        colors.append(hsv_to_rgb([angle/np.pi, 1, 1]))
    
    collection = PatchCollection(patches, facecolor=colors, edgecolor='none')
    ax.add_collection(collection)
    
    # 标注
    ax.text(w + 35, h * 0.0, '0°', fontsize=10, va='center')
    ax.text(w + 35, h * 0.5, '90°', fontsize=10, va='center')
    ax.text(w + 35, h * 1.0, '180°', fontsize=10, va='center')
    
    ax.set_xlim(-5, w + 60)
    ax.set_ylim(h + 5, -5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('方向场矢量可视化', fontsize=14, pad=20)
    
    return fig


def process_single_field(field_path, output_base_dir, smooth_iterations=0, smooth_sigma=1.5):
    """
    处理单个field文件
    
    Args:
        field_path: field掩码文件路径
        output_base_dir: 输出基础目录
        smooth_iterations: 平滑迭代次数（默认0，不平滑）
        smooth_sigma: 平滑强度（默认1.5）
    
    Returns:
        成功返回True，失败返回False
    """
    try:
        # 提取文件编号
        field_name = field_path.stem  # field_1, field_2, etc.
        field_number = field_name.replace('field_', '')
        
        # 加载掩码
        mask_img = cv2.imread(str(field_path), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            print(f"❌ 无法读取文件: {field_path}")
            return False
        
        # 转换为二值掩码
        mask = (mask_img > 0).astype(np.uint8)
        
        # 计算方向场
        orientation = compute_orientation_field(mask, smooth_sigma=0)
        
        # 可选的额外平滑
        if smooth_iterations > 0:
            orientation = smooth_orientation_field(orientation, mask, smooth_iterations, smooth_sigma)
        
        # 创建输出目录
        hif_dir = output_base_dir / 'hif'
        hif_image_dir = hif_dir / 'image'
        hif_dir.mkdir(parents=True, exist_ok=True)
        hif_image_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存.npy文件
        npy_path = hif_dir / f'human_intent_field_{field_number}.npy'
        np.save(npy_path, orientation)
        
        # 生成HSV可视化
        hsv_vis = visualize_orientation_hsv(orientation, mask)
        hsv_path = hif_image_dir / f'orientation_field_{field_number}.png'
        cv2.imwrite(str(hsv_path), cv2.cvtColor(hsv_vis, cv2.COLOR_RGB2BGR))
        
        # 生成矢量场可视化
        fig = visualize_orientation_vector(orientation, mask, density=25, scale=1.0)
        vector_path = hif_image_dir / f'orientation_field_{field_number}_vector.png'
        fig.savefig(vector_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        
        return True
        
    except Exception as e:
        print(f"❌ 处理 {field_path} 时出错: {e}")
        return False


def main():
    """主函数 - 批处理所有field文件"""
    parser = argparse.ArgumentParser(description='基于形状生成HIF方向场')
    parser.add_argument('map_dir', type=str, help='地图目录路径')
    parser.add_argument('--smooth', type=int, default=0, help='平滑迭代次数 (默认: 0，不平滑)')
    parser.add_argument('--sigma', type=float, default=1.5, help='平滑强度 (默认: 1.5)')
    parser.add_argument('--test', action='store_true', help='测试模式，只处理前3个文件')
    args = parser.parse_args()
    
    # 设置路径
    map_dir = Path(args.map_dir)
    field_dir = map_dir / 'field'
    
    if not field_dir.exists():
        print(f"❌ Field目录不存在: {field_dir}")
        return
    
    # 获取所有field文件
    field_files = sorted(field_dir.glob('field_*.png'))
    if not field_files:
        print(f"❌ 未找到field文件在: {field_dir}")
        return
    
    print(f"📂 找到 {len(field_files)} 个field文件")
    
    # 测试模式
    if args.test:
        field_files = field_files[:3]
        print(f"🧪 测试模式: 只处理前3个文件")
    
    # 处理统计
    success_count = 0
    failed_files = []
    
    print(f"\n🔄 开始处理...")
    print(f"   平滑迭代: {args.smooth}")
    print(f"   输出目录: {map_dir}/hif/\n")
    
    # 批处理
    for field_path in tqdm(field_files, desc="生成方向场"):
        if process_single_field(field_path, map_dir, args.smooth, args.sigma):
            success_count += 1
        else:
            failed_files.append(field_path.name)
    
    # 打印结果
    print(f"\n✅ 处理完成!")
    print(f"   成功: {success_count}/{len(field_files)}")
    
    if failed_files:
        print(f"   失败文件:")
        for name in failed_files:
            print(f"     - {name}")
    
    print(f"\n📁 输出文件:")
    print(f"   方向场数据: {map_dir}/hif/human_intent_field_*.npy")
    print(f"   HSV可视化: {map_dir}/hif/image/orientation_field_*.png")
    print(f"   矢量可视化: {map_dir}/hif/image/orientation_field_*_vector.png")


if __name__ == '__main__':
    main()