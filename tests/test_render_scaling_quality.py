#!/usr/bin/env python3
"""
测试不同缩放方法对渲染质量的影响
比较 repeat vs cv2.resize 的效果
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # 使用无头后端
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/lzh/NewCppRL')

def create_test_trajectory_image(size=(100, 100)):
    """创建一个包含轨迹线的测试图像"""
    img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    
    # 绘制一些测试元素
    # 1. 斜线轨迹（最容易看出锯齿）
    cv2.line(img, (10, 10), (90, 50), (255, 0, 255), 1)  # 紫色斜线
    
    # 2. 曲线轨迹
    points = []
    for i in range(20, 80, 2):
        y = int(30 + 20 * np.sin(i * 0.2))
        points.append([i, y])
    points = np.array(points, np.int32)
    cv2.polylines(img, [points], False, (255, 0, 0), 1)  # 红色曲线
    
    # 3. 垂直和水平线
    cv2.line(img, (30, 20), (30, 80), (0, 255, 0), 1)  # 绿色垂直线
    cv2.line(img, (40, 60), (70, 60), (0, 0, 255), 1)  # 蓝色水平线
    
    # 4. 小圆点（代表agent位置）
    cv2.circle(img, (50, 70), 3, (255, 0, 0), -1)
    
    return img

def scale_with_repeat(img, scale=2):
    """使用 repeat 方法缩放（当前的方法）"""
    return img.repeat(scale, axis=0).repeat(scale, axis=1)

def scale_with_resize_nearest(img, scale=2):
    """使用 cv2.resize 最近邻插值"""
    new_size = (img.shape[1] * scale, img.shape[0] * scale)
    return cv2.resize(img, new_size, interpolation=cv2.INTER_NEAREST)

def scale_with_resize_linear(img, scale=2):
    """使用 cv2.resize 线性插值"""
    new_size = (img.shape[1] * scale, img.shape[0] * scale)
    return cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

def scale_with_resize_cubic(img, scale=2):
    """使用 cv2.resize 三次插值"""
    new_size = (img.shape[1] * scale, img.shape[0] * scale)
    return cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)

def render_at_high_resolution(size=(200, 200), target_size=(100, 100)):
    """直接在高分辨率下渲染，然后缩小"""
    img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    
    # 在2倍分辨率下绘制（线条粗细也要相应调整）
    scale = size[0] // target_size[0]
    
    # 斜线轨迹
    cv2.line(img, (20, 20), (180, 100), (255, 0, 255), max(1, scale//2))
    
    # 曲线轨迹
    points = []
    for i in range(40, 160, 4):
        y = int(60 + 40 * np.sin(i * 0.1))
        points.append([i, y])
    points = np.array(points, np.int32)
    cv2.polylines(img, [points], False, (255, 0, 0), max(1, scale//2))
    
    # 垂直和水平线
    cv2.line(img, (60, 40), (60, 160), (0, 255, 0), max(1, scale//2))
    cv2.line(img, (80, 120), (140, 120), (0, 0, 255), max(1, scale//2))
    
    # 圆点
    cv2.circle(img, (100, 140), 6, (255, 0, 0), -1)
    
    # 如果需要缩小，使用高质量缩放
    if size != target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    return img

def analyze_edge_quality(img):
    """分析边缘质量（锯齿程度）"""
    # 使用Sobel算子检测边缘
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = np.sqrt(sobelx**2 + sobely**2)
    
    # 计算边缘的平滑度（方差越大说明越不平滑）
    edge_variance = np.var(edge_strength[edge_strength > 10])
    
    return edge_variance

def main():
    print("🔬 测试不同缩放方法对渲染质量的影响")
    print("=" * 60)
    
    # 创建测试图像
    original = create_test_trajectory_image()
    
    # 应用不同的缩放方法
    scale = 2
    scaled_repeat = scale_with_repeat(original, scale)
    scaled_nearest = scale_with_resize_nearest(original, scale)
    scaled_linear = scale_with_resize_linear(original, scale)
    scaled_cubic = scale_with_resize_cubic(original, scale)
    high_res = render_at_high_resolution()
    
    # 分析边缘质量
    print("\n📊 边缘质量分析（方差越小越平滑）：")
    print(f"  Repeat方法:      {analyze_edge_quality(scaled_repeat):.2f}")
    print(f"  最近邻插值:      {analyze_edge_quality(scaled_nearest):.2f}")
    print(f"  线性插值:        {analyze_edge_quality(scaled_linear):.2f}")
    print(f"  三次插值:        {analyze_edge_quality(scaled_cubic):.2f}")
    print(f"  高分辨率渲染:    {analyze_edge_quality(high_res):.2f}")
    
    # 保存对比图
    save_dir = '/home/lzh/NewCppRL/test_env_consistency/img'
    
    # 创建对比图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    images = [
        (original, "原始 (100x100)"),
        (scaled_repeat, "Repeat方法 (当前)"),
        (scaled_nearest, "最近邻插值"),
        (scaled_linear, "线性插值"),
        (scaled_cubic, "三次插值"),
        (high_res, "高分辨率渲染")
    ]
    
    for ax, (img, title) in zip(axes.flat, images):
        # 转换BGR到RGB用于matplotlib显示
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(title)
        ax.axis('off')
    
    plt.suptitle('不同缩放方法的渲染质量对比', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/scaling_comparison.png', dpi=150)
    print(f"\n💾 对比图保存到: {save_dir}/scaling_comparison.png")
    
    # 保存局部放大对比
    print("\n🔍 生成局部放大对比...")
    
    # 提取斜线区域进行放大对比
    roi = (10, 10, 50, 30)  # x, y, width, height
    
    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
    
    # Repeat方法的局部
    roi_repeat = scaled_repeat[roi[1]*scale:(roi[1]+roi[3])*scale, 
                               roi[0]*scale:(roi[0]+roi[2])*scale]
    axes2[0].imshow(cv2.cvtColor(roi_repeat, cv2.COLOR_BGR2RGB))
    axes2[0].set_title('Repeat方法（块状/锯齿明显）')
    axes2[0].axis('off')
    
    # 线性插值的局部
    roi_linear = scaled_linear[roi[1]*scale:(roi[1]+roi[3])*scale,
                               roi[0]*scale:(roi[0]+roi[2])*scale]
    axes2[1].imshow(cv2.cvtColor(roi_linear, cv2.COLOR_BGR2RGB))
    axes2[1].set_title('线性插值（较平滑）')
    axes2[1].axis('off')
    
    # 高分辨率渲染的局部
    roi_highres = high_res[roi[1]:(roi[1]+roi[3]), roi[0]:(roi[0]+roi[2])]
    roi_highres_scaled = cv2.resize(roi_highres, (roi[2]*scale, roi[3]*scale), 
                                    interpolation=cv2.INTER_NEAREST)
    axes2[2].imshow(cv2.cvtColor(roi_highres_scaled, cv2.COLOR_BGR2RGB))
    axes2[2].set_title('高分辨率渲染（最佳质量）')
    axes2[2].axis('off')
    
    plt.suptitle('斜线轨迹局部放大对比', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/scaling_zoom_comparison.png', dpi=150)
    print(f"💾 局部放大对比图保存到: {save_dir}/scaling_zoom_comparison.png")
    
    # 结论
    print("\n" + "=" * 60)
    print("📋 分析结论：")
    print()
    print("1. ❌ **Repeat方法的问题**：")
    print("   - 产生明显的块状效果（像素直接复制）")
    print("   - 斜线出现严重锯齿（阶梯状）")
    print("   - 轨迹线条变粗且不均匀")
    print("   - 曲线失去平滑性")
    print()
    print("2. ✅ **更好的替代方案**：")
    print("   - **短期改进**：使用 cv2.resize() 与 INTER_LINEAR 插值")
    print("   - **最佳方案**：直接在目标分辨率下渲染（render_repeat_times倍）")
    print("   - **折中方案**：使用 INTER_CUBIC 插值获得更平滑的效果")
    print()
    print("3. 🎯 **建议的修改**：")
    print("   ```python")
    print("   # 替换当前的repeat方法")
    print("   if self.config.render_repeat_times > 1:")
    print("       scale = self.config.render_repeat_times")
    print("       new_size = (rendered.shape[1] * scale, rendered.shape[0] * scale)")
    print("       rendered = cv2.resize(rendered, new_size, interpolation=cv2.INTER_LINEAR)")
    print("   ```")

if __name__ == "__main__":
    main()