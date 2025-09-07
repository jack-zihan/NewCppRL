#!/usr/bin/env python3
"""
HIF方向场角度表示分析
通过实验验证角度的真实含义
"""

import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, pi
import cv2

def test_angle_representation():
    """测试角度表示的实际含义"""
    
    print("=" * 60)
    print("HIF方向场角度表示分析")
    print("=" * 60)
    
    # 测试几个关键角度值
    test_angles = [0, pi/4, pi/2, 3*pi/4, pi]
    
    print("\n1. 分析non_maximum_suppression中的角度转换:")
    print("-" * 40)
    
    for angle in test_angles:
        # 模拟1.py中第118-121行的代码
        radians = angle
        theta_radians = np.pi / 2 - radians
        dx = np.cos(theta_radians)
        dy = np.sin(theta_radians)
        
        print(f"\norientations = {angle:.2f} rad ({angle*180/pi:.0f}°):")
        print(f"  theta_radians = π/2 - {angle:.2f} = {theta_radians:.2f}")
        print(f"  dx = cos({theta_radians:.2f}) = {dx:.2f}")
        print(f"  dy = sin({theta_radians:.2f}) = {dy:.2f}")
        
        # 解释方向
        if abs(dx) > abs(dy):
            if dx > 0:
                direction = "东 (→)"
            else:
                direction = "西 (←)"
        else:
            if dy > 0:
                direction = "南 (↓)"
            else:
                direction = "北 (↑)"
        
        print(f"  主方向: {direction}")
    
    print("\n" + "=" * 60)
    print("2. 分析Gabor滤波器的角度含义:")
    print("-" * 40)
    
    for angle in test_angles:
        # 模拟传给Gabor核的角度（第22行和第400行）
        gabor_angle = angle + pi/2
        
        print(f"\nangle = {angle:.2f} rad ({angle*180/pi:.0f}°):")
        print(f"  Gabor角度 = {angle:.2f} + π/2 = {gabor_angle:.2f} rad ({gabor_angle*180/pi:.0f}°)")
        
        # Gabor核的方向向量
        u_dir = cos(gabor_angle)
        v_dir = sin(gabor_angle)
        
        print(f"  Gabor方向向量: u={u_dir:.2f}, v={v_dir:.2f}")
        
        # Gabor检测的纹理方向（垂直于Gabor方向）
        texture_angle = gabor_angle - pi/2
        if texture_angle < 0:
            texture_angle += pi
        
        print(f"  检测的纹理方向: {texture_angle:.2f} rad ({texture_angle*180/pi:.0f}°)")
    
    print("\n" + "=" * 60)
    print("3. 角度表示总结:")
    print("-" * 40)
    
    print("""
根据代码分析，方向场的角度表示如下：

1. orientations值的范围：[0, π)
   - 这是一个无向的方向场，0和π表示同一个方向

2. 角度的实际含义（基于non_maximum_suppression的转换）：
   - orientations = 0   → 垂直方向 (南-北, ↕)
   - orientations = π/4 → 东北-西南方向 (↗↙)
   - orientations = π/2 → 水平方向 (东-西, ↔)
   - orientations = 3π/4 → 东南-西北方向 (↘↖)
   - orientations → π   → 接近垂直方向 (北-南, ↕)

3. 关于注释的验证：
   注释："滤波器求出来的角度是从9点钟方向顺时针到3点钟方向"
   
   这个注释是不准确的。实际情况是：
   - orientations从0到π变化
   - 表示的方向从垂直（12点-6点）顺时针旋转180度回到垂直
   - 中间经过水平方向（3点-9点）
   
   正确的描述应该是：
   "orientations从0到π表示从垂直方向（12点-6点）顺时针旋转180度的方向场"

4. 坐标系转换的原因：
   - Gabor滤波器使用标准数学坐标系（x向右，y向上，角度逆时针）
   - 图像坐标系是（x向右，y向下）
   - theta_radians = π/2 - radians 这个转换实际上是：
     将"从北开始顺时针"的角度转换为"从东开始逆时针"的标准数学角度
    """)
    
    print("\n" + "=" * 60)
    print("4. HSV可视化中的颜色映射:")
    print("-" * 40)
    
    print("""
在HSV可视化中：
- H (Hue) = orientations / π  (映射到[0, 1])
- 在OpenCV中，H值范围是[0, 180]，所以实际是 orientations * (180/π)

颜色对应关系：
- orientations = 0     → H = 0°   → 红色    → 垂直方向
- orientations = π/4   → H = 45°  → 橙黄色  → 东北-西南
- orientations = π/2   → H = 90°  → 绿色    → 水平方向  
- orientations = 3π/4  → H = 135° → 青绿色  → 东南-西北
- orientations → π     → H → 180° → 青色    → 接近垂直
    """)


def create_visual_demonstration():
    """创建可视化演示图"""
    
    print("\n" + "=" * 60)
    print("5. 创建可视化演示:")
    print("-" * 40)
    
    # 创建一个简单的测试图像
    size = 200
    angles = [0, pi/4, pi/2, 3*pi/4]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for i, angle in enumerate(angles):
        # 创建方向场
        orientation_field = np.full((size, size), angle, dtype=np.float32)
        
        # HSV可视化
        hsv = np.zeros((size, size, 3))
        hsv[:, :, 0] = angle / pi  # Hue
        hsv[:, :, 1] = 1.0  # Saturation
        hsv[:, :, 2] = 1.0  # Value
        
        # 转换为RGB
        from matplotlib.colors import hsv_to_rgb
        rgb = hsv_to_rgb(hsv)
        
        axes[i].imshow(rgb)
        axes[i].set_title(f'angle = {angle:.2f} ({angle*180/pi:.0f}°)')
        
        # 添加方向箭头
        cx, cy = size//2, size//2
        # 使用non_maximum_suppression中的转换
        theta_radians = np.pi / 2 - angle
        dx = np.cos(theta_radians) * 50
        dy = np.sin(theta_radians) * 50
        
        # 绘制双向箭头表示无向性
        axes[i].arrow(cx - dx, cy - dy, 2*dx, 2*dy, 
                     head_width=10, head_length=10, fc='black', ec='black')
        axes[i].arrow(cx + dx, cy + dy, -2*dx, -2*dy, 
                     head_width=10, head_length=10, fc='black', ec='black')
        
        axes[i].axis('off')
    
    plt.suptitle('HIF方向场角度可视化演示', fontsize=16)
    plt.tight_layout()
    plt.savefig('/home/lzh/NewCppRL/tests/hif_angle_visualization.png', dpi=150)
    print("可视化图像已保存到: /home/lzh/NewCppRL/tests/hif_angle_visualization.png")
    plt.show()


if __name__ == "__main__":
    test_angle_representation()
    create_visual_demonstration()
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)