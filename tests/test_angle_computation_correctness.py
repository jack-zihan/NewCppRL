#!/usr/bin/env python3
"""
验证_compute_angle_difference函数的正确性
测试坐标系转换和无向场角度差计算
"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi, degrees, radians

def _compute_angle_difference(agent_direction: float, hif_direction: float) -> float:
    """
    原始函数实现（从cpp_env_v5.py复制）
    """
    # Agent 朝向（度，图像系，有向）归一化到 [0, 360)
    agent_direction_deg = float(agent_direction) % 360.0
    # HIF：弧度→度（数学CCW，轴向），再映射为图像系轴向（CW），范围 [0, 180)
    hif_direction_deg = float(np.degrees(hif_direction))
    hif_direction_deg_image_coordinates = (180.0 - hif_direction_deg) % 180.0
    
    # 图像系下的轴向差，折叠到 [0, 90]
    delta_deg = abs(agent_direction_deg - hif_direction_deg_image_coordinates) % 180.0
    if delta_deg > 90.0: delta_deg = 180.0 - delta_deg
    return float(delta_deg)


def analyze_coordinate_conversion():
    """分析坐标系转换的正确性"""
    print("=" * 70)
    print("坐标系转换分析")
    print("=" * 70)
    print("\n坐标系定义：")
    print("- Agent：图像坐标系（顺时针），0°=东，90°=南，180°=西，270°=北")
    print("- HIF：数学坐标系（逆时针），0=东，π/2=北，π=西，3π/2=南")
    print("- HIF是无向轴向场，范围[0, π)")
    
    print("\n" + "=" * 70)
    print("测试HIF坐标系转换:")
    print("-" * 70)
    
    test_cases = [
        (0, "东-西轴(水平)"),
        (pi/4, "东北-西南轴(45°)"),
        (pi/2, "北-南轴(垂直)"),
        (3*pi/4, "西北-东南轴(135°)"),
        (pi*0.99, "接近西-东轴")
    ]
    
    for hif_rad, description in test_cases:
        hif_deg = degrees(hif_rad)
        hif_image_deg = (180.0 - hif_deg) % 180.0
        
        print(f"\nHIF = {hif_rad:.3f} rad ({hif_deg:.1f}°) - {description}")
        print(f"  数学坐标系角度: {hif_deg:.1f}°")
        print(f"  转换后图像坐标系: {hif_image_deg:.1f}°")
        
        # 分析轴向是否正确
        if abs(hif_rad - 0) < 0.01:  # 水平轴
            expected = "0°或180°(水平)"
            correct = abs(hif_image_deg - 0) < 1 or abs(hif_image_deg - 180) < 1
        elif abs(hif_rad - pi/2) < 0.01:  # 垂直轴
            expected = "90°(垂直)"
            correct = abs(hif_image_deg - 90) < 1
        else:
            expected = f"对角线轴"
            correct = True  # 需要更复杂的验证
            
        print(f"  期望: {expected}, 正确: {'✓' if correct else '✗'}")


def test_angle_difference_computation():
    """测试角度差计算的正确性"""
    print("\n" + "=" * 70)
    print("角度差计算测试")
    print("=" * 70)
    
    # 测试案例：(agent方向度, HIF方向弧度, 期望差异, 描述)
    test_cases = [
        # 完美对齐的情况
        (0, 0, 0, "Agent东，HIF东-西轴，完美对齐"),
        (180, 0, 0, "Agent西，HIF东-西轴，完美对齐(反向也是对齐)"),
        (90, pi/2, 0, "Agent南，HIF北-南轴，完美对齐"),
        (270, pi/2, 0, "Agent北，HIF北-南轴，完美对齐(反向也是对齐)"),
        
        # 垂直的情况
        (0, pi/2, 90, "Agent东，HIF北-南轴，垂直"),
        (90, 0, 90, "Agent南，HIF东-西轴，垂直"),
        
        # 45度角的情况
        (45, 0, 45, "Agent东南，HIF东-西轴，45°偏差"),
        (45, pi/2, 45, "Agent东南，HIF北-南轴，45°偏差"),
        
        # 测试无向性
        (0, pi*0.99, 0, "Agent东，HIF接近西-东轴(无向)，应该接近对齐"),
    ]
    
    print("\n测试结果：")
    print("-" * 70)
    print(f"{'Agent方向':^10} | {'HIF方向':^12} | {'计算结果':^8} | {'期望值':^8} | {'状态':^6} | 描述")
    print("-" * 70)
    
    for agent_deg, hif_rad, expected, desc in test_cases:
        result = _compute_angle_difference(agent_deg, hif_rad)
        status = "✓" if abs(result - expected) < 5 else "✗"
        
        print(f"{agent_deg:^10.1f}° | {hif_rad:^6.3f} rad | {result:^8.1f}° | {expected:^8.1f}° | {status:^6} | {desc}")


def visualize_angle_difference():
    """可视化角度差函数的行为"""
    print("\n" + "=" * 70)
    print("创建角度差可视化热力图")
    print("=" * 70)
    
    # 创建网格
    agent_angles = np.linspace(0, 360, 361)
    hif_angles = np.linspace(0, pi, 181)
    
    # 计算所有组合的角度差
    Z = np.zeros((len(hif_angles), len(agent_angles)))
    for i, hif in enumerate(hif_angles):
        for j, agent in enumerate(agent_angles):
            Z[i, j] = _compute_angle_difference(agent, hif)
    
    # 绘制热力图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 热力图
    im1 = ax1.imshow(Z, extent=[0, 360, 180, 0], aspect='auto', cmap='RdYlGn_r')
    ax1.set_xlabel('Agent方向 (度)')
    ax1.set_ylabel('HIF方向 (度)')
    ax1.set_title('角度差热力图 (红=90°垂直, 绿=0°对齐)')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(im1, ax=ax1, label='角度差 (度)')
    
    # 添加关键线
    ax1.axvline(x=0, color='b', linestyle='--', alpha=0.5, label='Agent东')
    ax1.axvline(x=90, color='b', linestyle='--', alpha=0.5, label='Agent南')
    ax1.axvline(x=180, color='b', linestyle='--', alpha=0.5, label='Agent西')
    ax1.axvline(x=270, color='b', linestyle='--', alpha=0.5, label='Agent北')
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='HIF东-西')
    ax1.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='HIF北-南')
    
    # 横截面图
    agent_test = 45  # 测试Agent朝向45度时的情况
    hif_range = np.linspace(0, pi, 181)
    differences = [_compute_angle_difference(agent_test, h) for h in hif_range]
    
    ax2.plot(np.degrees(hif_range), differences, 'b-', linewidth=2)
    ax2.set_xlabel('HIF方向 (度)')
    ax2.set_ylabel('角度差 (度)')
    ax2.set_title(f'Agent方向={agent_test}°时的角度差曲线')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 90])
    
    # 标记关键点
    ax2.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='HIF东-西轴')
    ax2.axvline(x=90, color='r', linestyle='--', alpha=0.5, label='HIF北-南轴')
    ax2.axhline(y=0, color='g', linestyle='--', alpha=0.5, label='完美对齐')
    ax2.axhline(y=45, color='y', linestyle='--', alpha=0.5, label='45°偏差')
    ax2.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='垂直')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('/home/lzh/NewCppRL/tests/angle_difference_analysis.png', dpi=150)
    print("热力图已保存到: /home/lzh/NewCppRL/tests/angle_difference_analysis.png")
    plt.show()


def verify_key_properties():
    """验证关键数学性质"""
    print("\n" + "=" * 70)
    print("关键数学性质验证")
    print("=" * 70)
    
    print("\n1. 无向性验证（HIF是无向场）:")
    print("-" * 40)
    # HIF的0和π应该表示同一个轴向
    agent = 45
    hif1 = 0
    hif2 = pi * 0.999  # 接近π
    diff1 = _compute_angle_difference(agent, hif1)
    diff2 = _compute_angle_difference(agent, hif2)
    print(f"Agent={agent}°, HIF={hif1:.3f}rad: {diff1:.1f}°")
    print(f"Agent={agent}°, HIF={hif2:.3f}rad: {diff2:.1f}°")
    print(f"差异是否相近: {'✓' if abs(diff1 - diff2) < 5 else '✗'}")
    
    print("\n2. 对称性验证（Agent反向）:")
    print("-" * 40)
    # Agent朝向相反方向，与同一HIF的角度差应该相同
    hif = pi/4
    agent1 = 30
    agent2 = 210  # 相反方向
    diff1 = _compute_angle_difference(agent1, hif)
    diff2 = _compute_angle_difference(agent2, hif)
    print(f"Agent={agent1}°, HIF={hif:.3f}rad: {diff1:.1f}°")
    print(f"Agent={agent2}°, HIF={hif:.3f}rad: {diff2:.1f}°")
    print(f"差异是否相同: {'✓' if abs(diff1 - diff2) < 1 else '✗'}")
    
    print("\n3. 范围验证:")
    print("-" * 40)
    # 所有结果应该在[0, 90]范围内
    import random
    random.seed(42)
    all_in_range = True
    for _ in range(1000):
        agent = random.uniform(0, 360)
        hif = random.uniform(0, pi)
        diff = _compute_angle_difference(agent, hif)
        if diff < 0 or diff > 90:
            all_in_range = False
            print(f"超出范围: Agent={agent:.1f}°, HIF={hif:.3f}rad, 差异={diff:.1f}°")
            break
    
    if all_in_range:
        print("✓ 1000次随机测试，所有结果都在[0, 90]范围内")
    else:
        print("✗ 发现超出范围的情况")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("_compute_angle_difference 函数正确性分析")
    print("=" * 70)
    
    # 运行所有测试
    analyze_coordinate_conversion()
    test_angle_difference_computation()
    verify_key_properties()
    visualize_angle_difference()
    
    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)