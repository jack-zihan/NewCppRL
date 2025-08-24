"""
调试45度角的问题
"""
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs_new.cpp_env_v5 import HIFCalculator

def debug_45_degrees():
    """调试45度角的映射问题"""
    print("=" * 70)
    print("45度角映射调试")
    print("=" * 70)
    
    # 问题案例
    agent_deg = 45
    hif_rad = 3*np.pi/4  # 135度
    
    print(f"\n问题：Agent {agent_deg}° vs HIF {np.degrees(hif_rad):.1f}°")
    print("预期：它们应该表示同一条斜线（差异0°）")
    
    # 分析映射
    print("\n📐 坐标系分析：")
    print("-" * 50)
    print("Agent 45° = 4:30方向（东南）")
    print("HIF 3π/4 (135°) = 东南方向吗？")
    
    # HIF坐标系分析
    print("\nHIF坐标系（0=9点钟开始，顺时针）：")
    print("- 0 = 9点钟（西）")
    print("- π/4 = 10:30方向")
    print("- π/2 = 12点钟（北）")
    print("- 3π/4 = 1:30方向")
    print("- π = 3点钟（东）")
    
    print("\n❌ 问题发现：")
    print("HIF 3π/4 实际是1:30方向，不是4:30方向！")
    
    # 正确的对应关系
    print("\n✅ 正确的对应关系：")
    print("-" * 50)
    print("Agent 45° (4:30) 应该对应 HIF的什么角度？")
    
    # 计算正确的HIF角度
    # Agent 45° = 东南方向
    # 在HIF中，东南方向是多少？
    # HIF: 0=西, π/2=北, π=东, 3π/2=南
    # 但HIF只有[0, π]范围（无向）
    
    print("\n重新理解HIF坐标系：")
    print("HIF是从9点钟开始，顺时针到3点钟")
    print("0 → π/2 → π")
    print("9点 → 12点 → 3点")
    
    # 关键洞察
    print("\n💡 关键洞察：")
    print("Agent 45° (3点钟偏下45°) 在无向场中")
    print("应该对应 HIF π - π/4 = 3π/4")
    print("因为HIF π是3点钟，往回45°就是3π/4")
    
    # 但是为什么测试失败了？
    print("\n🔍 深入分析当前实现：")
    hif_deg = np.degrees(hif_rad)
    print(f"HIF {hif_rad:.4f} rad = {hif_deg:.1f}°")
    
    hif_in_agent = (hif_deg + 180) % 360
    print(f"HIF在Agent系统中: {hif_in_agent:.1f}°")
    
    diff = abs(agent_deg - hif_in_agent)
    print(f"初始差异: {diff:.1f}°")
    
    if diff > 180:
        diff = 360 - diff
        print(f"周期性处理后: {diff:.1f}°")
    
    if diff > 90:
        diff = 180 - diff
        print(f"无向场处理后: {diff:.1f}°")
    
    print(f"\n最终结果: {diff:.1f}°")
    
    # 问题分析
    print("\n⚠️ 问题根源：")
    print("HIF 135° + 180° = 315°")
    print("Agent 45° vs 315° = 270°差异")
    print("270° > 180，所以取 360 - 270 = 90°")
    print("90° 不需要补角处理")
    print("所以最终是90°！")
    
    print("\n📊 验证理解：")
    print("Agent 45° 是东南方向（3点钟偏下45°）")
    print("HIF 3π/4 (135°) 在HIF系统中是什么方向？")
    print("从9点钟开始顺时针135°，到达1:30方向（东北）")
    print("东南 vs 东北 = 垂直！所以90°是正确的！")
    
    print("\n✅ 结论：")
    print("测试用例的期望值错误！")
    print("Agent 45° 和 HIF 135° 实际上是垂直的，不是同一条线")

def find_correct_mappings():
    """找出正确的角度映射关系"""
    print("\n" + "=" * 70)
    print("正确的角度映射关系")
    print("=" * 70)
    
    print("\n对于Agent的每个主要方向，找出对应的HIF角度：")
    print("-" * 50)
    
    # Agent主要方向
    agent_directions = [
        (0, "3点钟(东)"),
        (45, "4:30(东南)"),
        (90, "6点钟(南)"),
        (135, "7:30(西南)"),
        (180, "9点钟(西)"),
        (225, "10:30(西北)"),
        (270, "12点钟(北)"),
        (315, "1:30(东北)"),
    ]
    
    for agent_deg, desc in agent_directions:
        # 在HIF系统中找对应方向
        # Agent到HIF的映射：考虑180°相位差
        # Agent 0° (3点钟) = HIF π (3点钟)
        # Agent 180° (9点钟) = HIF 0 (9点钟)
        
        # 简单公式：HIF_deg = 180 - agent_deg
        # 但要归一化到[0, 180]
        hif_deg_equivalent = (180 - agent_deg) % 360
        if hif_deg_equivalent > 180:
            hif_deg_equivalent = 360 - hif_deg_equivalent
        
        hif_rad_equivalent = np.radians(hif_deg_equivalent)
        
        # 验证
        diff = HIFCalculator._compute_angle_difference(agent_deg, hif_rad_equivalent)
        
        print(f"Agent {agent_deg:3}° ({desc:15}) → HIF {hif_deg_equivalent:3.0f}° ({hif_rad_equivalent:.3f}π) → 差异: {diff:.1f}°")

if __name__ == "__main__":
    debug_45_degrees()
    find_correct_mappings()