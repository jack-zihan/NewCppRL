"""
深度数学分析：验证角度转换的边界条件和防护措施必要性
"""
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def analyze_current_implementation():
    """分析当前实现的边界条件问题"""
    print("=" * 70)
    print("当前实现的边界条件分析")
    print("=" * 70)
    
    # 模拟当前的计算逻辑
    def current_compute(agent_direction, hif_direction):
        """当前的实现逻辑"""
        agent_rad = np.radians(agent_direction)
        agent_in_hif = (np.pi - agent_rad) % (2 * np.pi)
        agent_in_hif_norm = agent_in_hif % np.pi
        hif_norm = hif_direction % np.pi
        diff_rad = abs(agent_in_hif_norm - hif_norm)
        if diff_rad > np.pi/2:
            diff_rad = np.pi - diff_rad
        return np.degrees(diff_rad)
    
    print("\n🔍 边界条件测试：")
    print("-" * 50)
    
    # 测试边界条件
    boundary_cases = [
        (0, np.pi, "Agent 0° → π in HIF"),
        (180, 0, "Agent 180° → 0 in HIF"),
        (90, np.pi/2, "Agent 90° → π/2 in HIF"),
        (270, np.pi/2, "Agent 270° → π/2 in HIF"),
    ]
    
    for agent_deg, hif_rad, desc in boundary_cases:
        agent_rad = np.radians(agent_deg)
        agent_in_hif = (np.pi - agent_rad) % (2 * np.pi)
        agent_in_hif_norm = agent_in_hif % np.pi
        
        print(f"\n{desc}:")
        print(f"  agent_rad = {agent_rad:.4f}")
        print(f"  π - agent_rad = {np.pi - agent_rad:.4f}")
        print(f"  agent_in_hif = {agent_in_hif:.4f}")
        print(f"  agent_in_hif % π = {agent_in_hif_norm:.4f}")
        
        # 检查边界问题
        if agent_in_hif == np.pi:
            print(f"  ⚠️ 边界问题：π % π = {np.pi % np.pi:.4f} (应该是π，不是0)")
        
        diff = current_compute(agent_deg, hif_rad)
        print(f"  最终差异: {diff:.2f}°")
    
    print("\n" + "=" * 70)
    print("问题分析")
    print("=" * 70)
    
    print("\n1. π % π = 0 的问题：")
    print(f"   np.pi % np.pi = {np.pi % np.pi}")
    print("   这会导致Agent 0°被错误地映射到HIF 0而不是π")
    
    print("\n2. 防护措施分析：")
    print("   - (π - agent_rad) % (2π): ✅ 必要 - 处理负值")
    print("   - agent_in_hif % π: ⚠️ 有边界问题")
    print("   - if diff > π/2: ✅ 必要 - 无向场核心逻辑")
    print("   - np.clip(0, 90): ❌ 不必要 - 掩盖逻辑错误")

def improved_implementation():
    """改进的实现，修复边界问题"""
    print("\n" + "=" * 70)
    print("改进的实现（无clip）")
    print("=" * 70)
    
    def improved_compute(agent_direction, hif_direction):
        """改进的实现"""
        # 转换到弧度
        agent_rad = np.radians(agent_direction)
        
        # 转换到HIF坐标系
        agent_in_hif = np.pi - agent_rad
        
        # 确保在[0, 2π]范围
        while agent_in_hif < 0:
            agent_in_hif += 2 * np.pi
        while agent_in_hif >= 2 * np.pi:
            agent_in_hif -= 2 * np.pi
        
        # 无向场归一化到[0, π]
        if agent_in_hif > np.pi:
            agent_in_hif = 2 * np.pi - agent_in_hif
        
        # HIF归一化（防御性）
        hif_norm = hif_direction
        if hif_norm < 0:
            hif_norm = 0
        while hif_norm > np.pi:
            hif_norm -= np.pi
        
        # 计算差异
        diff_rad = abs(agent_in_hif - hif_norm)
        
        # 无向场补角处理
        if diff_rad > np.pi/2:
            diff_rad = np.pi - diff_rad
        
        return np.degrees(diff_rad)
    
    print("\n🧪 改进实现测试：")
    print("-" * 50)
    
    # 全面测试
    test_cases = [
        # 边界条件
        (0, np.pi, 0, "Agent 0°, HIF π - 同向"),
        (180, 0, 0, "Agent 180°, HIF 0 - 同向"),
        (90, np.pi/2, 0, "Agent 90°, HIF π/2 - 同向"),
        (270, np.pi/2, 0, "Agent 270°, HIF π/2 - 同向(无向)"),
        
        # 垂直关系
        (0, np.pi/2, 90, "Agent 0°, HIF π/2 - 垂直"),
        (90, 0, 90, "Agent 90°, HIF 0 - 垂直"),
        
        # 45度角
        (45, 3*np.pi/4, 0, "Agent 45°, HIF 3π/4 - 同线"),
        (135, np.pi/4, 0, "Agent 135°, HIF π/4 - 同线"),
        
        # 极端情况
        (360, np.pi, 0, "Agent 360°(=0°), HIF π"),
        (-90, np.pi/2, 0, "Agent -90°(=270°), HIF π/2"),
    ]
    
    all_passed = True
    for agent_deg, hif_rad, expected, desc in test_cases:
        result = improved_compute(agent_deg, hif_rad)
        passed = abs(result - expected) < 1e-6
        
        status = "✅" if passed else "❌"
        print(f"{status} {desc}")
        print(f"   结果: {result:.2f}°, 期望: {expected:.2f}°")
        
        if not passed:
            all_passed = False
            print(f"   ⚠️ 差异: {abs(result - expected):.2f}°")
        
        # 检查是否需要clip
        if result < 0 or result > 90:
            print(f"   🚨 结果超出[0, 90]范围！值: {result:.2f}°")
            all_passed = False
    
    return all_passed, improved_compute

def alternative_simpler_implementation():
    """更简洁的替代实现"""
    print("\n" + "=" * 70)
    print("最简洁的实现（基于相位差）")
    print("=" * 70)
    
    def simplest_compute(agent_direction, hif_direction):
        """最简实现：直接处理180°相位差"""
        # 将HIF转换为度
        hif_deg = np.degrees(hif_direction)
        
        # 调整HIF到Agent坐标系（180°相位差）
        hif_in_agent = (hif_deg + 180) % 360
        
        # 计算角度差
        diff = abs(agent_direction - hif_in_agent)
        
        # 处理周期性
        if diff > 180:
            diff = 360 - diff
        
        # 无向场：超过90度取补角
        if diff > 90:
            diff = 180 - diff
        
        return diff
    
    print("\n📐 原理：")
    print("1. Agent 0° = 3点钟, HIF π = 3点钟")
    print("2. HIF 0 = 9点钟 = Agent 180°")
    print("3. 因此 HIF_in_Agent = (HIF_deg + 180) % 360")
    
    print("\n🧪 简洁实现测试：")
    print("-" * 50)
    
    test_cases = [
        (0, np.pi, 0, "同向测试1"),
        (180, 0, 0, "同向测试2"),
        (90, np.pi/2, 0, "同向测试3"),
        (0, np.pi/2, 90, "垂直测试1"),
        (90, 0, 90, "垂直测试2"),
    ]
    
    all_passed = True
    for agent_deg, hif_rad, expected, desc in test_cases:
        result = simplest_compute(agent_deg, hif_rad)
        passed = abs(result - expected) < 1e-6
        status = "✅" if passed else "❌"
        print(f"{status} {desc}: {result:.2f}° (期望: {expected}°)")
        if not passed:
            all_passed = False
    
    return all_passed, simplest_compute

if __name__ == "__main__":
    # 分析当前实现
    analyze_current_implementation()
    
    # 测试改进实现
    passed1, improved_func = improved_implementation()
    
    # 测试最简实现
    passed2, simple_func = alternative_simpler_implementation()
    
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    
    if passed1 and passed2:
        print("✅ 两种改进实现都通过测试！")
        print("\n推荐：使用'最简洁的实现'，因为：")
        print("1. 逻辑更清晰直观")
        print("2. 避免了π % π的边界问题")
        print("3. 不需要clip等防护措施")
    else:
        print("❌ 需要进一步调试")
    
    print("\n关键发现：")
    print("1. np.clip是不必要的，如果逻辑正确结果必然在[0, 90]")
    print("2. π % π = 0 是个边界陷阱")
    print("3. 直接处理180°相位差更简单可靠")