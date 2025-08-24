"""
全面验证_compute_angle_difference函数的逻辑正确性
"""
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs_new.cpp_env_v5 import HIFCalculator

def analyze_coordinate_systems():
    """深入分析两个坐标系的映射关系"""
    print("=" * 80)
    print("坐标系深度分析")
    print("=" * 80)
    
    print("\n📐 坐标系定义：")
    print("-" * 60)
    print("Agent系统（度）：0°=3点钟, 90°=6点钟, 180°=9点钟, 270°=12点钟")
    print("HIF系统（弧度）：0=9点钟, π/2=6点钟, π=3点钟")
    
    print("\n🔍 转换逻辑分析：")
    print("-" * 60)
    print("当前实现：hif_in_agent = (hif_deg + 180) % 360")
    
    # 验证关键映射点
    mappings = [
        (0, "9点钟"),
        (np.pi/4, "10:30"),
        (np.pi/2, "6点钟"),
        (3*np.pi/4, "1:30"),
        (np.pi, "3点钟"),
    ]
    
    print("\nHIF → Agent坐标系映射：")
    for hif_rad, clock_desc in mappings:
        hif_deg = np.degrees(hif_rad)
        hif_in_agent = (hif_deg + 180) % 360
        
        # 分析Agent系统中的位置
        agent_clock = {
            0: "3点钟", 45: "4:30", 90: "6点钟", 135: "7:30",
            180: "9点钟", 225: "10:30", 270: "12点钟", 315: "1:30"
        }
        
        closest_clock = min(agent_clock.keys(), key=lambda x: abs(x - hif_in_agent))
        
        print(f"  HIF {hif_rad:.3f}rad ({hif_deg:6.1f}°) [{clock_desc:6}] "
              f"→ Agent {hif_in_agent:6.1f}° [{agent_clock.get(closest_clock):6}]")
    
    print("\n⚠️ 注意：HIF π/2 (6点钟) 被映射到 Agent 270° (12点钟)")
    print("这在无向场中是正确的，因为6点钟和12点钟是同一条垂直线")

def test_critical_cases():
    """测试关键案例"""
    print("\n" + "=" * 80)
    print("关键案例测试")
    print("=" * 80)
    
    test_cases = [
        # (agent_deg, hif_rad, expected_diff, description)
        # 主方向测试
        (0, np.pi, 0, "Agent 3点钟, HIF 3点钟 - 同向"),
        (90, np.pi/2, 0, "Agent 6点钟, HIF 6点钟 - 同向(通过无向场处理)"),
        (180, 0, 0, "Agent 9点钟, HIF 9点钟 - 同向"),
        (270, np.pi/2, 0, "Agent 12点钟, HIF 6点钟 - 无向场同线"),
        
        # 垂直关系测试
        (0, np.pi/2, 90, "Agent 3点钟, HIF 6点钟 - 垂直"),
        (90, 0, 90, "Agent 6点钟, HIF 9点钟 - 垂直"),
        (180, np.pi/2, 90, "Agent 9点钟, HIF 6点钟 - 垂直"),
        (270, 0, 90, "Agent 12点钟, HIF 9点钟 - 垂直"),
        
        # 45度角测试（修正期望值）
        (45, 3*np.pi/4, 90, "Agent 4:30(东南), HIF 1:30(东北) - 垂直"),
        (135, np.pi/4, 90, "Agent 7:30(西南), HIF 10:30(西北) - 垂直"),
        (225, np.pi/4, 0, "Agent 10:30(西北), HIF 10:30(西北) - 同向"),
        (315, 3*np.pi/4, 0, "Agent 1:30(东北), HIF 1:30(东北) - 同向"),
        
        # 边界条件测试
        (0, 0, 0, "Agent 3点钟, HIF 9点钟 - 无向场水平线"),
        (360, np.pi, 0, "Agent 360°(=0°), HIF π"),
        (-90, np.pi/2, 0, "Agent -90°(=270°), HIF π/2"),
        (450, np.pi/2, 0, "Agent 450°(=90°), HIF π/2"),
    ]
    
    print("\n🧪 测试结果：")
    print("-" * 60)
    
    all_passed = True
    for agent_deg, hif_rad, expected_diff, desc in test_cases:
        result = HIFCalculator._compute_angle_difference(agent_deg, hif_rad)
        passed = abs(result - expected_diff) < 1e-6
        
        status = "✅" if passed else "❌"
        print(f"{status} {desc}")
        print(f"   输入: Agent {agent_deg:4.0f}°, HIF {np.degrees(hif_rad):6.1f}°")
        print(f"   结果: {result:5.1f}°, 期望: {expected_diff:5.1f}°")
        
        if not passed:
            all_passed = False
            print(f"   ⚠️ 差异: {abs(result - expected_diff):.3f}°")
    
    return all_passed

def test_symmetry_properties():
    """测试对称性和数学性质"""
    print("\n" + "=" * 80)
    print("对称性和数学性质测试")
    print("=" * 80)
    
    print("\n1️⃣ 测试交换律（无向场应该满足）：")
    print("-" * 60)
    
    # 在无向场中，两个角度的差异应该是对称的
    test_pairs = [
        (0, np.pi),
        (90, 0),
        (45, 3*np.pi/4),
    ]
    
    symmetric = True
    for agent_deg, hif_rad in test_pairs:
        # 正向计算
        diff1 = HIFCalculator._compute_angle_difference(agent_deg, hif_rad)
        
        # 反向：将HIF角度作为Agent角度，Agent角度转为HIF
        # Agent deg → HIF: hif_equiv = π - agent_rad
        agent_rad = np.radians(agent_deg)
        hif_equiv = np.pi - agent_rad
        while hif_equiv < 0:
            hif_equiv += np.pi
        while hif_equiv > np.pi:
            hif_equiv -= np.pi
            
        diff2 = HIFCalculator._compute_angle_difference(np.degrees(hif_rad), hif_equiv)
        
        match = abs(diff1 - diff2) < 1e-6
        status = "✅" if match else "❌"
        print(f"{status} Agent {agent_deg:3.0f}° vs HIF {np.degrees(hif_rad):5.1f}°: "
              f"diff={diff1:5.1f}°, reverse diff={diff2:5.1f}°")
        
        if not match:
            symmetric = False
    
    print("\n2️⃣ 测试范围约束：")
    print("-" * 60)
    
    # 测试大量随机值，确保结果始终在[0, 90]范围内
    np.random.seed(42)
    out_of_range = []
    
    for _ in range(1000):
        agent_deg = np.random.uniform(-720, 720)
        hif_rad = np.random.uniform(0, np.pi)
        
        result = HIFCalculator._compute_angle_difference(agent_deg, hif_rad)
        
        if result < 0 or result > 90:
            out_of_range.append((agent_deg, hif_rad, result))
    
    if out_of_range:
        print(f"❌ 发现 {len(out_of_range)} 个超出[0, 90]范围的结果：")
        for i, (ag, hf, res) in enumerate(out_of_range[:5]):  # 只显示前5个
            print(f"   Agent {ag:.1f}°, HIF {np.degrees(hf):.1f}° → {res:.1f}°")
    else:
        print("✅ 1000个随机测试全部在[0, 90]范围内")
    
    print("\n3️⃣ 测试周期性：")
    print("-" * 60)
    
    base_agent = 45
    hif_test = np.pi/3
    base_diff = HIFCalculator._compute_angle_difference(base_agent, hif_test)
    
    periodic_tests = [360, -360, 720, -720]
    periodic_ok = True
    
    for offset in periodic_tests:
        diff = HIFCalculator._compute_angle_difference(base_agent + offset, hif_test)
        match = abs(diff - base_diff) < 1e-6
        status = "✅" if match else "❌"
        print(f"{status} Agent {base_agent}° + {offset}° → diff={diff:.1f}° "
              f"(base={base_diff:.1f}°)")
        if not match:
            periodic_ok = False
    
    return symmetric and len(out_of_range) == 0 and periodic_ok

def test_implementation_consistency():
    """测试不同实现方式的一致性"""
    print("\n" + "=" * 80)
    print("实现一致性测试")
    print("=" * 80)
    
    def alternative_compute(agent_direction, hif_direction):
        """替代实现：使用三角函数"""
        # 将两个角度都转换为单位向量
        agent_rad = np.radians(agent_direction)
        agent_vec = np.array([np.cos(agent_rad), np.sin(agent_rad)])
        
        # HIF坐标系转换：考虑相位差
        hif_adjusted = hif_direction + np.pi  # 相位调整
        hif_vec = np.array([np.cos(hif_adjusted), np.sin(hif_adjusted)])
        
        # 计算向量点积得到cos(θ)
        cos_angle = np.dot(agent_vec, hif_vec)
        cos_angle = np.clip(cos_angle, -1, 1)  # 数值稳定性
        
        # 转换为角度
        angle_diff = np.degrees(np.arccos(abs(cos_angle)))  # abs处理无向场
        
        return angle_diff
    
    print("\n比较当前实现与向量法实现：")
    print("-" * 60)
    
    test_points = [
        (0, np.pi), (90, np.pi/2), (180, 0), (270, np.pi/2),
        (45, 3*np.pi/4), (135, np.pi/4), (225, np.pi/4), (315, 3*np.pi/4),
    ]
    
    max_diff = 0
    for agent_deg, hif_rad in test_points:
        current = HIFCalculator._compute_angle_difference(agent_deg, hif_rad)
        alternative = alternative_compute(agent_deg, hif_rad)
        diff = abs(current - alternative)
        
        status = "✅" if diff < 1e-6 else "⚠️"
        print(f"{status} Agent {agent_deg:3.0f}°, HIF {np.degrees(hif_rad):5.1f}°: "
              f"当前={current:5.1f}°, 向量法={alternative:5.1f}°, 差异={diff:.3f}°")
        
        max_diff = max(max_diff, diff)
    
    print(f"\n最大差异: {max_diff:.6f}°")
    
    return max_diff < 0.01  # 允许小的数值误差

def main():
    """主测试函数"""
    print("\n" + "🔬 " * 20)
    print("_compute_angle_difference 全面验证")
    print("🔬 " * 20)
    
    # 1. 分析坐标系
    analyze_coordinate_systems()
    
    # 2. 测试关键案例
    test1_passed = test_critical_cases()
    
    # 3. 测试数学性质
    test2_passed = test_symmetry_properties()
    
    # 4. 测试实现一致性
    test3_passed = test_implementation_consistency()
    
    # 总结
    print("\n" + "=" * 80)
    print("📊 测试总结")
    print("=" * 80)
    
    if test1_passed and test2_passed and test3_passed:
        print("\n✅ 所有测试通过！")
        print("\n结论：_compute_angle_difference的逻辑是正确的")
        print("\n关键发现：")
        print("1. 转换公式 (hif_deg + 180) % 360 正确处理了180°相位差")
        print("2. 无向场的两步处理（周期性+补角）确保结果在[0, 90]范围")
        print("3. 虽然HIF π/2映射到Agent 270°看似奇怪，但在无向场中是正确的")
    else:
        print("\n❌ 部分测试失败")
        print(f"- 关键案例测试: {'✅' if test1_passed else '❌'}")
        print(f"- 数学性质测试: {'✅' if test2_passed else '❌'}")
        print(f"- 实现一致性测试: {'✅' if test3_passed else '❌'}")

if __name__ == "__main__":
    main()