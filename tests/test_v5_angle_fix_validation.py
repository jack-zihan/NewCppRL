"""
验证v5环境中角度转换函数修复的正确性
"""
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs_new.cpp_env_v5 import HIFCalculator

def test_angle_conversion():
    """
    验证修复后的角度转换函数
    """
    print("=" * 70)
    print("v5环境角度转换函数验证")
    print("=" * 70)
    
    print("\n📐 坐标系定义：")
    print("-" * 50)
    print("Agent系统：0°=3点钟, 90°=6点钟, 180°=9点钟, 270°=12点钟")
    print("HIF系统：  0=9点钟, π/2=6点钟, π=3点钟")
    
    # 定义测试案例
    test_cases = [
        # (agent_deg, hif_rad, expected_diff, description)
        (0, np.pi, 0, "Agent和HIF都指向3点钟"),
        (180, 0, 0, "Agent和HIF都指向9点钟"),
        (90, np.pi/2, 0, "Agent和HIF都指向6点钟"),
        (270, np.pi/2, 0, "Agent指向12点，无向场中等价于6点钟"),
        (45, 3*np.pi/4, 0, "Agent 45°(3:30方向), HIF 135°(4:30方向) - 同一条斜线"),
        (135, np.pi/4, 0, "Agent 135°(7:30方向), HIF 45°(7:30方向) - 同一条斜线"),
        (0, 0, 0, "Agent 3点钟, HIF 9点钟 - 无向场中是同一条水平线"),
        (90, 0, 90, "Agent 6点钟, HIF 9点钟 - 垂直关系"),
        (0, np.pi/2, 90, "Agent 3点钟, HIF 6点钟 - 垂直关系"),
    ]
    
    print("\n🧪 测试结果：")
    print("-" * 50)
    
    all_passed = True
    for agent_deg, hif_rad, expected_diff, desc in test_cases:
        # 调用修复后的函数
        actual_diff = HIFCalculator._compute_angle_difference(agent_deg, hif_rad)
        
        # 检查结果是否符合预期（允许小误差）
        passed = abs(actual_diff - expected_diff) < 1e-6
        status = "✅" if passed else "❌"
        
        print(f"{status} {desc}:")
        print(f"   Agent {agent_deg:3}° vs HIF {np.degrees(hif_rad):6.1f}°")
        print(f"   期望差异: {expected_diff:5.1f}°, 实际差异: {actual_diff:5.1f}°")
        
        if not passed:
            all_passed = False
            print(f"   ⚠️ 测试失败！差异: {abs(actual_diff - expected_diff):.3f}°")
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 所有测试通过！角度转换函数修复成功！")
    else:
        print("❌ 部分测试失败，需要进一步调试")
    print("=" * 70)
    
    # 额外测试：完整的角度范围
    print("\n📊 完整角度范围测试（每45度）：")
    print("-" * 50)
    
    for agent_deg in range(0, 360, 45):
        # 对应的HIF角度（考虑相位差）
        agent_rad = np.radians(agent_deg)
        hif_equivalent = (np.pi - agent_rad) % np.pi
        
        diff = HIFCalculator._compute_angle_difference(agent_deg, hif_equivalent)
        
        clock_pos = {
            0: "3点钟", 45: "4:30", 90: "6点钟", 135: "7:30",
            180: "9点钟", 225: "10:30", 270: "12点钟", 315: "1:30"
        }
        
        print(f"Agent {agent_deg:3}° ({clock_pos.get(agent_deg, ''):6}) → "
              f"HIF {np.degrees(hif_equivalent):5.1f}° → 差异: {diff:5.1f}°")
    
    print("\n✨ 关键验证点：")
    print("-" * 50)
    print("1. 相同方向的角度差异应该为0°")
    print("2. 垂直方向的角度差异应该为90°")
    print("3. 无向场中，相差180°的方向应该被视为同一条线（差异0°）")
    
    return all_passed

if __name__ == "__main__":
    test_angle_conversion()