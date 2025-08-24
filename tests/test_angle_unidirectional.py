"""
深入分析无向场中的角度映射关系
"""
import numpy as np
import matplotlib.pyplot as plt

def analyze_unidirectional_field():
    """
    分析无向场中Agent和HIF坐标系的关系
    
    关键洞察：在无向场中，一条线有两个方向是等价的
    - 水平线：3点钟方向 = 9点钟方向（相差180°）
    - 垂直线：6点钟方向 = 12点钟方向（相差180°）
    """
    
    print("=" * 70)
    print("无向场角度系统分析")
    print("=" * 70)
    
    print("\n📐 坐标系定义：")
    print("-" * 50)
    print("Agent系统：0°=3点钟, 90°=6点钟, 180°=9点钟, 270°=12点钟")
    print("HIF系统：  0=9点钟, π/2=6点钟, π=3点钟")
    
    print("\n🔍 关键观察：")
    print("-" * 50)
    print("1. Agent: 0°到180° = 3点钟到9点钟（顺时针经过6点）")
    print("2. HIF: 0到π = 9点钟到3点钟（顺时针经过12点）")
    print("3. 两者是关于水平线对称的！")
    
    print("\n💡 核心洞察：")
    print("-" * 50)
    print("在无向场中，我们只关心'线的方向'，不关心'箭头指向'")
    print("因此：")
    print("- 0°和180°表示同一条线（水平线）")
    print("- 90°和270°表示同一条线（垂直线）")
    print("- 45°和225°表示同一条线（斜线）")
    
    # 测试映射关系
    print("\n🔄 坐标系映射关系：")
    print("-" * 50)
    
    test_angles = [
        (0, "3点钟/水平向右"),
        (45, "4:30方向"),
        (90, "6点钟/垂直向下"),
        (135, "7:30方向"),
        (180, "9点钟/水平向左"),
        (225, "10:30方向"),
        (270, "12点钟/垂直向上"),
        (315, "1:30方向"),
    ]
    
    print("\nAgent角度 → 对应的HIF角度（都表示同一条线）：")
    for agent_deg, desc in test_angles:
        # Agent到HIF的映射：考虑相位差
        # Agent 0° (3点钟) 应该对应 HIF π (3点钟)
        # Agent 180° (9点钟) 应该对应 HIF 0 (9点钟)
        # 转换公式：hif = π - agent_rad
        agent_rad = np.radians(agent_deg)
        hif_rad = np.pi - agent_rad
        
        # 处理负值（归一化到[0, 2π]）
        if hif_rad < 0:
            hif_rad += 2 * np.pi
        
        # 在无向场中归一化到[0, π]
        hif_rad_normalized = hif_rad % np.pi
        
        print(f"  Agent {agent_deg:3}° ({desc:15}) → HIF {hif_rad:.4f} rad "
              f"(归一化: {hif_rad_normalized:.4f} rad = {np.degrees(hif_rad_normalized):.1f}°)")
    
    print("\n✨ 统一表示方法：")
    print("-" * 50)
    print("如果我们定义'线角度'∈[0°, 90°]来表示所有可能的线：")
    print("- 0° = 水平线")
    print("- 45° = 斜线")
    print("- 90° = 垂直线")
    
    def to_line_angle_agent(agent_deg):
        """将Agent角度转换为线角度[0, 90]"""
        # 归一化到[0, 180]
        angle = agent_deg % 180
        # 再归一化到[0, 90]
        if angle > 90:
            angle = 180 - angle
        return angle
    
    def to_line_angle_hif(hif_rad):
        """将HIF角度转换为线角度[0, 90]"""
        # 转换为度
        hif_deg = np.degrees(hif_rad)
        # HIF的0（9点钟）对应水平线
        # HIF的π/2（6点钟）对应垂直线
        # 需要调整相位：HIF系统比标准系统领先90度
        adjusted_deg = (hif_deg + 90) % 180
        # 归一化到[0, 90]
        if adjusted_deg > 90:
            adjusted_deg = 180 - adjusted_deg
        return adjusted_deg
    
    print("\n📊 验证统一表示：")
    print("-" * 50)
    
    # 测试几个关键角度
    test_cases = [
        (0, np.pi, "Agent和HIF都指向3点钟"),
        (180, 0, "Agent和HIF都指向9点钟"),
        (90, np.pi/2, "Agent和HIF都指向6点钟"),
        (270, np.pi/2, "Agent指向12点，HIF无270°，但π/2的反向=12点"),
        (45, 3*np.pi/4, "Agent 45°, HIF 135°"),
    ]
    
    for agent_deg, hif_rad, desc in test_cases:
        agent_line = to_line_angle_agent(agent_deg)
        hif_line = to_line_angle_hif(hif_rad)
        print(f"{desc}:")
        print(f"  Agent {agent_deg}° → 线角度 {agent_line:.1f}°")
        print(f"  HIF {np.degrees(hif_rad):.1f}° → 线角度 {hif_line:.1f}°")
        print(f"  差异: {abs(agent_line - hif_line):.1f}°")
    
    print("\n🎯 最简单的角度差异计算方法：")
    print("-" * 50)
    
    def compute_angle_difference_simple(agent_deg, hif_rad):
        """
        最简单的方法：直接在各自坐标系中归一化，然后考虑相位差
        """
        # 将HIF转换为度
        hif_deg = np.degrees(hif_rad)
        
        # 关键洞察：Agent和HIF的0点相差180度
        # Agent 0° = 3点钟
        # HIF 0° = 9点钟
        # 所以需要调整180度相位差
        hif_adjusted = (hif_deg + 180) % 360
        
        # 计算角度差
        diff = abs(agent_deg - hif_adjusted)
        
        # 归一化到[0, 180]（因为360°和0°是同一个方向）
        if diff > 180:
            diff = 360 - diff
        
        # 在无向场中，180°差异意味着同一条线
        if diff > 90:
            diff = 180 - diff
        
        return diff
    
    def compute_angle_difference_transform(agent_deg, hif_rad):
        """
        另一种方法：将Agent转换到HIF坐标系
        """
        # Agent转HIF：hif_equivalent = π - agent_rad
        agent_rad = np.radians(agent_deg)
        agent_in_hif = np.pi - agent_rad
        
        # 处理负值
        while agent_in_hif < 0:
            agent_in_hif += 2 * np.pi
        
        # 在无向场中，归一化到[0, π]
        agent_in_hif = agent_in_hif % np.pi
        hif_normalized = hif_rad % np.pi
        
        # 计算差异
        diff_rad = abs(agent_in_hif - hif_normalized)
        
        # 如果差异大于π/2（90度），取补角
        if diff_rad > np.pi/2:
            diff_rad = np.pi - diff_rad
        
        return np.degrees(diff_rad)
    
    print("\n测试两种方法：")
    for agent_deg, hif_rad, desc in test_cases:
        diff1 = compute_angle_difference_simple(agent_deg, hif_rad)
        diff2 = compute_angle_difference_transform(agent_deg, hif_rad)
        print(f"{desc}:")
        print(f"  方法1（相位调整）: {diff1:.1f}°")
        print(f"  方法2（坐标变换）: {diff2:.1f}°")
    
    return compute_angle_difference_transform

if __name__ == "__main__":
    best_method = analyze_unidirectional_field()
    
    print("\n" + "=" * 70)
    print("🏆 推荐的实现方案")
    print("=" * 70)
    print("""
def _compute_angle_difference(agent_direction: float, hif_direction: float) -> float:
    '''
    计算无向场中agent朝向与HIF方向的角度差异
    
    坐标系：
    - Agent：0°=3点钟, 90°=6点钟, 180°=9点钟, 270°=12点钟
    - HIF：0=9点钟, π/2=6点钟, π=3点钟
    
    关键：两个坐标系的0点相差180度，但都是顺时针旋转
    '''
    # 方案1：相位调整法（更直观）
    hif_deg = np.degrees(hif_direction)
    hif_adjusted = (hif_deg + 180) % 360  # 调整HIF到Agent坐标系
    
    diff = abs(agent_direction - hif_adjusted)
    if diff > 180:
        diff = 360 - diff
    if diff > 90:  # 无向场：超过90度取补角
        diff = 180 - diff
    
    return diff
    
    # 方案2：坐标变换法（更数学）
    # agent_rad = np.radians(agent_direction)
    # agent_in_hif = (np.pi - agent_rad) % np.pi
    # hif_norm = hif_direction % np.pi
    # diff_rad = abs(agent_in_hif - hif_norm)
    # if diff_rad > np.pi/2:
    #     diff_rad = np.pi - diff_rad
    # return np.degrees(diff_rad)
    """)