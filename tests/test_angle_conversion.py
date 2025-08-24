"""
测试角度转换逻辑
"""
import numpy as np

def test_angle_conversion():
    """
    分析两个坐标系的转换关系
    
    Agent新坐标系（用户更新）：
    - 0° = 3点钟方向（向右）
    - 90° = 6点钟方向（向下）
    - 180° = 9点钟方向（向左）
    - 270° = 12点钟方向（向上）
    
    HIF坐标系（保持不变）：
    - 0 rad = 9点钟方向（向左）
    - π/2 rad = 6点钟方向（向下）
    - π rad = 3点钟方向（向右）
    
    注意：HIF是无向的，所以只有[0, π]范围
    """
    
    print("=" * 60)
    print("坐标系转换分析")
    print("=" * 60)
    
    # 定义关键方向的映射关系
    test_cases = [
        # (agent_deg, clock_position, hif_expected_rad)
        (0, "3点钟(向右)", np.pi),        # Agent 0° = 向右 → HIF π
        (90, "6点钟(向下)", np.pi/2),      # Agent 90° = 向下 → HIF π/2  
        (180, "9点钟(向左)", 0),           # Agent 180° = 向左 → HIF 0
        (270, "12点钟(向上)", np.pi/2),    # Agent 270° = 向上 → HIF π/2 (因为无向，向上=向下)
    ]
    
    print("\n关键方向映射：")
    print("-" * 40)
    for agent_deg, clock_pos, hif_rad in test_cases:
        print(f"Agent {agent_deg:3}° ({clock_pos:12}) → HIF {hif_rad:.4f} rad")
    
    # 推导转换公式
    print("\n转换公式推导：")
    print("-" * 40)
    print("观察规律：")
    print("- Agent 0° → HIF π")
    print("- Agent 180° → HIF 0")
    print("这是一个反向映射，可以用公式：hif_rad = π - (agent_deg * π/180)")
    print("简化后：hif_rad = π * (1 - agent_deg/180)")
    print("但要考虑无向性，需要映射到[0, π]范围")
    
    # 测试新公式
    def new_conversion(agent_deg):
        """新的转换公式"""
        # 先转换为弧度
        agent_rad = np.radians(agent_deg)
        
        # Agent坐标系转HIF坐标系
        # Agent 0° (3点钟) → HIF π (3点钟)
        # Agent 180° (9点钟) → HIF 0 (9点钟)
        # 公式：hif_rad = π - agent_rad (对于[0, π]范围)
        
        # 处理无向性：将[0, 2π]映射到[0, π]
        if agent_rad > np.pi:
            # [π, 2π] → [π, 0]
            hif_rad = 2 * np.pi - agent_rad
        else:
            # [0, π] → [π, 0]
            hif_rad = np.pi - agent_rad
            
        return hif_rad
    
    # 更简洁的转换公式
    def simple_conversion(agent_deg):
        """简化的转换公式"""
        # 转换为弧度
        agent_rad = np.radians(agent_deg)
        
        # 直接映射：考虑到HIF是无向的
        # Agent [0°, 180°] → HIF [π, 0]
        # Agent [180°, 360°] → HIF [0, π]
        
        # 使用模运算处理无向性
        agent_deg_mod = agent_deg % 360
        if agent_deg_mod > 180:
            agent_deg_mod = 360 - agent_deg_mod
            
        # 线性映射 [0, 180] → [π, 0]
        hif_rad = np.pi * (1 - agent_deg_mod / 180)
        
        return hif_rad
    
    print("\n测试转换公式：")
    print("-" * 40)
    test_angles = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    
    for angle in test_angles:
        hif_new = new_conversion(angle)
        hif_simple = simple_conversion(angle)
        print(f"Agent {angle:3}° → HIF {hif_new:.4f} rad (new), {hif_simple:.4f} rad (simple)")
    
    # 测试角度差异计算
    print("\n角度差异计算示例：")
    print("-" * 40)
    
    def compute_angle_difference(agent_deg, hif_rad):
        """计算角度差异"""
        # 将agent角度转换到HIF坐标系
        agent_hif_rad = simple_conversion(agent_deg)
        
        # 计算弧度差
        diff_rad = abs(agent_hif_rad - hif_rad)
        
        # 如果差异大于90度，取补角（因为是无向的）
        if diff_rad > np.pi/2:
            diff_rad = np.pi - diff_rad
            
        # 转换为度
        diff_deg = np.degrees(diff_rad)
        return diff_deg
    
    # 测试案例
    test_diff_cases = [
        (0, np.pi, "Agent向右，HIF也向右"),      # 应该是0度差异
        (0, 0, "Agent向右，HIF向左"),            # 应该是180度，但无向所以是0度
        (90, np.pi/2, "Agent向下，HIF也向下"),    # 应该是0度差异
        (180, 0, "Agent向左，HIF也向左"),         # 应该是0度差异
        (45, np.pi*3/4, "Agent45°，HIF 135°"),   # 应该是0度差异
    ]
    
    for agent_deg, hif_rad, desc in test_diff_cases:
        diff = compute_angle_difference(agent_deg, hif_rad)
        print(f"{desc}")
        print(f"  Agent {agent_deg}° vs HIF {np.degrees(hif_rad):.1f}° → 差异 {diff:.1f}°")
    
    return simple_conversion

if __name__ == "__main__":
    conversion_func = test_angle_conversion()
    
    print("\n" + "=" * 60)
    print("推荐的转换公式：")
    print("=" * 60)
    print("""
    def _compute_angle_difference(agent_direction: float, hif_direction: float) -> float:
        # Agent新坐标系：0°=3点钟, 90°=6点钟, 180°=9点钟, 270°=12点钟
        # HIF坐标系：0=9点钟, π/2=6点钟, π=3点钟
        
        # 处理无向性：将agent角度映射到[0, 180]
        agent_deg_mod = agent_direction % 360
        if agent_deg_mod > 180:
            agent_deg_mod = 360 - agent_deg_mod
        
        # 转换：Agent [0°, 180°] → HIF [π, 0]
        agent_hif_rad = np.pi * (1 - agent_deg_mod / 180)
        
        # 计算角度差异
        diff_rad = abs(agent_hif_rad - hif_direction)
        
        # 无向场处理：差异大于90度时取补角
        if diff_rad > np.pi/2:
            diff_rad = np.pi - diff_rad
        
        return np.degrees(diff_rad)
    """)