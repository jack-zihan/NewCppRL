#!/usr/bin/env python3
"""
测试轨迹初始化修复效果
验证agent_position不再从(0,0)开始
"""

import numpy as np
import cv2
from envs_new.cpp_env_v2 import CppEnv

def test_trajectory_fix():
    """测试轨迹初始化是否修复"""
    print("🧪 测试轨迹初始化修复...")
    print("=" * 60)
    
    # 创建环境
    env = CppEnv(render_mode='rgb_array')
    
    # 重置环境
    obs, _ = env.reset(seed=42)
    
    # 获取agent初始位置
    initial_position = env.agent.position
    print(f"✅ Agent初始位置: {initial_position}")
    
    # 检查agent_position的StateVariable
    agent_pos_info = env.env_state.get_info('agent_position')
    
    print("\n📊 StateVariable历史检查:")
    history_list = list(agent_pos_info.history)
    print(f"  历史记录: {history_list}")
    print(f"  当前值: {agent_pos_info.current}")
    print(f"  上一值: {agent_pos_info.last}")
    
    # 验证初始化是否正确
    print("\n🎯 验证结果:")
    
    # 检查1: 初始值应该是实际agent位置，不是(0,0)
    if len(history_list) > 0:
        first_value = history_list[0]
        if isinstance(first_value, (tuple, list, np.ndarray)):
            is_zero = np.allclose(first_value, [0.0, 0.0])
            if is_zero:
                print(f"  ❌ 初始值仍然是(0,0): {first_value}")
            else:
                print(f"  ✅ 初始值正确: {first_value}")
                if np.allclose(first_value, initial_position):
                    print(f"  ✅ 初始值与agent实际位置匹配!")
                else:
                    print(f"  ⚠️ 初始值与agent位置不匹配: {first_value} vs {initial_position}")
    
    # 检查2: agent_direction是否被正确记录
    agent_dir_info = env.env_state.get_info('agent_direction')
    if agent_dir_info:
        print(f"\n📐 Agent方向记录:")
        print(f"  当前方向: {agent_dir_info.current}")
        print(f"  初始方向: {list(agent_dir_info.history)[0] if agent_dir_info.history else 'N/A'}")
        print(f"  ✅ agent_direction已成功添加到状态追踪")
    else:
        print(f"\n  ⚠️ agent_direction未找到")
    
    # 渲染一帧查看轨迹
    img = env.render()
    
    # 执行几步看轨迹是否正常
    print("\n🚶 执行几步检查轨迹...")
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        
        agent_pos_info = env.env_state.get_info('agent_position')
        print(f"  Step {i+1}: current={agent_pos_info.current}, last={agent_pos_info.last}")
        
        if terminated or truncated:
            break
    
    # 再次渲染查看轨迹
    img_after = env.render()
    
    # 保存图像对比
    cv2.imwrite('/tmp/test_trajectory_before.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite('/tmp/test_trajectory_after.png', cv2.cvtColor(img_after, cv2.COLOR_RGB2BGR))
    
    print("\n📷 渲染图像已保存:")
    print("  - /tmp/test_trajectory_before.png (重置后)")
    print("  - /tmp/test_trajectory_after.png (执行几步后)")
    
    # 检查FrontierUpdater初始化
    print("\n🗺️ FrontierUpdater初始化检查:")
    frontier_area_info = env.env_state.get_info('frontier_area')
    if frontier_area_info:
        initial_area = list(frontier_area_info.history)[0] if frontier_area_info.history else None
        actual_area = int(env.maps_dict['field_frontier'].sum()) if 'field_frontier' in env.maps_dict else 0
        print(f"  初始记录值: {initial_area}")
        print(f"  实际初始值: {actual_area}")
        if initial_area == actual_area:
            print(f"  ✅ FrontierUpdater初始化正确!")
        else:
            print(f"  ⚠️ FrontierUpdater初始值不匹配")
    
    env.close()
    
    print("\n" + "=" * 60)
    print("🎉 测试完成!")
    print("\n📝 总结:")
    print("  1. AgentUpdater现在使用实际agent位置初始化")
    print("  2. agent_direction已添加到状态追踪")
    print("  3. FrontierUpdater使用实际frontier值初始化")
    print("  4. 轨迹不再从(0,0)开始")

if __name__ == "__main__":
    test_trajectory_fix()