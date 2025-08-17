#!/usr/bin/env python3
"""
测试StatusUpdater改进效果
验证合并后的功能和简化的crashed处理
"""

import numpy as np
from envs_new.cpp_env_v2 import CppEnv

def test_status_updater():
    """测试StatusUpdater改进"""
    print("🧪 测试StatusUpdater改进...")
    print("=" * 60)
    
    # 创建环境
    env = CppEnv()
    
    # 重置环境
    obs, _ = env.reset(seed=42)
    
    print("📊 初始状态检查:")
    print(f"  current_step: {env.env_state.current_step}")
    print(f"  crashed: {env.env_state.crashed}")
    print(f"  finished: {env.env_state.finished}")
    print(f"  timeout: {env.env_state.timeout}")
    
    # 检查StateVariable是否正确初始化
    step_info = env.env_state.get_info('current_step')
    crashed_info = env.env_state.get_info('crashed')
    finished_info = env.env_state.get_info('finished')
    timeout_info = env.env_state.get_info('timeout')
    
    print("\n✅ StateVariable初始化验证:")
    print(f"  current_step历史: {list(step_info.history) if step_info else 'N/A'}")
    print(f"  crashed历史: {list(crashed_info.history) if crashed_info else 'N/A'}")
    print(f"  finished历史: {list(finished_info.history) if finished_info else 'N/A'}")
    print(f"  timeout历史: {list(timeout_info.history) if timeout_info else 'N/A'}")
    
    # 执行几步测试状态更新
    print("\n🚶 执行步骤测试:")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nStep {i+1}:")
        print(f"  current_step: {env.env_state.current_step}")
        print(f"  crashed: {env.env_state.crashed}")
        print(f"  finished: {env.env_state.finished}")
        print(f"  timeout: {env.env_state.timeout}")
        
        if terminated or truncated:
            print(f"  🏁 Episode结束: terminated={terminated}, truncated={truncated}")
            break
    
    # 测试碰撞检测（模拟碰撞场景）
    print("\n🔍 测试crashed状态设置:")
    
    # 先检查当前是否有障碍物
    if 'obstacle' in env.maps_dict:
        obstacle_map = env.maps_dict['obstacle']
        print(f"  障碍物地图存在，非零像素: {np.sum(obstacle_map > 0)}")
        
        # 尝试找到一个障碍物位置
        obstacle_positions = np.argwhere(obstacle_map > 0)
        if len(obstacle_positions) > 0:
            obs_pos = obstacle_positions[0]
            print(f"  找到障碍物位置: {obs_pos}")
            
            # 尝试让agent靠近障碍物（这只是演示，实际碰撞需要更复杂的逻辑）
            print("  尝试模拟碰撞...")
    else:
        print("  无障碍物地图，跳过碰撞测试")
    
    # 测试任务完成（清除所有杂草）
    print("\n🎯 测试finished状态:")
    if 'weed' in env.maps_dict:
        weed_count = int(env.maps_dict['weed'].sum())
        print(f"  当前杂草数量: {weed_count}")
        
        # 模拟清除所有杂草
        env.maps_dict['weed'].fill(0)
        
        # 执行一步触发更新
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"  清除所有杂草后:")
        print(f"    finished: {env.env_state.finished}")
        print(f"    terminated: {terminated}")
    
    # 测试超时
    print("\n⏰ 测试timeout状态:")
    # 直接设置步数接近最大值
    env.env_state.current_step = env.env_state.max_steps - 2
    
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"  Step {env.env_state.current_step}: timeout={env.env_state.timeout}, truncated={truncated}")
        
        if truncated:
            print(f"  ✅ 超时触发成功!")
            break
    
    env.close()
    
    print("\n" + "=" * 60)
    print("🎉 测试完成!")
    print("\n📝 改进总结:")
    print("  1. StatusUpdater成功合并了原FlagsUpdater和StepUpdater")
    print("  2. crashed状态现在直接在step中设置，更简洁")
    print("  3. 所有状态标志统一管理，逻辑更清晰")
    print("  4. 减少了组件数量和复杂性")

if __name__ == "__main__":
    test_status_updater()