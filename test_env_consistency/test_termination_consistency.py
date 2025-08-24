#!/usr/bin/env python3
"""
终止条件一致性测试
检查碰撞和完成判定是否一致
"""

import sys
import numpy as np
sys.path.append('/home/lzh/NewCppRL')

from envs.cpp_env_v2 import CppEnv as OldCppEnvV2
from envs_new.cpp_env_v2 import CppEnv as NewCppEnvV2


def test_single_step_detail(seed=42):
    """详细测试单步奖励差异"""
    print("🔍 单步奖励详细分析")
    print("="*80)
    
    # 创建环境
    old_env = OldCppEnvV2(render_mode=None)
    new_env = NewCppEnvV2(render_mode=None)
    
    # 使用相同的种子重置
    old_obs, old_info = old_env.reset(seed=seed)
    new_obs, new_info = new_env.reset(seed=seed)
    
    # 设置相同的动作种子
    old_env.action_space.seed(seed)
    new_env.action_space.seed(seed)
    
    print("\n运行到碰撞/完成前...")
    
    # 运行到第23步（问题发生的地方）
    for step in range(23):
        action = old_env.action_space.sample()
        
        # 执行动作
        old_obs, old_reward, old_done, old_truncated, old_info = old_env.step(action)
        new_obs, new_reward, new_done, new_truncated, new_info = new_env.step(action)
        
        # 在第22步停下来，详细检查第23步
        if step == 21:
            print(f"\n第{step+1}步后的状态:")
            print(f"  旧环境:")
            print(f"    - 奖励: {old_reward:.4f}")
            print(f"    - done: {old_done}, truncated: {old_truncated}")
            print(f"    - crashed: {old_info.get('crashed', False)}")
            print(f"    - finished: {old_info.get('finished', False)}")
            print(f"    - weed_ratio: {old_info.get('weed_ratio', 0):.4f}")
            
            print(f"  新环境:")
            print(f"    - 奖励: {new_reward:.4f}")
            print(f"    - done: {new_done}, truncated: {new_truncated}")
            print(f"    - crashed: {new_info.get('crashed', False)}")
            print(f"    - finished: {new_info.get('finished', False)}")
            print(f"    - weed_ratio: {new_info.get('weed_ratio', 0):.4f}")
            
            # 获取下一个动作
            action = old_env.action_space.sample()
            print(f"\n执行第23步，动作: {action}")
            
            # 执行关键的第23步
            old_obs, old_reward, old_done, old_truncated, old_info = old_env.step(action)
            new_obs, new_reward, new_done, new_truncated, new_info = new_env.step(action)
            
            print(f"\n第23步的结果:")
            print(f"  旧环境:")
            print(f"    - 奖励: {old_reward:.4f}")
            print(f"    - done: {old_done}, truncated: {old_truncated}")
            print(f"    - crashed: {old_info.get('crashed', False)}")
            print(f"    - finished: {old_info.get('finished', False)}")
            
            print(f"  新环境:")
            print(f"    - 奖励: {new_reward:.4f}")
            print(f"    - done: {new_done}, truncated: {new_truncated}")
            print(f"    - crashed: {new_info.get('crashed', False)}")
            print(f"    - finished: {new_info.get('finished', False)}")
            
            # 分析奖励差异
            print(f"\n奖励差异分析:")
            diff = new_reward - old_reward
            print(f"  差异: {diff:.4f}")
            
            if abs(diff - 500) < 1:
                print("  💡 差异约500：可能是碰撞(-399) vs 完成(+100)的判定差异")
                print("     预期：碰撞-399 + 基础-0.1 = -399.1")
                print("     预期：完成+500 + 基础-0.1 + 其他 ≈ 100.9")
            
            break
    
    old_env.close()
    new_env.close()


def test_weed_completion_logic():
    """测试杂草完成逻辑"""
    print("\n\n🌱 测试杂草完成逻辑")
    print("="*80)
    
    # 创建环境
    old_env = OldCppEnvV2(render_mode=None)
    new_env = NewCppEnvV2(render_mode=None)
    
    # 重置环境，设置较少的杂草以便快速完成
    old_obs, old_info = old_env.reset(seed=42, options={'weed_num': 5})
    new_obs, new_info = new_env.reset(seed=42, options={'weed_num': 5})
    
    print(f"初始杂草数量:")
    print(f"  旧环境: {old_env.weed_num}")
    print(f"  新环境: {new_env.env_state.weed_count if hasattr(new_env, 'env_state') else 'N/A'}")
    
    # 检查完成阈值
    print(f"\n完成阈值:")
    if hasattr(old_env, 'weed_ratio_threshold'):
        print(f"  旧环境: {old_env.weed_ratio_threshold}")
    if hasattr(new_env, 'config') and hasattr(new_env.config, 'weed_ratio_threshold'):
        print(f"  新环境: {new_env.config.weed_ratio_threshold}")
    
    old_env.close()
    new_env.close()


def test_collision_detection():
    """测试碰撞检测逻辑"""
    print("\n\n💥 测试碰撞检测逻辑")
    print("="*80)
    
    # 这需要更深入的分析，可能需要查看agent位置和障碍物位置
    print("需要进一步分析碰撞检测的实现差异...")
    print("可能的原因：")
    print("1. 碰撞检测的精度差异（int vs round）")
    print("2. 边界条件判定差异")
    print("3. Agent尺寸或碰撞箱差异")
    print("4. 障碍物位置或大小差异")


if __name__ == "__main__":
    # 详细分析单步差异
    test_single_step_detail(seed=42)
    
    # 测试杂草完成逻辑
    test_weed_completion_logic()
    
    # 测试碰撞检测
    test_collision_detection()