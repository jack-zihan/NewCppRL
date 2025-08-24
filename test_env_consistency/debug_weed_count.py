#!/usr/bin/env python3
"""
调试杂草数量设置问题
"""

import sys
sys.path.append('/home/lzh/NewCppRL')

from envs.cpp_env_v2 import CppEnv as OldCppEnvV2
from envs_new.cpp_env_v2 import CppEnv as NewCppEnvV2


def debug_weed_count():
    """调试杂草数量设置"""
    print("🔍 调试杂草数量设置")
    print("="*80)
    
    # 测试不同的杂草数量设置
    test_values = [10, 50, 100, 200]
    
    for target_count in test_values:
        print(f"\n目标杂草数: {target_count}")
        print("-"*40)
        
        # 旧环境
        old_env = OldCppEnvV2(render_mode=None)
        old_env.reset(seed=42, options={'weed_num': target_count})
        old_actual = old_env.weed_num
        print(f"  旧环境实际: {old_actual} {'✅' if old_actual == target_count else '❌'}")
        old_env.close()
        
        # 新环境
        new_env = NewCppEnvV2(render_mode=None)
        new_env.reset(seed=42, options={'weed_count': target_count})
        new_actual = new_env.env_state.weed_count
        print(f"  新环境实际: {new_actual} {'✅' if new_actual == target_count else '❌'}")
        
        # 检查地图上的实际杂草数
        if hasattr(new_env, 'maps_dict') and 'weed' in new_env.maps_dict:
            map_weed_count = new_env.maps_dict['weed'].sum()
            print(f"  新环境地图杂草数: {map_weed_count}")
        
        new_env.close()
    
    # 测试默认值
    print("\n默认值测试（不传参数）:")
    print("-"*40)
    
    old_env = OldCppEnvV2(render_mode=None)
    old_env.reset(seed=42)
    print(f"  旧环境默认: {old_env.weed_num}")
    old_env.close()
    
    new_env = NewCppEnvV2(render_mode=None)
    new_env.reset(seed=42)
    print(f"  新环境默认: {new_env.env_state.weed_count}")
    new_env.close()


def debug_seed_consistency():
    """调试种子一致性"""
    print("\n\n🌱 调试种子一致性")
    print("="*80)
    
    seeds = [42, 100, 200]
    
    for seed in seeds:
        print(f"\nSeed: {seed}")
        print("-"*40)
        
        # 使用相同seed和weed_count
        old_env = OldCppEnvV2(render_mode=None)
        old_env.reset(seed=seed, options={'weed_num': 50})
        
        new_env = NewCppEnvV2(render_mode=None)
        new_env.reset(seed=seed, options={'weed_count': 50})
        
        # 比较初始agent位置
        if hasattr(old_env, 'agent') and hasattr(new_env, 'agent'):
            old_pos = old_env.agent.position
            new_pos = new_env.agent.position
            pos_match = np.allclose(old_pos, new_pos, atol=1e-6)
            print(f"  Agent位置: {'✅ 一致' if pos_match else '❌ 不一致'}")
            if not pos_match:
                print(f"    旧: {old_pos}")
                print(f"    新: {new_pos}")
        
        # 比较地图尺寸
        old_dims = old_env.dimensions
        new_dims = new_env.config.dimensions
        print(f"  地图尺寸: 旧={old_dims}, 新={new_dims}")
        
        old_env.close()
        new_env.close()


import numpy as np

if __name__ == "__main__":
    debug_weed_count()
    debug_seed_consistency()