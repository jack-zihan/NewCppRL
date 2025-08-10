#!/usr/bin/env python3
"""
快速验证脚本 - 验证关键修复是否生效
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def verify_fixes():
    """验证所有关键修复"""
    print("="*60)
    print("快速验证脚本 - 检查关键修复")
    print("="*60)
    
    results = {}
    
    # 1. 验证constants.py中的INITIAL_TURN_DIRECTION
    print("\n1. 检查turn_direction初始值...")
    try:
        from rules_new.algorithms.constants import AlgorithmDefaults
        if AlgorithmDefaults.INITIAL_TURN_DIRECTION == True:
            print("   ✅ INITIAL_TURN_DIRECTION = True (正确)")
            results['turn_direction'] = True
        else:
            print("   ❌ INITIAL_TURN_DIRECTION = False (错误)")
            results['turn_direction'] = False
    except Exception as e:
        print(f"   ❌ 错误: {e}")
        results['turn_direction'] = False
    
    # 2. 验证坐标转换
    print("\n2. 检查坐标转换...")
    try:
        from rules_new.utils.coordinate_system import CoordinateSystem
        test_pos = [10, 20]
        converted = CoordinateSystem.env_to_algo(test_pos)
        if converted == (20, 10):
            print(f"   ✅ [10,20] -> (20,10) (正确)")
            results['coordinate'] = True
        else:
            print(f"   ❌ [10,20] -> {converted} (错误)")
            results['coordinate'] = False
    except Exception as e:
        print(f"   ❌ 错误: {e}")
        results['coordinate'] = False
    
    # 3. 验证算法导入
    print("\n3. 检查算法模块...")
    try:
        from rules_new.algorithms.jump_planner import JumpPlanner
        from rules_new.algorithms.snake_planner import SnakePlanner
        from rules_new.algorithms.bcp_planner import BcpPlanner
        print("   ✅ 所有算法模块可导入")
        results['imports'] = True
    except Exception as e:
        print(f"   ❌ 导入失败: {e}")
        results['imports'] = False
    
    # 4. 验证算法初始化
    print("\n4. 检查算法初始化...")
    try:
        config = {'algorithm': 'test'}
        env_config = {'agent': {'car_width': 5}}
        
        jump = JumpPlanner(config, env_config)
        if hasattr(jump, 'turn_direction') and jump.turn_direction == True:
            print("   ✅ JumpPlanner初始化正确")
            results['init'] = True
        else:
            print("   ❌ JumpPlanner初始化错误")
            results['init'] = False
    except Exception as e:
        print(f"   ❌ 初始化失败: {e}")
        results['init'] = False
    
    # 5. 总结
    print("\n" + "="*60)
    print("验证结果总结:")
    print("="*60)
    
    all_pass = all(results.values())
    pass_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    for key, value in results.items():
        status = "✅ PASS" if value else "❌ FAIL"
        print(f"  {key:20s}: {status}")
    
    print(f"\n通过率: {pass_count}/{total_count} ({pass_count/total_count*100:.0f}%)")
    
    if all_pass:
        print("\n✅ 所有关键修复已生效！")
    else:
        print("\n❌ 部分修复未生效，请检查！")
    
    return all_pass

if __name__ == "__main__":
    success = verify_fixes()
    sys.exit(0 if success else 1)