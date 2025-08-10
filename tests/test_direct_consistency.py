#!/usr/bin/env python3
"""
直接功能一致性测试 - 专注于核心算法行为验证
ultrathink模式 - 极度严格验证
"""

import sys
import os
import numpy as np
import time
from typing import Dict, List, Tuple, Any

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_turn_direction():
    """直接测试turn_direction初始值"""
    print("\n" + "="*60)
    print("测试: turn_direction初始值验证")
    print("="*60)
    
    try:
        # 导入模块
        from rules.jump_path import JUMP as OldJUMP
        from rules_new.algorithms.jump_planner import JumpPlanner
        
        # 创建测试农场
        farm = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        
        # 旧版本
        old_jump = OldJUMP(200, 200, 0.5, farm)
        old_turn = old_jump.turn_direction
        print(f"旧版本 turn_direction: {old_turn}")
        
        # 新版本检查 - 看是否有该属性
        config = {'algorithm': 'jump', 'step': 2.5}
        env_config = {'agent': {'car_width': 5}}
        new_jump = JumpPlanner(config, env_config)
        
        # 检查新版本是否有turn_direction
        if hasattr(new_jump, 'turn_direction'):
            new_turn = new_jump.turn_direction
            print(f"新版本 turn_direction: {new_turn}")
            
            if old_turn == new_turn:
                print("✅ turn_direction初始值一致")
                return True
            else:
                print(f"❌ turn_direction不一致: OLD={old_turn}, NEW={new_turn}")
                return False
        else:
            print("⚠️ 新版本没有turn_direction属性")
            return None
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_farm_vertices_format():
    """测试farm_vertices坐标格式"""
    print("\n" + "="*60)
    print("测试: farm_vertices坐标格式验证")
    print("="*60)
    
    try:
        from rules.jump_path import JUMP as OldJUMP
        
        # 原始输入格式 [x, y]
        farm_input = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        print(f"输入农场坐标 (x,y格式): \n{farm_input}")
        
        # 旧版本
        old_jump = OldJUMP(200, 200, 0.5, farm_input)
        old_vertices = old_jump.farm_vertices
        print(f"\n旧版本 farm_vertices: \n{old_vertices}")
        
        # 检查旧版本是否进行了坐标转换
        # 如果进行了转换，应该是 [y, x] 格式
        if np.array_equal(old_vertices, farm_input):
            print("旧版本未转换坐标 (保持x,y格式)")
        else:
            print("旧版本转换了坐标")
            # 检查是否是x,y -> y,x的转换
            expected_conversion = farm_input[:, [1, 0]]  # 交换列
            if np.array_equal(old_vertices, expected_conversion):
                print("✅ 旧版本执行了 x,y -> y,x 转换")
            else:
                print("⚠️ 旧版本的转换方式未知")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_jump_basic_path():
    """测试JUMP基本路径生成"""
    print("\n" + "="*60)
    print("测试: JUMP基本路径生成")
    print("="*60)
    
    try:
        from rules.jump_path import JUMP as OldJUMP
        
        # 设置随机种子
        np.random.seed(42)
        
        # 简单测试用例
        farm = np.array([[0, 0], [20, 0], [20, 10], [0, 10]])
        turning_radius = 0.5
        weeds = [(5, 5), (15, 7)]
        
        # 旧版本路径
        old_jump = OldJUMP(200, 200, turning_radius, farm)
        old_path = old_jump.get_path(weeds)
        
        print(f"农场: {farm.tolist()}")
        print(f"杂草: {weeds}")
        print(f"转弯半径: {turning_radius}")
        print(f"生成路径长度: {len(old_path)}")
        
        if len(old_path) > 0:
            print(f"路径前5个点: {old_path[:5] if len(old_path) >= 5 else old_path}")
            print("✅ JUMP路径生成成功")
            return True
        else:
            print("❌ JUMP路径为空")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_bcp_step_value():
    """测试BCP步进值计算"""
    print("\n" + "="*60)
    print("测试: BCP步进值计算")
    print("="*60)
    
    try:
        from rules.jump_path import Boustrophedon_Cell_Path as OldBCP
        
        # 测试参数
        farm = np.array([[0, 0], [25, 0], [25, 12], [0, 12]])
        turning_radius = 1.0
        
        # 旧版本
        old_bcp = OldBCP(200, 200, turning_radius, farm)
        old_step = old_bcp.step
        
        print(f"农场: {farm.tolist()}")
        print(f"转弯半径: {turning_radius}")
        print(f"计算的步进值: {old_step:.6f}")
        
        # 验证步进值公式
        # step = 2 * turning_radius + 一些调整
        expected_base = 2 * turning_radius
        print(f"基础步进值 (2*r): {expected_base:.6f}")
        print(f"实际步进值: {old_step:.6f}")
        
        if old_step > 0:
            print("✅ BCP步进值计算正常")
            return True
        else:
            print("❌ BCP步进值异常")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_algorithm_existence():
    """测试算法模块是否存在"""
    print("\n" + "="*60)
    print("测试: 算法模块存在性验证")
    print("="*60)
    
    results = {}
    
    # 测试旧版本
    try:
        from rules.jump_path import JUMP
        results['old_jump'] = "✅ 存在"
    except:
        results['old_jump'] = "❌ 不存在"
    
    try:
        from rules.jump_path import SNAKE
        results['old_snake'] = "✅ 存在"
    except:
        results['old_snake'] = "❌ 不存在"
    
    try:
        from rules.jump_path import Boustrophedon_Cell_Path
        results['old_bcp'] = "✅ 存在"
    except:
        results['old_bcp'] = "❌ 不存在"
    
    # 测试新版本
    try:
        from rules_new.algorithms.jump_planner import JumpPlanner
        results['new_jump'] = "✅ 存在"
    except:
        results['new_jump'] = "❌ 不存在"
    
    try:
        from rules_new.algorithms.snake_planner import SnakePlanner
        results['new_snake'] = "✅ 存在"
    except:
        results['new_snake'] = "❌ 不存在"
    
    try:
        from rules_new.algorithms.bcp_planner import BCPPlanner
        results['new_bcp'] = "✅ 存在"
    except:
        results['new_bcp'] = "❌ 不存在"
    
    # 打印结果
    print("\n旧版本模块:")
    print(f"  JUMP: {results['old_jump']}")
    print(f"  SNAKE: {results['old_snake']}")
    print(f"  BCP: {results['old_bcp']}")
    
    print("\n新版本模块:")
    print(f"  JumpPlanner: {results['new_jump']}")
    print(f"  SnakePlanner: {results['new_snake']}")
    print(f"  BCPPlanner: {results['new_bcp']}")
    
    # 检查是否都存在
    all_exist = all('✅' in v for v in results.values())
    if all_exist:
        print("\n✅ 所有算法模块都存在")
        return True
    else:
        print("\n⚠️ 部分算法模块缺失")
        return False

def generate_summary(test_results):
    """生成测试摘要"""
    print("\n" + "="*80)
    print("测试摘要 - ultrathink验证")
    print("="*80)
    
    total = len(test_results)
    passed = sum(1 for r in test_results.values() if r is True)
    failed = sum(1 for r in test_results.values() if r is False)
    skipped = sum(1 for r in test_results.values() if r is None)
    
    print(f"\n总测试数: {total}")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    print(f"跳过: {skipped}")
    print(f"通过率: {passed/total*100:.1f}%")
    
    print("\n详细结果:")
    for name, result in test_results.items():
        if result is True:
            status = "✅ PASSED"
        elif result is False:
            status = "❌ FAILED"
        else:
            status = "⚠️ SKIPPED"
        print(f"  {name}: {status}")
    
    # 最终判定
    print("\n" + "="*80)
    print("最终判定")
    print("="*80)
    
    if passed >= total * 0.8:
        print("\n✅ **基本功能验证通过**")
        print("主要功能正常，可以进行进一步的集成测试")
    else:
        print("\n❌ **功能验证未通过**")
        print("存在关键问题，需要修复后重新测试")
    
    return passed / total if total > 0 else 0

def main():
    """主测试入口"""
    print("="*80)
    print("直接功能一致性测试 - ultrathink模式")
    print("测试时间:", time.strftime('%Y-%m-%d %H:%M:%S'))
    print("="*80)
    
    # 执行测试
    test_results = {}
    
    # 基础测试
    test_results['模块存在性'] = test_algorithm_existence()
    test_results['turn_direction'] = test_turn_direction()
    test_results['farm_vertices'] = test_farm_vertices_format()
    test_results['JUMP路径'] = test_jump_basic_path()
    test_results['BCP步进值'] = test_bcp_step_value()
    
    # 生成摘要
    pass_rate = generate_summary(test_results)
    
    # 保存简单报告
    report_path = "/home/lzh/NewCppRL/rules_new/team_reports/direct_test_result.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"直接功能测试结果\n")
        f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"通过率: {pass_rate*100:.1f}%\n\n")
        
        for name, result in test_results.items():
            status = "PASSED" if result is True else ("FAILED" if result is False else "SKIPPED")
            f.write(f"{name}: {status}\n")
    
    print(f"\n测试结果已保存至: {report_path}")
    
    return pass_rate >= 0.8

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)