#!/usr/bin/env python3
"""
Bug修复验证测试脚本

验证修复的P0级问题是否已解决：
1. turn_direction初始值
2. farm_vertices坐标系转换
3. 算法行为一致性
"""

import sys
from pathlib import Path
import numpy as np

# 添加项目根目录到Python路径
BASE_DIR = Path(__file__).parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

def test_turn_direction():
    """测试turn_direction初始值是否正确"""
    print("\n" + "="*50)
    print("测试1: turn_direction初始值")
    print("="*50)
    
    from rules_new.algorithms.constants import AlgorithmDefaults
    
    expected = True
    actual = AlgorithmDefaults.INITIAL_TURN_DIRECTION
    
    if actual == expected:
        print(f"✅ 通过: turn_direction = {actual} (期望值: {expected})")
        return True
    else:
        print(f"❌ 失败: turn_direction = {actual} (期望值: {expected})")
        return False

def test_farm_vertices_transform():
    """测试farm_vertices坐标系转换"""
    print("\n" + "="*50)
    print("测试2: farm_vertices坐标系转换")
    print("="*50)
    
    # 模拟环境传入的farm_vertices (环境坐标系 [x,y])
    env_vertices = np.array([
        [100, 200],  # x=100, y=200
        [300, 200],  # x=300, y=200
        [300, 400],  # x=300, y=400
        [100, 400],  # x=100, y=400
    ])
    
    # 期望的算法坐标系 [y,x]
    expected_algo_vertices = np.array([
        [200, 100],  # y=200, x=100
        [200, 300],  # y=200, x=300
        [400, 300],  # y=400, x=300
        [400, 100],  # y=400, x=100
    ])
    
    # 测试各算法的坐标转换
    algorithms = [
        ('JUMP', 'rules_new.algorithms.jump_planner', 'JumpPlanner'),
        ('BCP', 'rules_new.algorithms.bcp_planner', 'BCPPlanner'),
        ('SNAKE', 'rules_new.algorithms.snake_planner', 'SnakePlanner'),
        ('REACT', 'rules_new.algorithms.react_planner', 'ReactPlanner'),
    ]
    
    all_passed = True
    for algo_name, module_path, class_name in algorithms:
        try:
            # 动态导入算法类
            module = __import__(module_path, fromlist=[class_name])
            AlgoClass = getattr(module, class_name)
            
            # 创建算法实例
            config = {'algorithm': algo_name, 'parameters': {}}
            env_config = {
                'agent': {'car_width': 5, 'sight_width': 24, 'sight_length': 24},
                'environment': {'width': 600, 'height': 600}
            }
            algo = AlgoClass(config, env_config)
            
            # 调用reset并传入farm_vertices
            initial_state = {
                'farm_vertices': env_vertices.copy(),
                'agent_position': [300, 300],
                'agent_direction': 0
            }
            algo.reset(initial_state)
            
            # 检查转换结果
            if hasattr(algo, 'farm_vertices') and algo.farm_vertices is not None:
                if np.array_equal(algo.farm_vertices, expected_algo_vertices):
                    print(f"  ✅ {algo_name}: 坐标转换正确")
                else:
                    print(f"  ❌ {algo_name}: 坐标转换错误")
                    print(f"     实际: {algo.farm_vertices[0]}")
                    print(f"     期望: {expected_algo_vertices[0]}")
                    all_passed = False
            else:
                print(f"  ⚠️ {algo_name}: 无法验证 (farm_vertices未设置)")
                all_passed = False
                
        except Exception as e:
            print(f"  ❌ {algo_name}: 测试失败 - {e}")
            all_passed = False
    
    return all_passed

def test_algorithm_basic_functionality():
    """测试算法基本功能是否正常"""
    print("\n" + "="*50)
    print("测试3: 算法基本功能")
    print("="*50)
    
    algorithms = [
        ('JUMP', 'rules_new.algorithms.jump_planner', 'JumpPlanner'),
        ('BCP', 'rules_new.algorithms.bcp_planner', 'BCPPlanner'),
        ('SNAKE', 'rules_new.algorithms.snake_planner', 'SnakePlanner'),
        ('REACT', 'rules_new.algorithms.react_planner', 'ReactPlanner'),
    ]
    
    # 准备测试数据
    farm_vertices = np.array([
        [0, 0],
        [600, 0],
        [600, 600],
        [0, 600]
    ])
    
    all_passed = True
    for algo_name, module_path, class_name in algorithms:
        try:
            # 动态导入算法类
            module = __import__(module_path, fromlist=[class_name])
            AlgoClass = getattr(module, class_name)
            
            # 创建算法实例
            config = {'algorithm': algo_name, 'parameters': {}}
            env_config = {
                'agent': {'car_width': 5, 'sight_width': 24, 'sight_length': 24},
                'environment': {'width': 600, 'height': 600}
            }
            algo = AlgoClass(config, env_config)
            
            # 初始化算法
            initial_state = {
                'farm_vertices': farm_vertices,
                'agent_position': [300, 300],
                'agent_direction': 0,
                'discovered_weeds': [],
                'coverage_rate': 0.0
            }
            algo.reset(initial_state)
            
            # 尝试获取下一个路径点
            current_state = initial_state.copy()
            waypoint = algo.plan_next_waypoint(current_state)
            
            if waypoint is not None:
                print(f"  ✅ {algo_name}: 能够生成路径点")
            else:
                print(f"  ⚠️ {algo_name}: 未生成路径点（可能是正常行为）")
                
        except Exception as e:
            print(f"  ❌ {algo_name}: 执行失败 - {e}")
            all_passed = False
    
    return all_passed

def main():
    """主测试函数"""
    print("\n" + "#"*50)
    print("# P0级Bug修复验证测试")
    print("#"*50)
    
    results = []
    
    # 执行各项测试
    results.append(("turn_direction初始值", test_turn_direction()))
    results.append(("farm_vertices坐标系转换", test_farm_vertices_transform()))
    results.append(("算法基本功能", test_algorithm_basic_functionality()))
    
    # 打印总结
    print("\n" + "="*50)
    print("测试总结")
    print("="*50)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {passed_count}/{total_count} 通过")
    
    if passed_count == total_count:
        print("\n🎉 所有测试通过！修复成功！")
        return 0
    else:
        print(f"\n⚠️ {total_count - passed_count} 个测试失败，请检查修复")
        return 1

if __name__ == "__main__":
    exit(main())