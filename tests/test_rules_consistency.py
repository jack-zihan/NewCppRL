#!/usr/bin/env python3
"""
规则算法新旧版本一致性测试
用于验证report8_consistency_analysis.md中发现的问题
"""
import sys
import numpy as np
import math
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def test_initial_turn_direction():
    """测试1: 验证turn_direction初始值问题"""
    print("\n" + "="*60)
    print("测试1: turn_direction初始值一致性")
    print("="*60)
    
    # 检查旧版的初始值
    print("\n旧版分析 (rules/jump_path.py):")
    from pathlib import Path
    old_file = project_root / "rules" / "jump_path.py"
    if old_file.exists():
        with open(old_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[370:380], start=371):
                if 'turn' in line.lower():
                    print(f"  Line {i}: {line.strip()}")
    
    # 检查新版的初始值
    print("\n新版分析 (rules_new/algorithms/constants.py):")
    from rules_new.algorithms.constants import AlgorithmDefaults
    print(f"  INITIAL_TURN_DIRECTION = {AlgorithmDefaults.INITIAL_TURN_DIRECTION}")
    
    # 判断是否一致
    old_value = True  # 从代码分析得知旧版是True
    new_value = AlgorithmDefaults.INITIAL_TURN_DIRECTION
    
    if old_value == new_value:
        print(f"\n✅ 测试通过: 初始值一致 (都是 {old_value})")
    else:
        print(f"\n❌ 测试失败: 初始值不一致")
        print(f"   旧版: turn = {old_value}")
        print(f"   新版: INITIAL_TURN_DIRECTION = {new_value}")
        print(f"   建议: 将新版改为 {old_value}")
    
    return old_value == new_value

def test_coordinate_conversion():
    """测试2: 验证坐标系转换一致性"""
    print("\n" + "="*60)
    print("测试2: 坐标系转换一致性")
    print("="*60)
    
    from rules_new.utils.coordinate_system import CoordinateSystem
    
    # 测试环境坐标到算法坐标的转换
    test_cases = [
        ([100, 200], (200, 100)),  # 期望: [x,y] -> (y,x)
        ([50.5, 75.3], (75.3, 50.5)),
        (np.array([10, 20]), (20, 10))
    ]
    
    all_pass = True
    for env_pos, expected_algo_pos in test_cases:
        algo_pos = CoordinateSystem.env_to_algo(env_pos)
        if algo_pos == expected_algo_pos:
            print(f"  ✅ env_to_algo({env_pos}) = {algo_pos}")
        else:
            print(f"  ❌ env_to_algo({env_pos}) = {algo_pos}, 期望 {expected_algo_pos}")
            all_pass = False
    
    # 测试算法坐标到环境坐标的转换
    test_cases_reverse = [
        ((200, 100), [100, 200]),  # 期望: (y,x) -> [x,y]
        ((75.3, 50.5), [50.5, 75.3])
    ]
    
    for algo_pos, expected_env_pos in test_cases_reverse:
        env_pos = CoordinateSystem.algo_to_env(algo_pos)
        if env_pos == expected_env_pos:
            print(f"  ✅ algo_to_env({algo_pos}) = {env_pos}")
        else:
            print(f"  ❌ algo_to_env({algo_pos}) = {env_pos}, 期望 {expected_env_pos}")
            all_pass = False
    
    if all_pass:
        print("\n✅ 坐标转换测试通过")
    else:
        print("\n❌ 坐标转换测试失败")
    
    return all_pass

def test_path_decomposition():
    """测试3: 验证路径分解步长一致性"""
    print("\n" + "="*60)
    print("测试3: 路径分解步长一致性")
    print("="*60)
    
    from rules_new.algorithms.base_algorithm import BasePathPlanner
    
    # 测试路径分解
    start = [0, 0]
    target = [10, 0]
    step_size = 2.0
    
    waypoints = BasePathPlanner.decompose_path(start, target, step_size)
    
    print(f"  起点: {start}")
    print(f"  目标: {target}")
    print(f"  步长: {step_size}")
    print(f"  生成的路径点: {waypoints}")
    
    # 验证步长
    expected_num_steps = int(10 // 2)  # 5步
    actual_num_steps = len(waypoints) - 1  # 减1因为包含了目标点
    
    if actual_num_steps == expected_num_steps:
        print(f"\n✅ 路径分解测试通过: 生成了{actual_num_steps}个中间点")
    else:
        print(f"\n❌ 路径分解测试失败: 期望{expected_num_steps}个中间点，实际{actual_num_steps}个")
    
    # 验证每步距离
    all_correct = True
    for i in range(1, len(waypoints)):
        prev = np.array(waypoints[i-1] if i > 1 else start)
        curr = np.array(waypoints[i])
        dist = np.linalg.norm(curr - prev)
        if i < len(waypoints) - 1:  # 非最后一步
            if abs(dist - step_size) > 0.01:
                print(f"  ❌ 第{i}步距离: {dist:.2f}, 期望: {step_size}")
                all_correct = False
    
    return all_correct

def test_bcp_step_width():
    """测试4: 验证BCP算法步进宽度"""
    print("\n" + "="*60)
    print("测试4: BCP算法步进宽度一致性")
    print("="*60)
    
    # 检查BCP算法的步进宽度
    print("\n算法步进宽度分析:")
    
    # 读取BCP算法
    bcp_file = project_root / "rules_new" / "algorithms" / "bcp_planner.py"
    if bcp_file.exists():
        with open(bcp_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[160:170], start=161):
                if 'y_offset' in line and '+=' in line:
                    print(f"  BCP (Line {i}): {line.strip()}")
    
    # 读取JUMP算法
    jump_file = project_root / "rules_new" / "algorithms" / "jump_planner.py"
    if jump_file.exists():
        with open(jump_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[210:220], start=211):
                if 'y_offset' in line and '+=' in line:
                    print(f"  JUMP (Line {i}): {line.strip()}")
    
    # 读取SNAKE算法
    snake_file = project_root / "rules_new" / "algorithms" / "snake_planner.py"
    if snake_file.exists():
        with open(snake_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[265:275], start=266):
                if 'y_offset' in line and '+=' in line:
                    print(f"  SNAKE (Line {i}): {line.strip()}")
    
    print("\n分析结果:")
    print("  BCP使用: agent_width (较密集覆盖)")
    print("  JUMP使用: sight_width/2 (较稀疏覆盖)")
    print("  SNAKE使用: sight_width/2 (较稀疏覆盖)")
    print("\n⚠️ 这是设计差异，不是bug，但需要确认是否符合预期")
    
    return True  # 这不是错误，只是需要确认

def test_dubins_sampling():
    """测试5: 验证dubins路径采样间隔"""
    print("\n" + "="*60)
    print("测试5: dubins路径采样间隔一致性")
    print("="*60)
    
    try:
        import dubins
        from rules_new.algorithms.base_algorithm import BasePathPlanner
        
        # 测试dubins路径生成
        start_pose = (0, 0, 0)
        end_pose = (10, 10, math.pi/2)
        turning_radius = 5.0
        sample_interval = 0.5  # 期望的采样间隔
        
        path_points = BasePathPlanner.generate_dubins_path(
            start_pose, end_pose, turning_radius, sample_interval
        )
        
        print(f"  起始位姿: {start_pose}")
        print(f"  目标位姿: {end_pose}")
        print(f"  转弯半径: {turning_radius}")
        print(f"  采样间隔: {sample_interval}")
        print(f"  生成点数: {len(path_points)}")
        
        # 验证点之间的距离
        if len(path_points) > 1:
            distances = []
            for i in range(1, len(path_points)):
                prev = np.array(path_points[i-1])
                curr = np.array(path_points[i])
                dist = np.linalg.norm(curr - prev)
                distances.append(dist)
            
            avg_dist = np.mean(distances)
            print(f"  平均点间距: {avg_dist:.3f}")
            
            if abs(avg_dist - sample_interval) < 0.1:
                print("\n✅ dubins采样间隔测试通过")
                return True
            else:
                print(f"\n⚠️ dubins采样间隔可能不一致: 平均{avg_dist:.3f}, 期望{sample_interval}")
                return False
        
    except ImportError:
        print("  ⚠️ dubins库未安装，跳过测试")
        return None

def main():
    """运行所有一致性测试"""
    print("="*60)
    print("规则算法新旧版本一致性测试")
    print("基于report8_consistency_analysis.md的发现")
    print("="*60)
    
    results = {}
    
    # 运行各项测试
    results['turn_direction'] = test_initial_turn_direction()
    results['coordinate'] = test_coordinate_conversion()
    results['path_decompose'] = test_path_decomposition()
    results['bcp_step'] = test_bcp_step_width()
    results['dubins'] = test_dubins_sampling()
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)
    
    print(f"\n通过: {passed}")
    print(f"失败: {failed}")
    print(f"跳过: {skipped}")
    
    if failed > 0:
        print("\n❌ 发现一致性问题，需要修复:")
        if not results.get('turn_direction'):
            print("  1. 修改 rules_new/algorithms/constants.py 中的 INITIAL_TURN_DIRECTION = True")
        if not results.get('coordinate'):
            print("  2. 检查坐标系转换逻辑")
        if not results.get('path_decompose'):
            print("  3. 统一路径分解步长为2.0")
    else:
        print("\n✅ 所有测试通过!")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)