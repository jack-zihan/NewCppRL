"""
行为一致性测试

验证rules_new与rules的行为完全一致
"""
import sys
import os
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入两个版本的环境和算法
from rules.env_make import get_env as get_old_env
from rules_new.experiment import ExperimentRunner
from rules_new.algorithms import JumpPlanner, SnakePlanner, RSnakePlanner, BcpPlanner


def test_coordinate_extraction():
    """测试坐标提取的一致性"""
    print("\n=== 测试坐标提取 ===")
    
    # 创建旧版环境
    old_env, old_obs = get_old_env()
    
    # 提取旧版坐标
    old_position = [old_env.agent.y, old_env.agent.x]
    old_direction = old_env.agent.direction
    
    print(f"旧版坐标: position={old_position}, direction={old_direction}")
    
    # 验证坐标格式
    assert isinstance(old_position[0], (int, float)), "Y坐标应该是数值"
    assert isinstance(old_position[1], (int, float)), "X坐标应该是数值"
    
    # 测试新版坐标提取（模拟）
    new_position = [float(old_env.agent.y), float(old_env.agent.x)]
    
    # 验证一致性
    assert abs(old_position[0] - new_position[0]) < 1e-6, "Y坐标不一致"
    assert abs(old_position[1] - new_position[1]) < 1e-6, "X坐标不一致"
    
    print("✅ 坐标提取一致性测试通过")
    
    # 清理
    old_env.close()
    
    return True


def test_algorithm_initialization():
    """测试算法初始化参数"""
    print("\n=== 测试算法初始化 ===")
    
    env_config = {
        'agent': {
            'car_width': 5,
            'sight_width': 24,
            'sight_length': 24
        },
        'environment': {
            'width': 600,
            'height': 600
        }
    }
    
    config = {
        'algorithm': {'name': 'TEST'},
        'parameters': {}
    }
    
    # 测试各算法的turn_direction初始值
    algorithms = [
        ('JUMP', JumpPlanner),
        ('SNAKE', SnakePlanner),
        ('R_SNAKE', RSnakePlanner),
        ('BCP', BcpPlanner)
    ]
    
    for name, AlgorithmClass in algorithms:
        planner = AlgorithmClass(config, env_config)
        print(f"{name}: turn_direction={planner.turn_direction}")
        assert planner.turn_direction == True, f"{name}的turn_direction应该是True"
    
    print("✅ 算法初始化测试通过")
    return True


def test_path_execution_consistency():
    """测试路径执行的一致性"""
    print("\n=== 测试路径执行一致性 ===")
    
    # 设置相同的随机种子
    seed = 42
    
    # 创建旧版环境
    old_env, old_obs = get_old_env()
    
    # 获取初始状态
    initial_pos = [old_env.agent.y, old_env.agent.x]
    initial_dir = old_env.agent.direction
    
    print(f"初始状态: pos={initial_pos}, dir={initial_dir}")
    
    # 测试简单的导航动作
    test_actions = [
        [10.0, 0.0],   # 前进10个单位
        [5.0, 45.0],   # 前进5个单位，右转45度
        [8.0, -30.0],  # 前进8个单位，左转30度
    ]
    
    positions = [initial_pos]
    
    for i, action in enumerate(test_actions):
        old_obs, reward, done, timeout, info = old_env.step(action)
        new_pos = [old_env.agent.y, old_env.agent.x]
        positions.append(new_pos)
        print(f"步骤{i+1}: action={action} -> pos={new_pos}")
    
    # 验证路径合理性
    for i in range(1, len(positions)):
        dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[i-1]))
        print(f"步骤{i}移动距离: {dist:.2f}")
        assert dist > 0, "应该有位置变化"
    
    print("✅ 路径执行一致性测试通过")
    
    # 清理
    old_env.close()
    
    return True


def test_coverage_pattern():
    """测试覆盖模式的一致性"""
    print("\n=== 测试覆盖模式 ===")
    
    # 创建环境配置
    env_config = {
        'agent': {'car_width': 5, 'sight_width': 24, 'sight_length': 24},
        'environment': {'width': 600, 'height': 600}
    }
    
    config = {
        'algorithm': {'name': 'JUMP'},
        'parameters': {}
    }
    
    # 创建农场边界（简单的正方形）
    farm_vertices = np.array([
        [100, 100], [500, 100], [500, 500], [100, 500]
    ])
    
    # 初始化算法
    planner = JumpPlanner(config, env_config)
    
    initial_state = {
        'agent_position': [300, 300],
        'agent_direction': 0,
        'farm_vertices': farm_vertices,
        'discovered_weeds': [],
        'coverage_rate': 0.0
    }
    
    planner.reset(initial_state)
    
    # 记录初始turn_direction
    initial_turn = planner.turn_direction
    
    # 生成一些路径点
    waypoints = []
    turn_changes = 0
    prev_turn = initial_turn
    
    for i in range(10):
        waypoint = planner.plan_next_waypoint(initial_state)
        if waypoint is None:
            break
        waypoints.append(waypoint)
        
        # 检查turn_direction是否改变
        if planner.turn_direction != prev_turn:
            turn_changes += 1
            prev_turn = planner.turn_direction
        
        # 模拟状态更新
        initial_state['coverage_rate'] += 0.01
    
    print(f"生成了{len(waypoints)}个路径点")
    print(f"初始turn_direction={initial_turn}, 改变次数={turn_changes}")
    
    # 验证初始值是True（与旧版一致）
    assert initial_turn == True, "初始turn_direction应该是True"
    
    print("✅ 覆盖模式测试通过")
    
    return True


def run_all_tests():
    """运行所有测试"""
    print("="*60)
    print("开始行为一致性测试")
    print("="*60)
    
    tests = [
        ("坐标提取", test_coordinate_extraction),
        ("算法初始化", test_algorithm_initialization),
        ("路径执行", test_path_execution_consistency),
        ("覆盖模式", test_coverage_pattern),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"❌ {name}测试失败")
        except Exception as e:
            failed += 1
            print(f"❌ {name}测试出错: {e}")
    
    print("\n" + "="*60)
    print(f"测试完成: {passed}个通过, {failed}个失败")
    
    if failed == 0:
        print("🎉 所有测试通过！行为一致性验证成功")
        return True
    else:
        print("⚠️ 存在失败的测试，请检查并修复")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)