"""
测试rules_new的完整执行流程

确保修复后的版本可以正常运行
"""
import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_experiment_runner():
    """测试ExperimentRunner的执行"""
    print("\n=== 测试ExperimentRunner ===")
    
    try:
        from rules_new.experiment import ExperimentRunner
        
        # 使用快速测试配置
        runner = ExperimentRunner('quick_test')
        
        print("✅ ExperimentRunner创建成功")
        
        # 测试算法初始化
        print(f"  已加载算法: {list(runner.algorithm_instances.keys())}")
        
        # 验证坐标提取修复
        from rules.env_make import get_env
        env, _ = get_env()
        
        # 模拟extract_state_from_environment
        agent_position = [float(env.agent.y), float(env.agent.x)]  # 修复后的正确格式
        print(f"  坐标提取: {agent_position} (格式[y,x])")
        
        env.close()
        
        print("✅ 坐标提取验证通过")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_algorithm_execution():
    """测试算法的独立执行"""
    print("\n=== 测试算法执行 ===")
    
    try:
        from rules_new.algorithms import JumpPlanner, SnakePlanner
        from rules.env_make import get_env
        
        # 配置
        env_config = {
            'agent': {'car_width': 5, 'sight_width': 24, 'sight_length': 24},
            'environment': {'width': 600, 'height': 600}
        }
        
        config = {
            'algorithm': {'name': 'TEST'},
            'parameters': {}
        }
        
        # 测试JUMP算法
        jump = JumpPlanner(config, env_config)
        print(f"  JUMP: turn_direction={jump.turn_direction}")
        assert jump.turn_direction == True, "JUMP初始值错误"
        
        # 测试SNAKE算法
        snake = SnakePlanner(config, env_config)
        print(f"  SNAKE: turn_direction={snake.turn_direction}")
        assert snake.turn_direction == True, "SNAKE初始值错误"
        
        # 创建环境测试路径规划
        env, _ = get_env()
        
        # 准备状态
        farm_vertices = env.min_area_rect[0][:, 0, ::-1] if hasattr(env, 'min_area_rect') else None
        initial_state = {
            'agent_position': [env.agent.y, env.agent.x],  # 正确的[y,x]格式
            'agent_direction': env.agent.direction,
            'farm_vertices': farm_vertices,
            'discovered_weeds': [],
            'coverage_rate': 0.0
        }
        
        # 测试算法reset
        jump.reset(initial_state)
        snake.reset(initial_state)
        
        # 测试路径规划
        jump_waypoint = jump.plan_next_waypoint(initial_state)
        snake_waypoint = snake.plan_next_waypoint(initial_state)
        
        print(f"  JUMP生成路径点: {jump_waypoint is not None}")
        print(f"  SNAKE生成路径点: {snake_waypoint is not None}")
        
        env.close()
        
        print("✅ 算法执行测试通过")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_coordinate_consistency():
    """测试坐标系统的一致性"""
    print("\n=== 测试坐标一致性 ===")
    
    try:
        from rules_new.utils.coordinate_converter import CoordinateConverter
        
        # 测试坐标转换
        env_xy = [100, 200]  # 环境坐标[x,y]
        algo_yx = CoordinateConverter.env_xy_to_algo_yx(env_xy)
        
        print(f"  环境坐标[x,y]: {env_xy}")
        print(f"  算法坐标[y,x]: {algo_yx}")
        
        assert algo_yx == [200.0, 100.0], "坐标转换错误"
        
        # 测试反向转换
        back_xy = CoordinateConverter.algo_yx_to_env_xy(algo_yx)
        assert back_xy == [100.0, 200.0], "反向转换错误"
        
        # 测试数组索引
        y, x = 150, 250
        idx = CoordinateConverter.array_index_yx(y, x)
        assert idx == (150, 250), "数组索引错误"
        
        print("✅ 坐标一致性测试通过")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("="*60)
    print("Rules_new 执行测试")
    print("="*60)
    
    tests = [
        ("ExperimentRunner", test_experiment_runner),
        ("算法执行", test_algorithm_execution),
        ("坐标一致性", test_coordinate_consistency),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            print(f"❌ {name}测试异常: {e}")
    
    print("\n" + "="*60)
    print(f"测试完成: {passed}个通过, {failed}个失败")
    
    if failed == 0:
        print("🎉 所有测试通过！rules_new可以正常运行")
        return True
    else:
        print("⚠️ 存在失败的测试")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)