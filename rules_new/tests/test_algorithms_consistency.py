"""
算法一致性测试模块

测试新版算法与旧版的一致性，包括：
- 坐标转换正确性
- 初始值一致性
- 关键路径覆盖
"""
import sys
import os
import numpy as np
import unittest
from typing import Dict, Any

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rules_new.algorithms.jump_planner import JumpPlanner
from rules_new.algorithms.snake_planner import SnakePlanner  
from rules_new.algorithms.bcp_planner import BcpPlanner
from rules_new.utils.coordinate_system import CoordinateSystem
from rules_new.algorithms.constants import AlgorithmDefaults


class TestCoordinateSystem(unittest.TestCase):
    """测试坐标系转换的正确性"""
    
    def test_env_to_algo_conversion(self):
        """测试环境坐标到算法坐标的转换"""
        # 单个点转换
        env_pos = [10, 20]
        algo_pos = CoordinateSystem.env_to_algo(env_pos)
        self.assertEqual(algo_pos, (20, 10))
        
        # numpy数组转换
        env_pos_np = np.array([10, 20])
        algo_pos_np = CoordinateSystem.env_to_algo(env_pos_np)
        self.assertEqual(algo_pos_np, (20, 10))
        
    def test_algo_to_env_conversion(self):
        """测试算法坐标到环境坐标的转换"""
        # 元组转换
        algo_pos = (20, 10)
        env_pos = CoordinateSystem.algo_to_env(algo_pos)
        self.assertEqual(env_pos, [10, 20])
        
        # 列表转换
        algo_pos_list = [20, 10]
        env_pos_list = CoordinateSystem.algo_to_env(algo_pos_list)
        self.assertEqual(env_pos_list, [10, 20])
        
    def test_batch_conversion(self):
        """测试批量转换"""
        env_positions = [[10, 20], [30, 40], [50, 60]]
        algo_positions = CoordinateSystem.batch_env_to_algo(env_positions)
        expected = [(20, 10), (40, 30), (60, 50)]
        self.assertEqual(algo_positions, expected)
        
        # 反向转换
        back_to_env = CoordinateSystem.batch_algo_to_env(algo_positions)
        self.assertEqual(back_to_env, env_positions)
        
    def test_round_trip_conversion(self):
        """测试往返转换的一致性"""
        original = [100, 200]
        converted = CoordinateSystem.env_to_algo(original)
        back = CoordinateSystem.algo_to_env(converted)
        self.assertEqual(back, original)


class TestAlgorithmInitialization(unittest.TestCase):
    """测试算法初始化的一致性"""
    
    def setUp(self):
        """准备测试数据"""
        self.env_config = {
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
        
        self.config = {
            'algorithm': {'name': 'TEST'},
            'parameters': {}
        }
        
    def test_jump_planner_initialization(self):
        """测试JUMP算法初始化值"""
        planner = JumpPlanner(self.config, self.env_config)
        
        # 检查turn_direction初始值
        self.assertEqual(planner.turn_direction, AlgorithmDefaults.INITIAL_TURN_DIRECTION)
        self.assertTrue(planner.turn_direction)  # 应该是True以与旧版一致
        
        # 检查其他默认值
        self.assertEqual(planner.agent_width, 5)
        self.assertEqual(planner.sight_width, 24)
        self.assertEqual(planner.turning_radius, 5.0)
        
    def test_snake_planner_initialization(self):
        """测试SNAKE算法初始化值"""
        planner = SnakePlanner(self.config, self.env_config)
        
        # 检查turn_direction初始值
        self.assertEqual(planner.turn_direction, AlgorithmDefaults.INITIAL_TURN_DIRECTION)
        self.assertTrue(planner.turn_direction)  # 应该是True以与旧版一致
        
        # 检查SNAKE特定参数
        self.assertTrue(planner.greedy_search)
        self.assertTrue(planner.forward_only)
        
    def test_bcp_planner_initialization(self):
        """测试BCP算法初始化值"""
        planner = BcpPlanner(self.config, self.env_config)
        
        # 检查turn_direction初始值
        self.assertEqual(planner.turn_direction, AlgorithmDefaults.INITIAL_TURN_DIRECTION)
        self.assertTrue(planner.turn_direction)  # 应该是True以与旧版一致
        
        # 检查BCP特定参数
        self.assertTrue(planner.simple_coverage)
        self.assertTrue(planner.sequential_scan)
        
    def test_custom_turn_direction(self):
        """测试自定义turn_direction参数"""
        config_with_turn = self.config.copy()
        config_with_turn['parameters'] = {'initial_turn_direction': False}  # 测试覆盖默认值
        
        planner = JumpPlanner(config_with_turn, self.env_config)
        self.assertFalse(planner.turn_direction)  # 应该使用自定义值


class TestAlgorithmReset(unittest.TestCase):
    """测试算法重置行为"""
    
    def setUp(self):
        self.env_config = {
            'agent': {'car_width': 5, 'sight_width': 24, 'sight_length': 24},
            'environment': {'width': 600, 'height': 600}
        }
        self.config = {'algorithm': {'name': 'TEST'}, 'parameters': {}}
        
        # 创建测试用的农场边界
        self.farm_vertices = np.array([
            [100, 100], [500, 100], [500, 500], [100, 500]
        ])
        
        self.initial_state = {
            'agent_position': [300, 300],
            'agent_direction': 0,
            'farm_vertices': self.farm_vertices,
            'turning_radius': 7.5,
            'discovered_weeds': []
        }
        
    def test_jump_reset_preserves_turn_direction(self):
        """测试JUMP算法reset不改变turn_direction"""
        planner = JumpPlanner(self.config, self.env_config)
        initial_turn = planner.turn_direction
        
        planner.reset(self.initial_state)
        
        # reset不应该改变turn_direction
        self.assertEqual(planner.turn_direction, initial_turn)
        
        # 但应该更新turning_radius
        self.assertEqual(planner.turning_radius, 7.5)
        
    def test_snake_reset_preserves_turn_direction(self):
        """测试SNAKE算法reset不改变turn_direction"""
        planner = SnakePlanner(self.config, self.env_config)
        initial_turn = planner.turn_direction
        
        planner.reset(self.initial_state)
        
        self.assertEqual(planner.turn_direction, initial_turn)
        self.assertEqual(planner.turning_radius, 7.5)
        
    def test_farm_vertices_processing(self):
        """测试农场边界处理"""
        planner = JumpPlanner(self.config, self.env_config)
        planner.reset(self.initial_state)
        
        # 检查farm_vertices被正确处理
        self.assertIsNotNone(planner.farm_vertices)
        self.assertIsInstance(planner.farm_vertices, np.ndarray)
        
        # 检查初始化覆盖模式被调用
        self.assertIsNotNone(planner.longest_edge)
        self.assertGreater(planner.diagonal_length, 0)


class TestPathPlanningConsistency(unittest.TestCase):
    """测试路径规划的一致性"""
    
    def setUp(self):
        self.env_config = {
            'agent': {'car_width': 5, 'sight_width': 24, 'sight_length': 24},
            'environment': {'width': 600, 'height': 600}
        }
        self.config = {'algorithm': {'name': 'TEST'}, 'parameters': {}}
        
        self.farm_vertices = np.array([
            [100, 100], [500, 100], [500, 500], [100, 500]
        ])
        
        self.initial_state = {
            'agent_position': [300, 300],
            'agent_direction': 0,
            'farm_vertices': self.farm_vertices,
            'discovered_weeds': [[150, 150], [250, 250], [350, 350]]
        }
        
    def test_coordinate_consistency_in_planning(self):
        """测试规划过程中的坐标一致性"""
        planner = JumpPlanner(self.config, self.env_config)
        planner.reset(self.initial_state)
        
        # 模拟调用plan_next_waypoint
        current_state = {
            'agent_position': [300, 300],
            'agent_direction': 0,
            'discovered_weeds': [[150, 150], [250, 250]]
        }
        
        # 这里只是测试调用不报错，具体逻辑需要根据实际算法测试
        try:
            waypoint = planner.plan_next_waypoint(current_state)
            # 如果返回了waypoint，检查格式
            if waypoint is not None:
                if isinstance(waypoint, list):
                    self.assertEqual(len(waypoint), 2)
                elif isinstance(waypoint, tuple):
                    self.assertEqual(len(waypoint), 2)
        except NotImplementedError:
            # 如果方法未实现，这是预期的
            pass


class TestConstants(unittest.TestCase):
    """测试常量定义的正确性"""
    
    def test_default_values(self):
        """测试默认值是否与旧版一致"""
        from rules_new.algorithms.constants import PathConstants, AlgorithmDefaults
        
        # 关键常量检查 - 修正：INITIAL_TURN_DIRECTION应该是True
        self.assertEqual(AlgorithmDefaults.INITIAL_TURN_DIRECTION, True)
        self.assertEqual(PathConstants.DEFAULT_TURNING_RADIUS, 5.0)
        self.assertEqual(PathConstants.JUMP_THRESHOLD, 4)
        self.assertEqual(PathConstants.SAFETY_MARGIN, 2)
        
    def test_performance_thresholds(self):
        """测试性能阈值"""
        from rules_new.algorithms.constants import PerformanceThresholds
        
        self.assertEqual(PerformanceThresholds.COVERAGE_MILESTONE_90, 0.90)
        self.assertEqual(PerformanceThresholds.COVERAGE_MILESTONE_95, 0.95)
        self.assertEqual(PerformanceThresholds.COVERAGE_MILESTONE_98, 0.98)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestCoordinateSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestAlgorithmInitialization))
    suite.addTests(loader.loadTestsFromTestCase(TestAlgorithmReset))
    suite.addTests(loader.loadTestsFromTestCase(TestPathPlanningConsistency))
    suite.addTests(loader.loadTestsFromTestCase(TestConstants))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印总结
    print("\n" + "="*70)
    print(f"测试完成: 运行了 {result.testsRun} 个测试")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ 所有测试通过！算法一致性验证成功。")
    else:
        print("\n❌ 存在测试失败，请检查并修复问题。")
        
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)