#!/usr/bin/env python3
"""
Phase 1 优化测试脚本

验证：
1. 坐标系统统一性
2. 异常处理机制
3. 错误恢复功能
4. 行为一致性

作者：Rules_new优化团队
版本：2.0.0
"""

import sys
import numpy as np
from pathlib import Path
import traceback
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入核心组件
from rules_new.core import CoordinateSystem as CS
from rules_new.core.exceptions import (
    AlgorithmError, CoordinateError, StateError,
    handle_errors, RecoverableError
)
from rules_new.core.recovery_manager import RecoveryManager
from rules_new.core.state_validator import StateValidator
from rules_new.core.performance_monitor import PerformanceMonitor

# 导入算法
from rules_new.algorithms import JumpPlanner, SnakePlanner, BcpPlanner, ReactPlanner


class Phase1OptimizationTester:
    """Phase 1 优化测试器"""
    
    def __init__(self):
        self.test_results = []
        self.failed_tests = []
        
    def run_all_tests(self):
        """运行所有测试"""
        print("=" * 60)
        print("Phase 1 优化测试套件")
        print("=" * 60)
        
        # 测试1：坐标系统统一性
        self.test_coordinate_system()
        
        # 测试2：异常处理机制
        self.test_exception_handling()
        
        # 测试3：错误恢复功能
        self.test_error_recovery()
        
        # 测试4：算法集成
        self.test_algorithm_integration()
        
        # 测试5：性能监控
        self.test_performance_monitoring()
        
        # 测试6：状态验证
        self.test_state_validation()
        
        # 输出测试结果
        self.print_results()
        
    def test_coordinate_system(self):
        """测试坐标系统统一性"""
        print("\n[测试1] 坐标系统统一性")
        print("-" * 40)
        
        try:
            # 测试各种格式输入
            test_cases = [
                ([100, 200], (200, 100)),  # 列表 -> 元组
                ((100, 200), (100, 200)),  # 元组保持不变
                (np.array([100, 200]), (200, 100)),  # numpy数组
                ([100.5, 200.3], (200.3, 100.5)),  # 浮点数
            ]
            
            all_passed = True
            for input_pos, expected in test_cases:
                result = CS.normalize(input_pos)
                if result != expected:
                    print(f"  ❌ 失败: 输入{input_pos} -> 期望{expected}, 实际{result}")
                    all_passed = False
                else:
                    print(f"  ✅ 通过: {input_pos} -> {result}")
            
            # 测试距离计算
            dist = CS.distance((0, 0), (3, 4))
            if abs(dist - 5.0) < 0.001:
                print(f"  ✅ 距离计算正确: {dist}")
            else:
                print(f"  ❌ 距离计算错误: {dist}")
                all_passed = False
            
            # 测试批量操作
            positions = [[1, 2], [3, 4], [5, 6]]
            normalized = [CS.normalize(p) for p in positions]
            expected = [(2, 1), (4, 3), (6, 5)]
            if normalized == expected:
                print(f"  ✅ 批量转换正确")
            else:
                print(f"  ❌ 批量转换错误")
                all_passed = False
            
            self.test_results.append(("坐标系统", all_passed))
            
        except Exception as e:
            print(f"  ❌ 测试异常: {e}")
            self.test_results.append(("坐标系统", False))
            self.failed_tests.append(("坐标系统", str(e)))
    
    def test_exception_handling(self):
        """测试异常处理机制"""
        print("\n[测试2] 异常处理机制")
        print("-" * 40)
        
        try:
            # 测试算法错误
            alg_error = AlgorithmError(
                algorithm_name="TEST_ALG",
                phase="planning",
                message="测试算法错误",
                context={'test': True}
            )
            
            assert alg_error.error_code == "ALG_TEST_ALG_PLANNING"
            assert alg_error.recovery_hint is not None
            print(f"  ✅ 算法错误创建成功: {alg_error.error_code}")
            
            # 测试坐标错误
            coord_error = CoordinateError(
                operation="normalize",
                coordinate="invalid",
                message="无效坐标"
            )
            
            assert coord_error.error_code == "COORD_NORMALIZE"
            print(f"  ✅ 坐标错误创建成功: {coord_error.error_code}")
            
            # 测试错误处理装饰器
            @handle_errors(AlgorithmError, save_checkpoint=False, reraise=False)
            def test_function():
                raise AlgorithmError("TEST", "execution", "测试错误")
            
            result = test_function()
            assert result is None  # 错误被处理，返回None
            print(f"  ✅ 错误处理装饰器工作正常")
            
            # 测试可恢复错误
            recovery_called = False
            def recovery_action():
                nonlocal recovery_called
                recovery_called = True
            
            recoverable = RecoverableError(
                "可恢复错误",
                recovery_action=recovery_action,
                max_retries=1
            )
            
            success = recoverable.attempt_recovery()
            assert recovery_called
            assert success
            print(f"  ✅ 可恢复错误机制正常")
            
            self.test_results.append(("异常处理", True))
            
        except Exception as e:
            print(f"  ❌ 测试异常: {e}")
            self.test_results.append(("异常处理", False))
            self.failed_tests.append(("异常处理", str(e)))
    
    def test_error_recovery(self):
        """测试错误恢复功能"""
        print("\n[测试3] 错误恢复功能")
        print("-" * 40)
        
        try:
            # 创建临时恢复管理器
            recovery_manager = RecoveryManager(
                checkpoint_dir=Path("./test_checkpoints"),
                max_checkpoints=5,
                auto_recovery=True
            )
            
            # 测试检查点保存
            test_state = {
                'agent_position': [100, 200],
                'agent_direction': 45,
                'coverage_rate': 0.5,
                'discovered_weeds': [[10, 20], [30, 40]]
            }
            
            checkpoint_path = recovery_manager.save_checkpoint(test_state, "test_checkpoint")
            assert checkpoint_path.exists()
            print(f"  ✅ 检查点保存成功: {checkpoint_path.name}")
            
            # 测试检查点加载
            loaded_state = recovery_manager.load_checkpoint("test_checkpoint")
            
            # 验证坐标已标准化
            assert loaded_state['agent_position'] == (200, 100)  # 已转换为(y,x)
            print(f"  ✅ 检查点加载成功，坐标已标准化")
            
            # 测试错误恢复
            coord_error = CoordinateError("test", [999, 999], "测试坐标错误")
            recovered_state = recovery_manager.recover_from_error(coord_error, test_state)
            
            assert 'agent_position' in recovered_state
            print(f"  ✅ 错误恢复成功")
            
            # 测试恢复统计
            stats = recovery_manager.get_recovery_statistics()
            assert stats['total_errors'] > 0
            print(f"  ✅ 恢复统计: 错误数={stats['total_errors']}")
            
            # 清理测试文件
            import shutil
            if Path("./test_checkpoints").exists():
                shutil.rmtree("./test_checkpoints")
            
            self.test_results.append(("错误恢复", True))
            
        except Exception as e:
            print(f"  ❌ 测试异常: {e}")
            self.test_results.append(("错误恢复", False))
            self.failed_tests.append(("错误恢复", str(e)))
    
    def test_algorithm_integration(self):
        """测试算法集成"""
        print("\n[测试4] 算法集成")
        print("-" * 40)
        
        try:
            # 测试配置
            config = {
                'algorithm': {'name': 'JUMP', 'type': 'rule_based'},
                'parameters': {},
                'performance': {'max_iterations': 100, 'timeout_seconds': 10}
            }
            
            env_config = {
                'environment': {'width': 600, 'height': 600},
                'agent': {'car_width': 5, 'sight_width': 24}
            }
            
            # 测试各算法初始化
            algorithms = [
                ('JUMP', JumpPlanner),
                ('SNAKE', SnakePlanner),
                ('BCP', BcpPlanner),
                ('REACT', ReactPlanner)
            ]
            
            all_passed = True
            for alg_name, alg_class in algorithms:
                try:
                    config['algorithm']['name'] = alg_name
                    algorithm = alg_class(config, env_config)
                    
                    # 测试reset
                    initial_state = {
                        'agent_position': [300, 300],
                        'agent_direction': 0,
                        'discovered_weeds': [[100, 100], [200, 200]],
                        'coverage_rate': 0.0
                    }
                    
                    algorithm.reset(initial_state)
                    
                    # 验证坐标已统一
                    assert algorithm.current_position == (300, 300)
                    assert len(algorithm.discovered_weeds) == 2
                    assert algorithm.discovered_weeds[0] == (100, 100)
                    
                    print(f"  ✅ {alg_name} 算法集成成功")
                    
                except Exception as e:
                    print(f"  ❌ {alg_name} 算法集成失败: {e}")
                    all_passed = False
            
            self.test_results.append(("算法集成", all_passed))
            
        except Exception as e:
            print(f"  ❌ 测试异常: {e}")
            self.test_results.append(("算法集成", False))
            self.failed_tests.append(("算法集成", str(e)))
    
    def test_performance_monitoring(self):
        """测试性能监控"""
        print("\n[测试5] 性能监控")
        print("-" * 40)
        
        try:
            monitor = PerformanceMonitor("test_monitor")
            
            # 测试计时功能
            monitor.start_timer("test_operation")
            time.sleep(0.01)  # 模拟操作
            elapsed = monitor.stop_timer("test_operation")
            
            assert elapsed > 0.01
            print(f"  ✅ 计时功能正常: {elapsed:.3f}秒")
            
            # 测试内存监控
            memory_info = monitor.measure_memory()
            assert 'rss_mb' in memory_info
            assert memory_info['rss_mb'] > 0
            print(f"  ✅ 内存监控正常: {memory_info['rss_mb']:.1f} MB")
            
            # 测试CPU监控
            cpu_percent = monitor.measure_cpu()
            assert cpu_percent >= 0
            print(f"  ✅ CPU监控正常: {cpu_percent:.1f}%")
            
            # 测试统计信息
            stats = monitor.get_statistics()
            assert 'total_operations' in stats
            print(f"  ✅ 统计信息正常: 总操作数={stats['total_operations']}")
            
            # 测试报告生成
            report = monitor.generate_report()
            assert len(report) > 0
            print(f"  ✅ 报告生成成功")
            
            self.test_results.append(("性能监控", True))
            
        except Exception as e:
            print(f"  ❌ 测试异常: {e}")
            self.test_results.append(("性能监控", False))
            self.failed_tests.append(("性能监控", str(e)))
    
    def test_state_validation(self):
        """测试状态验证"""
        print("\n[测试6] 状态验证")
        print("-" * 40)
        
        try:
            validator = StateValidator(
                max_speed=10.0,
                max_angular_speed=90.0,
                position_tolerance=0.1,
                angle_tolerance=1.0
            )
            
            # 测试位置更新验证
            old_pos = [100, 100]
            new_pos = [102, 103]
            
            result = validator.validate_position_update(
                old_pos, new_pos, action=[5, 0], dt=1.0
            )
            
            assert 'valid' in result
            assert 'actual_distance' in result
            print(f"  ✅ 位置验证正常: 距离={result['actual_distance']:.2f}")
            
            # 测试角度更新验证
            angle_result = validator.validate_angle_update(
                old_angle=0, new_angle=45, action_angle=45, dt=1.0
            )
            
            assert angle_result['valid']
            print(f"  ✅ 角度验证正常: 变化={angle_result['angle_diff']:.1f}度")
            
            # 测试状态转换验证
            old_state = {
                'agent_position': [100, 100],
                'agent_direction': 0,
                'coverage_rate': 0.5,
                'weed_count': 10
            }
            
            new_state = {
                'agent_position': [105, 105],
                'agent_direction': 30,
                'coverage_rate': 0.55,
                'weed_count': 8
            }
            
            transition_result = validator.validate_state_transition(
                old_state, new_state, action=[7, 30]
            )
            
            assert transition_result['valid']
            print(f"  ✅ 状态转换验证正常")
            
            # 获取统计信息
            stats = validator.get_statistics()
            assert stats['stats']['total_validations'] > 0
            print(f"  ✅ 验证统计: 总验证数={stats['stats']['total_validations']}")
            
            self.test_results.append(("状态验证", True))
            
        except Exception as e:
            print(f"  ❌ 测试异常: {e}")
            self.test_results.append(("状态验证", False))
            self.failed_tests.append(("状态验证", str(e)))
    
    def print_results(self):
        """输出测试结果"""
        print("\n" + "=" * 60)
        print("测试结果汇总")
        print("=" * 60)
        
        passed_count = 0
        failed_count = 0
        
        for test_name, passed in self.test_results:
            status = "✅ 通过" if passed else "❌ 失败"
            print(f"{test_name:20} {status}")
            if passed:
                passed_count += 1
            else:
                failed_count += 1
        
        print("-" * 60)
        print(f"总计: {passed_count} 通过, {failed_count} 失败")
        
        if self.failed_tests:
            print("\n失败详情:")
            for test_name, error in self.failed_tests:
                print(f"  - {test_name}: {error}")
        
        # 返回是否所有测试都通过
        return failed_count == 0


def main():
    """主函数"""
    tester = Phase1OptimizationTester()
    
    try:
        tester.run_all_tests()
        
        # 判断测试结果
        all_passed = all(passed for _, passed in tester.test_results)
        
        if all_passed:
            print("\n🎉 所有Phase 1优化测试通过！")
            return 0
        else:
            print("\n⚠️ 部分测试失败，请检查日志")
            return 1
            
    except Exception as e:
        print(f"\n❌ 测试执行失败: {e}")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())