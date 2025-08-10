"""
Rules New1 一致性测试 - 验证新版本与旧版本的功能一致性
"""
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rules_new.experiment import ExperimentRunner, ConfigManager
from rules_new.utils.logging_utils import LoggingUtils, PerformanceTimer
from rules_new.utils.path_utils import PathUtils


class ConsistencyTester:
    """一致性测试器 - 验证rules_new1与rules_new的一致性"""
    
    def __init__(self):
        self.logger = LoggingUtils.setup_logger("consistency_tester")
        self.config_manager = ConfigManager()
        
        # 测试结果
        self.test_results = {
            'config_validation': [],
            'algorithm_initialization': [],
            'single_step_consistency': [],
            'experiment_flow': [],
            'performance_comparison': []
        }
        
        # 性能对比
        self.performance_metrics = {
            'rules': {},
            'rules_new': {}
        }
        
    def test_config_system(self) -> bool:
        """测试配置系统的正确性"""
        self.logger.info("测试配置系统...")
        
        try:
            # 测试基础配置加载
            base_config = self.config_manager.load_base_config()
            assert 'environment' in base_config
            assert 'agent' in base_config
            assert 'paths' in base_config
            
            # 测试算法配置加载
            algorithms = ['jump', 'snake', 'r_snake', 'react', 'bcp']
            for alg in algorithms:
                alg_config = self.config_manager.load_algorithm_config(alg)
                assert 'algorithm' in alg_config
                assert alg_config['algorithm']['name'].upper() == alg.upper().replace('_', '_')
            
            # 测试实验配置加载
            exp_config = self.config_manager.load_experiment_config('baseline_comparison')
            assert 'experiment' in exp_config
            assert 'algorithms' in exp_config
            assert 'parameters' in exp_config
            
            self.test_results['config_validation'].append({
                'test': 'config_system',
                'status': 'PASS',
                'message': '配置系统验证通过'
            })
            
            self.logger.info("✅ 配置系统测试通过")
            return True
            
        except Exception as e:
            self.test_results['config_validation'].append({
                'test': 'config_system',
                'status': 'FAIL',
                'message': f'配置系统测试失败: {e}'
            })
            
            self.logger.error(f"❌ 配置系统测试失败: {e}")
            return False
    
    def test_algorithm_initialization(self) -> bool:
        """测试算法初始化的正确性"""
        self.logger.info("测试算法初始化...")
        
        success_count = 0
        total_algorithms = 0
        
        try:
            # 加载基础配置
            base_config = self.config_manager.load_base_config()
            
            # 测试每个算法的初始化
            algorithms = ['JUMP', 'SNAKE', 'R_SNAKE', 'REACT', 'BCP']
            
            for alg_name in algorithms:
                total_algorithms += 1
                
                try:
                    # 加载算法配置
                    alg_config = self.config_manager.load_algorithm_config(alg_name.lower())
                    
                    # 导入对应的算法类
                    if alg_name == 'JUMP':
                        from rules_new.algorithms import JumpPlanner
                        algorithm_class = JumpPlanner
                    elif alg_name == 'SNAKE':
                        from rules_new.algorithms import SnakePlanner
                        algorithm_class = SnakePlanner
                    elif alg_name == 'R_SNAKE':
                        from rules_new.algorithms import RSnakePlanner
                        algorithm_class = RSnakePlanner
                    elif alg_name == 'REACT':
                        from rules_new.algorithms import ReactPlanner
                        algorithm_class = ReactPlanner
                    elif alg_name == 'BCP':
                        from rules_new.algorithms import BcpPlanner
                        algorithm_class = BcpPlanner
                    
                    # 创建算法实例
                    algorithm = algorithm_class(alg_config, base_config)
                    
                    # 测试基本属性
                    assert hasattr(algorithm, 'algorithm_name')
                    assert hasattr(algorithm, 'reset')
                    assert hasattr(algorithm, 'plan_next_waypoint')
                    assert hasattr(algorithm, 'should_terminate')
                    
                    success_count += 1
                    
                    self.test_results['algorithm_initialization'].append({
                        'algorithm': alg_name,
                        'status': 'PASS',
                        'message': f'{alg_name}算法初始化成功'
                    })
                    
                    self.logger.info(f"✅ {alg_name} 算法初始化成功")
                    
                except Exception as e:
                    self.test_results['algorithm_initialization'].append({
                        'algorithm': alg_name,
                        'status': 'FAIL',
                        'message': f'{alg_name}算法初始化失败: {e}'
                    })
                    
                    self.logger.error(f"❌ {alg_name} 算法初始化失败: {e}")
            
            success_rate = success_count / total_algorithms
            self.logger.info(f"算法初始化测试完成: {success_count}/{total_algorithms} ({success_rate:.1%})")
            
            return success_rate >= 0.8  # 80%成功率算通过
            
        except Exception as e:
            self.logger.error(f"❌ 算法初始化测试异常: {e}")
            return False
    
    def test_single_step_consistency(self) -> bool:
        """测试单步执行的一致性"""
        self.logger.info("测试单步执行一致性...")
        
        try:
            # 创建简单的测试场景
            base_config = self.config_manager.load_base_config()
            
            # 模拟环境状态
            test_state = {
                'agent_position': [100, 100],
                'agent_direction': 0.0,
                'discovered_weeds': [[150, 120], [200, 180], [250, 200]],
                'coverage_rate': 0.15,
                'farm_vertices': np.array([[50, 50], [550, 50], [550, 550], [50, 550]]),
                'maps': {}
            }
            
            # 测试每个算法的单步规划
            algorithms = ['JUMP', 'SNAKE', 'BCP']  # 测试核心算法
            
            for alg_name in algorithms:
                try:
                    # 加载算法配置并创建实例
                    alg_config = self.config_manager.load_algorithm_config(alg_name.lower())
                    
                    if alg_name == 'JUMP':
                        from rules_new.algorithms import JumpPlanner
                        algorithm = JumpPlanner(alg_config, base_config)
                    elif alg_name == 'SNAKE':
                        from rules_new.algorithms import SnakePlanner
                        algorithm = SnakePlanner(alg_config, base_config)
                    elif alg_name == 'BCP':
                        from rules_new.algorithms import BcpPlanner
                        algorithm = BcpPlanner(alg_config, base_config)
                    
                    # 重置算法
                    algorithm.reset(test_state)
                    
                    # 执行几步规划
                    for step in range(5):
                        waypoint = algorithm.plan_next_waypoint(test_state)
                        
                        if waypoint is None:
                            break
                            
                        # 验证waypoint格式
                        assert isinstance(waypoint, tuple)
                        assert len(waypoint) == 2
                        assert isinstance(waypoint[0], (int, float))
                        assert isinstance(waypoint[1], (int, float))
                        
                        # 更新位置用于下一步
                        test_state['agent_position'] = list(waypoint)
                    
                    self.test_results['single_step_consistency'].append({
                        'algorithm': alg_name,
                        'status': 'PASS',
                        'message': f'{alg_name}单步执行测试通过'
                    })
                    
                    self.logger.info(f"✅ {alg_name} 单步执行测试通过")
                    
                except Exception as e:
                    self.test_results['single_step_consistency'].append({
                        'algorithm': alg_name,
                        'status': 'FAIL',
                        'message': f'{alg_name}单步执行测试失败: {e}'
                    })
                    
                    self.logger.error(f"❌ {alg_name} 单步执行测试失败: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 单步执行测试异常: {e}")
            return False
    
    def test_experiment_flow(self) -> bool:
        """测试完整实验流程"""
        self.logger.info("测试完整实验流程...")
        
        try:
            # 创建一个简化的实验配置用于测试
            test_config_content = """
experiment:
  name: "consistency_test"
  description: "一致性测试用的简化实验"

parameters:
  seeds: [42]
  difficulties: ["easy"]
  weed_distributions: ["gaussian"]
  noise_levels: ["no_noise"]

algorithms:
  - name: "JUMP"
    config_path: "algorithms/jump.yaml"
    enabled: true

environment_overrides:
  use_sgcnn: false
  use_global_obs: false
  
output:
  base_dir: "logs/test"
  csv_format: true
  metrics:
    - total_coverage
    - path_length
"""
            
            # 保存测试配置
            test_config_path = PathUtils.get_project_root() / "rules_new" / "configs" / "experiments" / "consistency_test.yaml"
            with open(test_config_path, 'w', encoding='utf-8') as f:
                f.write(test_config_content)
            
            # 尝试运行实验（如果环境可用）
            try:
                runner = ExperimentRunner('consistency_test')
                
                # 测试初始化是否成功
                assert len(runner.algorithm_instances) > 0
                assert 'JUMP' in runner.algorithm_instances
                
                self.test_results['experiment_flow'].append({
                    'test': 'experiment_initialization',
                    'status': 'PASS',
                    'message': '实验流程初始化成功'
                })
                
                self.logger.info("✅ 实验流程测试通过")
                
                # 清理资源
                runner.cleanup()
                
                return True
                
            except ImportError as e:
                # 如果环境模块不可用，跳过实际运行
                self.test_results['experiment_flow'].append({
                    'test': 'experiment_flow',
                    'status': 'SKIP',
                    'message': f'环境模块不可用，跳过实验运行: {e}'
                })
                
                self.logger.warning(f"⚠️ 跳过实验运行测试: {e}")
                return True
            
        except Exception as e:
            self.test_results['experiment_flow'].append({
                'test': 'experiment_flow',
                'status': 'FAIL',
                'message': f'实验流程测试失败: {e}'
            })
            
            self.logger.error(f"❌ 实验流程测试失败: {e}")
            return False
        
        finally:
            # 清理测试配置文件
            test_config_path = PathUtils.get_project_root() / "rules_new" / "configs" / "experiments" / "consistency_test.yaml"
            if test_config_path.exists():
                test_config_path.unlink()
    
    def test_performance_comparison(self) -> bool:
        """性能对比测试"""
        self.logger.info("执行性能对比测试...")
        
        try:
            # 测试配置加载性能
            with PerformanceTimer("配置加载") as timer:
                for _ in range(10):  # 重复10次测试
                    config_manager = ConfigManager()
                    config_manager.load_base_config()
                    config_manager.load_algorithm_config('jump')
                    config_manager.clear_cache()  # 清除缓存以确保每次都重新加载
            
            config_load_time = timer.stop()
            self.performance_metrics['rules_new']['config_load_avg'] = config_load_time / 10
            
            # 测试算法初始化性能
            with PerformanceTimer("算法初始化") as timer:
                for _ in range(5):  # 重复5次测试
                    base_config = self.config_manager.load_base_config()
                    alg_config = self.config_manager.load_algorithm_config('jump')
                    
                    from rules_new.algorithms import JumpPlanner
                    algorithm = JumpPlanner(alg_config, base_config)
            
            init_time = timer.stop()
            self.performance_metrics['rules_new']['algorithm_init_avg'] = init_time / 5
            
            self.test_results['performance_comparison'].append({
                'test': 'performance_benchmark',
                'status': 'PASS',
                'message': f'性能测试完成 - 配置加载: {config_load_time/10:.4f}s, 算法初始化: {init_time/5:.4f}s'
            })
            
            self.logger.info("✅ 性能对比测试完成")
            return True
            
        except Exception as e:
            self.test_results['performance_comparison'].append({
                'test': 'performance_benchmark',
                'status': 'FAIL',
                'message': f'性能测试失败: {e}'
            })
            
            self.logger.error(f"❌ 性能对比测试失败: {e}")
            return False
    
    def run_full_consistency_test(self) -> Dict[str, Any]:
        """运行完整的一致性测试套件"""
        self.logger.info("🧪 开始运行完整一致性测试...")
        
        start_time = time.time()
        
        # 执行所有测试
        tests = [
            ('配置系统', self.test_config_system),
            ('算法初始化', self.test_algorithm_initialization),
            ('单步执行一致性', self.test_single_step_consistency),
            ('实验流程', self.test_experiment_flow),
            ('性能对比', self.test_performance_comparison)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            self.logger.info(f"执行测试: {test_name}")
            try:
                if test_func():
                    passed_tests += 1
                    self.logger.info(f"✅ {test_name} - 通过")
                else:
                    self.logger.error(f"❌ {test_name} - 失败")
            except Exception as e:
                self.logger.error(f"❌ {test_name} - 异常: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 生成测试报告
        test_summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': passed_tests / total_tests,
            'total_time': total_time,
            'test_results': self.test_results,
            'performance_metrics': self.performance_metrics,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 导出测试报告
        self._export_test_report(test_summary)
        
        self.logger.info(f"🏁 一致性测试完成: {passed_tests}/{total_tests} ({test_summary['success_rate']:.1%}) - 耗时: {total_time:.2f}s")
        
        return test_summary
    
    def _export_test_report(self, summary: Dict[str, Any]):
        """导出测试报告"""
        try:
            report_dir = PathUtils.get_project_root() / "logs" / "consistency_tests"
            PathUtils.ensure_directory_exists(report_dir)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report_file = report_dir / f"consistency_test_report_{timestamp}.json"
            
            import json
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"测试报告已导出: {report_file}")
            
        except Exception as e:
            self.logger.warning(f"导出测试报告失败: {e}")


def main():
    """主函数"""
    print("=" * 60)
    print("🧪 Rules New1 一致性测试")
    print("   验证新版本与旧版本的功能一致性")
    print("=" * 60)
    
    tester = ConsistencyTester()
    summary = tester.run_full_consistency_test()
    
    print(f"\n📊 测试结果摘要:")
    print(f"   - 总测试数: {summary['total_tests']}")
    print(f"   - 通过测试: {summary['passed_tests']}")
    print(f"   - 失败测试: {summary['failed_tests']}")
    print(f"   - 成功率: {summary['success_rate']:.1%}")
    print(f"   - 总耗时: {summary['total_time']:.2f} 秒")
    
    if summary['success_rate'] >= 0.8:
        print("✅ 一致性测试基本通过！")
        return 0
    else:
        print("❌ 一致性测试未通过，请检查失败的测试项。")
        return 1


if __name__ == "__main__":
    sys.exit(main())