#!/usr/bin/env python3
"""
测试基准测试系统的基本功能

验证各个组件是否正常工作
"""

import sys
from pathlib import Path
import logging

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """测试导入"""
    try:
        from rules_new.benchmark import (
            BenchmarkRunner,
            ScenarioGenerator,
            MetricCollector,
            VisualizationManager,
            ResultAnalyzer
        )
        logger.info("✅ 所有组件导入成功")
        return True
    except ImportError as e:
        logger.error(f"❌ 导入失败: {e}")
        return False


def test_scenario_generation():
    """测试场景生成的确定性"""
    try:
        from rules_new.benchmark import ScenarioGenerator
        
        # 创建基础配置
        config = {
            'difficulty_levels': {
                'easy': {
                    'obstacle_range': [0, 0],
                    'weed_num': 50,
                    'map_ids': [1, 2]
                }
            },
            'noise_sets': {
                'no_noise': [0, 0, 0]
            }
        }
        
        generator = ScenarioGenerator(config)
        
        # 测试相同seed生成相同场景
        scenario1 = generator.generate_scenario(
            seed=42,
            difficulty='easy',
            map_id=1,
            weed_distribution='uniform',
            noise_level='no_noise'
        )
        
        scenario2 = generator.generate_scenario(
            seed=42,
            difficulty='easy',
            map_id=1,
            weed_distribution='uniform',
            noise_level='no_noise'
        )
        
        # 验证场景ID相同
        assert scenario1['scenario_id'] == scenario2['scenario_id']
        # 验证杂草位置相同
        assert scenario1['weed_positions'] == scenario2['weed_positions']
        
        logger.info("✅ 场景生成确定性测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 场景生成测试失败: {e}")
        return False


def test_metric_collector():
    """测试指标收集器"""
    try:
        from rules_new.benchmark import MetricCollector
        
        collector = MetricCollector(coverage_thresholds=[0.90, 0.95, 0.98])
        
        # 创建模拟数据
        trajectory_data = {
            'coverage_history': [0.0, 0.3, 0.6, 0.9, 0.95, 0.98, 0.99],
            'distance_history': [0, 100, 200, 300, 400, 500, 600],
            'position_history': [(0, 0), (10, 0), (20, 0), (30, 0)],
            'collision_info': {
                'occurred': False
            },
            'time_info': {
                'total_time': 10.0,
                'planning_time': 2.0,
                'execution_time': 8.0
            }
        }
        
        env_info = {
            'scenario_id': 'test_scenario',
            'seed': 42,
            'difficulty': 'easy'
        }
        
        algorithm_info = {
            'name': 'TEST_ALGO'
        }
        
        # 收集指标
        metrics = collector.collect_metrics(trajectory_data, env_info, algorithm_info)
        
        # 验证关键指标
        assert metrics['algorithm'] == 'TEST_ALGO'
        assert metrics['final_coverage'] == 0.99
        assert metrics['coverage_90_distance'] == 300  # 第4个点达到90%
        assert metrics['coverage_95_distance'] == 400  # 第5个点达到95%
        assert metrics['coverage_98_distance'] == 500  # 第6个点达到98%
        
        logger.info("✅ 指标收集器测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 指标收集器测试失败: {e}")
        return False


def test_visualization_manager():
    """测试可视化管理器"""
    try:
        from rules_new.benchmark import VisualizationManager
        import tempfile
        import numpy as np
        
        # 使用临时目录
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = VisualizationManager(
                output_dir=tmpdir,
                save_finished_picture=True
            )
            
            # 创建模拟环境状态
            env_state = {
                'width': 600,
                'height': 600,
                'farm_vertices': [[100, 100], [500, 100], [500, 500], [100, 500]],
                'obstacles': [],
                'weeds': [[200, 200], [300, 300], [400, 400]]
            }
            
            trajectory_data = {
                'position_history': [(150, 150), (250, 250), (350, 350)],
                'coverage_history': [0.0, 0.5, 0.99],
                'distance_history': [0, 100, 200],
                'covered_weeds': {0, 1}
            }
            
            # 测试保存场景完成图片
            filepath = viz.save_scenario_completion(
                algorithm_name='TEST',
                scenario_id='test_s1',
                env_state=env_state,
                trajectory_data=trajectory_data,
                completion_reason='success'
            )
            
            assert filepath is not None
            assert filepath.exists()
            
            logger.info("✅ 可视化管理器测试通过")
            return True
            
    except Exception as e:
        logger.error(f"❌ 可视化管理器测试失败: {e}")
        return False


def test_result_analyzer():
    """测试结果分析器"""
    try:
        from rules_new.benchmark import ResultAnalyzer
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ResultAnalyzer(tmpdir)
            
            # 创建模拟结果
            results = [
                {
                    'algorithm': 'ALGO1',
                    'scenario': {
                        'scenario_id': 's1',
                        'seed': 1,
                        'difficulty': 'easy',
                        'weed_distribution': 'uniform',
                        'noise_level': 'no_noise'
                    },
                    'metrics': {
                        'final_coverage': 0.99,
                        'coverage_90_distance': 300,
                        'coverage_95_distance': 400,
                        'coverage_98_distance': 500,
                        'total_distance': 600,
                        'total_steps': 100,
                        'collision_occurred': False,
                        'overall_efficiency_score': 0.8
                    },
                    'runtime': 5.0
                },
                {
                    'algorithm': 'ALGO2',
                    'scenario': {
                        'scenario_id': 's1',
                        'seed': 1,
                        'difficulty': 'easy',
                        'weed_distribution': 'uniform',
                        'noise_level': 'no_noise'
                    },
                    'metrics': {
                        'final_coverage': 0.95,
                        'coverage_90_distance': 350,
                        'coverage_95_distance': 450,
                        'coverage_98_distance': -1,
                        'total_distance': 700,
                        'total_steps': 120,
                        'collision_occurred': True,
                        'overall_efficiency_score': 0.6
                    },
                    'runtime': 6.0
                }
            ]
            
            # 分析结果
            summary = analyzer.analyze(results)
            
            # 验证分析结果
            assert 'overall_statistics' in summary
            assert 'algorithm_rankings' in summary
            assert 'ALGO1' in summary['overall_statistics']
            assert 'ALGO2' in summary['overall_statistics']
            
            logger.info("✅ 结果分析器测试通过")
            return True
            
    except Exception as e:
        logger.error(f"❌ 结果分析器测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    logger.info("=" * 60)
    logger.info("开始测试基准测试系统")
    logger.info("=" * 60)
    
    tests = [
        ("导入测试", test_imports),
        ("场景生成测试", test_scenario_generation),
        ("指标收集测试", test_metric_collector),
        ("可视化管理测试", test_visualization_manager),
        ("结果分析测试", test_result_analyzer)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n运行: {test_name}")
        if test_func():
            passed += 1
        else:
            failed += 1
    
    logger.info("\n" + "=" * 60)
    logger.info(f"测试完成: {passed} 通过, {failed} 失败")
    logger.info("=" * 60)
    
    if failed == 0:
        logger.info("🎉 所有测试通过！基准测试系统工作正常。")
        logger.info("\n下一步：")
        logger.info("1. 运行快速测试: python run_benchmark.py --quick-test")
        logger.info("2. 使用自定义配置: python run_benchmark.py --config-dir ./configs")
        logger.info("3. 保存完成图片: python run_benchmark.py --save-finished-picture")
    else:
        logger.error("❌ 部分测试失败，请检查相关组件。")
        sys.exit(1)


if __name__ == '__main__':
    main()