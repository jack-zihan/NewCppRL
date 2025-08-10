#!/usr/bin/env python3
"""
架构验证测试 - 环境架构完整性和兼容性验证

整合架构相关的所有测试功能，替代多个final_*和architecture_*测试文件。
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from tests.utils.environment_manager import EnvironmentManager
from tests.utils.consistency_tester import UnifiedConsistencyTester


def test_environment_creation(versions: List[str]) -> Dict[str, Any]:
    """
    测试环境创建功能
    
    Args:
        versions: 要测试的版本列表
        
    Returns:
        测试结果
    """
    print("🏗️ 环境创建测试")
    
    manager = EnvironmentManager(verbose=False)
    results = {
        'test_type': 'environment_creation',
        'versions_tested': versions,
        'results': {},
        'overall_success': True
    }
    
    for version in versions:
        print(f"  测试版本 {version}...")
        
        version_result = {
            'version': version,
            'creation_success': False,
            'reset_success': False,
            'step_success': False,
            'cleanup_success': False,
            'error': None
        }
        
        try:
            # 测试环境创建
            with manager.create_environment_pair(version, seed=0) as (old_env, new_env):
                version_result['creation_success'] = True
                version_result['reset_success'] = True
                
                # 测试基本step操作
                action = old_env.action_space.sample()
                old_result = old_env.step(action)
                new_result = new_env.step(action)
                
                version_result['step_success'] = True
                version_result['cleanup_success'] = True
                
                print(f"    ✅ 版本 {version} 环境创建正常")
                
        except Exception as e:
            version_result['error'] = str(e)
            results['overall_success'] = False
            print(f"    ❌ 版本 {version} 环境创建失败: {str(e)}")
        
        results['results'][version] = version_result
    
    return results


def test_interface_compatibility(versions: List[str]) -> Dict[str, Any]:
    """
    测试接口兼容性
    
    Args:
        versions: 要测试的版本列表
        
    Returns:
        测试结果
    """
    print("🔌 接口兼容性测试")
    
    manager = EnvironmentManager(verbose=False)
    results = {
        'test_type': 'interface_compatibility',
        'versions_tested': versions,
        'results': {},
        'overall_compatible': True
    }
    
    for version in versions:
        print(f"  测试版本 {version}...")
        
        version_result = {
            'version': version,
            'compatible': True,
            'interface_checks': {},
            'errors': []
        }
        
        try:
            # 获取环境信息
            env_info = manager.get_environment_info(version)
            
            if 'error' in env_info:
                version_result['compatible'] = False
                version_result['errors'].append(env_info['error'])
                results['overall_compatible'] = False
                print(f"    ❌ 版本 {version} 信息获取失败")
                continue
            
            # 检查接口一致性
            old_info = env_info['old_env']
            new_info = env_info['new_env']
            
            # 动作空间检查
            if old_info['action_space_type'] != new_info['action_space_type']:
                version_result['compatible'] = False
                version_result['errors'].append(f"Action space type mismatch: {old_info['action_space_type']} vs {new_info['action_space_type']}")
            
            if old_info.get('action_space_n') != new_info.get('action_space_n'):
                version_result['compatible'] = False
                version_result['errors'].append(f"Action space size mismatch: {old_info.get('action_space_n')} vs {new_info.get('action_space_n')}")
            
            # 观测空间检查
            if old_info['observation_space_type'] != new_info['observation_space_type']:
                version_result['compatible'] = False
                version_result['errors'].append(f"Observation space type mismatch: {old_info['observation_space_type']} vs {new_info['observation_space_type']}")
            
            version_result['interface_checks'] = {
                'action_space_compatible': old_info['action_space_type'] == new_info['action_space_type'],
                'action_size_compatible': old_info.get('action_space_n') == new_info.get('action_space_n'),
                'observation_space_compatible': old_info['observation_space_type'] == new_info['observation_space_type']
            }
            
            if version_result['compatible']:
                print(f"    ✅ 版本 {version} 接口兼容")
            else:
                results['overall_compatible'] = False
                print(f"    ❌ 版本 {version} 接口不兼容")
                
        except Exception as e:
            version_result['compatible'] = False
            version_result['errors'].append(str(e))
            results['overall_compatible'] = False
            print(f"    ❌ 版本 {version} 兼容性测试异常: {str(e)}")
        
        results['results'][version] = version_result
    
    return results


def test_mathematical_equivalence(version: str, seeds: List[int], steps: int) -> Dict[str, Any]:
    """
    测试数学等价性
    
    Args:
        version: 环境版本
        seeds: 测试种子
        steps: 测试步数
        
    Returns:
        测试结果
    """
    print(f"🧮 数学等价性测试 (版本: {version})")
    
    tester = UnifiedConsistencyTester(tolerance=1e-15, verbose=False)
    
    # 执行高精度一致性测试
    result = tester.test_all_consistency(version, seeds, steps)
    
    mathematical_result = {
        'test_type': 'mathematical_equivalence',
        'version': version,
        'high_precision_tolerance': 1e-15,
        'seeds_tested': len(seeds),
        'steps_per_seed': steps,
        'mathematically_equivalent': result['overall_consistent'],
        'detailed_results': result['results'],
        'equivalence_analysis': {}
    }
    
    # 分析等价性
    dynamics_consistent = result['results']['dynamics']['overall_consistent']
    rewards_consistent = result['results']['rewards']['overall_consistent']
    observations_consistent = result['results']['observations']['overall_consistent']
    
    mathematical_result['equivalence_analysis'] = {
        'dynamics_equivalent': dynamics_consistent,
        'rewards_equivalent': rewards_consistent,
        'observations_equivalent': observations_consistent,
        'full_equivalence': dynamics_consistent and rewards_consistent and observations_consistent
    }
    
    if mathematical_result['mathematically_equivalent']:
        print("  ✅ 数学等价性验证通過")
    else:
        print("  ❌ 数学等价性验证失败")
        
        # 打印详细的不等价分析
        if not dynamics_consistent:
            print("    - 动力学不等价")
        if not rewards_consistent:
            print("    - 奖励计算不等价")
        if not observations_consistent:
            print("    - 观测输出不等价")
    
    return mathematical_result


def test_stress_conditions(version: str) -> Dict[str, Any]:
    """
    测试压力条件
    
    Args:
        version: 环境版本
        
    Returns:
        测试结果
    """
    print(f"💪 压力条件测试 (版本: {version})")
    
    manager = EnvironmentManager(verbose=False)
    results = {
        'test_type': 'stress_conditions',
        'version': version,
        'tests': {},
        'overall_robust': True
    }
    
    # 长序列测试
    print("  测试长序列稳定性...")
    try:
        tester = UnifiedConsistencyTester(tolerance=1e-6, verbose=False)
        long_sequence_result = tester.test_all_consistency(version, seeds=[0], num_steps=50)
        
        results['tests']['long_sequence'] = {
            'success': long_sequence_result['overall_consistent'],
            'steps_tested': 50,
            'description': '长序列稳定性测试'
        }
        
        if long_sequence_result['overall_consistent']:
            print("    ✅ 长序列测试通过")
        else:
            print("    ❌ 长序列测试失败")
            results['overall_robust'] = False
            
    except Exception as e:
        results['tests']['long_sequence'] = {
            'success': False,
            'error': str(e),
            'description': '长序列稳定性测试'
        }
        results['overall_robust'] = False
        print(f"    ❌ 长序列测试异常: {str(e)}")
    
    # 多种子一致性测试
    print("  测试多种子一致性...")
    try:
        multi_seed_result = tester.test_all_consistency(version, seeds=list(range(10)), num_steps=10)
        
        results['tests']['multi_seed'] = {
            'success': multi_seed_result['overall_consistent'],
            'seeds_tested': 10,
            'description': '多种子一致性测试'
        }
        
        if multi_seed_result['overall_consistent']:
            print("    ✅ 多种子测试通过")
        else:
            print("    ❌ 多种子测试失败")
            results['overall_robust'] = False
            
    except Exception as e:
        results['tests']['multi_seed'] = {
            'success': False,
            'error': str(e),
            'description': '多种子一致性测试'
        }
        results['overall_robust'] = False
        print(f"    ❌ 多种子测试异常: {str(e)}")
    
    # 边界动作测试
    print("  测试边界动作处理...")
    try:
        with manager.create_environment_pair(version, seed=0) as (old_env, new_env):
            # 测试边界动作值
            boundary_actions = [0, old_env.action_space.n - 1]  # 最小和最大动作
            
            boundary_success = True
            for action in boundary_actions:
                old_result = old_env.step(action)
                new_result = new_env.step(action)
                
                step_comparison = manager.compare_environment_step_results(old_result, new_result)
                if not step_comparison['consistent']:
                    boundary_success = False
                    break
            
            results['tests']['boundary_actions'] = {
                'success': boundary_success,
                'actions_tested': boundary_actions,
                'description': '边界动作测试'
            }
            
            if boundary_success:
                print("    ✅ 边界动作测试通过")
            else:
                print("    ❌ 边界动作测试失败")
                results['overall_robust'] = False
                
    except Exception as e:
        results['tests']['boundary_actions'] = {
            'success': False,
            'error': str(e),
            'description': '边界动作测试'
        }
        results['overall_robust'] = False
        print(f"    ❌ 边界动作测试异常: {str(e)}")
    
    return results


def run_comprehensive_architecture_test(versions: List[str], test_seeds: List[int], 
                                       test_steps: int, output_file: str = None) -> Dict[str, Any]:
    """
    运行综合架构测试
    
    Args:
        versions: 测试版本列表
        test_seeds: 测试种子
        test_steps: 测试步数
        output_file: 结果保存文件
        
    Returns:
        综合测试结果
    """
    print("🏗️ 综合架构验证测试")
    print("=" * 50)
    
    comprehensive_results = {
        'test_suite': 'comprehensive_architecture',
        'timestamp': datetime.now().isoformat(),
        'versions_tested': versions,
        'test_config': {
            'seeds': test_seeds,
            'steps': test_steps
        },
        'test_results': {},
        'overall_status': {
            'all_tests_passed': True,
            'total_tests': 0,
            'passed_tests': 0
        }
    }
    
    # 1. 环境创建测试
    creation_result = test_environment_creation(versions)
    comprehensive_results['test_results']['environment_creation'] = creation_result
    comprehensive_results['overall_status']['total_tests'] += 1
    if creation_result['overall_success']:
        comprehensive_results['overall_status']['passed_tests'] += 1
    else:
        comprehensive_results['overall_status']['all_tests_passed'] = False
    
    # 2. 接口兼容性测试
    compatibility_result = test_interface_compatibility(versions)
    comprehensive_results['test_results']['interface_compatibility'] = compatibility_result
    comprehensive_results['overall_status']['total_tests'] += 1
    if compatibility_result['overall_compatible']:
        comprehensive_results['overall_status']['passed_tests'] += 1
    else:
        comprehensive_results['overall_status']['all_tests_passed'] = False
    
    # 3. 数学等价性测试 (仅测试主版本)
    main_version = 'v2'  # 默认主版本
    if main_version in versions:
        math_equiv_result = test_mathematical_equivalence(main_version, test_seeds, test_steps)
        comprehensive_results['test_results']['mathematical_equivalence'] = math_equiv_result
        comprehensive_results['overall_status']['total_tests'] += 1
        if math_equiv_result['mathematically_equivalent']:
            comprehensive_results['overall_status']['passed_tests'] += 1
        else:
            comprehensive_results['overall_status']['all_tests_passed'] = False
    
    # 4. 压力条件测试 (仅测试主版本)
    if main_version in versions:
        stress_result = test_stress_conditions(main_version)
        comprehensive_results['test_results']['stress_conditions'] = stress_result
        comprehensive_results['overall_status']['total_tests'] += 1
        if stress_result['overall_robust']:
            comprehensive_results['overall_status']['passed_tests'] += 1
        else:
            comprehensive_results['overall_status']['all_tests_passed'] = False
    
    # 保存结果
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        print(f"\n📁 详细结果已保存到: {output_file}")
    
    return comprehensive_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='环境架构验证测试',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
测试类型:
  creation        环境创建测试 
  compatibility   接口兼容性测试
  equivalence     数学等价性测试
  stress          压力条件测试
  comprehensive   综合架构测试

使用示例:
  python test_architecture.py creation --versions v1 v2 v3
  python test_architecture.py equivalence --version v2 --steps 20
  python test_architecture.py comprehensive --output results.json
        """
    )
    
    parser.add_argument('test_type',
                       choices=['creation', 'compatibility', 'equivalence', 'stress', 'comprehensive'],
                       help='测试类型')
    
    parser.add_argument('--version', default='v2',
                       choices=['v1', 'v2', 'v3'],
                       help='环境版本, 默认: v2')
    
    parser.add_argument('--versions', nargs='+',
                       choices=['v1', 'v2', 'v3'],
                       help='测试版本列表')
    
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2],
                       help='测试种子列表, 默认: [0,1,2]')
    
    parser.add_argument('--steps', type=int, default=10,
                       help='测试步数, 默认: 10')
    
    parser.add_argument('--output', type=str,
                       help='结果保存文件路径')
    
    args = parser.parse_args()
    
    # 确定测试版本
    if args.versions:
        versions = args.versions
    else:
        versions = [args.version]
    
    print("🚀 NewCppRL 架构验证测试")
    print("=" * 50)
    print(f"测试类型: {args.test_type}")
    print(f"测试版本: {versions}")
    print("=" * 50)
    
    start_time = datetime.now()
    success = True
    
    try:
        if args.test_type == 'creation':
            result = test_environment_creation(versions)
            success = result['overall_success']
            
        elif args.test_type == 'compatibility':
            result = test_interface_compatibility(versions)
            success = result['overall_compatible']
            
        elif args.test_type == 'equivalence':
            result = test_mathematical_equivalence(args.version, args.seeds, args.steps)
            success = result['mathematically_equivalent']
            
        elif args.test_type == 'stress':
            result = test_stress_conditions(args.version)
            success = result['overall_robust']
            
        elif args.test_type == 'comprehensive':
            result = run_comprehensive_architecture_test(versions, args.seeds, args.steps, args.output)
            success = result['overall_status']['all_tests_passed']
            
            # 打印综合测试摘要
            status = result['overall_status']
            print(f"\n📊 综合测试摘要:")
            print(f"   总测试数: {status['total_tests']}")
            print(f"   通过测试: {status['passed_tests']}")
            print(f"   成功率: {status['passed_tests']/status['total_tests']*100:.1f}%")
    
    except Exception as e:
        print(f"❌ 测试执行异常: {str(e)}")
        success = False
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("=" * 50)
    print(f"⏱️ 测试耗时: {duration.total_seconds():.2f}秒")
    
    if success:
        print("🎉 架构验证通过!")
        sys.exit(0)
    else:
        print("❌ 架构验证失败!")
        sys.exit(1)


if __name__ == '__main__':
    main()