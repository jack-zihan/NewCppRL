#!/usr/bin/env python3
"""
统一一致性测试 - 动力学、奖励、观测一致性验证

统一的一致性测试入口，整合原本分散在多个文件中的测试功能。
替代 test_dynamics.py, test_rewards.py, test_observations.py 等文件。
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from tests.utils.consistency_tester import UnifiedConsistencyTester


def run_dynamics_test(version: str, seeds: List[int], steps: int, 
                     tolerance: float, verbose: bool) -> bool:
    """运行动力学一致性测试"""
    print(f"🔄 动力学一致性测试 (版本: {version})")
    
    tester = UnifiedConsistencyTester(tolerance=tolerance, verbose=verbose)
    result = tester.test_dynamics_consistency(version, seeds, steps)
    
    if result['overall_consistent']:
        print("✅ 动力学测试通过!")
        return True
    else:
        print("❌ 动力学测试失败!")
        if 'error_summary' in result:
            summary = result['error_summary']
            print(f"   智能体错误: {summary.get('agent_errors', 0)}")
            print(f"   地图错误: {summary.get('map_errors', 0)}")
            print(f"   环境状态错误: {summary.get('env_state_errors', 0)}")
            print(f"   观测错误: {summary.get('observation_errors', 0)}")
        return False


def run_rewards_test(version: str, seeds: List[int], steps: int,
                    tolerance: float, verbose: bool) -> bool:
    """运行奖励一致性测试"""
    print(f"🎁 奖励一致性测试 (版本: {version})")
    
    tester = UnifiedConsistencyTester(tolerance=tolerance, verbose=verbose)
    result = tester.test_rewards_consistency(version, seeds, steps)
    
    if result['overall_consistent']:
        print("✅ 奖励测试通过!")
        return True
    else:
        print("❌ 奖励测试失败!")
        if 'reward_statistics' in result:
            stats = result['reward_statistics']
            print(f"   总比较次数: {stats['total_comparisons']}")
            print(f"   一致次数: {stats['consistent_comparisons']}")
            print(f"   最大差异: {stats['max_difference']:.8f}")
            print(f"   平均差异: {stats['avg_difference']:.8f}")
        return False


def run_observations_test(version: str, seeds: List[int], steps: int,
                         tolerance: float, verbose: bool) -> bool:
    """运行观测一致性测试"""
    print(f"👁️ 观测一致性测试 (版本: {version})")
    
    tester = UnifiedConsistencyTester(tolerance=tolerance, verbose=verbose)
    result = tester.test_observations_consistency(version, seeds, steps)
    
    if result['overall_consistent']:
        print("✅ 观测测试通过!")
        return True
    else:
        print("❌ 观测测试失败!")
        if 'observation_statistics' in result:
            stats = result['observation_statistics']
            print(f"   总比较次数: {stats['total_comparisons']}")
            print(f"   一致次数: {stats['consistent_comparisons']}")
            if stats['failed_keys']:
                print(f"   失败键值: {list(stats['failed_keys'].keys())}")
        return False


def run_comprehensive_test(version: str, seeds: List[int], steps: int,
                          tolerance: float, verbose: bool, output_file: Optional[str] = None) -> bool:
    """运行综合一致性测试"""
    print(f"🚀 综合一致性测试 (版本: {version})")
    
    tester = UnifiedConsistencyTester(tolerance=tolerance, verbose=verbose)
    result = tester.test_all_consistency(version, seeds, steps)
    
    # 保存结果
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"📁 结果已保存到: {output_file}")
    
    return result['overall_consistent']


def run_batch_test(versions: List[str], seeds: List[int], steps: int,
                  tolerance: float, test_type: str, verbose: bool) -> bool:
    """运行批量测试"""
    print(f"📦 批量{test_type}测试")
    print(f"   版本: {versions}")
    print(f"   种子: {seeds}")
    print(f"   步数: {steps}")
    
    all_passed = True
    results = {}
    
    for version in versions:
        print(f"\n🧪 测试版本 {version}")
        
        if test_type == 'dynamics':
            passed = run_dynamics_test(version, seeds, steps, tolerance, False)
        elif test_type == 'rewards':
            passed = run_rewards_test(version, seeds, steps, tolerance, False)
        elif test_type == 'observations':
            passed = run_observations_test(version, seeds, steps, tolerance, False)
        elif test_type == 'all':
            passed = run_comprehensive_test(version, seeds, steps, tolerance, False)
        else:
            print(f"❌ 未知测试类型: {test_type}")
            passed = False
        
        results[version] = passed
        if not passed:
            all_passed = False
    
    # 打印批量测试摘要
    print(f"\n📊 批量测试摘要:")
    for version, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {version}: {'通过' if passed else '失败'}")
    
    return all_passed


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='统一一致性测试 - 新旧环境完全等价性验证',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
测试类型:
  dynamics      动力学一致性测试
  rewards       奖励一致性测试  
  observations  观测一致性测试
  all          综合一致性测试

使用示例:
  python test_consistency.py dynamics --version v2
  python test_consistency.py rewards --seeds 0 1 2 --steps 20
  python test_consistency.py all --batch --versions v1 v2 v3
  python test_consistency.py observations --tolerance 1e-8 --output results.json
        """
    )
    
    parser.add_argument('test_type', 
                       choices=['dynamics', 'rewards', 'observations', 'all'],
                       help='测试类型')
    
    parser.add_argument('--version', default='v2',
                       choices=['v1', 'v2', 'v3'],
                       help='环境版本, 默认: v2')
    
    parser.add_argument('--versions', nargs='+', 
                       choices=['v1', 'v2', 'v3'],
                       help='批量测试版本列表')
    
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4],
                       help='测试种子列表, 默认: [0,1,2,3,4]')
    
    parser.add_argument('--steps', type=int, default=10,
                       help='每种子测试步数, 默认: 10')
    
    parser.add_argument('--tolerance', type=float, default=1e-12,
                       help='数值比较容差, 默认: 1e-12')
    
    parser.add_argument('--batch', action='store_true',
                       help='批量测试模式')
    
    parser.add_argument('--verbose', action='store_true',
                       help='详细输出模式')
    
    parser.add_argument('--output', type=str,
                       help='结果保存文件路径')
    
    parser.add_argument('--quick', action='store_true',
                       help='快速模式 (1个种子, 5步)')
    
    args = parser.parse_args()
    
    # 快速模式参数调整
    if args.quick:
        args.seeds = [0]
        args.steps = 5
        args.tolerance = 1e-6
        print("⚡ 快速测试模式")
    
    print("🚀 NewCppRL 统一一致性测试")
    print("=" * 50)
    print(f"测试类型: {args.test_type}")
    print(f"容差: {args.tolerance}")
    print(f"种子: {args.seeds}")
    print(f"步数: {args.steps}")
    print("=" * 50)
    
    # 确定测试版本
    if args.batch and args.versions:
        versions = args.versions
    elif args.batch:
        versions = ['v1', 'v2', 'v3']  # 默认所有版本
    else:
        versions = [args.version]
    
    # 执行测试
    start_time = datetime.now()
    
    if args.batch:
        success = run_batch_test(versions, args.seeds, args.steps, 
                               args.tolerance, args.test_type, args.verbose)
    else:
        version = args.version
        
        if args.test_type == 'dynamics':
            success = run_dynamics_test(version, args.seeds, args.steps, 
                                      args.tolerance, args.verbose)
        elif args.test_type == 'rewards':
            success = run_rewards_test(version, args.seeds, args.steps,
                                     args.tolerance, args.verbose)
        elif args.test_type == 'observations':
            success = run_observations_test(version, args.seeds, args.steps,
                                          args.tolerance, args.verbose)
        elif args.test_type == 'all':
            success = run_comprehensive_test(version, args.seeds, args.steps,
                                           args.tolerance, args.verbose, args.output)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("=" * 50)
    print(f"⏱️ 测试耗时: {duration.total_seconds():.2f}秒")
    
    if success:
        print("🎉 所有测试通过!")
        sys.exit(0)
    else:
        print("❌ 部分测试失败!")
        sys.exit(1)


if __name__ == '__main__':
    main()