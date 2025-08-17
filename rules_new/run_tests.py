#!/usr/bin/env python3
"""
路径规划算法测试主入口脚本

简单直接的命令行入口，让用户能够方便地运行测试。
设计原则：保持简单，避免过度工程化。
"""

import argparse
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from tester import PathPlannerTester


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='运行路径规划算法测试',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置运行测试
  python run_tests.py
  
  # 使用自定义配置
  python run_tests.py --config configs/my_config.yaml
  
  # 快速测试（仅运行部分场景）
  python run_tests.py --quick-test
  
  # 并行执行
  python run_tests.py --workers 4
  
  # 仅测试特定算法
  python run_tests.py --algorithms JUMP SNAKE
"""
    )
    
    # 基本参数
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/simple_test_config.yaml',
        help='配置文件路径 (默认: configs/simple_test_config.yaml)'
    )
    
    # 测试选项
    parser.add_argument(
        '--quick-test', '-q',
        action='store_true',
        help='快速测试模式（仅运行少量场景）'
    )
    
    parser.add_argument(
        '--algorithms', '-a',
        nargs='+',
        choices=['JUMP', 'SNAKE', 'R-SNAKE', 'BCP', 'REACT'],
        help='指定要测试的算法'
    )
    
    # 性能选项
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='并行工作进程数 (默认: 4)'
    )
    
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='禁用并行执行'
    )
    
    # 输出选项
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='保存可视化图表'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出模式'
    )
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"错误：配置文件不存在: {config_path}")
        sys.exit(1)
    
    try:
        # 创建测试器
        print(f"📋 加载配置: {config_path}")
        tester = PathPlannerTester(str(config_path))
        
        # 应用命令行参数
        if args.quick_test:
            print("⚡ 快速测试模式")
            # 修改配置以减少测试场景
            tester.scenario_builder.seeds = tester.scenario_builder.seeds[:1]
            tester.scenario_builder.difficulties = ['easy']
            tester.scenario_builder.map_sizes = tester.scenario_builder.map_sizes[:1]
        
        if args.algorithms:
            print(f"🤖 测试算法: {', '.join(args.algorithms)}")
            # 仅保留指定的算法
            tester.algorithms = {
                name: algo for name, algo in tester.algorithms.items()
                if name in args.algorithms
            }
        
        if args.no_parallel:
            tester.config['parallel'] = False
        else:
            tester.config['max_workers'] = args.workers
        
        tester.config['output']['generate_plots'] = args.save_plots
        
        # 运行测试
        print(f"\n🚀 开始测试 ({len(tester.algorithms)} 个算法)")
        print("=" * 50)
        
        results = tester.run_tests()
        
        # 输出结果摘要
        print("\n" + "=" * 50)
        print("📊 测试完成！")
        print(f"📁 结果保存到: {results['output_dir']}")
        
        if 'summary' in results:
            print("\n📈 性能摘要:")
            for algo_name, metrics in results['summary'].items():
                if isinstance(metrics, dict):
                    avg_coverage = metrics.get('avg_coverage', 0)
                    avg_path_length = metrics.get('avg_path_length', 0)
                    print(f"  {algo_name:10s}: 覆盖率={avg_coverage:.2%}, 路径长度={avg_path_length:.1f}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
        return 1
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())