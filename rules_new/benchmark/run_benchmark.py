#!/usr/bin/env python3
"""
基准测试运行脚本

使用示例：
    # 使用默认配置运行所有算法
    python run_benchmark.py
    
    # 使用自定义配置文件（推荐）
    python run_benchmark.py --config my_experiment.yaml
    python run_benchmark.py --config ./configs/test_v2.yaml
    
    # 使用配置目录 + 配置名称
    python run_benchmark.py --config-dir ./configs --config-name my_benchmark
    
    # 启用场景完成图片保存
    python run_benchmark.py --save-finished-picture
    
    # 只测试特定算法
    python run_benchmark.py --algorithms JUMP SNAKE BCP
    
    # 禁用并行执行
    python run_benchmark.py --no-parallel
    
    # 组合使用
    python run_benchmark.py --config my_config.yaml --save-finished-picture --algorithms NN_baseline NN_ours
"""

import argparse
import logging
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from rules_new.benchmark import BenchmarkRunner


def setup_logging(level: str = 'INFO'):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('benchmark.log')
        ]
    )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='运行标准化基准测试',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 配置相关参数
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='直接指定配置文件路径（如：my_benchmark.yaml 或 ./configs/experiment.yaml）'
    )
    
    parser.add_argument(
        '--config-dir',
        type=str,
        default=None,
        help='配置目录路径（与--config-name配合使用，或查找默认的benchmark_config.yaml）'
    )
    
    parser.add_argument(
        '--config-name',
        type=str,
        default=None,
        help='配置文件名称（不含.yaml后缀，与--config-dir配合使用）'
    )
    
    parser.add_argument(
        '--save-finished-picture',
        action='store_true',
        help='在每个场景完成时保存渲染图片'
    )
    
    # 算法选择
    parser.add_argument(
        '--algorithms',
        nargs='+',
        choices=['JUMP', 'SNAKE', 'R_SNAKE', 'BCP', 'REACT', 'NN_baseline', 'NN_ours'],
        default=None,
        help='要测试的算法列表（默认测试所有启用的算法）'
    )
    
    # 场景配置
    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        default=None,
        help='测试种子列表（覆盖配置文件）'
    )
    
    parser.add_argument(
        '--difficulties',
        nargs='+',
        choices=['easy', 'medium', 'hard'],
        default=None,
        help='难度级别列表'
    )
    
    # 执行配置
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='禁用并行执行'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=None,
        help='最大并行工作进程数'
    )
    
    # 日志配置
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='日志级别'
    )
    
    # 快速测试模式
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='快速测试模式（只测试少量场景）'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # 创建基准测试运行器
        logger.info("初始化基准测试运行器...")
        runner = BenchmarkRunner(
            config_path=args.config,
            config_dir=args.config_dir,
            config_name=args.config_name,
            save_finished_picture=args.save_finished_picture,
            parallel=not args.no_parallel,
            max_workers=args.max_workers
        )
        
        # 处理快速测试模式
        scenarios = None
        if args.quick_test:
            logger.info("快速测试模式：只测试少量场景")
            # 生成少量测试场景
            quick_scenarios = runner.scenario_generator.generate_all_scenarios(
                seeds=[25],  # 只用一个种子
                difficulties=['easy'],  # 只测试简单难度
                weed_distributions=['uniform'],  # 只用一种分布
                noise_levels=['no_noise']
            )
            scenarios = quick_scenarios[:2]  # 只取前2个场景
        elif args.seeds or args.difficulties:
            # 使用命令行参数生成场景
            seeds = args.seeds or runner.config['benchmark']['scenarios']['seeds']
            difficulties = args.difficulties or runner.config['benchmark']['scenarios']['difficulties']
            
            scenarios = runner.scenario_generator.generate_all_scenarios(
                seeds=seeds,
                difficulties=difficulties,
                weed_distributions=runner.config['benchmark']['scenarios']['weed_distributions'],
                noise_levels=runner.config['benchmark']['scenarios']['noise_levels']
            )
        
        # 运行基准测试
        logger.info("开始运行基准测试...")
        logger.info(f"配置目录: {args.config_dir or '默认'}")
        logger.info(f"保存完成图片: {args.save_finished_picture}")
        logger.info(f"并行执行: {not args.no_parallel}")
        
        if args.algorithms:
            logger.info(f"测试算法: {args.algorithms}")
        else:
            logger.info("测试所有启用的算法")
        
        # 运行测试
        summary = runner.run_benchmark(
            algorithms=args.algorithms,
            scenarios=scenarios
        )
        
        # 输出摘要
        logger.info("=" * 80)
        logger.info("基准测试完成！")
        logger.info("=" * 80)
        
        # 输出排名
        if 'algorithm_rankings' in summary and 'overall' in summary['algorithm_rankings']:
            logger.info("\n算法综合排名：")
            for i, (alg, score) in enumerate(summary['algorithm_rankings']['overall'], 1):
                logger.info(f"  {i}. {alg}: {score:.3f}")
        
        # 输出关键统计
        if 'overall_statistics' in summary:
            logger.info("\n关键性能指标：")
            for alg, stats in summary['overall_statistics'].items():
                logger.info(f"  {alg}:")
                logger.info(f"    - 成功率: {stats['success_rate']:.1%}")
                logger.info(f"    - 碰撞率: {stats['collision_rate']:.1%}")
                logger.info(f"    - 平均覆盖率: {stats['coverage_mean']:.1%}")
        
        logger.info(f"\n详细结果保存在: {runner.output_dir}")
        
        # 清理资源
        runner.cleanup()
        
    except KeyboardInterrupt:
        logger.warning("用户中断测试")
        sys.exit(1)
    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()