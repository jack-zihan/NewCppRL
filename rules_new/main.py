#!/usr/bin/env python3
"""
Rules New1 主入口 - 优雅的实验管理系统
替代原有script.py的硬编码和注释切换方式
"""
import argparse
import sys
from pathlib import Path
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rules_new.experiment import ExperimentRunner, BatchManager, ConfigManager
from rules_new.utils.logging_utils import LoggingUtils
from rules_new.utils.path_utils import PathUtils


def setup_logging(verbose: bool = False):
    """设置日志级别"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def run_single_experiment(args):
    """运行单个实验"""
    print(f"🚀 运行实验: {args.config}")
    
    try:
        runner = ExperimentRunner(args.config)
        result = runner.run_experiment()
        
        print(f"✅ 实验完成!")
        print(f"   - 总实验数: {result['total_experiments']}")
        print(f"   - 成功率: {result['success_rate']:.1%}")
        print(f"   - 摘要文件: {result['summary_file']}")
        
        runner.cleanup()
        
    except Exception as e:
        print(f"❌ 实验失败: {e}")
        sys.exit(1)


def run_batch_experiments(args):
    """运行批量实验"""
    print(f"🔄 批量运行实验")
    
    try:
        manager = BatchManager(max_workers=args.workers)
        
        if args.configs:
            # 手动指定的配置列表
            manager.add_multiple_experiments(args.configs)
        else:
            # 自动发现所有实验配置
            discovered = manager.discover_experiments()
            manager.add_multiple_experiments(discovered)
        
        # 显示队列状态
        status = manager.get_queue_status()
        print(f"   - 队列中的实验: {status['queued_experiments']}")
        print(f"   - 最大并行数: {status['max_workers']}")
        
        # 执行实验
        if args.parallel:
            result = manager.run_parallel()
        else:
            result = manager.run_sequential()
        
        print(f"✅ 批量实验完成!")
        print(f"   - 执行模式: {result['execution_mode']}")
        print(f"   - 总实验数: {result['total_experiments']}")
        print(f"   - 成功率: {result['success_rate']:.1%}")
        print(f"   - 总耗时: {result['total_runtime_seconds']:.2f} 秒")
        
    except Exception as e:
        print(f"❌ 批量实验失败: {e}")
        sys.exit(1)


def list_available_configs(args):
    """列出可用的配置文件"""
    try:
        config_manager = ConfigManager()
        
        # 列出算法配置
        algorithms_dir = PathUtils.get_project_root() / "rules_new" / "configs" / "algorithms"
        if algorithms_dir.exists():
            print("📋 可用的算法配置:")
            for config_file in algorithms_dir.glob("*.yaml"):
                print(f"   - {config_file.stem}")
        
        # 列出实验配置
        experiments_dir = PathUtils.get_project_root() / "rules_new" / "configs" / "experiments"
        if experiments_dir.exists():
            print("\n🧪 可用的实验配置:")
            for config_file in experiments_dir.glob("*.yaml"):
                print(f"   - {config_file.stem}")
                
                # 尝试加载配置显示详细信息
                try:
                    config = config_manager.load_experiment_config(config_file.stem)
                    exp_info = config.get('experiment', {})
                    algorithms = config.get('algorithms', [])
                    enabled_algs = [alg['name'] for alg in algorithms if alg.get('enabled', True)]
                    
                    print(f"     描述: {exp_info.get('description', '无')}")
                    print(f"     算法: {', '.join(enabled_algs)}")
                    
                except Exception:
                    print(f"     (配置文件加载失败)")
                
                print()
        
    except Exception as e:
        print(f"❌ 列出配置失败: {e}")
        sys.exit(1)


def validate_config(args):
    """验证配置文件"""
    print(f"🔍 验证配置: {args.config}")
    
    try:
        config_manager = ConfigManager()
        
        if args.config.startswith('experiments/'):
            config = config_manager.load_experiment_config(args.config.replace('experiments/', ''))
            print("✅ 实验配置验证通过")
            
            # 显示配置摘要
            exp_info = config.get('experiment', {})
            print(f"   - 实验名称: {exp_info.get('name', '未知')}")
            print(f"   - 描述: {exp_info.get('description', '无')}")
            
            algorithms = config.get('algorithms', [])
            enabled_algs = [alg['name'] for alg in algorithms if alg.get('enabled', True)]
            print(f"   - 启用的算法: {', '.join(enabled_algs)}")
            
        elif args.config.startswith('algorithms/'):
            config = config_manager.load_algorithm_config(args.config.replace('algorithms/', ''))
            print("✅ 算法配置验证通过")
            
            alg_info = config.get('algorithm', {})
            print(f"   - 算法名称: {alg_info.get('name', '未知')}")
            print(f"   - 类型: {alg_info.get('type', '未知')}")
            
        else:
            config = config_manager.load_config(args.config)
            print("✅ 配置验证通过")
        
    except Exception as e:
        print(f"❌ 配置验证失败: {e}")
        sys.exit(1)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Rules New1 - 优雅的实验管理系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行单个实验
  python main.py run baseline_comparison
  
  # 运行批量实验（顺序执行）
  python main.py batch
  
  # 运行批量实验（并行执行）
  python main.py batch --parallel --workers 4
  
  # 列出可用配置
  python main.py list
  
  # 验证配置文件
  python main.py validate experiments/baseline_comparison
        """
    )
    
    parser.add_argument('-v', '--verbose', action='store_true', help='详细输出')
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 运行单个实验
    run_parser = subparsers.add_parser('run', help='运行单个实验')
    run_parser.add_argument('config', help='实验配置名称（不包含.yaml后缀）')
    
    # 批量运行实验
    batch_parser = subparsers.add_parser('batch', help='批量运行实验')
    batch_parser.add_argument('--configs', nargs='+', help='指定要运行的实验配置列表')
    batch_parser.add_argument('--parallel', action='store_true', help='并行执行（默认顺序执行）')
    batch_parser.add_argument('--workers', type=int, default=4, help='并行工作线程数（默认4）')
    
    # 列出配置
    list_parser = subparsers.add_parser('list', help='列出可用的配置文件')
    
    # 验证配置
    validate_parser = subparsers.add_parser('validate', help='验证配置文件')
    validate_parser.add_argument('config', help='配置文件路径（相对于configs目录）')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.verbose)
    
    # 显示欢迎信息
    print("=" * 60)
    print("🎯 Rules New1 - 优雅的实验管理系统")
    print("   替代原有的注释切换式实验执行方式")
    print("   采用配置驱动和模块化设计")
    print("=" * 60)
    
    # 执行对应命令
    if args.command == 'run':
        run_single_experiment(args)
    elif args.command == 'batch':
        run_batch_experiments(args)
    elif args.command == 'list':
        list_available_configs(args)
    elif args.command == 'validate':
        validate_config(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()