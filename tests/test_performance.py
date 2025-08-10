#!/usr/bin/env python3
"""
性能基准测试 - 环境执行性能评估

优化的性能测试工具，用于评估不同环境版本的执行效率。
"""

import sys
import time
import argparse
import statistics
from pathlib import Path
from typing import List, Dict, Any

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from tests.utils.environment_manager import EnvironmentManager


def benchmark_environment_version(version: str, num_steps: int = 100, 
                                 num_episodes: int = 3, render: bool = False) -> Dict[str, Any]:
    """
    性能基准测试单个环境版本
    
    Args:
        version: 环境版本
        num_steps: 每轮的步数
        num_episodes: 测试轮数
        render: 是否渲染
        
    Returns:
        性能测试结果
    """
    print(f"⚡ 性能测试版本 {version}")
    
    manager = EnvironmentManager(verbose=False)
    
    try:
        OldEnv, NewEnv = manager.get_environment_classes(version)
        
        results = {
            'version': version,
            'num_steps': num_steps,
            'num_episodes': num_episodes,
            'old_env_performance': {},
            'new_env_performance': {},
            'comparison': {}
        }
        
        # 测试旧环境性能
        print(f"  测试旧环境...")
        old_times = []
        
        for episode in range(num_episodes):
            old_env = OldEnv(render_mode='rgb_array' if render else None)
            obs, _ = old_env.reset(seed=episode)
            
            episode_times = []
            for step in range(num_steps):
                action = old_env.action_space.sample()
                
                start_time = time.time()
                obs, reward, done, truncated, _ = old_env.step(action)
                end_time = time.time()
                
                if step > 0:  # 跳过第一步的冷启动时间
                    episode_times.append(end_time - start_time)
                
                if done or truncated:
                    obs, _ = old_env.reset(seed=episode + 1000)
            
            old_times.extend(episode_times)
            old_env.close()
        
        # 测试新环境性能
        print(f"  测试新环境...")
        new_times = []
        
        for episode in range(num_episodes):
            new_env = NewEnv()
            obs, _ = new_env.reset(seed=episode)
            
            episode_times = []
            for step in range(num_steps):
                action = new_env.action_space.sample()
                
                start_time = time.time()
                obs, reward, done, truncated, _ = new_env.step(action)
                end_time = time.time()
                
                if step > 0:  # 跳过第一步的冷启动时间
                    episode_times.append(end_time - start_time)
                
                if done or truncated:
                    obs, _ = new_env.reset(seed=episode + 1000)
            
            new_times.extend(episode_times)
            new_env.close()
        
        # 计算统计数据
        results['old_env_performance'] = {
            'total_steps': len(old_times),
            'avg_time_ms': statistics.mean(old_times) * 1000,
            'median_time_ms': statistics.median(old_times) * 1000,
            'min_time_ms': min(old_times) * 1000,
            'max_time_ms': max(old_times) * 1000,
            'std_dev_ms': statistics.stdev(old_times) * 1000 if len(old_times) > 1 else 0
        }
        
        results['new_env_performance'] = {
            'total_steps': len(new_times),
            'avg_time_ms': statistics.mean(new_times) * 1000,
            'median_time_ms': statistics.median(new_times) * 1000,
            'min_time_ms': min(new_times) * 1000,
            'max_time_ms': max(new_times) * 1000,
            'std_dev_ms': statistics.stdev(new_times) * 1000 if len(new_times) > 1 else 0
        }
        
        # 性能比较
        old_avg = results['old_env_performance']['avg_time_ms']
        new_avg = results['new_env_performance']['avg_time_ms']
        
        results['comparison'] = {
            'speedup_ratio': old_avg / new_avg if new_avg > 0 else float('inf'),
            'time_difference_ms': new_avg - old_avg,
            'performance_improvement_percent': ((old_avg - new_avg) / old_avg * 100) if old_avg > 0 else 0
        }
        
        # 打印结果
        print(f"    旧环境: {old_avg:.2f}ms ± {results['old_env_performance']['std_dev_ms']:.2f}ms")
        print(f"    新环境: {new_avg:.2f}ms ± {results['new_env_performance']['std_dev_ms']:.2f}ms")
        
        if results['comparison']['performance_improvement_percent'] > 0:
            print(f"    ✅ 性能提升: {results['comparison']['performance_improvement_percent']:.1f}%")
        elif results['comparison']['performance_improvement_percent'] < -5:  # 容忍5%的性能下降
            print(f"    ⚠️ 性能下降: {abs(results['comparison']['performance_improvement_percent']):.1f}%")
        else:
            print(f"    ✅ 性能相当")
        
        return results
        
    except Exception as e:
        print(f"  ❌ 性能测试异常: {str(e)}")
        return {
            'version': version,
            'error': str(e),
            'success': False
        }


def run_comparative_benchmark(versions: List[str], num_steps: int = 100) -> Dict[str, Any]:
    """
    运行比较性能基准测试
    
    Args:
        versions: 要测试的版本列表
        num_steps: 每个版本的测试步数
        
    Returns:
        比较测试结果
    """
    print("📊 比较性能基准测试")
    
    benchmark_results = {
        'test_type': 'comparative_benchmark',
        'versions_tested': versions,
        'num_steps': num_steps,
        'version_results': {},
        'performance_ranking': []
    }
    
    for version in versions:
        result = benchmark_environment_version(version, num_steps)
        benchmark_results['version_results'][version] = result
    
    # 性能排名 (基于新环境性能)
    valid_results = []
    for version, result in benchmark_results['version_results'].items():
        if 'error' not in result and 'new_env_performance' in result:
            avg_time = result['new_env_performance']['avg_time_ms']
            valid_results.append((version, avg_time))
    
    # 按平均时间排序 (越小越好)
    valid_results.sort(key=lambda x: x[1])
    benchmark_results['performance_ranking'] = valid_results
    
    # 打印排名
    print(f"\n🏆 性能排名 (新环境):")
    for i, (version, avg_time) in enumerate(valid_results, 1):
        print(f"  {i}. {version}: {avg_time:.2f}ms")
    
    return benchmark_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='环境性能基准测试',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python test_performance.py                    # 默认测试v2版本
  python test_performance.py --version v1      # 测试特定版本
  python test_performance.py --comparative     # 比较所有版本
  python test_performance.py --steps 200       # 自定义测试步数
        """
    )
    
    parser.add_argument('--version', default='v2',
                       choices=['v1', 'v2', 'v3'],
                       help='环境版本, 默认: v2')
    
    parser.add_argument('--versions', nargs='+',
                       choices=['v1', 'v2', 'v3'],
                       help='批量测试版本列表')
    
    parser.add_argument('--comparative', action='store_true',
                       help='比较所有版本性能')
    
    parser.add_argument('--steps', type=int, default=100,
                       help='测试步数, 默认: 100')
    
    parser.add_argument('--episodes', type=int, default=3,
                       help='测试轮数, 默认: 3')
    
    parser.add_argument('--render', action='store_true',
                       help='启用渲染 (会显著影响性能)')
    
    args = parser.parse_args()
    
    print("🚀 NewCppRL 性能基准测试")
    print("=" * 40)
    
    if args.comparative:
        # 比较所有版本
        versions = ['v1', 'v2', 'v3']
        run_comparative_benchmark(versions, args.steps)
        
    elif args.versions:
        # 批量测试指定版本
        for version in args.versions:
            benchmark_environment_version(version, args.steps, args.episodes, args.render)
            print()
    
    else:
        # 单版本测试
        result = benchmark_environment_version(args.version, args.steps, args.episodes, args.render)
        
        if 'error' not in result:
            print(f"\n📈 详细统计 ({args.version}):")
            old_perf = result['old_env_performance']
            new_perf = result['new_env_performance']
            
            print(f"旧环境:")
            print(f"  平均: {old_perf['avg_time_ms']:.2f}ms")
            print(f"  中位数: {old_perf['median_time_ms']:.2f}ms")
            print(f"  范围: {old_perf['min_time_ms']:.2f}ms - {old_perf['max_time_ms']:.2f}ms")
            
            print(f"新环境:")
            print(f"  平均: {new_perf['avg_time_ms']:.2f}ms")
            print(f"  中位数: {new_perf['median_time_ms']:.2f}ms")
            print(f"  范围: {new_perf['min_time_ms']:.2f}ms - {new_perf['max_time_ms']:.2f}ms")
    
    print("=" * 40)
    print("🎉 性能测试完成!")


if __name__ == '__main__':
    main()
