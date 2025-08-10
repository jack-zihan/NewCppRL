#!/usr/bin/env python3
"""
快速测试入口 - 环境基础功能快速验证

提供最简洁的测试接口，用于快速验证环境是否正常工作。
替代复杂的多文件测试结构。
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from tests.utils.environment_manager import EnvironmentManager
from tests.utils.consistency_tester import UnifiedConsistencyTester


def quick_version_test(version: str = 'v2', steps: int = 5) -> bool:
    """
    快速版本测试
    
    Args:
        version: 环境版本
        steps: 测试步数
        
    Returns:
        测试是否通过
    """
    print(f"⚡ 快速测试版本 {version}")
    
    manager = EnvironmentManager(verbose=False)
    
    try:
        # 基础环境创建测试
        with manager.create_environment_pair(version, seed=0) as (old_env, new_env):
            print("  ✅ 环境创建成功")
            
            # 验证同步
            sync_result = manager.verify_environment_sync(old_env, new_env)
            if sync_result['synchronized']:
                print("  ✅ 环境同步正常")
            else:
                print("  ❌ 环境同步失败")
                return False
            
            # 快速步骤测试
            for step in range(steps):
                action = old_env.action_space.sample()
                
                old_result = old_env.step(action)
                new_result = new_env.step(action)
                
                step_comparison = manager.compare_environment_step_results(old_result, new_result)
                
                if not step_comparison['consistent']:
                    print(f"  ❌ 步骤 {step} 不一致")
                    return False
                
                if old_result[2] or old_result[3]:  # done or truncated
                    break
            
            print(f"  ✅ {steps}步测试通过")
            return True
            
    except Exception as e:
        print(f"  ❌ 测试失败: {str(e)}")
        return False


def quick_consistency_test(version: str = 'v2', test_type: str = 'all') -> bool:
    """
    快速一致性测试
    
    Args:
        version: 环境版本
        test_type: 测试类型 ('dynamics', 'rewards', 'observations', 'all')
        
    Returns:
        测试是否通过
    """
    print(f"🔍 快速一致性测试 - {test_type}")
    
    tester = UnifiedConsistencyTester(tolerance=1e-6, verbose=False)
    
    try:
        if test_type == 'dynamics':
            result = tester.test_dynamics_consistency(version, seeds=[0], num_steps=3)
            success = result['overall_consistent']
        elif test_type == 'rewards':
            result = tester.test_rewards_consistency(version, seeds=[0], num_steps=3)
            success = result['overall_consistent']
        elif test_type == 'observations':
            result = tester.test_observations_consistency(version, seeds=[0], num_steps=3)
            success = result['overall_consistent']
        elif test_type == 'all':
            result = tester.test_all_consistency(version, seeds=[0], num_steps=3)
            success = result['overall_consistent']
        else:
            print(f"  ❌ 未知测试类型: {test_type}")
            return False
        
        if success:
            print(f"  ✅ {test_type}一致性测试通过")
            return True
        else:
            print(f"  ❌ {test_type}一致性测试失败")
            return False
            
    except Exception as e:
        print(f"  ❌ 测试异常: {str(e)}")
        return False


def quick_compatibility_test(versions: List[str] = None) -> bool:
    """
    快速兼容性测试
    
    Args:
        versions: 要测试的版本列表
        
    Returns:
        测试是否通过
    """
    if versions is None:
        versions = ['v2']  # 默认只测试v2
    
    print(f"🔧 快速兼容性测试")
    
    manager = EnvironmentManager(verbose=False)
    
    try:
        result = manager.run_environment_compatibility_test(versions, num_steps=3)
        
        if result['overall_compatible']:
            print(f"  ✅ 兼容性测试通过")
            return True
        else:
            print(f"  ❌ 兼容性测试失败")
            # 打印简要错误信息
            for version, version_result in result['version_results'].items():
                if not version_result['compatible']:
                    print(f"    - {version}: {version_result['errors'][0] if version_result['errors'] else 'Unknown error'}")
            return False
            
    except Exception as e:
        print(f"  ❌ 测试异常: {str(e)}")
        return False


def main():
    """快速测试主函数"""
    parser = argparse.ArgumentParser(
        description='环境快速验证测试',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python test_quick.py                    # 默认快速测试
  python test_quick.py v2                 # 测试特定版本
  python test_quick.py --consistency all  # 一致性测试
  python test_quick.py --compatibility    # 兼容性测试
  python test_quick.py --full            # 完整快速测试
        """
    )
    
    parser.add_argument('version', nargs='?', default='v2',
                       help='环境版本 (v1, v2, v3), 默认: v2')
    parser.add_argument('--consistency', choices=['dynamics', 'rewards', 'observations', 'all'],
                       help='运行一致性测试')
    parser.add_argument('--compatibility', action='store_true',
                       help='运行兼容性测试')
    parser.add_argument('--full', action='store_true',
                       help='运行完整快速测试套件')
    parser.add_argument('--steps', type=int, default=5,
                       help='测试步数, 默认: 5')
    
    args = parser.parse_args()
    
    print("🚀 NewCppRL 环境快速测试")
    print("=" * 40)
    
    all_passed = True
    
    if args.full:
        # 完整快速测试
        print("运行完整快速测试套件...")
        
        # 基础功能测试
        if not quick_version_test(args.version, args.steps):
            all_passed = False
        
        # 一致性测试
        if not quick_consistency_test(args.version, 'all'):
            all_passed = False
        
        # 兼容性测试
        if not quick_compatibility_test([args.version]):
            all_passed = False
    
    elif args.consistency:
        # 一致性测试
        if not quick_consistency_test(args.version, args.consistency):
            all_passed = False
    
    elif args.compatibility:
        # 兼容性测试
        if not quick_compatibility_test([args.version]):
            all_passed = False
    
    else:
        # 默认基础功能测试
        if not quick_version_test(args.version, args.steps):
            all_passed = False
    
    print("=" * 40)
    if all_passed:
        print("🎉 所有快速测试通过!")
        sys.exit(0)
    else:
        print("❌ 部分测试失败!")
        sys.exit(1)


if __name__ == '__main__':
    main()