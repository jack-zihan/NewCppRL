#!/usr/bin/env python3
"""
观测和渲染系统测试运行脚本

快速运行所有观测和渲染系统的一致性测试。
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from tests.test_observation_rendering import (
    ObservationRenderingTester,
    compare_environments
)
from envs.cpp_env_v2 import CppEnv as CppEnvV2
from envs.cpp_env_v3 import CppEnv as CppEnvV3


def main():
    """主函数"""
    print("=" * 80)
    print("🔬 观测和渲染系统一致性测试")
    print("=" * 80)
    
    # 测试配置
    test_configs = [
        {
            'name': '基础配置',
            'config': {
                'map_size': (256, 256),
                'vision_length': 28,
                'vision_angle': 75,
                'state_size': (256, 256),
                'state_downsize': (128, 128),
                'use_apf': True,
                'use_traj': False,
                'render_mode': 'rgb_array',
            }
        },
        {
            'name': 'SGCNN配置',
            'config': {
                'map_size': (256, 256),
                'vision_length': 28,
                'vision_angle': 75,
                'state_size': (256, 256),
                'state_downsize': (128, 128),
                'sgcnn_size': 16,
                'use_sgcnn': True,
                'use_global_obs': False,
                'use_apf': True,
                'render_mode': 'rgb_array',
            }
        },
        {
            'name': 'SGCNN+全局观测',
            'config': {
                'map_size': (256, 256),
                'vision_length': 28,
                'vision_angle': 75,
                'state_size': (256, 256),
                'state_downsize': (128, 128),
                'sgcnn_size': 16,
                'use_sgcnn': True,
                'use_global_obs': True,
                'use_apf': True,
                'render_mode': 'rgb_array',
            }
        }
    ]
    
    # 运行测试
    for test_config in test_configs:
        print(f"\n📋 测试配置: {test_config['name']}")
        print("-" * 40)
        
        # 测试V2版本
        print(f"\n🔹 测试CppEnvV2...")
        tester_v2 = ObservationRenderingTester(CppEnvV2, test_config['config'])
        tester_v2.run_all_tests()
        
        # 生成V2报告
        report_path_v2 = f"/home/lzh/NewCppRL/test_env_consistency/reports/obs_render_v2_{test_config['name'].replace(' ', '_')}.md"
        tester_v2.generate_report(report_path_v2)
        print(f"✅ V2报告已保存: {report_path_v2}")
        
        # 测试V3版本
        print(f"\n🔹 测试CppEnvV3...")
        tester_v3 = ObservationRenderingTester(CppEnvV3, test_config['config'])
        tester_v3.run_all_tests()
        
        # 生成V3报告
        report_path_v3 = f"/home/lzh/NewCppRL/test_env_consistency/reports/obs_render_v3_{test_config['name'].replace(' ', '_')}.md"
        tester_v3.generate_report(report_path_v3)
        print(f"✅ V3报告已保存: {report_path_v3}")
    
    # 运行版本比较
    print("\n" + "=" * 80)
    print("🔀 运行环境版本比较...")
    print("=" * 80)
    
    comparison = compare_environments(
        test_configs[0]['config'],  # 使用基础配置进行比较
        test_configs[0]['config']
    )
    
    print("\n" + "=" * 80)
    print("✅ 所有测试完成！")
    print("=" * 80)
    print("\n📁 测试报告保存位置:")
    print("   /home/lzh/NewCppRL/test_env_consistency/reports/")
    print("\n📊 主要报告文件:")
    print("   - dataflow_validation_report.md (数据流验证报告)")
    print("   - environment_comparison.md (版本比较报告)")
    print("   - obs_render_v2_*.md (V2版本测试报告)")
    print("   - obs_render_v3_*.md (V3版本测试报告)")


if __name__ == "__main__":
    main()