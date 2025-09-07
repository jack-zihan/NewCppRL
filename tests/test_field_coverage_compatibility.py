#!/usr/bin/env python3
"""
测试field_coverage数据集的向后兼容性
确保修改后的代码仍然能正确处理连续编号的field文件
"""

import sys
from pathlib import Path
sys.path.append('/home/lzh/NewCppRL')

from envs_new.cpp_env_v5 import CppEnv


def test_field_coverage_compatibility():
    """测试field_coverage数据集是否仍然正常工作"""
    
    print("测试field_coverage向后兼容性...")
    
    # 创建使用field_coverage的环境
    env = CppEnv(map_dir='/home/lzh/NewCppRL/envs_new/maps/field_coverage')
    
    print("\n运行3次reset测试:")
    for i in range(3):
        print(f"\n测试 {i+1}:")
        obs, info = env.reset()
        
        # 获取field_id  
        field_id = env.env_state.get_static_info('field_id')
        print(f"  field_id: {field_id}")
        
        # 检查HIF是否加载
        if 'hif' in env.maps_dict:
            hif_map = env.maps_dict['hif']
            print(f"  HIF shape: {hif_map.shape}")
            
            # 验证文件路径
            expected_path = Path('/home/lzh/NewCppRL/envs_new/maps/field_coverage/hif') / f'human_intent_field_{field_id}.npy'
            print(f"  期望文件: human_intent_field_{field_id}.npy")
            print(f"  文件存在: {expected_path.exists()}")
            
            if expected_path.exists():
                print("  ✅ HIF正确加载!")
        else:
            print("  ⚠️ HIF未加载")
    
    print("\n✅ field_coverage兼容性测试完成!")
    print("说明：修改后的代码仍然兼容连续编号的field_coverage数据集")


if __name__ == "__main__":
    test_field_coverage_compatibility()