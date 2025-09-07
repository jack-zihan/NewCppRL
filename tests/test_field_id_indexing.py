#!/usr/bin/env python3
"""
测试field_id索引修复是否正确工作
测试weed_coverage数据集中不连续编号的field文件能否正确加载对应的HIF文件
"""

import sys
import numpy as np
from pathlib import Path
sys.path.append('/home/lzh/NewCppRL')

from envs_new.cpp_env_v5 import CppEnv


def test_field_id_indexing():
    """测试field_id索引是否正确匹配HIF文件"""
    
    print("测试field_id索引修复...")
    
    # 配置v5环境使用weed_coverage数据集
    # 创建环境
    env = CppEnv(map_dir='/home/lzh/NewCppRL/envs_new/maps/weed_coverage')
    
    # 测试多次reset，确保field_id正确传递和使用
    test_cases = []
    
    for i in range(5):
        print(f"\n测试 {i+1}:")
        
        # Reset环境
        obs, info = env.reset()
        
        # 获取field_id
        field_id = env.env_state.get_static_info('field_id')
        print(f"  field_id: {field_id}")
        
        # 检查是否成功加载了HIF
        if 'hif' in env.maps_dict:
            hif_map = env.maps_dict['hif']
            print(f"  HIF shape: {hif_map.shape}")
            print(f"  HIF dtype: {hif_map.dtype}")
            
            # 验证HIF文件路径是否正确
            expected_hif_path = Path('/home/lzh/NewCppRL/envs_new/maps/weed_coverage') / 'hif' / f'human_intent_field_{field_id}.npy'
            print(f"  期望的HIF文件: {expected_hif_path}")
            print(f"  文件存在: {expected_hif_path.exists()}")
            
            if expected_hif_path.exists():
                # 直接加载文件验证是否与环境中的一致
                expected_hif = np.load(str(expected_hif_path))
                if np.array_equal(expected_hif, hif_map):
                    print("  ✅ HIF文件正确匹配!")
                else:
                    print("  ❌ HIF内容不匹配!")
            
            test_cases.append({
                'field_id': field_id,
                'hif_loaded': True,
                'file_exists': expected_hif_path.exists()
            })
        else:
            print("  ❌ HIF未加载到环境中")
            test_cases.append({
                'field_id': field_id,
                'hif_loaded': False,
                'file_exists': False
            })
    
    # 总结测试结果
    print("\n" + "="*50)
    print("测试总结:")
    success_count = sum(1 for tc in test_cases if tc['hif_loaded'] and tc['file_exists'])
    print(f"成功加载HIF: {success_count}/{len(test_cases)}")
    
    # 显示所有测试的field_id
    field_ids = [tc['field_id'] for tc in test_cases]
    print(f"测试的field_id列表: {field_ids}")
    
    if success_count == len(test_cases):
        print("\n✅ 所有测试通过! field_id索引修复成功!")
    else:
        print("\n⚠️ 部分测试失败，请检查HIF文件是否已生成")
    
    return success_count == len(test_cases)


if __name__ == "__main__":
    # 首先检查weed_coverage的HIF文件是否存在
    hif_dir = Path('/home/lzh/NewCppRL/envs_new/maps/weed_coverage/hif')
    if not hif_dir.exists() or not list(hif_dir.glob('*.npy')):
        print("⚠️ 警告: weed_coverage的HIF文件尚未生成")
        print("请先运行: python utils/generate_shape_based_hif.py /home/lzh/NewCppRL/envs_new/maps/weed_coverage")
        print("")
    
    # 运行测试
    success = test_field_id_indexing()
    
    if not success:
        print("\n提示: 如果HIF文件不存在，请先生成它们:")
        print("python utils/generate_shape_based_hif.py /home/lzh/NewCppRL/envs_new/maps/weed_coverage")