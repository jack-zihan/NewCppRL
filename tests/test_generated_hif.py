#!/usr/bin/env python3
"""
测试生成的HIF方向场在v5环境中的兼容性
"""

import sys
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_hif_format():
    """测试HIF文件格式"""
    print("=" * 60)
    print("测试HIF文件格式和内容")
    print("=" * 60)
    
    hif_dir = Path("/home/lzh/NewCppRL/envs_new/maps/field_coverage/hif")
    
    # 测试几个HIF文件
    for i in [1, 5, 10]:
        hif_path = hif_dir / f"human_intent_field_{i}.npy"
        if not hif_path.exists():
            print(f"❌ 文件不存在: {hif_path}")
            continue
            
        hif = np.load(hif_path)
        
        print(f"\n📁 文件: human_intent_field_{i}.npy")
        print(f"   形状: {hif.shape}")
        print(f"   数据类型: {hif.dtype}")
        
        # 统计
        valid_mask = hif >= 0
        invalid_mask = hif < 0
        
        print(f"   有效像素: {np.sum(valid_mask):,} ({np.sum(valid_mask)/hif.size*100:.1f}%)")
        print(f"   无效像素: {np.sum(invalid_mask):,} ({np.sum(invalid_mask)/hif.size*100:.1f}%)")
        
        if np.any(valid_mask):
            valid_values = hif[valid_mask]
            print(f"   方向范围: [{np.min(valid_values):.4f}, {np.max(valid_values):.4f}] rad")
            print(f"   方向均值: {np.mean(valid_values):.4f} rad")
            print(f"   方向标准差: {np.std(valid_values):.4f} rad")
            
            # 检查是否在[0, π)范围内
            if np.all(valid_values >= 0) and np.all(valid_values < np.pi + 0.001):
                print(f"   ✅ 方向值在正确范围[0, π)内")
            else:
                print(f"   ❌ 方向值超出范围!")
    
    print("\n" + "=" * 60)


def test_hif_in_environment():
    """测试HIF在v5环境中的加载"""
    print("\n测试HIF在v5环境中的加载")
    print("=" * 60)
    
    try:
        from envs_new.cpp_env_v5 import HIFCreator
        
        # 模拟环境状态
        state = {
            'env_state': type('MockEnvState', (), {
                'get_static_info': lambda self, key: {
                    'dimensions': (400, 400),
                    'map_id': 0
                }[key]
            })(),
            'config': type('MockConfig', (), {
                'get_absolute_map_dir': lambda self: Path('/home/lzh/NewCppRL/envs_new/maps/field_coverage')
            })(),
            'options': {},
            'maps_dict': {}
        }
        
        # 创建HIFCreator并加载
        hif_creator = HIFCreator()
        hif_creator.generate(state, np.random.default_rng(42))
        
        if 'hif' in state['maps_dict']:
            hif = state['maps_dict']['hif']
            print(f"✅ HIF成功加载到环境")
            print(f"   形状: {hif.shape}")
            print(f"   数据类型: {hif.dtype}")
            print(f"   有效像素: {np.sum(hif >= 0):,}")
        else:
            print("❌ HIF加载失败")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
    
    print("=" * 60)


def test_angle_calculation():
    """测试角度差计算"""
    print("\n测试HIF角度差计算")
    print("=" * 60)
    
    try:
        from envs_new.cpp_env_v5 import HIFCalculator
        
        # 测试几个角度组合
        test_cases = [
            (0, 0, "Agent朝东，HIF指向东"),
            (90, np.pi/2, "Agent朝南，HIF指向南"),
            (45, np.pi/4, "Agent朝东南，HIF指向东南"),
            (0, np.pi/2, "Agent朝东，HIF指向南"),
            (180, 0, "Agent朝西，HIF指向东"),
        ]
        
        for agent_deg, hif_rad, desc in test_cases:
            diff = HIFCalculator._compute_angle_difference(agent_deg, hif_rad)
            print(f"   {desc}")
            print(f"     Agent: {agent_deg}°, HIF: {hif_rad:.3f} rad")
            print(f"     角度差: {diff:.1f}°")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
    
    print("=" * 60)


if __name__ == '__main__':
    print("\n🔬 开始测试生成的HIF方向场\n")
    
    test_hif_format()
    test_hif_in_environment()
    test_angle_calculation()
    
    print("\n✅ 测试完成!\n")