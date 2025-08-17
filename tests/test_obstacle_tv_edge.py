#!/usr/bin/env python3
"""测试obstacle TV边缘渲染"""

import numpy as np
import cv2
import sys
import os

sys.path.append('/home/lzh/NewCppRL')
sys.path.append('/home/lzh/NewCppRL/tests')

from envs_new.cpp_env_v2 import CppEnv as NewCppEnv
from envs.cpp_env_v2 import CppEnv as OldCppEnv
from envs_new.utils.math_utils import total_variation_mat
from envs.utils import total_variation_mat as old_total_variation_mat
from tests.utils.environment_state_synchronizer import EnvironmentStateSynchronizer

def test_obstacle_tv_edge():
    print("🔍 测试obstacle TV边缘渲染差异...")
    print("=" * 60)
    
    # 创建保存目录
    save_dir = '/home/lzh/NewCppRL/test_env_consistency/img'
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建环境
    print("创建环境...")
    new_env = NewCppEnv(render_mode='rgb_array')
    old_env = OldCppEnv(render_mode='rgb_array')
    
    # 使用同步器确保状态一致
    print("重置新版环境（seed=42）...")
    new_env.reset(seed=42)
    state_info = EnvironmentStateSynchronizer.extract_new_env_state(new_env)
    
    print("同步状态到旧版环境...")
    old_env.reset(seed=999)
    EnvironmentStateSynchronizer.sync_old_env_state(old_env, state_info)
    
    # 检查obstacle的TV边缘
    print("\n📊 检查obstacle的TV边缘:")
    old_tv = old_total_variation_mat(old_env.map_obstacle)
    new_tv = total_variation_mat(new_env.maps_dict['obstacle'])
    
    print(f"  旧版TV边缘像素数: {np.sum(old_tv)}")
    print(f"  新版TV边缘像素数: {np.sum(new_tv)}")
    print(f"  TV边缘一致性: {np.array_equal(old_tv, new_tv)}")
    
    # 获取渲染图
    print("\n🎨 获取渲染图...")
    old_render = old_env.render_map()
    new_render = new_env.renderer._render_map(
        new_env.maps_dict, new_env.agent, new_env.env_state.dimensions
    )
    
    # 确保数据类型一致
    if old_render.dtype != np.uint8:
        old_render = old_render.astype(np.uint8)
    if new_render.dtype != np.uint8:
        new_render = new_render.astype(np.uint8)
    
    # 检查TV边缘位置的颜色
    tv_positions = np.where(old_tv)
    if len(tv_positions[0]) > 0:
        print("\n🔍 TV边缘颜色采样（前5个位置）:")
        print("  期望颜色（旧版TV边缘）: RGB(47, 82, 143)")
        print("  " + "-" * 50)
        
        num_samples = min(5, len(tv_positions[0]))
        for i in range(num_samples):
            y, x = tv_positions[0][i], tv_positions[1][i]
            old_color = old_render[y, x]
            new_color = new_render[y, x]
            print(f"  位置({y:3d}, {x:3d}): 旧版{old_color} | 新版{new_color}")
            
            # 检查是否为TV边缘颜色
            is_old_tv_color = np.array_equal(old_color, [47, 82, 143])
            is_new_tv_color = np.array_equal(new_color, [47, 82, 143])
            if is_old_tv_color and not is_new_tv_color:
                print(f"    ⚠️ 新版缺少TV边缘渲染！")
    
    # 统计颜色差异
    print("\n📊 颜色统计分析:")
    
    # 统计旧版中TV边缘颜色的像素数
    tv_color = np.array([47, 82, 143])
    old_tv_pixels = np.all(old_render == tv_color, axis=2).sum()
    new_tv_pixels = np.all(new_render == tv_color, axis=2).sum()
    
    print(f"  旧版TV边缘颜色(47,82,143)像素数: {old_tv_pixels}")
    print(f"  新版TV边缘颜色(47,82,143)像素数: {new_tv_pixels}")
    print(f"  差异: {old_tv_pixels - new_tv_pixels} 像素")
    
    # 保存对比图
    print("\n💾 保存对比图...")
    
    # 创建差异图（专门标记TV边缘差异）
    diff_img = np.zeros_like(old_render)
    tv_diff_mask = np.zeros((old_render.shape[0], old_render.shape[1]), dtype=bool)
    
    for y, x in zip(tv_positions[0], tv_positions[1]):
        if not np.array_equal(old_render[y, x], new_render[y, x]):
            tv_diff_mask[y, x] = True
            diff_img[y, x] = [255, 0, 255]  # 紫色标记差异
    
    cv2.imwrite(f'{save_dir}/tv_edge_diff.png', diff_img)
    print(f"  TV边缘差异图保存到: {save_dir}/tv_edge_diff.png")
    
    # 总结
    print("\n" + "=" * 60)
    print("📋 总结:")
    if old_tv_pixels > 0 and new_tv_pixels == 0:
        print("  ❌ 新版完全缺少obstacle的TV边缘渲染！")
        print("  📝 建议：在新版renderer中添加TV边缘渲染逻辑")
    elif old_tv_pixels > new_tv_pixels:
        print("  ⚠️ 新版TV边缘渲染不完整")
        print(f"  📝 缺少 {old_tv_pixels - new_tv_pixels} 个TV边缘像素")
    else:
        print("  ✅ TV边缘渲染一致")

if __name__ == "__main__":
    test_obstacle_tv_edge()