#!/usr/bin/env python3
"""
保存新旧环境的第一帧渲染图片进行对比
"""

import numpy as np
import cv2
import sys
import os

sys.path.append('/home/lzh/NewCppRL')

from envs_new.cpp_env_v2 import CppEnv as NewCppEnv
from envs.cpp_env_v2 import CppEnv as OldCppEnv

def save_first_frames():
    print("🔍 保存新旧环境第一帧渲染图片...")
    print("=" * 60)
    
    # 创建保存目录
    save_dir = '/home/lzh/NewCppRL/test_env_consistency/img'
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建环境
    print("创建环境...")
    new_env = NewCppEnv(render_mode='rgb_array')
    old_env = OldCppEnv(render_mode='rgb_array')
    
    # 使用相同的种子重置，确保初始状态一致
    print("重置环境（seed=42）...")
    new_obs, _ = new_env.reset(seed=42)
    old_obs, _ = old_env.reset(seed=42)
    
    # 输出基本信息
    print("\n📊 环境信息:")
    print(f"  新版agent位置: {new_env.agent.position}")
    print(f"  旧版agent位置: {old_env.agent.position}")
    print(f"  新版agent方向: {new_env.agent.direction}")
    print(f"  旧版agent方向: {old_env.agent.direction}")
    
    # 检查render_repeat_times
    if hasattr(new_env, 'config'):
        print(f"  新版render_repeat_times: {new_env.config.render_repeat_times}")
    if hasattr(old_env, 'render_repeat_times'):
        print(f"  旧版render_repeat_times: {old_env.render_repeat_times}")
    
    # 渲染第一帧
    print("\n🎨 渲染第一帧...")
    new_img = new_env.render()
    
    # 旧版使用render_map方法
    if hasattr(old_env, 'render_map'):
        old_img = old_env.render_map()
    else:
        old_img = old_env.render()
    
    # 确保数据类型正确
    if old_img.dtype == np.float64 or old_img.dtype == np.float32:
        old_img = (old_img * 1).astype(np.uint8)  # 如果是浮点数，转换为uint8
    
    print(f"\n🖼️ 渲染图像信息:")
    print(f"  新版形状: {new_img.shape}, 类型: {new_img.dtype}")
    print(f"  旧版形状: {old_img.shape}, 类型: {old_img.dtype}")
    
    # 保存原始图片
    new_path = os.path.join(save_dir, 'new_first_frame.png')
    old_path = os.path.join(save_dir, 'old_first_frame.png')
    
    cv2.imwrite(new_path, cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(old_path, cv2.cvtColor(old_img, cv2.COLOR_RGB2BGR))
    
    print(f"\n📷 原始图片已保存:")
    print(f"  新版: {new_path}")
    print(f"  旧版: {old_path}")
    
    # 如果尺寸相同，直接对比
    if new_img.shape == old_img.shape:
        print("\n✅ 图像尺寸相同，进行直接对比...")
        diff = np.abs(new_img.astype(float) - old_img.astype(float))
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)
        similarity = 100 * (1 - mean_diff / 255)
        
        print(f"  平均差异: {mean_diff:.2f}")
        print(f"  最大差异: {max_diff:.2f}")
        print(f"  相似度: {similarity:.1f}%")
        
        # 保存差异图
        diff_img = (diff * 10).clip(0, 255).astype(np.uint8)  # 放大差异10倍便于观察
        diff_path = os.path.join(save_dir, 'difference_map.png')
        cv2.imwrite(diff_path, diff_img)
        print(f"  差异图: {diff_path}")
        
        # 如果相似度低，测试转置
        if similarity < 95:
            print("\n🔄 相似度较低，测试转置...")
            new_transposed = np.transpose(new_img, (1, 0, 2))
            if new_transposed.shape == old_img.shape:
                trans_diff = np.mean(np.abs(new_transposed.astype(float) - old_img.astype(float)))
                trans_similarity = 100 * (1 - trans_diff / 255)
                print(f"  转置后相似度: {trans_similarity:.1f}%")
                
                if trans_similarity > similarity:
                    print(f"  ⚠️ 转置后相似度更高！可能存在坐标系问题")
                    trans_path = os.path.join(save_dir, 'new_transposed.png')
                    cv2.imwrite(trans_path, cv2.cvtColor(new_transposed, cv2.COLOR_RGB2BGR))
                    print(f"  转置图: {trans_path}")
    else:
        print(f"\n⚠️ 图像尺寸不同!")
        print(f"  新版: {new_img.shape}")
        print(f"  旧版: {old_img.shape}")
        
        # 尝试缩放对比
        if new_img.shape[0] == 2 * old_img.shape[0]:
            print("\n  新版是2倍大小，尝试缩小对比...")
            new_downscaled = new_img[::2, ::2]
            downscaled_path = os.path.join(save_dir, 'new_downscaled.png')
            cv2.imwrite(downscaled_path, cv2.cvtColor(new_downscaled, cv2.COLOR_RGB2BGR))
            print(f"  缩小图: {downscaled_path}")
            
            if new_downscaled.shape == old_img.shape:
                diff = np.mean(np.abs(new_downscaled.astype(float) - old_img.astype(float)))
                similarity = 100 * (1 - diff / 255)
                print(f"  缩小后相似度: {similarity:.1f}%")
        elif old_img.shape[0] == 2 * new_img.shape[0]:
            print("\n  旧版是2倍大小，尝试缩小对比...")
            old_downscaled = old_img[::2, ::2]
            downscaled_path = os.path.join(save_dir, 'old_downscaled.png')
            cv2.imwrite(downscaled_path, cv2.cvtColor(old_downscaled, cv2.COLOR_RGB2BGR))
            print(f"  缩小图: {downscaled_path}")
            
            if old_downscaled.shape == new_img.shape:
                diff = np.mean(np.abs(old_downscaled.astype(float) - new_img.astype(float)))
                similarity = 100 * (1 - diff / 255)
                print(f"  缩小后相似度: {similarity:.1f}%")
    
    # 分析地图内容
    print("\n🗺️ 地图内容分析:")
    
    # 新版地图
    if hasattr(new_env, 'maps_dict'):
        print("  新版地图:")
        for map_name in ['field', 'field_frontier', 'weed', 'obstacle', 'trajectory']:
            if map_name in new_env.maps_dict:
                map_data = new_env.maps_dict[map_name]
                print(f"    {map_name}: shape={map_data.shape}, 非零={np.sum(map_data > 0)}")
    
    # 旧版地图
    print("  旧版地图:")
    for map_name in ['field', 'field_frontier', 'weed', 'obstacle', 'trajectory']:
        attr_name = f'map_{map_name}'
        if hasattr(old_env, attr_name):
            map_data = getattr(old_env, attr_name)
            print(f"    {attr_name}: shape={map_data.shape}, 非零={np.sum(map_data > 0)}")
    
    # 执行一步并保存
    print("\n🚶 执行一步后对比...")
    action = 7  # 使用相同的动作
    
    new_obs2, _, _, _, _ = new_env.step(action)
    old_obs2, _, _, _, _ = old_env.step(action)
    
    print(f"  执行动作: {action}")
    print(f"  新版位置: {new_env.agent.position}")
    print(f"  旧版位置: {old_env.agent.position}")
    
    # 渲染第二帧
    new_img2 = new_env.render()
    old_img2 = old_env.render_map() if hasattr(old_env, 'render_map') else old_env.render()
    
    if old_img2.dtype == np.float64 or old_img2.dtype == np.float32:
        old_img2 = (old_img2 * 1).astype(np.uint8)
    
    new_path2 = os.path.join(save_dir, 'new_second_frame.png')
    old_path2 = os.path.join(save_dir, 'old_second_frame.png')
    
    cv2.imwrite(new_path2, cv2.cvtColor(new_img2, cv2.COLOR_RGB2BGR))
    cv2.imwrite(old_path2, cv2.cvtColor(old_img2, cv2.COLOR_RGB2BGR))
    
    print(f"\n📷 第二帧已保存:")
    print(f"  新版: {new_path2}")
    print(f"  旧版: {old_path2}")
    
    new_env.close()
    old_env.close()
    
    print("\n" + "=" * 60)
    print("✅ 分析完成！")
    print(f"📁 所有图片保存在: {save_dir}")
    print("\n请查看以下文件:")
    print("  - new_first_frame.png vs old_first_frame.png (第一帧对比)")
    print("  - new_second_frame.png vs old_second_frame.png (第二帧对比)")
    print("  - difference_map.png (差异图，如果存在)")

if __name__ == "__main__":
    save_first_frames()