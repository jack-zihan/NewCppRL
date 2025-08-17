#!/usr/bin/env python3
"""
对比新旧版本的基础渲染图（render_map / _render_map）
不包含缩放的原始渲染结果
"""

import numpy as np
import cv2
import sys
import os

sys.path.append('/home/lzh/NewCppRL')

from envs_new.cpp_env_v2 import CppEnv as NewCppEnv
from envs.cpp_env_v2 import CppEnv as OldCppEnv

def compare_base_render_maps():
    print("🔍 对比基础渲染图（未缩放）...")
    print("=" * 60)
    
    # 创建保存目录
    save_dir = '/home/lzh/NewCppRL/test_env_consistency/img'
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建环境
    print("创建环境...")
    new_env = NewCppEnv(render_mode='rgb_array')
    old_env = OldCppEnv(render_mode='rgb_array')
    
    # 使用相同的种子重置
    print("重置环境（seed=42）...")
    new_obs, _ = new_env.reset(seed=42)
    old_obs, _ = old_env.reset(seed=42)
    
    print("\n📊 环境状态:")
    print(f"  新版agent位置: {new_env.agent.position}")
    print(f"  旧版agent位置: {old_env.agent.position}")
    print(f"  新版agent方向: {new_env.agent.direction}")
    print(f"  旧版agent方向: {old_env.agent.direction}")
    
    # ============ 获取基础渲染图 ============
    print("\n🎨 获取基础渲染图（未缩放）...")
    
    # 旧版：直接调用render_map()
    old_base_img = old_env.render_map()
    print(f"  旧版render_map(): {old_base_img.shape}, dtype={old_base_img.dtype}")
    
    # 新版：需要直接调用Renderer的_render_map方法
    # 获取基础渲染（不含缩放）
    new_base_img = new_env.renderer._render_map(
        new_env.maps_dict,
        new_env.agent,
        new_env.env_state.dimensions
    )
    print(f"  新版_render_map(): {new_base_img.shape}, dtype={new_base_img.dtype}")
    
    # 确保数据类型一致
    if old_base_img.dtype != np.uint8:
        old_base_img = old_base_img.astype(np.uint8)
    if new_base_img.dtype != np.uint8:
        new_base_img = new_base_img.astype(np.uint8)
    
    # ============ 对比分析 ============
    print("\n📊 基础渲染图对比:")
    print(f"  形状是否相同: {new_base_img.shape == old_base_img.shape}")
    
    if new_base_img.shape == old_base_img.shape:
        # 计算差异
        diff = np.abs(new_base_img.astype(float) - old_base_img.astype(float))
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)
        similarity = 100 * (1 - mean_diff / 255)
        
        print(f"  平均像素差异: {mean_diff:.2f}")
        print(f"  最大像素差异: {max_diff:.2f}")
        print(f"  相似度: {similarity:.1f}%")
        
        # 找出差异最大的区域
        if max_diff > 10:
            diff_gray = np.mean(diff, axis=2)
            max_pos = np.unravel_index(np.argmax(diff_gray), diff_gray.shape)
            print(f"  最大差异位置: {max_pos}")
            
            # 在该位置周围取一个小区域查看
            y, x = max_pos
            window = 5
            y_start = max(0, y-window)
            y_end = min(old_base_img.shape[0], y+window)
            x_start = max(0, x-window)
            x_end = min(old_base_img.shape[1], x+window)
            
            print(f"\n  差异区域详情 ({y_start}:{y_end}, {x_start}:{x_end}):")
            print(f"    新版像素值: {new_base_img[y, x]}")
            print(f"    旧版像素值: {old_base_img[y, x]}")
        
        # 保存差异图
        diff_img = (diff * 10).clip(0, 255).astype(np.uint8)  # 放大10倍便于观察
        diff_path = os.path.join(save_dir, 'base_render_diff.png')
        cv2.imwrite(diff_path, diff_img)
        print(f"\n  差异图已保存: {diff_path}")
    
    # ============ 保存基础渲染图 ============
    new_base_path = os.path.join(save_dir, 'new_base_render.png')
    old_base_path = os.path.join(save_dir, 'old_base_render.png')
    
    cv2.imwrite(new_base_path, cv2.cvtColor(new_base_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(old_base_path, cv2.cvtColor(old_base_img, cv2.COLOR_RGB2BGR))
    
    print(f"\n📷 基础渲染图已保存:")
    print(f"  新版: {new_base_path}")
    print(f"  旧版: {old_base_path}")
    
    # ============ 测试转置 ============
    if similarity < 95:
        print("\n🔄 测试转置...")
        new_transposed = np.transpose(new_base_img, (1, 0, 2))
        if new_transposed.shape == old_base_img.shape:
            trans_diff = np.mean(np.abs(new_transposed.astype(float) - old_base_img.astype(float)))
            trans_similarity = 100 * (1 - trans_diff / 255)
            print(f"  转置后相似度: {trans_similarity:.1f}%")
            
            if trans_similarity > similarity:
                print(f"  ⚠️ 转置后相似度更高！存在坐标系问题")
                trans_path = os.path.join(save_dir, 'new_base_transposed.png')
                cv2.imwrite(trans_path, cv2.cvtColor(new_transposed, cv2.COLOR_RGB2BGR))
                print(f"  转置图已保存: {trans_path}")
    
    # ============ 执行一步后再对比 ============
    print("\n🚶 执行一步后对比...")
    action = 7
    
    new_env.step(action)
    old_env.step(action)
    
    print(f"  新版位置: {new_env.agent.position}")
    print(f"  旧版位置: {old_env.agent.position}")
    
    # 获取第二帧的基础渲染
    new_base_img2 = new_env.renderer._render_map(
        new_env.maps_dict,
        new_env.agent,
        new_env.env_state.dimensions
    )
    old_base_img2 = old_env.render_map()
    
    if old_base_img2.dtype != np.uint8:
        old_base_img2 = old_base_img2.astype(np.uint8)
    if new_base_img2.dtype != np.uint8:
        new_base_img2 = new_base_img2.astype(np.uint8)
    
    # 保存第二帧
    new_base_path2 = os.path.join(save_dir, 'new_base_render_frame2.png')
    old_base_path2 = os.path.join(save_dir, 'old_base_render_frame2.png')
    
    cv2.imwrite(new_base_path2, cv2.cvtColor(new_base_img2, cv2.COLOR_RGB2BGR))
    cv2.imwrite(old_base_path2, cv2.cvtColor(old_base_img2, cv2.COLOR_RGB2BGR))
    
    if new_base_img2.shape == old_base_img2.shape:
        diff2 = np.mean(np.abs(new_base_img2.astype(float) - old_base_img2.astype(float)))
        similarity2 = 100 * (1 - diff2 / 255)
        print(f"  第二帧相似度: {similarity2:.1f}%")
    
    # ============ 分析具体差异 ============
    print("\n🔬 详细差异分析:")
    
    # 分析各个地图层的差异
    if hasattr(new_env, 'maps_dict') and hasattr(old_env, 'map_weed'):
        print("\n  地图内容对比:")
        
        # 杂草
        new_weed = new_env.maps_dict.get('weed', None)
        old_weed = old_env.map_weed
        if new_weed is not None:
            print(f"    杂草数量 - 新版: {np.sum(new_weed)}, 旧版: {np.sum(old_weed)}")
        
        # 障碍物
        new_obstacle = new_env.maps_dict.get('obstacle', None)
        old_obstacle = old_env.map_obstacle
        if new_obstacle is not None:
            print(f"    障碍物 - 新版: {np.sum(new_obstacle)}, 旧版: {np.sum(old_obstacle)}")
        
        # 轨迹
        new_traj = new_env.maps_dict.get('trajectory', None)
        old_traj = old_env.map_trajectory
        if new_traj is not None:
            print(f"    轨迹 - 新版: {np.sum(new_traj)}, 旧版: {np.sum(old_traj)}")
    
    new_env.close()
    old_env.close()
    
    print("\n" + "=" * 60)
    print("✅ 基础渲染图对比完成！")
    print(f"\n📁 所有图片保存在: {save_dir}")
    print("\n重点查看:")
    print("  - new_base_render.png vs old_base_render.png")
    print("  - base_render_diff.png (差异图)")

if __name__ == "__main__":
    compare_base_render_maps()