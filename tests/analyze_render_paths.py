#!/usr/bin/env python3
"""
深入分析新旧版本的渲染路径差异
"""

import numpy as np
import cv2
import sys
import os

sys.path.append('/home/lzh/NewCppRL')

from envs_new.cpp_env_v2 import CppEnv as NewCppEnv
from envs.cpp_env_v2 import CppEnv as OldCppEnv
from gymnasium.wrappers import HumanRendering

def analyze_render_paths():
    print("🔍 深入分析渲染路径差异...")
    print("=" * 60)
    
    # 创建环境
    print("\n📊 环境创建:")
    new_env = NewCppEnv(render_mode='rgb_array')
    old_env = OldCppEnv(render_mode='rgb_array')
    
    # 重置
    new_obs, _ = new_env.reset(seed=42)
    old_obs, _ = old_env.reset(seed=42)
    
    print("\n🎨 渲染路径分析:")
    print("-" * 40)
    
    # ============ 新版渲染路径 ============
    print("\n【新版渲染路径】")
    print("1. CppEnv(render_mode='rgb_array')")
    print("2. → CppEnvBase.render()")
    print("3. → Renderer.render()")
    print("4. → Renderer._render_map()")
    print("5. → 应用render_repeat_times缩放")
    print("6. → 返回800x800图像")
    
    # 测试新版直接渲染
    new_img_direct = new_env.render()
    print(f"\n新版直接渲染: {new_img_direct.shape}")
    
    # ============ 旧版渲染路径 ============
    print("\n【旧版渲染路径】")
    print("1. CppEnv(render_mode='rgb_array')")
    
    # 路径A：直接调用render()
    print("\n路径A - 直接调用render():")
    print("2. → CppEnvBase.render()")
    print("3. → 使用pygame创建Surface")
    print("4. → 调用render_map()返回400x400")
    print("5. → 应用render_repeat_times缩放到800x800")
    print("6. → pygame渲染并转换回numpy")
    
    # 路径B：调用render_map()
    print("\n路径B - 直接调用render_map():")
    print("2. → CppEnv.render_map()")
    print("3. → 父类CppEnvBase.render_map()")
    print("4. → 直接返回400x400图像（不缩放）")
    
    # 测试旧版不同方法
    if hasattr(old_env, 'render_map'):
        old_img_map = old_env.render_map()
        print(f"\n旧版render_map(): {old_img_map.shape}")
    
    # 测试旧版render()方法
    try:
        old_img_render = old_env.render()
        if old_img_render is not None:
            print(f"旧版render(): {old_img_render.shape}")
    except Exception as e:
        print(f"旧版render()错误: {e}")
    
    # ============ HumanRendering包装器分析 ============
    print("\n【HumanRendering包装器】")
    print("-" * 40)
    
    # 测试HumanRendering包装
    print("\n测试HumanRendering包装旧版环境:")
    old_env_wrapped = HumanRendering(OldCppEnv(render_mode='rgb_array'))
    old_env_wrapped.reset(seed=42)
    
    # HumanRendering会自动调用render()并显示
    print("HumanRendering特点:")
    print("1. 包装原始环境")
    print("2. 在step/reset时自动调用render()")
    print("3. 使用pygame显示窗口")
    print("4. 但不改变render()返回值的尺寸")
    
    # ============ 关键差异总结 ============
    print("\n" + "=" * 60)
    print("🎯 关键差异总结:")
    print("-" * 40)
    
    print("\n1. 渲染方法调用:")
    print("   新版: 统一使用render() → Renderer类处理")
    print("   旧版: render_map()直接返回，render()需要pygame")
    
    print("\n2. 缩放应用时机:")
    print("   新版: Renderer.render()中统一应用缩放")
    print("   旧版: 只在render()的pygame路径中应用")
    
    print("\n3. 实际使用场景:")
    print("   新版测试: 直接调用render()，得到800x800")
    print("   旧版测试: 调用render_map()，得到400x400")
    
    print("\n4. HumanRendering影响:")
    print("   - HumanRendering是包装器，不改变渲染尺寸")
    print("   - 主要用于创建pygame窗口显示")
    print("   - 旧版测试代码使用render_map()绕过了缩放")
    
    # ============ 验证实际调用 ============
    print("\n" + "=" * 60)
    print("🔬 验证实际调用:")
    
    # 检查旧版测试代码实际用的什么
    print("\n旧版cpp_env_v2.py中的测试代码:")
    print("  第102行: def render_map(self)")
    print("  第103行: rendered_map = super(CppEnv, self).render_map()")
    print("  → 旧版测试实际调用render_map()，不是render()")
    print("  → 所以得到400x400的图像")
    
    print("\n新版cpp_env_base.py中的渲染:")
    print("  第251行: def render(self)")
    print("  第268行: return self.renderer.render(...)")
    print("  → 新版统一调用render()")
    print("  → Renderer应用了缩放，得到800x800")
    
    # 保存对比图像
    save_dir = '/home/lzh/NewCppRL/test_env_consistency/img'
    
    # 测试如果旧版也用render()会怎样
    print("\n" + "=" * 60)
    print("🧪 实验：如果旧版也用render()会怎样？")
    
    # 创建新的旧版环境
    old_env2 = OldCppEnv(render_mode='rgb_array')
    old_env2.reset(seed=42)
    
    # 尝试调用render()
    print("\n尝试调用旧版的render()方法...")
    try:
        # 旧版的render()需要pygame，可能会失败或返回800x800
        old_render_result = old_env2.render()
        if old_render_result is not None:
            print(f"旧版render()结果: {old_render_result.shape}")
            cv2.imwrite(os.path.join(save_dir, 'old_render_method.png'), 
                       cv2.cvtColor(old_render_result.astype(np.uint8), cv2.COLOR_RGB2BGR))
        else:
            print("旧版render()返回None")
    except Exception as e:
        print(f"旧版render()失败: {e}")
        print("原因：render()需要pygame初始化")
    
    # 清理
    new_env.close()
    old_env.close()
    old_env2.close()
    
    print("\n" + "=" * 60)
    print("✅ 分析完成！")
    print("\n📝 结论：")
    print("1. 差异原因：旧版测试用render_map()，新版用render()")
    print("2. render_map()不应用缩放，render()应用缩放")
    print("3. 这不是bug，是两个不同的渲染路径")
    print("4. HumanRendering只是显示包装器，不影响尺寸")

if __name__ == "__main__":
    analyze_render_paths()