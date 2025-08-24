#!/usr/bin/env python3
"""
最终综合测试：修复参数名称后的完整一致性测试
"""

import sys
import numpy as np
sys.path.append('/home/lzh/NewCppRL')

from envs.cpp_env_v2 import CppEnv as OldCppEnvV2
from envs_new.cpp_env_v2 import CppEnv as NewCppEnvV2


def test_reward_consistency_fixed():
    """测试修复参数后的奖励一致性"""
    print("🎯 最终奖励一致性测试（修复参数名称）")
    print("="*80)
    
    # 创建环境
    old_env = OldCppEnvV2(render_mode=None)
    new_env = NewCppEnvV2(render_mode=None)
    
    seed = 42
    weed_num = 100
    
    # 使用正确的参数名称
    old_options = {'weed_num': weed_num}
    new_options = {'weed_count': weed_num}  # 新环境使用weed_count
    
    # 重置环境
    old_obs, old_info = old_env.reset(seed=seed, options=old_options)
    new_obs, new_info = new_env.reset(seed=seed, options=new_options)
    
    # 验证杂草数量
    print(f"\n初始化验证:")
    print(f"  旧环境杂草数: {old_env.weed_num}")
    print(f"  新环境杂草数: {new_env.env_state.weed_count}")
    
    if old_env.weed_num != new_env.env_state.weed_count:
        print("  ❌ 杂草数量不一致！")
        return False
    else:
        print("  ✅ 杂草数量一致")
    
    # 设置相同的动作种子
    old_env.action_space.seed(seed)
    new_env.action_space.seed(seed)
    
    # 运行测试
    print("\n运行Episode测试:")
    print(f"{'Step':<6} {'旧版本':>10} {'新版本':>10} {'差异':>10} {'状态':<10}")
    print("-"*50)
    
    all_match = True
    for step in range(50):
        action = old_env.action_space.sample()
        
        # 执行动作
        old_obs, old_reward, old_done, old_truncated, old_info = old_env.step(action)
        new_obs, new_reward, new_done, new_truncated, new_info = new_env.step(action)
        
        # 计算差异
        diff = float(new_reward) - float(old_reward)
        
        # 显示前10步和有差异的步
        if step < 10 or abs(diff) > 1e-6:
            status = "✅" if abs(diff) < 1e-6 else "❌"
            print(f"{step+1:<6} {old_reward:>10.4f} {new_reward:>10.4f} {diff:>10.6f} {status}")
            
            if abs(diff) > 1e-6:
                all_match = False
        
        # 检查终止条件
        if old_done != new_done:
            print(f"\n⚠️ 终止状态不一致在第{step+1}步!")
            print(f"   旧: done={old_done}, crashed={old_info.get('crashed')}, finished={old_info.get('finished')}")
            print(f"   新: done={new_done}, crashed={new_info.get('crashed')}, finished={new_info.get('finished')}")
            all_match = False
            break
        
        if old_done:
            print(f"\n✓ Episode在第{step+1}步结束")
            break
    
    old_env.close()
    new_env.close()
    
    return all_match


def test_base_and_apf_rewards():
    """单独测试基础环境和APF奖励"""
    print("\n\n🔬 基础环境(base)和APF奖励测试")
    print("="*80)
    
    # 测试cpp_env_base
    print("\n1. 测试cpp_env_base（无APF）:")
    from envs.cpp_env_base_copy import CppEnvBase as OldBase
    from envs_new.cpp_env_base import CppEnvBase as NewBase
    
    old_base = OldBase(render_mode=None)
    new_base = NewBase(render_mode=None)
    
    seed = 42
    old_obs, _ = old_base.reset(seed=seed, options={'weed_num': 100})
    new_obs, _ = new_base.reset(seed=seed, options={'weed_count': 100})
    
    old_base.action_space.seed(seed)
    new_base.action_space.seed(seed)
    
    base_match = True
    for i in range(10):
        action = old_base.action_space.sample()
        _, old_r, _, _, _ = old_base.step(action)
        _, new_r, _, _, _ = new_base.step(action)
        
        diff = float(new_r) - float(old_r)
        if abs(diff) > 1e-6:
            print(f"  Step {i+1}: 差异 = {diff:.6f} ❌")
            base_match = False
        else:
            if i == 0:  # 只打印第一步作为示例
                print(f"  Step {i+1}: 差异 = {diff:.6f} ✅")
    
    if base_match:
        print("  ✅ 基础环境奖励完全一致")
    else:
        print("  ❌ 基础环境存在差异")
    
    old_base.close()
    new_base.close()
    
    return base_match


def test_without_apf():
    """测试禁用APF的情况"""
    print("\n\n🔧 测试禁用APF的cpp_env_v2")
    print("="*80)
    
    old_env = OldCppEnvV2(render_mode=None, use_apf=False)
    new_env = NewCppEnvV2(render_mode=None, use_apf=False)
    
    seed = 42
    old_obs, _ = old_env.reset(seed=seed, options={'weed_num': 100})
    new_obs, _ = new_env.reset(seed=seed, options={'weed_count': 100})
    
    old_env.action_space.seed(seed)
    new_env.action_space.seed(seed)
    
    no_apf_match = True
    for i in range(10):
        action = old_env.action_space.sample()
        _, old_r, _, _, _ = old_env.step(action)
        _, new_r, _, _, _ = new_env.step(action)
        
        diff = float(new_r) - float(old_r)
        if abs(diff) > 1e-6:
            print(f"  Step {i+1}: 差异 = {diff:.6f} ❌")
            no_apf_match = False
        else:
            if i == 0:
                print(f"  Step {i+1}: 差异 = {diff:.6f} ✅")
    
    if no_apf_match:
        print("  ✅ 禁用APF后奖励一致")
    else:
        print("  ❌ 禁用APF后仍有差异")
    
    old_env.close()
    new_env.close()
    
    return no_apf_match


if __name__ == "__main__":
    print("="*80)
    print("🏁 最终综合一致性测试")
    print("="*80)
    
    # 测试1：修复参数后的完整测试
    test1_pass = test_reward_consistency_fixed()
    
    # 测试2：基础环境和APF分离测试
    test2_pass = test_base_and_apf_rewards()
    
    # 测试3：禁用APF测试
    test3_pass = test_without_apf()
    
    # 总结
    print("\n" + "="*80)
    print("📊 测试总结:")
    print(f"  1. 完整环境测试: {'✅ 通过' if test1_pass else '❌ 失败'}")
    print(f"  2. 基础环境测试: {'✅ 通过' if test2_pass else '❌ 失败'}")
    print(f"  3. 禁用APF测试: {'✅ 通过' if test3_pass else '❌ 失败'}")
    
    if all([test1_pass, test2_pass, test3_pass]):
        print("\n🎉 所有测试通过！奖励系统完全一致！")
    else:
        print("\n⚠️ 仍存在问题需要解决")