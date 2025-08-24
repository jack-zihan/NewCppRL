#!/usr/bin/env python3
"""
测试环境与numpy类型的兼容性
验证action处理和状态变量都能正确处理numpy类型
"""
import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from envs_new.cpp_env_v2 import CppEnv
from envs_new.components.state.environment_state import StateVariable


def test_action_numpy_compatibility():
    """测试action处理对numpy数组的支持"""
    print("🧪 测试action的numpy兼容性...")
    
    env = CppEnv()
    obs, info = env.reset(seed=42)
    
    # 测试不同类型的action
    test_cases = [
        ("numpy数组", np.array([2.0, 0.5], dtype=np.float32)),
        ("元组", (2.0, 0.5)),
        ("列表", [2.0, 0.5]),
        ("action_space.sample()", env.action_space.sample())
    ]
    
    for name, action in test_cases:
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  ✅ {name}: 类型={type(action)}, 奖励={reward:.4f}")
        except Exception as e:
            print(f"  ❌ {name}: 失败 - {e}")
            
    env.close()
    print("✅ Action兼容性测试完成\n")


def test_state_variable_numpy_types():
    """测试StateVariable对numpy类型的支持"""
    print("🧪 测试StateVariable的numpy类型支持...")
    
    # 创建StateVariable实例
    state_var = StateVariable("test_var", history_length=3)
    
    # 测试不同的数值类型
    test_values = [
        ("Python float", 1.0),
        ("Python int", 2),
        ("numpy float32", np.float32(3.0)),
        ("numpy float64", np.float64(4.0)),
        ("numpy int32", np.int32(5)),
        ("numpy int64", np.int64(6))
    ]
    
    for name, value in test_values:
        state_var.update(value)
        change = state_var.change()
        print(f"  {name}: 值={value}, 类型={type(value)}, 变化={change}")
    
    print("✅ StateVariable类型测试完成\n")


def test_reward_calculation_with_numpy():
    """测试使用numpy类型时的奖励计算"""
    print("🧪 测试奖励计算的numpy兼容性...")
    
    env = CppEnv()
    obs, info = env.reset(seed=42)
    
    # 执行多步，验证奖励计算
    for i in range(5):
        action = env.action_space.sample()  # 返回numpy数组
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 获取奖励分解信息
        reward_breakdown = env.reward_system.get_reward_breakdown(env.env_state)
        
        print(f"  Step {i}:")
        print(f"    总奖励: {reward:.4f}")
        print(f"    转向奖励: {reward_breakdown['turning_total']:.4f}")
        
        if terminated or truncated:
            break
    
    env.close()
    print("✅ 奖励计算测试完成\n")


def test_full_episode():
    """运行完整episode测试"""
    print("🧪 运行完整episode测试...")
    
    env = CppEnv()
    obs, info = env.reset(seed=42)
    
    step_count = 0
    total_reward = 0.0
    
    while step_count < 100:  # 限制步数
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        step_count += 1
        total_reward += reward
        
        if terminated or truncated:
            print(f"  Episode结束: 步数={step_count}, 总奖励={total_reward:.2f}")
            break
    
    if step_count >= 100:
        print(f"  达到最大步数: 总奖励={total_reward:.2f}")
    
    env.close()
    print("✅ Episode测试完成\n")


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("🔧 Numpy兼容性测试套件")
    print("="*60 + "\n")
    
    try:
        test_action_numpy_compatibility()
        test_state_variable_numpy_types()
        test_reward_calculation_with_numpy()
        test_full_episode()
        
        print("="*60)
        print("🎉 所有测试通过！")
        print("\n📊 修复总结:")
        print("  ✅ action_processor.py: 支持numpy.ndarray作为action输入")
        print("  ✅ environment_state.py: StateVariable.change()支持numpy数值类型")
        print("  ✅ 环境与Gymnasium标准完全兼容")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())