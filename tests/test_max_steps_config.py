"""测试max_steps配置是否正确生效"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs_new.cpp_env_v2 import CppEnv
from envs_new.components.config.environment_config import EnvironmentConfig


def test_default_max_steps():
    """测试默认的max_steps配置"""
    print("=" * 50)
    print("测试默认max_steps配置")
    print("=" * 50)
    
    # 创建环境
    env = CppEnv()
    
    # 重置环境
    obs, info = env.reset(seed=42)
    
    # 检查配置值
    config_max_steps = env.config.max_episode_steps
    print(f"配置中的max_episode_steps: {config_max_steps}")
    
    # 检查env_state中的值
    env_state_max_steps = env.env_state.max_steps
    print(f"env_state中的max_steps: {env_state_max_steps}")
    
    # 验证是否一致
    if config_max_steps == env_state_max_steps:
        print("✅ max_steps配置正确传递！")
    else:
        print(f"❌ 错误：配置值({config_max_steps}) != 状态值({env_state_max_steps})")
    
    # 检查是否能通过__getattr__访问
    try:
        accessed_max_steps = env.env_state.max_steps
        print(f"通过__getattr__访问的max_steps: {accessed_max_steps}")
    except AttributeError as e:
        print(f"❌ 无法通过__getattr__访问max_steps: {e}")
    
    env.close()
    return config_max_steps == env_state_max_steps


def test_custom_max_steps():
    """测试自定义的max_steps配置"""
    print("\n" + "=" * 50)
    print("测试自定义max_steps配置")
    print("=" * 50)
    
    # 创建自定义配置
    custom_config = EnvironmentConfig()
    custom_config.max_episode_steps = 10000  # 设置为10000步
    
    # 创建环境（CppEnv的__init__不直接接受config参数）
    # 需要通过kwargs传递配置参数
    env = CppEnv(max_episode_steps=10000)
    
    # 重置环境
    obs, info = env.reset(seed=42)
    
    print(f"自定义配置的max_episode_steps: {env.config.max_episode_steps}")
    print(f"env_state中的max_steps: {env.env_state.max_steps}")
    
    # 验证是否一致
    if env.config.max_episode_steps == env.env_state.max_steps:
        print("✅ 自定义max_steps配置正确传递！")
    else:
        print(f"❌ 错误：配置值({env.config.max_episode_steps}) != 状态值({env.env_state.max_steps})")
    
    env.close()
    return env.config.max_episode_steps == env.env_state.max_steps


def test_timeout_behavior():
    """测试timeout行为是否使用正确的max_steps"""
    print("\n" + "=" * 50)
    print("测试timeout行为")
    print("=" * 50)
    
    # 创建一个很小的max_steps来快速测试
    env = CppEnv(max_episode_steps=100)  # 只允许100步
    obs, info = env.reset(seed=42)
    
    print(f"设置max_episode_steps为: {env.config.max_episode_steps}")
    
    # 运行到超时
    step_count = 0
    timeout_occurred = False
    
    for _ in range(150):  # 运行150步，应该在100步时超时
        # 使用固定的安全动作（直行）来避免碰撞
        if env.config.action_type == "discrete":
            # 选择中间的动作（通常是直行）
            action = env.action_space.n // 2
        else:
            action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        
        if truncated:  # truncated表示超时
            timeout_occurred = True
            print(f"✅ 在第{step_count}步发生超时（预期{env.config.max_episode_steps}步）")
            break
        
        if terminated:  # 其他原因终止
            print(f"在第{step_count}步因其他原因终止")
            break
    
    if not timeout_occurred:
        print(f"❌ 运行了{step_count}步但没有发生超时！")
    
    # 验证超时是否在正确的步数发生
    expected_timeout_step = env.config.max_episode_steps
    if timeout_occurred and step_count == expected_timeout_step:
        print("✅ Timeout在正确的步数发生！")
        result = True
    else:
        print(f"❌ Timeout步数不正确：实际{step_count}，预期{expected_timeout_step}")
        result = False
    
    env.close()
    return result


def test_static_info_storage():
    """测试max_steps是否正确存储在static_info中"""
    print("\n" + "=" * 50)
    print("测试static_info存储")
    print("=" * 50)
    
    env = CppEnv()
    obs, info = env.reset(seed=42)
    
    # 直接检查_static_info
    if 'max_steps' in env.env_state._static_info:
        stored_value = env.env_state._static_info['max_steps']
        print(f"✅ max_steps存储在_static_info中: {stored_value}")
        
        # 验证通过get_static_info访问
        retrieved_value = env.env_state.get_static_info('max_steps')
        print(f"通过get_static_info访问: {retrieved_value}")
        
        result = stored_value == env.config.max_episode_steps
    else:
        print("❌ max_steps未存储在_static_info中！")
        result = False
    
    env.close()
    return result


if __name__ == "__main__":
    print("开始测试max_steps配置")
    print("=" * 70)
    
    all_tests_passed = True
    
    # 运行所有测试
    all_tests_passed &= test_default_max_steps()
    all_tests_passed &= test_custom_max_steps()
    all_tests_passed &= test_timeout_behavior()
    all_tests_passed &= test_static_info_storage()
    
    print("\n" + "=" * 70)
    if all_tests_passed:
        print("🎉 所有测试通过！max_steps配置正确实现。")
    else:
        print("❌ 有测试失败，请检查实现。")
    print("=" * 70)