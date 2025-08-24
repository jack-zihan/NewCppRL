#!/usr/bin/env python3
"""
验证frontier→field重命名后的功能正确性
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from envs_new.cpp_env_base import CppEnvBase
from envs_new.cpp_env_v1 import CppEnv as CppEnvV1
from envs_new.cpp_env_v2 import CppEnv as CppEnvV2
from envs_new.cpp_env_v3 import CppEnv as CppEnvV3
from envs_new.components.config.environment_config import EnvironmentConfig
from envs_new.components.state.environment_state import EnvironmentState
from envs_new.components.reward.reward_system import RewardSystem


def test_config_field_attributes():
    """测试配置中的field相关属性"""
    print("\n1️⃣ 测试配置属性...")
    config = EnvironmentConfig()
    
    # 验证field相关属性存在
    assert hasattr(config, 'reward_field_coverage'), "Missing reward_field_coverage"
    assert hasattr(config, 'reward_field_variation'), "Missing reward_field_variation"
    assert hasattr(config, 'reward_field_group_coef'), "Missing reward_field_group_coef"
    
    # 验证默认值
    assert config.reward_field_coverage == 1.0
    assert config.reward_field_variation == 0.5
    assert config.reward_field_group_coef == 0.125
    
    print("   ✅ 配置属性正确")


def test_reward_system_field_calculators():
    """测试奖励系统中的field计算器"""
    print("\n2️⃣ 测试奖励系统...")
    config = EnvironmentConfig()
    reward_system = RewardSystem(config)
    
    # 验证field计算器存在
    assert 'field_coverage' in reward_system.AVAILABLE_CALCULATORS, "Missing field_coverage calculator"
    assert 'field_variation' in reward_system.AVAILABLE_CALCULATORS, "Missing field_variation calculator"
    
    # 验证组定义
    assert 'field' in reward_system.REWARD_GROUPS, "Missing field reward group"
    assert reward_system.REWARD_GROUPS['field'] == 'reward_field_group_coef'
    assert reward_system.REWARD_GROUPS['turning'] == 'reward_turning_group_coef'
    
    print("   ✅ 奖励系统正确")


def test_environment_state_field_tracking():
    """测试环境状态中的field跟踪"""
    print("\n3️⃣ 测试环境状态...")
    env_state = EnvironmentState()
    
    # 添加field相关状态
    field_area = env_state.add_state_info('field_area', history_length=2)
    field_variation = env_state.add_state_info('field_variation', history_length=2)
    
    # 更新和验证
    field_area.update(1000)
    field_variation.update(50)
    
    assert env_state.field_area == 1000
    assert env_state.field_variation == 50
    
    print("   ✅ 环境状态正确")


def test_environment_creation_and_reset():
    """测试环境创建和重置"""
    print("\n4️⃣ 测试环境创建...")
    
    # 测试基础环境
    env_base = CppEnvBase()
    obs, info = env_base.reset(seed=42)
    assert 'field' in env_base.maps_dict, "Missing field map in base environment"
    print("   ✅ 基础环境正确")
    
    # 测试V1环境
    env_v1 = CppEnvV1()
    obs, info = env_v1.reset(seed=42)
    assert 'field' in env_v1.maps_dict, "Missing field map in V1 environment"
    print("   ✅ V1环境正确")
    
    # 测试V2环境
    env_v2 = CppEnvV2()
    obs, info = env_v2.reset(seed=42)
    assert 'field' in env_v2.maps_dict, "Missing field map in V2 environment"
    print("   ✅ V2环境正确")
    
    # 测试V3环境
    env_v3 = CppEnvV3()
    obs, info = env_v3.reset(seed=42)
    assert 'field' in env_v3.maps_dict, "Missing field map in V3 environment"
    print("   ✅ V3环境正确")
    
    env_base.close()
    env_v1.close()
    env_v2.close()
    env_v3.close()


def test_environment_step_execution():
    """测试环境step执行"""
    print("\n5️⃣ 测试环境步进...")
    
    env = CppEnvV2()
    obs, info = env.reset(seed=42)
    
    # 执行几步
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 验证field_area在环境状态中被跟踪
        assert hasattr(env.env_state, 'field_area'), "field_area not tracked"
        assert hasattr(env.env_state, 'field_variation'), "field_variation not tracked"
        
        if terminated or truncated:
            break
    
    env.close()
    print("   ✅ 环境步进正确")


def test_map_generation():
    """测试地图生成"""
    print("\n6️⃣ 测试地图生成...")
    
    from envs_new.components.map.map_generator import ScenarioGenerator
    from envs_new.components.config.environment_config import EnvironmentConfig
    
    config = EnvironmentConfig()
    generator = ScenarioGenerator(config)
    
    # 设置随机数生成器
    import numpy as np
    rng = np.random.Generator(np.random.PCG64(seed=42))
    generator.set_random_generator(rng)
    
    # 生成场景
    agent, maps_dict, env_state = generator.generate_scenario()
    
    # 验证field地图存在
    assert 'field' in maps_dict, "Missing field map in generated scenario"
    # 地图形状应该与环境状态中的dimensions一致
    height, width = env_state.dimensions
    assert maps_dict['field'].shape == (height, width)
    
    # 验证环境状态包含field信息
    assert hasattr(env_state, 'total_field_area'), "Missing total_field_area"
    assert env_state.total_field_area > 0, "Field area should be positive"
    
    print("   ✅ 地图生成正确")


def test_reward_calculation():
    """测试奖励计算"""
    print("\n7️⃣ 测试奖励计算...")
    
    env = CppEnvV2()
    obs, info = env.reset(seed=42)
    
    # 获取初始field面积
    initial_field_area = env.env_state.field_area
    
    # 执行一步
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 验证奖励被计算
    assert reward is not None, "Reward should not be None"
    
    # 获取奖励分解
    reward_breakdown = env.get_reward_breakdown()
    
    # 检查field相关奖励是否在分解中
    if 'components' in reward_breakdown:
        # field_coverage可能为0（如果没有覆盖新区域）
        assert 'field_coverage' in reward_breakdown['components'] or \
               'field_variation' in reward_breakdown['components'], \
               "Field-related rewards should be in breakdown"
    
    env.close()
    print("   ✅ 奖励计算正确")


def main():
    """运行所有验证测试"""
    print("\n" + "="*60)
    print("🧪 开始验证frontier→field重命名...")
    print("="*60)
    
    try:
        test_config_field_attributes()
        test_reward_system_field_calculators()
        test_environment_state_field_tracking()
        test_environment_creation_and_reset()
        test_environment_step_execution()
        test_map_generation()
        test_reward_calculation()
        
        print("\n" + "="*60)
        print("🎉 所有验证测试通过！重命名成功完成")
        print("="*60)
        
        print("\n📊 重命名总结：")
        print("  ✅ 配置系统：frontier → field")
        print("  ✅ 奖励系统：FrontierCalculator → FieldCalculator")
        print("  ✅ 状态管理：frontier_area → field_area")
        print("  ✅ 地图组件：FrontierCreator → FieldCreator")
        print("  ✅ 环境文件：field_frontier → field")
        print("  ✅ 渲染系统：frontier相关颜色 → field")
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ 运行错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())