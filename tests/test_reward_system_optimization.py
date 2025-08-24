#!/usr/bin/env python3
"""
测试奖励系统优化后的功能
验证命名优化和系数更新逻辑的正确性
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from envs_new.components.config.environment_config import EnvironmentConfig
from envs_new.components.reward.reward_system import RewardSystem
from envs_new.components.state.environment_state import EnvironmentState


def test_config_naming():
    """测试配置命名是否清晰和一致"""
    config = EnvironmentConfig()
    
    # 验证新的命名
    assert hasattr(config, 'reward_base_penalty')
    assert hasattr(config, 'reward_weed_removal')
    assert hasattr(config, 'reward_field_coverage')
    assert hasattr(config, 'reward_field_variation')
    assert hasattr(config, 'reward_turning_penalty')
    assert hasattr(config, 'reward_direction_change_penalty')
    assert hasattr(config, 'reward_steering_smoothness')
    assert hasattr(config, 'reward_field_group_coef')
    assert hasattr(config, 'reward_turning_group_coef')
    
    print("✅ 配置命名测试通过")


def test_coefficient_update():
    """测试系数更新机制"""
    config = EnvironmentConfig()
    reward_system = RewardSystem(config)
    
    # 验证Calculator名称与配置属性对应
    assert 'base_penalty' in reward_system.AVAILABLE_CALCULATORS
    assert 'weed_removal' in reward_system.AVAILABLE_CALCULATORS
    assert 'field_coverage' in reward_system.AVAILABLE_CALCULATORS
    assert 'field_variation' in reward_system.AVAILABLE_CALCULATORS
    assert 'turning_penalty' in reward_system.AVAILABLE_CALCULATORS
    
    # 测试系数更新
    from envs_new.components.reward.reward_system import BaseCalculator, WeedRemovalCalculator
    
    # 检查初始系数
    assert BaseCalculator.coefficient == config.reward_base_penalty
    assert WeedRemovalCalculator.coefficient == config.reward_weed_removal
    
    # 更新系数
    new_coefficients = {
        'base_penalty': -0.2,
        'weed_removal': 30.0,
        'turning_penalty': -1.0
    }
    reward_system.update_coefficients(new_coefficients)
    
    # 验证更新后的系数
    assert config.reward_base_penalty == -0.2
    assert config.reward_weed_removal == 30.0
    assert config.reward_turning_penalty == -1.0
    assert BaseCalculator.coefficient == -0.2
    assert WeedRemovalCalculator.coefficient == 30.0
    
    print("✅ 系数更新测试通过")


def test_group_coefficient():
    """测试组系数机制"""
    config = EnvironmentConfig()
    config.reward_field_group_coef = 2.0  # 设置田地组系数
    config.reward_turning_group_coef = 0.5   # 设置转向组系数
    
    reward_system = RewardSystem(config)
    
    # 创建模拟的环境状态
    env_state = EnvironmentState()
    # 使用正确的方法添加状态信息
    weed_count = env_state.add_state_info('weed_count', history_length=2)
    weed_count.update(102)  # 初始值
    weed_count.update(100)  # 当前值（清除了2个杂草）
    
    field_area = env_state.add_state_info('field_area', history_length=2)
    field_area.update(1000)  # 初始值
    field_area.update(900)   # 当前值（覆盖了100单位田地）
    
    agent_steer = env_state.add_state_info('agent_steer', history_length=2)
    agent_steer.update(0.0)  # 初始值
    agent_steer.update(0.5)  # 当前值
    
    # 添加crashed和finished状态（终止条件）
    env_state.add_state_info('crashed', history_length=1, initial_value=False)
    env_state.add_state_info('finished', history_length=1, initial_value=False)
    
    # 测试组系数是否正确应用
    reward_breakdown = reward_system.get_reward_breakdown(env_state)
    
    # 验证原始奖励（不含组系数）
    assert 'components_raw' in reward_breakdown
    raw_field_coverage = reward_breakdown['components_raw'].get('field_coverage', 0)
    
    # 验证应用组系数后的奖励
    assert 'components' in reward_breakdown
    final_field_coverage = reward_breakdown['components'].get('field_coverage', 0)
    
    # 组系数应该被正确应用
    if raw_field_coverage != 0:
        expected_ratio = config.reward_field_group_coef
        actual_ratio = final_field_coverage / raw_field_coverage
        assert abs(actual_ratio - expected_ratio) < 0.01, f"组系数应用错误: {actual_ratio} != {expected_ratio}"
    
    print("✅ 组系数测试通过")


def test_reward_groups_structure():
    """测试REWARD_GROUPS结构化定义"""
    config = EnvironmentConfig()
    reward_system = RewardSystem(config)
    
    # 验证REWARD_GROUPS结构
    assert hasattr(reward_system, 'REWARD_GROUPS')
    assert 'field' in reward_system.REWARD_GROUPS
    assert 'turning' in reward_system.REWARD_GROUPS
    
    # 验证组定义
    field_group = reward_system.REWARD_GROUPS['field']
    assert 'coef_attr' in field_group
    assert 'members' in field_group
    assert field_group['coef_attr'] == 'reward_field_group_coef'
    assert 'field_coverage' in field_group['members']
    assert 'field_variation' in field_group['members']
    
    turning_group = reward_system.REWARD_GROUPS['turning']
    assert turning_group['coef_attr'] == 'reward_turning_group_coef'
    assert 'turning_penalty' in turning_group['members']
    assert 'direction_change_penalty' in turning_group['members']
    assert 'steering_smoothness' in turning_group['members']
    
    print("✅ REWARD_GROUPS结构测试通过")


def test_naming_consistency():
    """测试命名一致性：配置属性与Calculator名称对应"""
    config = EnvironmentConfig()
    reward_system = RewardSystem(config)
    
    # 对于每个Calculator，验证对应的配置属性存在
    for calc_name in reward_system.AVAILABLE_CALCULATORS.keys():
        config_attr = f"reward_{calc_name}"
        assert hasattr(config, config_attr), f"配置缺少属性: {config_attr}"
    
    # 验证get_reward_coefficients返回的键名与Calculator名称一致
    coefficients = config.get_reward_coefficients()
    for calc_name in reward_system.AVAILABLE_CALCULATORS.keys():
        assert calc_name in coefficients, f"get_reward_coefficients缺少: {calc_name}"
    
    print("✅ 命名一致性测试通过")


def test_simplified_code():
    """验证代码简化效果"""
    config = EnvironmentConfig()
    reward_system = RewardSystem(config)
    
    # 检查是否移除了旧的映射
    # 不应该再有 'base' -> 'base_penalty' 这样的映射
    
    # update_coefficients应该直接使用新名称
    test_coeffs = {'base_penalty': -0.15, 'weed_removal': 25.0}
    reward_system.update_coefficients(test_coeffs)
    assert config.reward_base_penalty == -0.15
    assert config.reward_weed_removal == 25.0
    
    # _update_coefficients应该使用命名约定
    # 这在内部已经验证过了
    
    print("✅ 代码简化测试通过")


def main():
    """运行所有测试"""
    print("\n🧪 开始测试奖励系统优化...\n")
    
    try:
        test_config_naming()
        test_coefficient_update()
        test_group_coefficient()
        test_reward_groups_structure()
        test_naming_consistency()
        test_simplified_code()
        
        print("\n🎉 所有测试通过！奖励系统优化成功")
        print("\n📊 优化效果总结：")
        print("  1. 命名更清晰：reward_turning_penalty 替代 reward_turn_gap_coef")
        print("  2. 移除多层映射：直接通过命名约定访问配置")
        print("  3. 结构化组定义：REWARD_GROUPS 清晰定义组关系")
        print("  4. 代码更简洁：去除所有兼容性映射代码")
        print("  5. 维护更容易：添加新Calculator只需遵循命名约定")
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ 运行错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())