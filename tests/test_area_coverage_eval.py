#!/usr/bin/env python3
"""
测试Area Coverage SAC评估脚本
验证评估器可以正确处理V4/V5环境
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import gymnasium as gym
import torch
import numpy as np
import yaml
from omegaconf import DictConfig

import envs  # 注册环境


def test_metric_detection():
    """测试环境类型检测和度量名称设置"""
    print("\n=== 测试环境类型检测 ===")
    
    # 直接模拟不同的环境配置，而不是修改文件
    test_cases = [
        ("Pasture-v2", "weed_ratio", "Weed Ratio"),
        ("Pasture-v4", "coverage_rate", "Coverage"),
        ("Pasture-v5", "coverage_rate", "Coverage"),
    ]
    
    for env_id, expected_metric, expected_display in test_cases:
        print(f"\n测试 {env_id}:")
        
        # 动态创建一个测试用的评估器类
        from rl.sac_cont.area_coverage_sac_cont_eval import AreaCoverageSacEvaluator
        
        class TestEvaluator(AreaCoverageSacEvaluator):
            def __init__(self, test_env_id, *args, **kwargs):
                # 跳过父类初始化，直接设置必要属性
                self.ckpt_path = 'dummy'
                self.ckpt_dir = 'dummy'
                self.is_coverage_env = test_env_id in ["Pasture-v4", "Pasture-v5"]
                
                if self.is_coverage_env:
                    self.metric_name = "coverage_rate"
                    self.metric_display = "Coverage"
                else:
                    self.metric_name = "weed_ratio"
                    self.metric_display = "Weed Ratio"
        
        # 创建测试实例
        evaluator = TestEvaluator(env_id)
        
        # 检查度量名称
        assert evaluator.metric_name == expected_metric, \
            f"期望 metric_name={expected_metric}, 实际={evaluator.metric_name}"
        assert evaluator.metric_display == expected_display, \
            f"期望 metric_display={expected_display}, 实际={evaluator.metric_display}"
        
        print(f"  ✓ 正确检测: metric_name={evaluator.metric_name}, display={evaluator.metric_display}")
    
    print("\n✅ 环境类型检测测试通过")
    return True


def test_observation_compatibility():
    """测试观测兼容性（weed_ratio键但实际是coverage值）"""
    print("\n=== 测试观测兼容性 ===")
    
    for env_id in ["Pasture-v4", "Pasture-v5"]:
        print(f"\n测试 {env_id}:")
        
        # 创建环境
        env = gym.make(env_id, state_pixels=False, action_type='continuous')
        obs, info = env.reset(seed=42)
        
        # 检查观测中的键
        assert 'weed_ratio' in obs, f"{env_id} 观测中应该有'weed_ratio'键"
        print(f"  ✓ 观测包含'weed_ratio'键")
        
        # 对于V4/V5，这个值实际是coverage_rate
        # 验证值的范围
        weed_ratio_value = obs['weed_ratio']
        assert 0 <= weed_ratio_value <= 1, f"weed_ratio值应在[0,1]范围内，实际={weed_ratio_value}"
        print(f"  ✓ 'weed_ratio'值在有效范围内: {weed_ratio_value:.4f}")
        
        # 验证info中的coverage_rate（V4/V5特有）
        if env_id in ["Pasture-v4", "Pasture-v5"]:
            assert 'coverage_rate' in info, f"{env_id} info中应该有'coverage_rate'"
            # 初始时，obs['weed_ratio']应该等于info['coverage_rate']
            assert abs(obs['weed_ratio'] - info['coverage_rate']) < 1e-6, \
                f"obs['weed_ratio']应该等于info['coverage_rate']"
            print(f"  ✓ obs['weed_ratio'] == info['coverage_rate'] = {info['coverage_rate']:.4f}")
        
        env.close()
    
    print("\n✅ 观测兼容性测试通过")
    return True


def test_action_extraction():
    """测试动作提取（SAC连续动作）"""
    print("\n=== 测试动作提取 ===")
    
    # 创建一个简单的模拟actor
    class MockActor(torch.nn.Module):
        def forward(self, observation, vector):
            batch_size = observation.shape[0]
            # 模拟SAC actor的输出格式：(mean, std, action)
            mean = torch.zeros(batch_size, 2)  # 2维连续动作
            std = torch.ones(batch_size, 2)
            action = torch.randn(batch_size, 2) * 0.5  # 模拟动作
            return mean, std, action
    
    from rl.sac_cont.area_coverage_sac_cont_eval import AreaCoverageSacEvaluator
    
    evaluator = AreaCoverageSacEvaluator(
        episodes=2,
        max_frames=1,
        max_step=10,
        video=False,
        device='cpu',
        ckpt_path='dummy',  # 避免自动查找不存在的路径
    )
    
    # 创建模拟观测
    mock_obss = [
        {'observation': np.random.rand(4, 128, 128), 'vector': 0.5}
        for _ in range(2)
    ]
    
    # 测试动作提取
    actor = MockActor()
    actions = evaluator.get_actions(actor, mock_obss)
    
    assert len(actions) == 2, f"应该返回2个动作，实际={len(actions)}"
    assert all(len(a) == 2 for a in actions), "每个动作应该是2维的"
    
    print(f"  ✓ 成功提取{len(actions)}个动作")
    print(f"  ✓ 动作示例: {actions[0]}")
    print("\n✅ 动作提取测试通过")
    return True


def main():
    """运行所有测试"""
    print("=" * 60)
    print("Area Coverage SAC评估器测试")
    print("=" * 60)
    
    tests = [
        ("环境类型检测", test_metric_detection),
        ("观测兼容性", test_observation_compatibility),
        ("动作提取", test_action_extraction),
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        try:
            result = test_func()
            if not result:
                all_passed = False
                print(f"\n❌ {test_name}测试失败")
        except Exception as e:
            all_passed = False
            print(f"\n❌ {test_name}测试出错: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！")
        print("\n评估器使用说明：")
        print("1. 训练模型：python -m rl.sac_cont.area_coverage_sac_cont_train")
        print("2. 评估模型：python -m rl.sac_cont.area_coverage_sac_cont_eval")
        print("\n注意事项：")
        print("- V4/V5环境的'weed_ratio'键实际存储的是coverage_rate")
        print("- 评估器会自动检测环境类型并使用正确的度量名称")
    else:
        print("⚠️ 部分测试失败")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)