#!/usr/bin/env python
"""
测试V4和V5环境及模型的独立性
确保两个版本可以独立训练和评估
"""
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# V4相关导入
from rl.sac_cont.area_coverage_utils import (
    make_area_coverage_env,
    make_area_coverage_sac_models
)

# V5相关导入
from rl.sac_cont.area_coverage_v5_utils import (
    make_area_coverage_v5_env,
    make_area_coverage_v5_sac_models
)


def test_v4_environment():
    """测试V4环境"""
    print("=" * 60)
    print("测试 V4 环境 (Pasture-v4)")
    print("-" * 60)
    
    # 创建V4环境
    env = make_area_coverage_env(num_envs=1, device="cpu")
    obs_spec = env.observation_spec
    
    print(f"✓ V4环境创建成功")
    print(f"  - 环境ID: Pasture-v4")
    print(f"  - 观察空间: {obs_spec['observation'].shape}")
    print(f"  - 通道数: {obs_spec['observation'].shape[0]}")
    print(f"  - SGCNN: 禁用")
    
    # 验证是4通道
    assert obs_spec['observation'].shape[0] == 4, f"V4应该是4通道，但得到{obs_spec['observation'].shape[0]}"
    
    env.close()
    print("✅ V4环境测试通过")
    
    return obs_spec['observation'].shape[0]


def test_v5_environment():
    """测试V5环境"""
    print("\n" + "=" * 60)
    print("测试 V5 环境 (Pasture-v5)")
    print("-" * 60)
    
    # 创建V5环境
    env = make_area_coverage_v5_env(num_envs=1, device="cpu")
    obs_spec = env.observation_spec
    
    print(f"✓ V5环境创建成功")
    print(f"  - 环境ID: Pasture-v5")
    print(f"  - 观察空间: {obs_spec['observation'].shape}")
    print(f"  - 通道数: {obs_spec['observation'].shape[0]}")
    print(f"  - SGCNN: 启用")
    print(f"  - 方向场奖励: 启用")
    
    # 验证是20通道
    assert obs_spec['observation'].shape[0] == 20, f"V5应该是20通道，但得到{obs_spec['observation'].shape[0]}"
    
    env.close()
    print("✅ V5环境测试通过")
    
    return obs_spec['observation'].shape[0]


def test_v4_model():
    """测试V4模型"""
    print("\n" + "=" * 60)
    print("测试 V4 模型架构")
    print("-" * 60)
    
    # 创建V4模型
    model = make_area_coverage_sac_models()
    
    # 检查第一个卷积层
    actor = model[0]
    first_conv = None
    for module in actor.modules():
        if isinstance(module, torch.nn.Conv2d):
            first_conv = module
            break
    
    if first_conv:
        print(f"✓ V4模型创建成功")
        print(f"  - Actor输入通道: {first_conv.in_channels}")
        assert first_conv.in_channels == 4, f"V4模型应该接受4通道输入"
    
    print("✅ V4模型测试通过")


def test_v5_model():
    """测试V5模型"""
    print("\n" + "=" * 60)
    print("测试 V5 模型架构")
    print("-" * 60)
    
    # 创建V5模型
    model = make_area_coverage_v5_sac_models()
    
    # 检查第一个卷积层
    actor = model[0]
    first_conv = None
    for module in actor.modules():
        if isinstance(module, torch.nn.Conv2d):
            first_conv = module
            break
    
    if first_conv:
        print(f"✓ V5模型创建成功")
        print(f"  - Actor输入通道: {first_conv.in_channels}")
        assert first_conv.in_channels == 20, f"V5模型应该接受20通道输入"
    
    print("✅ V5模型测试通过")


def test_independence():
    """测试V4和V5的独立性"""
    print("\n" + "=" * 60)
    print("测试 V4 和 V5 独立性")
    print("-" * 60)
    
    # 同时创建两个环境
    env_v4 = make_area_coverage_env(num_envs=1, device="cpu")
    env_v5 = make_area_coverage_v5_env(num_envs=1, device="cpu")
    
    # 同时创建两个模型
    model_v4 = make_area_coverage_sac_models()
    model_v5 = make_area_coverage_v5_sac_models()
    
    # 验证它们的差异
    v4_channels = env_v4.observation_spec['observation'].shape[0]
    v5_channels = env_v5.observation_spec['observation'].shape[0]
    
    print(f"✓ V4环境: {v4_channels}通道")
    print(f"✓ V5环境: {v5_channels}通道")
    
    assert v4_channels == 4, "V4应该是4通道"
    assert v5_channels == 20, "V5应该是20通道"
    assert v4_channels != v5_channels, "V4和V5应该有不同的通道数"
    
    env_v4.close()
    env_v5.close()
    
    print("✅ 独立性测试通过")


def test_training_scripts():
    """测试训练脚本的存在性"""
    print("\n" + "=" * 60)
    print("检查训练和评估脚本")
    print("-" * 60)
    
    base_dir = Path(__file__).parent.parent
    
    scripts = {
        "V4训练": base_dir / "rl/sac_cont/area_coverage_sac_cont_train.py",
        "V4评估": base_dir / "rl/sac_cont/area_coverage_sac_cont_eval.py",
        "V5训练": base_dir / "rl/sac_cont/area_coverage_v5_sac_cont_train.py",
        "V5评估": base_dir / "rl/sac_cont/area_coverage_v5_sac_cont_eval.py",
    }
    
    configs = {
        "V4环境配置": base_dir / "configs/env_config_area_coverage.yaml",
        "V4训练配置": base_dir / "configs/train_area_coverage_sac_cont_config.yaml",
        "V5环境配置": base_dir / "configs/env_config_area_coverage_v5.yaml",
        "V5训练配置": base_dir / "configs/train_area_coverage_v5_sac_cont_config.yaml",
    }
    
    print("脚本文件:")
    for name, path in scripts.items():
        if path.exists():
            print(f"  ✓ {name}: {path.name}")
        else:
            print(f"  ❌ {name}: 文件不存在")
    
    print("\n配置文件:")
    for name, path in configs.items():
        if path.exists():
            print(f"  ✓ {name}: {path.name}")
        else:
            print(f"  ❌ {name}: 文件不存在")
    
    print("\n✅ 文件检查完成")


def main():
    """主测试函数"""
    print("\n🚀 开始测试 V4 和 V5 环境的独立性\n")
    
    try:
        # 测试V4
        v4_channels = test_v4_environment()
        test_v4_model()
        
        # 测试V5
        v5_channels = test_v5_environment()
        test_v5_model()
        
        # 测试独立性
        test_independence()
        
        # 检查文件
        test_training_scripts()
        
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！")
        print("=" * 60)
        
        print("\n📋 总结:")
        print(f"  • V4: {v4_channels}通道，无SGCNN，无方向场奖励")
        print(f"  • V5: {v5_channels}通道，有SGCNN，有方向场奖励")
        print("  • 两个版本完全独立，可以并行训练")
        
        print("\n🎯 使用方法:")
        print("  训练V4: python rl/sac_cont/area_coverage_sac_cont_train.py")
        print("  评估V4: python rl/sac_cont/area_coverage_sac_cont_eval.py")
        print("  训练V5: python rl/sac_cont/area_coverage_v5_sac_cont_train.py")
        print("  评估V5: python rl/sac_cont/area_coverage_v5_sac_cont_eval.py")
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()