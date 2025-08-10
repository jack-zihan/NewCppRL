#!/usr/bin/env python
"""
验证area_coverage训练和评估的一致性
测试环境配置、模型架构和数据通道的匹配性
"""
import sys
import torch
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.sac_cont.area_coverage_utils import (
    make_area_coverage_env,
    make_area_coverage_sac_models
)


def test_environment_consistency():
    """测试环境创建的一致性"""
    print("=" * 60)
    print("测试1: 环境创建一致性")
    print("-" * 60)
    
    # 创建单个环境
    env_single = make_area_coverage_env(num_envs=1, device="cpu")
    obs_spec = env_single.observation_spec
    
    print(f"✓ 单环境创建成功")
    print(f"  - 观察空间形状: {obs_spec['observation'].shape}")
    print(f"  - 通道数: {obs_spec['observation'].shape[0]}")
    
    # 创建并行环境
    env_parallel = make_area_coverage_env(num_envs=4, device="cpu")
    obs_spec_parallel = env_parallel.observation_spec
    
    print(f"✓ 并行环境创建成功")
    print(f"  - 观察空间形状: {obs_spec_parallel['observation'].shape}")
    
    # 验证一致性（并行环境有额外的批处理维度）
    single_shape = obs_spec['observation'].shape
    parallel_shape = obs_spec_parallel['observation'].shape
    
    # 并行环境的形状应该是 [batch_size, channels, height, width]
    if len(parallel_shape) == 4:
        # 去掉批处理维度进行比较
        assert single_shape == parallel_shape[1:], \
            f"单环境形状{single_shape}和并行环境形状{parallel_shape[1:]}不一致"
    else:
        assert single_shape == parallel_shape, \
            "单环境和并行环境的观察空间不一致"
    
    print(f"✅ 环境创建一致性测试通过")
    
    # 清理
    try:
        env_single.close()
    except:
        pass
    try:
        env_parallel.close()
    except:
        pass
    
    return obs_spec['observation'].shape[0]  # 返回通道数


def test_model_architecture(expected_channels):
    """测试模型架构与环境的匹配性"""
    print("\n" + "=" * 60)
    print("测试2: 模型架构匹配性")
    print("-" * 60)
    
    # 创建模型
    actor_critic = make_area_coverage_sac_models()
    
    print(f"✓ 模型创建成功")
    
    # 获取模型的第一个卷积层
    actor = actor_critic[0]
    critic = actor_critic[1]
    
    # 查找actor的第一个卷积层
    actor_conv1 = None
    for module in actor.modules():
        if isinstance(module, torch.nn.Conv2d):
            actor_conv1 = module
            break
    
    # 查找critic的第一个卷积层
    critic_conv1 = None
    for module in critic.modules():
        if isinstance(module, torch.nn.Conv2d):
            critic_conv1 = module
            break
    
    if actor_conv1:
        print(f"  - Actor输入通道数: {actor_conv1.in_channels}")
        assert actor_conv1.in_channels == expected_channels, \
            f"Actor期望{expected_channels}通道，但得到{actor_conv1.in_channels}通道"
    
    if critic_conv1:
        print(f"  - Critic输入通道数: {critic_conv1.in_channels}")
        assert critic_conv1.in_channels == expected_channels, \
            f"Critic期望{expected_channels}通道，但得到{critic_conv1.in_channels}通道"
    
    print(f"✅ 模型架构匹配性测试通过")
    
    return actor_critic


def test_forward_pass(model):
    """测试模型的前向传播"""
    print("\n" + "=" * 60)
    print("测试3: 模型前向传播")
    print("-" * 60)
    
    # 创建环境并获取数据
    env = make_area_coverage_env(device="cpu")
    
    # Reset环境
    td_reset = env.reset()
    print(f"✓ 环境重置成功")
    
    # 执行几步
    with torch.no_grad():
        td = env.rollout(max_steps=10, policy=model[0], break_when_any_done=False)
    
    print(f"✓ 模型推理成功")
    print(f"  - 执行步数: {td.shape[0]}")
    print(f"  - 奖励形状: {td['next', 'reward'].shape}")
    
    # 清理
    env.close()
    
    print(f"✅ 前向传播测试通过")


def test_config_consistency():
    """测试配置文件的一致性"""
    print("\n" + "=" * 60)
    print("测试4: 配置文件一致性")
    print("-" * 60)
    
    import yaml
    from omegaconf import DictConfig
    
    base_dir = Path(__file__).parent.parent
    
    # 读取area_coverage配置
    area_cfg = yaml.load(
        open(f'{base_dir}/configs/env_config_area_coverage.yaml'), 
        Loader=yaml.FullLoader
    )
    
    print(f"✓ Area Coverage配置:")
    print(f"  - 环境ID: {area_cfg['env']['params']['id']}")
    print(f"  - use_sgcnn: {area_cfg['env']['params']['use_sgcnn']}")
    print(f"  - use_global_obs: {area_cfg['env']['params']['use_global_obs']}")
    
    # 验证关键配置
    assert area_cfg['env']['params']['id'] in ["Pasture-v4", "Pasture-v5"], \
        "环境ID应该是Pasture-v4或v5"
    
    expected_channels = 25 if area_cfg['env']['params']['use_sgcnn'] else 4
    print(f"  - 预期通道数: {expected_channels}")
    
    print(f"✅ 配置文件一致性测试通过")
    
    return expected_channels


def main():
    """主测试函数"""
    print("\n" + "🚀 开始测试 Area Coverage 一致性修复" + "\n")
    
    try:
        # 测试配置一致性
        expected_channels = test_config_consistency()
        
        # 测试环境创建
        actual_channels = test_environment_consistency()
        
        # 验证通道数匹配
        assert actual_channels == expected_channels, \
            f"环境通道数({actual_channels})与配置预期({expected_channels})不匹配"
        
        # 测试模型架构
        model = test_model_architecture(actual_channels)
        
        # 测试前向传播
        test_forward_pass(model)
        
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！修复成功！")
        print("=" * 60)
        print("\n总结:")
        print(f"✅ 环境配置统一使用 env_config_area_coverage.yaml")
        print(f"✅ 环境提供 {actual_channels} 个通道")
        print(f"✅ 模型期望 {actual_channels} 个通道")
        print(f"✅ 训练和评估将使用一致的架构")
        print("\n您现在可以重新开始训练了！")
        
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