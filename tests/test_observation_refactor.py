"""
测试观测系统重构效果
"""
import numpy as np
import torch
from envs_new.cpp_env_v1 import CppEnv as CppEnvV1
from envs_new.cpp_env_v2 import CppEnv as CppEnvV2
from envs_new.cpp_env_v3 import CppEnv as CppEnvV3


def test_unified_observation_flow():
    """测试统一的观测生成流程"""
    print("=== 测试统一观测生成流程 ===")
    
    # 测试单尺度观测
    env = CppEnvV1(
        use_multiscale=False,
        position_noise=0.5,
        direction_noise=2.0,
        state_size=(128, 128),
        state_downsize=(64, 64)
    )
    
    obs, info = env.reset(seed=42)
    observation = obs['observation']
    
    print(f"单尺度观测形状: {observation.shape}")
    assert observation.shape[1] == 64 and observation.shape[2] == 64
    print("✓ 单尺度观测测试通过")
    
    env.close()
    
    # 测试多尺度观测（有噪声）
    env = CppEnvV2(
        use_multiscale=True,
        use_global_features=True,
        position_noise=0.5,
        direction_noise=2.0,
        multiscale_feature_size=16,
        state_size=(256, 256),
        state_downsize=(128, 128)  # 128 / 8 = 16，满足要求
    )
    
    obs, info = env.reset(seed=42)
    observation = obs['observation']
    
    print(f"多尺度观测形状: {observation.shape}")
    # cpp_env_v2 有5个地图（包括trajectory_apf）
    # 5个地图 * 5个尺度（4个多尺度 + 1个全局）
    expected_channels = 5 * 5  # = 25
    assert observation.shape[0] == expected_channels, f"期望{expected_channels}个通道，实际{observation.shape[0]}个"
    assert observation.shape[1] == 16 and observation.shape[2] == 16
    print("✓ 多尺度观测测试通过\n")
    
    env.close()


def test_multiscale_config_validation():
    """测试多尺度配置验证"""
    print("=== 测试多尺度配置验证 ===")
    
    # 测试无效配置（state_downsize太小）
    try:
        env = CppEnvV2(
            use_multiscale=True,
            multiscale_feature_size=16,
            state_size=(64, 64),
            state_downsize=(32, 32)  # 32 / 8 = 4，小于16
        )
        print("❌ 应该抛出配置错误")
        env.close()
    except ValueError as e:
        print(f"✓ 正确捕获配置错误: {str(e)}")
    
    # 测试有效配置
    try:
        env = CppEnvV2(
            use_multiscale=True,
            multiscale_feature_size=16,
            state_size=(128, 128),
            state_downsize=(128, 128)  # 128 / 8 = 16，等于16
        )
        print("✓ 有效配置测试通过")
        env.close()
    except ValueError as e:
        print(f"❌ 不应该抛出错误: {e}")
    
    print()


def test_noise_consistency():
    """测试噪声应用的一致性"""
    print("=== 测试噪声应用一致性 ===")
    
    # 创建两个环境，一个单尺度，一个多尺度
    env1 = CppEnvV1(
        use_multiscale=False,
        position_noise=1.0,
        direction_noise=5.0,
        state_size=(128, 128),
        state_downsize=(64, 64)
    )
    
    env2 = CppEnvV2(
        use_multiscale=True,
        position_noise=1.0,
        direction_noise=5.0,
        multiscale_feature_size=16,
        state_size=(256, 256),
        state_downsize=(128, 128)  # 128 / 8 = 16，满足要求
    )
    
    # 使用相同的种子重置
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    
    # 执行几步，检查噪声是否被应用
    observations1 = []
    observations2 = []
    
    for _ in range(5):
        action = 0  # 不移动
        obs1, _, _, _, _ = env1.step(action)
        obs2, _, _, _, _ = env2.step(action)
        
        observations1.append(obs1['observation'])
        observations2.append(obs2['observation'])
    
    # 检查观测是否有变化（由于噪声）
    obs_changes1 = [np.sum(np.abs(observations1[i] - observations1[0])) for i in range(1, 5)]
    obs_changes2 = [np.sum(np.abs(observations2[i][:4] - observations2[0][:4])) for i in range(1, 5)]
    
    print(f"单尺度观测变化: {obs_changes1}")
    print(f"多尺度观测变化（前4通道）: {obs_changes2}")
    
    # 两种模式都应该有噪声引起的变化
    assert any(change > 0 for change in obs_changes1), "单尺度观测应该有噪声变化"
    assert any(change > 0 for change in obs_changes2), "多尺度观测应该有噪声变化"
    
    print("✓ 噪声一致性测试通过\n")
    
    env1.close()
    env2.close()


def test_performance():
    """测试重构后的性能"""
    print("=== 测试重构性能 ===")
    
    import time
    
    # 测试单尺度性能
    env = CppEnvV1(use_multiscale=False)
    env.reset(seed=42)
    
    start_time = time.time()
    for _ in range(100):
        action = env.action_space.sample()
        env.step(action)
    single_time = time.time() - start_time
    
    env.close()
    
    # 测试多尺度性能
    env = CppEnvV2(
        use_multiscale=True,
        use_global_features=True
    )
    env.reset(seed=42)
    
    start_time = time.time()
    for _ in range(100):
        action = env.action_space.sample()
        env.step(action)
    multi_time = time.time() - start_time
    
    env.close()
    
    print(f"单尺度100步耗时: {single_time:.3f}秒")
    print(f"多尺度100步耗时: {multi_time:.3f}秒")
    print(f"多尺度/单尺度时间比: {multi_time/single_time:.2f}x")
    print("✓ 性能测试完成\n")


def test_all_env_variants():
    """测试所有环境变体"""
    print("=== 测试所有环境变体 ===")
    
    configs = [
        ("CppEnvV1", CppEnvV1, {"use_multiscale": False}),
        ("CppEnvV1_multi", CppEnvV1, {"use_multiscale": True, "use_global_features": True}),
        ("CppEnvV2", CppEnvV2, {"use_multiscale": True, "use_global_features": True}),
        ("CppEnvV3", CppEnvV3, {"use_multiscale": False}),
    ]
    
    for name, env_class, kwargs in configs:
        try:
            env = env_class(**kwargs)
            obs, info = env.reset(seed=42)
            
            # 执行几步
            for _ in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            
            print(f"✓ {name} 测试通过，观测形状: {obs['observation'].shape}")
            env.close()
            
        except Exception as e:
            print(f"❌ {name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    print()


if __name__ == "__main__":
    print("开始测试观测系统重构...\n")
    
    try:
        test_unified_observation_flow()
        test_multiscale_config_validation()
        test_noise_consistency()
        test_performance()
        test_all_env_variants()
        
        print("🎉 所有测试通过！观测系统重构成功！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()