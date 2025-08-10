"""
测试观测系统修复效果
"""
import numpy as np
from envs_new.cpp_env_v1 import CppEnv as CppEnvV1
from envs_new.cpp_env_v2 import CppEnv as CppEnvV2


def test_mask_padding():
    """测试mask填充逻辑是否正确"""
    print("=== 测试Mask填充逻辑 ===")
    
    # 创建v1环境
    env = CppEnvV1()
    obs, info = env.reset(seed=42)
    
    # 获取观测地图
    obs_maps = env._get_observation_maps()
    
    # 验证每个地图都有正确的pad值
    print("地图pad值检查：")
    for name, map_info in obs_maps.items():
        pad_value = map_info['pad']
        print(f"  {name}: pad={pad_value}")
        
        # 验证obstacle的pad值为1.0
        if 'obstacle' in name:
            assert pad_value == 1.0, f"{name}的pad值应该为1.0，实际为{pad_value}"
        else:
            assert pad_value == 0.0, f"{name}的pad值应该为0.0，实际为{pad_value}"
    
    env.close()
    print("✓ Mask填充逻辑测试通过\n")


def test_multiscale_observation():
    """测试多尺度观测实现"""
    print("=== 测试多尺度观测 ===")
    
    # 创建启用多尺度的环境
    env = CppEnvV2(
        use_multiscale=True,
        use_global_features=True,
        multiscale_feature_size=16,
        state_size=(128, 128),
        state_downsize=(64, 64)
    )
    
    obs, info = env.reset(seed=42)
    observation = obs['observation']
    
    print(f"观测形状: {observation.shape}")
    
    # 验证通道数
    # 基础地图通道数
    obs_maps = env._get_observation_maps()
    base_channels = len(obs_maps)
    
    # 多尺度：4个尺度 + 1个全局（如果启用）
    expected_channels = base_channels * 5  # 4个尺度 + 1个全局
    
    assert observation.shape[0] == expected_channels, \
        f"期望{expected_channels}个通道，实际得到{observation.shape[0]}个"
    
    # 验证每个尺度的大小
    feature_size = env.config.multiscale_feature_size
    assert observation.shape[1] == feature_size, \
        f"期望高度为{feature_size}，实际为{observation.shape[1]}"
    assert observation.shape[2] == feature_size, \
        f"期望宽度为{feature_size}，实际为{observation.shape[2]}"
    
    env.close()
    print("✓ 多尺度观测测试通过\n")


def test_single_scale_observation():
    """测试单尺度观测"""
    print("=== 测试单尺度观测 ===")
    
    # 创建不使用多尺度的环境
    env = CppEnvV1(
        use_multiscale=False,
        state_size=(128, 128),
        state_downsize=(64, 64)
    )
    
    obs, info = env.reset(seed=42)
    observation = obs['observation']
    
    print(f"观测形状: {observation.shape}")
    
    # 验证形状
    obs_maps = env._get_observation_maps()
    expected_channels = len(obs_maps)
    expected_height = env.config.state_downsize[0]
    expected_width = env.config.state_downsize[1]
    
    assert observation.shape == (expected_channels, expected_height, expected_width), \
        f"期望形状为{(expected_channels, expected_height, expected_width)}，" \
        f"实际为{observation.shape}"
    
    env.close()
    print("✓ 单尺度观测测试通过\n")


def test_obstacle_boundary():
    """测试障碍物边界处理"""
    print("=== 测试障碍物边界处理 ===")
    
    env = CppEnvV2(num_obstacles_range=[1, 1])
    obs, info = env.reset(seed=42)
    
    # 执行几步，确保agent接近边界
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    
    # 检查观测是否正常生成（不会因为边界问题崩溃）
    assert obs['observation'].shape[0] > 0, "观测生成失败"
    
    env.close()
    print("✓ 障碍物边界处理测试通过\n")


if __name__ == "__main__":
    print("开始测试观测系统修复...\n")
    
    try:
        test_mask_padding()
        test_multiscale_observation()
        test_single_scale_observation()
        test_obstacle_boundary()
        
        print("🎉 所有测试通过！观测系统修复成功！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()