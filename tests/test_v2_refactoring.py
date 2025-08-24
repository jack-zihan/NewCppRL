#!/usr/bin/env python3
"""
测试v2环境重构后的正确性
验证：
1. 环境能正常创建和运行
2. 观察地图生成正确
3. APF奖励计算正确
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs_new.cpp_env_v2 import CppEnv


def test_env_creation():
    """测试环境创建"""
    print("测试1：环境创建...")
    try:
        env = CppEnv(render_mode=None)
        print("✅ 环境创建成功")
        return env
    except Exception as e:
        print(f"❌ 环境创建失败: {e}")
        raise


def test_env_reset(env):
    """测试环境重置"""
    print("\n测试2：环境重置...")
    try:
        obs, info = env.reset(seed=42)
        assert 'observation' in obs
        assert 'vector' in obs
        assert 'completion_ratio' in obs
        print(f"✅ 环境重置成功")
        print(f"  - 观察形状: {obs['observation'].shape}")
        print(f"  - 观察地图数量: {len(env._get_observation_maps())}")
        return obs
    except Exception as e:
        print(f"❌ 环境重置失败: {e}")
        raise


def test_observation_maps(env):
    """测试观察地图生成"""
    print("\n测试3：观察地图生成...")
    try:
        obs_maps = env._get_observation_maps()
        
        # 检查返回的地图名称
        expected_keys = ['field_obs', 'mist_inv', 'obstacle_obs', 'weed_obs']
        if env.use_traj:
            expected_keys.append('trajectory_obs')
            
        for key in expected_keys:
            assert key in obs_maps, f"缺少地图: {key}"
            assert 'map' in obs_maps[key], f"{key} 缺少 'map' 字段"
            assert 'pad' in obs_maps[key], f"{key} 缺少 'pad' 字段"
        
        # 检查pad值
        assert obs_maps['obstacle_obs']['pad'] == 1.0, "obstacle_obs 的 pad 值应该是 1.0"
        assert obs_maps['field_obs']['pad'] == 0.0, "field_obs 的 pad 值应该是 0.0"
        
        print("✅ 观察地图生成正确")
        print(f"  - 地图键: {list(obs_maps.keys())}")
        
        # 检查APF转换
        if env.use_apf:
            # APF地图应该是浮点数且有渐变值
            field_map = obs_maps['field_obs']['map']
            unique_vals = np.unique(field_map)
            print(f"  - APF场值范围: [{field_map.min():.3f}, {field_map.max():.3f}]")
            print(f"  - APF场唯一值数量: {len(unique_vals)}")
            if len(unique_vals) > 2:
                print("  - ✅ APF转换生效（存在渐变值）")
            else:
                print("  - ⚠️  APF可能未生效（只有二值）")
        else:
            print("  - APF未启用")
            
    except Exception as e:
        print(f"❌ 观察地图生成失败: {e}")
        raise


def test_env_step(env):
    """测试环境步进"""
    print("\n测试4：环境步进...")
    try:
        # 执行几步
        total_reward = 0
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if i == 0:
                print(f"✅ 第一步执行成功")
                print(f"  - 奖励: {reward:.4f}")
                print(f"  - 终止: {terminated}, 截断: {truncated}")
            
            if terminated or truncated:
                break
        
        print(f"✅ 多步执行成功")
        print(f"  - 总奖励: {total_reward:.4f}")
        
    except Exception as e:
        print(f"❌ 环境步进失败: {e}")
        raise


def test_apf_calculator():
    """测试APF奖励计算器"""
    print("\n测试5：APF奖励计算...")
    try:
        env = CppEnv()
        obs, info = env.reset(seed=42)
        
        # 执行一步获取奖励
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"✅ APF奖励计算执行成功")
        print(f"  - 奖励值: {reward:.4f}")
        
        # 检查obs_apf是否正确生成
        assert env.obs_apf is not None, "obs_apf未生成"
        print(f"  - obs_apf形状: {env.obs_apf.shape}")
        
        # 验证APF地图包含正确的内容
        assert env.obs_apf.shape[0] >= 4, "APF地图通道数不足"
        
        # 检查APF场是否有渐变值（表示APF转换生效）
        if env.use_apf:
            unique_vals = np.unique(env.obs_apf[0])  # field_obs
            if len(unique_vals) > 2:
                print(f"  - APF场包含渐变值（{len(unique_vals)}个不同值）")
            else:
                print(f"  - ⚠️ APF场可能未正确转换")
        
    except Exception as e:
        print(f"❌ APF奖励计算失败: {e}")
        raise


def test_parameter_names():
    """测试参数命名是否正确"""
    print("\n测试6：参数命名检查...")
    try:
        import inspect
        
        # 检查get_discounted_apf的参数名
        sig = inspect.signature(CppEnv.get_discounted_apf)
        params = list(sig.parameters.keys())
        
        expected_params = ['binary_map', 'propagate_distance', 'eps', 'pad']
        for param in expected_params:
            assert param in params, f"参数 {param} 不存在"
        
        print("✅ 参数命名正确")
        print(f"  - get_discounted_apf参数: {params}")
        
    except Exception as e:
        print(f"❌ 参数命名检查失败: {e}")
        raise


def main():
    """运行所有测试"""
    print("=" * 60)
    print("v2环境重构验证测试")
    print("=" * 60)
    
    try:
        # 测试基本功能
        env = test_env_creation()
        obs = test_env_reset(env)
        test_observation_maps(env)
        test_env_step(env)
        env.close()
        
        # 测试APF相关功能
        test_apf_calculator()
        test_parameter_names()
        
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！重构成功！")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ 测试失败，请检查重构代码")
        print("=" * 60)
        raise


if __name__ == "__main__":
    main()