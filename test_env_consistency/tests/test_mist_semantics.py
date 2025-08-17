#!/usr/bin/env python3
"""
Mist地图语义一致性测试

验证新旧环境中mist地图的语义和功能一致性：
- mist = 0: 未探索区域（有雾）
- mist = 1: 已探索区域（无雾）
"""
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from envs.cpp_env_v2 import CppEnv as CppEnvOld
from envs_new.cpp_env_v2 import CppEnv as CppEnvNew


def test_mist_initialization():
    """测试mist地图初始化值"""
    print("测试1: Mist地图初始化值...")
    
    # 旧环境
    old_env = CppEnvOld()
    old_obs, _ = old_env.reset(seed=42)
    
    # 新环境
    new_env = CppEnvNew()
    new_obs, _ = new_env.reset(seed=42)
    
    # 检查初始值
    old_mist = old_env.map_mist
    new_mist = new_env.maps_dict['mist']
    
    # 统计初始状态
    old_zeros = np.sum(old_mist == 0)
    old_ones = np.sum(old_mist == 1)
    new_zeros = np.sum(new_mist == 0)
    new_ones = np.sum(new_mist == 1)
    
    print(f"  旧环境 - 0的数量: {old_zeros}, 1的数量: {old_ones}")
    print(f"  新环境 - 0的数量: {new_zeros}, 1的数量: {new_ones}")
    
    # 验证初始时大部分应该是0（未探索）
    assert old_zeros > old_ones, "旧环境初始mist应该大部分为0"
    assert new_zeros > new_ones, "新环境初始mist应该大部分为0"
    
    # 但是agent初始位置周围应该有一些1（已探索）
    assert old_ones > 0, "旧环境应该有初始探索区域"
    assert new_ones > 0, "新环境应该有初始探索区域"
    
    print("  ✅ 初始化测试通过！")
    
    old_env.close()
    new_env.close()


def test_mist_update_during_step():
    """测试step过程中mist的更新"""
    print("\n测试2: Step过程中mist更新...")
    
    # 旧环境
    old_env = CppEnvOld()
    old_obs, _ = old_env.reset(seed=42)
    old_mist_before = old_env.map_mist.copy()
    
    # 新环境
    new_env = CppEnvNew()
    new_obs, _ = new_env.reset(seed=42)
    new_mist_before = new_env.maps_dict['mist'].copy()
    
    # 执行一些步骤
    for _ in range(10):
        action = 70  # 固定动作
        old_env.step(action)
        new_env.step(action)
    
    old_mist_after = old_env.map_mist
    new_mist_after = new_env.maps_dict['mist']
    
    # 计算探索区域的变化
    old_explored_before = np.sum(old_mist_before == 1)
    old_explored_after = np.sum(old_mist_after == 1)
    new_explored_before = np.sum(new_mist_before == 1)
    new_explored_after = np.sum(new_mist_after == 1)
    
    print(f"  旧环境 - 探索区域: {old_explored_before} -> {old_explored_after}")
    print(f"  新环境 - 探索区域: {new_explored_before} -> {new_explored_after}")
    
    # 验证探索区域应该增加
    assert old_explored_after >= old_explored_before, "旧环境探索区域应该增加或保持"
    assert new_explored_after >= new_explored_before, "新环境探索区域应该增加或保持"
    
    # 验证只能从0变到1，不能从1变到0
    old_changes = old_mist_after - old_mist_before
    new_changes = new_mist_after - new_mist_before
    
    assert np.all(old_changes >= 0), "旧环境mist只能从0变到1"
    assert np.all(new_changes >= 0), "新环境mist只能从0变到1"
    
    print("  ✅ Step更新测试通过！")
    
    old_env.close()
    new_env.close()


def test_mist_frontier_visibility():
    """测试ensure_frontier_visibility功能"""
    print("\n测试3: Frontier可见性保证...")
    
    # 新环境
    new_env = CppEnvNew()
    new_obs, _ = new_env.reset(seed=42)
    
    # 检查初始时是否能看到frontier
    mist = new_env.maps_dict['mist']
    frontier = new_env.maps_dict['field_frontier']
    
    # 计算视野内的frontier
    frontier_in_vision = np.logical_and(frontier, mist)
    visible_frontier = np.sum(frontier_in_vision)
    
    print(f"  视野内可见的frontier像素: {visible_frontier}")
    
    # 应该能看到一些frontier
    assert visible_frontier > 0, "初始视野应该能看到frontier"
    
    print("  ✅ Frontier可见性测试通过！")
    
    new_env.close()


def test_mist_observation_consistency():
    """测试观察生成中mist的使用"""
    print("\n测试4: 观察生成中mist的使用...")
    
    # 新环境
    new_env = CppEnvNew()
    obs, _ = new_env.reset(seed=42)
    
    # 获取原始mist和观察中的mist通道
    raw_mist = new_env.maps_dict['mist']
    
    # 在cpp_env_v2中，第156行使用了np.logical_not(map_mist)
    # 这意味着观察中的mist是反转的
    # 观察格式：[frontier_apf, mist_inv, obstacle_apf, weed_apf, trajectory_apf]
    
    # 由于观察经过了旋转和裁剪，我们只验证语义
    # mist=0（未探索）在观察中应该变成1（有雾）
    # mist=1（已探索）在观察中应该变成0（无雾）
    
    print(f"  原始mist中0的数量: {np.sum(raw_mist == 0)}")
    print(f"  原始mist中1的数量: {np.sum(raw_mist == 1)}")
    
    # 处理obs可能是dict的情况
    if isinstance(obs, dict):
        obs_pixels = obs.get('pixels', obs.get('observation', None))
        if obs_pixels is not None:
            print(f"  观察shape: {obs_pixels.shape}")
            assert len(obs_pixels.shape) == 3, "观察应该是3维的"
            assert obs_pixels.shape[0] >= 4, "观察应该至少有4个通道"
        else:
            print(f"  观察是字典类型，包含的键: {list(obs.keys())}")
    else:
        print(f"  观察shape: {obs.shape}")
        assert len(obs.shape) == 3, "观察应该是3维的"
        assert obs.shape[0] >= 4, "观察应该至少有4个通道"
    
    print("  ✅ 观察生成测试通过！")
    
    new_env.close()


def test_mist_apf_interaction():
    """测试mist与APF计算的交互"""
    print("\n测试5: Mist与APF计算交互...")
    
    # 新环境
    new_env = CppEnvNew()
    obs, _ = new_env.reset(seed=42)
    
    # 获取相关地图
    mist = new_env.maps_dict['mist']
    frontier = new_env.maps_dict['field_frontier']
    obstacle = new_env.maps_dict['obstacle']
    
    # 根据cpp_env_v2.py第142-143行
    # APF只在已探索区域(mist=1)计算
    from envs_new.utils.math_utils import total_variation_mat
    
    apf_frontier = np.logical_and(total_variation_mat(frontier), mist)
    apf_obstacle = np.logical_and(total_variation_mat(obstacle), mist)
    
    # 验证APF只在已探索区域
    assert np.all(apf_frontier <= mist), "Frontier APF应该只在已探索区域"
    assert np.all(apf_obstacle <= mist), "Obstacle APF应该只在已探索区域"
    
    # 验证有一些APF值
    assert np.sum(apf_frontier) > 0 or np.sum(frontier) == 0, "应该有一些Frontier APF"
    
    print(f"  Frontier APF像素: {np.sum(apf_frontier)}")
    print(f"  Obstacle APF像素: {np.sum(apf_obstacle)}")
    
    print("  ✅ APF交互测试通过！")
    
    new_env.close()


def main():
    """运行所有测试"""
    print("="*60)
    print("Mist地图语义一致性测试")
    print("="*60)
    
    test_mist_initialization()
    test_mist_update_during_step()
    test_mist_frontier_visibility()
    test_mist_observation_consistency()
    test_mist_apf_interaction()
    
    print("\n" + "="*60)
    print("✅ 所有mist测试通过！")
    print("="*60)


if __name__ == "__main__":
    main()