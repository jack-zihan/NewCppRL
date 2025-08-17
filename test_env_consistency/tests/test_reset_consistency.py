"""
Reset函数一致性测试脚本
用于验证新旧环境reset后的状态一致性
"""
import sys
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from envs.cpp_env_base_copy import CppEnvBase as OldEnv
from envs_new.cpp_env_base import CppEnvBase as NewEnv


def test_initial_step_value():
    """测试初始步数值"""
    print("\n=== 测试初始步数值 ===")
    
    # 创建环境
    old_env = OldEnv(action_type='discrete')
    new_env = NewEnv(action_type='discrete')
    
    # Reset
    old_env.reset(seed=42)
    new_env.reset(seed=42)
    
    # 检查步数
    old_step = old_env.t
    new_step = new_env.env_state.current_step
    
    print(f"旧环境 self.t = {old_step}")
    print(f"新环境 current_step = {new_step}")
    
    if old_step != new_step:
        print(f"❌ 步数不一致: 旧环境={old_step}, 新环境={new_step}")
        return False
    else:
        print("✅ 步数一致")
        return True


def test_initial_state_variables():
    """测试所有状态变量的初始值"""
    print("\n=== 测试状态变量初始值 ===")
    
    old_env = OldEnv(action_type='discrete')
    new_env = NewEnv(action_type='discrete')
    
    old_env.reset(seed=42)
    new_env.reset(seed=42)
    
    # 检查各种状态变量
    checks = []
    
    # 1. 步数
    old_val = old_env.t
    new_val = new_env.env_state.current_step
    status = "✅" if old_val == new_val else "❌"
    print(f"{status} t/current_step: 旧={old_val}, 新={new_val}")
    checks.append(old_val == new_val)
    
    # 2. 转向值
    old_val = old_env.steer_t
    new_val = new_env.agent.last_steer
    status = "✅" if old_val == new_val else "❌"
    print(f"{status} steer_t/last_steer: 旧={old_val}, 新={new_val}")
    checks.append(old_val == new_val)
    
    # 3. 杂草数量
    old_val = old_env.weed_num_t
    new_val = new_env.env_state.weed_count if hasattr(new_env.env_state, 'weed_count') else new_env.env_state.get_info('weed_count').current
    status = "✅" if abs(old_val - new_val) < 5 else "❌"  # 允许小差异（随机性）
    print(f"{status} weed_num_t/weed_count: 旧={old_val}, 新={new_val}")
    checks.append(abs(old_val - new_val) < 5)
    
    # 4. frontier面积
    old_val = old_env.frontier_area_t
    new_val = new_env.env_state.frontier_area if hasattr(new_env.env_state, 'frontier_area') else new_env.env_state.get_info('frontier_area').current
    status = "✅" if abs(old_val - new_val) < 100 else "❌"  # 允许小差异
    print(f"{status} frontier_area: 旧={old_val}, 新={new_val}")
    checks.append(abs(old_val - new_val) < 100)
    
    # 5. Agent位置
    old_pos = old_env.agent.position
    new_pos = new_env.agent.position
    pos_diff = np.linalg.norm(np.array(old_pos) - np.array(new_pos))
    status = "✅" if pos_diff < 1.0 else "❌"
    print(f"{status} agent.position: 旧={old_pos}, 新={new_pos}, 差异={pos_diff:.2f}")
    checks.append(pos_diff < 1.0)
    
    # 6. Agent方向
    old_dir = old_env.agent.direction
    new_dir = new_env.agent.direction
    dir_diff = abs(old_dir - new_dir)
    status = "✅" if dir_diff < 1.0 else "❌"
    print(f"{status} agent.direction: 旧={old_dir:.2f}, 新={new_dir:.2f}, 差异={dir_diff:.2f}")
    checks.append(dir_diff < 1.0)
    
    return all(checks)


def test_map_initialization():
    """测试地图初始化状态"""
    print("\n=== 测试地图初始化 ===")
    
    old_env = OldEnv(action_type='discrete', use_traj=True)
    new_env = NewEnv(action_type='discrete', use_traj=True)
    
    old_env.reset(seed=42)
    new_env.reset(seed=42)
    
    checks = []
    
    # 1. Trajectory地图
    old_traj = old_env.map_trajectory
    new_traj = new_env.maps_dict.get('trajectory', None)
    
    if new_traj is not None:
        traj_sum_old = old_traj.sum()
        traj_sum_new = new_traj.sum()
        status = "✅" if traj_sum_old == traj_sum_new == 0 else "❌"
        print(f"{status} trajectory初始化: 旧sum={traj_sum_old}, 新sum={traj_sum_new}")
        checks.append(traj_sum_old == traj_sum_new == 0)
    else:
        print("⚠️ 新环境没有trajectory地图")
        checks.append(False)
    
    # 2. Mist地图（如果存在）
    if hasattr(old_env, 'map_mist'):
        old_mist = old_env.map_mist
        new_mist = new_env.maps_dict.get('mist', None)
        
        if new_mist is not None:
            # 注意：旧环境初始化为0，新环境可能初始化为1
            old_init_val = 0 if old_mist.sum() == 0 else 1
            new_init_val = 1 if new_mist.sum() > 0 else 0
            print(f"ℹ️ mist初始化值: 旧={old_init_val}, 新={new_init_val}")
            # 这是已知差异，不作为错误
    
    # 3. 检查初始时agent位置的杂草是否被清除
    agent_pos = (int(old_env.agent.position[1]), int(old_env.agent.position[0]))
    
    # 获取agent周围5x5区域的杂草
    y, x = agent_pos
    y_start, y_end = max(0, y-2), min(old_env.map_weed.shape[0], y+3)
    x_start, x_end = max(0, x-2), min(old_env.map_weed.shape[1], x+3)
    
    old_weed_area = old_env.map_weed[y_start:y_end, x_start:x_end].sum()
    new_weed_area = new_env.maps_dict['weed'][y_start:y_end, x_start:x_end].sum()
    
    status = "✅" if old_weed_area == new_weed_area == 0 else "⚠️"
    print(f"{status} agent位置杂草清除: 旧={old_weed_area}, 新={new_weed_area}")
    
    return all(checks) if checks else True


def test_observation_shape():
    """测试观察空间形状"""
    print("\n=== 测试观察空间 ===")
    
    old_env = OldEnv(action_type='discrete', use_sgcnn=True)
    new_env = NewEnv(action_type='discrete', use_sgcnn=True)
    
    old_obs, _ = old_env.reset(seed=42)
    new_obs, _ = new_env.reset(seed=42)
    
    # 检查观察形状
    old_shape = old_obs['observation'].shape
    new_shape = new_obs['observation'].shape
    
    status = "✅" if old_shape == new_shape else "❌"
    print(f"{status} observation形状: 旧={old_shape}, 新={new_shape}")
    
    # 检查其他组件
    for key in ['vector', 'weed_ratio']:
        if key in old_obs and key in new_obs:
            old_val = old_obs[key]
            new_val = new_obs[key]
            status = "✅" if old_val.shape == new_val.shape else "❌"
            print(f"{status} {key}形状: 旧={old_val.shape}, 新={new_val.shape}")
    
    return old_shape == new_shape


def main():
    """运行所有测试"""
    print("=" * 50)
    print("Reset函数一致性测试")
    print("=" * 50)
    
    results = []
    
    # 运行各项测试
    results.append(("初始步数值", test_initial_step_value()))
    results.append(("状态变量", test_initial_state_variables()))
    results.append(("地图初始化", test_map_initialization()))
    results.append(("观察空间", test_observation_shape()))
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    
    for test_name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name}: {status}")
    
    total = len(results)
    passed_count = sum(1 for _, passed in results if passed)
    
    print(f"\n总计: {passed_count}/{total} 测试通过")
    
    if passed_count < total:
        print("\n⚠️ 发现不一致性，请查看上述详细信息")
        print("\n关键问题:")
        print("1. current_step初始值差异 (0 vs 1)")
        print("2. mist地图初始化差异 (zeros vs ones)")
        print("3. 状态变量初始化时机差异")
    else:
        print("\n✅ 所有测试通过，reset函数行为一致")


if __name__ == "__main__":
    main()