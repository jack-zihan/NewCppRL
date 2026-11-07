"""
Dubins导航功能测试脚本
验证基础的Dubins路径规划和导航是否正确工作
"""
import sys
sys.path.insert(0, '/home/lzh/NewCppRL')

import math
import numpy as np
import dubins
import gymnasium as gym

# 触发环境注册
import envs_new

def test_dubins_library():
    """测试dubins库的基础功能"""
    print("=" * 60)
    print("测试1: Dubins库基础功能")
    print("=" * 60)

    # 测试参数
    turning_radius = 5.0

    # 测试案例：从原点向右出发，到目标点
    test_cases = [
        # (起点x, 起点y, 起点角度, 终点x, 终点y, 终点角度)
        (0, 0, 0, 20, 10, 0),           # 直行+转弯
        (0, 0, 0, 0, 20, math.pi/2),    # 90度转弯
        (0, 0, 0, 20, 0, 0),             # 纯直行
        (100, 100, 0, 150, 120, math.pi), # 从中间位置
    ]

    for i, (sx, sy, sth, ex, ey, eth) in enumerate(test_cases):
        print(f"\n案例 {i+1}: ({sx}, {sy}, {math.degrees(sth):.1f}°) -> ({ex}, {ey}, {math.degrees(eth):.1f}°)")

        try:
            q0 = (sx, sy, sth)
            q1 = (ex, ey, eth)
            path = dubins.shortest_path(q0, q1, turning_radius)
            configs, _ = path.sample_many(1.0)

            print(f"   路径长度: {path.path_length():.2f}")
            print(f"   采样点数: {len(configs)}")
            print(f"   起点: ({configs[0][0]:.2f}, {configs[0][1]:.2f})")
            print(f"   终点: ({configs[-1][0]:.2f}, {configs[-1][1]:.2f})")
            print(f"   ✅ 成功")
        except Exception as e:
            print(f"   ❌ 失败: {e}")

def test_env_creation():
    """测试环境创建"""
    print("\n" + "=" * 60)
    print("测试2: 环境创建和基础参数")
    print("=" * 60)

    try:
        env = gym.make(
            "NewPasture-v2",
            state_size=(256, 256),
            action_type="continuous",
            use_trajectory=True,
            use_box_boundary=True,
            use_apf=True,
            use_mist=True,
            multiscale_feature_size=32,
            num_obstacles_range=[0, 0],
            weed_num=50,
            weed_dist="gaussian",
            random_seed=25,
            map_id=2,
        )

        obs, info = env.reset()
        base = env.unwrapped

        print(f"环境创建成功!")
        print(f"v_max: {base.config.v_max}")
        print(f"w_max: {base.config.w_max}")
        print(f"agent_width: {getattr(base.config, 'agent_width', 'N/A')}")
        print(f"agent_vision_length: {getattr(base.config, 'agent_vision_length', 'N/A')}")
        print(f"agent_vision_angle: {getattr(base.config, 'agent_vision_angle', 'N/A')}")

        # 计算转弯半径
        w_max_rad = abs(base.config.w_max) * math.pi / 180.0
        turning_radius = base.config.v_max / w_max_rad
        print(f"计算的转弯半径: {turning_radius:.2f}")

        # 获取boundingbox
        bbox = base.env_state.get_static_info("bounding_box")
        if bbox:
            print(f"Bounding box: {bbox[0]}")

        # 获取agent初始位置
        print(f"Agent位置: ({base.agent.x}, {base.agent.y})")
        print(f"Agent方向: {base.agent.direction}")

        # 检查maps
        print(f"\nmaps_dict keys: {list(base.maps_dict.keys())}")
        for key, val in base.maps_dict.items():
            print(f"  {key}: shape={val.shape}, sum={val.sum()}")

        env.close()
        print("✅ 环境测试通过")
        return True

    except Exception as e:
        print(f"❌ 环境创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dubins_in_env():
    """测试在环境中使用Dubins导航"""
    print("\n" + "=" * 60)
    print("测试3: 环境中的Dubins导航")
    print("=" * 60)

    try:
        env = gym.make(
            "NewPasture-v2",
            state_size=(256, 256),
            action_type="continuous",
            use_trajectory=True,
            use_box_boundary=True,
            use_apf=True,
            use_mist=True,
            multiscale_feature_size=32,
            num_obstacles_range=[0, 0],
            weed_num=50,
            weed_dist="gaussian",
            random_seed=25,
            map_id=2,
        )

        obs, info = env.reset()
        base = env.unwrapped

        # 获取转弯半径
        w_max_rad = abs(base.config.w_max) * math.pi / 180.0
        v_max = base.config.v_max
        turning_radius = v_max / w_max_rad

        # 当前位置
        x, y = base.agent.x, base.agent.y
        dir_deg = base.agent.direction

        print(f"起始位置: ({x:.2f}, {y:.2f}), 方向: {dir_deg:.2f}°")

        # 选择一个目标点（向前50像素）
        target_x = x + 50 * math.cos(math.radians(dir_deg))
        target_y = y + 50 * math.sin(math.radians(dir_deg))
        target_dir = dir_deg

        print(f"目标位置: ({target_x:.2f}, {target_y:.2f}), 方向: {target_dir:.2f}°")

        # 转换角度到dubins格式
        def env_deg_to_dubins(theta_deg):
            return -math.radians(float(theta_deg))

        q0 = (x, y, env_deg_to_dubins(dir_deg))
        q1 = (target_x, target_y, env_deg_to_dubins(target_dir))

        path = dubins.shortest_path(q0, q1, turning_radius)
        configs, _ = path.sample_many(1.0)

        print(f"Dubins路径长度: {path.path_length():.2f}")
        print(f"采样点数: {len(configs)}")

        # 模拟沿路径导航
        print("\n开始导航测试（跟踪前10个点）:")
        positions = [(x, y)]

        for i, (px, py, pth) in enumerate(configs[1:11]):
            # 计算到目标点的动作
            dx = px - base.agent.x
            dy = py - base.agent.y
            dist = math.hypot(dx, dy)

            if dist > 0.1:
                goal_dir = math.degrees(math.atan2(dy, dx))
                delta = (goal_dir - base.agent.direction + 180) % 360 - 180

                # 限制角速度
                w_bounds = base.action_processor.get_action_bounds()[1]
                steer = max(w_bounds[0], min(delta, w_bounds[1]))
                speed = min(v_max, dist)

                obs, reward, done, truncated, info = env.step([speed, steer])
                positions.append((base.agent.x, base.agent.y))

                if i < 5:
                    print(f"  步{i+1}: 目标({px:.1f},{py:.1f}) -> 实际({base.agent.x:.1f},{base.agent.y:.1f})")

        print(f"\n总移动距离: {sum(math.hypot(positions[i+1][0]-positions[i][0], positions[i+1][1]-positions[i][1]) for i in range(len(positions)-1)):.2f}")
        print("✅ Dubins导航测试完成")

        env.close()
        return True

    except Exception as e:
        print(f"❌ Dubins导航测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_weed_discovery():
    """测试杂草发现机制"""
    print("\n" + "=" * 60)
    print("测试4: 杂草发现机制")
    print("=" * 60)

    try:
        env = gym.make(
            "NewPasture-v2",
            state_size=(256, 256),
            action_type="continuous",
            use_trajectory=True,
            use_box_boundary=True,
            use_apf=True,
            use_mist=True,
            multiscale_feature_size=32,
            num_obstacles_range=[0, 0],
            weed_num=50,
            weed_dist="gaussian",
            random_seed=25,
            map_id=2,
        )

        obs, info = env.reset()
        base = env.unwrapped

        weed_map = base.maps_dict.get("weed")
        mist_map = base.maps_dict.get("mist")

        if weed_map is None or mist_map is None:
            print("❌ weed或mist地图不存在！")
            return False

        total_weeds = weed_map.sum()

        # 初始状态
        discovered_initial = (weed_map > 0) & (mist_map > 0)
        print(f"总杂草数: {total_weeds}")
        print(f"初始已发现: {discovered_initial.sum()}")

        # 走几步看看能否发现更多
        v_max = base.config.v_max
        for i in range(10):
            obs, reward, done, truncated, info = env.step([v_max, 0])
            if done:
                break

        discovered_after = (base.maps_dict["weed"] > 0) & (base.maps_dict["mist"] > 0)
        print(f"走动后已发现: {discovered_after.sum()}")
        print(f"mist覆盖面积: {(base.maps_dict['mist'] > 0).sum()}")

        env.close()
        print("✅ 杂草发现机制测试完成")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Dubins导航功能测试\n")

    test_dubins_library()

    if test_env_creation():
        test_dubins_in_env()
        test_weed_discovery()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
