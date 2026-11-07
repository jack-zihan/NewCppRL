#!/usr/bin/env python
"""REACT theta跟踪修复验证测试

验证目标：
1. _plan_dubins_path 返回正确的 (x, y, theta) 三元组
2. path_theta 被正确填充
3. _compute_action_with_theta 使用正确的切线方向
4. 消除"月牙形"轨迹（弦方向≠切线方向导致的问题）
"""
import sys
sys.path.insert(0, '/home/lzh/NewCppRL')

import gymnasium as gym
import numpy as np
import math
import envs_new

# 创建环境
env = gym.make('NewPasture-v2',
    state_size=(256, 256),
    state_downsize=(256, 256),
    action_type="continuous",
    use_trajectory=True,
    boundary_source="box",
    use_apf=True,
    use_mist=True,
    num_obstacles_range=(0, 0),
    render_mode='rgb_array'
)

from rules_new.policies_rules import ReactController, _plan_dubins_path, _env_deg_to_dubins, _dubins_to_env_deg

print("=" * 60)
print("测试1: 验证 _plan_dubins_path 返回三元组")
print("=" * 60)

# 测试Dubins路径规划
obs, info = env.reset(seed=42)
start_pose = (100.0, 100.0, 0.0)  # env坐标系：顺时针度数
end_pose = (150.0, 150.0, 90.0)

try:
    path_with_theta = _plan_dubins_path(env, start_pose, end_pose, step_size=2.0)
    print(f"✓ 返回路径长度: {len(path_with_theta)}")

    if len(path_with_theta) > 0:
        first = path_with_theta[0]
        last = path_with_theta[-1]
        print(f"✓ 第一个点: (x={first[0]:.2f}, y={first[1]:.2f}, theta={first[2]:.2f}°)")
        print(f"✓ 最后一个点: (x={last[0]:.2f}, y={last[1]:.2f}, theta={last[2]:.2f}°)")

        # 验证三元组格式
        assert len(first) == 3, "第一个点应该是三元组"
        assert len(last) == 3, "最后一个点应该是三元组"
        print("✓ 三元组格式正确")

        # 检查theta是否合理（应该在-180到180度之间）
        for i, (x, y, theta) in enumerate(path_with_theta[:5]):
            assert -180 <= theta <= 180, f"theta {theta} 超出范围"
        print("✓ theta值范围正确 (-180° ~ 180°)")
except Exception as e:
    print(f"✗ 错误: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("测试2: 验证 ReactController.path_theta 正确填充")
print("=" * 60)

controller = ReactController()
obs, info = env.reset(seed=42, options={
    'weed_dist': 'gaussian',
    'map_id': 2,
    'weed_num': 50
})
controller.reset(env)

# 执行一步以触发路径规划
action = controller.act(obs, env)

print(f"当前模式: {controller.mode}")
print(f"path 长度: {len(controller.path)}")
print(f"path_theta 长度: {len(controller.path_theta)}")

if len(controller.path) > 0 and len(controller.path_theta) > 0:
    assert len(controller.path) == len(controller.path_theta), \
        f"path和path_theta长度不匹配: {len(controller.path)} vs {len(controller.path_theta)}"
    print("✓ path 和 path_theta 长度匹配")

    print(f"\n前5个路径点:")
    for i in range(min(5, len(controller.path))):
        x, y = controller.path[i]
        theta = controller.path_theta[i]
        print(f"  [{i}] pos=({x:.2f}, {y:.2f}), theta={theta:.2f}°")
else:
    print("⚠ 路径为空，可能需要更多步骤来触发路径规划")

print()
print("=" * 60)
print("测试3: 验证弦方向 vs 切线方向差异")
print("=" * 60)

if len(controller.path) >= 2 and len(controller.path_theta) >= 2:
    print("比较弦方向(atan2)和切线方向(Dubins theta)的差异:")
    max_diff = 0
    for i in range(min(10, len(controller.path) - 1)):
        x1, y1 = controller.path[i]
        x2, y2 = controller.path[i + 1]

        # 弦方向（旧方法：atan2）
        chord_dir = math.degrees(math.atan2(y2 - y1, x2 - x1))

        # 切线方向（新方法：Dubins theta）
        tangent_dir = controller.path_theta[i]

        # 计算差值
        diff = abs(((tangent_dir - chord_dir + 180) % 360) - 180)
        max_diff = max(max_diff, diff)

        marker = "⚠" if diff > 10 else "✓"
        print(f"  [{i}] 弦方向={chord_dir:.1f}°, 切线方向={tangent_dir:.1f}°, 差值={diff:.1f}° {marker}")

    if max_diff > 20:
        print(f"\n⚠ 最大差值 {max_diff:.1f}° > 20°，这就是导致'月牙形'的原因！")
        print("  使用切线方向(Dubins theta)可以消除这个问题")
    else:
        print(f"\n✓ 最大差值 {max_diff:.1f}°，在可接受范围内")

print()
print("=" * 60)
print("测试4: 运行300步并检查轨迹平滑性")
print("=" * 60)

controller = ReactController()
obs, info = env.reset(seed=123, options={
    'weed_dist': 'gaussian',
    'map_id': 1,
    'weed_num': 30
})
controller.reset(env)

positions = []
headings = []
actions = []

for step in range(300):
    x, y = controller._pose(env)
    heading = controller._heading_deg(env)
    positions.append((x, y))
    headings.append(heading)

    action = controller.act(obs, env)
    actions.append(action)

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print(f"Episode结束于step {step}")
        break

# 分析轨迹平滑性
print(f"总步数: {len(positions)}")

# 检查是否有v=0的情况（纯转向）
v_zero_count = sum(1 for v, w in actions if abs(v) < 0.1)
print(f"v≈0的步数（纯转向）: {v_zero_count} ({100*v_zero_count/len(actions):.1f}%)")

# 检查角度跳变
angle_jumps = []
for i in range(1, len(headings)):
    diff = abs(((headings[i] - headings[i-1] + 180) % 360) - 180)
    if diff > 30:  # 大于30度的跳变
        angle_jumps.append((i, diff))

print(f"角度跳变(>30°)次数: {len(angle_jumps)}")
if angle_jumps[:5]:
    print(f"  前5个跳变: {angle_jumps[:5]}")

# 保存最终渲染
import cv2
frame = env.render()
if frame is not None:
    cv2.imwrite('/tmp/react_theta_test.png', frame[:, :, ::-1])
    print(f"\n渲染图保存到 /tmp/react_theta_test.png")

# 获取完成率
if isinstance(obs, dict) and 'completion_ratio' in obs:
    completion = obs['completion_ratio'][0] * 100
else:
    completion = info.get('completion_ratio', 0) * 100
print(f"最终完成率: {completion:.1f}%")

print()
print("=" * 60)
print("测试结论")
print("=" * 60)

if v_zero_count < len(actions) * 0.05:  # 小于5%
    print("✓ 很少出现v=0纯转向，修复有效！")
else:
    print("⚠ 仍有较多v=0纯转向，需要进一步检查")

if len(angle_jumps) < 10:
    print("✓ 角度跳变很少，轨迹平滑")
else:
    print("⚠ 仍有较多角度跳变，轨迹可能不平滑")

env.close()
print("\n测试完成")
