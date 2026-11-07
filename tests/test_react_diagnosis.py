#!/usr/bin/env python
"""REACT算法详细诊断测试"""
import sys
sys.path.insert(0, '/home/lzh/NewCppRL')

import gymnasium as gym
import numpy as np
import cv2
import envs_new

def test_react_diagnosis():
    """诊断REACT为什么只有4%完成率"""
    print("="*60)
    print("REACT 详细诊断测试")
    print("="*60)

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

    from rules_new.policies_rules import ReactController
    controller = ReactController()

    obs, info = env.reset(seed=42, options={
        'weed_dist': 'gaussian',
        'map_id': 2,
        'weed_num': 50
    })
    controller.reset(env)

    base = env.unwrapped
    maps = base.maps_dict

    # 统计初始状态
    field_mask = (maps['field'] > 0).astype(np.uint8)
    field_pixels = np.sum(field_mask)
    mist = maps.get('mist')
    weed = maps.get('weed')

    print(f"\n初始状态：")
    print(f"  - 田地像素总数: {field_pixels}")
    if mist is not None:
        explored_initial = np.sum((mist > 0) & (field_mask > 0))
        print(f"  - 初始已探索像素: {explored_initial} ({100*explored_initial/field_pixels:.1f}%)")
    if weed is not None:
        weed_count = np.sum(weed > 0)
        print(f"  - 杂草像素总数: {weed_count}")

    # 记录历史
    mode_history = []
    target_history = []
    pos_history = []
    path_len_history = []

    max_steps = 500
    for step in range(max_steps):
        # 获取动作前记录状态
        old_mode = controller.mode
        old_target = controller.target
        old_path_len = len(controller.path)

        action = controller.act(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)

        # 记录
        mode_history.append(controller.mode)
        target_history.append(controller.target)
        pos_history.append(controller._pose(env))
        path_len_history.append(len(controller.path))

        # 关键时刻打印
        if step < 20 or step % 100 == 0 or controller.mode != old_mode:
            x, y = controller._pose(env)
            weeds = controller._get_discovered_weeds_world(env)
            print(f"Step {step}: pos=({x:.1f},{y:.1f}), mode={controller.mode}, "
                  f"path_len={len(controller.path)}, idx={controller.idx}, "
                  f"target={controller.target}, discovered_weeds={weeds.shape[0]}")

        if terminated or truncated:
            print(f"\n终止 @ step {step}")
            break

    # 统计分析
    print("\n" + "="*60)
    print("行为统计分析")
    print("="*60)

    # 模式分布
    explore_count = sum(1 for m in mode_history if m == 'explore')
    weed_count = sum(1 for m in mode_history if m == 'weed')
    print(f"\n模式分布：")
    print(f"  - explore模式: {explore_count} 步 ({100*explore_count/len(mode_history):.1f}%)")
    print(f"  - weed模式: {weed_count} 步 ({100*weed_count/len(mode_history):.1f}%)")

    # 目标变化次数
    target_changes = sum(1 for i in range(1, len(target_history))
                        if target_history[i] != target_history[i-1])
    print(f"\n目标变化次数: {target_changes}")

    # 移动距离
    total_dist = 0
    for i in range(1, len(pos_history)):
        dx = pos_history[i][0] - pos_history[i-1][0]
        dy = pos_history[i][1] - pos_history[i-1][1]
        total_dist += np.sqrt(dx*dx + dy*dy)
    print(f"总移动距离: {total_dist:.1f} 像素")

    # 覆盖范围
    xs = [p[0] for p in pos_history]
    ys = [p[1] for p in pos_history]
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)
    print(f"X范围: [{min(xs):.1f}, {max(xs):.1f}], 跨度={x_range:.1f}")
    print(f"Y范围: [{min(ys):.1f}, {max(ys):.1f}], 跨度={y_range:.1f}")

    # 最终完成率
    if isinstance(obs, dict) and 'completion_ratio' in obs:
        completion = obs['completion_ratio'][0] * 100
    else:
        completion = info.get('completion_ratio', 0) * 100
    print(f"\n最终完成率: {completion:.1f}%")

    # 分析mist探索情况
    mist_final = maps.get('mist')
    if mist_final is not None:
        explored_final = np.sum((mist_final > 0) & (field_mask > 0))
        print(f"最终已探索像素: {explored_final} ({100*explored_final/field_pixels:.1f}%)")

    # 保存图片
    frame = env.render()
    if frame is not None:
        cv2.imwrite('/tmp/react_diagnosis.png', frame[:, :, ::-1])
        print("\n保存图片到 /tmp/react_diagnosis.png")

    env.close()

if __name__ == "__main__":
    test_react_diagnosis()
