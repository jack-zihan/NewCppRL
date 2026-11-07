#!/usr/bin/env python
"""
单独测试SNAKE算法的运行行为
"""
import sys
sys.path.insert(0, '/home/lzh/NewCppRL')

import gymnasium as gym
import numpy as np
import envs_new  # Register environments

def test_snake_single():
    """测试SNAKE单次运行"""
    from rules_new.policies_rules import SnakeController

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

    # 创建SNAKE控制器
    from rules_new.policies_rules import SnakeController
    controller = SnakeController()

    # 重置环境
    obs, info = env.reset(seed=25, options={
        'weed_dist': 'gaussian',
        'map_id': 2,
        'weed_num': 50
    })
    controller.reset(env)

    print("=== SNAKE单次运行测试 ===")
    print(f"Initial position: {controller._pose(env)}")

    max_steps = 1000
    collision_count = 0
    truncated_early = False

    for step in range(max_steps):
        # 获取动作
        action = controller.act(obs, env)

        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)

        # 每100步打印状态
        if step % 100 == 0:
            x, y = controller._pose(env)
            path_len = len(controller.path) if controller.path else 0
            idx = controller.idx
            mode = getattr(controller, '_mode', 'unknown')
            detour = getattr(controller, '_detour_active', False)
            # completion_ratio可能在obs或info中
            if isinstance(obs, dict) and 'completion_ratio' in obs:
                completion = obs['completion_ratio'][0] * 100
            else:
                completion = info.get('completion_ratio', 0) * 100
            # 检查weeds removed
            weeds_removed = info.get('weeds_removed', 0)
            total_weeds = info.get('total_weeds', '?')
            print(f"Step {step}: pos=({x:.1f},{y:.1f}), mode={mode}, path_len={path_len}, idx={idx}, "
                  f"detour={detour}, completion={completion:.1f}%, weeds={weeds_removed}/{total_weeds}")

        # 检查碰撞
        if info.get('collision', False):
            collision_count += 1
            if collision_count <= 5:
                x, y = controller._pose(env)
                print(f"  ! 碰撞 @ step {step}: pos=({x:.1f},{y:.1f})")

        if terminated or truncated:
            completion = info.get('completion_ratio', 0) * 100
            print(f"\n终止 @ step {step}: terminated={terminated}, truncated={truncated}")
            print(f"completion_ratio: {completion:.1f}%")
            print(f"collision_count: {collision_count}")
            truncated_early = True
            break

    if not truncated_early:
        completion = info.get('completion_ratio', 0) * 100
        print(f"\n达到最大步数 {max_steps}")
        print(f"completion_ratio: {completion:.1f}%")
        print(f"collision_count: {collision_count}")

    # 保存最终帧
    frame = env.render()
    if frame is not None:
        import cv2
        cv2.imwrite('/tmp/snake_test_final.png', frame[:, :, ::-1])
        print("保存最终帧到 /tmp/snake_test_final.png")

    env.close()
    print("\n✅ 测试完成")

if __name__ == "__main__":
    test_snake_single()
