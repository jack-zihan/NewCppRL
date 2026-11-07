#!/usr/bin/env python
"""
5算法快速测试：BCP, JUMP, SNAKE, R_SNAKE, REACT
每个算法运行1000步，保存最终帧图片
"""
import sys
sys.path.insert(0, '/home/lzh/NewCppRL')

import gymnasium as gym
import numpy as np
import cv2
import envs_new  # Register environments

def test_algorithm(alg_name, controller_class, max_steps=1000):
    """测试单个算法"""
    print(f"\n{'='*60}")
    print(f"测试 {alg_name}")
    print('='*60)

    # 创建环境 - v2带杂草
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

    # 创建控制器
    controller = controller_class()

    # 重置环境
    obs, info = env.reset(seed=42, options={
        'weed_dist': 'gaussian',
        'map_id': 2,
        'weed_num': 50
    })
    controller.reset(env)

    collision_count = 0
    final_completion = 0.0

    for step in range(max_steps):
        # 获取动作
        action = controller.act(obs, env)

        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)

        # 获取completion_ratio
        if isinstance(obs, dict) and 'completion_ratio' in obs:
            final_completion = obs['completion_ratio'][0] * 100

        # 检查碰撞
        if info.get('collision', False):
            collision_count += 1

        # 每200步打印
        if step % 200 == 0:
            print(f"  Step {step}: completion={final_completion:.1f}%")

        if terminated or truncated:
            print(f"  终止 @ step {step}")
            break

    # 保存最终帧
    frame = env.render()
    if frame is not None:
        output_path = f'/tmp/test_{alg_name.lower()}_final.png'
        cv2.imwrite(output_path, frame[:, :, ::-1])
        print(f"  保存图片: {output_path}")

    env.close()

    result = {
        'name': alg_name,
        'completion': final_completion,
        'collisions': collision_count,
        'steps': step + 1
    }
    print(f"  结果: completion={final_completion:.1f}%, collisions={collision_count}")
    return result


def main():
    from rules_new.policies_rules import (
        BCPController,
        JumpController,
        SnakeController,
        RSnakeController,
        ReactController
    )

    algorithms = [
        ('BCP', BCPController),
        ('JUMP', JumpController),
        ('SNAKE', SnakeController),
        ('R_SNAKE', RSnakeController),
        ('REACT', ReactController),
    ]

    results = []
    for alg_name, controller_class in algorithms:
        try:
            result = test_algorithm(alg_name, controller_class)
            results.append(result)
        except Exception as e:
            print(f"  ❌ {alg_name} 失败: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'name': alg_name,
                'completion': 0,
                'collisions': -1,
                'steps': 0,
                'error': str(e)
            })

    # 打印汇总
    print("\n" + "="*60)
    print("5算法测试汇总")
    print("="*60)
    print(f"{'算法':<12} {'完成率':<12} {'碰撞次数':<12} {'步数':<10}")
    print("-"*60)
    for r in results:
        if 'error' in r:
            print(f"{r['name']:<12} {'ERROR':<12} {'-':<12} {'-':<10}")
        else:
            print(f"{r['name']:<12} {r['completion']:.1f}%{'':<6} {r['collisions']:<12} {r['steps']:<10}")

    print("\n✅ 所有测试完成")
    print("图片保存在 /tmp/test_*_final.png")

if __name__ == "__main__":
    main()
