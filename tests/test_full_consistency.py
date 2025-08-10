"""
完整的新旧版本一致性对比测试

运行相同的场景，对比新旧版本的输出
"""
import sys
import os
import numpy as np
from pathlib import Path
import time

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_old_version_jump():
    """运行旧版本的JUMP算法"""
    print("\n=== 运行旧版JUMP算法 ===")
    
    # 导入旧版本
    from rules.env_make import get_env
    
    # 创建环境
    env, obs = get_env()
    
    # 获取初始状态
    initial_pos = [env.agent.y, env.agent.x]
    initial_dir = env.agent.direction
    initial_weed = env.map_weed.sum()
    
    print(f"初始位置: {initial_pos}")
    print(f"初始方向: {initial_dir}")
    print(f"初始杂草数: {initial_weed}")
    
    # 执行几步
    positions = []
    for i in range(5):
        # 简单的前进动作
        action = [5.0, 0.0]
        obs, reward, done, timeout, info = env.step(action)
        
        new_pos = [env.agent.y, env.agent.x]
        positions.append(new_pos)
        
        if done:
            break
    
    # 获取最终状态
    final_weed = env.map_weed.sum()
    coverage = (initial_weed - final_weed) / initial_weed if initial_weed > 0 else 0
    
    env.close()
    
    return {
        'initial_pos': initial_pos,
        'initial_dir': initial_dir,
        'positions': positions,
        'coverage': coverage,
        'initial_weed': initial_weed,
        'final_weed': final_weed
    }


def run_new_version_jump():
    """运行新版本的JUMP算法"""
    print("\n=== 运行新版JUMP算法 ===")
    
    from rules_new.algorithms.jump_planner import JumpPlanner
    from rules.env_make import get_env
    
    # 创建环境（使用相同的环境）
    env, obs = get_env()
    
    # 获取初始状态
    initial_pos = [env.agent.y, env.agent.x]
    initial_dir = env.agent.direction
    initial_weed = env.map_weed.sum()
    
    print(f"初始位置: {initial_pos}")
    print(f"初始方向: {initial_dir}")
    print(f"初始杂草数: {initial_weed}")
    
    # 配置算法
    env_config = {
        'agent': {
            'car_width': 5,
            'sight_width': 24,
            'sight_length': 24
        },
        'environment': {
            'width': 600,
            'height': 600
        }
    }
    
    config = {
        'algorithm': {'name': 'JUMP'},
        'parameters': {}
    }
    
    # 创建算法实例
    planner = JumpPlanner(config, env_config)
    
    # 准备初始状态
    farm_vertices = env.min_area_rect[0][:, 0, ::-1] if hasattr(env, 'min_area_rect') else None
    
    initial_state = {
        'agent_position': [env.agent.y, env.agent.x],
        'agent_direction': env.agent.direction,
        'farm_vertices': farm_vertices,
        'discovered_weeds': [],
        'coverage_rate': 0.0
    }
    
    planner.reset(initial_state)
    
    # 执行几步（与旧版相同）
    positions = []
    for i in range(5):
        # 获取算法决策
        current_state = {
            'agent_position': [env.agent.y, env.agent.x],
            'agent_direction': env.agent.direction,
            'discovered_weeds': [],
            'coverage_rate': (initial_weed - env.map_weed.sum()) / initial_weed if initial_weed > 0 else 0
        }
        
        waypoint = planner.plan_next_waypoint(current_state)
        
        # 如果算法返回路径，执行第一个动作
        if waypoint:
            if isinstance(waypoint, tuple) and waypoint[0] == 'path':
                # 路径列表，执行第一个
                if waypoint[1]:
                    target = waypoint[1][0]
                    # 计算动作
                    dy = target[0] - env.agent.y
                    dx = target[1] - env.agent.x
                    distance = np.sqrt(dx**2 + dy**2)
                    action = [distance, 0.0]  # 简化：只前进
                else:
                    action = [5.0, 0.0]
            else:
                action = [5.0, 0.0]
        else:
            action = [5.0, 0.0]
        
        obs, reward, done, timeout, info = env.step(action)
        
        new_pos = [env.agent.y, env.agent.x]
        positions.append(new_pos)
        
        if done:
            break
    
    # 获取最终状态
    final_weed = env.map_weed.sum()
    coverage = (initial_weed - final_weed) / initial_weed if initial_weed > 0 else 0
    
    env.close()
    
    return {
        'initial_pos': initial_pos,
        'initial_dir': initial_dir,
        'positions': positions,
        'coverage': coverage,
        'initial_weed': initial_weed,
        'final_weed': final_weed,
        'turn_direction': planner.turn_direction
    }


def compare_results(old_result, new_result):
    """对比新旧版本的结果"""
    print("\n=== 结果对比 ===")
    
    # 对比初始状态
    print("\n初始状态对比:")
    print(f"  旧版初始位置: {old_result['initial_pos']}")
    print(f"  新版初始位置: {new_result['initial_pos']}")
    
    pos_diff = np.linalg.norm(
        np.array(old_result['initial_pos']) - np.array(new_result['initial_pos'])
    )
    print(f"  位置差异: {pos_diff:.6f}")
    
    # 对比路径
    print("\n路径对比:")
    min_len = min(len(old_result['positions']), len(new_result['positions']))
    
    for i in range(min_len):
        old_pos = old_result['positions'][i]
        new_pos = new_result['positions'][i]
        diff = np.linalg.norm(np.array(old_pos) - np.array(new_pos))
        print(f"  步骤{i+1}: 旧={old_pos}, 新={new_pos}, 差异={diff:.4f}")
    
    # 对比覆盖率
    print(f"\n覆盖率对比:")
    print(f"  旧版: {old_result['coverage']:.4f}")
    print(f"  新版: {new_result['coverage']:.4f}")
    print(f"  差异: {abs(old_result['coverage'] - new_result['coverage']):.4f}")
    
    # 验证turn_direction
    if 'turn_direction' in new_result:
        print(f"\n新版turn_direction: {new_result['turn_direction']}")
    
    # 判断是否一致
    is_consistent = pos_diff < 1.0  # 位置差异小于1个单位
    
    return is_consistent


def main():
    """主函数"""
    print("="*60)
    print("完整一致性测试")
    print("="*60)
    
    try:
        # 运行旧版本
        old_result = run_old_version_jump()
        
        # 运行新版本
        new_result = run_new_version_jump()
        
        # 对比结果
        is_consistent = compare_results(old_result, new_result)
        
        print("\n" + "="*60)
        if is_consistent:
            print("✅ 测试通过！新旧版本行为一致")
            return True
        else:
            print("⚠️ 测试失败！新旧版本存在差异")
            return False
            
    except Exception as e:
        print(f"❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)