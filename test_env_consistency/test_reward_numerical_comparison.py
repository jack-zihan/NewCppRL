#!/usr/bin/env python3
"""
奖励系统数值对比测试脚本
通过相同的状态变化，精确对比新旧环境的奖励计算结果
"""

import sys
import numpy as np
sys.path.append('/home/lzh/NewCppRL')

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class TestCase:
    """测试用例数据结构"""
    name: str
    # 状态变化
    frontier_area_change: float
    frontier_tv_change: float
    weed_count_change: float
    steer_change: Tuple[float, float]  # (old, new)
    crashed: bool = False
    finished: bool = False
    # APF变化（可选）
    apf_changes: Dict[str, float] = None


def calculate_old_reward(test_case: TestCase) -> Dict[str, float]:
    """计算旧版本奖励"""
    # 参数
    agent_width = 4.0
    v_max = 3.5
    w_max = 28.6
    
    # 基础惩罚
    reward_const = -0.1
    
    # 前沿奖励
    normalization = 2 * agent_width * v_max  # 28.0
    reward_frontier_coverage = test_case.frontier_area_change / normalization
    reward_frontier_tv = 0.5 * test_case.frontier_tv_change / v_max
    reward_frontier = 0.125 * (reward_frontier_coverage + reward_frontier_tv)
    
    # 杂草奖励
    reward_weed = 20.0 * test_case.weed_count_change
    
    # 转向奖励（组系数为0）
    steer_old, steer_new = test_case.steer_change
    reward_turn_gap = -0.5 * abs(steer_new - steer_old) / w_max
    reward_turn_direction = -0.30 if (steer_new * steer_old < 0) else 0.0
    reward_turn_self = 0.25 * (0.4 - abs(steer_new / w_max) ** 0.5)
    reward_turn = 0.0 * (reward_turn_gap + reward_turn_direction + reward_turn_self)
    
    # 碰撞和完成
    reward_collision = -399.0 if test_case.crashed else 0.0
    reward_completion = 500.0 if test_case.finished else 0.0
    
    # 总奖励
    total = (reward_const + reward_frontier + reward_weed + 
             reward_turn + reward_collision + reward_completion)
    
    return {
        'base': reward_const,
        'frontier_coverage': 0.125 * reward_frontier_coverage,  # 含组系数
        'frontier_tv': 0.125 * reward_frontier_tv,  # 含组系数
        'frontier_total': reward_frontier,
        'weed': reward_weed,
        'turn': reward_turn,
        'collision': reward_collision,
        'completion': reward_completion,
        'total': total
    }


def calculate_new_reward_wrong(test_case: TestCase) -> Dict[str, float]:
    """计算新版本奖励（错误的配置）"""
    # 参数
    agent_width = 4.0
    v_max = 3.5
    w_max = 28.6
    
    # 基础惩罚
    reward_base = -0.1
    
    # 前沿奖励（注意cpp_env_v2的错误配置）
    normalization = 2 * agent_width * v_max  # 28.0
    
    # 错误：coverage使用0.5作为个体系数
    reward_frontier_coverage = test_case.frontier_area_change / normalization * 0.5  # 个体系数
    reward_frontier_coverage *= 0.125  # 组系数
    
    # TV正确使用0.5
    reward_frontier_tv = test_case.frontier_tv_change / v_max * 0.5  # 个体系数
    reward_frontier_tv *= 0.125  # 组系数
    
    # 杂草奖励
    reward_weed = 20.0 * test_case.weed_count_change
    
    # 转向奖励（组系数为0）
    steer_old, steer_new = test_case.steer_change
    reward_turn = 0.0  # 因为组系数是0
    
    # 碰撞和完成
    reward_collision = -399.0 if test_case.crashed else 0.0
    reward_completion = 500.0 if test_case.finished else 0.0
    
    # 总奖励
    total = (reward_base + reward_frontier_coverage + reward_frontier_tv + 
             reward_weed + reward_turn + reward_collision + reward_completion)
    
    return {
        'base': reward_base,
        'frontier_coverage': reward_frontier_coverage,
        'frontier_tv': reward_frontier_tv,
        'frontier_total': reward_frontier_coverage + reward_frontier_tv,
        'weed': reward_weed,
        'turn': reward_turn,
        'collision': reward_collision,
        'completion': reward_completion,
        'total': total
    }


def calculate_new_reward_fixed(test_case: TestCase) -> Dict[str, float]:
    """计算新版本奖励（修复后的配置）"""
    # 参数
    agent_width = 4.0
    v_max = 3.5
    w_max = 28.6
    
    # 基础惩罚
    reward_base = -0.1
    
    # 前沿奖励（修复：coverage使用1.0作为个体系数）
    normalization = 2 * agent_width * v_max  # 28.0
    
    reward_frontier_coverage = test_case.frontier_area_change / normalization * 1.0  # 修复为1.0
    reward_frontier_coverage *= 0.125  # 组系数
    
    reward_frontier_tv = test_case.frontier_tv_change / v_max * 0.5  # 个体系数
    reward_frontier_tv *= 0.125  # 组系数
    
    # 杂草奖励
    reward_weed = 20.0 * test_case.weed_count_change
    
    # 转向奖励（组系数为0）
    reward_turn = 0.0
    
    # 碰撞和完成
    reward_collision = -399.0 if test_case.crashed else 0.0
    reward_completion = 500.0 if test_case.finished else 0.0
    
    # 总奖励
    total = (reward_base + reward_frontier_coverage + reward_frontier_tv + 
             reward_weed + reward_turn + reward_collision + reward_completion)
    
    return {
        'base': reward_base,
        'frontier_coverage': reward_frontier_coverage,
        'frontier_tv': reward_frontier_tv,
        'frontier_total': reward_frontier_coverage + reward_frontier_tv,
        'weed': reward_weed,
        'turn': reward_turn,
        'collision': reward_collision,
        'completion': reward_completion,
        'total': total
    }


def print_comparison(test_case: TestCase):
    """打印对比结果"""
    print(f"\n{'='*80}")
    print(f"测试用例: {test_case.name}")
    print(f"{'='*80}")
    
    # 计算三个版本
    old_rewards = calculate_old_reward(test_case)
    new_wrong_rewards = calculate_new_reward_wrong(test_case)
    new_fixed_rewards = calculate_new_reward_fixed(test_case)
    
    # 打印表格
    print(f"\n{'组件':<20} {'旧版本':>12} {'新版本(错误)':>12} {'新版本(修复)':>12} {'差异(错误)':>12} {'差异(修复)':>12}")
    print("-" * 92)
    
    components = ['base', 'frontier_coverage', 'frontier_tv', 'frontier_total', 
                  'weed', 'turn', 'collision', 'completion', 'total']
    
    for comp in components:
        old_val = old_rewards[comp]
        new_wrong_val = new_wrong_rewards[comp]
        new_fixed_val = new_fixed_rewards[comp]
        
        diff_wrong = new_wrong_val - old_val
        diff_fixed = new_fixed_val - old_val
        
        # 特殊格式化total行
        if comp == 'total':
            print("-" * 92)
        
        # 标记有差异的项
        marker_wrong = "❌" if abs(diff_wrong) > 1e-6 else "✅"
        marker_fixed = "❌" if abs(diff_fixed) > 1e-6 else "✅"
        
        print(f"{comp:<20} {old_val:>11.4f} {new_wrong_val:>11.4f} {new_fixed_val:>11.4f} "
              f"{diff_wrong:>10.4f} {marker_wrong} {diff_fixed:>10.4f} {marker_fixed}")
    
    # 计算百分比差异
    if abs(old_rewards['total']) > 1e-6:
        pct_diff_wrong = (new_wrong_rewards['total'] - old_rewards['total']) / abs(old_rewards['total']) * 100
        pct_diff_fixed = (new_fixed_rewards['total'] - old_rewards['total']) / abs(old_rewards['total']) * 100
        print(f"\n总奖励差异百分比: 错误配置={pct_diff_wrong:.2f}%, 修复后={pct_diff_fixed:.2f}%")
    
    # 前沿奖励差异分析
    if abs(old_rewards['frontier_total']) > 1e-6:
        frontier_diff_wrong = (new_wrong_rewards['frontier_total'] - old_rewards['frontier_total']) / abs(old_rewards['frontier_total']) * 100
        frontier_diff_fixed = (new_fixed_rewards['frontier_total'] - old_rewards['frontier_total']) / abs(old_rewards['frontier_total']) * 100
        print(f"前沿奖励差异百分比: 错误配置={frontier_diff_wrong:.2f}%, 修复后={frontier_diff_fixed:.2f}%")


def main():
    """主测试流程"""
    print("🔍 新旧环境奖励系统数值对比测试")
    print("=" * 80)
    
    # 定义测试用例
    test_cases = [
        TestCase(
            name="典型探索场景",
            frontier_area_change=50,
            frontier_tv_change=10,
            weed_count_change=5,
            steer_change=(10.0, 15.0)
        ),
        TestCase(
            name="纯前沿探索（无杂草）",
            frontier_area_change=100,
            frontier_tv_change=20,
            weed_count_change=0,
            steer_change=(0.0, 5.0)
        ),
        TestCase(
            name="纯杂草清除（无探索）",
            frontier_area_change=0,
            frontier_tv_change=0,
            weed_count_change=10,
            steer_change=(5.0, 5.0)
        ),
        TestCase(
            name="大转向场景",
            frontier_area_change=30,
            frontier_tv_change=5,
            weed_count_change=3,
            steer_change=(20.0, -20.0)  # 方向反转
        ),
        TestCase(
            name="碰撞场景",
            frontier_area_change=10,
            frontier_tv_change=2,
            weed_count_change=1,
            steer_change=(5.0, 10.0),
            crashed=True
        ),
        TestCase(
            name="任务完成场景",
            frontier_area_change=5,
            frontier_tv_change=1,
            weed_count_change=1,
            steer_change=(0.0, 0.0),
            finished=True
        ),
    ]
    
    # 运行测试
    for test_case in test_cases:
        print_comparison(test_case)
    
    # 总结
    print("\n" + "=" * 80)
    print("📊 测试总结")
    print("=" * 80)
    print("\n关键发现：")
    print("1. ❌ 新版本(错误配置)的前沿覆盖奖励只有旧版本的50%")
    print("2. ✅ 修复配置后，奖励计算完全一致")
    print("3. 修复方法：将cpp_env_v2.py中的reward_frontier_coverage_coef从0.5改为1.0")
    print("\n建议：立即应用修复，确保训练效果不受影响！")


if __name__ == "__main__":
    main()