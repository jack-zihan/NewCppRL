#!/usr/bin/env python3
"""
深入排查新旧环境的零散状态记录和边界条件处理差异

本测试脚本专注于：
1. 零散状态记录差异
2. 边界条件和异常处理
3. 隐含的状态依赖
4. 动力学更新顺序
"""

import numpy as np
import cv2
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from envs.cpp_env_base_copy import CppEnvBase as OldEnv


class StateConsistencyAnalyzer:
    """新旧环境状态一致性分析器"""
    
    def __init__(self):
        self.old_env = OldEnv(
            action_type='discrete',
            render_mode=None,
            state_pixels=False,
            use_sgcnn=False,
            use_global_obs=False,
            use_apf=False,
            use_box_boundary=True,
            use_traj=True
        )
        # 暂时只测试旧环境，因为新环境位置未知
        
    def analyze_scattered_states(self):
        """分析零散状态记录"""
        print("\n" + "="*80)
        print("1. 零散状态记录分析")
        print("="*80)
        
        # 重置环境
        self.old_env.reset(seed=42)
        
        # 分析旧环境中的状态变量
        state_vars = {
            # 核心状态
            'agent': self.old_env.agent,
            'map_id': self.old_env.map_id,
            't': self.old_env.t,
            
            # 地图状态
            'map_frontier': hasattr(self.old_env, 'map_frontier'),
            'map_obstacle': hasattr(self.old_env, 'map_obstacle'), 
            'map_weed': hasattr(self.old_env, 'map_weed'),
            'map_trajectory': hasattr(self.old_env, 'map_trajectory'),
            'map_mist': hasattr(self.old_env, 'map_mist'),
            
            # 度量状态
            'weed_num_t': self.old_env.weed_num_t,
            'frontier_area_t': self.old_env.frontier_area_t,
            'frontier_tv_t': self.old_env.frontier_tv_t,
            'steer_t': self.old_env.steer_t,
            
            # 隐藏状态
            'weed_num': self.old_env.weed_num,
            'map_frontier_full': hasattr(self.old_env, 'map_frontier_full'),
            'map_weed_ori': hasattr(self.old_env, 'map_weed_ori'),
            'map_weed_noisy': hasattr(self.old_env, 'map_weed_noisy'),
        }
        
        print("\n旧环境状态变量清单：")
        for name, value in state_vars.items():
            if isinstance(value, bool):
                print(f"  - {name}: {'存在' if value else '不存在'}")
            elif isinstance(value, (int, float)):
                print(f"  - {name}: {value}")
            else:
                print(f"  - {name}: {type(value).__name__}")
        
        # 执行一步，追踪状态更新
        print("\n执行step后的状态变化：")
        action = 0
        
        # 记录step前的状态
        before_state = {
            'weed_num_t': self.old_env.weed_num_t,
            'frontier_area_t': self.old_env.frontier_area_t,
            'frontier_tv_t': self.old_env.frontier_tv_t,
            'steer_t': self.old_env.steer_t,
            't': self.old_env.t,
            'agent_x': self.old_env.agent.x,
            'agent_y': self.old_env.agent.y,
            'agent_direction': self.old_env.agent.direction,
        }
        
        # 执行step
        obs, reward, done, timeout, info = self.old_env.step(action)
        
        # 记录step后的状态
        after_state = {
            'weed_num_t': self.old_env.weed_num_t,
            'frontier_area_t': self.old_env.frontier_area_t,
            'frontier_tv_t': self.old_env.frontier_tv_t,
            'steer_t': self.old_env.steer_t,
            't': self.old_env.t,
            'agent_x': self.old_env.agent.x,
            'agent_y': self.old_env.agent.y,
            'agent_direction': self.old_env.agent.direction,
        }
        
        # 分析变化
        for key in before_state:
            if before_state[key] != after_state[key]:
                print(f"  {key}: {before_state[key]} -> {after_state[key]}")
        
        return state_vars
    
    def analyze_boundary_conditions(self):
        """分析边界条件处理"""
        print("\n" + "="*80)
        print("2. 边界条件分析")
        print("="*80)
        
        test_cases = []
        
        # 测试1：空地图（没有frontier区域）
        print("\n测试1：空地图场景")
        try:
            # 创建一个全零的frontier地图
            self.old_env.reset(seed=42)
            original_frontier = self.old_env.map_frontier.copy()
            self.old_env.map_frontier = np.zeros_like(self.old_env.map_frontier)
            self.old_env.frontier_area_t = 0
            
            # 尝试执行step
            obs, reward, done, timeout, info = self.old_env.step(0)
            print(f"  空地图执行成功: reward={reward}, done={done}")
            
            # 恢复
            self.old_env.map_frontier = original_frontier
            test_cases.append(("空地图", "通过"))
        except Exception as e:
            print(f"  空地图执行失败: {e}")
            test_cases.append(("空地图", f"失败: {e}"))
        
        # 测试2：Agent初始位置在障碍物上
        print("\n测试2：Agent在障碍物上")
        try:
            self.old_env.reset(seed=42)
            
            # 找到一个障碍物位置
            obstacle_positions = np.argwhere(self.old_env.map_obstacle)
            if len(obstacle_positions) > 0:
                obs_pos = obstacle_positions[0]
                self.old_env.agent.x = float(obs_pos[1])
                self.old_env.agent.y = float(obs_pos[0])
                
                # 检查碰撞
                crashed = self.old_env.check_collision()
                print(f"  Agent在障碍物上，碰撞检测: {crashed}")
                test_cases.append(("Agent在障碍物", f"碰撞={crashed}"))
            else:
                print("  没有障碍物可测试")
                test_cases.append(("Agent在障碍物", "跳过"))
        except Exception as e:
            print(f"  测试失败: {e}")
            test_cases.append(("Agent在障碍物", f"失败: {e}"))
        
        # 测试3：无效的动作输入
        print("\n测试3：无效动作输入")
        self.old_env.reset(seed=42)
        
        invalid_actions = [
            -1,  # 负数
            1000,  # 超大数
            None,  # None
            [1, 2, 3],  # 错误类型
        ]
        
        for invalid_action in invalid_actions:
            try:
                obs, reward, done, timeout, info = self.old_env.step(invalid_action)
                print(f"  动作{invalid_action}: 执行成功")
                test_cases.append((f"无效动作{invalid_action}", "通过"))
            except Exception as e:
                print(f"  动作{invalid_action}: 执行失败 - {type(e).__name__}")
                test_cases.append((f"无效动作{invalid_action}", f"失败: {type(e).__name__}"))
        
        # 测试4：超出地图边界
        print("\n测试4：超出地图边界")
        try:
            self.old_env.reset(seed=42)
            
            # 将Agent移到边界外
            self.old_env.agent.x = -10.0
            self.old_env.agent.y = -10.0
            
            crashed = self.old_env.check_collision()
            print(f"  Agent在边界外，碰撞检测: {crashed}")
            
            # 执行step看是否会自动修正
            obs, reward, done, timeout, info = self.old_env.step(0)
            print(f"  执行step后: x={self.old_env.agent.x}, y={self.old_env.agent.y}")
            test_cases.append(("超出边界", f"碰撞={crashed}, 位置修正"))
        except Exception as e:
            print(f"  测试失败: {e}")
            test_cases.append(("超出边界", f"失败: {e}"))
        
        return test_cases
    
    def analyze_implicit_dependencies(self):
        """分析隐含的状态依赖"""
        print("\n" + "="*80)
        print("3. 隐含状态依赖分析")
        print("="*80)
        
        dependencies = []
        
        # 分析get_reward的依赖
        print("\n分析get_reward依赖：")
        self.old_env.reset(seed=42)
        
        # 保存初始状态
        initial_states = {
            'weed_num_t': self.old_env.weed_num_t,
            'frontier_area_t': self.old_env.frontier_area_t,
            'frontier_tv_t': self.old_env.frontier_tv_t,
            'steer_t': self.old_env.steer_t,
            'map_weed': self.old_env.map_weed.copy(),
            'map_frontier': self.old_env.map_frontier.copy(),
        }
        
        # 直接调用get_reward
        x_t, y_t = self.old_env.agent.position_discrete
        reward = self.old_env.get_reward(0.0, x_t, y_t, x_t+1, y_t+1)
        
        # 检查状态变化
        print("  get_reward会更新以下状态：")
        if self.old_env.weed_num_t != initial_states['weed_num_t']:
            print(f"    - weed_num_t: {initial_states['weed_num_t']} -> {self.old_env.weed_num_t}")
            dependencies.append(('get_reward', 'weed_num_t', '更新'))
        if self.old_env.frontier_area_t != initial_states['frontier_area_t']:
            print(f"    - frontier_area_t: {initial_states['frontier_area_t']} -> {self.old_env.frontier_area_t}")
            dependencies.append(('get_reward', 'frontier_area_t', '更新'))
        if self.old_env.frontier_tv_t != initial_states['frontier_tv_t']:
            print(f"    - frontier_tv_t: {initial_states['frontier_tv_t']} -> {self.old_env.frontier_tv_t}")
            dependencies.append(('get_reward', 'frontier_tv_t', '更新'))
        if self.old_env.steer_t != initial_states['steer_t']:
            print(f"    - steer_t: {initial_states['steer_t']} -> {self.old_env.steer_t}")
            dependencies.append(('get_reward', 'steer_t', '更新'))
        
        # 分析observation的依赖
        print("\n分析observation依赖：")
        obs = self.old_env.observation()
        print("  observation依赖的状态：")
        print(f"    - agent.last_steer: {self.old_env.agent.last_steer}")
        print(f"    - weed_num_t: {self.old_env.weed_num_t}")
        print(f"    - weed_num: {self.old_env.weed_num}")
        dependencies.append(('observation', 'agent.last_steer', '读取'))
        dependencies.append(('observation', 'weed_num_t', '读取'))
        dependencies.append(('observation', 'weed_num', '读取'))
        
        return dependencies
    
    def analyze_update_order(self):
        """分析动力学更新顺序"""
        print("\n" + "="*80)
        print("4. 动力学更新顺序分析")
        print("="*80)
        
        # 分析step函数中的更新顺序
        print("\n旧环境step函数更新顺序：")
        update_order = [
            "1. 获取当前Agent位置 (x_t, y_t)",
            "2. 解析动作得到速度和角速度 (acc, steer)",
            "3. 更新Agent位置 (agent.control)",
            "4. 更新map_weed (cv2.fillPoly)",
            "5. 更新map_frontier (cv2.ellipse)",
            "6. 更新map_mist (cv2.ellipse)",
            "7. 检查碰撞 (check_collision)",
            "8. 获取新位置 (x_tp1, y_tp1)",
            "9. 裁剪位置到地图范围",
            "10. 更新map_trajectory (cv2.line)",
            "11. 计算奖励 (get_reward) - 同时更新状态",
            "12. 处理碰撞惩罚",
            "13. 更新时间步 (t += 1)",
            "14. 检查终止条件",
            "15. 获取观测 (observation)",
        ]
        
        for step in update_order:
            print(f"  {step}")
        
        # 测试更新顺序的影响
        print("\n测试更新顺序的影响：")
        self.old_env.reset(seed=42)
        
        # 保存初始状态快照
        snapshot_before = {
            'agent_pos': (self.old_env.agent.x, self.old_env.agent.y),
            'agent_dir': self.old_env.agent.direction,
            'weed_sum': self.old_env.map_weed.sum(),
            'frontier_sum': self.old_env.map_frontier.sum(),
            'mist_sum': self.old_env.map_mist.sum() if hasattr(self.old_env, 'map_mist') else 0,
            'trajectory_sum': self.old_env.map_trajectory.sum(),
        }
        
        # 执行一步
        action = 10  # 选择一个中等动作
        obs, reward, done, timeout, info = self.old_env.step(action)
        
        # 保存step后状态快照
        snapshot_after = {
            'agent_pos': (self.old_env.agent.x, self.old_env.agent.y),
            'agent_dir': self.old_env.agent.direction,
            'weed_sum': self.old_env.map_weed.sum(),
            'frontier_sum': self.old_env.map_frontier.sum(),
            'mist_sum': self.old_env.map_mist.sum() if hasattr(self.old_env, 'map_mist') else 0,
            'trajectory_sum': self.old_env.map_trajectory.sum(),
        }
        
        print("\n状态变化总结：")
        for key in snapshot_before:
            before = snapshot_before[key]
            after = snapshot_after[key]
            if before != after:
                print(f"  {key}: {before} -> {after}")
        
        return update_order
    
    def generate_test_cases(self):
        """生成针对发现的风险点的测试用例"""
        print("\n" + "="*80)
        print("5. 建议的测试用例")
        print("="*80)
        
        test_cases = [
            {
                'name': '状态更新完整性测试',
                'description': '验证所有状态变量在step中正确更新',
                'steps': [
                    '1. 记录step前所有状态',
                    '2. 执行多个不同的动作',
                    '3. 验证每个状态变量的更新逻辑',
                    '4. 检查状态之间的一致性'
                ]
            },
            {
                'name': '边界碰撞精度测试',
                'description': '验证碰撞检测的精度和一致性',
                'steps': [
                    '1. 将Agent移动到接近边界的位置',
                    '2. 以小步长逐渐接近边界',
                    '3. 记录碰撞触发的精确位置',
                    '4. 对比新旧环境的碰撞阈值'
                ]
            },
            {
                'name': '奖励计算依赖测试',
                'description': '验证奖励计算的前置条件',
                'steps': [
                    '1. 人为修改地图状态',
                    '2. 调用get_reward',
                    '3. 验证状态更新的正确性',
                    '4. 检查奖励值的合理性'
                ]
            },
            {
                'name': '观测生成一致性测试',
                'description': '验证observation的生成逻辑',
                'steps': [
                    '1. 设置特定的环境状态',
                    '2. 生成观测',
                    '3. 验证观测中的每个组件',
                    '4. 对比新旧环境的观测差异'
                ]
            },
            {
                'name': '长期运行稳定性测试',
                'description': '验证长时间运行的数值稳定性',
                'steps': [
                    '1. 运行1000步固定动作序列',
                    '2. 每100步记录关键状态',
                    '3. 检查数值累积误差',
                    '4. 验证状态一致性'
                ]
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n测试用例{i}: {test_case['name']}")
            print(f"描述: {test_case['description']}")
            print("步骤:")
            for step in test_case['steps']:
                print(f"  {step}")
        
        return test_cases
    
    def run_full_analysis(self):
        """运行完整分析"""
        print("\n" + "="*80)
        print("新旧环境状态一致性深度分析报告")
        print("="*80)
        
        # 1. 零散状态记录
        state_vars = self.analyze_scattered_states()
        
        # 2. 边界条件
        boundary_tests = self.analyze_boundary_conditions()
        
        # 3. 隐含依赖
        dependencies = self.analyze_implicit_dependencies()
        
        # 4. 更新顺序
        update_order = self.analyze_update_order()
        
        # 5. 测试用例
        test_cases = self.generate_test_cases()
        
        # 生成总结报告
        print("\n" + "="*80)
        print("风险评估总结")
        print("="*80)
        
        print("\n高风险问题（会影响RL训练）：")
        print("1. get_reward中的状态更新可能导致新旧环境不一致")
        print("2. 碰撞检测的边界处理可能存在细微差异")
        print("3. 地图更新顺序的差异可能累积误差")
        
        print("\n中风险问题（可能影响训练稳定性）：")
        print("1. 零散的状态变量管理容易遗漏")
        print("2. observation对状态的依赖可能不完整")
        print("3. 浮点数累积误差在长期运行中放大")
        
        print("\n低风险问题（需要注意但影响较小）：")
        print("1. 异常输入的处理方式差异")
        print("2. 渲染相关的状态更新")
        print("3. 噪声注入的实现细节")
        
        return {
            'state_vars': state_vars,
            'boundary_tests': boundary_tests,
            'dependencies': dependencies,
            'update_order': update_order,
            'test_cases': test_cases
        }


if __name__ == "__main__":
    analyzer = StateConsistencyAnalyzer()
    results = analyzer.run_full_analysis()
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)