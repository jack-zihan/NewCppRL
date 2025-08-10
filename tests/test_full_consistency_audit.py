#!/usr/bin/env python3
"""
Rules_new vs Rules_new1 完整一致性审查
验证两个版本算法的完全一致性，包括指标、轨迹和可视化
"""

import sys
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import logging
import yaml
from omegaconf import DictConfig

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置环境变量，禁用GUI
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'

import envs  # noqa - 注册环境
import gymnasium as gym

# 导入自定义模块
from rules_new_adapter import RulesNewAdapter
from metrics_collector import MetricsCollector
from visualization_generator import VisualizationGenerator
from report_generator import ReportGenerator

# 导入rules_new1
from rules_new.algorithms import (
    JumpPlanner, SnakePlanner, RSnakePlanner, 
    ReactPlanner, BcpPlanner
)


class ConsistencyAuditor:
    """一致性审查器"""
    
    def __init__(self, config_path: str = None):
        """初始化审查器"""
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 创建日志目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = project_root / 'logs' / f'consistency_audit_{timestamp}'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 初始化组件
        self.metrics_collector = MetricsCollector(self.log_dir)
        self.visualizer = VisualizationGenerator(self.log_dir)
        self.report_generator = ReportGenerator(self.log_dir)
        
        # rules_new适配器
        self.rules_new_adapter = RulesNewAdapter()
        
        # rules_new1算法映射
        self.rules_new1_algorithms = {
            'BCP': BcpPlanner,
            'JUMP': JumpPlanner,
            'SNAKE': SnakePlanner,
            'R_SNAKE': RSnakePlanner,
            'REACT': ReactPlanner
        }
        
        # 测试结果存储
        self.results = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """加载测试配置"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # 默认配置
            return {
                'test_config': {
                    'seeds': [42],  # 减少测试种子数以加快执行
                    'algorithms': ['BCP', 'JUMP', 'SNAKE', 'R_SNAKE', 'REACT'],
                    'test_episodes': 1,
                    'max_steps': 200,  # 减少最大步数以加快执行
                    'render': True,
                    'save_frequency': 50
                },
                'environment': {
                    'difficulty': 'medium',
                    'map_id': 4,
                    'weed_distribution': 'uniform',
                    'noise_level': [0, 0, 0]
                }
            }
    
    def _setup_logging(self):
        """设置日志系统"""
        log_file = self.log_dir / 'execution_logs' / 'audit.log'
        log_file.parent.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('ConsistencyAuditor')
        
    def create_environment(self, seed: int) -> gym.Env:
        """创建统一的测试环境"""
        # 加载环境配置
        cfg = DictConfig(yaml.load(
            open(f'{project_root}/configs/env_config.yaml'), 
            Loader=yaml.FullLoader
        ))
        
        # 创建环境（启用渲染以获取图像）
        env = gym.make(
            render_mode='rgb_array' if self.config['test_config']['render'] else None,
            **cfg.env.params,
        )
        
        # 重置环境
        obs, info = env.reset(seed=seed)
        
        return env, obs, info
    
    def run_rules_new(self, algorithm: str, seed: int) -> Dict[str, Any]:
        """运行rules_new版本的算法"""
        self.logger.info(f"运行rules_new - 算法: {algorithm}, Seed: {seed}")
        
        # 创建环境
        env, obs, info = self.create_environment(seed)
        
        # 运行算法
        try:
            result = self.rules_new_adapter.run_algorithm(
                algorithm=algorithm,
                env=env,
                obs=obs,
                info=info,
                max_steps=self.config['test_config']['max_steps'],
                render=self.config['test_config']['render']
            )
            
            # 收集指标
            metrics = self.metrics_collector.collect_from_rules_new(result)
            
            env.close()
            return metrics
            
        except Exception as e:
            self.logger.error(f"rules_new执行失败: {e}")
            env.close()
            return None
    
    def run_rules_new1(self, algorithm: str, seed: int) -> Dict[str, Any]:
        """运行rules_new1版本的算法"""
        self.logger.info(f"运行rules_new1 - 算法: {algorithm}, Seed: {seed}")
        
        # 创建环境
        env, obs, info = self.create_environment(seed)
        
        # 创建算法配置
        algorithm_config = {
            'algorithm': {'name': algorithm},
            'parameters': {},
            'performance': {
                'max_iterations': self.config['test_config']['max_steps'],
                'timeout_seconds': 300
            }
        }
        
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
        
        try:
            # 创建算法实例
            AlgorithmClass = self.rules_new1_algorithms[algorithm]
            planner = AlgorithmClass(algorithm_config, env_config)
            
            # 准备初始状态
            initial_state = {
                'agent_position': [float(env.agent.x), float(env.agent.y)],
                'agent_direction': float(env.agent.direction),
                'discovered_weeds': [],
                'weed_count': 100,
                'coverage_rate': 0.0,
                'farm_vertices': env.min_area_rect[0][:, 0, ::-1] if hasattr(env, 'min_area_rect') else np.array([[50, 50], [550, 50], [550, 550], [50, 550]]),
                'seed': seed,
                'turning_radius': env.v_range.max / (abs(env.w_range.max) * np.pi / 180),
                'maps': {
                    'weed': env.map_weed if hasattr(env, 'map_weed') else None,
                    'obstacle': env.map_obstacle if hasattr(env, 'map_obstacle') else None,
                    'frontier': env.map_frontier if hasattr(env, 'map_frontier') else None
                }
            }
            
            # 重置算法
            planner.reset(initial_state)
            
            # 运行算法并收集数据
            metrics = {
                'algorithm': algorithm,
                'seed': seed,
                'trajectory': [],
                'actions': [],
                'rewards': [],
                'total_reward': 0,
                'steps': 0,
                'coverage_rate': 0,
                'execution_time': 0,
                'final_frame': None
            }
            
            start_time = time.time()
            
            for step in range(self.config['test_config']['max_steps']):
                # 获取决策
                decision = planner.plan_next_waypoint(initial_state)
                
                if decision is None:
                    break
                
                # 处理决策（转换为动作）
                if isinstance(decision, tuple) and decision[0] == 'path':
                    # 执行路径中的第一个点
                    path_points = decision[1]
                    if path_points:
                        waypoint = path_points[0]
                        # 转换为动作
                        action = self._waypoint_to_action(waypoint, env)
                    else:
                        break
                else:
                    break
                
                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)
                
                # 收集数据
                metrics['trajectory'].append([env.agent.x, env.agent.y])
                metrics['actions'].append(action)
                metrics['rewards'].append(reward)
                metrics['total_reward'] += reward
                metrics['steps'] += 1
                metrics['coverage_rate'] = info.get('coverage_rate', 0)
                
                # 更新状态
                initial_state['agent_position'] = [env.agent.x, env.agent.y]
                initial_state['agent_direction'] = env.agent.direction
                initial_state['coverage_rate'] = metrics['coverage_rate']
                
                # 保存最后一帧
                if self.config['test_config']['render']:
                    metrics['final_frame'] = env.render()
                
                if terminated or truncated:
                    break
            
            metrics['execution_time'] = time.time() - start_time
            
            env.close()
            return metrics
            
        except Exception as e:
            self.logger.error(f"rules_new1执行失败: {e}")
            import traceback
            traceback.print_exc()
            env.close()
            return None
    
    def _waypoint_to_action(self, waypoint: List[float], env: gym.Env) -> Tuple[float, float]:
        """将路径点转换为动作"""
        import math
        
        agent_pos = [env.agent.x, env.agent.y]
        agent_dir = env.agent.direction
        agent_rad = np.pi / 2 - math.radians(agent_dir)
        
        target_rad = math.atan2(
            waypoint[1] - agent_pos[1],
            waypoint[0] - agent_pos[0]
        )
        
        length = math.sqrt(
            (waypoint[0] - agent_pos[0])**2 + 
            (waypoint[1] - agent_pos[1])**2
        )
        
        delta_angle = -(target_rad - agent_rad) % (2 * math.pi)
        if delta_angle > math.pi:
            delta_angle = delta_angle - 2 * math.pi
        delta_angle = math.degrees(delta_angle)
        
        return (length, delta_angle)
    
    def compare_metrics(self, metrics_old: Dict, metrics_new: Dict) -> Dict[str, Any]:
        """比较两个版本的指标"""
        if not metrics_old or not metrics_new:
            return {'consistent': False, 'error': 'Missing metrics'}
        
        comparison = {
            'consistent': True,
            'similarity_score': 0,
            'details': {}
        }
        
        # 比较各项指标
        # 1. 奖励差异
        reward_diff = abs(metrics_old['total_reward'] - metrics_new['total_reward'])
        comparison['details']['reward_diff'] = reward_diff
        comparison['details']['reward_consistent'] = reward_diff < 1.0
        
        # 2. 覆盖率差异
        coverage_diff = abs(metrics_old['coverage_rate'] - metrics_new['coverage_rate'])
        comparison['details']['coverage_diff'] = coverage_diff
        comparison['details']['coverage_consistent'] = coverage_diff < 0.01
        
        # 3. 步数差异
        steps_diff = abs(metrics_old['steps'] - metrics_new['steps'])
        comparison['details']['steps_diff'] = steps_diff
        comparison['details']['steps_consistent'] = steps_diff < 10
        
        # 4. 轨迹相似度
        if metrics_old['trajectory'] and metrics_new['trajectory']:
            min_len = min(len(metrics_old['trajectory']), len(metrics_new['trajectory']))
            if min_len > 0:
                traj_old = np.array(metrics_old['trajectory'][:min_len])
                traj_new = np.array(metrics_new['trajectory'][:min_len])
                avg_distance = np.mean(np.linalg.norm(traj_old - traj_new, axis=1))
                comparison['details']['trajectory_distance'] = avg_distance
                comparison['details']['trajectory_consistent'] = avg_distance < 5.0
            else:
                comparison['details']['trajectory_consistent'] = False
        
        # 5. 计算总体相似度
        consistent_items = [
            comparison['details'].get('reward_consistent', False),
            comparison['details'].get('coverage_consistent', False),
            comparison['details'].get('steps_consistent', False),
            comparison['details'].get('trajectory_consistent', False)
        ]
        
        comparison['similarity_score'] = sum(consistent_items) / len(consistent_items) * 100
        comparison['consistent'] = comparison['similarity_score'] >= 90
        
        return comparison
    
    def run_algorithm_test(self, algorithm: str) -> Dict[str, Any]:
        """测试单个算法的一致性"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"测试算法: {algorithm}")
        self.logger.info(f"{'='*60}")
        
        algorithm_results = {
            'algorithm': algorithm,
            'seeds': {},
            'overall_consistency': 0
        }
        
        # 为算法创建目录
        alg_dir = self.log_dir / 'algorithms' / algorithm
        alg_dir.mkdir(parents=True, exist_ok=True)
        
        consistency_scores = []
        
        for seed in self.config['test_config']['seeds']:
            self.logger.info(f"\nSeed {seed}:")
            
            # 运行两个版本
            metrics_old = self.run_rules_new(algorithm, seed)
            metrics_new = self.run_rules_new1(algorithm, seed)
            
            # 比较结果
            comparison = self.compare_metrics(metrics_old, metrics_new)
            
            # 保存结果
            seed_results = {
                'metrics_old': metrics_old,
                'metrics_new': metrics_new,
                'comparison': comparison
            }
            
            algorithm_results['seeds'][seed] = seed_results
            
            # 生成可视化
            if metrics_old and metrics_new:
                # 轨迹对比图
                traj_plot = self.visualizer.create_trajectory_comparison(
                    metrics_old['trajectory'],
                    metrics_new['trajectory'],
                    algorithm, seed
                )
                
                # 保存最后一帧
                if metrics_old.get('final_frame') is not None:
                    frame_path = alg_dir / f'final_frame_old_seed{seed}.png'
                    plt.imsave(frame_path, metrics_old['final_frame'])
                
                if metrics_new.get('final_frame') is not None:
                    frame_path = alg_dir / f'final_frame_new_seed{seed}.png'
                    plt.imsave(frame_path, metrics_new['final_frame'])
            
            # 输出结果
            if comparison['consistent']:
                self.logger.info(f"  ✅ 一致 (相似度: {comparison['similarity_score']:.1f}%)")
            else:
                self.logger.info(f"  ❌ 不一致 (相似度: {comparison['similarity_score']:.1f}%)")
            
            consistency_scores.append(comparison['similarity_score'])
        
        # 计算总体一致性
        algorithm_results['overall_consistency'] = np.mean(consistency_scores)
        
        # 保存算法结果
        with open(alg_dir / 'results.json', 'w') as f:
            json.dump(algorithm_results, f, indent=2, default=str)
        
        return algorithm_results
    
    def run_full_audit(self):
        """运行完整的一致性审查"""
        self.logger.info("\n🔍 Rules_new vs Rules_new1 一致性审查")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        # 测试所有算法
        for algorithm in self.config['test_config']['algorithms']:
            result = self.run_algorithm_test(algorithm)
            self.results[algorithm] = result
        
        # 计算总体一致性
        overall_scores = [r['overall_consistency'] for r in self.results.values()]
        total_consistency = np.mean(overall_scores)
        
        # 生成报告
        self.logger.info("\n" + "=" * 60)
        self.logger.info("生成报告...")
        
        # 生成HTML报告
        report_path = self.report_generator.generate_html_report(self.results)
        
        # 生成CSV摘要
        csv_path = self.metrics_collector.export_summary(self.results)
        
        # 输出总结
        self.logger.info("\n" + "=" * 60)
        self.logger.info("审查完成")
        self.logger.info(f"总体一致性: {total_consistency:.1f}%")
        self.logger.info(f"执行时间: {time.time() - start_time:.1f}秒")
        self.logger.info(f"报告已保存至: {self.log_dir}")
        
        return total_consistency >= 90  # 90%以上认为一致


def main():
    """主函数"""
    auditor = ConsistencyAuditor()
    success = auditor.run_full_audit()
    
    if success:
        print("\n🎉 一致性审查通过！")
        return 0
    else:
        print("\n⚠️ 一致性审查未通过，请检查报告")
        return 1


if __name__ == "__main__":
    sys.exit(main())