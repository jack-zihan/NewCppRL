"""
实验运行器 - 核心实验执行引擎
"""
import sys
import importlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Type
import logging
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ..algorithms.base_algorithm import BasePathPlanner
from ..algorithms import (
    JumpPlanner, SnakePlanner, RSnakePlanner, 
    ReactPlanner, BcpPlanner
)
from ..utils.path_utils import PathUtils
from ..utils.logging_utils import LoggingUtils, PerformanceTimer
from .config_manager import ConfigManager
from .result_collector import ResultCollector


class ExperimentRunner:
    """实验运行器 - 替代原有script.py的功能"""
    
    def __init__(self, experiment_config_path: str):
        """
        初始化实验运行器
        
        Args:
            experiment_config_path: 实验配置文件路径
        """
        # 设置日志
        self.logger = LoggingUtils.get_experiment_logger("experiment_runner")
        
        # 配置管理器
        self.config_manager = ConfigManager()
        
        # 加载配置
        self.experiment_config = self.config_manager.load_experiment_config(experiment_config_path)
        self.base_config = self.config_manager.load_base_config()
        
        # 结果收集器
        output_config = self.experiment_config.get('output', {'base_dir': 'logs', 'csv_format': True})
        self.result_collector = ResultCollector(output_config)
        
        # 算法映射
        self.algorithm_map = {
            'JUMP': JumpPlanner,
            'SNAKE': SnakePlanner,
            'R_SNAKE': RSnakePlanner,
            'REACT': ReactPlanner,
            'BCP': BcpPlanner
        }
        
        # 初始化算法实例
        self.algorithm_instances: Dict[str, BasePathPlanner] = {}
        self._initialize_algorithms()
        
        self.logger.info(f"实验运行器初始化完成: {self.experiment_config['experiment']['name']}")
        
    def _initialize_algorithms(self):
        """初始化算法实例"""
        algorithms_config = self.experiment_config.get('algorithms', [])
        
        for alg_config in algorithms_config:
            if not alg_config.get('enabled', True):
                continue
                
            alg_name = alg_config['name']
            if alg_name not in self.algorithm_map:
                self.logger.warning(f"未知算法: {alg_name}")
                continue
                
            # 加载算法配置
            try:
                algorithm_config = self.config_manager.load_algorithm_config(alg_name)
                
                # 合并环境配置
                env_overrides = self.experiment_config.get('environment_overrides', {})
                merged_env_config = self.config_manager.merge_configs(self.base_config, {'environment': env_overrides})
                
                # 创建算法实例
                algorithm_class = self.algorithm_map[alg_name]
                algorithm_instance = algorithm_class(algorithm_config, merged_env_config)
                
                self.algorithm_instances[alg_name] = algorithm_instance
                self.logger.info(f"算法初始化成功: {alg_name}")
                
            except Exception as e:
                self.logger.error(f"算法初始化失败 {alg_name}: {e}")
                raise
    
    def _create_environment(self, difficulty: str, seed: int, map_id: int, 
                          weed_distribution: str, noise_level: str) -> Any:
        """创建环境实例"""
        try:
            # 导入环境类（动态导入以避免循环依赖）
            from envs_new.cpp_env_v2 import CppEnv
            
            # 获取难度配置
            difficulty_config = self.base_config['difficulty_levels'][difficulty]
            noise_config = self.base_config['noise_sets'][noise_level]
            
            # 创建环境配置
            env_config = {
                'map_id': map_id,
                'num_obstacle_min': difficulty_config['obstacle_range'][0],
                'num_obstacle_max': difficulty_config['obstacle_range'][1],
                'weed_num': difficulty_config['weed_num'],
                'weed_type': weed_distribution,
                'position_noise': noise_config[0],
                'direction_noise': noise_config[1],
                'perception_noise': noise_config[2],
            }
            
            # 合并环境覆盖配置
            env_overrides = self.experiment_config.get('environment_overrides', {})
            env_config.update(env_overrides)
            
            # 创建环境
            env = CppEnv(**env_config)
            
            return env
            
        except Exception as e:
            self.logger.error(f"创建环境失败: {e}")
            raise
    
    def _run_single_experiment(self, algorithm_name: str, algorithm: BasePathPlanner,
                             seed: int, difficulty: str, map_id: int, 
                             weed_distribution: str, noise_level: str) -> Dict[str, Any]:
        """运行单个实验"""
        experiment_info = {
            'experiment_name': self.experiment_config['experiment']['name'],
            'algorithm': algorithm_name,
            'seed': seed,
            'difficulty': difficulty,
            'map_id': map_id,
            'weed_distribution': weed_distribution,
            'noise_level': noise_level
        }
        
        self.logger.info(f"开始实验: {algorithm_name} - 种子:{seed} - 难度:{difficulty} - 地图:{map_id}")
        
        try:
            # 创建环境
            env = self._create_environment(difficulty, seed, map_id, weed_distribution, noise_level)
            
            # 重置环境和算法
            obs, info = env.reset(seed=seed)
            
            # 准备初始状态
            initial_state = {
                'agent_position': info.get('agent_position', [0, 0]),
                'agent_direction': info.get('agent_direction', 0),
                'discovered_weeds': info.get('discovered_weeds', []),
                'farm_vertices': info.get('farm_vertices'),
                'seed': seed
            }
            
            algorithm.reset(initial_state)
            
            # 运行实验
            with PerformanceTimer(f"{algorithm_name}实验") as timer:
                step_count = 0
                while True:
                    # 准备当前状态
                    current_state = {
                        'agent_position': info.get('agent_position', [0, 0]),
                        'agent_direction': info.get('agent_direction', 0),
                        'discovered_weeds': info.get('discovered_weeds', []),
                        'coverage_rate': info.get('coverage_rate', 0.0),
                        'maps': info.get('maps', {})
                    }
                    
                    # 算法规划下一步
                    next_waypoint = algorithm.plan_next_waypoint(current_state)
                    
                    # 检查终止条件
                    if next_waypoint is None or algorithm.should_terminate(current_state):
                        break
                    
                    # 执行动作（这里简化处理，实际需要根据环境接口调整）
                    # 将waypoint转换为环境动作
                    action = self._waypoint_to_action(next_waypoint, current_state['agent_position'])
                    
                    # 环境步进
                    obs, reward, terminated, truncated, info = env.step(action)
                    step_count += 1
                    
                    if terminated or truncated:
                        break
                    
                    # 防止无限循环
                    if step_count > 10000:
                        self.logger.warning(f"实验步数超限: {algorithm_name}")
                        break
            
            # 获取算法性能指标
            algorithm_metrics = algorithm.get_performance_metrics()
            algorithm_metrics['final_coverage'] = info.get('coverage_rate', 0.0)
            algorithm_metrics['total_steps'] = step_count
            
            # 清理环境
            env.close()
            
            self.logger.info(f"实验完成: {algorithm_name} - "
                           f"覆盖率: {algorithm_metrics.get('final_coverage', 0):.3f} - "
                           f"步数: {step_count}")
            
            return {
                'experiment_info': experiment_info,
                'algorithm_metrics': algorithm_metrics,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"实验失败 {algorithm_name}: {e}")
            return {
                'experiment_info': experiment_info,
                'algorithm_metrics': {'error': str(e)},
                'success': False
            }
    
    def _waypoint_to_action(self, waypoint: tuple, current_position: list) -> int:
        """将路径点转换为环境动作（简化实现）"""
        # 这里需要根据具体环境的动作空间实现
        # 暂时返回随机动作
        import random
        return random.randint(0, 6)  # 假设动作空间为0-6
    
    def run_experiment(self) -> Dict[str, Any]:
        """运行完整的实验"""
        experiment_name = self.experiment_config['experiment']['name']
        self.logger.info(f"开始运行实验: {experiment_name}")
        
        # 获取实验参数
        parameters = self.experiment_config.get('parameters', {})
        seeds = parameters.get('seeds', [42])
        difficulties = parameters.get('difficulties', ['easy'])
        weed_distributions = parameters.get('weed_distributions', ['gaussian'])
        noise_levels = parameters.get('noise_levels', ['no_noise'])
        
        # 统计信息
        total_experiments = 0
        successful_experiments = 0
        failed_experiments = 0
        
        # 运行所有实验组合
        for seed in seeds:
            for difficulty in difficulties:
                # 获取该难度的地图ID列表
                difficulty_config = self.base_config['difficulty_levels'][difficulty]
                map_ids = difficulty_config.get('map_ids', [1])
                
                for map_id in map_ids:
                    for weed_distribution in weed_distributions:
                        for noise_level in noise_levels:
                            for algorithm_name, algorithm in self.algorithm_instances.items():
                                total_experiments += 1
                                
                                # 运行单个实验
                                result = self._run_single_experiment(
                                    algorithm_name, algorithm, seed, difficulty, 
                                    map_id, weed_distribution, noise_level
                                )
                                
                                # 收集结果
                                if result['success']:
                                    self.result_collector.collect_result(
                                        result['experiment_info'], 
                                        result['algorithm_metrics']
                                    )
                                    successful_experiments += 1
                                else:
                                    failed_experiments += 1
        
        # 导出摘要
        summary_file = self.result_collector.export_summary()
        
        # 最终统计
        final_stats = {
            'experiment_name': experiment_name,
            'total_experiments': total_experiments,
            'successful_experiments': successful_experiments, 
            'failed_experiments': failed_experiments,
            'success_rate': successful_experiments / total_experiments if total_experiments > 0 else 0.0,
            'summary_file': str(summary_file),
            'result_statistics': self.result_collector.get_statistics()
        }
        
        self.logger.info(f"实验完成: {experiment_name}")
        self.logger.info(f"成功: {successful_experiments}/{total_experiments} ({final_stats['success_rate']:.1%})")
        
        return final_stats
    
    def cleanup(self):
        """清理资源"""
        self.result_collector.cleanup()
        self.config_manager.clear_cache()
        self.logger.info("实验运行器已清理")