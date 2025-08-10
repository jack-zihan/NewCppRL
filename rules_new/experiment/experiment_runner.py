"""
实验运行器 - 核心实验执行引擎（完全重构版）
"""
import sys
import torch
import gymnasium as gym
import yaml
import numpy as np
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Tuple
from omegaconf import DictConfig
import logging
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入老版本环境
import envs  # noqa - 这会注册老版本环境

from ..algorithms.base_algorithm import BasePathPlanner
from ..algorithms import (
    JumpPlanner, SnakePlanner, RSnakePlanner, 
    ReactPlanner, BcpPlanner, NNPlanner
)
from ..utils.path_utils import PathUtils
from ..utils.logging_utils import LoggingUtils, PerformanceTimer
from ..utils.trajectory_collector import TrajectoryCollector
from .config_manager import ConfigManager
from .result_collector import ResultCollector
from ..core.recovery_manager import RecoveryManager
from ..core.exceptions import RulesNewError, AlgorithmError, EnvironmentError


class ExperimentRunner:
    """实验运行器 - 真实环境集成版本"""
    
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
        self.experiment_name = self.experiment_config['experiment']['name']
        
        # 创建实验专用目录
        output_config = self.experiment_config.get('output', {'base_dir': 'logs', 'csv_format': True})
        self.experiment_dir = PathUtils.get_experiment_output_directory(self.experiment_name, {'output': output_config})
        
        # 初始化结果收集器
        self.result_collector = ResultCollector(self.experiment_name, output_config, self.experiment_dir)
        
        # 初始化轨迹收集器
        self.trajectory_collector = TrajectoryCollector(
            self.result_collector.get_trajectories_directory(),
            target_seed=42  # 固定使用种子42进行轨迹收集
        )
        
        # 初始化恢复管理器
        checkpoint_dir = self.experiment_dir / "checkpoints"
        self.recovery_manager = RecoveryManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=10,
            auto_recovery=True
        )
        self.logger.info(f"恢复管理器初始化完成，检查点目录：{checkpoint_dir}")
        
        # 算法映射（包含神经网络算法）
        self.algorithm_map = {
            'JUMP': JumpPlanner,
            'SNAKE': SnakePlanner,
            'R_SNAKE': RSnakePlanner,
            'REACT': ReactPlanner,
            'BCP': BcpPlanner,
            'NN_baseline': NNPlanner,
            'NN_ours': NNPlanner
        }
        
        # 初始化算法实例
        self.algorithm_instances: Dict[str, BasePathPlanner] = {}
        self._initialize_algorithms()
        
        # 路径执行队列（极简状态管理）
        self._path_queue = []
        
        self.logger.info(f"实验运行器初始化完成: {self.experiment_name}")
        self.logger.info(f"实验目录: {self.experiment_dir}")
        
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
                
            try:
                # 神经网络算法特殊处理
                if alg_name.startswith('NN_'):
                    # 直接使用配置中的model_path
                    algorithm_config = {
                        'algorithm': {'name': alg_name, 'type': 'neural_network'},
                        'model_path': alg_config.get('model_path'),
                        'device': alg_config.get('device', 'cpu'),
                        'performance': {'max_iterations': 5000, 'timeout_seconds': 300}
                    }
                else:
                    # 传统算法加载配置文件
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
    
    def _validate_nn_observation_data(self, obs_data: Dict[str, Any], context: str):
        """
        验证NN算法的观测数据格式和内容
        
        Args:
            obs_data: 观测数据字典
            context: 验证上下文（用于错误信息）
        """
        try:
            # 验证必需字段存在
            required_keys = ['observation', 'vector']
            missing_keys = [key for key in required_keys if key not in obs_data]
            if missing_keys:
                raise ValueError(f"{context} - NN算法观测数据缺少必需字段: {missing_keys}")
            
            # 验证observation字段
            observation = obs_data['observation']
            if not isinstance(observation, np.ndarray):
                raise TypeError(f"{context} - observation应该是numpy数组，实际类型: {type(observation)}")
            
            if observation.shape != (25, 16, 16):
                raise ValueError(f"{context} - observation形状错误: {observation.shape}, 期望: (25, 16, 16)")
            
            if observation.dtype != np.float32:
                self.logger.warning(f"{context} - observation数据类型: {observation.dtype}, 期望: float32")
            
            # 验证数值范围（应该在[0, 1]范围内）
            if observation.min() < -0.1 or observation.max() > 1.1:
                self.logger.warning(f"{context} - observation数值范围异常: [{observation.min():.3f}, {observation.max():.3f}]")
            
            # 验证转换后的20通道数据
            observation_20ch = observation[:20, :, :]
            self.logger.debug(f"{context} - 25通道->20通道转换: {observation.shape} -> {observation_20ch.shape}")
            
            # 验证vector字段
            vector = obs_data['vector']
            if not isinstance(vector, (int, float, np.number)) and not (hasattr(vector, '__len__') and len(vector) == 1):
                self.logger.warning(f"{context} - vector格式异常: {type(vector)}, 值: {vector}")
            
            self.logger.debug(f"{context} - NN观测数据验证通过")
            
        except Exception as e:
            self.logger.error(f"{context} - NN观测数据验证失败: {e}")
            raise
    
    def _create_environment(self, difficulty: str, seed: int, map_id: int, 
                          weed_distribution: str, noise_level: str, algorithm_name: str = None):
        """创建环境实例 - 统一使用老版本envs环境（与sac_cont_test.py相同）"""
        try:
            # 配置无头渲染环境变量
            import os
            os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # 禁用Qt GUI
            os.environ['MPLBACKEND'] = 'Agg'  # 使用非交互式matplotlib后端
            
            # 统一使用老版本envs环境 (与sac_cont_test.py相同)
            import gymnasium as gym
            import yaml
            from omegaconf import DictConfig
            
            # 加载环境配置
            project_root = Path(__file__).parents[2]
            env_config_path = project_root / 'configs' / 'env_config.yaml'
            cfg = DictConfig(yaml.load(open(env_config_path), Loader=yaml.FullLoader))
            
            # 创建老版本环境（所有算法都使用相同方式）
            env = gym.make(
                render_mode='rgb_array',  # 用于轨迹可视化
                **cfg.env.params,
            )
            
            self.logger.info(f"为 {algorithm_name} 创建老版本环境: {cfg.env.params.id}")
            
            return env
            
        except Exception as e:
            self.logger.error(f"创建环境失败: {e}")
            raise
    
    def _waypoint_to_action(self, waypoint: Tuple[float, float], current_state: Dict[str, Any], 
                           action_space) -> Any:
        """
        将路径点转换为环境动作
        
        支持连续动作空间，返回[length, delta_angle]格式（与rules_new一致）
        """
        try:
            if waypoint is None:
                # 无路径点，返回停止动作
                return (0.0, 0.0)
            
            # 获取当前状态
            agent_pos = current_state['agent_position']
            agent_dir = current_state['agent_direction']
            
            # 当前朝向弧度（注意坐标系转换，与rules_new一致）
            current_rad = np.pi / 2 - math.radians(agent_dir)
            
            # 目标方向
            target_rad = math.atan2(
                waypoint[1] - agent_pos[1],
                waypoint[0] - agent_pos[0]
            )
            
            # 计算距离（作为length）
            length = math.sqrt(
                (waypoint[0] - agent_pos[0]) ** 2 + 
                (waypoint[1] - agent_pos[1]) ** 2
            )
            
            # 计算角度差（与rules_new的go函数完全一致）
            delta_angle = -(target_rad - current_rad) % (2 * math.pi)
            if delta_angle > math.pi:
                delta_angle = delta_angle - 2 * math.pi
            delta_angle = math.degrees(delta_angle)
            
            # 返回连续动作[length, delta_angle]
            return (length, delta_angle)
            
        except Exception as e:
            self.logger.error(f"waypoint转action失败: {e}")
            return (0.0, 0.0)  # 返回停止动作作为安全默认值
    
    def _run_single_experiment(self, algorithm_name: str, algorithm: BasePathPlanner,
                             seed: int, difficulty: str, map_id: int, 
                             weed_distribution: str, noise_level: str) -> Dict[str, Any]:
        """运行单个实验 - 真实环境交互版本"""
        experiment_info = {
            'experiment_name': self.experiment_name,
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
            env = self._create_environment(difficulty, seed, map_id, weed_distribution, noise_level, algorithm_name)
            
            # 重置环境
            obs, info = env.reset(seed=seed)
            
            # 从老版本环境提取状态信息
            def extract_state_from_environment(env, info, algorithm_name):
                """从老版本envs环境中提取算法需要的状态信息"""
                # 统一使用老版本环境的属性访问方式
                # 重要修复：使用[y,x]格式与旧版保持一致
                agent_position = [float(env.agent.y), float(env.agent.x)]
                agent_direction = float(env.agent.direction)
                
                # 获取农场边界（如果存在）
                if hasattr(env, 'min_area_rect') and env.min_area_rect is not None:
                    # 从min_area_rect提取顶点坐标
                    farm_vertices = env.min_area_rect[0][:, 0, ::-1]
                else:
                    # 默认农场边界
                    farm_vertices = np.array([
                        [50, 50], [350, 50], [350, 350], [50, 350]
                    ])
                
                # 获取杂草和障碍物地图
                map_weed = env.map_weed if hasattr(env, 'map_weed') else None
                map_obstacle = env.map_obstacle if hasattr(env, 'map_obstacle') else None
                map_frontier = env.map_frontier if hasattr(env, 'map_frontier') else None
                
                # 从info获取覆盖率等信息
                if info and isinstance(info, dict):
                    weed_count = info.get('weed_count', 0)
                    coverage_rate = info.get('coverage_rate', 0.0)
                else:
                    # 初始状态计算杂草数量
                    weed_count = int(map_weed.sum()) if map_weed is not None else 0
                    coverage_rate = 0.0
                
                # 提取已发现的杂草位置（如果需要）
                discovered_weeds = []
                if map_weed is not None and map_frontier is not None:
                    # 已发现的杂草是不在frontier区域内的杂草
                    discovered_weed_mask = np.logical_and(map_weed, np.logical_not(map_frontier))
                    weed_positions = np.argwhere(discovered_weed_mask)
                    discovered_weeds = [(int(pos[1]), int(pos[0])) for pos in weed_positions]  # (x, y)格式
                
                # 计算turning_radius（与rules_new保持一致）
                import math
                turning_radius = None
                if hasattr(env, 'v_range') and hasattr(env, 'w_range'):
                    w_max_rad = abs(env.w_range.max) * (math.pi / 180)
                    turning_radius = env.v_range.max / w_max_rad
                
                # 构造算法期望的状态格式
                return {
                    'agent_position': agent_position,
                    'agent_direction': agent_direction,
                    'discovered_weeds': discovered_weeds,
                    'weed_count': weed_count,
                    'coverage_rate': coverage_rate,
                    'farm_vertices': farm_vertices,
                    'seed': seed,
                    'turning_radius': turning_radius,  # 添加turning_radius
                    'maps': {
                        'weed': map_weed,
                        'obstacle': map_obstacle,
                        'frontier': map_frontier
                    }
                }
            
            # 准备初始状态
            initial_state = extract_state_from_environment(env, info, algorithm_name)
            
            # 为NN算法添加观测数据
            if algorithm_name and algorithm_name.startswith('NN_'):
                if isinstance(obs, dict):
                    # 老版本环境返回字典格式观测，直接传递
                    initial_state.update(obs)  # 将obs中的observation, vector, weed_ratio等添加到状态中
                    
                    # 验证NN算法必需的观测数据
                    self._validate_nn_observation_data(obs, "初始状态")
                    
                else:
                    # 如果obs不是字典，作为observation传递
                    initial_state['observation'] = obs
            
            # 重置算法
            algorithm.reset(initial_state)
            
            # 开始轨迹记录
            self.trajectory_collector.start_recording(algorithm_name, seed)
            
            # 运行实验主循环
            with PerformanceTimer(f"{algorithm_name}实验") as timer:
                step_count = 0
                total_reward = 0.0
                
                # 清空路径队列
                self._path_queue = []
                
                while True:
                    # 从环境提取当前状态
                    current_state = extract_state_from_environment(env, info, algorithm_name)
                    
                    # 添加观测数据给NN算法使用
                    if isinstance(obs, dict):
                        # 老版本环境返回字典格式观测，直接传递
                        current_state.update(obs)  # 将obs中的observation, vector, weed_ratio等添加到状态中
                        
                        # 对NN算法验证观测数据
                        if algorithm_name and algorithm_name.startswith('NN_'):
                            self._validate_nn_observation_data(obs, f"步骤{step_count}")
                            
                    else:
                        # 如果obs不是字典，作为observation传递
                        current_state['observation'] = obs
                    
                    # 记录轨迹位置
                    self.trajectory_collector.record_position(tuple(current_state['agent_position']))
                    
                    # 获取下一个动作
                    action = None
                    
                    # 如果路径队列中有路径点，优先执行
                    if self._path_queue:
                        waypoint = self._path_queue.pop(0)
                        action = self._waypoint_to_action(waypoint, current_state, env.action_space)
                    else:
                        # 获取算法决策（带错误恢复）
                        try:
                            # 定期保存检查点（每100步）
                            if step_count % 100 == 0:
                                checkpoint_state = {
                                    'current_state': current_state,
                                    'step_count': step_count,
                                    'total_reward': total_reward,
                                    'algorithm_name': algorithm_name,
                                    'seed': seed
                                }
                                self.recovery_manager.save_checkpoint(
                                    checkpoint_state,
                                    f"{algorithm_name}_seed{seed}_step{step_count}"
                                )
                            
                            decision = algorithm.plan_next_waypoint(current_state)
                            
                        except (AlgorithmError, RulesNewError) as e:
                            # 尝试恢复
                            self.logger.warning(f"算法错误，尝试恢复: {e}")
                            try:
                                recovered_state = self.recovery_manager.recover_from_error(e, current_state)
                                
                                # 检查是否需要重置环境
                                if recovered_state.get('needs_env_reset'):
                                    self.logger.info("需要重置环境")
                                    obs, info = env.reset(seed=seed)
                                    if recovered_state.get('last_checkpoint'):
                                        # 恢复到检查点状态
                                        current_state = recovered_state['last_checkpoint']['current_state']
                                        step_count = recovered_state['last_checkpoint'].get('step_count', 0)
                                        total_reward = recovered_state['last_checkpoint'].get('total_reward', 0)
                                    continue
                                else:
                                    # 使用恢复的状态继续
                                    current_state.update(recovered_state)
                                    decision = algorithm.plan_next_waypoint(current_state)
                                    
                            except Exception as recovery_error:
                                self.logger.error(f"恢复失败，跳过该实验: {recovery_error}")
                                break
                        
                        # 检查终止条件
                        if decision is None or algorithm.should_terminate(current_state):
                            break
                        
                        # 处理不同类型的算法输出
                        if isinstance(decision, tuple) and len(decision) == 2:
                            if decision[0] == 'action':
                                # 神经网络算法直接返回action
                                action = decision[1]
                            elif decision[0] == 'path':
                                # 返回路径列表，加入队列
                                self._path_queue.extend(decision[1])
                                # 递归处理第一个路径点
                                if self._path_queue:
                                    waypoint = self._path_queue.pop(0)
                                    action = self._waypoint_to_action(waypoint, current_state, env.action_space)
                                else:
                                    # 路径列表为空，跳过这一步
                                    continue
                            else:
                                # 简单waypoint
                                action = self._waypoint_to_action(decision, current_state, env.action_space)
                        else:
                            # 向后兼容：简单waypoint
                            action = self._waypoint_to_action(decision, current_state, env.action_space)
                    
                    # 如果没有有效动作，跳过
                    if action is None:
                        continue
                    
                    # 环境步进
                    obs, reward, terminated, truncated, info = env.step(action)
                    step_count += 1
                    total_reward += reward
                    
                    # 检查环境终止条件
                    if terminated or truncated:
                        break
                    
                    # 防止无限循环
                    if step_count > 10000:
                        self.logger.warning(f"实验步数超限: {algorithm_name}")
                        break
            
            # 停止轨迹记录
            self.trajectory_collector.stop_recording()
            
            # 获取算法性能指标
            algorithm_metrics = algorithm.get_performance_metrics()
            algorithm_metrics.update({
                'final_coverage': info.get('coverage_rate', 0.0),
                'total_steps': step_count,
                'total_reward': total_reward,
                'success': True
            })
            
            # 记录轨迹性能指标
            self.trajectory_collector.record_performance(algorithm_name, algorithm_metrics)
            
            # 在关闭环境前保存最后一帧render图像
            try:
                final_render = env.render()
                if final_render is not None:
                    self.trajectory_collector.save_final_render(algorithm_name, seed, final_render)
                else:
                    self.logger.warning(f"环境render返回None: {algorithm_name}")
            except Exception as e:
                self.logger.error(f"保存render图像失败 {algorithm_name}: {e}")
            
            # 清理环境
            env.close()
            
            self.logger.info(f"实验完成: {algorithm_name} - "
                           f"覆盖率: {algorithm_metrics.get('final_coverage', 0):.3f} - "
                           f"步数: {step_count} - 奖励: {total_reward:.2f}")
            
            return {
                'experiment_info': experiment_info,
                'algorithm_metrics': algorithm_metrics,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"实验失败 {algorithm_name}: {e}")
            # 停止轨迹记录
            self.trajectory_collector.stop_recording()
            
            return {
                'experiment_info': experiment_info,
                'algorithm_metrics': {'error': str(e), 'success': False},
                'success': False
            }
    
    def run_experiment(self) -> Dict[str, Any]:
        """运行完整的实验"""
        self.logger.info(f"开始运行实验: {self.experiment_name}")
        
        # 获取实验参数
        parameters = self.experiment_config.get('parameters', {})
        seeds = parameters.get('seeds', [42])
        difficulties = parameters.get('difficulties', ['easy'])
        weed_distributions = parameters.get('weed_distributions', ['gaussian'])
        noise_levels = parameters.get('noise_levels', ['no_noise'])
        
        # 记录环境信息（用于轨迹可视化）
        self.trajectory_collector.record_environment_info({
            'width': self.base_config.get('environment', {}).get('width', 600),
            'height': self.base_config.get('environment', {}).get('height', 600),
            'seeds': seeds,
            'difficulties': difficulties
        })
        
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
        
        # 生成轨迹对比图
        trajectory_plot = self.trajectory_collector.generate_comparison_plot(self.experiment_name)
        
        # 导出摘要
        summary_file = self.result_collector.export_summary()
        
        # 最终统计
        final_stats = {
            'experiment_name': self.experiment_name,
            'experiment_directory': str(self.experiment_dir),
            'total_experiments': total_experiments,
            'successful_experiments': successful_experiments, 
            'failed_experiments': failed_experiments,
            'success_rate': successful_experiments / total_experiments if total_experiments > 0 else 0.0,
            'summary_file': str(summary_file.relative_to(self.experiment_dir)) if summary_file else None,
            'trajectory_plot': str(trajectory_plot.relative_to(self.experiment_dir)) if trajectory_plot else None,
            'result_statistics': self.result_collector.get_statistics(),
            'trajectory_statistics': self.trajectory_collector.get_summary()
        }
        
        self.logger.info(f"实验完成: {self.experiment_name}")
        self.logger.info(f"成功: {successful_experiments}/{total_experiments} ({final_stats['success_rate']:.1%})")
        self.logger.info(f"实验目录: {self.experiment_dir}")
        
        return final_stats
    
    def cleanup(self):
        """清理资源"""
        self.result_collector.cleanup()
        self.config_manager.clear_cache()
        self.logger.info("实验运行器已清理")