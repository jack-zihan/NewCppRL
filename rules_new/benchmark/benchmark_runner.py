"""
基准测试运行器 - 主协调器

支持config_dir参数进行灵活的配置切换
"""

import os
import sys
import time
import yaml
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from .scenario_generator import ScenarioGenerator
from .metric_collector import MetricCollector
from .visualization_manager import VisualizationManager
from .result_analyzer import ResultAnalyzer

# 算法导入（延迟导入，避免依赖问题）
# 实际使用时才导入具体算法

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    基准测试运行器
    
    协调所有组件，运行完整的基准测试流程
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 config_dir: Optional[Path] = None,
                 config_name: Optional[str] = None,
                 save_finished_picture: bool = False,
                 parallel: bool = True,
                 max_workers: Optional[int] = None):
        """
        初始化基准测试运行器
        
        Args:
            config_path: 直接指定的配置文件路径
            config_dir: 配置目录路径
            config_name: 配置文件名称（不含.yaml后缀）
            save_finished_picture: 是否在场景完成时保存图片
            parallel: 是否并行运行算法
            max_workers: 最大并行工作进程数
        """
        # 处理配置参数
        self.config_path = config_path
        self.config_name = config_name
        
        # 确定配置目录
        if config_path:
            # 如果直接指定了配置文件，从中提取目录
            self.config_dir = Path(config_path).parent
        elif config_dir:
            self.config_dir = Path(config_dir)
        else:
            # 默认配置目录
            self.config_dir = Path(__file__).parent.parent / 'configs'
        self.save_finished_picture = save_finished_picture
        self.parallel = parallel
        self.max_workers = max_workers or mp.cpu_count()
        
        # 加载配置
        self.config = self._load_config()
        
        # 创建输出目录
        self.output_dir = self._create_output_directory()
        
        # 初始化组件
        self._initialize_components()
        
        # 算法映射（延迟加载）
        self.algorithm_classes = None
        self._init_algorithm_classes()
        
        logger.info(f"基准测试运行器初始化完成 - 配置: {self.config_path or self.config_dir}")
    
    def _init_algorithm_classes(self):
        """初始化算法类映射（延迟加载）"""
        self.algorithm_classes = {}
        
        # 尝试导入各个算法
        algorithm_modules = {
            'JUMP': ('..algorithms.jump', 'JumpPathPlanner'),
            'SNAKE': ('..algorithms.snake', 'SnakePathPlanner'),
            'R_SNAKE': ('..algorithms.r_snake', 'RSnakePathPlanner'),
            'BCP': ('..algorithms.bcp', 'BCPPathPlanner'),
            'REACT': ('..algorithms.react', 'ReactPathPlanner'),
            'NN_baseline': ('..algorithms.nn_planner', 'NNPathPlanner'),
            'NN_ours': ('..algorithms.nn_planner', 'NNPathPlanner')
        }
        
        for alg_name, (module_path, class_name) in algorithm_modules.items():
            try:
                module = __import__(module_path, fromlist=[class_name])
                self.algorithm_classes[alg_name] = getattr(module, class_name)
                logger.debug(f"成功加载算法: {alg_name}")
            except ImportError as e:
                logger.warning(f"无法加载算法 {alg_name}: {e}")
                # 使用占位类
                self.algorithm_classes[alg_name] = None
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        config = {}
        
        # 1. 先加载基础配置（如果存在）
        base_config_path = self.config_dir / 'base_config.yaml'
        if base_config_path.exists():
            with open(base_config_path, 'r') as f:
                base_config = yaml.safe_load(f)
                if base_config:
                    config.update(base_config)
                    logger.debug(f"加载基础配置: {base_config_path}")
        
        # 2. 确定要加载的基准测试配置文件
        benchmark_config_path = None
        
        if self.config_path:
            # 优先级1：直接指定的配置文件
            benchmark_config_path = Path(self.config_path)
            if not benchmark_config_path.exists():
                raise FileNotFoundError(f"指定的配置文件不存在: {benchmark_config_path}")
                
        elif self.config_name:
            # 优先级2：config_dir + config_name
            if not self.config_name.endswith('.yaml'):
                benchmark_config_path = self.config_dir / f"{self.config_name}.yaml"
            else:
                benchmark_config_path = self.config_dir / self.config_name
                
            if not benchmark_config_path.exists():
                # 列出可用的配置文件
                available_configs = list(self.config_dir.glob("*.yaml"))
                config_list = "\n".join([f"  - {c.name}" for c in available_configs])
                raise FileNotFoundError(
                    f"配置文件 {benchmark_config_path} 不存在。\n"
                    f"可用的配置文件：\n{config_list}"
                )
        else:
            # 优先级3：查找默认的benchmark_config.yaml
            benchmark_config_path = self.config_dir / 'benchmark_config.yaml'
            
            if not benchmark_config_path.exists():
                # 尝试查找任何包含benchmark的yaml文件
                benchmark_configs = list(self.config_dir.glob("*benchmark*.yaml"))
                
                if len(benchmark_configs) == 1:
                    benchmark_config_path = benchmark_configs[0]
                    logger.info(f"自动使用找到的配置文件: {benchmark_config_path.name}")
                elif len(benchmark_configs) > 1:
                    config_list = "\n".join([f"  - {c.name}" for c in benchmark_configs])
                    raise ValueError(
                        f"找到多个基准测试配置文件，请明确指定：\n{config_list}\n"
                        f"使用 --config <文件路径> 或 --config-name <配置名>"
                    )
                else:
                    # 没有找到任何配置，使用默认配置
                    logger.warning("未找到基准测试配置文件，使用默认配置")
                    config.update(self._get_default_benchmark_config())
                    return config
        
        # 3. 加载基准测试配置
        if benchmark_config_path and benchmark_config_path.exists():
            with open(benchmark_config_path, 'r') as f:
                benchmark_config = yaml.safe_load(f)
                if benchmark_config:
                    config.update(benchmark_config)
                    logger.info(f"加载基准测试配置: {benchmark_config_path}")
        
        return config
    
    def _get_default_benchmark_config(self) -> Dict[str, Any]:
        """获取默认基准测试配置"""
        return {
            'benchmark': {
                'algorithms': {
                    'JUMP': {'enabled': True},
                    'SNAKE': {'enabled': True},
                    'R_SNAKE': {'enabled': True},
                    'BCP': {'enabled': True},
                    'REACT': {'enabled': True},
                    'NN_baseline': {
                        'enabled': True,
                        'model_path': 'ckpt/sac_baseline_continuous_t[01100]_r[2570.25=2509.63~2623.36].pt'
                    },
                    'NN_ours': {
                        'enabled': True,
                        'model_path': 'ckpt/t[02600]_r[2731.41=2717.75~2750.74].pt'
                    }
                },
                'scenarios': {
                    'seeds': [25, 27, 47, 21, 31],
                    'difficulties': ['easy', 'medium', 'hard'],
                    'weed_distributions': ['gaussian', 'uniform'],
                    'noise_levels': ['no_noise']
                },
                'metrics': {
                    'coverage_thresholds': [0.90, 0.95, 0.98],
                    'collect_trajectory': True,
                    'collect_collision': True
                },
                'output': {
                    'save_finished_picture': False,
                    'save_trajectories': True,
                    'save_statistics': True,
                    'create_comparison_plots': True
                }
            }
        }
    
    def _create_output_directory(self) -> Path:
        """创建输出目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path('benchmark_results') / f"benchmark_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置副本
        config_copy = output_dir / 'config'
        config_copy.mkdir(exist_ok=True)
        
        # 复制配置文件
        for config_file in self.config_dir.glob('*.yaml'):
            import shutil
            shutil.copy2(config_file, config_copy / config_file.name)
        
        return output_dir
    
    def _initialize_components(self):
        """初始化所有组件"""
        # 场景生成器
        self.scenario_generator = ScenarioGenerator(self.config)
        
        # 指标收集器
        # 使用默认值如果配置中没有指定
        metrics_config = self.config.get('benchmark', {}).get('metrics', {})
        coverage_thresholds = metrics_config.get('coverage_thresholds', [0.90, 0.95, 0.98])
        self.metric_collector = MetricCollector(coverage_thresholds)
        
        # 可视化管理器
        output_config = self.config.get('benchmark', {}).get('output', {})
        save_pic = self.save_finished_picture or output_config.get('save_finished_picture', False)
        self.visualization_manager = VisualizationManager(
            self.output_dir,
            save_finished_picture=save_pic
        )
        
        # 结果分析器
        self.result_analyzer = ResultAnalyzer(self.output_dir)
    
    def run_benchmark(self, 
                     algorithms: Optional[List[str]] = None,
                     scenarios: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        运行基准测试
        
        Args:
            algorithms: 要测试的算法列表，None则使用配置中的所有算法
            scenarios: 要测试的场景列表，None则生成所有场景
            
        Returns:
            测试结果汇总
        """
        start_time = time.time()
        
        # 确定要测试的算法
        if algorithms is None:
            algorithms = [
                name for name, cfg in self.config['benchmark']['algorithms'].items()
                if cfg.get('enabled', True)
            ]
        
        logger.info(f"准备测试 {len(algorithms)} 个算法: {algorithms}")
        
        # 生成或使用指定的场景
        if scenarios is None:
            scenarios = self._generate_scenarios()
        
        logger.info(f"准备测试 {len(scenarios)} 个场景")
        
        # 运行测试
        if self.parallel:
            results = self._run_parallel(algorithms, scenarios)
        else:
            results = self._run_sequential(algorithms, scenarios)
        
        # 分析结果
        summary = self._analyze_results(results)
        
        # 生成报告
        self._generate_report(summary, time.time() - start_time)
        
        logger.info(f"基准测试完成，耗时: {time.time() - start_time:.2f}秒")
        
        return summary
    
    def _generate_scenarios(self) -> List[Dict[str, Any]]:
        """生成测试场景"""
        scenario_config = self.config['benchmark']['scenarios']
        
        scenarios = self.scenario_generator.generate_all_scenarios(
            seeds=scenario_config['seeds'],
            difficulties=scenario_config['difficulties'],
            weed_distributions=scenario_config['weed_distributions'],
            noise_levels=scenario_config.get('noise_levels', ['no_noise'])
        )
        
        return scenarios
    
    def _run_sequential(self, 
                       algorithms: List[str],
                       scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """顺序运行测试"""
        results = []
        total_tests = len(algorithms) * len(scenarios)
        completed = 0
        
        for scenario in scenarios:
            scenario_results = {}
            
            for alg_name in algorithms:
                logger.info(f"运行测试 [{completed+1}/{total_tests}]: {alg_name} - {scenario['scenario_id']}")
                
                # 运行算法
                result = self._run_single_test(alg_name, scenario)
                
                # 收集结果
                scenario_results[alg_name] = result
                results.append(result)
                
                completed += 1
                
            # 创建场景对比图
            if self.config['benchmark']['output']['create_comparison_plots']:
                self.visualization_manager.create_algorithm_comparison(
                    scenario['scenario_id'],
                    scenario_results
                )
        
        return results
    
    def _run_parallel(self,
                     algorithms: List[str],
                     scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """并行运行测试"""
        results = []
        total_tests = len(algorithms) * len(scenarios)
        
        # 创建任务列表
        tasks = []
        for scenario in scenarios:
            for alg_name in algorithms:
                tasks.append((alg_name, scenario))
        
        # 并行执行
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(self._run_single_test, alg, scn): (alg, scn)
                for alg, scn in tasks
            }
            
            completed = 0
            for future in as_completed(future_to_task):
                alg_name, scenario = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    logger.info(f"完成测试 [{completed}/{total_tests}]: {alg_name} - {scenario['scenario_id']}")
                except Exception as e:
                    logger.error(f"测试失败 {alg_name} - {scenario['scenario_id']}: {e}")
        
        # 生成对比图
        if self.config['benchmark']['output']['create_comparison_plots']:
            self._create_comparison_plots(results)
        
        return results
    
    def _run_single_test(self,
                        algorithm_name: str,
                        scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行单次测试
        
        Args:
            algorithm_name: 算法名称
            scenario: 场景配置
            
        Returns:
            测试结果
        """
        try:
            # 创建环境
            env = self._create_environment(scenario)
            
            # 创建算法实例
            algorithm = self._create_algorithm(algorithm_name, env)
            
            # 运行算法
            start_time = time.time()
            trajectory_data = self._run_algorithm(algorithm, env)
            runtime = time.time() - start_time
            
            # 收集指标
            env_info = {
                'scenario_id': scenario['scenario_id'],
                'seed': scenario['seed'],
                'difficulty': scenario['difficulty'],
                'map_id': scenario['map_id'],
                'weed_distribution': scenario['weed_distribution'],
                'noise_level': scenario['noise_level']
            }
            
            algorithm_info = {
                'name': algorithm_name,
                'config': self.config['benchmark']['algorithms'].get(algorithm_name, {})
            }
            
            # 添加运行时间
            trajectory_data['time_info'] = {
                'total_time': runtime,
                'planning_time': 0,  # 可以细分规划时间
                'execution_time': runtime,
                'avg_step_time': runtime / len(trajectory_data.get('position_history', [1]))
            }
            
            metrics = self.metric_collector.collect_metrics(
                trajectory_data,
                env_info,
                algorithm_info
            )
            
            # 保存完成图片（如果启用）
            if trajectory_data.get('terminated') or trajectory_data.get('truncated'):
                completion_reason = self._get_completion_reason(trajectory_data)
                
                self.visualization_manager.save_scenario_completion(
                    algorithm_name,
                    scenario['scenario_id'],
                    env.get_state(),
                    trajectory_data,
                    completion_reason
                )
            
            # 组装完整结果
            result = {
                'algorithm': algorithm_name,
                'scenario': scenario,
                'metrics': metrics,
                'trajectory_data': trajectory_data if self.config['benchmark']['output']['save_trajectories'] else None,
                'runtime': runtime
            }
            
            return result
            
        except Exception as e:
            logger.error(f"测试执行失败 {algorithm_name} - {scenario['scenario_id']}: {e}")
            return {
                'algorithm': algorithm_name,
                'scenario': scenario,
                'metrics': {'error': str(e)},
                'trajectory_data': None,
                'runtime': -1
            }
    
    def _create_environment(self, scenario: Dict[str, Any]):
        """创建环境实例"""
        try:
            # 尝试导入实际的环境类
            from ..environment.rules_env import RulesEnv
            
            env_config = {
                'seed': scenario['seed'],
                'farm_vertices': scenario['farm_vertices'],
                'obstacles': scenario['obstacle_positions'],
                'weeds': scenario['weed_positions'],
                'noise_params': scenario['noise_params']
            }
            
            return RulesEnv(**env_config)
        except ImportError:
            logger.warning("无法导入RulesEnv，使用模拟环境")
            # 返回模拟环境用于测试
            class MockEnv:
                def __init__(self, **kwargs):
                    self.config = kwargs
                    
                def reset(self):
                    return {'observation': 0}
                
                def step(self, action):
                    return {'observation': 0}, 0, False, False, {}
                
                def get_state(self):
                    return {
                        'width': 600,
                        'height': 600,
                        'farm_vertices': self.config.get('farm_vertices', []),
                        'obstacles': self.config.get('obstacles', []),
                        'weeds': self.config.get('weeds', [])
                    }
            
            return MockEnv(**scenario)
    
    def _create_algorithm(self, algorithm_name: str, env) -> Any:
        """创建算法实例"""
        algorithm_class = self.algorithm_classes.get(algorithm_name)
        
        if algorithm_class is None:
            logger.warning(f"算法 {algorithm_name} 未加载，使用模拟算法")
            # 返回一个模拟算法对象
            class MockAlgorithm:
                def __init__(self, env, **kwargs):
                    self.env = env
                
                def get_action(self, obs):
                    # 返回随机动作用于测试
                    return 0
            
            return MockAlgorithm(env)
        
        algorithm_config = self.config['benchmark']['algorithms'].get(algorithm_name, {})
        
        # 对于NN算法，需要加载模型
        if 'NN' in algorithm_name:
            model_path = algorithm_config.get('model_path')
            return algorithm_class(env, model_path=model_path)
        else:
            return algorithm_class(env)
    
    def _run_algorithm(self, algorithm, env) -> Dict[str, Any]:
        """运行算法并收集轨迹数据"""
        trajectory_data = {
            'position_history': [],
            'coverage_history': [],
            'distance_history': [],
            'covered_weeds': set(),
            'collision_info': {},
            'terminated': False,
            'truncated': False
        }
        
        # 重置环境
        obs = env.reset()
        
        # 运行算法
        max_steps = 1000
        for step in range(max_steps):
            # 获取动作
            action = algorithm.get_action(obs)
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 记录轨迹
            trajectory_data['position_history'].append(info.get('position', [0, 0]))
            trajectory_data['coverage_history'].append(info.get('coverage', 0))
            trajectory_data['distance_history'].append(info.get('total_distance', 0))
            
            # 检查碰撞
            if info.get('collision', False):
                trajectory_data['collision_info'] = {
                    'occurred': True,
                    'step': step,
                    'position': info.get('position'),
                    'distance': info.get('total_distance', 0),
                    'type': info.get('collision_type', 'unknown')
                }
            
            # 更新覆盖的杂草
            if 'covered_weeds' in info:
                trajectory_data['covered_weeds'].update(info['covered_weeds'])
            
            trajectory_data['terminated'] = terminated
            trajectory_data['truncated'] = truncated
            
            if terminated or truncated:
                break
        
        return trajectory_data
    
    def _get_completion_reason(self, trajectory_data: Dict[str, Any]) -> str:
        """获取完成原因"""
        if trajectory_data.get('collision_info', {}).get('occurred'):
            return 'collision'
        elif trajectory_data.get('coverage_history', [0])[-1] >= 0.98:
            return 'success'
        elif trajectory_data.get('truncated'):
            return 'timeout'
        else:
            return 'unknown'
    
    def _create_comparison_plots(self, results: List[Dict[str, Any]]):
        """创建对比图"""
        # 按场景分组结果
        scenarios_results = {}
        for result in results:
            scenario_id = result['scenario']['scenario_id']
            if scenario_id not in scenarios_results:
                scenarios_results[scenario_id] = {}
            
            alg_name = result['algorithm']
            scenarios_results[scenario_id][alg_name] = result['trajectory_data']
        
        # 为每个场景创建对比图
        for scenario_id, alg_results in scenarios_results.items():
            self.visualization_manager.create_algorithm_comparison(scenario_id, alg_results)
    
    def _analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析测试结果"""
        return self.result_analyzer.analyze(results)
    
    def _generate_report(self, summary: Dict[str, Any], total_time: float):
        """生成测试报告"""
        report_path = self.output_dir / 'benchmark_report.yaml'
        
        report = {
            'benchmark_info': {
                'timestamp': datetime.now().isoformat(),
                'total_runtime': total_time,
                'config_dir': str(self.config_dir),
                'output_dir': str(self.output_dir),
                'save_finished_picture': self.save_finished_picture
            },
            'summary': summary,
            'visualization_summary': self.visualization_manager.export_visualization_summary()
        }
        
        with open(report_path, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)
        
        logger.info(f"测试报告已保存: {report_path}")
    
    def cleanup(self):
        """清理资源"""
        pass