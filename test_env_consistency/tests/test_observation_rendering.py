#!/usr/bin/env python3
"""
观测和渲染系统一致性测试套件

验证环境观测生成和渲染系统的数据格式一致性、边界条件处理、性能影响。
作者：Test Engineer
日期：2025-01-15
"""

import sys
import time
import json
import numpy as np
import pytest
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
from contextlib import contextmanager
import tracemalloc
import psutil
import gc

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from envs.cpp_env_v2 import CppEnv as CppEnvV2
from envs.cpp_env_v3 import CppEnv as CppEnvV3
from envs.cpp_env_base_copy import CppEnvBase


@dataclass
class ObservationTestResult:
    """观测测试结果"""
    test_name: str
    passed: bool
    shape_match: bool
    dtype_match: bool
    range_valid: bool
    details: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class RenderTestResult:
    """渲染测试结果"""
    test_name: str
    passed: bool
    shape_match: bool
    dtype_match: bool
    time_ms: float
    memory_mb: float
    details: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """性能指标"""
    avg_time_ms: float
    max_time_ms: float
    min_time_ms: float
    std_time_ms: float
    memory_peak_mb: float
    memory_avg_mb: float


class ObservationRenderingTester:
    """观测和渲染系统测试器"""
    
    def __init__(self, env_class, env_config: Optional[Dict] = None):
        """
        初始化测试器
        
        Args:
            env_class: 环境类
            env_config: 环境配置
        """
        self.env_class = env_class
        self.env_config = env_config or self._get_default_config()
        self.test_results = {
            'observation': [],
            'rendering': [],
            'performance': {},
            'compatibility': []
        }
        
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'map_size': (256, 256),
            'vision_length': 28,
            'vision_angle': 75,
            'state_size': (256, 256),
            'state_downsize': (128, 128),
            'sgcnn_size': 16,
            'use_sgcnn': False,
            'use_global_obs': False,
            'use_apf': True,
            'use_traj': False,
            'render_mode': None,
            'render_repeat_times': 1,
        }
    
    @contextmanager
    def measure_performance(self):
        """性能测量上下文管理器"""
        gc.collect()
        tracemalloc.start()
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.perf_counter()
        yield
        end_time = time.perf_counter()
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        
        self.last_perf = {
            'time_ms': (end_time - start_time) * 1000,
            'memory_peak_mb': peak / 1024 / 1024,
            'memory_delta_mb': mem_after - mem_before
        }
    
    # ==================== 观测空间测试 ====================
    
    def test_observation_space_definition(self) -> ObservationTestResult:
        """测试观测空间定义一致性"""
        test_configs = [
            {'use_sgcnn': False, 'use_traj': False},
            {'use_sgcnn': False, 'use_traj': True},
            {'use_sgcnn': True, 'use_global_obs': False},
            {'use_sgcnn': True, 'use_global_obs': True},
        ]
        
        results = []
        for config in test_configs:
            env_config = {**self.env_config, **config}
            env = self.env_class(**env_config)
            obs, _ = env.reset()
            
            # 获取观测
            if isinstance(obs, dict):
                obs_data = obs.get('observation')
            else:
                obs_data = obs
            
            # 计算期望shape
            expected_shape = self._calculate_expected_shape(env_config)
            
            # 验证
            shape_match = obs_data.shape == expected_shape
            dtype_match = obs_data.dtype in [np.float32, np.float64]
            
            result = ObservationTestResult(
                test_name=f"obs_space_{config}",
                passed=shape_match and dtype_match,
                shape_match=shape_match,
                dtype_match=dtype_match,
                range_valid=True,  # 将在下一个测试中验证
                details={
                    'config': config,
                    'expected_shape': expected_shape,
                    'actual_shape': obs_data.shape,
                    'dtype': str(obs_data.dtype)
                }
            )
            results.append(result)
            env.close()
        
        self.test_results['observation'].extend(results)
        return results[-1] if results else None
    
    def _calculate_expected_shape(self, config: Dict) -> Tuple:
        """计算期望的观测形状"""
        base_channels = 4 if not config.get('use_traj') else 5
        
        if config.get('use_sgcnn'):
            # SGCNN模式：4层金字塔
            channels = base_channels * 4
            if config.get('use_global_obs'):
                # 添加全局观测通道
                global_size = config.get('sgcnn_size', 16)
                channels += base_channels
            return (channels, config.get('sgcnn_size', 16), config.get('sgcnn_size', 16))
        else:
            # 普通模式
            downsize = config.get('state_downsize', (128, 128))
            return (base_channels, downsize[0], downsize[1])
    
    def test_observation_data_range(self) -> ObservationTestResult:
        """测试观测数据范围"""
        env = self.env_class(**self.env_config)
        obs, _ = env.reset()
        
        test_steps = 100
        range_violations = []
        
        for i in range(test_steps):
            action = env.action_space.sample()
            obs, _, done, _, _ = env.step(action)
            
            if isinstance(obs, dict):
                obs_data = obs.get('observation')
            else:
                obs_data = obs
            
            # 检查范围
            min_val, max_val = obs_data.min(), obs_data.max()
            
            # APF值应该在[0, 1]范围内（经过gamma折扣）
            if min_val < -0.01 or max_val > 1.01:  # 允许小的浮点误差
                range_violations.append({
                    'step': i,
                    'min': float(min_val),
                    'max': float(max_val)
                })
            
            if done:
                obs, _ = env.reset()
        
        env.close()
        
        result = ObservationTestResult(
            test_name="observation_range",
            passed=len(range_violations) == 0,
            shape_match=True,
            dtype_match=True,
            range_valid=len(range_violations) == 0,
            details={
                'test_steps': test_steps,
                'violations': range_violations[:10],  # 只记录前10个违反
                'violation_count': len(range_violations)
            }
        )
        
        self.test_results['observation'].append(result)
        return result
    
    def test_observation_boundary_conditions(self) -> List[ObservationTestResult]:
        """测试边界条件"""
        boundary_configs = [
            {'vision_length': 1, 'test_name': 'min_vision'},
            {'vision_length': 100, 'test_name': 'max_vision'},
            {'vision_angle': 10, 'test_name': 'narrow_angle'},
            {'vision_angle': 360, 'test_name': 'full_angle'},
            {'sgcnn_size': 8, 'test_name': 'small_sgcnn'},
            {'sgcnn_size': 64, 'test_name': 'large_sgcnn'},
        ]
        
        results = []
        for config in boundary_configs:
            test_name = config.pop('test_name')
            env_config = {**self.env_config, **config}
            
            try:
                env = self.env_class(**env_config)
                obs, _ = env.reset()
                
                # 测试边缘位置
                env.agent.x = 0
                env.agent.y = 0
                obs_edge = env.observation()
                
                # 测试中心位置
                env.agent.x = env.dimensions[0] // 2
                env.agent.y = env.dimensions[1] // 2
                obs_center = env.observation()
                
                passed = True
                error_msg = None
                env.close()
                
            except Exception as e:
                passed = False
                error_msg = str(e)
                obs_edge = None
                obs_center = None
            
            result = ObservationTestResult(
                test_name=f"boundary_{test_name}",
                passed=passed,
                shape_match=True,
                dtype_match=True,
                range_valid=True,
                details={
                    'config': config,
                    'edge_obs_shape': obs_edge['observation'].shape if obs_edge else None,
                    'center_obs_shape': obs_center['observation'].shape if obs_center else None,
                },
                error_message=error_msg
            )
            results.append(result)
        
        self.test_results['observation'].extend(results)
        return results
    
    def test_observation_temporal_consistency(self) -> ObservationTestResult:
        """测试观测时序一致性"""
        env = self.env_class(**self.env_config)
        obs_t0, _ = env.reset()
        
        # 不执行动作，观测应该保持不变
        obs_t1 = env.observation()
        
        if isinstance(obs_t0, dict):
            obs_t0_data = obs_t0['observation']
            obs_t1_data = obs_t1['observation']
        else:
            obs_t0_data = obs_t0
            obs_t1_data = obs_t1
        
        # 验证两次观测是否相同
        consistency = np.allclose(obs_t0_data, obs_t1_data, rtol=1e-5)
        
        # 执行动作后观测应该改变
        action = env.action_space.sample()
        obs_t2, _, _, _, _ = env.step(action)
        
        if isinstance(obs_t2, dict):
            obs_t2_data = obs_t2['observation']
        else:
            obs_t2_data = obs_t2
        
        # 验证观测是否改变
        changed = not np.allclose(obs_t1_data, obs_t2_data, rtol=1e-5)
        
        env.close()
        
        result = ObservationTestResult(
            test_name="temporal_consistency",
            passed=consistency and changed,
            shape_match=True,
            dtype_match=True,
            range_valid=True,
            details={
                'static_consistency': consistency,
                'action_changes_obs': changed,
                'max_static_diff': float(np.max(np.abs(obs_t0_data - obs_t1_data))),
                'max_action_diff': float(np.max(np.abs(obs_t1_data - obs_t2_data)))
            }
        )
        
        self.test_results['observation'].append(result)
        return result
    
    # ==================== 渲染测试 ====================
    
    def test_render_output_format(self) -> RenderTestResult:
        """测试渲染输出格式"""
        render_configs = [
            {'render_mode': None},
            {'render_mode': 'rgb_array'},
            {'render_mode': 'rgb_array', 'render_repeat_times': 2},
            {'render_mode': 'rgb_array', 'render_repeat_times': 4},
        ]
        
        results = []
        for config in render_configs:
            env_config = {**self.env_config, **config}
            env = self.env_class(**env_config)
            env.reset()
            
            with self.measure_performance():
                if config['render_mode'] is not None:
                    rendered = env.render()
                else:
                    rendered = None
            
            if rendered is not None:
                # 验证格式
                expected_h = env.dimensions[1] * config.get('render_repeat_times', 1)
                expected_w = env.dimensions[0] * config.get('render_repeat_times', 1)
                expected_shape = (expected_h, expected_w, 3)
                
                shape_match = rendered.shape == expected_shape
                dtype_match = rendered.dtype == np.uint8
                
                # 验证值范围
                value_valid = (rendered.min() >= 0) and (rendered.max() <= 255)
            else:
                shape_match = True  # No render mode
                dtype_match = True
                value_valid = True
            
            result = RenderTestResult(
                test_name=f"render_{config}",
                passed=shape_match and dtype_match and value_valid,
                shape_match=shape_match,
                dtype_match=dtype_match,
                time_ms=self.last_perf['time_ms'],
                memory_mb=self.last_perf['memory_peak_mb'],
                details={
                    'config': config,
                    'rendered_shape': rendered.shape if rendered is not None else None,
                    'value_valid': value_valid
                }
            )
            results.append(result)
            env.close()
        
        self.test_results['rendering'].extend(results)
        return results[-1] if results else None
    
    def test_render_performance(self) -> Dict[str, PerformanceMetrics]:
        """测试渲染性能"""
        configs = [
            {'name': 'no_render', 'render_mode': None},
            {'name': 'basic', 'render_mode': 'rgb_array'},
            {'name': 'scaled_2x', 'render_mode': 'rgb_array', 'render_repeat_times': 2},
            {'name': 'scaled_4x', 'render_mode': 'rgb_array', 'render_repeat_times': 4},
        ]
        
        performance_results = {}
        
        for config in configs:
            name = config.pop('name')
            env_config = {**self.env_config, **config}
            env = self.env_class(**env_config)
            env.reset()
            
            times = []
            memories = []
            
            # 运行多次测试
            for _ in range(50):
                with self.measure_performance():
                    if config.get('render_mode') is not None:
                        _ = env.render()
                    # 执行一个step来模拟真实使用
                    action = env.action_space.sample()
                    env.step(action)
                
                times.append(self.last_perf['time_ms'])
                memories.append(self.last_perf['memory_peak_mb'])
            
            env.close()
            
            # 计算统计数据
            metrics = PerformanceMetrics(
                avg_time_ms=float(np.mean(times)),
                max_time_ms=float(np.max(times)),
                min_time_ms=float(np.min(times)),
                std_time_ms=float(np.std(times)),
                memory_peak_mb=float(np.max(memories)),
                memory_avg_mb=float(np.mean(memories))
            )
            
            performance_results[name] = metrics
        
        self.test_results['performance']['rendering'] = performance_results
        return performance_results
    
    def test_render_consistency(self) -> RenderTestResult:
        """测试渲染一致性"""
        env = self.env_class(**{**self.env_config, 'render_mode': 'rgb_array'})
        env.reset()
        
        # 相同状态应该产生相同的渲染
        render1 = env.render()
        render2 = env.render()
        
        same_state_match = np.array_equal(render1, render2)
        
        # 动作后渲染应该改变
        action = env.action_space.sample()
        env.step(action)
        render3 = env.render()
        
        action_changes_render = not np.array_equal(render2, render3)
        
        env.close()
        
        result = RenderTestResult(
            test_name="render_consistency",
            passed=same_state_match and action_changes_render,
            shape_match=True,
            dtype_match=True,
            time_ms=0,
            memory_mb=0,
            details={
                'same_state_consistency': same_state_match,
                'action_changes_render': action_changes_render,
                'pixel_diff_count': int(np.sum(render2 != render3))
            }
        )
        
        self.test_results['rendering'].append(result)
        return result
    
    # ==================== 数据流验证 ====================
    
    def test_observation_dataflow(self) -> Dict[str, Any]:
        """测试观测数据流"""
        env = self.env_class(**self.env_config)
        env.reset()
        
        dataflow_checks = {
            'apf_calculation': self._check_apf_calculation(env),
            'noise_application': self._check_noise_application(env),
            'rotation_cropping': self._check_rotation_cropping(env),
            'sgcnn_processing': self._check_sgcnn_processing(env) if self.env_config.get('use_sgcnn') else None,
        }
        
        env.close()
        
        self.test_results['compatibility'].append({
            'test': 'observation_dataflow',
            'results': dataflow_checks
        })
        
        return dataflow_checks
    
    def _check_apf_calculation(self, env) -> Dict[str, Any]:
        """检查APF计算"""
        if hasattr(env, 'get_discounted_apf'):
            # 测试APF计算
            test_map = np.random.randint(0, 2, (10, 10), dtype=np.uint8)
            apf_result = env.get_discounted_apf(test_map, max_step=30)
            
            return {
                'has_method': True,
                'output_range': (float(apf_result.min()), float(apf_result.max())),
                'contains_zeros': bool(np.any(apf_result == 0)),
                'shape_preserved': apf_result.shape == test_map.shape
            }
        return {'has_method': False}
    
    def _check_noise_application(self, env) -> Dict[str, Any]:
        """检查噪声应用"""
        results = {}
        
        # 检查位置噪声
        if hasattr(env, 'noise_position'):
            results['position_noise'] = {
                'enabled': env.noise_position is not None and env.noise_position > 0,
                'value': float(env.noise_position) if env.noise_position else 0
            }
        
        # 检查方向噪声
        if hasattr(env, 'noise_direction'):
            results['direction_noise'] = {
                'enabled': env.noise_direction is not None and env.noise_direction > 0,
                'value': float(env.noise_direction) if env.noise_direction else 0
            }
        
        # 检查杂草噪声
        if hasattr(env, 'noise_weed'):
            results['weed_noise'] = {
                'enabled': env.noise_weed is not None and env.noise_weed > 0,
                'value': float(env.noise_weed) if env.noise_weed else 0
            }
        
        return results
    
    def _check_rotation_cropping(self, env) -> Dict[str, Any]:
        """检查旋转裁剪"""
        if hasattr(env, 'get_rotated_obs'):
            # 创建测试数据
            test_maps = np.random.rand(env.dimensions[1], env.dimensions[0], 4).astype(np.float32)
            test_mask = [0., 0., 1., 0.]
            
            try:
                rotated_obs = env.get_rotated_obs(test_maps, test_mask)
                return {
                    'has_method': True,
                    'output_shape': rotated_obs.shape,
                    'preserves_channels': rotated_obs.shape[-1] == test_maps.shape[-1]
                }
            except Exception as e:
                return {
                    'has_method': True,
                    'error': str(e)
                }
        return {'has_method': False}
    
    def _check_sgcnn_processing(self, env) -> Dict[str, Any]:
        """检查SGCNN处理"""
        if hasattr(env, 'get_sgcnn_obs'):
            # 创建测试观测
            test_obs = np.random.rand(4, 128, 128).astype(np.float32)
            
            try:
                sgcnn_obs = env.get_sgcnn_obs(test_obs, None, None)
                return {
                    'has_method': True,
                    'output_shape': sgcnn_obs.shape,
                    'pyramid_levels': sgcnn_obs.shape[0] // test_obs.shape[0]
                }
            except Exception as e:
                return {
                    'has_method': True,
                    'error': str(e)
                }
        return {'has_method': False}
    
    # ==================== 兼容性测试 ====================
    
    def test_torchrl_compatibility(self) -> Dict[str, Any]:
        """测试与TorchRL的兼容性"""
        try:
            import torch
            from torchrl.envs import GymWrapper
            
            env = self.env_class(**self.env_config)
            
            # 尝试包装环境
            try:
                wrapped_env = GymWrapper(env)
                obs_spec = wrapped_env.observation_spec
                action_spec = wrapped_env.action_spec
                
                # 测试reset和step
                td = wrapped_env.reset()
                action = wrapped_env.action_spec.rand()
                td = wrapped_env.step(action)
                
                compatible = True
                error = None
            except Exception as e:
                compatible = False
                error = str(e)
                obs_spec = None
                action_spec = None
            
            env.close()
            
            result = {
                'compatible': compatible,
                'has_obs_spec': obs_spec is not None,
                'has_action_spec': action_spec is not None,
                'error': error
            }
            
        except ImportError:
            result = {
                'compatible': False,
                'error': 'TorchRL not installed'
            }
        
        self.test_results['compatibility'].append({
            'test': 'torchrl',
            'result': result
        })
        
        return result
    
    # ==================== 报告生成 ====================
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """生成测试报告"""
        report = []
        report.append("# 观测和渲染系统测试报告\n")
        report.append(f"测试时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"环境类：{self.env_class.__name__}\n\n")
        
        # 观测测试结果
        report.append("## 1. 观测系统测试结果\n\n")
        report.append("### 1.1 观测空间定义\n")
        report.append("| 测试项 | 通过 | Shape匹配 | 类型匹配 | 范围有效 | 详情 |\n")
        report.append("|--------|------|-----------|----------|----------|------|\n")
        
        for result in self.test_results['observation']:
            report.append(f"| {result.test_name} | {'✅' if result.passed else '❌'} | "
                         f"{'✅' if result.shape_match else '❌'} | "
                         f"{'✅' if result.dtype_match else '❌'} | "
                         f"{'✅' if result.range_valid else '❌'} | "
                         f"{json.dumps(result.details, ensure_ascii=False)[:50]}... |\n")
        
        # 渲染测试结果
        report.append("\n## 2. 渲染系统测试结果\n\n")
        report.append("### 2.1 渲染格式和性能\n")
        report.append("| 测试项 | 通过 | 时间(ms) | 内存(MB) | 详情 |\n")
        report.append("|--------|------|----------|----------|------|\n")
        
        for result in self.test_results['rendering']:
            report.append(f"| {result.test_name} | {'✅' if result.passed else '❌'} | "
                         f"{result.time_ms:.2f} | {result.memory_mb:.2f} | "
                         f"{json.dumps(result.details, ensure_ascii=False)[:50]}... |\n")
        
        # 性能分析
        if 'rendering' in self.test_results['performance']:
            report.append("\n### 2.2 渲染性能对比\n")
            report.append("| 配置 | 平均时间(ms) | 最大时间(ms) | 内存峰值(MB) |\n")
            report.append("|------|--------------|--------------|-------------|\n")
            
            for name, metrics in self.test_results['performance']['rendering'].items():
                report.append(f"| {name} | {metrics.avg_time_ms:.2f} | "
                             f"{metrics.max_time_ms:.2f} | {metrics.memory_peak_mb:.2f} |\n")
        
        # 兼容性测试
        report.append("\n## 3. 兼容性测试结果\n\n")
        for compat_test in self.test_results['compatibility']:
            report.append(f"### {compat_test['test']}\n")
            report.append(f"```json\n{json.dumps(compat_test, indent=2, ensure_ascii=False)}\n```\n\n")
        
        # 总结
        report.append("\n## 4. 测试总结\n\n")
        
        obs_passed = sum(1 for r in self.test_results['observation'] if r.passed)
        obs_total = len(self.test_results['observation'])
        render_passed = sum(1 for r in self.test_results['rendering'] if r.passed)
        render_total = len(self.test_results['rendering'])
        
        report.append(f"- 观测测试通过率：{obs_passed}/{obs_total} ({obs_passed/obs_total*100:.1f}%)\n")
        report.append(f"- 渲染测试通过率：{render_passed}/{render_total} ({render_passed/render_total*100:.1f}%)\n")
        
        # 保存报告
        report_text = ''.join(report)
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
        
        return report_text
    
    def run_all_tests(self) -> None:
        """运行所有测试"""
        print("🔍 开始观测和渲染系统测试...")
        
        # 观测测试
        print("\n📊 测试观测空间定义...")
        self.test_observation_space_definition()
        
        print("📊 测试观测数据范围...")
        self.test_observation_data_range()
        
        print("📊 测试边界条件...")
        self.test_observation_boundary_conditions()
        
        print("📊 测试时序一致性...")
        self.test_observation_temporal_consistency()
        
        # 渲染测试
        print("\n🎨 测试渲染输出格式...")
        self.test_render_output_format()
        
        print("🎨 测试渲染性能...")
        self.test_render_performance()
        
        print("🎨 测试渲染一致性...")
        self.test_render_consistency()
        
        # 数据流测试
        print("\n🔄 测试观测数据流...")
        self.test_observation_dataflow()
        
        # 兼容性测试
        print("\n🔗 测试TorchRL兼容性...")
        self.test_torchrl_compatibility()
        
        print("\n✅ 测试完成！")


def compare_environments(env_v2_config: Dict, env_v3_config: Dict) -> Dict[str, Any]:
    """比较两个环境版本的一致性"""
    print("\n🔀 比较环境版本差异...")
    
    # 测试V2版本
    print("\n📦 测试环境V2...")
    tester_v2 = ObservationRenderingTester(CppEnvV2, env_v2_config)
    tester_v2.run_all_tests()
    report_v2 = tester_v2.generate_report(
        "/home/lzh/NewCppRL/test_env_consistency/reports/obs_render_test_v2.md"
    )
    
    # 测试V3版本
    print("\n📦 测试环境V3...")
    tester_v3 = ObservationRenderingTester(CppEnvV3, env_v3_config)
    tester_v3.run_all_tests()
    report_v3 = tester_v3.generate_report(
        "/home/lzh/NewCppRL/test_env_consistency/reports/obs_render_test_v3.md"
    )
    
    # 比较结果
    comparison = {
        'v2_results': tester_v2.test_results,
        'v3_results': tester_v3.test_results,
        'differences': []
    }
    
    # 找出差异
    for test_type in ['observation', 'rendering']:
        v2_tests = {r.test_name: r for r in tester_v2.test_results[test_type]}
        v3_tests = {r.test_name: r for r in tester_v3.test_results[test_type]}
        
        for test_name in v2_tests:
            if test_name in v3_tests:
                v2_result = v2_tests[test_name]
                v3_result = v3_tests[test_name]
                
                if v2_result.passed != v3_result.passed:
                    comparison['differences'].append({
                        'test_type': test_type,
                        'test_name': test_name,
                        'v2_passed': v2_result.passed,
                        'v3_passed': v3_result.passed,
                        'details': {
                            'v2': v2_result.details,
                            'v3': v3_result.details
                        }
                    })
    
    # 生成比较报告
    comparison_report = []
    comparison_report.append("# 环境版本比较报告\n\n")
    comparison_report.append(f"发现 {len(comparison['differences'])} 个差异\n\n")
    
    for diff in comparison['differences']:
        comparison_report.append(f"### {diff['test_type']} - {diff['test_name']}\n")
        comparison_report.append(f"- V2: {'✅' if diff['v2_passed'] else '❌'}\n")
        comparison_report.append(f"- V3: {'✅' if diff['v3_passed'] else '❌'}\n")
        comparison_report.append(f"- 详情：{json.dumps(diff['details'], indent=2, ensure_ascii=False)}\n\n")
    
    comparison_report_text = ''.join(comparison_report)
    
    # 保存比较报告
    with open("/home/lzh/NewCppRL/test_env_consistency/reports/environment_comparison.md", 'w', encoding='utf-8') as f:
        f.write(comparison_report_text)
    
    print(f"\n📝 比较报告已保存")
    print(f"发现 {len(comparison['differences'])} 个差异")
    
    return comparison


if __name__ == "__main__":
    # 默认配置
    default_config = {
        'map_size': (256, 256),
        'vision_length': 28,
        'vision_angle': 75,
        'state_size': (256, 256),
        'state_downsize': (128, 128),
        'sgcnn_size': 16,
        'use_sgcnn': False,
        'use_global_obs': False,
        'use_apf': True,
        'use_traj': False,
        'render_mode': 'rgb_array',
        'render_repeat_times': 1,
    }
    
    # 运行单个环境测试
    print("=" * 80)
    print("观测和渲染系统一致性测试")
    print("=" * 80)
    
    # 测试V2版本
    tester = ObservationRenderingTester(CppEnvV2, default_config)
    tester.run_all_tests()
    report = tester.generate_report(
        "/home/lzh/NewCppRL/test_env_consistency/reports/obs_render_test_report.md"
    )
    print("\n" + report)
    
    # 比较V2和V3版本
    print("\n" + "=" * 80)
    print("环境版本比较测试")
    print("=" * 80)
    
    comparison = compare_environments(default_config, default_config)
    
    print("\n✅ 所有测试完成！")
    print(f"📁 报告已保存到: /home/lzh/NewCppRL/test_env_consistency/reports/")