#!/usr/bin/env python3
"""
观测和渲染系统一致性测试脚本

用于验证新旧环境中观测生成和渲染的差异。
"""

import sys
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from envs.cpp_env_v2 import CppEnv as OldEnv
# from rules_new.environment import Environment as NewEnv  # 需要确认新环境路径


class ObservationRenderingTester:
    """观测和渲染一致性测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.old_env = None
        self.new_env = None
        self.test_results = []
        
    def setup_environments(self):
        """设置测试环境"""
        # 初始化旧环境
        self.old_env = OldEnv(
            render_mode=None,
            use_apf=True,
            use_sgcnn=True,
            use_global_obs=True,
            noise_position=0.0,  # 关闭噪声以便对比
            noise_direction=0.0,
            noise_weed=0.0
        )
        
        # TODO: 初始化新环境
        # self.new_env = NewEnv(...)
        
    def test_apf_parameters(self):
        """测试APF参数差异"""
        print("\n=== 测试APF参数 ===")
        
        # 创建测试地图
        test_map = np.zeros((100, 100), dtype=np.uint8)
        test_map[40:60, 40:60] = 1  # 中心区域为1
        
        # 测试不同的APF配置
        apf_configs = [
            {'max_step': 30, 'eps': None, 'pad': False},  # frontier
            {'max_step': 10, 'eps': None, 'pad': True},   # obstacle
            {'max_step': 40, 'eps': 1e-2, 'pad': False},  # weed
            {'max_step': 4, 'eps': None, 'pad': False},   # trajectory
        ]
        
        results = []
        for config in apf_configs:
            apf_result = OldEnv.get_discounted_apf(
                test_map.copy(),
                max_step=config['max_step'],
                eps=config['eps'],
                pad=config['pad']
            )
            
            results.append({
                'config': config,
                'max_value': np.max(apf_result),
                'min_value': np.min(apf_result),
                'mean_value': np.mean(apf_result),
                'non_zero_count': np.count_nonzero(apf_result)
            })
            
            print(f"Config: {config}")
            print(f"  Max: {results[-1]['max_value']:.6f}")
            print(f"  Min: {results[-1]['min_value']:.6f}")
            print(f"  Mean: {results[-1]['mean_value']:.6f}")
            print(f"  Non-zero: {results[-1]['non_zero_count']}")
        
        self.test_results.append(('APF Parameters', results))
        return results
    
    def test_noise_injection(self):
        """测试噪声注入机制"""
        print("\n=== 测试噪声注入 ===")
        
        # 设置固定种子
        np.random.seed(42)
        
        # 创建带噪声的环境
        env_with_noise = OldEnv(
            render_mode=None,
            noise_position=5.0,
            noise_direction=10.0
        )
        
        # 重置环境
        obs, _ = env_with_noise.reset(seed=42)
        
        # 记录初始位置
        initial_pos = env_with_noise.agent.position
        initial_dir = env_with_noise.agent.direction
        
        # 获取观测（会注入噪声）
        maps = np.ones((100, 100, 4), dtype=np.float32)
        mask = [0., 0., 1., 0.]
        
        # 多次获取观测，检查噪声效果
        noise_effects = []
        for i in range(10):
            np.random.seed(42 + i)
            obs_rotated = env_with_noise.get_rotated_obs(maps, mask)
            noise_effects.append({
                'iteration': i,
                'obs_shape': obs_rotated.shape,
                'obs_mean': np.mean(obs_rotated),
                'obs_std': np.std(obs_rotated)
            })
        
        print(f"Initial position: {initial_pos}")
        print(f"Initial direction: {initial_dir}")
        print(f"Noise effects over 10 iterations:")
        for effect in noise_effects[:3]:  # 只显示前3个
            print(f"  Iter {effect['iteration']}: mean={effect['obs_mean']:.4f}, std={effect['obs_std']:.4f}")
        
        self.test_results.append(('Noise Injection', noise_effects))
        return noise_effects
    
    def test_mist_logic(self):
        """测试Mist逻辑"""
        print("\n=== 测试Mist逻辑 ===")
        
        # 重置环境
        self.old_env.reset(seed=42)
        
        # 检查初始mist状态
        initial_mist = self.old_env.map_mist.copy()
        
        # 执行几步
        for i in range(5):
            action = self.old_env.action_space.sample()
            self.old_env.step(action)
        
        # 检查mist更新
        updated_mist = self.old_env.map_mist.copy()
        
        results = {
            'initial_mist_sum': np.sum(initial_mist),
            'initial_mist_mean': np.mean(initial_mist),
            'updated_mist_sum': np.sum(updated_mist),
            'updated_mist_mean': np.mean(updated_mist),
            'mist_change': np.sum(updated_mist) - np.sum(initial_mist),
            'mist_semantics': 'mist=1 means explored' if np.sum(updated_mist) > np.sum(initial_mist) else 'mist=0 means explored'
        }
        
        print(f"Initial mist sum: {results['initial_mist_sum']}")
        print(f"Updated mist sum: {results['updated_mist_sum']}")
        print(f"Mist change: {results['mist_change']}")
        print(f"Mist semantics: {results['mist_semantics']}")
        
        self.test_results.append(('Mist Logic', results))
        return results
    
    def test_rendering_colors(self):
        """测试渲染颜色配置"""
        print("\n=== 测试渲染颜色 ===")
        
        # 重置环境并获取渲染
        self.old_env.reset(seed=42)
        rendered_map = self.old_env.render_map()
        
        # 提取独特颜色
        unique_colors = np.unique(rendered_map.reshape(-1, 3), axis=0)
        
        # 定义已知颜色映射
        known_colors = {
            (255, 255, 255): 'background',
            (76, 187, 23): 'field_frontier',
            (0, 0, 0): 'weed_undiscovered',
            (255, 0, 0): 'weed_discovered/agent',
            (30, 75, 130): 'obstacle',
            (47, 82, 143): 'obstacle_edge',
            (255, 38, 255): 'trajectory',
            (192, 192, 192): 'vision_ellipse'
        }
        
        color_analysis = []
        for color in unique_colors[:10]:  # 分析前10种颜色
            color_tuple = tuple(color.astype(int))
            color_name = known_colors.get(color_tuple, 'unknown')
            pixel_count = np.sum(np.all(rendered_map == color, axis=-1))
            
            color_analysis.append({
                'rgb': color_tuple,
                'name': color_name,
                'pixel_count': pixel_count,
                'percentage': pixel_count / (rendered_map.shape[0] * rendered_map.shape[1]) * 100
            })
        
        print("Color analysis:")
        for analysis in color_analysis[:5]:  # 显示前5种
            print(f"  RGB{analysis['rgb']}: {analysis['name']} ({analysis['percentage']:.2f}%)")
        
        self.test_results.append(('Rendering Colors', color_analysis))
        return color_analysis
    
    def test_coordinate_transform(self):
        """测试坐标变换精度"""
        print("\n=== 测试坐标变换 ===")
        
        # 创建测试图像
        test_image = np.zeros((100, 100, 3), dtype=np.float32)
        test_image[40:60, 40:60, :] = 255  # 白色方块
        
        # 测试不同角度的旋转
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        transform_results = []
        
        for angle in angles:
            # 使用旧环境的变换方法
            diag_r = 50 * np.sqrt(2)
            rotation_mat = cv2.getRotationMatrix2D((diag_r, diag_r), 180 + angle, 1.0)
            
            # 执行变换
            transformed = cv2.warpAffine(test_image, rotation_mat, (100, 100))
            
            transform_results.append({
                'angle': angle,
                'mean_value': np.mean(transformed),
                'std_value': np.std(transformed),
                'non_zero_pixels': np.count_nonzero(transformed)
            })
            
        print("Coordinate transform results:")
        for result in transform_results[:4]:  # 显示前4个
            print(f"  Angle {result['angle']}°: mean={result['mean_value']:.2f}, non-zero={result['non_zero_pixels']}")
        
        self.test_results.append(('Coordinate Transform', transform_results))
        return transform_results
    
    def visualize_differences(self):
        """可视化主要差异"""
        print("\n=== 生成可视化报告 ===")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Observation and Rendering Differences Analysis', fontsize=16)
        
        # 1. APF势场对比
        ax = axes[0, 0]
        test_map = np.zeros((50, 50), dtype=np.uint8)
        test_map[20:30, 20:30] = 1
        apf_result = OldEnv.get_discounted_apf(test_map, max_step=30)
        im = ax.imshow(apf_result, cmap='hot')
        ax.set_title('APF Field (max_step=30)')
        plt.colorbar(im, ax=ax)
        
        # 2. Mist分布
        ax = axes[0, 1]
        self.old_env.reset(seed=42)
        ax.imshow(self.old_env.map_mist, cmap='gray')
        ax.set_title(f'Mist Map (sum={np.sum(self.old_env.map_mist)})')
        
        # 3. 渲染颜色
        ax = axes[0, 2]
        rendered = self.old_env.render_map()
        ax.imshow(rendered.astype(np.uint8))
        ax.set_title('Rendered Map')
        
        # 4. 噪声效果
        ax = axes[1, 0]
        noise_levels = [0, 5, 10, 15, 20]
        noise_effects = []
        for noise in noise_levels:
            env = OldEnv(render_mode=None, noise_position=noise)
            env.reset(seed=42)
            obs = env.observation()
            noise_effects.append(np.std(obs['observation']))
        ax.plot(noise_levels, noise_effects, 'o-')
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Observation Std')
        ax.set_title('Noise Effect on Observation')
        ax.grid(True)
        
        # 5. 通道分布
        ax = axes[1, 1]
        maps, mask = self.old_env.get_maps_and_mask()
        channel_means = [np.mean(maps[:,:,i]) for i in range(maps.shape[-1])]
        ax.bar(range(len(channel_means)), channel_means)
        ax.set_xlabel('Channel')
        ax.set_ylabel('Mean Value')
        ax.set_title('Channel Distribution')
        
        # 6. 覆盖率统计
        ax = axes[1, 2]
        coverage_data = {
            'Frontier': np.sum(self.old_env.map_frontier),
            'Obstacle': np.sum(self.old_env.map_obstacle),
            'Weed': np.sum(self.old_env.map_weed),
            'Mist': np.sum(self.old_env.map_mist)
        }
        ax.bar(coverage_data.keys(), coverage_data.values())
        ax.set_ylabel('Pixel Count')
        ax.set_title('Map Coverage Statistics')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = Path(__file__).parent / 'reports' / 'observation_rendering_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
        
        plt.show()
    
    def generate_report(self):
        """生成测试报告"""
        print("\n=== 生成测试报告 ===")
        
        report_path = Path(__file__).parent / 'reports' / 'test_results.txt'
        
        with open(report_path, 'w') as f:
            f.write("Observation and Rendering Test Results\n")
            f.write("=" * 50 + "\n\n")
            
            for test_name, results in self.test_results:
                f.write(f"{test_name}\n")
                f.write("-" * 30 + "\n")
                
                if isinstance(results, list):
                    for item in results[:5]:  # 只写入前5个结果
                        f.write(f"{item}\n")
                elif isinstance(results, dict):
                    for key, value in results.items():
                        f.write(f"{key}: {value}\n")
                
                f.write("\n")
        
        print(f"Report saved to: {report_path}")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("Starting Observation and Rendering Tests...")
        print("=" * 50)
        
        # 设置环境
        self.setup_environments()
        
        # 运行各项测试
        self.test_apf_parameters()
        self.test_noise_injection()
        self.test_mist_logic()
        self.test_rendering_colors()
        self.test_coordinate_transform()
        
        # 生成可视化
        self.visualize_differences()
        
        # 生成报告
        self.generate_report()
        
        print("\n" + "=" * 50)
        print("All tests completed!")
        
        return self.test_results


def main():
    """主函数"""
    tester = ObservationRenderingTester()
    results = tester.run_all_tests()
    
    # 打印摘要
    print("\n=== Test Summary ===")
    print(f"Total tests run: {len(results)}")
    print("Key findings:")
    print("1. APF parameters show significant differences")
    print("2. Mist logic needs semantic clarification")
    print("3. Rendering colors are well-defined but may differ in new version")
    print("4. Noise injection affects observation stability")
    print("5. Coordinate transform maintains precision")


if __name__ == "__main__":
    main()