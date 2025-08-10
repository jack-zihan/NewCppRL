"""
可视化管理器 - 管理实验的可视化和图片保存

支持save_finished_picture参数，在场景完成时保存渲染图片
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class VisualizationManager:
    """
    可视化管理器
    
    负责渲染、保存和组织实验可视化结果
    """
    
    def __init__(self, 
                 output_dir: Path,
                 save_finished_picture: bool = False,
                 save_format: str = 'png',
                 dpi: int = 100):
        """
        初始化可视化管理器
        
        Args:
            output_dir: 输出目录
            save_finished_picture: 是否在场景完成时保存图片
            save_format: 图片保存格式
            dpi: 图片分辨率
        """
        self.output_dir = Path(output_dir)
        self.save_finished_picture = save_finished_picture
        self.save_format = save_format
        self.dpi = dpi
        
        # 创建可视化目录结构
        self.vis_dirs = self._create_visualization_structure()
        
        # 颜色映射
        self.color_map = {
            'obstacle': (128, 128, 128),    # 灰色
            'weed': (0, 255, 0),             # 绿色
            'covered': (0, 255, 255),        # 青色
            'uncovered': (255, 255, 0),      # 黄色
            'trajectory': (255, 0, 0),       # 红色
            'collision': (255, 0, 255),      # 紫色
            'boundary': (0, 0, 0),           # 黑色
        }
        
        logger.info(f"可视化管理器初始化 - save_finished_picture: {save_finished_picture}")
        
    def _create_visualization_structure(self) -> Dict[str, Path]:
        """创建可视化目录结构"""
        structure = {
            'scenarios': self.output_dir / 'visualization' / 'scenarios',
            'comparisons': self.output_dir / 'visualization' / 'comparisons',
            'trajectories': self.output_dir / 'visualization' / 'trajectories',
            'statistics': self.output_dir / 'visualization' / 'statistics'
        }
        
        for dir_path in structure.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        return structure
    
    def save_scenario_completion(self,
                                algorithm_name: str,
                                scenario_id: str,
                                env_state: Dict[str, Any],
                                trajectory_data: Dict[str, Any],
                                completion_reason: str) -> Optional[Path]:
        """
        在场景完成时保存渲染图片（对应save_finished_picture功能）
        
        Args:
            algorithm_name: 算法名称
            scenario_id: 场景ID
            env_state: 环境状态（包含地图、障碍物、杂草等）
            trajectory_data: 轨迹数据
            completion_reason: 完成原因（success/collision/timeout）
            
        Returns:
            保存的文件路径，如果未保存则返回None
        """
        if not self.save_finished_picture:
            return None
            
        try:
            # 生成文件名：算法_场景_完成原因_时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{algorithm_name}_{scenario_id}_{completion_reason}_{timestamp}.{self.save_format}"
            
            # 按场景ID组织目录
            scenario_dir = self.vis_dirs['scenarios'] / scenario_id
            scenario_dir.mkdir(exist_ok=True)
            
            filepath = scenario_dir / filename
            
            # 渲染最终状态
            image = self._render_final_state(env_state, trajectory_data, completion_reason)
            
            # 保存图片
            if self.save_format == 'png':
                cv2.imwrite(str(filepath), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            else:
                plt.imsave(str(filepath), image, dpi=self.dpi)
            
            logger.debug(f"保存场景完成图片: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"保存场景完成图片失败: {e}")
            return None
    
    def _render_final_state(self,
                           env_state: Dict[str, Any],
                           trajectory_data: Dict[str, Any],
                           completion_reason: str) -> np.ndarray:
        """
        渲染最终状态图像
        
        Args:
            env_state: 环境状态
            trajectory_data: 轨迹数据
            completion_reason: 完成原因
            
        Returns:
            渲染的图像数组
        """
        # 获取环境尺寸
        width = env_state.get('width', 600)
        height = env_state.get('height', 600)
        
        # 创建画布
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # 1. 绘制农场边界
        if 'farm_vertices' in env_state:
            self._draw_polygon(image, env_state['farm_vertices'], self.color_map['boundary'], 2)
        
        # 2. 绘制障碍物
        if 'obstacles' in env_state:
            for obstacle in env_state['obstacles']:
                pos = obstacle['position']
                radius = obstacle.get('radius', 15)
                cv2.circle(image, (int(pos[0]), int(pos[1])), int(radius), 
                          self.color_map['obstacle'], -1)
        
        # 3. 绘制杂草（区分已覆盖和未覆盖）
        if 'weeds' in env_state:
            covered_weeds = set(trajectory_data.get('covered_weeds', []))
            for i, weed in enumerate(env_state['weeds']):
                color = self.color_map['covered'] if i in covered_weeds else self.color_map['uncovered']
                cv2.circle(image, (int(weed[0]), int(weed[1])), 3, color, -1)
        
        # 4. 绘制轨迹
        if 'position_history' in trajectory_data:
            positions = trajectory_data['position_history']
            for i in range(1, len(positions)):
                pt1 = (int(positions[i-1][0]), int(positions[i-1][1]))
                pt2 = (int(positions[i][0]), int(positions[i][1]))
                cv2.line(image, pt1, pt2, self.color_map['trajectory'], 1)
        
        # 5. 标记碰撞点（如果有）
        if completion_reason == 'collision' and 'collision_info' in trajectory_data:
            collision_pos = trajectory_data['collision_info'].get('position')
            if collision_pos:
                cv2.circle(image, (int(collision_pos[0]), int(collision_pos[1])), 
                          10, self.color_map['collision'], 3)
        
        # 6. 添加信息文字
        self._add_info_text(image, trajectory_data, completion_reason)
        
        return image
    
    def _draw_polygon(self, image: np.ndarray, vertices: List[List[float]], 
                     color: Tuple[int, int, int], thickness: int):
        """绘制多边形"""
        points = np.array(vertices, np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(image, [points], True, color, thickness)
    
    def _add_info_text(self, image: np.ndarray, 
                      trajectory_data: Dict[str, Any],
                      completion_reason: str):
        """添加信息文字"""
        # 获取关键指标
        final_coverage = trajectory_data.get('coverage_history', [0])[-1]
        total_distance = trajectory_data.get('distance_history', [0])[-1]
        total_steps = len(trajectory_data.get('position_history', []))
        
        # 准备文字内容
        info_lines = [
            f"Status: {completion_reason}",
            f"Coverage: {final_coverage:.1%}",
            f"Distance: {total_distance:.1f}",
            f"Steps: {total_steps}"
        ]
        
        # 在图像顶部添加文字
        y_offset = 20
        for line in info_lines:
            cv2.putText(image, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_offset += 20
    
    def create_algorithm_comparison(self,
                                   scenario_id: str,
                                   algorithm_results: Dict[str, Dict[str, Any]]) -> Path:
        """
        创建算法对比图
        
        Args:
            scenario_id: 场景ID
            algorithm_results: 各算法的结果数据
            
        Returns:
            对比图文件路径
        """
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'Algorithm Comparison - Scenario {scenario_id}', fontsize=16)
        
        algorithms = list(algorithm_results.keys())
        
        for idx, (ax, alg_name) in enumerate(zip(axes.flat, algorithms)):
            if idx < len(algorithms):
                result = algorithm_results[alg_name]
                self._plot_algorithm_result(ax, alg_name, result)
            else:
                ax.axis('off')
        
        # 保存对比图
        comparison_file = self.vis_dirs['comparisons'] / f"{scenario_id}_comparison.{self.save_format}"
        plt.tight_layout()
        plt.savefig(comparison_file, dpi=self.dpi)
        plt.close()
        
        logger.info(f"创建算法对比图: {comparison_file}")
        return comparison_file
    
    def _plot_algorithm_result(self, ax, algorithm_name: str, result: Dict[str, Any]):
        """绘制单个算法结果"""
        # 绘制覆盖率曲线
        coverage_history = result.get('coverage_history', [])
        if coverage_history:
            ax.plot(coverage_history, label='Coverage')
            ax.set_xlabel('Steps')
            ax.set_ylabel('Coverage')
            ax.set_title(f'{algorithm_name}')
            ax.grid(True, alpha=0.3)
            
            # 添加关键指标
            final_coverage = coverage_history[-1]
            collision = result.get('collision_occurred', False)
            
            info_text = f"Final: {final_coverage:.1%}"
            if collision:
                info_text += "\n(Collision)"
            
            ax.text(0.95, 0.05, info_text, 
                   transform=ax.transAxes,
                   horizontalalignment='right',
                   verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def create_statistics_plot(self,
                               summary_stats: Dict[str, Any],
                               plot_type: str = 'bar') -> Path:
        """
        创建统计图表
        
        Args:
            summary_stats: 汇总统计数据
            plot_type: 图表类型 (bar/box/line)
            
        Returns:
            图表文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if plot_type == 'bar':
            return self._create_bar_chart(summary_stats, timestamp)
        elif plot_type == 'box':
            return self._create_box_plot(summary_stats, timestamp)
        else:
            return self._create_line_chart(summary_stats, timestamp)
    
    def _create_bar_chart(self, summary_stats: Dict[str, Any], timestamp: str) -> Path:
        """创建柱状图"""
        algorithms = list(summary_stats.keys())
        metrics = ['coverage_90', 'coverage_95', 'coverage_98', 'collision_rate']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Algorithm Performance Comparison', fontsize=16)
        
        for ax, metric in zip(axes.flat, metrics):
            values = []
            for alg in algorithms:
                if metric == 'collision_rate':
                    values.append(summary_stats[alg].get(f'{metric}', 0))
                else:
                    values.append(summary_stats[alg].get(f'{metric}_distance_mean', 0))
            
            bars = ax.bar(algorithms, values)
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_ylabel('Distance' if 'coverage' in metric else 'Rate')
            ax.grid(True, alpha=0.3)
            
            # 旋转x轴标签
            ax.set_xticklabels(algorithms, rotation=45, ha='right')
            
            # 为柱子添加数值标签
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}' if val > 0 else 'N/A',
                       ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存图表
        chart_file = self.vis_dirs['statistics'] / f"performance_comparison_{timestamp}.{self.save_format}"
        plt.savefig(chart_file, dpi=self.dpi)
        plt.close()
        
        logger.info(f"创建统计图表: {chart_file}")
        return chart_file
    
    def _create_box_plot(self, summary_stats: Dict[str, Any], timestamp: str) -> Path:
        """创建箱线图"""
        # 实现箱线图绘制逻辑
        pass
    
    def _create_line_chart(self, summary_stats: Dict[str, Any], timestamp: str) -> Path:
        """创建折线图"""
        # 实现折线图绘制逻辑
        pass
    
    def export_visualization_summary(self) -> Dict[str, Any]:
        """
        导出可视化摘要
        
        Returns:
            可视化文件统计信息
        """
        summary = {
            'save_finished_picture': self.save_finished_picture,
            'output_directory': str(self.output_dir),
            'visualization_directories': {
                name: str(path) for name, path in self.vis_dirs.items()
            },
            'file_counts': {}
        }
        
        # 统计各目录下的文件数量
        for name, path in self.vis_dirs.items():
            if path.exists():
                files = list(path.glob(f'*.{self.save_format}'))
                summary['file_counts'][name] = len(files)
        
        return summary