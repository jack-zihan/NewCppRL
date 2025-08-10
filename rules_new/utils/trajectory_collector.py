"""
轨迹收集器 - 收集算法最终render图像和生成对比展示
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from PIL import Image


class TrajectoryCollector:
    """轨迹收集器 - 用于收集环境render图像和生成统一的算法对比可视化"""
    
    def __init__(self, trajectories_dir: Path, target_seed: int = 42):
        """
        初始化轨迹收集器
        
        Args:
            trajectories_dir: render图像保存目录
            target_seed: 目标种子（只为此种子收集render图像）
        """
        self.trajectories_dir = trajectories_dir  
        self.target_seed = target_seed
        self.logger = logging.getLogger(__name__)
        
        # render图像存储: {algorithm_name: image_path}
        self.render_images: Dict[str, Path] = {}
        
        # 性能指标存储: {algorithm_name: metrics_dict}
        self.performance_metrics: Dict[str, Dict[str, Any]] = {}
        
        # 环境信息存储
        self.environment_info = None
        
        # 当前正在记录的算法和种子
        self.current_algorithm = None
        self.current_seed = None
        self.recording = False
        
        self.logger.info(f"轨迹收集器初始化完成，目标种子: {target_seed}")
        
        # 确保目录存在
        self.trajectories_dir.mkdir(parents=True, exist_ok=True)
    
    def start_recording(self, algorithm_name: str, seed: int):
        """开始记录算法"""
        self.current_algorithm = algorithm_name
        self.current_seed = seed
        
        # 只为目标种子记录
        if seed == self.target_seed:
            self.recording = True
            self.logger.info(f"开始记录算法: {algorithm_name}, 种子: {seed}")
        else:
            self.recording = False
    
    def save_final_render(self, algorithm_name: str, seed: int, render_image: np.ndarray):
        """
        保存算法的最终render图像
        
        Args:
            algorithm_name: 算法名称
            seed: 种子值  
            render_image: 环境render的RGB图像数组
        """
        # 只为目标种子保存图像
        if seed != self.target_seed:
            return
        
        try:
            # 生成文件名
            image_filename = f"{algorithm_name}_seed{seed}_final_render.png"
            image_path = self.trajectories_dir / image_filename
            
            # 保存图像
            if isinstance(render_image, np.ndarray):
                # 确保是RGB格式
                if render_image.dtype != np.uint8:
                    render_image = (render_image * 255).astype(np.uint8)
                
                image = Image.fromarray(render_image)
                image.save(image_path)
                
                # 存储图像路径
                self.render_images[algorithm_name] = image_path
                
                self.logger.info(f"render图像已保存: {algorithm_name} -> {image_path}")
            else:
                self.logger.error(f"render图像格式错误: {type(render_image)}")
                
        except Exception as e:
            self.logger.error(f"保存render图像失败 {algorithm_name}: {e}")
    
    def record_position(self, position: Tuple[float, float]):
        """保留兼容性，实际不再使用"""
        pass
    
    def record_performance(self, algorithm_name: str, metrics: Dict[str, Any]):
        """记录性能指标"""
        self.performance_metrics[algorithm_name] = metrics
    
    def record_environment_info(self, env_info: Dict[str, Any]):
        """记录环境信息（只需要一次）"""
        if self.environment_info is None:
            self.environment_info = env_info
            self.logger.info("环境信息已记录")
    
    def stop_recording(self):
        """停止记录当前算法"""
        if self.recording and self.current_algorithm:
            self.logger.info(f"记录完成: {self.current_algorithm}")
        
        self.recording = False
        self.current_algorithm = None
        self.current_seed = None
    
    def generate_comparison_plot(self, experiment_name: str) -> Path:
        """
        生成基于render图像的算法对比展示
        
        Returns:
            生成的对比图片文件路径
        """
        if not self.render_images:
            self.logger.warning("没有render图像数据，无法生成对比图")
            return None
        
        # 创建图片文件路径
        plot_file = self.trajectories_dir / f"algorithm_comparison_seed{self.target_seed}.png"
        
        try:
            # 分离传统算法和神经网络算法
            traditional_algs = [alg for alg in self.render_images.keys() 
                              if not alg.startswith('NN_')]
            nn_algs = [alg for alg in self.render_images.keys() 
                      if alg.startswith('NN_')]
            
            # 计算布局（传统算法 + NN对比）
            total_plots = len(traditional_algs) + (1 if len(nn_algs) > 0 else 0)
            if total_plots == 0:
                self.logger.warning("没有有效的算法图像")
                return None
            
            # 计算网格大小（3列布局）
            cols = 3
            rows = (total_plots + cols - 1) // cols
            
            # 设置图片大小
            fig, axes = plt.subplots(rows, cols, figsize=(18, 6*rows))
            if rows == 1:
                axes = [axes] if cols == 1 else axes
            else:
                axes = axes.flatten()
            
            fig.suptitle(f'{experiment_name} - Algorithm Comparison (Seed {self.target_seed})', 
                        fontsize=16, fontweight='bold')
            
            plot_idx = 0
            
            # 展示传统算法的render图像
            for alg_name in traditional_algs:
                if plot_idx < len(axes):
                    ax = axes[plot_idx]
                    self._display_render_image(ax, alg_name)
                    plot_idx += 1
            
            # 展示神经网络对比
            if nn_algs and plot_idx < len(axes):
                ax = axes[plot_idx]
                self._display_nn_comparison(ax, nn_algs)
                plot_idx += 1
            
            # 隐藏多余的子图
            for i in range(plot_idx, len(axes)):
                axes[i].axis('off')
            
            # 保存图片
            plt.tight_layout()
            plt.savefig(plot_file, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            
            self.logger.info(f"算法对比图已生成: {plot_file}")
            return plot_file
            
        except Exception as e:
            self.logger.error(f"生成对比图失败: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None
    
    def _display_render_image(self, ax, algorithm_name: str):
        """在子图中展示单个算法的render图像"""
        if algorithm_name not in self.render_images:
            ax.text(0.5, 0.5, f'{algorithm_name}\n(No render image)', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12)
            ax.set_title(algorithm_name)
            ax.axis('off')
            return
        
        try:
            # 加载并显示图像
            image_path = self.render_images[algorithm_name]
            image = Image.open(image_path)
            ax.imshow(image)
            ax.axis('off')
            
            # 设置标题
            ax.set_title(algorithm_name, fontweight='bold', fontsize=14)
            
            # 添加性能指标文本
            if algorithm_name in self.performance_metrics:
                metrics = self.performance_metrics[algorithm_name]
                info_text = self._format_performance_text(metrics)
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
        
        except Exception as e:
            self.logger.error(f"显示render图像失败 {algorithm_name}: {e}")
            ax.text(0.5, 0.5, f'{algorithm_name}\n(Error loading image)', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(algorithm_name)
            ax.axis('off')
    
    def _display_nn_comparison(self, ax, nn_algorithms: List[str]):
        """创建神经网络算法对比展示"""
        if len(nn_algorithms) == 0:
            ax.axis('off')
            return
        
        # 如果只有一个NN算法，直接展示
        if len(nn_algorithms) == 1:
            self._display_render_image(ax, nn_algorithms[0])
            return
        
        # 多个NN算法：创建并排对比
        try:
            # 加载图像
            images = []
            labels = []
            for alg_name in nn_algorithms[:2]:  # 最多显示2个
                if alg_name in self.render_images:
                    image = Image.open(self.render_images[alg_name])
                    images.append(np.array(image))
                    
                    # 确定标签
                    if 'baseline' in alg_name.lower():
                        labels.append('Baseline')
                    elif 'ours' in alg_name.lower():
                        labels.append('Ours')
                    else:
                        labels.append(alg_name)
                else:
                    labels.append(f"{alg_name}\n(No image)")
            
            if len(images) == 2:
                # 水平拼接两个图像
                combined_image = np.concatenate(images, axis=1)
                ax.imshow(combined_image)
                
                # 添加分隔线和标签
                height, width = combined_image.shape[:2]
                ax.axvline(x=width//2, color='white', linewidth=3)
                
                # 左侧标签
                ax.text(width//4, height*0.05, labels[0], 
                       ha='center', va='top', fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.7, edgecolor='white'))
                
                # 右侧标签  
                ax.text(3*width//4, height*0.05, labels[1] if len(labels) > 1 else 'N/A',
                       ha='center', va='top', fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7, edgecolor='white'))
            
            elif len(images) == 1:
                ax.imshow(images[0])
                ax.text(0.5, 0.05, labels[0], transform=ax.transAxes,
                       ha='center', va='top', fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.7))
            
            ax.axis('off')
            ax.set_title('Neural Network Comparison', fontweight='bold', fontsize=14)
            
            # 添加性能对比信息
            if len(nn_algorithms) >= 2:
                comparison_text = self._format_nn_comparison_text(nn_algorithms)
                ax.text(0.02, 0.98, comparison_text, transform=ax.transAxes,
                       verticalalignment='top', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.9))
        
        except Exception as e:
            self.logger.error(f"创建NN对比失败: {e}")
            ax.text(0.5, 0.5, 'Neural Network\nComparison\n(Error)', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Neural Network Comparison')
            ax.axis('off')
    
    def _format_performance_text(self, metrics: Dict[str, Any]) -> str:
        """格式化性能指标文本"""
        text_lines = []
        
        # 主要指标
        if 'final_coverage' in metrics:
            text_lines.append(f"Coverage: {metrics['final_coverage']:.1%}")
        if 'total_distance' in metrics:
            text_lines.append(f"Distance: {metrics['total_distance']:.1f}")
        if 'runtime' in metrics:
            text_lines.append(f"Time: {metrics['runtime']:.1f}s")
        if 'iterations' in metrics:
            text_lines.append(f"Steps: {metrics['iterations']}")
        
        return '\n'.join(text_lines)
    
    def _format_nn_comparison_text(self, nn_algorithms: List[str]) -> str:
        """格式化神经网络对比文本"""
        if len(nn_algorithms) < 2:
            return ""
        
        text_lines = ["Performance Comparison:"]
        
        for alg_name in nn_algorithms[:2]:
            if alg_name in self.performance_metrics:
                metrics = self.performance_metrics[alg_name]
                label = "Baseline" if 'baseline' in alg_name.lower() else "Ours"
                
                coverage = metrics.get('final_coverage', 0)
                distance = metrics.get('total_distance', 0)
                
                text_lines.append(f"{label}: {coverage:.1%}, {distance:.1f}m")
        
        return '\n'.join(text_lines)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取轨迹收集摘要"""
        return {
            'target_seed': self.target_seed,
            'algorithms_recorded': list(self.render_images.keys()),
            'render_image_counts': len(self.render_images),
            'render_image_paths': {
                alg: str(path) for alg, path in self.render_images.items()
            },
            'performance_metrics_available': list(self.performance_metrics.keys()),
            'environment_info_recorded': self.environment_info is not None
        }