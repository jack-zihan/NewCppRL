#!/usr/bin/env python3
"""
可视化生成器
生成轨迹对比图、像素差异图等可视化内容
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import List, Optional, Tuple, Any
import cv2


class VisualizationGenerator:
    """可视化生成器"""
    
    def __init__(self, log_dir: Path):
        """初始化生成器"""
        self.log_dir = log_dir
        self.viz_dir = log_dir / 'visualizations'
        self.viz_dir.mkdir(exist_ok=True)
        
        # 设置matplotlib样式
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 颜色方案
        self.colors = {
            'rules': '#FF6B6B',     # 红色
            'rules_new': '#4ECDC4',    # 青色
            'difference': '#FFE66D',     # 黄色
            'farm': '#95E77E',          # 绿色
            'obstacle': '#A8A8A8',      # 灰色
            'weed': '#FF1744'           # 深红色
        }
    
    def create_trajectory_comparison(self, traj_old: List, traj_new: List, 
                                    algorithm: str, seed: int) -> Path:
        """创建轨迹对比图"""
        fig = plt.figure(figsize=(16, 6))
        gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1])
        
        # 左图：rules_new轨迹
        ax1 = fig.add_subplot(gs[0])
        self._plot_single_trajectory(ax1, traj_old, 'rules', self.colors['rules'])
        ax1.set_title(f'{algorithm} - rules\n(Seed: {seed})', fontsize=12, fontweight='bold')
        
        # 中图：rules_new1轨迹
        ax2 = fig.add_subplot(gs[1])
        self._plot_single_trajectory(ax2, traj_new, 'rules_new', self.colors['rules_new'])
        ax2.set_title(f'{algorithm} - rules_new\n(Seed: {seed})', fontsize=12, fontweight='bold')
        
        # 右图：叠加对比
        ax3 = fig.add_subplot(gs[2])
        self._plot_trajectory_overlay(ax3, traj_old, traj_new)
        ax3.set_title(f'{algorithm} - 轨迹叠加\n(Seed: {seed})', fontsize=12, fontweight='bold')
        
        # 添加总标题
        fig.suptitle(f'轨迹对比分析 - {algorithm} 算法', fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        # 保存图像
        save_path = self.viz_dir / f'{algorithm}_seed{seed}_trajectory_comparison.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return save_path
    
    def _plot_single_trajectory(self, ax, trajectory: List, label: str, color: str):
        """绘制单条轨迹"""
        if not trajectory:
            ax.text(0.5, 0.5, '无轨迹数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(0, 600)
            ax.set_ylim(0, 600)
            return
        
        trajectory = np.array(trajectory)
        
        # 绘制轨迹线
        ax.plot(trajectory[:, 0], trajectory[:, 1], 
               color=color, linewidth=1.5, alpha=0.7, label=label)
        
        # 标记起点和终点
        if len(trajectory) > 0:
            ax.scatter(trajectory[0, 0], trajectory[0, 1], 
                      color='green', s=100, marker='o', zorder=5, label='起点')
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], 
                      color='red', s=100, marker='s', zorder=5, label='终点')
        
        # 每50步添加一个方向箭头
        for i in range(0, len(trajectory)-1, 50):
            if i+1 < len(trajectory):
                dx = trajectory[i+1, 0] - trajectory[i, 0]
                dy = trajectory[i+1, 1] - trajectory[i, 1]
                ax.arrow(trajectory[i, 0], trajectory[i, 1], dx*0.3, dy*0.3,
                        head_width=3, head_length=2, fc=color, ec=color, alpha=0.5)
        
        # 设置坐标轴
        ax.set_xlim(0, 600)
        ax.set_ylim(0, 600)
        ax.set_xlabel('X坐标', fontsize=10)
        ax.set_ylabel('Y坐标', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=8)
        
        # 添加统计信息
        total_length = self._calculate_trajectory_length(trajectory)
        ax.text(0.02, 0.98, f'轨迹长度: {total_length:.1f}\n步数: {len(trajectory)}',
               transform=ax.transAxes, fontsize=9, va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _plot_trajectory_overlay(self, ax, traj_old: List, traj_new: List):
        """绘制轨迹叠加对比图"""
        if not traj_old and not traj_new:
            ax.text(0.5, 0.5, '无轨迹数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(0, 600)
            ax.set_ylim(0, 600)
            return
        
        # 绘制两条轨迹
        if traj_old:
            traj_old = np.array(traj_old)
            ax.plot(traj_old[:, 0], traj_old[:, 1], 
                   color=self.colors['rules'], linewidth=2, alpha=0.6, label='rules')
        
        if traj_new:
            traj_new = np.array(traj_new)
            ax.plot(traj_new[:, 0], traj_new[:, 1], 
                   color=self.colors['rules_new'], linewidth=2, alpha=0.6, label='rules_new')
        
        # 计算并显示差异热图
        if traj_old is not None and traj_new is not None:
            self._add_difference_heatmap(ax, traj_old, traj_new)
        
        ax.set_xlim(0, 600)
        ax.set_ylim(0, 600)
        ax.set_xlabel('X坐标', fontsize=10)
        ax.set_ylabel('Y坐标', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=8)
        
        # 添加相似度信息
        if traj_old is not None and traj_new is not None:
            similarity = self._calculate_trajectory_similarity(traj_old, traj_new)
            ax.text(0.02, 0.98, f'轨迹相似度: {similarity:.1f}%',
                   transform=ax.transAxes, fontsize=9, va='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    def _add_difference_heatmap(self, ax, traj_old: np.ndarray, traj_new: np.ndarray):
        """添加轨迹差异热图"""
        # 确保是numpy数组
        if not isinstance(traj_old, np.ndarray):
            traj_old = np.array(traj_old)
        if not isinstance(traj_new, np.ndarray):
            traj_new = np.array(traj_new)
        
        # 检查是否有有效轨迹
        if len(traj_old) == 0 or len(traj_new) == 0:
            return
        
        # 确保是2D数组
        if len(traj_old.shape) == 1:
            return
        if len(traj_new.shape) == 1:
            return
        
        # 创建网格
        grid_size = 20
        x_bins = np.linspace(0, 600, grid_size)
        y_bins = np.linspace(0, 600, grid_size)
        
        # 计算每个网格中的差异
        diff_grid = np.zeros((grid_size-1, grid_size-1))
        
        for i in range(grid_size-1):
            for j in range(grid_size-1):
                # 查找在这个网格中的点
                mask_old = ((traj_old[:, 0] >= x_bins[i]) & (traj_old[:, 0] < x_bins[i+1]) &
                           (traj_old[:, 1] >= y_bins[j]) & (traj_old[:, 1] < y_bins[j+1]))
                mask_new = ((traj_new[:, 0] >= x_bins[i]) & (traj_new[:, 0] < x_bins[i+1]) &
                           (traj_new[:, 1] >= y_bins[j]) & (traj_new[:, 1] < y_bins[j+1]))
                
                # 计算访问次数差异
                diff_grid[j, i] = abs(np.sum(mask_old) - np.sum(mask_new))
        
        # 绘制热图
        if np.max(diff_grid) > 0:
            im = ax.imshow(diff_grid, extent=[0, 600, 0, 600], 
                          origin='lower', cmap='YlOrRd', alpha=0.3)
            plt.colorbar(im, ax=ax, label='访问差异', shrink=0.8)
    
    def _calculate_trajectory_length(self, trajectory: np.ndarray) -> float:
        """计算轨迹总长度"""
        if len(trajectory) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(trajectory)):
            total_length += np.linalg.norm(trajectory[i] - trajectory[i-1])
        
        return total_length
    
    def _calculate_trajectory_similarity(self, traj1: np.ndarray, traj2: np.ndarray) -> float:
        """计算轨迹相似度"""
        min_len = min(len(traj1), len(traj2))
        if min_len == 0:
            return 0.0
        
        # 采样对齐
        sample_indices = np.linspace(0, min_len-1, min(100, min_len), dtype=int)
        
        total_distance = 0.0
        for idx in sample_indices:
            total_distance += np.linalg.norm(traj1[idx] - traj2[idx])
        
        avg_distance = total_distance / len(sample_indices)
        
        # 转换为相似度百分比
        similarity = max(0, 100 - avg_distance)
        
        return similarity
    
    def create_pixel_difference_map(self, frame_old: np.ndarray, frame_new: np.ndarray,
                                   algorithm: str, seed: int) -> Path:
        """创建像素差异图"""
        if frame_old is None or frame_new is None:
            return None
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # 原始图像 - rules
        axes[0].imshow(frame_old)
        axes[0].set_title('rules - 最后一帧', fontsize=12)
        axes[0].axis('off')
        
        # 原始图像 - rules_new
        axes[1].imshow(frame_new)
        axes[1].set_title('rules_new - 最后一帧', fontsize=12)
        axes[1].axis('off')
        
        # 计算差异
        if frame_old.shape == frame_new.shape:
            # 转换为灰度图进行比较
            gray_old = cv2.cvtColor(frame_old, cv2.COLOR_RGB2GRAY)
            gray_new = cv2.cvtColor(frame_new, cv2.COLOR_RGB2GRAY)
            
            # 计算绝对差异
            diff = cv2.absdiff(gray_old, gray_new)
            
            # 显示差异图
            axes[2].imshow(diff, cmap='hot')
            axes[2].set_title('像素差异热图', fontsize=12)
            axes[2].axis('off')
            
            # 创建差异标记图
            threshold = 30
            diff_mask = diff > threshold
            marked_image = frame_new.copy()
            marked_image[diff_mask] = [255, 0, 0]  # 标记差异区域为红色
            
            axes[3].imshow(marked_image)
            axes[3].set_title(f'差异标记 (阈值>{threshold})', fontsize=12)
            axes[3].axis('off')
            
            # 计算相似度
            similarity = 100 * (1 - np.mean(diff) / 255)
            fig.suptitle(f'{algorithm} - 像素级对比 (Seed: {seed}) - 相似度: {similarity:.1f}%',
                        fontsize=14, fontweight='bold')
        else:
            axes[2].text(0.5, 0.5, '图像尺寸不匹配', ha='center', va='center')
            axes[2].axis('off')
            axes[3].axis('off')
            fig.suptitle(f'{algorithm} - 像素级对比 (Seed: {seed}) - 无法比较', 
                        fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图像
        save_path = self.viz_dir / f'{algorithm}_seed{seed}_pixel_comparison.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return save_path
    
    def create_metrics_radar_chart(self, metrics_old: dict, metrics_new: dict,
                                   algorithm: str) -> Path:
        """创建指标雷达图"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='polar')
        
        # 准备数据
        categories = ['总奖励\n(归一化)', '覆盖率', '步数\n(归一化)', 
                     '执行效率', '轨迹长度\n(归一化)']
        
        # 归一化指标
        def normalize_metrics(metrics):
            return [
                min(metrics.get('total_reward', 0) / 1000 + 0.5, 1),  # 归一化到[0,1]
                metrics.get('coverage_rate', 0),
                1 - min(metrics.get('steps', 0) / 1000, 1),  # 步数越少越好
                1 - min(metrics.get('execution_time', 0) / 10, 1),  # 时间越短越好
                min(metrics.get('trajectory_length', 0) / 1000, 1)
            ]
        
        values_old = normalize_metrics(metrics_old) if metrics_old else [0] * 5
        values_new = normalize_metrics(metrics_new) if metrics_new else [0] * 5
        
        # 设置角度
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values_old += values_old[:1]
        values_new += values_new[:1]
        angles += angles[:1]
        
        # 绘制雷达图
        ax.plot(angles, values_old, 'o-', linewidth=2, 
               label='rules', color=self.colors['rules'])
        ax.fill(angles, values_old, alpha=0.25, color=self.colors['rules'])
        
        ax.plot(angles, values_new, 'o-', linewidth=2,
               label='rules_new', color=self.colors['rules_new'])
        ax.fill(angles, values_new, alpha=0.25, color=self.colors['rules_new'])
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        
        # 添加网格
        ax.grid(True)
        
        # 添加图例和标题
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.title(f'{algorithm} - 性能指标对比', size=14, fontweight='bold', pad=20)
        
        # 保存图像
        save_path = self.viz_dir / f'{algorithm}_metrics_radar.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return save_path
    
    def create_summary_bar_chart(self, all_results: dict) -> Path:
        """创建汇总条形图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        algorithms = list(all_results.keys())
        
        # 准备数据
        rewards_old = []
        rewards_new = []
        coverage_old = []
        coverage_new = []
        steps_old = []
        steps_new = []
        time_old = []
        time_new = []
        
        for alg in algorithms:
            # 计算每个算法的平均值
            seeds_data = all_results[alg].get('seeds', {})
            
            r_old = [s['metrics_old']['total_reward'] for s in seeds_data.values() 
                    if s.get('metrics_old')]
            r_new = [s['metrics_new']['total_reward'] for s in seeds_data.values() 
                    if s.get('metrics_new')]
            
            rewards_old.append(np.mean(r_old) if r_old else 0)
            rewards_new.append(np.mean(r_new) if r_new else 0)
            
            c_old = [s['metrics_old']['coverage_rate'] for s in seeds_data.values() 
                    if s.get('metrics_old')]
            c_new = [s['metrics_new']['coverage_rate'] for s in seeds_data.values() 
                    if s.get('metrics_new')]
            
            coverage_old.append(np.mean(c_old) if c_old else 0)
            coverage_new.append(np.mean(c_new) if c_new else 0)
            
            s_old = [s['metrics_old']['steps'] for s in seeds_data.values() 
                    if s.get('metrics_old')]
            s_new = [s['metrics_new']['steps'] for s in seeds_data.values() 
                    if s.get('metrics_new')]
            
            steps_old.append(np.mean(s_old) if s_old else 0)
            steps_new.append(np.mean(s_new) if s_new else 0)
            
            t_old = [s['metrics_old']['execution_time'] for s in seeds_data.values() 
                    if s.get('metrics_old')]
            t_new = [s['metrics_new']['execution_time'] for s in seeds_data.values() 
                    if s.get('metrics_new')]
            
            time_old.append(np.mean(t_old) if t_old else 0)
            time_new.append(np.mean(t_new) if t_new else 0)
        
        x = np.arange(len(algorithms))
        width = 0.35
        
        # 总奖励对比
        axes[0, 0].bar(x - width/2, rewards_old, width, label='rules',
                      color=self.colors['rules'])
        axes[0, 0].bar(x + width/2, rewards_new, width, label='rules_new',
                      color=self.colors['rules_new'])
        axes[0, 0].set_xlabel('算法')
        axes[0, 0].set_ylabel('平均总奖励')
        axes[0, 0].set_title('总奖励对比')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(algorithms)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 覆盖率对比
        axes[0, 1].bar(x - width/2, np.array(coverage_old)*100, width, label='rules',
                      color=self.colors['rules'])
        axes[0, 1].bar(x + width/2, np.array(coverage_new)*100, width, label='rules_new',
                      color=self.colors['rules_new'])
        axes[0, 1].set_xlabel('算法')
        axes[0, 1].set_ylabel('平均覆盖率 (%)')
        axes[0, 1].set_title('覆盖率对比')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(algorithms)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 步数对比
        axes[1, 0].bar(x - width/2, steps_old, width, label='rules',
                      color=self.colors['rules'])
        axes[1, 0].bar(x + width/2, steps_new, width, label='rules_new',
                      color=self.colors['rules_new'])
        axes[1, 0].set_xlabel('算法')
        axes[1, 0].set_ylabel('平均步数')
        axes[1, 0].set_title('执行步数对比')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(algorithms)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 执行时间对比
        axes[1, 1].bar(x - width/2, time_old, width, label='rules',
                      color=self.colors['rules'])
        axes[1, 1].bar(x + width/2, time_new, width, label='rules_new',
                      color=self.colors['rules_new'])
        axes[1, 1].set_xlabel('算法')
        axes[1, 1].set_ylabel('平均执行时间 (秒)')
        axes[1, 1].set_title('执行时间对比')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(algorithms)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        fig.suptitle('Rules_new vs Rules_new1 性能对比汇总', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存图像
        save_path = self.viz_dir / 'summary_comparison.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return save_path