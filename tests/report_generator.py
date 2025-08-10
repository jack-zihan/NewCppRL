#!/usr/bin/env python3
"""
HTML报告生成器
生成交互式的一致性审查HTML报告
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import base64


class ReportGenerator:
    """HTML报告生成器"""
    
    def __init__(self, log_dir: Path):
        """初始化报告生成器"""
        self.log_dir = log_dir
        self.report_dir = log_dir / 'reports'
        self.report_dir.mkdir(exist_ok=True)
    
    def generate_html_report(self, all_results: Dict) -> Path:
        """生成完整的HTML报告"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 计算总体统计
        overall_stats = self._calculate_overall_stats(all_results)
        
        # 生成HTML内容
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rules_new vs Rules_new1 一致性审查报告</title>
    <style>
        {self._get_css_styles()}
    </style>
    <script>
        {self._get_javascript()}
    </script>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔍 Rules_new vs Rules_new1 一致性审查报告</h1>
            <div class="report-info">
                <span>生成时间: {timestamp}</span>
                <span>总体一致性: <strong class="{self._get_consistency_class(overall_stats['overall_consistency'])}">{overall_stats['overall_consistency']:.1f}%</strong></span>
            </div>
        </header>
        
        <section class="summary">
            <h2>📊 测试摘要</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>测试算法</h3>
                    <div class="metric-value">{overall_stats['num_algorithms']}</div>
                    <div class="metric-label">个算法</div>
                </div>
                <div class="summary-card">
                    <h3>测试种子</h3>
                    <div class="metric-value">{overall_stats['num_seeds']}</div>
                    <div class="metric-label">个种子</div>
                </div>
                <div class="summary-card">
                    <h3>总测试数</h3>
                    <div class="metric-value">{overall_stats['total_tests']}</div>
                    <div class="metric-label">次测试</div>
                </div>
                <div class="summary-card {self._get_consistency_class(overall_stats['overall_consistency'])}">
                    <h3>总体一致性</h3>
                    <div class="metric-value">{overall_stats['overall_consistency']:.1f}%</div>
                    <div class="metric-label">{self._get_consistency_label(overall_stats['overall_consistency'])}</div>
                </div>
            </div>
        </section>
        
        <section class="algorithms">
            <h2>🎯 算法详细对比</h2>
            <div class="tabs">
                {self._generate_algorithm_tabs(all_results)}
            </div>
            <div class="tab-content">
                {self._generate_algorithm_contents(all_results)}
            </div>
        </section>
        
        <section class="visualizations">
            <h2>📈 可视化分析</h2>
            <div class="viz-grid">
                {self._generate_visualization_section(all_results)}
            </div>
        </section>
        
        <section class="detailed-metrics">
            <h2>📋 详细指标对比</h2>
            {self._generate_detailed_metrics_table(all_results)}
        </section>
        
        <footer>
            <p>报告目录: <code>{str(self.log_dir)}</code></p>
            <p>© 2024 CppRL项目 - 一致性审查系统</p>
        </footer>
    </div>
</body>
</html>
"""
        
        # 保存HTML文件
        report_path = self.report_dir / 'consistency_audit_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path
    
    def _get_css_styles(self) -> str:
        """获取CSS样式"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background: white;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        header h1 {
            color: #333;
            margin-bottom: 10px;
        }
        
        .report-info {
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: #666;
            font-size: 14px;
        }
        
        section {
            background: white;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        h2 {
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #f0f0f0;
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .summary-card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            transition: transform 0.3s;
        }
        
        .summary-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .summary-card h3 {
            color: #666;
            font-size: 14px;
            margin-bottom: 10px;
        }
        
        .metric-value {
            font-size: 36px;
            font-weight: bold;
            color: #333;
        }
        
        .metric-label {
            color: #999;
            font-size: 12px;
        }
        
        .high-consistency {
            background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%) !important;
        }
        
        .medium-consistency {
            background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%) !important;
        }
        
        .low-consistency {
            background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%) !important;
        }
        
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #f0f0f0;
        }
        
        .tab-btn {
            padding: 10px 20px;
            background: none;
            border: none;
            color: #666;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.3s;
            border-bottom: 3px solid transparent;
        }
        
        .tab-btn:hover {
            color: #333;
        }
        
        .tab-btn.active {
            color: #667eea;
            border-bottom-color: #667eea;
        }
        
        .tab-pane {
            display: none;
            animation: fadeIn 0.5s;
        }
        
        .tab-pane.active {
            display: block;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #f0f0f0;
        }
        
        th {
            background: #f8f9fa;
            font-weight: 600;
            color: #333;
        }
        
        tr:hover {
            background: #f8f9fa;
        }
        
        .viz-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .viz-card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }
        
        .viz-card img {
            max-width: 100%;
            border-radius: 5px;
            margin-top: 10px;
        }
        
        .consistency-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }
        
        .consistency-high {
            background: #d4edda;
            color: #155724;
        }
        
        .consistency-medium {
            background: #fff3cd;
            color: #856404;
        }
        
        .consistency-low {
            background: #f8d7da;
            color: #721c24;
        }
        
        footer {
            text-align: center;
            padding: 20px;
            color: white;
            font-size: 14px;
        }
        
        code {
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        """
    
    def _get_javascript(self) -> str:
        """获取JavaScript代码"""
        return """
        document.addEventListener('DOMContentLoaded', function() {
            // 标签页切换功能
            const tabBtns = document.querySelectorAll('.tab-btn');
            const tabPanes = document.querySelectorAll('.tab-pane');
            
            tabBtns.forEach(btn => {
                btn.addEventListener('click', () => {
                    const target = btn.getAttribute('data-target');
                    
                    // 移除所有active类
                    tabBtns.forEach(b => b.classList.remove('active'));
                    tabPanes.forEach(p => p.classList.remove('active'));
                    
                    // 添加active类到当前选中的
                    btn.classList.add('active');
                    document.getElementById(target).classList.add('active');
                });
            });
            
            // 默认显示第一个标签
            if (tabBtns.length > 0) {
                tabBtns[0].click();
            }
        });
        """
    
    def _calculate_overall_stats(self, all_results: Dict) -> Dict:
        """计算总体统计信息"""
        stats = {
            'num_algorithms': len(all_results),
            'num_seeds': 0,
            'total_tests': 0,
            'overall_consistency': 0
        }
        
        consistency_scores = []
        seeds_set = set()
        
        for alg_name, alg_data in all_results.items():
            seeds_data = alg_data.get('seeds', {})
            seeds_set.update(seeds_data.keys())
            stats['total_tests'] += len(seeds_data)
            
            # 收集一致性分数
            for seed_data in seeds_data.values():
                if seed_data.get('comparison'):
                    consistency_scores.append(
                        seed_data['comparison'].get('similarity_score', 0)
                    )
        
        stats['num_seeds'] = len(seeds_set)
        
        if consistency_scores:
            import numpy as np
            stats['overall_consistency'] = np.mean(consistency_scores)
        
        return stats
    
    def _get_consistency_class(self, score: float) -> str:
        """根据一致性分数返回CSS类"""
        if score >= 90:
            return 'high-consistency'
        elif score >= 70:
            return 'medium-consistency'
        else:
            return 'low-consistency'
    
    def _get_consistency_label(self, score: float) -> str:
        """根据一致性分数返回标签"""
        if score >= 90:
            return '优秀'
        elif score >= 70:
            return '良好'
        else:
            return '需改进'
    
    def _generate_algorithm_tabs(self, all_results: Dict) -> str:
        """生成算法标签页"""
        tabs_html = []
        for i, alg_name in enumerate(all_results.keys()):
            active_class = 'active' if i == 0 else ''
            tabs_html.append(
                f'<button class="tab-btn {active_class}" data-target="tab-{alg_name}">{alg_name}</button>'
            )
        return '\n'.join(tabs_html)
    
    def _generate_algorithm_contents(self, all_results: Dict) -> str:
        """生成算法内容面板"""
        contents_html = []
        
        for i, (alg_name, alg_data) in enumerate(all_results.items()):
            active_class = 'active' if i == 0 else ''
            consistency = alg_data.get('overall_consistency', 0)
            
            # 生成种子测试结果表格
            seeds_table = self._generate_seeds_table(alg_data.get('seeds', {}))
            
            content = f"""
            <div id="tab-{alg_name}" class="tab-pane {active_class}">
                <h3>{alg_name} 算法测试结果</h3>
                <div class="algorithm-summary">
                    <p>总体一致性: <span class="consistency-badge {self._get_consistency_badge_class(consistency)}">{consistency:.1f}%</span></p>
                </div>
                {seeds_table}
            </div>
            """
            contents_html.append(content)
        
        return '\n'.join(contents_html)
    
    def _generate_seeds_table(self, seeds_data: Dict) -> str:
        """生成种子测试结果表格"""
        if not seeds_data:
            return '<p>无测试数据</p>'
        
        rows = []
        for seed, data in seeds_data.items():
            metrics_old = data.get('metrics_old', {})
            metrics_new = data.get('metrics_new', {})
            comparison = data.get('comparison', {})
            
            consistency = comparison.get('similarity_score', 0)
            badge_class = self._get_consistency_badge_class(consistency)
            
            row = f"""
            <tr>
                <td>{seed}</td>
                <td>{metrics_old.get('total_reward', 0):.2f}</td>
                <td>{metrics_new.get('total_reward', 0):.2f}</td>
                <td>{metrics_old.get('coverage_rate', 0)*100:.1f}%</td>
                <td>{metrics_new.get('coverage_rate', 0)*100:.1f}%</td>
                <td>{metrics_old.get('steps', 0)}</td>
                <td>{metrics_new.get('steps', 0)}</td>
                <td><span class="consistency-badge {badge_class}">{consistency:.1f}%</span></td>
            </tr>
            """
            rows.append(row)
        
        return f"""
        <table>
            <thead>
                <tr>
                    <th>种子</th>
                    <th>奖励(旧)</th>
                    <th>奖励(新)</th>
                    <th>覆盖率(旧)</th>
                    <th>覆盖率(新)</th>
                    <th>步数(旧)</th>
                    <th>步数(新)</th>
                    <th>一致性</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        """
    
    def _get_consistency_badge_class(self, score: float) -> str:
        """获取一致性徽章CSS类"""
        if score >= 90:
            return 'consistency-high'
        elif score >= 70:
            return 'consistency-medium'
        else:
            return 'consistency-low'
    
    def _generate_visualization_section(self, all_results: Dict) -> str:
        """生成可视化部分"""
        viz_cards = []
        
        # 添加汇总图表
        summary_chart_path = self.log_dir / 'visualizations' / 'summary_comparison.png'
        if summary_chart_path.exists():
            viz_cards.append(f"""
            <div class="viz-card">
                <h4>性能对比汇总</h4>
                <img src="{self._get_relative_path(summary_chart_path)}" alt="性能对比汇总">
            </div>
            """)
        
        # 添加算法轨迹对比图（示例）
        for alg_name in list(all_results.keys())[:3]:  # 只显示前3个算法的轨迹
            traj_path = self.log_dir / 'visualizations' / f'{alg_name}_seed42_trajectory_comparison.png'
            if traj_path.exists():
                viz_cards.append(f"""
                <div class="viz-card">
                    <h4>{alg_name} 轨迹对比</h4>
                    <img src="{self._get_relative_path(traj_path)}" alt="{alg_name} 轨迹对比">
                </div>
                """)
        
        return '\n'.join(viz_cards) if viz_cards else '<p>暂无可视化数据</p>'
    
    def _generate_detailed_metrics_table(self, all_results: Dict) -> str:
        """生成详细指标对比表"""
        rows = []
        
        for alg_name, alg_data in all_results.items():
            for seed, seed_data in alg_data.get('seeds', {}).items():
                metrics_old = seed_data.get('metrics_old', {})
                metrics_new = seed_data.get('metrics_new', {})
                comparison = seed_data.get('comparison', {})
                
                row = f"""
                <tr>
                    <td>{alg_name}</td>
                    <td>{seed}</td>
                    <td>{metrics_old.get('total_reward', 0):.2f} / {metrics_new.get('total_reward', 0):.2f}</td>
                    <td>{metrics_old.get('coverage_rate', 0)*100:.1f}% / {metrics_new.get('coverage_rate', 0)*100:.1f}%</td>
                    <td>{metrics_old.get('steps', 0)} / {metrics_new.get('steps', 0)}</td>
                    <td>{metrics_old.get('execution_time', 0):.2f}s / {metrics_new.get('execution_time', 0):.2f}s</td>
                    <td>{metrics_old.get('trajectory_length', 0):.1f} / {metrics_new.get('trajectory_length', 0):.1f}</td>
                    <td><span class="consistency-badge {self._get_consistency_badge_class(comparison.get('similarity_score', 0))}">{comparison.get('similarity_score', 0):.1f}%</span></td>
                </tr>
                """
                rows.append(row)
        
        if not rows:
            return '<p>无详细数据</p>'
        
        return f"""
        <table>
            <thead>
                <tr>
                    <th>算法</th>
                    <th>种子</th>
                    <th>总奖励 (旧/新)</th>
                    <th>覆盖率 (旧/新)</th>
                    <th>步数 (旧/新)</th>
                    <th>执行时间 (旧/新)</th>
                    <th>轨迹长度 (旧/新)</th>
                    <th>一致性分数</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        """
    
    def _get_relative_path(self, path: Path) -> str:
        """获取相对路径"""
        try:
            return str(path.relative_to(self.report_dir))
        except:
            return str(path)