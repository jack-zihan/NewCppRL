# 标准化基准测试系统

一个清晰、易用的实验评估系统，用于对比7个路径规划算法在相同场景下的性能表现。

## 功能特点

- ✅ **确定性场景生成**：相同seed生成完全一致的测试场景
- ✅ **统一指标收集**：90%/95%/98%覆盖率对应的路径长度、碰撞检测等
- ✅ **灵活配置管理**：支持config_dir参数切换不同配置
- ✅ **可视化保存**：save_finished_picture参数保存场景完成时的渲染图
- ✅ **并行执行**：支持多进程并行测试提高效率
- ✅ **自动分析**：生成排名、统计报告和对比图表

## 支持的算法

1. **JUMP** - 跳跃式路径规划
2. **SNAKE** - 蛇形路径规划
3. **R_SNAKE** - 改进的蛇形算法
4. **BCP** - 边界覆盖路径规划
5. **REACT** - 反应式路径规划
6. **NN_baseline** - 神经网络基线模型
7. **NN_ours** - 我们的神经网络模型

## 快速开始

### 1. 基本使用

```bash
# 运行所有算法的完整基准测试
python run_benchmark.py

# 快速测试模式（只测试少量场景）
python run_benchmark.py --quick-test
```

### 2. 使用自定义配置

#### 方法1：直接指定配置文件（推荐）
```bash
# 使用您自己的配置文件
python run_benchmark.py --config my_benchmark.yaml
python run_benchmark.py --config ./configs/experiment_v2.yaml
python run_benchmark.py --config /absolute/path/to/config.yaml
```

#### 方法2：使用配置目录 + 配置名称
```bash
# 指定目录和配置名（不需要.yaml后缀）
python run_benchmark.py --config-dir ./configs --config-name my_benchmark
python run_benchmark.py --config-dir ./experiments --config-name test_v1
```

#### 方法3：使用默认配置目录
```bash
# 在默认目录查找benchmark_config.yaml
python run_benchmark.py --config-dir ./my_configs

# 如果目录中只有一个*benchmark*.yaml文件，会自动使用
# 如果有多个，会提示您选择
```

#### 创建自定义配置
```bash
# 1. 复制配置模板
cp configs/benchmark_config_template.yaml configs/my_experiment.yaml

# 2. 编辑您的配置
vim configs/my_experiment.yaml

# 3. 运行测试
python run_benchmark.py --config configs/my_experiment.yaml
```

### 3. 保存场景完成图片

```bash
# 在每个场景完成时保存渲染图片
python run_benchmark.py --save-finished-picture

# 图片会按场景ID组织保存在：
# benchmark_results/benchmark_*/visualization/scenarios/
```

### 4. 选择特定算法测试

```bash
# 只测试特定算法
python run_benchmark.py --algorithms JUMP SNAKE BCP

# 只测试神经网络模型
python run_benchmark.py --algorithms NN_baseline NN_ours
```

### 5. 自定义场景参数

```bash
# 使用特定种子
python run_benchmark.py --seeds 42 100 200

# 只测试特定难度
python run_benchmark.py --difficulties easy medium
```

### 6. 并行执行控制

```bash
# 禁用并行（用于调试）
python run_benchmark.py --no-parallel

# 设置最大工作进程数
python run_benchmark.py --max-workers 4
```

## 配置文件说明

### 配置文件结构
```
configs/
├── base_config.yaml              # 基础环境配置（所有实验共享）
├── benchmark_config.yaml         # 默认基准测试配置
├── benchmark_config_template.yaml # 配置模板（复制此文件创建新配置）
└── my_experiment.yaml           # 您的自定义配置
```

### base_config.yaml（基础配置）
```yaml
environment:
  width: 600
  height: 600
  
difficulty_levels:
  easy:
    obstacle_range: [0, 0]
    weed_num: 50
    map_ids: [2, 3, 6, 16, 20]
```

### 自定义配置示例
```yaml
# my_experiment.yaml
benchmark:
  algorithms:
    JUMP:
      enabled: true  # 启用JUMP算法
      params:
        step_size: 15  # 自定义参数
    SNAKE:
      enabled: false  # 禁用SNAKE算法
    NN_baseline:
      enabled: true
      model_path: "ckpt/my_model.pt"  # 您的模型路径
      
  scenarios:
    seeds: [42, 100]  # 只测试2个种子
    difficulties: ["easy"]  # 只测试简单难度
    
  output:
    save_finished_picture: true  # 保存完成图片
    create_comparison_plots: true
```

### 配置优先级
1. 命令行参数 > 配置文件
2. 自定义配置 > 默认配置
3. `--config` 指定的文件 > `--config-dir` 中的文件

## 输出结果结构

```
benchmark_results/
└── benchmark_20241210_143052/
    ├── config/                    # 配置文件副本
    │   ├── base_config.yaml
    │   └── benchmark_config.yaml
    ├── visualization/              # 可视化结果
    │   ├── scenarios/             # 场景完成图片（按场景ID分组）
    │   │   ├── s25_easy_m2_g/
    │   │   │   ├── JUMP_s25_easy_m2_g_success_*.png
    │   │   │   ├── SNAKE_s25_easy_m2_g_collision_*.png
    │   │   │   └── ...
    │   ├── comparisons/           # 算法对比图
    │   └── statistics/            # 统计图表
    ├── analysis/                  # 分析结果
    │   ├── raw_results.csv       # 原始数据
    │   ├── analysis_results.json # 分析结果
    │   └── analysis_report.md    # Markdown报告
    └── benchmark_report.yaml      # 总体报告
```

## 关键指标说明

### 覆盖率指标
- **coverage_90_distance**: 达到90%覆盖率时的路径长度
- **coverage_95_distance**: 达到95%覆盖率时的路径长度
- **coverage_98_distance**: 达到98%覆盖率时的路径长度

### 碰撞指标
- **collision_occurred**: 是否发生碰撞
- **collision_distance**: 碰撞时的累计路径长度
- **collision_type**: 碰撞类型（boundary/obstacle）

### 效率指标
- **coverage_efficiency**: 覆盖率/路径长度
- **time_efficiency**: 覆盖率/运行时间
- **overall_efficiency_score**: 综合效率评分

## Python API使用

```python
from rules_new.benchmark import BenchmarkRunner

# 创建运行器
runner = BenchmarkRunner(
    config_dir="./configs",
    save_finished_picture=True,
    parallel=True
)

# 运行测试
summary = runner.run_benchmark(
    algorithms=["JUMP", "SNAKE"],
    scenarios=None  # 使用配置生成
)

# 获取排名
rankings = summary['algorithm_rankings']['overall']
for rank, (alg, score) in enumerate(rankings, 1):
    print(f"{rank}. {alg}: {score:.3f}")
```

## 高级功能

### 1. 自定义场景生成

```python
from rules_new.benchmark import ScenarioGenerator

generator = ScenarioGenerator(config)
scenario = generator.generate_scenario(
    seed=42,
    difficulty="medium",
    map_id=5,
    weed_distribution="gaussian",
    noise_level="low_noise"
)
```

### 2. 单独使用指标收集器

```python
from rules_new.benchmark import MetricCollector

collector = MetricCollector(coverage_thresholds=[0.90, 0.95, 0.98])
metrics = collector.collect_metrics(
    trajectory_data,
    env_info,
    algorithm_info
)
```

### 3. 自定义可视化

```python
from rules_new.benchmark import VisualizationManager

viz = VisualizationManager(
    output_dir="./results",
    save_finished_picture=True,
    save_format='png',
    dpi=150
)

# 保存场景完成图
viz.save_scenario_completion(
    algorithm_name="JUMP",
    scenario_id="s25_easy_m2",
    env_state=env_state,
    trajectory_data=trajectory,
    completion_reason="success"
)
```

## 常见问题

### Q: 如何使用自己的配置文件？
A: 有三种方法：
1. 直接指定：`python run_benchmark.py --config my_config.yaml`
2. 目录+名称：`python run_benchmark.py --config-dir ./configs --config-name my_config`
3. 复制模板：`cp configs/benchmark_config_template.yaml configs/my_config.yaml` 然后编辑

### Q: 配置文件找不到怎么办？
A: 系统会列出可用的配置文件。确保：
1. 文件路径正确
2. 文件扩展名是 `.yaml`
3. 如果使用 `--config-name`，不要包含 `.yaml` 后缀

### Q: 如何确保不同算法在完全相同的场景下测试？
A: 系统通过固定seed和确定性场景生成保证相同参数生成完全一致的场景。

### Q: save_finished_picture会影响性能吗？
A: 会略微增加I/O开销，但通过异步保存最小化了影响。

### Q: 如何添加新的算法？
A: 1. 在algorithms/目录下实现算法类
   2. 在benchmark_runner.py的algorithm_classes中注册
   3. 在配置文件中添加算法配置

### Q: 并行执行出错怎么办？
A: 使用`--no-parallel`禁用并行，逐个算法调试。

## 贡献指南

欢迎提交Issue和Pull Request来改进系统。

## 许可证

MIT License