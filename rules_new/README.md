# Rules New - 高性能路径规划算法系统

## 📋 系统概述

Rules New 是一个高性能的路径规划算法实验平台，专为农业机器人覆盖路径规划设计。系统提供了多种经典和创新的路径规划算法实现，配备了完整的基准测试框架、优化组件和实验管理系统。

### 🎯 核心特性

- **🚀 高性能算法实现**：5种经典路径规划算法 + 2种神经网络模型
- **📊 标准化基准测试**：确定性场景生成，统一指标收集
- **⚡ 优化架构**：统一坐标系统、性能监控、状态验证
- **🔧 灵活配置**：YAML配置驱动，支持自定义实验
- **📈 完整分析**：自动生成排名、统计报告和可视化

## 🏗️ 系统架构

```
rules_new/
├── algorithms/                 # 路径规划算法实现
│   ├── base_algorithm.py      # 算法基类
│   ├── jump_planner.py        # JUMP算法
│   ├── snake_planner.py       # SNAKE/R-SNAKE算法
│   ├── bcp_planner.py         # BCP边界覆盖算法
│   ├── react_planner.py       # REACT反应式算法
│   └── nn_planner.py          # 神经网络算法
│
├── benchmark/                  # 标准化基准测试系统
│   ├── scenario_generator.py  # 确定性场景生成
│   ├── metric_collector.py    # 统一指标收集
│   ├── visualization_manager.py # 可视化管理
│   ├── benchmark_runner.py    # 主协调器
│   ├── result_analyzer.py     # 结果分析器
│   └── run_benchmark.py       # 命令行入口
│
├── core/                       # 核心优化组件
│   ├── coordinate_system.py   # 统一坐标系统
│   ├── state_validator.py     # 状态验证器
│   ├── performance_monitor.py # 性能监控器
│   ├── recovery_manager.py    # 错误恢复管理
│   └── exceptions.py          # 分层异常体系
│
├── experiment/                 # 实验管理系统
│   ├── experiment_runner.py   # 实验运行器
│   ├── config_manager.py      # 配置管理器
│   ├── result_collector.py    # 结果收集器
│   └── batch_manager.py       # 批量管理器
│
├── configs/                    # 配置文件
│   ├── base_config.yaml      # 基础环境配置
│   ├── benchmark_config.yaml # 基准测试配置
│   ├── benchmark_config_template.yaml # 配置模板
│   └── algorithms/            # 算法配置目录
│
├── utils/                      # 工具函数
│   ├── coordinate_converter.py # 坐标转换
│   ├── geometry_utils.py      # 几何计算
│   ├── path_utils.py          # 路径处理
│   ├── trajectory_collector.py # 轨迹收集
│   └── logging_utils.py       # 日志工具
│
├── tests/                      # 测试文件
│   └── test_algorithms_consistency.py # 一致性测试
│
├── docs/                       # 文档
│   ├── COORDINATE_SYSTEM.md   # 坐标系统文档
│   └── PHASE1_OPTIMIZATION_SUMMARY.md # 优化总结
│
└── main.py                     # 主入口程序
```

## 🚀 快速开始

### 安装依赖

```bash
# 基础依赖
pip install numpy pyyaml matplotlib opencv-python shapely pandas

# 如果需要运行神经网络模型
pip install torch torchrl tensordict
```

### 运行基准测试

#### 1. 使用默认配置
```bash
cd rules_new
python benchmark/run_benchmark.py

# 快速测试（少量场景）
python benchmark/run_benchmark.py --quick-test
```

#### 2. 使用自定义配置
```bash
# 复制模板创建自定义配置
cp configs/benchmark_config_template.yaml configs/my_experiment.yaml

# 编辑配置（启用/禁用算法，设置参数等）
vim configs/my_experiment.yaml

# 运行自定义实验
python benchmark/run_benchmark.py --config configs/my_experiment.yaml
```

#### 3. 保存场景完成图片
```bash
# 在每个场景完成时保存渲染图
python benchmark/run_benchmark.py --save-finished-picture

# 图片保存在: benchmark_results/*/visualization/scenarios/
```

#### 4. 并行执行加速
```bash
# 使用4个进程并行测试
python benchmark/run_benchmark.py --max-workers 4

# 禁用并行（用于调试）
python benchmark/run_benchmark.py --no-parallel
```

### 运行实验管理系统

```bash
# 运行单个实验
python main.py run baseline_comparison

# 批量运行实验
python main.py batch --parallel --workers 4

# 列出可用配置
python main.py list

# 验证配置文件
python main.py validate experiments/baseline_comparison
```

## 📊 算法介绍

### 传统路径规划算法

| 算法 | 描述 | 特点 | 适用场景 |
|------|------|------|----------|
| **JUMP** | 跳跃式路径规划 | 快速覆盖，避障能力强 | 障碍物稀疏环境 |
| **SNAKE** | 蛇形路径规划 | 规则往复，效率高 | 规则形状区域 |
| **R-SNAKE** | 改进蛇形算法 | 优化转弯，减少重复 | 复杂边界区域 |
| **BCP** | 边界覆盖规划 | 从边界向内覆盖 | 不规则区域 |
| **REACT** | 反应式规划 | 实时决策，适应性强 | 动态环境 |

### 神经网络模型

- **NN_baseline**: 基线深度强化学习模型
- **NN_ours**: 改进的深度强化学习模型

## 🔧 核心组件详解

### 1. 坐标系统 (Coordinate System)

统一的坐标处理系统，解决了原系统中坐标格式不一致的问题：

```python
from rules_new.core.coordinate_system import CoordinateSystem

# 统一坐标格式转换
pos_tuple = CoordinateSystem.normalize([x, y])  # 列表转元组
pos_array = CoordinateSystem.to_array((y, x))   # 元组转数组

# 坐标验证
is_valid = CoordinateSystem.validate_position(pos)
```

### 2. 基准测试系统 (Benchmark System)

#### 核心功能
- **确定性场景生成**：相同seed生成完全一致的测试场景
- **统一指标收集**：90%/95%/98%覆盖率对应的路径长度
- **灵活配置管理**：支持自定义配置文件
- **自动分析报告**：生成算法排名和统计分析

#### 配置示例
```yaml
benchmark:
  algorithms:
    JUMP:
      enabled: true
      params:
        step_size: 10
        max_iterations: 1000
    
  scenarios:
    seeds: [42, 100, 200]
    difficulties: ["easy", "medium"]
    
  metrics:
    coverage_thresholds: [0.90, 0.95, 0.98]
    
  output:
    save_finished_picture: true
    create_comparison_plots: true
```

### 3. 性能监控 (Performance Monitor)

实时监控算法性能：

```python
from rules_new.core.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start_timer("algorithm_execution")
# ... 算法执行 ...
monitor.end_timer("algorithm_execution")

# 获取性能报告
report = monitor.get_performance_report()
```

### 4. 状态验证 (State Validator)

确保算法状态的一致性：

```python
from rules_new.core.state_validator import StateValidator

validator = StateValidator()
validator.validate_consistency(current_state, expected_state)
```

## 📈 输出结果

### 目录结构
```
benchmark_results/
└── benchmark_YYYYMMDD_HHMMSS/
    ├── config/                    # 配置备份
    ├── visualization/
    │   ├── scenarios/            # 场景完成图片
    │   ├── comparisons/          # 算法对比图
    │   └── statistics/           # 统计图表
    ├── analysis/
    │   ├── raw_results.csv      # 原始数据
    │   ├── analysis_results.json # 分析结果
    │   └── analysis_report.md   # Markdown报告
    └── benchmark_report.yaml     # 总体报告
```

### 关键指标

- **覆盖率指标**：最终覆盖率、达到90%/95%/98%的路径长度
- **效率指标**：覆盖效率、时间效率、综合效率评分
- **碰撞指标**：碰撞发生率、碰撞距离
- **路径指标**：总路径长度、路径平滑度

## 🔬 优化历程

### Phase 1 优化（已完成）✅

1. **统一坐标系统**
   - 创建CoordinateSystem类统一处理坐标格式
   - 解决了y,x vs x,y的不一致问题
   - 所有算法使用统一接口

2. **分层异常体系**
   - 定义了清晰的异常层次结构
   - 实现了错误恢复机制
   - 提高了系统鲁棒性

3. **性能监控**
   - 添加了PerformanceMonitor组件
   - 实时跟踪算法性能指标
   - 识别性能瓶颈

4. **状态验证**
   - 实现StateValidator确保一致性
   - 自动检测状态异常
   - 提供详细的验证报告

### Phase 2 优化（计划中）

- 向量化计算优化
- 路径分解算法优化
- 内存使用优化

### Phase 3 优化（计划中）

- 分布式计算支持
- GPU加速
- 实时可视化

## 🧪 测试

### 运行测试
```bash
# 算法一致性测试
cd tests
python test_algorithms_consistency.py

# 基准系统测试
python ../benchmark/test_benchmark_system.py

# 配置系统测试
python ../benchmark/test_config_system.py
```

### 测试覆盖

- ✅ 坐标系统一致性
- ✅ 场景生成确定性
- ✅ 指标收集准确性
- ✅ 配置加载灵活性
- ✅ 可视化功能
- ✅ 结果分析正确性

## 🔧 扩展指南

### 添加新算法

1. **创建算法类**
```python
# algorithms/my_algorithm.py
from .base_algorithm import BasePathPlanner

class MyAlgorithm(BasePathPlanner):
    def plan_next_waypoint(self, current_state):
        # 实现路径规划逻辑
        pass
```

2. **注册算法**
```python
# 在benchmark_runner.py中注册
self.algorithm_classes['MY_ALGORITHM'] = MyAlgorithm
```

3. **添加配置**
```yaml
# configs/my_experiment.yaml
algorithms:
  MY_ALGORITHM:
    enabled: true
    params:
      custom_param: value
```

### 自定义指标

在`metric_collector.py`中添加新指标：

```python
def collect_custom_metric(self, trajectory_data):
    # 实现自定义指标计算
    return metric_value
```

## 🐛 故障排除

### 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 配置文件找不到 | 路径错误 | 使用`--config`指定完整路径 |
| 算法导入失败 | 缺少依赖 | 安装相应的Python包 |
| 内存不足 | 并行进程过多 | 减少`--max-workers`数量 |
| 结果不一致 | 种子未固定 | 确保使用相同的seed |

### 调试技巧

```bash
# 启用详细日志
python run_benchmark.py --log-level DEBUG

# 单算法测试
python run_benchmark.py --algorithms JUMP

# 单场景测试
python run_benchmark.py --seeds 42 --difficulties easy
```

## 📚 相关文档

- [坐标系统详解](docs/COORDINATE_SYSTEM.md)
- [Phase 1优化总结](docs/PHASE1_OPTIMIZATION_SUMMARY.md)
- [基准测试使用指南](benchmark/README.md)
- [配置系统说明](benchmark/CONFIG_IMPROVEMENTS.md)

## 🤝 贡献指南

1. **代码规范**
   - 遵循PEP 8规范
   - 添加类型注解
   - 编写清晰的文档字符串

2. **测试要求**
   - 新功能需包含单元测试
   - 确保现有测试通过
   - 更新相关文档

3. **提交规范**
   - 使用清晰的commit信息
   - 一个PR解决一个问题
   - 包含必要的测试和文档

## 📊 性能基准

基于标准测试集的性能对比（覆盖率98%的平均路径长度）：

| 算法 | Easy | Medium | Hard | 平均 |
|------|------|--------|------|------|
| JUMP | 520m | 780m | 1250m | 850m |
| SNAKE | 480m | 720m | 1180m | 793m |
| R-SNAKE | 465m | 695m | 1120m | 760m |
| BCP | 510m | 750m | 1200m | 820m |
| REACT | 495m | 735m | 1165m | 798m |

*注：数据基于默认参数配置，实际性能可能因参数调整而变化*

## 🔮 未来规划

### 短期目标（1-2月）
- [ ] 完成Phase 2性能优化
- [ ] 添加实时可视化界面
- [ ] 支持更多评估指标

### 中期目标（3-6月）
- [ ] 实现分布式计算
- [ ] 添加深度学习模型训练框架
- [ ] 开发Web界面

### 长期目标（6-12月）
- [ ] 支持3D环境
- [ ] 集成真实机器人接口
- [ ] 发布开源版本

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](../LICENSE) 文件。

---

**Rules New - 让路径规划更智能、更高效！** 🚀

*最后更新：2024年8月*