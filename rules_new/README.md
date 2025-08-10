# Rules New1 - 优雅的实验管理系统

## 📋 概述

Rules New1 是对原有 `rules_new/` 系统的完全重构版本，解决了原系统"注释一段运行一段"的不优雅实现模式，采用**配置驱动**和**模块化设计**理念，提供了专业级的实验管理解决方案。

### 🚨 解决的核心问题

**原系统 (rules_new) 的问题：**
- ❌ 硬编码macOS路径，不可移植
- ❌ 通过文件修改方式切换参数，极其脆弱
- ❌ 大量注释代码块管理实验配置，易错难维护
- ❌ 全局变量泛滥，职责耦合严重
- ❌ 无法并行执行，效率低下

**新系统 (rules_new1) 的优势：**
- ✅ YAML配置驱动，跨平台兼容
- ✅ 模块化算法实现，易于扩展
- ✅ 统一实验管理，支持批量和并行执行
- ✅ 专业级日志和结果收集
- ✅ 完善的测试和验证体系

## 🏗️ 架构设计

### 目录结构
```
rules_new1/
├── configs/                    # 配置文件集中管理
│   ├── algorithms/            # 算法配置
│   │   ├── jump.yaml          # JUMP算法配置
│   │   ├── snake.yaml         # SNAKE算法配置
│   │   ├── r_snake.yaml       # R_SNAKE算法配置
│   │   ├── react.yaml         # REACT算法配置
│   │   └── bcp.yaml           # BCP算法配置
│   ├── experiments/           # 实验配置
│   │   └── baseline_comparison.yaml  # 基准对比实验
│   └── base_config.yaml       # 基础配置
├── algorithms/                # 算法实现
│   ├── base_algorithm.py      # 算法基类
│   ├── jump_planner.py        # JUMP算法实现
│   ├── snake_planner.py       # SNAKE/R_SNAKE算法实现
│   ├── react_planner.py       # REACT算法实现
│   └── bcp_planner.py         # BCP算法实现
├── experiment/                # 实验管理
│   ├── config_manager.py      # 配置管理器
│   ├── experiment_runner.py   # 实验运行器
│   ├── result_collector.py    # 结果收集器
│   └── batch_manager.py       # 批量管理器
├── utils/                     # 工具函数
│   ├── path_utils.py          # 路径处理工具
│   ├── geometry_utils.py      # 几何计算工具
│   └── logging_utils.py       # 日志工具
└── main.py                    # 统一入口点
```

### 核心组件

#### 1. 配置系统 (Configuration System)
- **YAML配置文件**：替代硬编码和文件修改
- **分层配置**：基础配置 + 算法配置 + 实验配置
- **配置验证**：自动验证配置文件的完整性和正确性
- **配置缓存**：提高重复加载性能

#### 2. 算法架构 (Algorithm Architecture)
- **抽象基类** (`BasePathPlanner`)：定义统一接口
- **具体实现**：每个算法独立的类实现
- **性能监控**：内置性能指标收集
- **状态管理**：完善的算法状态跟踪

#### 3. 实验管理 (Experiment Management)
- **实验运行器**：统一的实验执行引擎
- **批量管理器**：支持批量和并行执行
- **结果收集器**：专业的结果收集和导出
- **配置管理器**：配置文件的加载和验证

## 🚀 快速开始

### 安装依赖
```bash
# 安装基础依赖
pip install pyyaml numpy matplotlib shapely

# 如果需要运行完整实验，确保环境模块可用
# 项目依赖的 envs_new.cpp_env_v2 等模块
```

### 基本使用

#### 1. 运行单个实验
```bash
# 进入rules_new1目录
cd rules_new

# 运行基准对比实验
python main.py run baseline_comparison

# 查看详细输出
python main.py run baseline_comparison --verbose
```

#### 2. 批量运行实验
```bash
# 顺序执行所有实验
python main.py batch

# 并行执行（4个线程）
python main.py batch --parallel --workers 4

# 指定特定实验
python main.py batch --configs baseline_comparison
```

#### 3. 配置管理
```bash
# 列出所有可用配置
python main.py list

# 验证配置文件
python main.py validate experiments/baseline_comparison
python main.py validate algorithms/jump
```

### 配置文件示例

#### 实验配置 (`experiments/baseline_comparison.yaml`)
```yaml
experiment:
  name: "baseline_algorithm_comparison"
  description: "对比JUMP, SNAKE, BCP, R_SNAKE, REACT等基准算法的性能"

parameters:
  seeds: [25, 27, 47, 21, 31]
  difficulties: ["easy", "medium", "hard"]
  weed_distributions: ["gaussian", "uniform"]
  noise_levels: ["no_noise"]

algorithms:
  - name: "JUMP"
    config_path: "algorithms/jump.yaml"
    enabled: true
  - name: "SNAKE"
    config_path: "algorithms/snake.yaml"
    enabled: true
  # ... 更多算法

output:
  base_dir: "logs"
  csv_format: true
  metrics:
    - coverage_90
    - coverage_95
    - coverage_98
    - total_coverage
    - path_length
```

#### 算法配置 (`algorithms/jump.yaml`)
```yaml
algorithm:
  name: "JUMP"
  type: "coverage_planning"

parameters:
  jump_threshold: 4
  safety_margin: 2
  forward_search: true
  vertical_search: true

performance:
  max_iterations: 5000
  timeout_seconds: 300
```

## 🧪 测试和验证

### 一致性测试
```bash
# 运行完整一致性测试
cd tests
python test_rules_new1_consistency.py

# 这将验证：
# - 配置系统正确性
# - 算法初始化
# - 单步执行一致性
# - 实验流程
# - 性能对比
```

### 功能对比测试
```bash
# 对比新旧版本功能
python test_rules_comparison.py

# 生成详细对比报告：
# - 架构对比
# - 配置系统对比
# - 代码可维护性对比
# - 算法覆盖度对比
```

## 📊 结果输出

### CSV结果文件
```csv
experiment_name,algorithm,seed,difficulty,map_id,coverage_90,coverage_95,coverage_98,total_coverage,runtime_seconds
baseline_comparison,JUMP,25,easy,2,156.23,189.45,234.67,0.982,45.23
baseline_comparison,SNAKE,25,easy,2,145.67,178.23,211.45,0.975,42.18
...
```

### 日志文件
- **实验日志**：`logs/experiments/experiment_name_timestamp.log`
- **批量执行日志**：`logs/batch_experiments/batch_summary_timestamp.json`
- **测试报告**：`logs/consistency_tests/consistency_test_report_timestamp.json`

## 🔧 扩展和定制

### 添加新算法

1. **创建算法配置文件**
```yaml
# configs/algorithms/my_algorithm.yaml
algorithm:
  name: "MY_ALGORITHM"
  type: "coverage_planning"

parameters:
  # 自定义参数
  
performance:
  max_iterations: 5000
  timeout_seconds: 300
```

2. **实现算法类**
```python
# algorithms/my_algorithm_planner.py
from .base_algorithm import BasePathPlanner

class MyAlgorithmPlanner(BasePathPlanner):
    def plan_next_waypoint(self, current_state):
        # 实现算法逻辑
        pass
        
    def should_terminate(self, current_state):
        # 实现终止条件
        pass
```

3. **注册算法**
```python
# 在 experiment_runner.py 的 algorithm_map 中添加
self.algorithm_map['MY_ALGORITHM'] = MyAlgorithmPlanner
```

### 创建新实验

1. **创建实验配置文件**
```yaml
# configs/experiments/my_experiment.yaml
experiment:
  name: "my_custom_experiment"
  description: "我的自定义实验"

parameters:
  seeds: [42, 43, 44]
  difficulties: ["easy"]
  # ...

algorithms:
  - name: "MY_ALGORITHM"
    config_path: "algorithms/my_algorithm.yaml"
    enabled: true
```

2. **运行实验**
```bash
python main.py run my_experiment
```

## 🔍 故障排除

### 常见问题

#### 1. 配置文件加载失败
```
错误：配置文件未找到: configs/experiments/xxx.yaml
解决：检查配置文件路径和文件名是否正确
```

#### 2. 算法初始化失败
```
错误：算法初始化失败 JUMP: No module named 'envs_new'
解决：确保项目环境模块可用，或跳过需要环境的测试
```

#### 3. 权限问题
```
错误：无法写入CSV文件: Permission denied
解决：检查logs目录权限，或更改输出目录配置
```

### 调试技巧

1. **启用详细输出**
```bash
python main.py run experiment_name --verbose
```

2. **查看日志文件**
```bash
tail -f logs/experiments/experiment_name_*.log
```

3. **验证配置**
```bash
python main.py validate experiments/experiment_name
```

## 🚀 性能优化

### 批量执行优化
- 使用并行执行：`--parallel --workers N`
- 合理设置工作线程数（通常为CPU核数）
- 监控内存使用情况

### 配置缓存
- 配置文件自动缓存，避免重复解析
- 手动清除缓存：`config_manager.clear_cache()`

### 结果收集优化
- 批量写入CSV：`batch_append_rows()`
- 适当的缓冲区大小设置

## 📈 迁移指南

### 从 rules_new 迁移到 rules_new1

1. **识别现有实验配置**
   - 分析 `script.py` 中的注释块
   - 提取参数组合和算法配置

2. **创建等效的YAML配置**
   - 将硬编码参数转换为配置文件
   - 设置相同的种子、难度、算法组合

3. **验证结果一致性**
   - 运行一致性测试
   - 对比关键指标的结果

4. **逐步替换**
   - 先并行运行两个版本
   - 验证结果后完全切换到新版本

## 📚 API参考

### 主要类和方法

#### ExperimentRunner
```python
runner = ExperimentRunner('experiment_config')
result = runner.run_experiment()
runner.cleanup()
```

#### BatchManager
```python
manager = BatchManager(max_workers=4)
manager.add_experiment('experiment_config')
result = manager.run_parallel()
```

#### ConfigManager
```python
config_manager = ConfigManager()
config = config_manager.load_experiment_config('experiment_name')
```

## 🤝 贡献指南

1. **代码风格**：遵循项目的简洁、清晰、优雅原则
2. **测试**：添加新功能时包含相应测试
3. **文档**：更新相关文档和示例
4. **配置**：为新算法提供完整配置示例

## 📄 许可证

本项目遵循原项目的许可证条款。

---

**Rules New1 - 让实验管理变得优雅而简单！** 🎯