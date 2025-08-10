# 配置系统改进总结

## 问题解决

原问题：`--config-dir` 参数不够灵活，硬编码了必须使用 `benchmark_config.yaml` 文件名，用户无法使用自定义配置文件。

## 实现的改进

### 1. 新增 `--config` 参数
直接指定配置文件路径，最灵活的方式：
```bash
python run_benchmark.py --config my_experiment.yaml
python run_benchmark.py --config ./configs/test_v2.yaml
python run_benchmark.py --config /absolute/path/to/config.yaml
```

### 2. 新增 `--config-name` 参数
与 `--config-dir` 配合使用，指定配置名称：
```bash
python run_benchmark.py --config-dir ./configs --config-name my_benchmark
python run_benchmark.py --config-dir ./experiments --config-name test_v1
```

### 3. 智能配置发现
当只指定 `--config-dir` 时：
- 如果目录中只有一个 `*benchmark*.yaml` 文件，自动使用
- 如果有多个，列出所有可选项让用户选择
- 如果没有找到，使用默认配置

### 4. 配置模板文件
创建了 `benchmark_config_template.yaml`，包含详细注释，方便用户创建自定义配置。

## 使用优先级

1. **`--config` 直接指定**（最高优先级）
2. **`--config-dir` + `--config-name`**
3. **`--config-dir` + 自动发现**
4. **默认配置**（最低优先级）

## 错误处理

### 友好的错误提示
```
配置文件 xxx.yaml 不存在。
可用的配置文件：
  - benchmark_config.yaml
  - my_experiment.yaml
  - test_config.yaml
```

### 多配置冲突提示
```
找到多个基准测试配置文件，请明确指定：
  - benchmark_v1.yaml
  - benchmark_v2.yaml
使用 --config <文件路径> 或 --config-name <配置名>
```

## 向后兼容

- 原有的 `--config-dir ./configs` 仍然可用
- 默认查找 `benchmark_config.yaml`
- 不会破坏现有用户的使用习惯

## 测试验证

✅ 所有测试通过：
- 直接配置路径测试
- 配置目录+名称测试
- 自动发现配置测试
- 多配置错误提示测试

## 使用建议

### 推荐方式
```bash
# 1. 复制模板
cp configs/benchmark_config_template.yaml configs/my_experiment.yaml

# 2. 编辑配置
vim configs/my_experiment.yaml

# 3. 运行测试
python run_benchmark.py --config configs/my_experiment.yaml
```

### 快速实验
```bash
# 创建多个实验配置
configs/
├── experiment_fast.yaml      # 快速测试配置
├── experiment_full.yaml      # 完整测试配置
├── experiment_nn_only.yaml   # 只测试神经网络
└── experiment_traditional.yaml # 只测试传统算法

# 轻松切换
python run_benchmark.py --config configs/experiment_fast.yaml
python run_benchmark.py --config configs/experiment_full.yaml
```

## 总结

现在用户可以：
1. ✅ 使用任意名称的配置文件
2. ✅ 灵活组织配置文件结构
3. ✅ 快速切换不同实验配置
4. ✅ 获得清晰的错误提示

配置系统更加灵活、清晰、易用！