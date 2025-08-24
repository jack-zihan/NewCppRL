# SAC单进程异步训练版本 (sac_cont_single)

## 概述

这是SAC算法的单进程异步训练实现，使用`aSyncDataCollector`替代`MultiaSyncDataCollector`，专门设计用于：
- **解决PrioritizedReplayBuffer的多进程限制**
- **支持双GPU训练架构**（一张卡训练，一张卡收集）
- **简化配置和设备管理**

## 主要特性

### 1. 单进程异步架构
- 使用`aSyncDataCollector`实现单进程异步收集
- 完美兼容`TensorDictPrioritizedReplayBuffer`
- 避免了`PrioritizedSampler`的跨进程共享限制

### 2. 双GPU优化
- **训练GPU (cuda:0)**：专注于模型训练和梯度更新
- **收集GPU (cuda:1)**：负责策略推理和数据收集
- 环境在CPU运行，最大化GPU利用率

### 3. 简化配置
- 单个`gpu_device`参数配置收集设备
- `num_envs`控制单进程内并行环境数量
- 移除了复杂的多进程设备管理逻辑

## 配置说明

### config.yaml 关键参数

```yaml
# 数据收集配置
collector:
  num_envs: 32           # 单进程内并行环境数量
  gpu_device: cuda:1     # 收集GPU（null表示使用CPU）
  
# 训练配置
training:
  device: cuda:0         # 训练GPU
```

## 使用方法

### 基础训练
```bash
cd /home/lzh/NewCppRL/rl_new/sac_cont_single
python train.py
```

### 自定义配置

#### 1. 双GPU配置（推荐）
```bash
python train.py collector.gpu_device=cuda:1 training.device=cuda:0
```

#### 2. 单GPU配置
```bash
python train.py collector.gpu_device=null training.device=cuda:0
```

#### 3. 增加并行环境数量
```bash
python train.py collector.num_envs=64
```

#### 4. 使用预训练模型
```bash
python train.py pretrained_model=path/to/model.pt
```

#### 5. 调整批量大小
```bash
python train.py buffer.batch_size=4096 collector.frames_per_batch=16384
```

## 与多进程版本的对比

| 特性 | sac_cont_single (单进程) | sac_cont_new (多进程) |
|------|-------------------------|---------------------|
| **收集器** | aSyncDataCollector | MultiaSyncDataCollector |
| **进程数** | 1 | N (可配置) |
| **PrioritizedReplayBuffer** | ✅ 完全支持 | ❌ 不支持 |
| **GPU利用** | 双GPU分离 | 多GPU并行 |
| **配置复杂度** | 简单 | 复杂 |
| **适用场景** | 双GPU服务器 | 多GPU集群 |

## 性能优化建议

### 1. 环境并行度
- 根据CPU核心数调整`num_envs`
- 建议值：16-64（取决于环境复杂度）

### 2. GPU分配
- 确保训练和收集使用不同GPU
- 监控GPU利用率，动态调整`frames_per_batch`

### 3. 内存优化
- 使用`temp_dir`指定本地SSD路径
- 启用`pin_memory`加速GPU传输

### 4. 编译优化
- 保持`compile.enable=true`和`cudagraph=true`
- 使用`compile.mode=reduce-overhead`获得最佳性能

## 监控和调试

### 查看训练日志
```bash
# 实时查看训练进度
tail -f ckpt/sac_cont_single/*/train.log

# 使用wandb监控
wandb login
python train.py logger.backend=wandb
```

### 性能分析
训练脚本会自动记录：
- FPS（每秒帧数）
- 更新频率
- Episode奖励统计
- 损失曲线

## 常见问题

### Q: 为什么选择单进程架构？
A: `PrioritizedSampler`内部使用Sum-tree和Min-tree数据结构，无法在进程间安全共享。单进程架构完美解决了这个限制。

### Q: 单进程会影响性能吗？
A: 通过在单进程内并行多个环境（`num_envs`）和GPU分离策略，性能与多进程版本相当。

### Q: 如何充分利用多GPU？
A: 当前版本专注于双GPU优化。如需利用更多GPU，可以：
1. 运行多个独立训练实例
2. 使用不同的随机种子进行集成学习

## 技术细节

### 关键改动
1. **train.py**: 
   - 导入`aSyncDataCollector`
   - 简化collector创建逻辑
   - 单设备管理

2. **sac_utils.py**:
   - 简化`setup_devices`函数
   - 返回单个收集设备

3. **config.yaml**:
   - 移除多进程配置
   - 添加`num_envs`和`gpu_device`

### 架构图
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   训练GPU   │ ←── │ ReplayBuffer │ ←── │  收集GPU    │
│  (cuda:0)   │     │ (Prioritized)│     │  (cuda:1)   │
└─────────────┘     └──────────────┘     └─────────────┘
      ↑                                         ↑
      │                                         │
  模型更新                                  策略推理
                                               │
                                         ┌─────▼─────┐
                                         │    CPU    │
                                         │  环境×32  │
                                         └───────────┘
```

## 更新日志

### v1.0.0 (2024-01-XX)
- 初始版本
- 实现单进程异步收集架构
- 支持PrioritizedReplayBuffer
- 双GPU优化配置

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交Issue或联系维护者。