# RL_NEW - TorchRL 0.9.2 迁移版本

这是从TorchRL 0.6.0迁移到0.9.2的RL训练代码。

## 📁 文件结构

```
rl_new/
├── sac_cont/                      # SAC算法实现
│   ├── area_coverage_sac_cont_train.py    ✅ 已修复，可用于V4环境训练
│   ├── area_coverage_sac_cont_eval.py     ✅ 无需修改，可直接评估V4模型
│   ├── area_coverage_v5_sac_cont_train.py ✅ 已修复，可用于V5环境训练
│   ├── area_coverage_v5_sac_cont_eval.py  ✅ 无需修改，可直接评估V5模型
│   ├── area_coverage_utils.py             # V4环境工具函数
│   ├── area_coverage_v5_utils.py          # V5环境工具函数
│   └── backup/                            # 备份文件存放目录
│
├── log/                           # 迁移日志
│   └── torchrl_migration_log.md  # 详细的迁移问题和解决方案记录
│
├── test_scripts/                  # 测试脚本（可选删除）
│   └── test_*.py                  # 各种兼容性测试脚本
│
├── migration_tools/               # 迁移工具（可选删除）
│   └── quick_fix_sac.py          # 自动修复脚本
│
└── torchrl_config.py             # TorchRL配置文件
```

## 🚀 快速开始

### 训练模型（笔记本测试配置）

```bash
# V4环境训练（4通道，无SGCNN）
python rl_new/sac_cont/area_coverage_sac_cont_train.py \
    collector.frames_per_batch=100 \
    collector.total_frames=1000 \
    collector.num_envs=1

# V5环境训练（20通道，SGCNN）
python rl_new/sac_cont/area_coverage_v5_sac_cont_train.py \
    collector.frames_per_batch=100 \
    collector.total_frames=1000 \
    collector.num_envs=1
```

### 评估模型

```bash
# 评估V4模型
python rl_new/sac_cont/area_coverage_sac_cont_eval.py \
    --ckpt_path ckpt/area_coverage_sac_cont/xxx/model.pt \
    --episodes 5

# 评估V5模型
python rl_new/sac_cont/area_coverage_v5_sac_cont_eval.py \
    --ckpt_path ckpt/area_coverage_v5_sac_cont/xxx/model.pt \
    --episodes 5
```

## 🔧 主要修复

1. **SACLoss参数更新**: 添加了`alpha_init=1.0`和`target_entropy=-2`
2. **临时目录管理**: 使用独立的临时目录避免文件冲突
3. **多进程收集器**: 如遇问题，可改用SyncDataCollector

## ⚠️ 注意事项

- 如果使用MultiaSyncDataCollector遇到多进程问题，考虑改用SyncDataCollector
- 在笔记本上测试时，建议减小batch_size和环境数量
- 评估脚本完全兼容，无需修改

## 📝 更多信息

详细的迁移问题和解决方案请查看 `log/torchrl_migration_log.md`