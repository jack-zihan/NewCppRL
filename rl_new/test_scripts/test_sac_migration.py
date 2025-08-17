#!/usr/bin/env python
"""测试脚本，用于检测TorchRL版本迁移问题"""

import sys
import traceback
import torch

print(f"Python版本: {sys.version}")
print(f"PyTorch版本: {torch.__version__}")

# 尝试导入TorchRL并检查版本
try:
    import torchrl
    print(f"TorchRL版本: {torchrl.__version__}")
except ImportError as e:
    print(f"无法导入TorchRL: {e}")
    sys.exit(1)

# 测试导入area_coverage_sac_cont_train的关键模块
print("\n=== 测试导入基础模块 ===")
errors = []

try:
    from torchrl.collectors import MultiaSyncDataCollector
    print("✓ MultiaSyncDataCollector导入成功")
except ImportError as e:
    errors.append(f"✗ MultiaSyncDataCollector导入失败: {e}")

try:
    from torchrl.data import LazyMemmapStorage, TensorDictPrioritizedReplayBuffer
    print("✓ LazyMemmapStorage, TensorDictPrioritizedReplayBuffer导入成功")
except ImportError as e:
    errors.append(f"✗ LazyMemmapStorage/TensorDictPrioritizedReplayBuffer导入失败: {e}")

try:
    from torchrl.objectives import SoftUpdate, SACLoss
    print("✓ SoftUpdate, SACLoss导入成功")
except ImportError as e:
    errors.append(f"✗ SoftUpdate/SACLoss导入失败: {e}")

try:
    from torchrl.record.loggers import get_logger
    print("✓ get_logger导入成功")
except ImportError as e:
    errors.append(f"✗ get_logger导入失败: {e}")

# 尝试导入自定义模块
print("\n=== 测试导入自定义模块 ===")
try:
    from rl.sac_cont.area_coverage_utils import (
        make_area_coverage_sac_models,
        make_area_coverage_env
    )
    print("✓ area_coverage_utils导入成功")
except ImportError as e:
    errors.append(f"✗ area_coverage_utils导入失败: {e}")
    traceback.print_exc()

# 打印所有错误
if errors:
    print("\n=== 发现的错误 ===")
    for error in errors:
        print(error)
else:
    print("\n✅ 所有基础模块导入成功！")

# 如果基础模块没问题，尝试创建一个简单的环境
if not errors:
    print("\n=== 测试创建环境 ===")
    try:
        from omegaconf import OmegaConf
        
        # 创建最小配置
        cfg = OmegaConf.create({
            'env': {
                'name': 'area_coverage',
                'frame_skip': 1,
                'num_envs': 1,  # 使用单个环境测试
                'device': 'cpu',
                'seed': 42,
            },
            'replay_buffer': {
                'size': 1000,
                'prb': True,  # 使用优先级回放
                'alpha': 0.6,
                'beta': 0.4,
            }
        })
        
        print("尝试创建环境...")
        env = make_area_coverage_env(
            num_envs=cfg.env.num_envs,
            device=cfg.env.device,
        )
        print(f"✓ 环境创建成功: {env}")
        
    except Exception as e:
        print(f"✗ 创建环境失败: {e}")
        traceback.print_exc()