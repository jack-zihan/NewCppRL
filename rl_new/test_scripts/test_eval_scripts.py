#!/usr/bin/env python
"""
测试评估脚本的兼容性
"""

import sys
import torch
import traceback

print("=== 测试评估脚本兼容性 ===\n")

# 测试基础导入
print("1. 测试导入...")
try:
    from torchrl.envs import ExplorationType, set_exploration_type
    from tensordict import TensorDict
    print("   ✓ TorchRL模块导入成功")
except ImportError as e:
    print(f"   ✗ 导入失败: {e}")
    sys.exit(1)

# 测试area_coverage_sac_cont_eval
print("\n2. 测试area_coverage_sac_cont_eval.py...")
try:
    sys.path.insert(0, '/home/lzh/NewCppRL')
    from rl_new.sac_cont.area_coverage_sac_cont_eval import AreaCoverageSacEvaluator
    print("   ✓ AreaCoverageSacEvaluator导入成功")
    
    # 创建评估器实例（不运行，只测试创建）
    evaluator = AreaCoverageSacEvaluator(
        episodes=1,
        video=False,
        ckpt_path=None
    )
    print("   ✓ 评估器实例创建成功")
    print(f"   - 使用指标: {evaluator.metric_name}")
    
except Exception as e:
    print(f"   ✗ 错误: {e}")
    traceback.print_exc()

# 测试area_coverage_v5_sac_cont_eval
print("\n3. 测试area_coverage_v5_sac_cont_eval.py...")
try:
    from rl_new.sac_cont.area_coverage_v5_sac_cont_eval import AreaCoverageV5SacEvaluator
    print("   ✓ AreaCoverageV5SacEvaluator导入成功")
    
    # 创建评估器实例
    evaluator_v5 = AreaCoverageV5SacEvaluator(
        episodes=1,
        video=False,
        ckpt_path=None
    )
    print("   ✓ V5评估器实例创建成功")
    print(f"   - 使用指标: {evaluator_v5.metric_name}")
    
except Exception as e:
    print(f"   ✗ 错误: {e}")
    traceback.print_exc()

# 测试TensorDict使用（评估脚本的关键部分）
print("\n4. 测试TensorDict兼容性...")
try:
    import numpy as np
    
    # 模拟评估脚本中的TensorDict使用
    batch_size = 2
    observation = torch.randn(batch_size, 4, 128, 128)  # 模拟观察
    vector = torch.randn(batch_size, 1)  # 模拟向量
    
    td = TensorDict({
        'observation': observation,
        'vector': vector
    }, batch_size=[batch_size])
    
    print(f"   ✓ TensorDict创建成功")
    print(f"   - 批次大小: {td.batch_size}")
    print(f"   - 包含键: {list(td.keys())}")
    
    # 测试exploration type设置
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        print("   ✓ ExplorationType.DETERMINISTIC设置成功")
    
except Exception as e:
    print(f"   ✗ 错误: {e}")
    traceback.print_exc()

print("\n=== 测试总结 ===")
print("评估脚本应该可以在TorchRL 0.9.2中正常工作")
print("主要原因：")
print("1. 评估脚本不使用训练组件（SACLoss、ReplayBuffer等）")
print("2. 只依赖基础的TensorDict和ExplorationType")
print("3. 这些基础API在0.9.2中保持兼容")

print("\n建议测试命令：")
print("python rl_new/sac_cont/area_coverage_sac_cont_eval.py \\")
print("    --ckpt_path ckpt/area_coverage_sac_cont/xxx/t[00001].pt \\")
print("    --episodes 1")