#!/usr/bin/env python
"""测试SAC损失函数创建"""

import torch
import sys

print("=== 测试SAC损失函数创建 ===")

try:
    from torchrl.objectives import SACLoss, SoftUpdate
    from rl.sac_cont.area_coverage_utils import make_area_coverage_sac_models
    
    print("1. 创建模型...")
    actor_critic = make_area_coverage_sac_models()
    actor = actor_critic[0]
    qvalue = actor_critic[1]
    print("✓ 模型创建成功")
    
    print("\n2. 创建损失函数...")
    device = torch.device("cpu")
    
    # 尝试创建损失函数
    loss_kwargs = {
        "actor_network": actor,
        "qvalue_network": qvalue,
        "num_qvalue_nets": 2,
        "loss_function": "smooth_l1",
        "alpha_init": 1.0,
        "target_entropy": -2,  # 使用正确的参数名
    }
    
    print("损失函数参数:")
    for k, v in loss_kwargs.items():
        if k in ["actor_network", "qvalue_network"]:
            print(f"  {k}: {type(v)}")
        else:
            print(f"  {k}: {v}")
    
    loss_module = SACLoss(**loss_kwargs)
    print("✓ 损失函数创建成功")
    
    print("\n3. 创建目标网络更新器...")
    target_net_updater = SoftUpdate(loss_module, tau=0.005)
    print("✓ 目标网络更新器创建成功")
    
    print("\n4. 创建优化器...")
    # 测试获取参数
    actor_params = loss_module.actor_network_params.values(True, True)
    qvalue_params = loss_module.qvalue_network_params.values(True, True)
    
    optimizer_actor = torch.optim.Adam(actor_params, lr=3e-4)
    optimizer_qvalue = torch.optim.Adam(qvalue_params, lr=3e-4)
    
    # 检查是否有log_alpha参数
    if hasattr(loss_module, "log_alpha"):
        optimizer_alpha = torch.optim.Adam([loss_module.log_alpha], lr=3e-4)
        print("✓ 三个优化器创建成功（包括alpha）")
    else:
        print("✓ 两个优化器创建成功（无alpha）")
    
    print("\n🎉 所有组件创建成功！")
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)