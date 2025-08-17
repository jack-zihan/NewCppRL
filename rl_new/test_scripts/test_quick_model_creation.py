#!/usr/bin/env python
"""快速测试模型创建是否有问题"""

import torch
import sys

print("测试模型创建...")

try:
    from rl.sac_cont.area_coverage_utils import make_area_coverage_sac_models
    
    # 设置超时，防止卡住
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("模型创建超时")
    
    # 设置10秒超时
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)
    
    print("开始创建模型...")
    actor_critic = make_area_coverage_sac_models()
    
    # 取消超时
    signal.alarm(0)
    
    print(f"✓ 模型创建成功")
    print(f"  Actor: {type(actor_critic[0])}")
    print(f"  QValue: {type(actor_critic[1])}")
    
except TimeoutError as e:
    print(f"✗ {e}")
    print("模型创建函数可能在rollout步骤卡住了")
    sys.exit(1)
except Exception as e:
    print(f"✗ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)