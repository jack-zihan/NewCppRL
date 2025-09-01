#!/usr/bin/env python3
"""测试v4环境的step操作"""

import sys
sys.path.append('/home/lzh/NewCppRL')
sys.path.append('/home/lzh/NewCppRL/rl_new/sac_cont_sy')

from omegaconf import OmegaConf
from rl_new.sac_cont_sy.env_utils import make_single_environment
import torch

# 加载配置
cfg = OmegaConf.load("/home/lzh/NewCppRL/rl_new/sac_cont_sy/config-async.yaml")
cfg.logger.backend = None
cfg.compile.enable = False
cfg.compile.cudagraph = False
cfg.collector.env_per_collector = 1
cfg.env.seed = 42
if 'env_kwargs' not in cfg.env or cfg.env.env_kwargs is None:
    cfg.env.env_kwargs = {}

cfg.env.env_id = "NewPasture-v4"

print("创建v4环境...")
env = make_single_environment(cfg, device="cpu")

print(f"环境observation shape: {env.observation_spec['observation'].shape}")

print("\nReset环境...")
td = env.reset()
print(f"Reset成功，observation shape: {td['observation'].shape}")

print("\nStep环境...")
try:
    # 创建一个随机动作
    action = env.action_spec.rand()
    td['action'] = action
    print(f"Action shape: {action.shape}")
    
    # 执行step
    td_next = env.step(td)
    print(f"Step成功！")
    print(f"Reward: {td_next.get('reward', 0)}")
    
except Exception as e:
    print(f"Step失败: {e}")
    import traceback
    traceback.print_exc()

env.close()