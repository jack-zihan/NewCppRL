#!/usr/bin/env python3
"""检查环境的实际观察空间形状"""

import sys
sys.path.append('/home/lzh/NewCppRL')
sys.path.append('/home/lzh/NewCppRL/rl_new/sac_cont_sy')

from omegaconf import OmegaConf
from rl_new.sac_cont_sy.env_utils import make_single_environment

# 加载配置
cfg = OmegaConf.load("/home/lzh/NewCppRL/rl_new/sac_cont_sy/config-async.yaml")
cfg.logger.backend = None
cfg.compile.enable = False
cfg.compile.cudagraph = False
cfg.collector.env_per_collector = 1
cfg.env.seed = 42
if 'env_kwargs' not in cfg.env or cfg.env.env_kwargs is None:
    cfg.env.env_kwargs = {}

# 测试三个环境
for env_id in ["NewPasture-v2", "NewPasture-v4", "NewPasture-v5"]:
    print(f"\n{'='*60}")
    print(f"环境: {env_id}")
    print(f"{'='*60}")
    
    cfg.env.env_id = env_id
    
    try:
        env = make_single_environment(cfg, device="cpu")
        
        # 获取observation spec
        obs_spec = env.observation_spec["observation"]
        print(f"Observation shape: {obs_spec.shape}")
        print(f"Observation dtype: {obs_spec.dtype}")
        
        # 实际reset获取数据
        td = env.reset()
        actual_obs = td["observation"]
        print(f"实际observation shape: {actual_obs.shape}")
        print(f"实际observation dtype: {actual_obs.dtype}")
        
        # 获取action spec
        action_spec = env.action_spec
        print(f"Action shape: {action_spec.shape}")
        print(f"Action space: {action_spec.space}")
        
        env.close()
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()