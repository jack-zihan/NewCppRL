#!/usr/bin/env python3
"""调试make_sac_modules函数"""

import sys
sys.path.append('/home/lzh/NewCppRL')
sys.path.append('/home/lzh/NewCppRL/rl_new/sac_cont_sy')

from omegaconf import OmegaConf
from rl_new.sac_cont_sy.env_utils import make_single_environment
from rl_new.sac_cont_sy.model_utils import make_sac_modules

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
    print(f"测试环境: {env_id}")
    print(f"{'='*60}")
    
    cfg.env.env_id = env_id
    
    try:
        # 创建环境
        env = make_single_environment(cfg, device="cpu")
        
        # 打印环境的observation spec
        obs_spec = env.observation_spec["observation"]
        print(f"环境observation_spec.shape: {obs_spec.shape}")
        print(f"环境observation_spec.dtype: {obs_spec.dtype}")
        
        # 在make_sac_modules之前，直接检查input_shape
        input_shape = env.observation_spec["observation"].shape
        print(f"传给make_sac_modules的input_shape: {input_shape}")
        print(f"input_shape类型: {type(input_shape)}")
        print(f"input_shape[0] (通道数): {input_shape[0]}")
        
        # 调用make_sac_modules
        print("\n调用make_sac_modules...")
        policy_module, qvalue_module = make_sac_modules(env)
        
        print("✓ 成功创建SAC modules")
        
        env.close()
        
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()