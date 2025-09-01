#!/usr/bin/env python3
"""最小测试案例"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import torch
from omegaconf import OmegaConf

# 导入方式1：与test_sac_comprehensive.py相同
from rl_new.sac_cont_sy.model_utils import make_sac_models as make_sac_models1
from rl_new.sac_cont_sy.env_utils import make_single_environment as make_env1

# 导入方式2：相对导入
sys.path.append('/home/lzh/NewCppRL/rl_new/sac_cont_sy')
from model_utils import make_sac_models as make_sac_models2
from env_utils import make_single_environment as make_env2

# 加载配置
cfg = OmegaConf.load("/home/lzh/NewCppRL/rl_new/sac_cont_sy/config-async.yaml")
cfg.logger.backend = None
cfg.compile.enable = False
cfg.compile.cudagraph = False
cfg.collector.env_per_collector = 1
cfg.env.seed = 42
if 'env_kwargs' not in cfg.env or cfg.env.env_kwargs is None:
    cfg.env.env_kwargs = {}

cfg.env.env_id = "NewPasture-v2"

print("测试导入方式1...")
try:
    env = make_env1(cfg, device="cpu")
    print(f"环境observation shape: {env.observation_spec['observation'].shape}")
    actor_critic = make_sac_models1(env=env)
    print("✓ 成功")
    env.close()
except Exception as e:
    print(f"✗ 失败: {e}")

print("\n测试导入方式2...")
try:
    env = make_env2(cfg, device="cpu")
    print(f"环境observation shape: {env.observation_spec['observation'].shape}")
    actor_critic = make_sac_models2(env=env)
    print("✓ 成功")
    env.close()
except Exception as e:
    print(f"✗ 失败: {e}")