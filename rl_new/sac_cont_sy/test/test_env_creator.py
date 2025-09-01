#!/usr/bin/env python3
"""测试EnvCreator创建v4环境"""

import sys
sys.path.append('/home/lzh/NewCppRL')
sys.path.append('/home/lzh/NewCppRL/rl_new/sac_cont_sy')

import functools
from omegaconf import OmegaConf
from torchrl.envs import EnvCreator, ParallelEnv
from rl_new.sac_cont_sy.env_utils import make_env_lambda

# 加载配置
cfg = OmegaConf.load("/home/lzh/NewCppRL/rl_new/sac_cont_sy/config-async.yaml")
cfg.logger.backend = None
cfg.compile.enable = False
cfg.compile.cudagraph = False
cfg.collector.env_per_collector = 1
cfg.env.seed = 42
if 'env_kwargs' not in cfg.env or cfg.env.env_kwargs is None:
    cfg.env.env_kwargs = {}

print("测试v4环境的EnvCreator...")
cfg.env.env_id = "NewPasture-v4"

# 创建partial函数（与make_train_environment相同）
partial = functools.partial(make_env_lambda, env_id=cfg.env.env_id, device="cpu", 
                            from_pixels=False, **(cfg.env.get('env_kwargs') or {}))

print("\n1. 直接调用partial函数...")
try:
    env = partial()
    print(f"✓ 成功创建环境，observation shape: {env.observation_spec['observation'].shape}")
    td = env.reset()
    print(f"✓ Reset成功")
    td_next = env.step(td)
    print(f"✓ Step成功，reward: {td_next.get('reward', 0)}")
    env.close()
except Exception as e:
    print(f"✗ 失败: {e}")

print("\n2. 使用EnvCreator...")
try:
    env_creator = EnvCreator(partial)
    print(f"✓ EnvCreator初始化成功")
    
    env = env_creator()
    print(f"✓ 从EnvCreator创建环境成功")
    
    td = env.reset()
    print(f"✓ Reset成功")
    td_next = env.step(td)
    print(f"✓ Step成功，reward: {td_next.get('reward', 0)}")
    env.close()
except Exception as e:
    print(f"✗ 失败: {e}")
    import traceback
    traceback.print_exc()

print("\n3. 使用ParallelEnv...")
try:
    parallel_env = ParallelEnv(1, EnvCreator(partial), serial_for_single=True)
    print(f"✓ ParallelEnv创建成功")
    
    td = parallel_env.reset()
    print(f"✓ Reset成功")
    td_next = parallel_env.step(td)
    print(f"✓ Step成功")
    parallel_env.close()
except Exception as e:
    print(f"✗ 失败: {e}")
    import traceback
    traceback.print_exc()