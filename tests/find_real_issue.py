"""
找出V5评估脚本真正的问题
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import envs
import gymnasium as gym
import yaml
from omegaconf import DictConfig
from torchrl.envs import ExplorationType, set_exploration_type
from tensordict import TensorDict

def find_the_issue():
    """找出V5评估脚本的真正问题"""
    
    print("=" * 60)
    print("找出V5评估脚本的真正问题")
    print("=" * 60)
    
    base_dir = Path(__file__).parent.parent
    
    # 1. 测试V5环境的不同创建方式
    print("\n1. 测试不同的环境创建方式...")
    
    # 方式1：直接使用gym.make（评估脚本使用的方式）
    v5_cfg = DictConfig(yaml.load(
        open(f'{base_dir}/configs/env_config_area_coverage_v5.yaml'), 
        Loader=yaml.FullLoader
    ))
    
    env1 = gym.make(render_mode=None, **v5_cfg.env.params)
    obs1, _ = env1.reset()
    print(f"   方式1 (gym.make): observation shape {obs1['observation'].shape}")
    print(f"                     vector shape {obs1['vector'].shape}")
    print(f"                     vector type: {type(obs1['vector'])}")
    
    # 方式2：使用V5的make_env函数
    from rl.sac_cont.area_coverage_v5_utils import make_area_coverage_v5_env
    env2 = make_area_coverage_v5_env(num_envs=1, device='cpu')
    td2 = env2.reset()
    print(f"\n   方式2 (make_area_coverage_v5_env): 返回类型 {type(td2)}")
    if hasattr(td2, 'keys'):
        for k in td2.keys():
            if hasattr(td2[k], 'shape'):
                print(f"                                      {k}: shape {td2[k].shape}")
    
    # 2. 测试观察数据处理
    print("\n2. 测试观察数据处理...")
    
    # 模拟评估脚本的观察处理
    obss = [obs1, obs1]  # 使用两个相同的观察
    
    observation_list = []
    vector_list = []
    for obs in obss:
        if isinstance(obs, dict):
            observation_list.append(obs['observation'])
            # 关键问题可能在这里！
            if isinstance(obs['vector'], np.ndarray):
                if obs['vector'].ndim == 1:
                    vector_list.append([obs['vector'][0]])  # 取第一个元素并包装
                else:
                    vector_list.append(obs['vector'].tolist())
            else:
                vector_list.append([obs['vector']])
    
    print(f"   observation_list[0] shape: {observation_list[0].shape}")
    print(f"   vector_list[0]: {vector_list[0]}, type: {type(vector_list[0])}")
    
    observation_tensor = torch.from_numpy(np.stack(observation_list, axis=0)).float()
    vector_tensor = torch.tensor(np.array(vector_list)).float()
    
    print(f"   转换后observation: {observation_tensor.shape}")
    print(f"   转换后vector: {vector_tensor.shape}")
    
    # 3. 加载V5模型并测试
    print("\n3. 测试V5模型调用...")
    ckpt_path = '/home/lzh/NewCppRL/ckpt/area_coverage_v5_sac_cont/2025-08-11_05-56-26_area_coverage_v5_with_direction_field/t[00042].pt'
    
    model = torch.load(ckpt_path, weights_only=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    actor = model[0].to(device)
    
    observation_tensor = observation_tensor.to(device)
    vector_tensor = vector_tensor.to(device)
    
    # 方式1：直接调用（评估脚本方式）
    print("\n   方式1：直接调用...")
    try:
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            output = actor(observation=observation_tensor, vector=vector_tensor)
            if isinstance(output, tuple) and len(output) >= 3:
                actions = output[2].tolist()
                print(f"      ✓ 成功！动作: {actions[0]}")
    except Exception as e:
        print(f"      ✗ 失败: {e}")
    
    # 方式2：使用TensorDict（推荐方式）
    print("\n   方式2：使用TensorDict...")
    try:
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            td = TensorDict({
                'observation': observation_tensor,
                'vector': vector_tensor
            }, batch_size=[len(obss)])
            td_out = actor(td)
            if 'action' in td_out:
                actions = td_out['action'].tolist()
                print(f"      ✓ 成功！动作: {actions[0]}")
    except Exception as e:
        print(f"      ✗ 失败: {e}")
    
    # 4. 问题分析
    print("\n" + "=" * 60)
    print("问题分析")
    print("=" * 60)
    
    print("问题可能的原因：")
    print("1. CustomEvaluator.run()创建的环境与make_env()不同")
    print("2. 环境返回的vector格式可能在不同情况下有差异")
    print("3. ProbabilisticActor在某些特定输入下会出现维度问题")
    
    # 5. 建议解决方案
    print("\n建议解决方案：")
    print("1. 在get_actions中使用TensorDict调用actor")
    print("2. 或者重写run()方法使用make_env()创建环境")
    print("3. 或者在vector处理时确保维度正确")

if __name__ == "__main__":
    find_the_issue()