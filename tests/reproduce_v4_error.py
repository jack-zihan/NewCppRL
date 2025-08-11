"""
复现V4评估脚本的错误
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

def reproduce_v4_error():
    """复现V4评估的错误"""
    
    print("=" * 60)
    print("复现V4 SAC评估错误")
    print("=" * 60)
    
    base_dir = Path(__file__).parent.parent
    
    # 加载V4配置
    v4_cfg = DictConfig(yaml.load(
        open(f'{base_dir}/configs/env_config_area_coverage.yaml'), 
        Loader=yaml.FullLoader
    ))
    
    print(f"\n1. V4环境配置:")
    print(f"   环境ID: {v4_cfg.env.params.id}")
    
    # 创建V4环境
    print("\n2. 创建V4环境...")
    envs_list = []
    for i in range(2):  # 创建2个环境模拟评估
        env = gym.make(render_mode=None, **v4_cfg.env.params)
        envs_list.append(env)
    
    # 重置环境获取观察
    print("\n3. 重置环境...")
    obss = []
    for env in envs_list:
        obs, _ = env.reset()
        obss.append(obs)
        print(f"   环境观察: observation shape {obs['observation'].shape}")
        print(f"            vector shape {obs['vector'].shape}")
    
    # 处理观察（模拟get_actions）
    print("\n4. 处理观察数据（模拟get_actions）...")
    observation_list = []
    vector_list = []
    for obs in obss:
        if isinstance(obs, dict):
            observation_list.append(obs['observation'])
            vector_list.append([obs['vector'][0]])  # 注意取第一个元素并包装
    
    print(f"   observation_list[0] shape: {observation_list[0].shape}")
    print(f"   vector_list[0]: {vector_list[0]}")
    
    observation = torch.from_numpy(np.stack(observation_list, axis=0)).float()
    vector = torch.tensor(np.array(vector_list)).float()
    
    print(f"   堆叠后observation shape: {observation.shape}")
    print(f"   堆叠后vector shape: {vector.shape}")
    
    # 尝试加载并调用模型
    print("\n5. 尝试加载V4模型...")
    
    # 查找V4的checkpoint
    v4_ckpt_dir = base_dir / 'ckpt' / 'area_coverage_sac_cont'
    
    if v4_ckpt_dir.exists():
        # 找到最新的目录
        dirs = [d for d in v4_ckpt_dir.iterdir() if d.is_dir()]
        if dirs:
            latest_dir = sorted(dirs)[-1]
            print(f"   找到checkpoint目录: {latest_dir.name}")
            
            # 查找.pt文件
            pt_files = list(latest_dir.glob("*.pt"))
            if pt_files:
                pt_file = pt_files[0]
                print(f"   找到模型文件: {pt_file.name}")
                
                try:
                    model = torch.load(str(pt_file), weights_only=False)
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    actor = model[0].to(device)
                    
                    observation = observation.to(device)
                    vector = vector.to(device)
                    
                    print(f"\n6. 调用actor...")
                    print(f"   输入observation shape: {observation.shape}")
                    print(f"   输入vector shape: {vector.shape}")
                    
                    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
                        try:
                            output = actor(observation=observation, vector=vector)
                            print(f"   ✓ 调用成功!")
                            if isinstance(output, tuple) and len(output) >= 3:
                                actions = output[2].tolist()
                                print(f"   获取动作: {actions}")
                        except Exception as e:
                            print(f"   ✗ 调用失败!")
                            print(f"   错误: {e}")
                            
                            # 详细分析错误
                            print("\n   错误分析:")
                            if "dimensions" in str(e):
                                print("   - 这是维度不匹配错误")
                                print("   - 可能原因：")
                                print("     1. 模型期望的输入shape与实际不符")
                                print("     2. ConvEncoder中concatenate操作失败")
                                print("     3. Vector在处理过程中维度变化")
                            
                            # 尝试使用TensorDict
                            print("\n7. 尝试使用TensorDict调用...")
                            from tensordict import TensorDict
                            td = TensorDict({
                                'observation': observation,
                                'vector': vector
                            }, batch_size=[len(obss)])
                            
                            try:
                                td_out = actor(td)
                                print(f"   ✓ TensorDict调用成功!")
                                if 'action' in td_out:
                                    print(f"   action shape: {td_out['action'].shape}")
                            except Exception as e2:
                                print(f"   ✗ TensorDict调用也失败: {e2}")
                                
                except Exception as e:
                    print(f"   ✗ 加载模型失败: {e}")
            else:
                print("   未找到.pt文件")
        else:
            print("   未找到checkpoint目录")
    else:
        print(f"   V4 checkpoint目录不存在: {v4_ckpt_dir}")
        
    print("\n" + "=" * 60)
    print("问题诊断")
    print("=" * 60)
    print("V4的问题可能是：")
    print("1. 环境输出shape (4, 128, 128) 但模型期望 (4, 16, 16)")
    print("2. 或者模型训练时用的环境配置与评估时不同")
    print("3. Vector处理方式可能有问题")

if __name__ == "__main__":
    reproduce_v4_error()