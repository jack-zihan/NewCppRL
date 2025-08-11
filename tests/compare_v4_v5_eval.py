"""
对比V4和V5评估脚本的差异，找出问题所在
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import envs  # 注册环境
import gymnasium as gym
import yaml
from omegaconf import DictConfig
import numpy as np
import torch

def test_environments():
    """测试V4和V5环境的差异"""
    
    print("=" * 60)
    print("测试V4和V5环境差异")
    print("=" * 60)
    
    base_dir = Path(__file__).parent.parent
    
    # 测试V4环境
    print("\n1. 测试V4环境（area_coverage）")
    try:
        v4_cfg = DictConfig(yaml.load(
            open(f'{base_dir}/configs/env_config_area_coverage.yaml'), 
            Loader=yaml.FullLoader
        ))
        print(f"   环境ID: {v4_cfg.env.params.id}")
        
        v4_env = gym.make(render_mode=None, **v4_cfg.env.params)
        obs, info = v4_env.reset()
        
        print(f"   ✓ V4环境创建成功")
        print(f"   观察空间键: {list(obs.keys())}")
        print(f"   observation shape: {obs['observation'].shape}")
        print(f"   vector shape: {obs['vector'].shape}")
        print(f"   vector值: {obs['vector']}")
        print(f"   weed_ratio值: {obs['weed_ratio']}")
        
    except Exception as e:
        print(f"   ✗ V4环境创建失败: {e}")
    
    # 测试V5环境
    print("\n2. 测试V5环境（area_coverage_v5）")
    try:
        v5_cfg = DictConfig(yaml.load(
            open(f'{base_dir}/configs/env_config_area_coverage_v5.yaml'), 
            Loader=yaml.FullLoader
        ))
        print(f"   环境ID: {v5_cfg.env.params.id}")
        
        v5_env = gym.make(render_mode=None, **v5_cfg.env.params)
        obs, info = v5_env.reset()
        
        print(f"   ✓ V5环境创建成功")
        print(f"   观察空间键: {list(obs.keys())}")
        print(f"   observation shape: {obs['observation'].shape}")
        print(f"   vector shape: {obs['vector'].shape}")
        print(f"   vector值: {obs['vector']}")
        print(f"   weed_ratio值: {obs['weed_ratio']}")
        
    except Exception as e:
        print(f"   ✗ V5环境创建失败: {e}")

def test_evaluator_classes():
    """测试评估类的差异"""
    
    print("\n" + "=" * 60)
    print("测试评估类的差异")
    print("=" * 60)
    
    # 导入评估类
    from rl.sac_cont.area_coverage_sac_cont_eval import AreaCoverageSacEvaluator
    from rl.sac_cont.area_coverage_v5_sac_cont_eval import AreaCoverageV5SacEvaluator
    
    print("\n1. V4评估器（AreaCoverageSacEvaluator）")
    print(f"   algo_name: {AreaCoverageSacEvaluator.algo_name}")
    print(f"   env_cfg文件: env_config_area_coverage.yaml")
    
    print("\n2. V5评估器（AreaCoverageV5SacEvaluator）")
    print(f"   algo_name: {AreaCoverageV5SacEvaluator.algo_name}")
    print(f"   env_cfg文件: env_config_area_coverage_v5.yaml")
    
    # 检查get_actions方法
    print("\n3. 检查get_actions方法签名")
    import inspect
    
    v4_get_actions = inspect.getsource(AreaCoverageSacEvaluator.get_actions)
    v5_get_actions = inspect.getsource(AreaCoverageV5SacEvaluator.get_actions)
    
    print("\nV4 get_actions方法核心逻辑:")
    # 提取关键行
    for line in v4_get_actions.split('\n'):
        if 'actor(' in line:
            print(f"   {line.strip()}")
    
    print("\nV5 get_actions方法核心逻辑:")
    # 提取关键行
    for line in v5_get_actions.split('\n'):
        if 'actor(' in line or 'output' in line:
            print(f"   {line.strip()}")

def test_actual_evaluation():
    """测试实际的评估流程"""
    
    print("\n" + "=" * 60)
    print("测试实际评估流程")
    print("=" * 60)
    
    # 创建一个简单的测试
    print("\n创建测试数据...")
    
    # 模拟环境观察
    v4_obs = {
        'observation': np.random.randn(4, 16, 16).astype(np.float32),
        'vector': np.array([0.5], dtype=np.float32),
        'weed_ratio': 0.1
    }
    
    v5_obs = {
        'observation': np.random.randn(20, 16, 16).astype(np.float32),
        'vector': np.array([0.5], dtype=np.float32),
        'weed_ratio': 0.1
    }
    
    print(f"V4观察: observation shape {v4_obs['observation'].shape}")
    print(f"V5观察: observation shape {v5_obs['observation'].shape}")
    
    # 检查vector处理
    print("\n检查vector处理...")
    vector_wrapped = [v4_obs['vector'][0]]  # 评估脚本这样处理
    print(f"原始vector: {v4_obs['vector']}, shape: {v4_obs['vector'].shape}")
    print(f"包装后: {vector_wrapped}")
    
    vector_tensor = torch.tensor([vector_wrapped])
    print(f"转换为tensor: shape {vector_tensor.shape}")

if __name__ == "__main__":
    test_environments()
    test_evaluator_classes()
    test_actual_evaluation()