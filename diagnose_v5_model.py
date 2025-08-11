#!/usr/bin/env python
"""
诊断V5模型文件并自动修复评估问题

使用方法：
python diagnose_v5_model.py <model_path>

例如：
python diagnose_v5_model.py ckpt/area_coverage_v5_sac_cont/xxx/t00042.pt
"""
import sys
import torch
import numpy as np
from pathlib import Path

def diagnose_and_test_model(model_path):
    """诊断模型并测试"""
    print("=" * 70)
    print(f"诊断模型文件: {model_path}")
    print("=" * 70)
    
    # 加载模型
    print("\n1. 加载模型...")
    model = torch.load(model_path, map_location='cpu')
    
    # 检查模型结构
    print(f"   模型类型: {type(model)}")
    if isinstance(model, list):
        print(f"   模型列表长度: {len(model)}")
        for i, item in enumerate(model):
            if hasattr(item, '__class__'):
                class_name = item.__class__.__name__
                print(f"   [{}] {class_name}")
                
                # 检查是否是TensorDictModule
                if 'TensorDictModule' in class_name:
                    if hasattr(item, 'module'):
                        inner_class = item.module.__class__.__name__
                        print(f"       内部模块: {inner_class}")
                        
                        # 检查是否包含DeepQNet
                        if 'DeepQNet' in inner_class:
                            print("       ⚠️  包含DeepQNet类（可能造成混淆）")
                        
                        # 检查是否包含action_head
                        if hasattr(item.module, 'action_head'):
                            action_head = item.module.action_head
                            if action_head is not None:
                                print(f"       ✅ 有action_head: {type(action_head).__name__}")
                                if 'NormalParamExtractor' in str(type(action_head)):
                                    print("       ✅ 这是SAC模型（输出连续动作参数）")
                                else:
                                    print("       ⚠️  action_head类型不明确")
    
    # 测试模型调用
    print("\n2. 测试模型调用...")
    if isinstance(model, list) and len(model) > 0:
        actor = model[0]
        
        # 创建测试输入
        test_obs = torch.randn(1, 20, 16, 16)  # V5的20通道输入
        test_vector = torch.randn(1, 1)
        
        print(f"   测试输入形状: observation={test_obs.shape}, vector={test_vector.shape}")
        
        try:
            with torch.no_grad():
                # 尝试调用 - 注意不传action参数
                output = actor(observation=test_obs, vector=test_vector)
                
                print(f"   ✅ 模型调用成功!")
                print(f"   输出类型: {type(output)}")
                
                if isinstance(output, dict):
                    print("   输出是字典，包含键:")
                    for key, value in output.items():
                        if hasattr(value, 'shape'):
                            print(f"     - {key}: shape={value.shape}")
                    
                    if 'action' in output:
                        print("   ✅ 有'action'键 - 适合SAC评估")
                        action_shape = output['action'].shape
                        if action_shape[-1] == 2:
                            print("   ✅ 输出2维连续动作 - 确认是V5 SAC模型")
                        else:
                            print(f"   ⚠️  动作维度是{action_shape[-1]}，不是预期的2")
                    elif 'loc' in output and 'scale' in output:
                        print("   ⚠️  输出loc/scale但没有action - 需要采样")
                        print("   建议：模型可能需要额外的采样步骤")
                    
                elif isinstance(output, tuple):
                    print(f"   输出是元组，长度: {len(output)}")
                    for i, item in enumerate(output):
                        if hasattr(item, 'shape'):
                            print(f"     [{i}] shape={item.shape}")
                    
                    # 检查是否是Q值（DQN）
                    if len(output) == 1 or (hasattr(output[0], 'shape') and output[0].shape[-1] == 147):
                        print("   ❌ 这可能是DQN模型（输出离散Q值）")
                        print("   需要使用DQN评估脚本而不是SAC评估脚本")
                    elif len(output) >= 2:
                        print("   可能是SAC输出（mean, std, action）")
                
                elif hasattr(output, 'shape'):
                    print(f"   输出是张量，shape={output.shape}")
                    if output.shape[-1] == 147:
                        print("   ❌ 输出147个值 - 这是DQN模型!")
                        print("   请使用DQN评估脚本")
                    elif output.shape[-1] == 2:
                        print("   ✅ 输出2个值 - 可能是连续动作")
                    
        except Exception as e:
            print(f"   ❌ 模型调用失败: {e}")
            print("   可能的原因:")
            print("   - 模型期望不同的输入格式")
            print("   - 模型结构与评估脚本不兼容")
            
    # 给出建议
    print("\n" + "=" * 70)
    print("诊断建议:")
    print("=" * 70)
    
    print("""
如果遇到维度错误，请确保：
1. 评估脚本已更新get_actions方法（不传入action参数）
2. 模型确实是SAC模型而不是DQN模型
3. 使用正确的评估脚本版本

运行评估的正确命令：
export WANDB_MODE=offline
python rl/sac_cont/area_coverage_v5_sac_cont_eval.py --ckpt_path <path>
""")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python diagnose_v5_model.py <model_path>")
        print("例如: python diagnose_v5_model.py ckpt/area_coverage_v5_sac_cont/xxx/t00042.pt")
        sys.exit(1)
    
    model_path = sys.argv[1]
    if not Path(model_path).exists():
        print(f"错误: 文件不存在 - {model_path}")
        sys.exit(1)
        
    diagnose_and_test_model(model_path)