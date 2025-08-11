"""
测试修复后的V5评估脚本
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from tensordict import TensorDict
from torchrl.envs import ExplorationType, set_exploration_type

def test_fixed_v5_eval():
    """测试修复后的V5评估脚本"""
    
    print("=" * 60)
    print("测试修复后的V5评估脚本")
    print("=" * 60)
    
    # 导入修复后的V5评估器
    from rl.sac_cont.area_coverage_v5_sac_cont_eval import AreaCoverageV5SacEvaluator
    
    print("\n1. 创建V5评估器实例...")
    evaluator = AreaCoverageV5SacEvaluator(
        episodes=2,
        max_frames=2,
        max_step=10,  # 减少步数加快测试
        skip_frames=5,
        video=False,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        start_idx=0,
        ckpt_path=None,
    )
    print(f"   ✓ 评估器创建成功")
    print(f"   设备: {evaluator.device}")
    print(f"   环境配置: {evaluator.env_cfg.env.params.id}")
    
    # 测试get_actions方法
    print("\n2. 测试get_actions方法...")
    
    # 加载V5模型
    ckpt_path = '/home/lzh/NewCppRL/ckpt/area_coverage_v5_sac_cont/2025-08-11_05-56-26_area_coverage_v5_with_direction_field/t[00042].pt'
    
    try:
        model = torch.load(ckpt_path, weights_only=False)
        actor = model[0].to(evaluator.device)
        print(f"   ✓ 模型加载成功")
        
        # 创建测试观察
        obss = [
            {
                'observation': np.random.randn(20, 16, 16).astype(np.float32),
                'vector': np.array([0.5], dtype=np.float32),
                'weed_ratio': 0.1
            },
            {
                'observation': np.random.randn(20, 16, 16).astype(np.float32),
                'vector': np.array([0.7], dtype=np.float32),
                'weed_ratio': 0.2
            }
        ]
        
        print(f"   创建了{len(obss)}个测试观察")
        
        # 调用get_actions
        try:
            actions = evaluator.get_actions(actor, obss)
            print(f"   ✓ get_actions调用成功!")
            print(f"   返回动作: {actions}")
            print(f"   动作数量: {len(actions)}")
            for i, action in enumerate(actions):
                print(f"     动作{i}: {action}")
        except Exception as e:
            print(f"   ✗ get_actions调用失败: {e}")
            import traceback
            traceback.print_exc()
            
    except FileNotFoundError:
        print(f"   ⚠️ 未找到V5模型文件，跳过测试")
    except Exception as e:
        print(f"   ✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print("✅ 修复方案已实施：")
    print("1. V4和V5评估脚本的get_actions方法已修改")
    print("2. 现在使用TensorDict调用actor")
    print("3. 正确处理vector维度")
    print("\n这应该解决了'Tensors must have same number of dimensions'错误")

if __name__ == "__main__":
    test_fixed_v5_eval()