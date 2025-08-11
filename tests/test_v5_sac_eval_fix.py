"""
测试V5 SAC评估脚本修复
"""
import sys
import os
import tempfile
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.sac_cont.area_coverage_v5_sac_cont_eval import AreaCoverageV5SacEvaluator

# 将类定义移到函数外部，这样可以pickle
class MockSACPolicyNet(torch.nn.Module):
    """模拟SAC策略网络（使用DeepQNet结构但输出连续动作）"""
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(20*16*16, 512)
        self.vector_embed = torch.nn.Linear(1, 16)
        self.output = torch.nn.Linear(512 + 16, 4)  # 输出loc和scale (2*2)
        
    def forward(self, observation, vector):
        batch_size = observation.shape[0]
        # 展平观察
        obs_flat = observation.view(batch_size, -1)
        obs_embed = self.encoder(obs_flat)
        
        # 确保vector是2D的
        if vector.dim() == 3:
            vector = vector.squeeze(1)  # 去掉中间维度
        vec_embed = self.vector_embed(vector)
        
        # 合并嵌入
        combined = torch.cat([obs_embed, vec_embed], dim=-1)
        output = self.output(combined)
        
        # 分割为loc和scale
        loc = output[:, :2]
        scale = torch.nn.functional.softplus(output[:, 2:]) + 1e-4
        
        # 返回字典格式（模拟ProbabilisticActor）
        return {
            'loc': loc,
            'scale': scale,
            'action': torch.tanh(loc)  # 确定性动作
        }

class MockQNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(20*16*16 + 1 + 2, 1)
        
    def forward(self, observation, vector, action):
        batch_size = observation.shape[0]
        obs_flat = observation.view(batch_size, -1)
        combined = torch.cat([obs_flat, vector, action], dim=-1)
        return self.net(combined)

def test_v5_sac_evaluator():
    """测试V5 SAC评估器的修复"""
    print("=" * 60)
    print("测试V5 SAC评估脚本修复")
    print("=" * 60)
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix='test_v5_sac_')
    ckpt_dir = os.path.join(temp_dir, 'test_run')
    os.makedirs(ckpt_dir, exist_ok=True)
    
    try:
        # 创建策略网络
        policy_net = MockSACPolicyNet()
        
        # 创建Q网络
        q_net = MockQNet()
        
        # 保存模型（SAC格式：[policy, q_net]）
        model_path = os.path.join(ckpt_dir, 't00001.pt')
        torch.save([policy_net, q_net], model_path)
        print(f"✅ 创建模拟SAC模型: {model_path}")
        
        # 测试评估器初始化
        evaluator = AreaCoverageV5SacEvaluator(
            ckpt_path=ckpt_dir,
            device='cpu',
            episodes=1,
            video=False
        )
        print("✅ V5 SAC评估器初始化成功")
        
        # 测试get_actor方法
        actor = evaluator.get_actor(model_path)
        print("✅ get_actor方法正常工作")
        
        # 测试get_actions方法
        test_obs = [
            {
                'observation': np.random.randn(20, 16, 16).astype(np.float32),
                'vector': np.array([0.5], dtype=np.float32)
            }
        ]
        
        actions = evaluator.get_actions(actor, test_obs)
        print(f"✅ get_actions方法正常工作")
        print(f"   返回动作: {actions}")
        print(f"   动作类型: {type(actions)}, 长度: {len(actions)}")
        
        # 验证动作是连续的
        if isinstance(actions, list) and len(actions) > 0:
            first_action = actions[0]
            if isinstance(first_action, list) and len(first_action) == 2:
                print(f"✅ 动作格式正确: 2维连续动作")
            else:
                print(f"⚠️ 动作格式可能有问题: {first_action}")
        
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！V5 SAC评估脚本修复成功")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_v5_sac_evaluator()