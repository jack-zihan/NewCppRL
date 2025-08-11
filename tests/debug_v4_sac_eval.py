"""
调试V4 SAC评估脚本的问题
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from tensordict import TensorDict
from torchrl.envs import ExplorationType, set_exploration_type

def debug_actor_call():
    """调试actor调用问题"""
    
    print("=" * 60)
    print("调试V4 SAC评估脚本问题")
    print("=" * 60)
    
    # 加载V5模型进行测试（V4没有checkpoint，用V5的）
    ckpt_path = '/home/lzh/NewCppRL/ckpt/area_coverage_v5_sac_cont/2025-08-11_05-56-26_area_coverage_v5_with_direction_field/t[00042].pt'
    
    try:
        print("\n1. 加载模型...")
        model = torch.load(ckpt_path, weights_only=False)
        actor = model[0]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        actor = actor.to(device)
        print(f"   ✓ 模型加载成功，设备: {device}")
        print(f"   Actor类型: {type(actor).__name__}")
        
        # 检查actor结构
        print("\n2. 检查Actor结构...")
        print(f"   Actor.module类型: {type(actor.module).__name__}")
        if hasattr(actor.module, '__len__'):
            print(f"   Actor.module包含 {len(actor.module)} 个模块")
            for i, m in enumerate(actor.module):
                print(f"     [{i}] {type(m).__name__}")
        
        # 准备测试数据
        print("\n3. 准备测试数据...")
        batch_size = 2
        observation = torch.randn(batch_size, 20, 16, 16).to(device)  # V5是20通道
        vector = torch.randn(batch_size, 1).to(device)
        print(f"   observation shape: {observation.shape}")
        print(f"   vector shape: {vector.shape}")
        
        # 方法1：直接调用（评估脚本的方式）
        print("\n4. 测试方法1：直接关键字参数调用（当前评估脚本方式）")
        try:
            with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
                output = actor(observation=observation, vector=vector)
            print(f"   ✓ 调用成功！")
            print(f"   输出类型: {type(output)}")
            if isinstance(output, tuple):
                print(f"   元组长度: {len(output)}")
                for i, item in enumerate(output):
                    if hasattr(item, 'shape'):
                        print(f"     [{i}] shape: {item.shape}")
        except Exception as e:
            print(f"   ✗ 调用失败: {e}")
            import traceback
            print("   详细错误:")
            traceback.print_exc()
        
        # 方法2：使用TensorDict（推荐方式）
        print("\n5. 测试方法2：使用TensorDict调用（推荐方式）")
        try:
            with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
                td = TensorDict({
                    'observation': observation,
                    'vector': vector
                }, batch_size=[batch_size])
                td_out = actor(td)
            print(f"   ✓ 调用成功！")
            print(f"   输出类型: {type(td_out)}")
            if hasattr(td_out, 'keys'):
                print(f"   输出键: {list(td_out.keys())}")
                if 'action' in td_out:
                    print(f"   action shape: {td_out['action'].shape}")
        except Exception as e:
            print(f"   ✗ 调用失败: {e}")
            import traceback
            traceback.print_exc()
            
        # 方法3：模拟评估脚本的完整流程
        print("\n6. 模拟评估脚本的完整流程...")
        # 模拟从环境获得的观察
        obss = [
            {'observation': np.random.randn(20, 16, 16).astype(np.float32), 'vector': 0.5},
            {'observation': np.random.randn(20, 16, 16).astype(np.float32), 'vector': 0.7}
        ]
        
        print("   处理观察数据...")
        observation_list = []
        vector_list = []
        for obs in obss:
            observation_list.append(obs['observation'])
            vector_list.append([obs['vector']])  # 注意这里包装成列表
        
        print(f"   observation_list[0] shape: {observation_list[0].shape}")
        print(f"   vector_list[0]: {vector_list[0]}")
        
        observation = torch.from_numpy(np.stack(observation_list, axis=0)).float().to(device)
        vector = torch.tensor(np.array(vector_list)).float().to(device)
        
        print(f"   最终observation shape: {observation.shape}")
        print(f"   最终vector shape: {vector.shape}")
        
        # 尝试调用
        print("   尝试调用actor...")
        try:
            with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
                output = actor(observation=observation, vector=vector)
                if isinstance(output, tuple) and len(output) >= 3:
                    actions = output[2].tolist()
                    print(f"   ✓ 成功获取动作: {actions}")
                else:
                    print(f"   ✗ 输出格式不符合预期")
        except Exception as e:
            print(f"   ✗ 调用失败: {e}")
            
    except FileNotFoundError:
        print("   ✗ 未找到checkpoint文件，跳过模型测试")
        print("   继续进行理论分析...")
        
    print("\n" + "=" * 60)
    print("问题分析总结")
    print("=" * 60)
    print("1. 问题根源：ProbabilisticActor在接收关键字参数时的内部处理")
    print("2. 维度不匹配发生在ConvEncoder的concatenate操作")
    print("3. 解决方案：使用TensorDict调用actor，而不是直接传递关键字参数")
    
if __name__ == "__main__":
    debug_actor_call()