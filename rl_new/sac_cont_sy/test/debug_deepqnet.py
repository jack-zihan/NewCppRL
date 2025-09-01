#!/usr/bin/env python3
"""调试DeepQNet的通道数问题"""

import sys
sys.path.append('/home/lzh/NewCppRL')

import torch
from torchrl_utils.model.deep_q_net import DeepQNet
from tensordict.nn import NormalParamExtractor

# 测试不同的输入形状
test_shapes = [
    (25, 16, 16),  # v2环境
    (15, 16, 16),  # v4环境
    (20, 16, 16),  # v5环境
    (1, 128, 128), # 标准单通道
]

for shape in test_shapes:
    print(f"\n测试 raster_shape={shape}")
    print("="*50)
    
    try:
        # 创建policy网络（带action_head）
        policy_net = DeepQNet(
            raster_shape=shape,
            cnn_channels=(32, 64, 64),
            kernel_sizes=(3, 3, 3),
            strides=(1, 1, 1),
            vec_dim=1,  # vector输入维度
            hidden_dim=512,
            output_num=4,  # 2 * action_dim (loc + scale)
            cnn_activation_class=torch.nn.SiLU,
            mlp_activation_class=torch.nn.SiLU,
            action_head=NormalParamExtractor(
                scale_mapping="biased_softplus_1.0",
                scale_lb=1e-4,
            ),
        )
        
        print(f"✓ 成功创建Policy DeepQNet")
        
        # 创建Q网络（不带action_head）
        q_net = DeepQNet(
            raster_shape=shape,
            cnn_channels=(32, 64, 64),
            kernel_sizes=(3, 3, 3),
            strides=(1, 1, 1),
            vec_dim=3,  # 1 (vector) + 2 (action)
            hidden_dim=512,
            output_num=1,  # Q值
            cnn_activation_class=torch.nn.SiLU,
            mlp_activation_class=torch.nn.SiLU,
        )
        
        print(f"✓ 成功创建Q DeepQNet")
        
        # 测试前向传播
        batch_size = 4
        test_obs = torch.randn(batch_size, *shape)
        test_vector = torch.randn(batch_size, 1)
        test_action = torch.randn(batch_size, 2)
        
        policy_output = policy_net(test_obs, test_vector)
        print(f"✓ Policy前向传播成功，输出形状: {policy_output.shape}")
        
        q_output = q_net(test_obs, test_vector, test_action)
        print(f"✓ Q前向传播成功，输出形状: {q_output.shape}")
        
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()