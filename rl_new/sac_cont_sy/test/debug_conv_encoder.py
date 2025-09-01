#!/usr/bin/env python3
"""调试ConvEncoder的通道数问题"""

import sys
sys.path.append('/home/lzh/NewCppRL')

import torch
from torchrl_utils.model.conv_encoder import ConvEncoder

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
        encoder = ConvEncoder(
            raster_shape=shape,
            cnn_channels=(32, 64, 64),
            kernel_sizes=(3, 3, 3),
            strides=(1, 1, 1),
            vec_dim=1,
            vec_out=512,
            cnn_activation_class=torch.nn.SiLU,
            mlp_activation_class=torch.nn.SiLU,
        )
        
        print(f"✓ 成功创建ConvEncoder")
        
        # 测试前向传播
        batch_size = 4
        test_input = torch.randn(batch_size, *shape)
        test_vector = torch.randn(batch_size, 1)
        
        output = encoder(test_input, test_vector)
        print(f"✓ 前向传播成功，输出形状: {output.shape}")
        
    except Exception as e:
        print(f"✗ 错误: {e}")
        
        # 打印更多调试信息
        print(f"  raster_shape[0] = {shape[0]}")
        
        # 手动检查第一个Conv层会被创建成什么
        in_ch = shape[0]
        print(f"  第一个_ConvNetBlock应该接收 in_ch={in_ch}")