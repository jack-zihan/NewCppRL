from typing import Sequence, Optional, Type

import torch
from torch import nn
import torch.nn.functional as F

from torchrl_utils.model.conv_encoder import ConvEncoder
from torchrl_utils.model.dueling_head import DuelingHead


class DeepQNet(nn.Module):
    def __init__(self,
                 raster_shape: Sequence[int],
                 cnn_channels: Sequence[int] = (16, 32, 64),
                 kernel_sizes: Sequence[int] = (8, 4, 3),
                 strides: Sequence[int] = (1, 1, 1),
                 vec_dim=14,
                 hidden_dim=256,
                 output_num=15,
                 cnn_activation_class: Optional[Type[nn.Module]] = nn.ELU,
                 mlp_activation_class: Optional[Type[nn.Module]] = nn.ReLU,
                 dueling_head: bool = False,
                 action_head: Optional[nn.Module] = None):
        super(DeepQNet, self).__init__()
        self.encoder = ConvEncoder(
            raster_shape=raster_shape,
            cnn_channels=cnn_channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            vec_dim=vec_dim,
            vec_out=hidden_dim,
            cnn_activation_class=cnn_activation_class,
            mlp_activation_class=mlp_activation_class,
        )
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
        )
        if mlp_activation_class:
            self.q_head.append(mlp_activation_class(inplace=False))
        if dueling_head:
            self.q_head.append(DuelingHead(hidden_dim, output_num))
        else:
            self.q_head.append(nn.Linear(hidden_dim, output_num))
        self.action_head = action_head

    def forward(self, observation, vector=None, action=None):
        embed = self.encoder(observation, vector, action)
        q_values = self.q_head(embed)
        if self.action_head:
            q_values = self.action_head(q_values)
        return q_values


class CNNHIFDecoder(nn.Module):
    """轻量级HIF解码器：从CNN空间特征上采样到HIF方向场.

    设计目标：
    - 对输入尺寸无假设：只依赖最后 target_size 进行插值，兼容32×32/96×96等不同HIF分辨率；
    - 结构尽量轻量：少量3×3卷积 + 逐步上采样，避免引入比主干更重的UNet。
    """

    def __init__(self, in_channels: int, mid_channels: int = 64):
        super().__init__()
        # 使用GroupNorm提高数值稳定性（与ResNet-FPN保持一致的归纳偏置）
        def gn(c: int) -> nn.Module:
            return nn.GroupNorm(min(32, max(1, c // 4)), c)

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            gn(mid_channels),
            nn.SiLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            gn(mid_channels),
            nn.SiLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            gn(mid_channels),
            nn.SiLU(inplace=True),
        )
        # 输出2通道轴向场（cos2, sin2）
        self.out = nn.Conv2d(mid_channels, 2, kernel_size=1)

    def forward(self, x: torch.Tensor, target_size: Sequence[int]) -> torch.Tensor:
        """从空间特征解码到目标HIF尺寸.

        Args:
            x: [B, C_f, H_f, W_f] 的空间特征
            target_size: (H_target, W_target)，通常等于环境HIF patch或观测的空间尺寸
        """
        # 在原始分辨率上做一层卷积以聚合局部特征
        x = self.block1(x)

        # 逐步上采样两次，在中等分辨率上建模空间结构
        for block in (self.block2, self.block3):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = block(x)

        # 最后一层统一插值到目标HIF尺寸，确保与label保持一致
        x = F.interpolate(x, size=tuple(target_size), mode="bilinear", align_corners=False)
        hif_pred = self.out(x)  # [B, 2, H_target, W_target]
        return hif_pred


class CNNDualHeadActor(nn.Module):
    """CNN双头Actor：动作预测 + HIF方向场预测.

    - 主干：ConvEncoder（IMPALA风格CNN），与DeepQNet保持一致的卷积架构；
    - 动作头：ConvEncoder → 512维embedding → MLP输出action_params；
    - HIF头：从ConvEncoder的空间特征上采样生成 pred_ego_hif (cos2, sin2)。

    该结构与ResNet-FPN版dual-head保持接口一致：
    forward(observation, vector, return_hif=False) -> (action_params, hif_pred或None)
    """

    def __init__(
        self,
        raster_shape: Sequence[int],
        vec_dim: int,
        action_dim: int,
        cnn_channels: Sequence[int] = (32, 64, 128),
        kernel_sizes: Sequence[int] = (3, 3, 3),
        strides: Sequence[int] = (1, 1, 1),
        hidden_dim: int = 512,
        cnn_activation_class: Optional[Type[nn.Module]] = nn.SiLU,
        mlp_activation_class: Optional[Type[nn.Module]] = nn.SiLU,
        enable_hif: bool = True,
    ):
        super().__init__()
        self.enable_hif = enable_hif

        # 与DeepQNet一致的ConvEncoder配置（仅使用其cnn_encoder和post_encoder）
        self.encoder = ConvEncoder(
            raster_shape=raster_shape,
            cnn_channels=cnn_channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            vec_dim=vec_dim,
            vec_out=hidden_dim,
            cnn_activation_class=cnn_activation_class,
            mlp_activation_class=mlp_activation_class,
        )

        # 动作MLP头：512→512→2*action_dim，保持与DeepQNet.q_head一致的形态
        mlp_act = mlp_activation_class
        policy_layers = [nn.Linear(hidden_dim, hidden_dim)]
        if mlp_act is not None:
            policy_layers.append(mlp_act(inplace=False))
        policy_layers.append(nn.Linear(hidden_dim, 2 * action_dim))
        self.policy_head = nn.Sequential(*policy_layers)

        # 轻量HIF解码器：从最后一个CNN通道的空间特征生成HIF方向场
        self.hif_decoder = CNNHIFDecoder(in_channels=cnn_channels[-1], mid_channels=64)

    def forward(
        self,
        observation: torch.Tensor,
        vector: torch.Tensor,
        return_hif: bool = False,
    ):
        """前向推理.

        Args:
            observation: [B, C, H, W] 观测图
            vector: [B, vec_dim] 向量特征
            return_hif: 是否返回HIF预测（在HIF预训练/辅助阶段为True）

        Returns:
            action_params: [B, 2*action_dim]
            hif_pred: [B, 2, H, W] 或 None
        """
        # 1. 通过cnn_encoder（保留空间特征）
        x = observation
        modules = list(self.encoder.cnn_encoder.children())
        # 最后一层是SquashDims，保留之前的空间卷积模块
        for layer in modules[:-1]:
            x = layer(x)
        feat_spatial = x  # [B, C_f, H_f, W_f]

        # 2. 展平成embedding并拼接vector，复用ConvEncoder的post_encoder
        x_flat = modules[-1](feat_spatial)  # SquashDims → [B, C_flat]
        to_concat = [x_flat]
        if vector is not None:
            to_concat.append(vector)
        embed_in = torch.cat(to_concat, dim=-1)
        embed = self.encoder.post_encoder(embed_in)  # [B, hidden_dim]

        # 3. 动作参数头
        action_params = self.policy_head(embed)

        # 4. 可选HIF预测头
        hif_pred = None
        if return_hif and self.enable_hif:
            target_size = observation.shape[-2:]  # 与观测/label_ego_hif保持一致
            hif_pred = self.hif_decoder(feat_spatial, target_size)

        return action_params, hif_pred
