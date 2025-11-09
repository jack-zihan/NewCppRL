"""
优化版ResNetFPNDualHeadActor - 遵循Less is More原则
拆分统一forward为独立的action和hif方法，消除返回类型不一致
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

# 从原版导入基础组件（不重复实现）
from torchrl_utils.model.resnet_fpn_dual import (
    ResNetFPNEncoder,
    ResNetFPNActionHead,
    TwoStageHIFDecoder,
    FullUNetHIFDecoder,
)


class ResNetFPNDualHeadActorOptimized(nn.Module):
    """优化版双头Actor - 清晰的接口设计

    核心改进：
    1. 拆分forward为forward_action和forward_hif
    2. 每个方法单一职责，返回类型一致
    3. 避免不必要的计算（预训练时不计算action）

    业务本质：
    - 共享编码器提取特征
    - Action头用于SAC训练
    - HIF头用于预训练和辅助
    """

    def __init__(
        self,
        in_channels: int = 5,
        vec_dim: int = 14,
        action_dim: int = 2,
        fpn_channels: int = 256,
        hidden_dim: int = 512,
        pretrained: bool = True,
        hif_decoder_type: str = 'two_stage',
        decoder_channels: int = 128
    ):
        super().__init__()

        # 共享编码器
        self.encoder = ResNetFPNEncoder(
            in_channels=in_channels,
            fpn_channels=fpn_channels,
            pretrained=pretrained,
            use_groupnorm=True
        )

        # Action头
        self.action_head = ResNetFPNActionHead(
            fpn_channels=fpn_channels,
            vec_dim=vec_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            output_loc_scale=True
        )

        # HIF解码器头
        if hif_decoder_type == 'two_stage':
            self.hif_decoder = TwoStageHIFDecoder(
                fpn_channels=fpn_channels,
                decoder_channels=decoder_channels,
                output_channels=2,
                use_groupnorm=True,
            )
        elif hif_decoder_type == 'full':
            self.hif_decoder = FullUNetHIFDecoder(
                fpn_channels=fpn_channels,
                decoder_channels=decoder_channels,
                output_channels=2,
                use_groupnorm=True,
            )
        else:
            raise ValueError(f"Unknown decoder type: {hif_decoder_type}")

        self.hif_decoder_type = hif_decoder_type

    def forward_action(
        self,
        observation: torch.Tensor,
        vector: torch.Tensor
    ) -> torch.Tensor:
        """仅预测action - SAC训练时调用

        单一职责：计算action参数，不涉及HIF
        返回类型一致：始终返回action_params

        Args:
            observation: 栅格观察 [B, C, H, W]
            vector: 向量观察 [B, vec_dim]

        Returns:
            action_params: Action参数 [B, 2*action_dim]
        """
        fpn_features = self.encoder(observation)
        return self.action_head(fpn_features, vector)

    def forward_hif(self, observation: torch.Tensor) -> torch.Tensor:
        """仅预测HIF - 预训练时调用

        单一职责：计算HIF预测，不涉及action
        返回类型一致：始终返回hif_pred

        Args:
            observation: 栅格观察 [B, C, H, W]

        Returns:
            hif_pred: HIF预测 [B, 2, 96, 96]
        """
        fpn_features = self.encoder(observation)

        if self.hif_decoder_type == 'two_stage':
            # Two-stage解码器使用P2和P3
            hif_pred_48 = self.hif_decoder(
                fpn_features['P2'],  # [B, 256, 48, 48]
                fpn_features['P3']   # [B, 256, 24, 24]
            )
        else:
            # Full UNet解码器
            hif_pred_48 = self.hif_decoder(fpn_features)

        # 上采样到96x96
        return F.interpolate(
            hif_pred_48,
            size=(96, 96),
            mode='bilinear',
            align_corners=False
        )

    def forward(
        self,
        observation: torch.Tensor,
        vector: Optional[torch.Tensor] = None,
        mode: str = 'action'
    ) -> torch.Tensor:
        """统一接口 - 保持兼容性

        为了兼容现有代码，提供统一forward接口
        但推荐直接调用forward_action或forward_hif

        Args:
            observation: 栅格观察 [B, C, H, W]
            vector: 向量观察 [B, vec_dim] (action模式必需)
            mode: 'action' 或 'hif'

        Returns:
            根据mode返回action_params或hif_pred
        """
        if mode == 'action':
            if vector is None:
                raise ValueError("vector is required for action mode")
            return self.forward_action(observation, vector)
        elif mode == 'hif':
            return self.forward_hif(observation)
        else:
            raise ValueError(f"Unknown mode: {mode}")


def create_dual_head_actor(cfg) -> ResNetFPNDualHeadActorOptimized:
    """工厂函数 - 创建优化版双头Actor

    直接从配置创建，无需复杂的映射逻辑
    """
    return ResNetFPNDualHeadActorOptimized(
        in_channels=cfg.model.in_channels,
        vec_dim=cfg.model.vec_dim,
        action_dim=cfg.model.action_dim,
        fpn_channels=cfg.model.fpn_channels,
        hidden_dim=cfg.model.hidden_dim,
        pretrained=cfg.model.pretrained,
        hif_decoder_type=cfg.model.hif_decoder_type,
        decoder_channels=cfg.model.decoder_channels,
    )