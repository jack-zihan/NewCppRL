"""
ResNet‑FPN dual‑head model for SAC + online HIF regularization.

Less‑is‑More principles applied:
- Single, self‑contained model module: encoder, action head, critic head,
  lightweight HIF decoder, and HIF loss in one place.
- Clear boundaries: training script only performs a short online HIF step
  and pops labels before replay; no extra glue abstractions.

Exposed classes
- ResNetFPNEncoder: ResNet34 backbone + FPN (imported from resnet_fpn).
- ResNetFPNActionHead: multi‑scale pooling + MLP to action params.
- ResNetFPNDualHeadActor: shared encoder, action head, optional HIF forward.
- ResNetFPNCritic: independent encoder + Q head.
- HIFReconModule: reuses the actor’s encoder to predict ego HIF.
- HIFReconstructionLoss: cosine similarity (+ optional TV) for HIF.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from tensordict import TensorDict

from .resnet_fpn import ResNetFPNEncoder


# ---------------------------------------------------------------------
# HIF decoders (inlined)
# ---------------------------------------------------------------------

class TwoStageHIFDecoder(nn.Module):
    """Lightweight two‑stage decoder using P2/P3.

    Inputs
      - p2: [B, C, H2, W2]  (highest resolution in FPN, here ~48×48)
      - p3: [B, C, H3, W3]
    Output
      - hif48: [B, 2, H2, W2]  (sin2θ, cos2θ) at P2 resolution
    """

    def __init__(self, fpn_channels: int = 256, decoder_channels: int = 128,
                 output_channels: int = 2, use_groupnorm: bool = True):
        super().__init__()
        gn = lambda c: nn.GroupNorm(min(32, max(1, c // 4)), c) if use_groupnorm else nn.Identity()

        self.p3_up = nn.ConvTranspose2d(fpn_channels, fpn_channels, 2, stride=2)
        self.fuse = nn.Sequential(
            nn.Conv2d(fpn_channels * 2, decoder_channels, 3, padding=1),
            gn(decoder_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1),
            gn(decoder_channels),
            nn.SiLU(inplace=True),
        )
        self.out = nn.Conv2d(decoder_channels, output_channels, 1)

    def forward(self, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
        p3_up = self.p3_up(p3)  # → size of p2
        if p3_up.shape[-2:] != p2.shape[-2:]:
            p3_up = F.interpolate(p3_up, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([p2, p3_up], dim=1)
        x = self.fuse(x)
        return self.out(x)


class FullUNetHIFDecoder(nn.Module):
    """Heavier UNet‑style decoder over P5→P2 pyramid.

    Accepts a dict with keys 'P2'..'P5' and returns [B,2,H2,W2].
    """

    def __init__(self, fpn_channels: int = 256, decoder_channels: int = 128,
                 output_channels: int = 2, use_groupnorm: bool = True):
        super().__init__()
        gn = lambda c: nn.GroupNorm(min(32, max(1, c // 4)), c) if use_groupnorm else nn.Identity()

        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), gn(co), nn.SiLU(True),
                nn.Conv2d(co, co, 3, padding=1), gn(co), nn.SiLU(True),
            )

        self.up54 = nn.ConvTranspose2d(fpn_channels, fpn_channels, 2, stride=2)
        self.up43 = nn.ConvTranspose2d(fpn_channels, fpn_channels, 2, stride=2)
        self.up32 = nn.ConvTranspose2d(fpn_channels, fpn_channels, 2, stride=2)

        self.b4 = block(fpn_channels * 2, decoder_channels)
        self.b3 = block(fpn_channels * 2, decoder_channels)
        self.b2 = block(fpn_channels * 2, decoder_channels)
        self.out = nn.Conv2d(decoder_channels, output_channels, 1)

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        p2, p3, p4, p5 = feats['P2'], feats['P3'], feats['P4'], feats['P5']
        x4 = self.b4(torch.cat([self.up54(p5), p4], dim=1))
        x3 = self.b3(torch.cat([self.up43(x4), p3], dim=1))
        x2 = self.b2(torch.cat([self.up32(x3), p2], dim=1))
        return self.out(x2)


class ResNetFPNActionHead(nn.Module):
    """
    Action prediction head for ResNet-FPN encoder.

    Takes multi-scale FPN features and vector observations to predict actions.
    """

    def __init__(
        self,
        fpn_channels: int = 256,
        vec_dim: int = 14,  # v4: 14, v6: 98
        action_dim: int = 2,  # (v, ω)
        hidden_dim: int = 512,
        pool_size: int = 6,  # Target size for adaptive pooling
        output_loc_scale: bool = True  # If True, output 2*action_dim for SAC
    ):
        super().__init__()

        self.pool_size = pool_size
        self.output_loc_scale = output_loc_scale

        # Calculate total feature dimension after pooling and concatenation
        # 4 FPN levels * fpn_channels * pool_size^2
        pooled_dim = 4 * fpn_channels * pool_size * pool_size

        # MLP for action prediction
        self.mlp = nn.Sequential(
            nn.Linear(pooled_dim + vec_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 2 * action_dim if output_loc_scale else action_dim)
        )

    def forward(
        self,
        fpn_features: Dict[str, torch.Tensor],
        vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for action prediction.

        Args:
            fpn_features: Dictionary with 'P2', 'P3', 'P4', 'P5' keys
            vector: Vector observations [B, vec_dim]

        Returns:
            Action parameters [B, 2*action_dim] or [B, action_dim]
        """
        # Pool each FPN level to fixed size
        p2_pooled = F.adaptive_avg_pool2d(fpn_features['P2'], self.pool_size)
        p3_pooled = F.adaptive_avg_pool2d(fpn_features['P3'], self.pool_size)
        p4_pooled = F.adaptive_avg_pool2d(fpn_features['P4'], self.pool_size)
        p5_pooled = F.adaptive_avg_pool2d(fpn_features['P5'], self.pool_size)

        # Flatten and concatenate
        features = torch.cat([
            p2_pooled.flatten(1),
            p3_pooled.flatten(1),
            p4_pooled.flatten(1),
            p5_pooled.flatten(1),
            vector
        ], dim=1)

        # MLP forward
        return self.mlp(features)


class ResNetFPNDualHeadActor(nn.Module):
    """
    Dual-head actor network with shared ResNet-FPN encoder.

    Provides both action prediction and HIF reconstruction capabilities.
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

        # Shared encoder
        self.encoder = ResNetFPNEncoder(
            in_channels=in_channels,
            fpn_channels=fpn_channels,
            pretrained=pretrained,
            use_groupnorm=True
        )

        # Action head
        self.action_head = ResNetFPNActionHead(
            fpn_channels=fpn_channels,
            vec_dim=vec_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            output_loc_scale=True
        )

        # HIF decoder head
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

    def forward(
        self,
        observation: torch.Tensor,
        vector: torch.Tensor,
        return_hif: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for dual-head network.

        Args:
            observation: Raster observation [B, C, H, W]
            vector: Vector observation [B, vec_dim]
            return_hif: Whether to compute and return HIF prediction

        Returns:
            action_params: Action parameters [B, 2*action_dim]
            hif_pred: HIF prediction [B, 2, H, W] if return_hif else None
        """
        # Shared encoding
        fpn_features = self.encoder(observation)

        # Action prediction (always computed)
        action_params = self.action_head(fpn_features, vector)

        # HIF prediction (optional)
        hif_pred = None
        if return_hif:
            if self.hif_decoder_type == 'two_stage':
                # Two-stage decoder uses P2 and P3
                # Note: P2 is 48×48, not 96×96 due to maxpool in ResNet
                # We need to upsample final result to 96×96
                hif_pred_48 = self.hif_decoder(
                    fpn_features['P2'],  # [B, 256, 48, 48]
                    fpn_features['P3']   # [B, 256, 24, 24]
                )
                hif_pred = F.interpolate(hif_pred_48, size=(96, 96), mode='bilinear', align_corners=False)
            else:
                # Full UNet decoder
                hif_pred_48 = self.hif_decoder(fpn_features)
                hif_pred = F.interpolate(hif_pred_48, size=(96, 96), mode='bilinear', align_corners=False)

        return action_params, hif_pred


class ResNetFPNCritic(nn.Module):
    """
    Critic network with independent ResNet-FPN encoder.

    Uses its own encoder to avoid destabilizing actor training.
    """

    def __init__(
        self,
        in_channels: int = 5,
        vec_dim: int = 14,
        action_dim: int = 2,
        fpn_channels: int = 256,
        hidden_dim: int = 512,
        pretrained: bool = True
    ):
        super().__init__()

        # Independent encoder for critic
        self.encoder = ResNetFPNEncoder(
            in_channels=in_channels,
            fpn_channels=fpn_channels,
            pretrained=pretrained,
            use_groupnorm=True
        )

        # Q-value head (similar to action head but includes action in input)
        pool_size = 6
        pooled_dim = 4 * fpn_channels * pool_size * pool_size

        self.q_head = nn.Sequential(
            nn.Linear(pooled_dim + vec_dim + action_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1)  # Single Q-value
        )

        self.pool_size = pool_size

    def forward(
        self,
        observation: torch.Tensor,
        vector: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for critic.

        Args:
            observation: Raster observation [B, C, H, W]
            vector: Vector observation [B, vec_dim]
            action: Action [B, action_dim]

        Returns:
            Q-value [B, 1]
        """
        # Encode observation
        fpn_features = self.encoder(observation)

        # Pool features
        pooled = []
        for key in ['P2', 'P3', 'P4', 'P5']:
            pooled.append(
                F.adaptive_avg_pool2d(fpn_features[key], self.pool_size).flatten(1)
            )

        # Concatenate all inputs
        features = torch.cat(pooled + [vector, action], dim=1)

        # Compute Q-value
        return self.q_head(features)


class HIFReconModule(nn.Module):
    """
    HIF reconstruction module for online auxiliary training.

    This module shares the encoder with the actor but is logically separate
    for clean training integration.
    """

    def __init__(
        self,
        shared_encoder: ResNetFPNEncoder,
        fpn_channels: int = 256,
        decoder_type: str = 'two_stage',
        decoder_channels: int = 128
    ):
        super().__init__()

        # Share encoder with actor (no duplication)
        self.encoder = shared_encoder

        # Create decoder
        if decoder_type == 'two_stage':
            self.decoder = TwoStageHIFDecoder(fpn_channels, decoder_channels, 2, True)
        else:
            self.decoder = FullUNetHIFDecoder(fpn_channels, decoder_channels, 2, True)

        self.decoder_type = decoder_type

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for HIF reconstruction.

        Args:
            observation: Raster observation [B, C, H, W]

        Returns:
            HIF prediction [B, 2, 96, 96]
        """
        # Encode with shared encoder
        fpn_features = self.encoder(observation)

        # Decode
        if self.decoder_type == 'two_stage':
            hif_pred_48 = self.decoder(fpn_features['P2'], fpn_features['P3'])
        else:
            hif_pred_48 = self.decoder(fpn_features)

        # Upsample to original resolution
        hif_pred = F.interpolate(hif_pred_48, size=(96, 96), mode='bilinear', align_corners=False)

        return hif_pred


# ---------------------------------------------------------------------
# HIF loss (inlined)
# ---------------------------------------------------------------------

class HIFReconstructionLoss(nn.Module):
    """Cosine similarity for axial field (sin2θ, cos2θ) + confidence weighting + optional TV.

    TorchRL风格接口：接收tensordict，内部提取需要的keys

    数学原理：
    - 轴向场特性：角度θ和θ+180°等价，双角编码(cos2θ, sin2θ)处理周期性
    - 余弦相似度：天然衡量方向相似性，不受幅度影响
    - 置信度加权：只在有效区域（confidence>0）计算损失，避免无效区域干扰
    - TV正则：鼓励空间平滑性，减少噪声

    Args:
        lambda_tv: TV正则化权重
        use_tv: 是否使用TV正则
        eps: 数值稳定性常数
    """

    def __init__(self, lambda_tv: float = 1e-5,
                 use_tv: bool = True, eps: float = 1e-6):
        super().__init__()
        self.lambda_tv = float(lambda_tv)
        self.use_tv = bool(use_tv)
        self.eps = float(eps)

    def forward(self, tensordict: TensorDict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        TorchRL风格：从tensordict提取数据并计算损失

        Args:
            tensordict: 包含以下keys的TensorDict
                - "pred_ego_hif": 预测HIF [B, 2, H, W]
                - ("label_ego_hif"): 标签HIF（3通道）[B, 3, H, W]
                    - channel 0: cos(2θ)
                    - channel 1: sin(2θ)
                    - channel 2: confidence

        Returns:
            total_loss: 总损失（余弦 + TV）
            metrics: 损失组成的详细信息dict
        """
        # 1. 从tensordict提取数据（TorchRL风格）
        pred_hif = tensordict["pred_ego_hif"]  # [B, 2, H, W]
        label_hif_3ch = tensordict[("label_ego_hif")]  # [B, 3, H, W]

        # 2. 解包三通道标签数据
        target_cos2 = label_hif_3ch[:, 0:1]  # [B, 1, H, W]
        target_sin2 = label_hif_3ch[:, 1:2]  # [B, 1, H, W]
        confidence = label_hif_3ch[:, 2]     # [B, H, W]

        # 3. 构建目标向量（去掉confidence的幅度影响，仅保留方向）
        target_hif = torch.cat([target_cos2, target_sin2], dim=1)  # [B, 2, H, W]

        # 归一化为单位向量（去除confidence权重，恢复纯方向）
        target_norm = target_hif.norm(dim=1, keepdim=True).clamp(min=self.eps)
        target_hif_unit = target_hif / target_norm

        # 预测向量也归一化
        pred_norm = pred_hif.norm(dim=1, keepdim=True).clamp(min=self.eps)
        pred_hif_unit = pred_hif / pred_norm

        # 4. 计算余弦相似度（逐像素点积）
        # 转置到[B, H, W, 2]便于计算
        p = pred_hif_unit.permute(0, 2, 3, 1)
        y = target_hif_unit.permute(0, 2, 3, 1)
        cos_sim = (p * y).sum(dim=-1)  # [B, H, W]

        # 5. 置信度加权的损失
        # (1 - cosine_similarity)范围[0, 2]，越小越好
        ang_loss = (1.0 - cos_sim) * confidence.clamp(0, 1)  # [B, H, W]

        # ⭐ 计算per-sample error（模仿SACLoss的td_error计算模式）
        valid_pixels_per_sample = confidence.sum(dim=[1, 2]) + self.eps  # [B]
        per_sample_error = ang_loss.sum(dim=[1, 2]) / valid_pixels_per_sample  # [B]

        # Batch mean用于反向传播
        cos_loss = per_sample_error.mean()  # 标量

        # 6. TV正则化（鼓励空间平滑性，可选）
        if self.use_tv and self.lambda_tv > 0:
            # 水平和垂直梯度的L2范数
            dh = pred_hif[:, :, :, 1:] - pred_hif[:, :, :, :-1]
            dv = pred_hif[:, :, 1:, :] - pred_hif[:, :, :-1, :]
            tv = dh.norm(dim=1).mean() + dv.norm(dim=1).mean()
        else:
            tv = pred_hif.new_tensor(0.0)

        # 7. 总损失
        total = cos_loss + self.lambda_tv * tv

        # ⭐ 返回metadata包含td_error（模仿SACLoss返回格式）
        return total, {
            'td_error': per_sample_error.detach(),  # [B] - 用于优先级采样
            'hif_cosine_loss': float(cos_loss.detach().cpu()),
            'hif_tv_loss': float(tv.detach().cpu()),
            'hif_total_loss': float(total.detach().cpu()),
        }
