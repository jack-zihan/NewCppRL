"""
ResNet34-FPN Architecture for RL Coverage Tasks

This module implements a ResNet34 backbone with Feature Pyramid Network (FPN)
specifically designed for bird's-eye-view (BEV) coverage tasks in RL.

Key features:
- ResNet34 pretrained on ImageNet, adapted for multi-channel input
- Feature Pyramid Network for multi-scale feature fusion
- GroupNorm instead of BatchNorm for RL stability
- Shared encoder design for Actor and HIF Recon heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet34_Weights
from typing import Dict, List, Tuple, Optional


class ResNet34Backbone(nn.Module):
    """
    ResNet34 backbone adapted for multi-channel BEV input.

    Adapts ImageNet pretrained weights to handle 4-7 channel input
    through weight repetition and energy normalization.
    """

    def __init__(
        self,
        in_channels: int = 5,  # v4: 5ch, v5: 7ch, v6: 5ch
        pretrained: bool = True,
        freeze_bn: bool = False,
        replace_stride_with_dilation: Optional[List[bool]] = None
    ):
        super().__init__()

        # Load pretrained ResNet34 (new weights API)
        if pretrained:
            resnet = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet34(weights=None)

        # Adapt first conv layer for multi-channel input
        self.conv1 = self._adapt_first_conv(resnet.conv1, in_channels, pretrained)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # ResNet stages
        self.layer1 = resnet.layer1  # stride 4,  64 channels
        self.layer2 = resnet.layer2  # stride 8,  128 channels
        self.layer3 = resnet.layer3  # stride 16, 256 channels
        self.layer4 = resnet.layer4  # stride 32, 512 channels

        # Optionally freeze BatchNorm
        if freeze_bn:
            self._freeze_bn()

    def _adapt_first_conv(
        self,
        conv_layer: nn.Conv2d,
        in_channels: int,
        pretrained: bool
    ) -> nn.Conv2d:
        """Adapt pretrained 3-channel conv to multi-channel input."""
        # Create new conv layer
        new_conv = nn.Conv2d(
            in_channels, conv_layer.out_channels,
            kernel_size=conv_layer.kernel_size,
            stride=1,  # Changed from stride=2 to maintain 96×96
            padding=conv_layer.padding,
            bias=conv_layer.bias is not None
        )

        if pretrained and in_channels != 3:
            # Adapt pretrained weights
            with torch.no_grad():
                # Original weights: [64, 3, 7, 7]
                pretrained_weight = conv_layer.weight.data

                # Strategy: Repeat RGB channels cyclically
                repeats = (in_channels + 2) // 3
                repeated = pretrained_weight.repeat(1, repeats, 1, 1)  # [64, 3*repeats, 7, 7]
                adapted = repeated[:, :in_channels, :, :]  # [64, in_channels, 7, 7]

                # Energy normalization to maintain activation magnitude
                adapted *= (3.0 / in_channels) ** 0.5

                new_conv.weight.data = adapted

                if conv_layer.bias is not None:
                    new_conv.bias.data = conv_layer.bias.data.clone()
        elif pretrained:
            # If in_channels == 3, just copy weights
            new_conv.weight.data = conv_layer.weight.data.clone()
            if conv_layer.bias is not None:
                new_conv.bias.data = conv_layer.bias.data.clone()

        return new_conv

    def _freeze_bn(self):
        """Freeze BatchNorm layers."""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass returning intermediate features for FPN.

        Args:
            x: Input tensor [B, C, 96, 96]

        Returns:
            Tuple of (C2, C3, C4, C5) features at different scales
        """
        # Stem
        x = self.conv1(x)  # [B, 64, 96, 96] (no downsampling due to stride=1)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # [B, 64, 48, 48]

        # ResNet stages
        c2 = self.layer1(x)  # [B, 64, 48, 48] - Note: effectively stride 2 due to maxpool
        c3 = self.layer2(c2)  # [B, 128, 24, 24]
        c4 = self.layer3(c3)  # [B, 256, 12, 12]
        c5 = self.layer4(c4)  # [B, 512, 6, 6]

        return c2, c3, c4, c5


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature fusion.

    Creates a top-down pathway with lateral connections from ResNet features.
    Uses GroupNorm instead of BatchNorm for RL stability.
    """

    def __init__(
        self,
        in_channels_list: List[int] = [64, 128, 256, 512],
        out_channels: int = 256,
        use_groupnorm: bool = True,
        groups: int = 32
    ):
        super().__init__()

        # Lateral connections (1×1 conv to unify channels)
        self.lateral_c2 = nn.Conv2d(in_channels_list[0], out_channels, 1)
        self.lateral_c3 = nn.Conv2d(in_channels_list[1], out_channels, 1)
        self.lateral_c4 = nn.Conv2d(in_channels_list[2], out_channels, 1)
        self.lateral_c5 = nn.Conv2d(in_channels_list[3], out_channels, 1)

        # Top-down pathway (3×3 conv after fusion)
        self.fpn_c5 = self._make_fpn_block(out_channels, use_groupnorm, groups)
        self.fpn_c4 = self._make_fpn_block(out_channels, use_groupnorm, groups)
        self.fpn_c3 = self._make_fpn_block(out_channels, use_groupnorm, groups)
        self.fpn_c2 = self._make_fpn_block(out_channels, use_groupnorm, groups)

    def _make_fpn_block(self, channels: int, use_gn: bool, groups: int) -> nn.Module:
        """Create FPN smoothing block."""
        layers = [nn.Conv2d(channels, channels, 3, padding=1)]
        if use_gn:
            layers.append(nn.GroupNorm(min(groups, channels // 4), channels))
        layers.append(nn.SiLU(inplace=True))
        return nn.Sequential(*layers)

    def _upsample_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Upsample x and add to y."""
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(
        self,
        c2: torch.Tensor,
        c3: torch.Tensor,
        c4: torch.Tensor,
        c5: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass creating pyramid features.

        Args:
            c2, c3, c4, c5: ResNet features at different scales

        Returns:
            Tuple of (P2, P3, P4, P5) pyramid features, all with 256 channels
            P2: [B, 256, ~48, ~48] - Highest resolution (由于maxpool，约为输入一半)
            P3: [B, 256, ~24, ~24]
            P4: [B, 256, ~12, ~12]
            P5: [B, 256, ~6, ~6]   - Lowest resolution
        """
        # Lateral connections
        l2 = self.lateral_c2(c2)  # [B, 256, 48, 48]
        l3 = self.lateral_c3(c3)  # [B, 256, 24, 24]
        l4 = self.lateral_c4(c4)  # [B, 256, 12, 12]
        l5 = self.lateral_c5(c5)  # [B, 256, 6, 6]

        # Top-down pathway with lateral fusion
        p5 = self.fpn_c5(l5)  # [B, 256, 6, 6]
        p4 = self.fpn_c4(self._upsample_add(p5, l4))  # [B, 256, 12, 12]
        p3 = self.fpn_c3(self._upsample_add(p4, l3))  # [B, 256, 24, 24]
        p2 = self.fpn_c2(self._upsample_add(p3, l2))  # [B, 256, 48, 48]

        return p2, p3, p4, p5


class ResNetFPNEncoder(nn.Module):
    """
    Complete ResNet34-FPN encoder for RL tasks.

    Combines ResNet34 backbone with FPN for multi-scale feature extraction.
    Provides features at multiple scales for downstream tasks.
    """

    def __init__(
        self,
        in_channels: int = 5,
        fpn_channels: int = 256,
        pretrained: bool = True,
        freeze_bn: bool = False,
        use_groupnorm: bool = True
    ):
        super().__init__()

        self.backbone = ResNet34Backbone(in_channels, pretrained, freeze_bn)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[64, 128, 256, 512],
            out_channels=fpn_channels,
            use_groupnorm=use_groupnorm
        )

        self.fpn_channels = fpn_channels

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning FPN features.

        Args:
            x: Input tensor [B, C, 96, 96]

        Returns:
            Dictionary with keys 'P2', 'P3', 'P4', 'P5' containing pyramid features
        """
        # Backbone forward
        c2, c3, c4, c5 = self.backbone(x)

        # FPN forward
        p2, p3, p4, p5 = self.fpn(c2, c3, c4, c5)

        return {
            'P2': p2,  # [B, 256, 48, 48] - Note: effective resolution after maxpool
            'P3': p3,  # [B, 256, 24, 24]
            'P4': p4,  # [B, 256, 12, 12]
            'P5': p5,  # [B, 256, 6, 6]
        }

    def get_output_channels(self) -> int:
        """Return number of output channels per FPN level."""
        return self.fpn_channels
