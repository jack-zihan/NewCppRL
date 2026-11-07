"""
训练端模型 / 批大小 对内存的影响诊断（不改训练代码）。

目的：
- 用一个结构接近 ResNet+FPN 的简化 CNN，在 CPU 上模拟 batch_size 变化时
  中间特征图占用的内存趋势；
- 证明“卷积网络激活 + 多头 (actor+critic+HIF) + 多副本 (num_collectors)”
  足以把 24×96×96×30 这样看似“轻量级”的任务推到几十 GB 级别；
- 为你在服务器上看到“128 -> 256 -> 55GB 甚至无法 512”的现象提供定量支撑。

说明：
- 这里用 CPU 的 RSS 作为 proxy，重点看「随 batch_size 的比例关系」，而不是绝对值；
- 由于当前环境没有 GPU，这个脚本不直接测显存，但 scaling 规律是一致的。
"""

import gc
import math
import os
from typing import List, Tuple

import torch
from torch import nn
from tensordict import TensorDict


def get_rss_mb() -> float:
    """读取当前进程 RSS（MB）。"""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = float(line.split()[1])
                    return kb / 1024.0
    except FileNotFoundError:
        return 0.0
    return 0.0


class ResNetFPNStyleBlock(nn.Module):
    """
    极简版的 ResNet+FPN 模块：
    - 3 个 stage，每个 stage 若干 conv + ReLU + downsample
    - 输出 3 个尺度的特征图，模拟 FPN 输入
    这里只用于制造具有代表性的激活规模。
    """

    def __init__(self, in_channels: int = 24):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # Stage 1: 64 -> 64, 96x96 -> 48x48
        self.stage1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Stage 2: 64 -> 128, 48x48 -> 24x24
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Stage 3: 128 -> 256, 24x24 -> 12x12
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        x = self.stem(x)       # [B,64,96,96]
        c3 = self.stage1(x)    # [B,64,48,48]
        c4 = self.stage2(c3)   # [B,128,24,24]
        c5 = self.stage3(c4)   # [B,256,12,12]
        return c3, c4, c5


class SimpleFPNHead(nn.Module):
    """
    极简 FPN + HIF decoder / actor head：
    - 把多尺度特征上采样到 96x96，concat 后做几层 conv，再输出 1 通道“方向场”
    """

    def __init__(self):
        super().__init__()
        # 简单起见，先统一调整到 96x96，再通过 1x1 conv 调整通道
        self.c5_proj = nn.Conv2d(256, 64, kernel_size=1)
        self.c4_proj = nn.Conv2d(128, 64, kernel_size=1)
        self.c3_proj = nn.Conv2d(64, 64, kernel_size=1)
        self.merge = nn.Sequential(
            nn.Conv2d(64 * 3, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, c3, c4, c5):
        # 统一插值到 96x96，并裁剪到相同通道数
        u5 = torch.nn.functional.interpolate(c5, size=(96, 96), mode="bilinear", align_corners=False)
        u4 = torch.nn.functional.interpolate(c4, size=(96, 96), mode="bilinear", align_corners=False)
        u3 = torch.nn.functional.interpolate(c3, size=(96, 96), mode="bilinear", align_corners=False)

        u5 = self.c5_proj(u5)  # [B,64,96,96]
        u4 = self.c4_proj(u4)  # [B,64,96,96]
        u3 = self.c3_proj(u3)  # [B,64,96,96]

        x = torch.cat([u3, u4, u5], dim=1)  # [B,192,96,96]
        out = self.merge(x)  # [B,1,96,96]
        return out


class SimpleResNetFPNModel(nn.Module):
    """
    模拟“ResNet-FPN + HIF decoder”的整体结构，用于估算激活内存随 batch_size 的变化。
    """

    def __init__(self, in_channels: int = 24):
        super().__init__()
        self.backbone = ResNetFPNStyleBlock(in_channels=in_channels)
        self.head = SimpleFPNHead()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        c3, c4, c5 = self.backbone(obs)
        out = self.head(c3, c4, c5)
        return out


def bytes_to_mb(n_bytes: int) -> float:
    return n_bytes / 1024.0 / 1024.0


def estimate_activation_bytes(model: nn.Module, batch_size: int) -> Tuple[int, List[Tuple[str, torch.Size]]]:
    """
    用 forward hooks 统计所有中间激活的 numel，估算一次前向的激活内存占用（bytes）。
    """
    activations: List[Tuple[str, torch.Size]] = []
    handles = []

    def hook(name):
        def _hook(_, __, output):
            if isinstance(output, torch.Tensor):
                activations.append((name, output.shape))
            elif isinstance(output, (list, tuple)):
                for idx, out in enumerate(output):
                    if isinstance(out, torch.Tensor):
                        activations.append((f"{name}[{idx}]", out.shape))
            return output

        return _hook

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 叶子模块
            handles.append(module.register_forward_hook(hook(name)))

    obs = torch.randn(batch_size, 24, 96, 96)
    _ = model(obs)

    total_bytes = 0
    for _, shape in activations:
        total_bytes += math.prod(shape) * 4  # float32

    for h in handles:
        h.remove()

    return total_bytes, activations


def run_memory_scaling_experiment():
    print("===== SimpleResNetFPNModel activation scaling experiment (CPU proxy) =====")
    model = SimpleResNetFPNModel(in_channels=24)

    batch_sizes = [64, 128, 256, 512]
    results = []
    for b in batch_sizes:
        total_bytes, _ = estimate_activation_bytes(model, b)
        mb = bytes_to_mb(total_bytes)
        results.append((b, mb))
        print(f"batch_size={b:4d} -> estimated activations: {mb:8.2f} MB")

    print("\n=== Summary (approximate linear scaling) ===")
    base_b, base_mb = results[0]
    for b, mb in results:
        factor = mb / base_mb if base_mb > 0 else float("nan")
        print(f"batch_size={b:4d}: {mb:8.2f} MB (~{factor:5.2f}x of batch_size={base_b})")

    print(
        "\n解释：即使是这个简化版 ResNet-FPN，每次前向的激活内存也会近似随 batch_size 线性增长。\n"
        "在真实训练中：\n"
        "  - 还有 critic 网络（至少一个 ResNet/FPN 分支）；\n"
        "  - HIF decoder 比这里的 SimpleFPNHead 更重；\n"
        "  - grad_accum_steps>1 时，同一 batch 在多个 micro-step 上复用部分激活；\n"
        "  - MultiaSyncDataCollector 会在多个进程复制一份 actor（policy_device=cuda:0）。\n"
        "综合下来，即使单个网络看起来“不大”，在 128->256->512 批量时，显存占用的绝对值也会非常可观。"
    )


def main():
    gc.collect()
    base = get_rss_mb()
    print(f"Initial RSS: {base:.2f} MB\n")
    run_memory_scaling_experiment()
    gc.collect()
    final = get_rss_mb()
    print(f"\nFinal RSS: {final:.2f} MB (delta={final - base:.2f} MB)")


if __name__ == "__main__":
    main()
