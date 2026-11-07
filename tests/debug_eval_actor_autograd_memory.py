"""
小规模内存实验：验证“保存带梯度的前向输出”会导致内存随步数快速累积。

说明：
- 这里只在 CPU 上构造一个简化版的“小型 CNN + MLP actor”，输入形状与 NewPasture-v6 类似：
  observation: [B, 24, 96, 96]，vector: [B, 98]
- 分别在两种模式下重复 N 次前向，并把每一步的输出都存入列表：
    1）默认有梯度的模式（模拟当前 eval_env.rollout 调用 policy 的行为）
    2）torch.no_grad() 模式（理想的评估方式）
- 对比两种模式的进程 RSS 增量，直观展示“计算图是否被保留”对内存增长的影响。

注意：
- 这是一个独立脚本，不作为 pytest 用例自动运行；需要时手动执行：
    python tests/debug_eval_actor_autograd_memory.py
"""

import os
import gc
import torch
from torch import nn
from tensordict import TensorDict


def get_rss_mb() -> float:
    """读取当前进程的常驻内存（RSS，MB）。在 Linux 下通过 /proc/self/status。"""
    status_path = "/proc/self/status"
    if not os.path.exists(status_path):
        return 0.0
    with open(status_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    kb = float(parts[1])
                    return kb / 1024.0
    return 0.0


class SimpleActor(nn.Module):
    """
    简化版 actor：
    - CNN 处理 observation: [B, 24, 96, 96]
    - MLP 处理 vector: [B, 98]
    - 拼接后输出 2 维 action

    目的不是精确还原 ResNet-FPN，而是保留“有较深网络 + 大特征图 + autograd 图”的特征，
    用于验证“保存带梯度输出”会导致内存线性累积。
    """

    def __init__(self, obs_shape, vec_dim, action_dim=2):
        super().__init__()
        c, h, w = obs_shape
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 48x48
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 24x24
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 1x1
        )
        self.vec_mlp = nn.Sequential(
            nn.Linear(vec_dim, 128),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, td: TensorDict) -> TensorDict:
        obs = td["observation"]  # [B,C,H,W]
        vec = td["vector"]  # [B,D]
        feat_img = self.cnn(obs).flatten(1)  # [B,128]
        feat_vec = self.vec_mlp(vec)  # [B,128]
        feat = torch.cat([feat_img, feat_vec], dim=-1)  # [B,256]
        action = self.head(feat)  # [B,2]
        return TensorDict({"action": action}, batch_size=td.batch_size)


def make_random_batch(batch_size: int, obs_shape=(24, 96, 96), vec_dim: int = 98) -> TensorDict:
    """构造一批随机观测。"""
    obs = torch.randn(batch_size, *obs_shape, dtype=torch.float32)
    vec = torch.randn(batch_size, vec_dim, dtype=torch.float32)
    return TensorDict({"observation": obs, "vector": vec}, batch_size=[batch_size])


def run_forward_loop(actor, steps: int, batch_size: int, use_grad: bool) -> float:
    """
    连续 steps 次前向，并把每一步的输出都保存到列表里，返回整个过程的 RSS 内存增量（MB）。
    use_grad=True 时会构建计算图；use_grad=False 时在 torch.no_grad() 中执行。
    """
    gc.collect()
    base = get_rss_mb()

    outputs = []
    for i in range(steps):
        td = make_random_batch(batch_size)
        if use_grad:
            out = actor(td)
        else:
            with torch.no_grad():
                out = actor(td)
        # 模拟 rollout 把每一步结果都存起来
        outputs.append(out)
        gc.collect()
        cur = get_rss_mb()
        print(
            f"  step {i+1:02d} | use_grad={use_grad} | "
            f"delta={cur - base:.2f} MB (current RSS={cur:.2f} MB)"
        )

    final = get_rss_mb()
    return final - base


def main():
    # 控制规模：CPU 上 batch=2, steps=10 足够看出趋势，又不会占太多内存
    batch_size = 2
    steps = 10

    obs_shape = (24, 96, 96)
    vec_dim = 98
    actor = SimpleActor(obs_shape=obs_shape, vec_dim=vec_dim)

    print("===== SimpleActor forward memory experiment (CPU) =====")
    print(f"Batch size = {batch_size}, steps = {steps}")
    print(f"Actor class: {type(actor).__name__}")
    print()

    print(">>> Case 1: 正常有梯度的前向（模拟当前 eval_env.rollout 行为）")
    delta_with_grad = run_forward_loop(actor, steps=steps, batch_size=batch_size, use_grad=True)
    print(f"[Summary] use_grad=True total RSS delta: {delta_with_grad:.2f} MB\n")

    print(">>> Case 2: torch.no_grad() 前向（理想的评估行为）")
    delta_no_grad = run_forward_loop(actor, steps=steps, batch_size=batch_size, use_grad=False)
    print(f"[Summary] use_grad=False total RSS delta: {delta_no_grad:.2f} MB\n")

    print("===== Comparison =====")
    print(f"Total RSS delta with grad   : {delta_with_grad:.2f} MB")
    print(f"Total RSS delta without grad: {delta_no_grad:.2f} MB")
    if delta_no_grad > 0:
        ratio = delta_with_grad / delta_no_grad
        print(f"Ratio (with_grad / no_grad): {ratio:.2f}x")
    else:
        print("Ratio: N/A (no_grad case delta ≈ 0)")

    print(
        "\n结论：如果在评估时像 Case 1 那样，每一步前向都保留带梯度的输出并长期保存，"
        "内存会随步数线性甚至超线性增长；理想的评估应该像 Case 2 一样在 no_grad 下执行，"
        "只保存数值结果，而不保留整条计算图。"
    )


if __name__ == "__main__":
    main()

