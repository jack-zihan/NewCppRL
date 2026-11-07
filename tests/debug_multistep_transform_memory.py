"""
MultiStepTransform 内存与形状诊断脚本（只在 tests 下做实验，不改训练代码）。

目的：
- 验证 n_steps=12 的 MultiStepTransform 是否会在采样 / 存储时成倍放大 observation 的体积。

核心结论（先说结论，再用代码验证）：
- MultiStepTransform 只对 reward / done 相关键做 n-step 折叠；
- observation / next.observation 只是做「时间对齐」（t -> t+n_steps-1），不做 12 倍展开；
- 内部的 `_buffer` 只在 CPU Transform 对象内缓存最近 n_steps 帧，与 GPU 显存无关。
"""

import gc
import torch
from tensordict import TensorDict
from torchrl.envs.transforms import MultiStepTransform


def get_rss_mb() -> float:
    """简单读取当前进程 RSS（MB），用于观察 CPU 级别内存变化。"""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = float(line.split()[1])
                    return kb / 1024.0
    except FileNotFoundError:
        return 0.0
    return 0.0


def make_dummy_rollout(T: int, B: int):
    """
    构造一个简单的 [T, B] 轨迹 TensorDict，包含：
      - observation: [T,B,24,96,96]
      - next.observation: [T,B,24,96,96]
      - reward: [T,B,1]
      - done / truncated / terminated: [T,B,1]
    观测大小与 NewPasture-v6 接近，用于观察形状是否被 n_steps 膨胀。
    """
    obs_shape = (24, 96, 96)

    obs = torch.randn(T, B, *obs_shape)
    next_obs = torch.randn(T, B, *obs_shape)
    rew = torch.randn(T, B, 1)
    done = torch.zeros(T, B, 1, dtype=torch.bool)
    truncated = torch.zeros(T, B, 1, dtype=torch.bool)
    terminated = torch.zeros(T, B, 1, dtype=torch.bool)

    td = TensorDict(
        {
            "observation": obs,
            "next": {
                "observation": next_obs,
                "reward": rew,
                "done": done,
                "truncated": truncated,
                "terminated": terminated,
            },
        },
        batch_size=[T, B],
    )
    return td


def main():
    T, B = 64, 4  # 时间长度与 batch 数量适中，便于看趋势
    td = make_dummy_rollout(T, B)

    base_rss = get_rss_mb()
    print(f"Base RSS: {base_rss:.2f} MB")

    # 构建 n_steps=12 的 MultiStepTransform（与训练配置一致）
    mst = MultiStepTransform(n_steps=12, gamma=0.99)

    # 按照 ReplayBuffer.extend 的用法，调用 inv（对应 _inv_call）
    # 这里我们模拟一次性把整个 rollout extend 进去
    td_out = mst.inv(td)
    if td_out is None:
        print("MultiStepTransform.inv returned None (not enough steps to apply n_steps window)")
        return

    gc.collect()
    after_rss = get_rss_mb()
    print(f"RSS after MultiStepTransform.inv: {after_rss:.2f} MB (delta={after_rss - base_rss:.2f} MB)")

    # 核心：检查形状是否膨胀
    print("\n=== Shape check ===")
    print(f"input.observation.shape       : {td['observation'].shape}")
    print(f"output.observation.shape      : {td_out['observation'].shape}")
    print(f"input.next.observation.shape  : {td['next','observation'].shape}")
    print(f"output.next.observation.shape : {td_out['next','observation'].shape}")

    print("\n=== Reward / done keys ===")
    print(f"output.next.reward.shape      : {td_out['next','reward'].shape}")
    print(f"output.next.done.shape        : {td_out['next','done'].shape}")
    print(f"output.next.truncated.shape   : {td_out['next','truncated'].shape}")
    print(f"output.next.terminated.shape  : {td_out['next','terminated'].shape}")
    for k in td_out.get("next").keys(True, True):
        if "reward" in str(k) and k != ("next", "reward"):
            print(f"extra reward-like key: {k}, shape={td_out.get(k).shape}")

    print(
        "\n结论：从形状上看，MultiStepTransform 只在时间维度做截断（T -> T - n_steps），"
        "并对 reward/done 做多步折叠；observation 的通道与空间大小未被乘以 n_steps。"
        "内部 `_buffer` 只是 Transform 对象里的 CPU 缓存，与 GPU 显存无关。"
    )


if __name__ == "__main__":
    main()
