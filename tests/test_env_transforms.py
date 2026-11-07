"""
测试环境Transform系统

验证关键Transform组件的功能：
- Steps95ToDoneCounter: 95%→done步数统计（矢量化）
- KeepLastPixels: 黑帧缓存机制
- DropPixels: 内存优化
"""
import pytest
import torch
import numpy as np
from tensordict import TensorDict


class TestSteps95ToDoneCounter:
    """测试Steps95ToDoneCounter Transform（矢量化版本）"""

    def test_counter_initialization(self):
        """计数器初始化

        验证点:
        - _armed和_count初始为None
        - 首次调用时自动初始化为per-env状态
        """
        from rl_new.sac_cont_sy.env_utils import Steps95ToDoneCounter

        counter = Steps95ToDoneCounter(threshold=0.95)

        # 初始状态
        assert counter._armed is None
        assert counter._count is None

        # 首次调用
        B = 4  # 4个并行环境
        td = TensorDict({
            "completion_ratio": torch.tensor([0.5, 0.6, 0.7, 0.8]).reshape(B, 1)
        }, batch_size=[B])

        counter._call(td)

        # 验证per-env状态初始化
        assert counter._armed is not None
        assert counter._armed.shape == torch.Size([B])
        assert counter._count is not None
        assert counter._count.shape == torch.Size([B])

    def test_counter_arming_at_threshold(self):
        """达到阈值时arming

        验证点:
        - completion >= 0.95时_armed变为True
        - completion < 0.95时_armed保持False
        """
        from rl_new.sac_cont_sy.env_utils import Steps95ToDoneCounter

        counter = Steps95ToDoneCounter(threshold=0.95)

        B = 4
        # env0/1/2/3: 0.90, 0.95, 0.97, 1.00
        td = TensorDict({
            "completion_ratio": torch.tensor([0.90, 0.95, 0.97, 1.00]).reshape(B, 1)
        }, batch_size=[B])

        counter._call(td)

        # env0未达标，env1/2/3达标
        expected_armed = torch.tensor([False, True, True, True])
        assert torch.equal(counter._armed, expected_armed)

    def test_counter_increments_when_armed(self):
        """armed后每步递增

        验证点:
        - armed=True时，count每步+1
        - armed=False时，count保持0
        """
        from rl_new.sac_cont_sy.env_utils import Steps95ToDoneCounter

        counter = Steps95ToDoneCounter(threshold=0.95)

        B = 2
        # env0: 未达标，env1: 已达标
        td1 = TensorDict({
            "completion_ratio": torch.tensor([0.80, 0.96]).reshape(B, 1)
        }, batch_size=[B])

        counter._call(td1)

        # env1 armed，count=1
        assert torch.equal(counter._count, torch.tensor([0, 1], dtype=torch.int64))

        # 再次调用（env0仍未达标，env1保持armed）
        td2 = TensorDict({
            "completion_ratio": torch.tensor([0.85, 0.97]).reshape(B, 1)
        }, batch_size=[B])

        counter._call(td2)

        # env0仍为0，env1递增到2
        assert torch.equal(counter._count, torch.tensor([0, 2], dtype=torch.int64))

    def test_counter_arms_once_reached(self):
        """一旦达到阈值就永久armed

        验证点:
        - completion从<0.95 → >=0.95 → <0.95，armed保持True
        - count继续递增
        """
        from rl_new.sac_cont_sy.env_utils import Steps95ToDoneCounter

        counter = Steps95ToDoneCounter(threshold=0.95)

        B = 1
        # Step 1: 0.90（未armed）
        td1 = TensorDict({"completion_ratio": torch.tensor([[0.90]])}, batch_size=[B])
        counter._call(td1)
        assert counter._count[0] == 0

        # Step 2: 0.96（armed）
        td2 = TensorDict({"completion_ratio": torch.tensor([[0.96]])}, batch_size=[B])
        counter._call(td2)
        assert counter._count[0] == 1

        # Step 3: 0.94（下降但armed保持）
        td3 = TensorDict({"completion_ratio": torch.tensor([[0.94]])}, batch_size=[B])
        counter._call(td3)
        assert counter._count[0] == 2  # 继续递增

    def test_counter_writes_to_tensordict(self):
        """写入TensorDict

        验证点:
        - _call后td["steps_95_to_done"]包含count
        - 形状与completion_ratio一致
        """
        from rl_new.sac_cont_sy.env_utils import Steps95ToDoneCounter

        counter = Steps95ToDoneCounter(threshold=0.95)

        B = 3
        td = TensorDict({
            "completion_ratio": torch.tensor([0.90, 0.96, 0.98]).reshape(B, 1)
        }, batch_size=[B])

        result = counter._call(td)

        # 验证写入
        assert "steps_95_to_done" in result.keys()
        assert result["steps_95_to_done"].shape == torch.Size([B, 1])

        # env0=0, env1/2=1
        expected = torch.tensor([[0], [1], [1]], dtype=torch.int64)
        assert torch.equal(result["steps_95_to_done"], expected)

    def test_counter_reset_clears_state(self):
        """Reset清零状态

        验证点:
        - _reset后_armed和_count归零
        - tensordict_reset包含初始值（全0）
        """
        from rl_new.sac_cont_sy.env_utils import Steps95ToDoneCounter

        counter = Steps95ToDoneCounter(threshold=0.95)

        B = 2
        # 先调用一次
        td1 = TensorDict({"completion_ratio": torch.tensor([[0.96], [0.97]])}, batch_size=[B])
        counter._call(td1)
        assert counter._count.sum() > 0  # 有计数

        # Reset
        td_in = TensorDict({}, batch_size=[B])
        td_reset = TensorDict({}, batch_size=[B])
        result = counter._reset(td_in, td_reset)

        # 验证状态清零
        assert torch.equal(counter._armed, torch.zeros(B, dtype=torch.bool))
        assert torch.equal(counter._count, torch.zeros(B, dtype=torch.int64))

        # 验证reset返回值
        assert "steps_95_to_done" in result.keys()
        assert torch.equal(result["steps_95_to_done"], torch.zeros(B, dtype=torch.int64))


class TestKeepLastPixels:
    """测试KeepLastPixels Transform"""

    def test_keep_last_pixels_initialization(self):
        """KeepLastPixels初始化

        验证点:
        - _last初始为None
        - 首次调用时自动初始化
        """
        from rl_new.sac_cont_sy.env_utils import KeepLastPixels

        transform = KeepLastPixels()

        assert transform._last is None

        # 首次调用
        B, H, W, C = 2, 64, 64, 3
        td = TensorDict({
            "pixels": torch.randint(0, 255, (B, H, W, C), dtype=torch.uint8)
        }, batch_size=[B])

        transform._call(td)

        # 验证初始化
        assert transform._last is not None
        assert transform._last.shape == torch.Size([B, H, W, C])

    def test_keep_last_pixels_caches_valid_frames(self):
        """缓存有效帧

        验证点:
        - 非零帧被缓存到_last
        """
        from rl_new.sac_cont_sy.env_utils import KeepLastPixels

        transform = KeepLastPixels()

        B, H, W, C = 2, 4, 4, 3
        # 两个非零帧
        valid_pixels = torch.randint(1, 255, (B, H, W, C), dtype=torch.uint8)
        td = TensorDict({"pixels": valid_pixels.clone()}, batch_size=[B])

        transform._call(td)

        # 验证缓存
        assert torch.equal(transform._last, valid_pixels)

    def test_keep_last_pixels_replaces_black_frames(self):
        """黑帧替换

        验证点:
        - 全零帧被替换为上一帧
        - 有效帧保持不变
        """
        from rl_new.sac_cont_sy.env_utils import KeepLastPixels

        transform = KeepLastPixels()

        B, H, W, C = 2, 4, 4, 3

        # Step 1: 两个有效帧
        valid_pixels = torch.randint(1, 255, (B, H, W, C), dtype=torch.uint8)
        td1 = TensorDict({"pixels": valid_pixels.clone()}, batch_size=[B])
        transform._call(td1)

        # Step 2: env0黑帧，env1有效帧
        pixels_step2 = torch.randint(1, 255, (B, H, W, C), dtype=torch.uint8)
        pixels_step2[0] = 0  # env0黑帧
        td2 = TensorDict({"pixels": pixels_step2.clone()}, batch_size=[B])

        result = transform._call(td2)

        # 验证env0被替换为上一帧
        assert torch.equal(result["pixels"][0], valid_pixels[0])
        # env1保持新帧
        assert torch.equal(result["pixels"][1], pixels_step2[1])

    def test_keep_last_pixels_handles_done_flag(self):
        """done标志触发替换

        验证点:
        - done=True时也使用上一帧（避免done后的黑帧）
        """
        from rl_new.sac_cont_sy.env_utils import KeepLastPixels

        transform = KeepLastPixels()

        B, H, W, C = 2, 4, 4, 3

        # Step 1: 有效帧
        valid_pixels = torch.randint(1, 255, (B, H, W, C), dtype=torch.uint8)
        td1 = TensorDict({"pixels": valid_pixels.clone(), "done": torch.tensor([False, False])}, batch_size=[B])
        transform._call(td1)

        # Step 2: env0 done（即使pixels非零也替换）
        pixels_step2 = torch.randint(1, 255, (B, H, W, C), dtype=torch.uint8)
        td2 = TensorDict({
            "pixels": pixels_step2.clone(),
            "done": torch.tensor([[True], [False]])  # env0 done
        }, batch_size=[B])

        result = transform._call(td2)

        # env0应被替换（因为done）
        assert torch.equal(result["pixels"][0], valid_pixels[0])

    def test_keep_last_pixels_reset_updates_cache(self):
        """Reset更新缓存

        验证点:
        - reset时用新帧更新_last
        """
        from rl_new.sac_cont_sy.env_utils import KeepLastPixels

        transform = KeepLastPixels()

        B, H, W, C = 2, 4, 4, 3

        # 初始帧
        old_pixels = torch.randint(1, 50, (B, H, W, C), dtype=torch.uint8)
        td_old = TensorDict({"pixels": old_pixels.clone()}, batch_size=[B])
        transform._call(td_old)

        # Reset with new pixels
        new_pixels = torch.randint(100, 255, (B, H, W, C), dtype=torch.uint8)
        td_reset = TensorDict({"pixels": new_pixels.clone()}, batch_size=[B])

        transform._reset(TensorDict({}, batch_size=[B]), td_reset)

        # 验证缓存更新
        assert torch.equal(transform._last, new_pixels)


class TestDropPixels:
    """测试DropPixels Transform"""

    def test_drop_pixels_removes_current_pixels(self):
        """删除当前步pixels

        验证点:
        - td["pixels"]被删除
        """
        from rl_new.sac_cont_sy.env_utils import DropPixels

        transform = DropPixels()

        B = 2
        td = TensorDict({
            "pixels": torch.randn(B, 64, 64, 3),
            "observation": torch.randn(B, 3, 64, 64)
        }, batch_size=[B])

        result = transform._call(td)

        # pixels应被删除
        assert "pixels" not in result.keys()
        # observation保留
        assert "observation" in result.keys()

    def test_drop_pixels_removes_next_pixels(self):
        """删除next.pixels

        验证点:
        - td["next"]["pixels"]被删除
        - 其他next键保留
        """
        from rl_new.sac_cont_sy.env_utils import DropPixels

        transform = DropPixels()

        B = 2
        td = TensorDict({
            "observation": torch.randn(B, 3, 64, 64),
            "next": TensorDict({
                "pixels": torch.randn(B, 64, 64, 3),
                "reward": torch.randn(B, 1),
                "done": torch.tensor([[False], [True]])
            }, batch_size=[B])
        }, batch_size=[B])

        result = transform._call(td)

        # next.pixels应被删除
        assert "pixels" not in result["next"].keys()
        # next其他键保留
        assert "reward" in result["next"].keys()
        assert "done" in result["next"].keys()

    def test_drop_pixels_handles_missing_pixels(self):
        """Pixels不存在时不报错

        验证点:
        - td中无pixels时正常执行
        """
        from rl_new.sac_cont_sy.env_utils import DropPixels

        transform = DropPixels()

        B = 2
        td = TensorDict({
            "observation": torch.randn(B, 3, 64, 64),
            "action": torch.randn(B, 2)
        }, batch_size=[B])

        # 不应抛出异常
        result = transform._call(td)
        assert "observation" in result.keys()

    def test_drop_pixels_in_reset(self):
        """Reset时删除pixels

        验证点:
        - tensordict_reset中的pixels被删除
        """
        from rl_new.sac_cont_sy.env_utils import DropPixels

        transform = DropPixels()

        B = 2
        td_reset = TensorDict({
            "pixels": torch.randn(B, 64, 64, 3),
            "observation": torch.randn(B, 3, 64, 64)
        }, batch_size=[B])

        result = transform._reset(TensorDict({}, batch_size=[B]), td_reset)

        # pixels应被删除
        assert "pixels" not in result.keys()
        # observation保留
        assert "observation" in result.keys()


class TestTransformEdgeCases:
    """测试边界情况"""

    def test_steps_95_counter_with_varying_batch_size(self):
        """不同batch_size时的动态调整"""
        from rl_new.sac_cont_sy.env_utils import Steps95ToDoneCounter

        counter = Steps95ToDoneCounter()

        # 首次B=2
        td1 = TensorDict({"completion_ratio": torch.tensor([[0.96], [0.97]])}, batch_size=[2])
        counter._call(td1)
        assert counter._count.shape[0] == 2

        # 后续B=4（重新初始化）
        td2 = TensorDict({"completion_ratio": torch.tensor([[0.90], [0.95], [0.96], [0.98]])}, batch_size=[4])
        counter._call(td2)
        assert counter._count.shape[0] == 4

    def test_keep_last_pixels_with_float_dtype(self):
        """Float类型的pixels

        验证点:
        - 支持float32类型的pixels
        - 零检测使用abs().sum()
        """
        from rl_new.sac_cont_sy.env_utils import KeepLastPixels

        transform = KeepLastPixels()

        B, H, W, C = 2, 4, 4, 3
        # Float pixels
        valid_pixels = torch.rand(B, H, W, C, dtype=torch.float32)
        td = TensorDict({"pixels": valid_pixels.clone()}, batch_size=[B])

        transform._call(td)

        # 验证缓存
        assert transform._last.dtype == torch.float32
        assert torch.equal(transform._last, valid_pixels)
