"""
测试分桶经验回放系统

验证BucketedTensorDictPrioritizedReplayBuffer的核心功能：
- 三桶分类逻辑（SUCCESS/NEAR_END/MID）
- 动态采样比例
- Fallback机制
- 优先级更新路由
- 清桶功能
"""
import pytest
import torch
import tempfile
from tensordict import TensorDict
from torchrl.data import LazyMemmapStorage
from rl_new.sac_cont_sy.bucketed_replay import (
    BucketedTensorDictPrioritizedReplayBuffer,
    BucketId,
)


class TestBucketedReplayClassification:
    """测试三桶分类逻辑"""

    @pytest.fixture
    def buffer(self):
        """创建测试用的分桶buffer"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LazyMemmapStorage(max_size=100, scratch_dir=tmpdir)
            buffer = BucketedTensorDictPrioritizedReplayBuffer(
                storage=storage,
                alpha=0.6,
                beta=0.4,
                batch_size=2,
                prefetch=0,
                pin_memory=False,
                success_threshold=0.99,
                near_end_threshold=0.90,
            )
            yield buffer

    def test_classification_success(self, buffer):
        """SUCCESS分类: done=True, completion>=0.99, truncated=False

        验证点:
        - 满足条件 → SUCCESS bucket
        - bucket_sizes['success'] 增加
        """
        # 创建SUCCESS样本
        td = TensorDict({
            'observation': torch.randn(1, 4, 64, 64),
            'action': torch.randn(1, 2),
            'next': TensorDict({
                'observation': torch.randn(1, 4, 64, 64),
                'done': torch.tensor([[True]]),
                'truncated': torch.tensor([[False]]),
                'completion_ratio': torch.tensor([[0.99]]),
                'reward': torch.tensor([[10.0]]),
            }, batch_size=[1]),
        }, batch_size=[1])

        buffer.extend(td)

        # 验证分类
        sizes = buffer.bucket_sizes
        assert sizes['success'] == 1
        assert sizes['near_end'] == 0
        assert sizes['mid'] == 0

    def test_classification_near_end(self, buffer):
        """NEAR_END分类: completion>=0.90, not SUCCESS

        验证点:
        - completion>=0.90但not done → NEAR_END
        - bucket_sizes['near_end'] 增加
        """
        # 创建NEAR_END样本
        td = TensorDict({
            'observation': torch.randn(1, 4, 64, 64),
            'action': torch.randn(1, 2),
            'next': TensorDict({
                'observation': torch.randn(1, 4, 64, 64),
                'done': torch.tensor([[False]]),
                'truncated': torch.tensor([[False]]),
                'completion_ratio': torch.tensor([[0.95]]),
                'reward': torch.tensor([[5.0]]),
            }, batch_size=[1]),
        }, batch_size=[1])

        buffer.extend(td)

        sizes = buffer.bucket_sizes
        assert sizes['success'] == 0
        assert sizes['near_end'] == 1
        assert sizes['mid'] == 0

    def test_classification_mid(self, buffer):
        """MID分类: 其他情况

        验证点:
        - completion<0.90 → MID
        - bucket_sizes['mid'] 增加
        """
        # 创建MID样本
        td = TensorDict({
            'observation': torch.randn(1, 4, 64, 64),
            'action': torch.randn(1, 2),
            'next': TensorDict({
                'observation': torch.randn(1, 4, 64, 64),
                'done': torch.tensor([[False]]),
                'truncated': torch.tensor([[False]]),
                'completion_ratio': torch.tensor([[0.50]]),
                'reward': torch.tensor([[1.0]]),
            }, batch_size=[1]),
        }, batch_size=[1])

        buffer.extend(td)

        sizes = buffer.bucket_sizes
        assert sizes['success'] == 0
        assert sizes['near_end'] == 0
        assert sizes['mid'] == 1

    def test_classification_boundary_cases(self, buffer):
        """边界情况分类"""
        # Case 1: completion=0.99, done=False → NEAR_END
        td1 = TensorDict({
            'observation': torch.randn(1, 4, 64, 64),
            'action': torch.randn(1, 2),
            'next': TensorDict({
                'observation': torch.randn(1, 4, 64, 64),
                'done': torch.tensor([[False]]),
                'truncated': torch.tensor([[False]]),
                'completion_ratio': torch.tensor([[0.99]]),
                'reward': torch.tensor([[5.0]]),
            }, batch_size=[1]),
        }, batch_size=[1])

        buffer.extend(td1)
        sizes = buffer.bucket_sizes
        assert sizes['near_end'] == 1  # NEAR_END而非SUCCESS

        # Case 2: completion=0.99, done=True, truncated=True → NEAR_END
        td2 = TensorDict({
            'observation': torch.randn(1, 4, 64, 64),
            'action': torch.randn(1, 2),
            'next': TensorDict({
                'observation': torch.randn(1, 4, 64, 64),
                'done': torch.tensor([[True]]),
                'truncated': torch.tensor([[True]]),  # 超时截断
                'completion_ratio': torch.tensor([[0.99]]),
                'reward': torch.tensor([[5.0]]),
            }, batch_size=[1]),
        }, batch_size=[1])

        buffer.extend(td2)
        sizes = buffer.bucket_sizes
        assert sizes['near_end'] == 2  # NEAR_END而非SUCCESS

        # Case 3: completion=0.89 → MID
        td3 = TensorDict({
            'observation': torch.randn(1, 4, 64, 64),
            'action': torch.randn(1, 2),
            'next': TensorDict({
                'observation': torch.randn(1, 4, 64, 64),
                'done': torch.tensor([[False]]),
                'truncated': torch.tensor([[False]]),
                'completion_ratio': torch.tensor([[0.89]]),
                'reward': torch.tensor([[1.0]]),
            }, batch_size=[1]),
        }, batch_size=[1])

        buffer.extend(td3)
        sizes = buffer.bucket_sizes
        assert sizes['mid'] == 1  # MID而非NEAR_END

    def test_batch_classification(self, buffer):
        """批量分类（并行环境）"""
        # 创建包含所有三种类型的batch
        td = TensorDict({
            'observation': torch.randn(3, 4, 64, 64),
            'action': torch.randn(3, 2),
            'next': TensorDict({
                'observation': torch.randn(3, 4, 64, 64),
                'done': torch.tensor([[True], [False], [False]]),
                'truncated': torch.tensor([[False], [False], [False]]),
                'completion_ratio': torch.tensor([[0.99], [0.95], [0.50]]),
                'reward': torch.tensor([[10.0], [5.0], [1.0]]),
            }, batch_size=[3]),
        }, batch_size=[3])

        buffer.extend(td)

        # 验证分类
        sizes = buffer.bucket_sizes
        assert sizes['success'] == 1
        assert sizes['near_end'] == 1
        assert sizes['mid'] == 1


class TestBucketedReplaySampling:
    """测试采样逻辑和fallback机制"""

    @pytest.fixture
    def buffer_with_data(self):
        """创建包含数据的buffer"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LazyMemmapStorage(max_size=100, scratch_dir=tmpdir)
            buffer = BucketedTensorDictPrioritizedReplayBuffer(
                storage=storage,
                alpha=0.6,
                beta=0.4,
                batch_size=10,
                prefetch=0,
                pin_memory=False,
                success_threshold=0.99,
                near_end_threshold=0.90,
            )

            # 添加各类型数据
            # 5个SUCCESS
            for _ in range(5):
                td = TensorDict({
                    'observation': torch.randn(1, 4, 64, 64),
                    'action': torch.randn(1, 2),
                    'next': TensorDict({
                        'observation': torch.randn(1, 4, 64, 64),
                        'done': torch.tensor([[True]]),
                        'truncated': torch.tensor([[False]]),
                        'completion_ratio': torch.tensor([[0.99]]),
                        'reward': torch.tensor([[10.0]]),
                    }, batch_size=[1]),
                }, batch_size=[1])
                buffer.extend(td)

            # 5个NEAR_END
            for _ in range(5):
                td = TensorDict({
                    'observation': torch.randn(1, 4, 64, 64),
                    'action': torch.randn(1, 2),
                    'next': TensorDict({
                        'observation': torch.randn(1, 4, 64, 64),
                        'done': torch.tensor([[False]]),
                        'truncated': torch.tensor([[False]]),
                        'completion_ratio': torch.tensor([[0.95]]),
                        'reward': torch.tensor([[5.0]]),
                    }, batch_size=[1]),
                }, batch_size=[1])
                buffer.extend(td)

            # 10个MID
            for _ in range(10):
                td = TensorDict({
                    'observation': torch.randn(1, 4, 64, 64),
                    'action': torch.randn(1, 2),
                    'next': TensorDict({
                        'observation': torch.randn(1, 4, 64, 64),
                        'done': torch.tensor([[False]]),
                        'truncated': torch.tensor([[False]]),
                        'completion_ratio': torch.tensor([[0.50]]),
                        'reward': torch.tensor([[1.0]]),
                    }, batch_size=[1]),
                }, batch_size=[1])
                buffer.extend(td)

            yield buffer

    def test_sampling_with_ratio(self, buffer_with_data):
        """按比例采样

        验证点:
        - batch_size=10, ratio=(0.3,0.3,0.4) → n=(3,3,4)
        - 返回的tensordict包含bucket_id字段
        """
        buffer = buffer_with_data
        buffer.set_sampling_ratio((0.3, 0.3, 0.4))

        sampled = buffer.sample()

        # 验证batch_size
        assert sampled.batch_size[0] == 10

        # 验证bucket_id存在
        assert 'bucket_id' in sampled.keys()

        # 统计各bucket数量
        bucket_ids = sampled['bucket_id'].squeeze()
        success_count = (bucket_ids == BucketId.SUCCESS.value).sum().item()
        near_end_count = (bucket_ids == BucketId.NEAR_END.value).sum().item()
        mid_count = (bucket_ids == BucketId.MID.value).sum().item()

        # 应该接近3,3,4（有随机性）
        assert success_count >= 2 and success_count <= 4
        assert near_end_count >= 2 and near_end_count <= 4
        assert mid_count >= 3 and mid_count <= 5
        assert success_count + near_end_count + mid_count == 10

    def test_fallback_to_mid_when_success_empty(self):
        """SUCCESS bucket空时fallback到MID"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LazyMemmapStorage(max_size=100, scratch_dir=tmpdir)
            buffer = BucketedTensorDictPrioritizedReplayBuffer(
                storage=storage,
                alpha=0.6,
                beta=0.4,
                batch_size=6,
                prefetch=0,
                pin_memory=False,
                success_threshold=0.99,
                near_end_threshold=0.90,
            )

            # 只添加MID数据
            for _ in range(10):
                td = TensorDict({
                    'observation': torch.randn(1, 4, 64, 64),
                    'action': torch.randn(1, 2),
                    'next': TensorDict({
                        'observation': torch.randn(1, 4, 64, 64),
                        'done': torch.tensor([[False]]),
                        'truncated': torch.tensor([[False]]),
                        'completion_ratio': torch.tensor([[0.50]]),
                        'reward': torch.tensor([[1.0]]),
                    }, batch_size=[1]),
                }, batch_size=[1])
                buffer.extend(td)

            # 设置采样比例（要求采样SUCCESS，但SUCCESS为空）
            buffer.set_sampling_ratio((0.5, 0.0, 0.5))  # 3 from SUCCESS, 3 from MID

            # 采样应该成功（fallback到MID）
            sampled = buffer.sample()
            assert sampled.batch_size[0] == 6

            # 所有样本应该来自MID（因为SUCCESS为空）
            bucket_ids = sampled['bucket_id'].squeeze()
            mid_count = (bucket_ids == BucketId.MID.value).sum().item()
            assert mid_count == 6  # 全部fallback到MID

    def test_mid_bucket_failure_fatal(self):
        """MID bucket空且被请求采样 → 抛RuntimeError"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LazyMemmapStorage(max_size=100, scratch_dir=tmpdir)
            buffer = BucketedTensorDictPrioritizedReplayBuffer(
                storage=storage,
                alpha=0.6,
                beta=0.4,
                batch_size=2,
                prefetch=0,
                pin_memory=False,
                success_threshold=0.99,
                near_end_threshold=0.90,
            )

            # 只添加SUCCESS数据（MID为空）
            td = TensorDict({
                'observation': torch.randn(1, 4, 64, 64),
                'action': torch.randn(1, 2),
                'next': TensorDict({
                    'observation': torch.randn(1, 4, 64, 64),
                    'done': torch.tensor([[True]]),
                    'truncated': torch.tensor([[False]]),
                    'completion_ratio': torch.tensor([[0.99]]),
                    'reward': torch.tensor([[10.0]]),
                }, batch_size=[1]),
            }, batch_size=[1])
            buffer.extend(td)

            # 设置采样比例（要求从空的MID采样）
            buffer.set_sampling_ratio((0.0, 0.0, 1.0))  # 全部从MID

            # 应该抛出异常（MID为空）
            with pytest.raises((RuntimeError, ValueError)):
                buffer.sample()


class TestBucketedReplayPriorityUpdate:
    """测试优先级更新路由"""

    @pytest.fixture
    def buffer_with_samples(self):
        """创建buffer并采样"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LazyMemmapStorage(max_size=100, scratch_dir=tmpdir)
            buffer = BucketedTensorDictPrioritizedReplayBuffer(
                storage=storage,
                alpha=0.6,
                beta=0.4,
                batch_size=6,
                prefetch=0,
                pin_memory=False,
                success_threshold=0.99,
                near_end_threshold=0.90,
            )

            # 添加数据
            for completion, done in [(0.99, True), (0.95, False), (0.50, False)]:
                for _ in range(3):
                    td = TensorDict({
                        'observation': torch.randn(1, 4, 64, 64),
                        'action': torch.randn(1, 2),
                        'next': TensorDict({
                            'observation': torch.randn(1, 4, 64, 64),
                            'done': torch.tensor([[done]]),
                            'truncated': torch.tensor([[False]]),
                            'completion_ratio': torch.tensor([[completion]]),
                            'reward': torch.tensor([[1.0]]),
                        }, batch_size=[1]),
                    }, batch_size=[1])
                    buffer.extend(td)

            buffer.set_sampling_ratio((0.33, 0.33, 0.34))
            sampled = buffer.sample()

            yield buffer, sampled

    def test_priority_update_routing(self, buffer_with_samples):
        """优先级更新路由

        验证点:
        - 根据bucket_id路由到正确的子buffer
        - 更新成功（不抛异常）
        """
        buffer, sampled = buffer_with_samples

        # 设置td_error
        sampled.set('td_error', torch.rand(sampled.batch_size[0], 1))

        # 更新优先级应该成功
        buffer.update_tensordict_priority(sampled)
        # 无异常即通过

    def test_priority_update_missing_bucket_id(self):
        """缺失bucket_id时更新应该失败"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LazyMemmapStorage(max_size=100, scratch_dir=tmpdir)
            buffer = BucketedTensorDictPrioritizedReplayBuffer(
                storage=storage,
                alpha=0.6,
                beta=0.4,
                batch_size=2,
                prefetch=0,
                pin_memory=False,
                success_threshold=0.99,
                near_end_threshold=0.90,
            )

            # 添加数据
            td = TensorDict({
                'observation': torch.randn(1, 4, 64, 64),
                'action': torch.randn(1, 2),
                'next': TensorDict({
                    'observation': torch.randn(1, 4, 64, 64),
                    'done': torch.tensor([[True]]),
                    'truncated': torch.tensor([[False]]),
                    'completion_ratio': torch.tensor([[0.99]]),
                    'reward': torch.tensor([[1.0]]),
                }, batch_size=[1]),
            }, batch_size=[1])
            buffer.extend(td)

            # 创建没有bucket_id的tensordict
            fake_td = TensorDict({
                'observation': torch.randn(1, 4, 64, 64),
                'td_error': torch.tensor([[0.5]]),
            }, batch_size=[1])

            # 更新应该失败
            with pytest.raises(KeyError):
                buffer.update_tensordict_priority(fake_td)


class TestBucketedReplayReset:
    """测试清桶功能"""

    def test_reset_buckets(self):
        """清桶功能

        验证点:
        - reset_buckets()后所有bucket_sizes归零
        - 采样比例保持不变
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LazyMemmapStorage(max_size=100, scratch_dir=tmpdir)
            buffer = BucketedTensorDictPrioritizedReplayBuffer(
                storage=storage,
                alpha=0.6,
                beta=0.4,
                batch_size=2,
                prefetch=0,
                pin_memory=False,
                success_threshold=0.99,
                near_end_threshold=0.90,
            )

            # 添加数据
            for completion, done in [(0.99, True), (0.95, False), (0.50, False)]:
                for _ in range(3):
                    td = TensorDict({
                        'observation': torch.randn(1, 4, 64, 64),
                        'action': torch.randn(1, 2),
                        'next': TensorDict({
                            'observation': torch.randn(1, 4, 64, 64),
                            'done': torch.tensor([[done]]),
                            'truncated': torch.tensor([[False]]),
                            'completion_ratio': torch.tensor([[completion]]),
                            'reward': torch.tensor([[1.0]]),
                        }, batch_size=[1]),
                    }, batch_size=[1])
                    buffer.extend(td)

            # 验证有数据
            sizes_before = buffer.bucket_sizes
            assert sizes_before['success'] > 0
            assert sizes_before['near_end'] > 0
            assert sizes_before['mid'] > 0

            # 设置采样比例
            buffer.set_sampling_ratio((0.2, 0.3, 0.5))

            # 清桶
            buffer.reset_buckets()

            # 验证清空
            sizes_after = buffer.bucket_sizes
            assert sizes_after['success'] == 0
            assert sizes_after['near_end'] == 0
            assert sizes_after['mid'] == 0

            # 验证采样比例保持
            assert buffer._sampling_ratio == (0.2, 0.3, 0.5)
