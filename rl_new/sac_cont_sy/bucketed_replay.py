from __future__ import annotations
from enum import Enum
from math import floor
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple
import shutil
import torch
from tensordict import TensorDict
from torchrl.data import (LazyMemmapStorage, TensorDictPrioritizedReplayBuffer)

class BucketId(Enum):
    SUCCESS = 0
    NEAR_END = 1
    MID = 2

class BucketedTensorDictPrioritizedReplayBuffer:
    """
    Drop-in replacement for TensorDictPrioritizedReplayBuffer that internally
    manages three prioritized buffers (success / near_end / mid) and samples
    a batch according to a configurable ratio.

    Public API kept compatible where relevant:
      - append_transform(fn)
      - extend(tensordict)
      - sample()
      - update_tensordict_priority(sampled_td)

    Additional helpers:
      - set_sampling_ratio((s, n, m))
      - reset_buckets()
      - bucket_sizes (property)
    """

    def __init__(self, *, alpha: float, beta: float, batch_size: int, pin_memory: bool, prefetch: int,
                 storage: LazyMemmapStorage, success_threshold: float = 0.99,
                 near_end_threshold: float = 0.90) -> None:
        # public config
        self.alpha, self.beta = alpha, beta
        self.batch_size, self.pin_memory, self.prefetch = batch_size, pin_memory, prefetch
        # 子缓冲区默认不启用预取，避免动态 batch_size 与后台线程冲突；如需开启，可显式传入 bucket_prefetch
        self.success_threshold, self.near_end_threshold = success_threshold, near_end_threshold
        self._sampling_ratio = (0.30, 0.30, 0.40)  # sampling ratio defaults (S2 default: 0.30/0.30/0.40)

        # create three sub-storages under the provided scratch dir
        assert isinstance(storage, LazyMemmapStorage), ("storage must be LazyMemmapStorage with scratch_dir set")
        self._root_dir, self._max_size = Path(storage.scratch_dir), int(storage.max_size)

        # capacity per bucket (fixed fractions; keep sum within max_size), success:near:mid = 1:1:2 by default to ensure mid has room for fallback
        capacity_success = max(1, self._max_size // 4)
        capacity_near = max(1, self._max_size // 4)
        capacity_mid = max(1, self._max_size - capacity_success - capacity_near)

        # transforms to mirror onto each bucket
        self._transforms: List[Callable[[TensorDict], TensorDict]] = []
        
        self._buffers: Dict[BucketId, TensorDictPrioritizedReplayBuffer] = {}
        self._init_bucket(BucketId.SUCCESS, capacity_success)
        self._init_bucket(BucketId.NEAR_END, capacity_near)
        self._init_bucket(BucketId.MID, capacity_mid)



    # ------------- initialization helpers -------------
    def _init_bucket(self, bucket_id: BucketId, capacity: int) -> None:
        bucket_dir = self._root_dir / bucket_id.name.lower()
        bucket_dir.mkdir(parents=True, exist_ok=True)
        sub_storage = LazyMemmapStorage(max_size=capacity, scratch_dir=str(bucket_dir))
        buffer = TensorDictPrioritizedReplayBuffer(alpha=self.alpha, beta=self.beta, batch_size=None,
                                                   pin_memory=self.pin_memory, prefetch=self.prefetch,
                                                   storage=sub_storage)
        for t in self._transforms: buffer.append_transform(t) # mirror transforms already registered
        self._buffers[bucket_id] = buffer

    # ------------- public helpers -------------
    def set_sampling_ratio(self, ratio: Tuple[float, float, float]) -> None:
        self._sampling_ratio = ratio

    def reset_buckets(self) -> None:
        """Clear on-disk storages for all buckets and recreate empty buffers."""
        self._buffers.clear()  # Clear existing buffers
        for sub in ["success", "near_end", "mid"]:  # Remove on-disk storage directories
            bucket_path = self._root_dir / sub
            if bucket_path.exists(): shutil.rmtree(bucket_path, ignore_errors=True)

        # Recreate empty buckets with same capacity allocation
        capacity_success, capacity_near = max(1, self._max_size // 4), max(1, self._max_size // 4)
        capacity_mid = max(1, self._max_size - capacity_success - capacity_near)
        self._init_bucket(BucketId.SUCCESS, capacity_success)
        self._init_bucket(BucketId.NEAR_END, capacity_near)
        self._init_bucket(BucketId.MID, capacity_mid)

    @property
    def bucket_sizes(self) -> Dict[str, int]:
        return {"success": int(self._buffers[BucketId.SUCCESS].storage._len),
                "near_end": int(self._buffers[BucketId.NEAR_END].storage._len),
                "mid": int(self._buffers[BucketId.MID].storage._len)}

    # ------------- API compatibility -------------
    def append_transform(self, fn: Callable[[TensorDict], TensorDict]):
        self._transforms.append(fn)
        for buffer in self._buffers.values():
            buffer.append_transform(fn)

    def extend(self, td: TensorDict) -> None:
        """Route incoming transitions into the appropriate bucket.

        Classification (per step):
        - success: next.done & completion>=success_thr & not truncated
        - near_end: completion>=near_end_thr and not success
        - mid: otherwise
        """
        # Flatten to 1D and get Flags directly - fail-fast if keys missing (indicates config error)
        flat = td.reshape(-1)
        completion_ratio = flat[("next", "completion_ratio")].squeeze(-1)
        done, truncated = flat[("next", "done")].squeeze(-1), flat[("next", "truncated")].squeeze(-1)

        # Classify transitions into buckets
        success_mask = done & (completion_ratio >= self.success_threshold) & (~truncated)
        near_end_mask = (completion_ratio >= self.near_end_threshold) & (~success_mask)
        mid_mask = ~(success_mask | near_end_mask)

        # Route to appropriate buckets
        if success_mask.any(): self._buffers[BucketId.SUCCESS].extend(flat[success_mask])
        if near_end_mask.any(): self._buffers[BucketId.NEAR_END].extend(flat[near_end_mask])
        if mid_mask.any(): self._buffers[BucketId.MID].extend(flat[mid_mask])

    def sample(self) -> TensorDict:
        success_ratio, near_end_ratio, mid_ratio = self._sampling_ratio
        # derive integer per-bucket sample sizes
        total = int(self.batch_size)
        n_success, n_near_end = max(0, int(floor(total * success_ratio))), max(0, int(floor(total * near_end_ratio)))
        n_mid = max(0, total - n_success - n_near_end)

        parts: List[TensorDict] = []

        # helper to sample with fallback to mid
        def _safe_sample(bucket_id: BucketId, n: int) -> Optional[TensorDict]:
            """Sample n items from bucket, fallback to MID if insufficient data."""
            if n <= 0: return None
            try:
                sampled, actual_bucket_id = self._buffers[bucket_id].sample(batch_size=n), bucket_id
            except (RuntimeError, ValueError) as err:
                if bucket_id == BucketId.MID:
                    raise RuntimeError(f"MID bucket sampling failed with n={n}: {err}") from err
                sampled, actual_bucket_id = self._buffers[BucketId.MID].sample(batch_size=n), BucketId.MID # Fallback to MID buffer when requested bucket缺样本

            sampled.set("bucket_id", torch.full_like(sampled.get("index"), int(actual_bucket_id.value)))
            return sampled

        for bucket_id, n in ((BucketId.SUCCESS, n_success), (BucketId.NEAR_END, n_near_end), (BucketId.MID, n_mid)):
            out = _safe_sample(bucket_id, n)
            if out is not None:
                parts.append(out)

        if not parts: raise RuntimeError("Bucketed replay could not sample any data.")
        if len(parts) == 1: return parts[0]
        return torch.cat(parts, dim=0)

    def update_tensordict_priority(self, td: TensorDict) -> None:
        """Update priorities by redirecting to the proper bucket using the auxiliary 'bucket_id' returned at sampling time."""
        if "bucket_id" not in td.keys():
            raise KeyError("Sampled tensordict missing 'bucket_id' key for priority update routing.")
        bucket_ids = td.get("bucket_id").reshape(-1)
        for bucket in BucketId:
            mask = bucket_ids == int(bucket.value)
            if mask.any(): self._buffers[bucket].update_tensordict_priority(td[mask])
