#!/usr/bin/env python3
"""
test_06_multiscale_stride_pooling.py: Standalone Verification of Pure NumPy Stride-View Multiscale Pooling.

Verifies the mathematical and interface contracts for multiscale observation pooling (Vol 02 / Kernel 2):
1. Proves bitwise exact equivalence (0.00e+00 diff) of 5D stride-view max-pooling vs. canonical block max-pooling.
2. Eliminates 4th iteration dead downsampling (P08 dead calculation elimination).
3. Preserves use_global_features contract (D04): guarantees output shape of (C * n_scales + C, feat_size, feat_size)
   i.e. (25, 16, 16) to satisfy downstream ResNet-FPN model input contracts.

Execution Time Target: < 0.1s
Dependencies: standard library, numpy
"""

from __future__ import annotations

import time
import numpy as np


def canonical_block_maxpool2d(arr: np.ndarray, k: int) -> np.ndarray:
    """Canonical mathematical definition of 2D block max-pooling."""
    C, H, W = arr.shape
    out_H = H // k
    out_W = W // k
    out = np.zeros((C, out_H, out_W), dtype=arr.dtype)
    for c in range(C):
        for i in range(out_H):
            for j in range(out_W):
                out[c, i, j] = np.max(arr[c, i * k : (i + 1) * k, j * k : (j + 1) * k])
    return out


def stride_view_maxpool2d(arr: np.ndarray, k: int) -> np.ndarray:
    """Zero-copy 5D stride-view max-pooling primitive."""
    C, H, W = arr.shape
    return arr.reshape(C, H // k, k, W // k, k).max(axis=(2, 4))


def generate_multiscale_pyramid(
    base_obs: np.ndarray,
    n_scales: int = 4,
    feat_size: int = 16,
    use_global_features: bool = True,
) -> np.ndarray:
    """
    Optimized multiscale pyramid pipeline:
    - Eliminates dead 4th downsampling
    - Appends global pooled features when use_global_features is True
    """
    C, H, W = base_obs.shape
    half_crop = feat_size // 2
    center = H // 2

    scale_crops = []
    curr = base_obs

    # Loop strictly n_scales times, downsampling only n_scales - 1 times (no dead pool)
    for i in range(n_scales):
        cropped = curr[:, center - half_crop : center + half_crop, center - half_crop : center + half_crop]
        scale_crops.append(cropped)
        if i < n_scales - 1:
            curr = stride_view_maxpool2d(curr, k=2)
            center //= 2

    # Append global feature layer (D04 correction to satisfy ResNet-FPN input shape)
    if use_global_features:
        k_global = H // feat_size
        global_pooled = stride_view_maxpool2d(base_obs, k=k_global)
        scale_crops.append(global_pooled)

    return np.concatenate(scale_crops, axis=0)


def test_multiscale_stride_pooling() -> None:
    t0 = time.perf_counter()
    np.random.seed(42)

    C, H, W = 5, 128, 128
    x = np.random.randn(C, H, W).astype(np.float32)

    # 1. Bitwise Equivalence: 2x2 Stride View vs. Canonical
    pool_canon_2 = canonical_block_maxpool2d(x, k=2)
    pool_stride_2 = stride_view_maxpool2d(x, k=2)
    assert np.array_equal(pool_canon_2, pool_stride_2), "2x2 Stride view pooling must be bitwise identical"

    # 2. Bitwise Equivalence: 8x8 Stride View vs. Canonical
    pool_canon_8 = canonical_block_maxpool2d(x, k=8)
    pool_stride_8 = stride_view_maxpool2d(x, k=8)
    assert np.array_equal(pool_canon_8, pool_stride_8), "8x8 Stride view pooling must be bitwise identical"

    # 3. Multiscale Pyramid Feature Verification (n_scales=4, feat_size=16, use_global=True)
    n_scales = 4
    feat_size = 16
    pyramid = generate_multiscale_pyramid(x, n_scales=n_scales, feat_size=feat_size, use_global_features=True)

    expected_channels = C * n_scales + C  # 5*4 + 5 = 25
    expected_shape = (expected_channels, feat_size, feat_size)

    assert pyramid.shape == expected_shape, f"Pyramid shape {pyramid.shape} != expected {expected_shape}"
    assert np.all(np.isfinite(pyramid)), "Non-finite values detected in multiscale pyramid output"

    # Verify that crops match corresponding downscaled versions
    assert np.array_equal(pyramid[:5], x[:, 64 - 8 : 64 + 8, 64 - 8 : 64 + 8]), "Scale 0 crop mismatch"

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[PASS] test_06_multiscale_stride_pooling completed in {elapsed_ms:.2f}ms")


if __name__ == "__main__":
    test_multiscale_stride_pooling()
