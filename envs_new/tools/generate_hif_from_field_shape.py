#!/usr/bin/env python3
"""
Generate HIF (Human Intention Field) from field shape masks.

Design goals
- Only operate on weed_coverage maps (never touch field_coverage).
- Output format identical to existing HIF files used by v5:
  - 2D float32 array, shape (H, W), pixels in field: angle in [0, π), background: -1.0
  - Saved as {map_dir}/hif/human_intent_field_{id}.npy
  - Visualization: {map_dir}/hif/image/orientation_field_{id}.png

Methods
- harmonic (default): boundary tangents -> double-angle Dirichlet conditions -> harmonic extension inside.
- edt: distance-transform tangent everywhere + double-angle smoothing (fast fallback).

CLI
  python envs_new/tools/generate_hif_from_field_shape.py \
      --map-dir envs_new/maps/weed_coverage \
      --method harmonic \
      --overwrite false

Notes
- HIF angle convention (as in v5): 0 rad = West, π/2 = South, π = East (undirected, modulo π).
- We derive a standard angle θ_std from image-space tangent (0=East, π/2=South) and map to HIF: θ_hif = (π - θ_std) mod π.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla


ANGLE_PI = math.pi


@dataclass
class Options:
    map_dir: Path
    method: str = "harmonic"  # or "edt"
    overwrite: bool = False
    sigma: float = 1.0  # used by edt smoothing (in pixels)
    max_files: Optional[int] = None
    dry_run: bool = False
    # Vector visualization
    vector_visualize: bool = True
    vector_style: str = "arrow"  # or "segment"
    vector_step: int = 12  # grid stride in pixels
    vector_half_len: int = 7  # half length for segment; length for arrow
    vector_thickness: int = 1  # line/arrow thickness in pixels (thin, as in first preview)
    vector_outline: bool = False  # no outline by default (matches first preview)


def _guard_map_dir(map_dir: Path) -> None:
    """Ensure we operate only on weed_coverage; avoid field_coverage by default."""
    parts = {p for p in map_dir.parts}
    if "field_coverage" in parts:
        raise ValueError(f"Refusing to operate on field_coverage: {map_dir}")
    if "weed_coverage" not in parts:
        # Allow override via env var if absolutely necessary, but default is strict.
        if os.environ.get("ALLOW_NON_WEED_COVERAGE", "0") != "1":
            raise ValueError(
                f"Map dir must be under weed_coverage (got: {map_dir}). Set ALLOW_NON_WEED_COVERAGE=1 to override.")


def _list_field_masks(field_dir: Path, max_files: Optional[int]) -> List[Path]:
    files = sorted([p for p in field_dir.glob("field_*.png") if p.is_file()])
    if max_files is not None:
        files = files[:max_files]
    return files


def _extract_id(p: Path) -> Optional[str]:
    m = re.search(r"field_(\d+)\.png$", p.name)
    return m.group(1) if m else None


def _ensure_dirs(hif_dir: Path) -> Tuple[Path, Path]:
    image_dir = hif_dir / "image"
    hif_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    return hif_dir, image_dir


def _load_mask(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    # The repository uses non-black as field in some loaders; here we assume binary field masks (1/0 or 255/0)
    mask = (img > 0).astype(np.uint8)
    return mask


def _compute_boundary(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask, kernel)
    boundary = (mask & (~eroded))
    return boundary.astype(bool)


def _distance_field(mask: np.ndarray) -> np.ndarray:
    # distanceTransform expects non-zero as foreground, computes distance to zero (background)
    return cv2.distanceTransform(mask, cv2.DIST_L2, 5)


def _sobel_grad(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return dx, dy


def _std_angle_from_tangent(tx: np.ndarray, ty: np.ndarray) -> np.ndarray:
    # Standard image-space angle: 0=+x(East), pi/2=+y(South), range (-pi, pi]
    return np.arctan2(ty, tx)


def _to_hif_angle(theta_std: np.ndarray) -> np.ndarray:
    # Map standard angle to HIF convention: 0=West, pi/2=South, pi=East (undirected field modulo pi)
    theta = (ANGLE_PI - theta_std)  # shift 180 degrees
    # Reduce to [0, pi)
    theta = np.mod(theta, ANGLE_PI)
    return theta.astype(np.float32)


def _double_angle(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.cos(2.0 * theta, dtype=np.float32), np.sin(2.0 * theta, dtype=np.float32)


def _solve_harmonic(mask: np.ndarray, boundary_mask: np.ndarray,
                    cos2_b: np.ndarray, sin2_b: np.ndarray,
                    max_iter: int = 10_000, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Solve Dirichlet Laplace for cos(2θ) and sin(2θ) inside the domain.
    Returns U, V defined on all pixels (background filled with 0),
    where interior unknowns are solved and boundary are fixed to the provided values.
    """
    h, w = mask.shape
    inside = mask.astype(bool)
    # Unknown interior = inside & ~boundary
    unknown = inside & (~boundary_mask)
    num_unknown = int(np.count_nonzero(unknown))

    if num_unknown == 0:
        # Degenerate: no interior; just return boundary values and zeros elsewhere.
        U = np.zeros((h, w), dtype=np.float32)
        V = np.zeros((h, w), dtype=np.float32)
        U[boundary_mask] = cos2_b[boundary_mask]
        V[boundary_mask] = sin2_b[boundary_mask]
        return U, V

    index = -np.ones((h, w), dtype=np.int32)
    index[unknown] = np.arange(num_unknown, dtype=np.int32)

    # Prepare sparse matrix (5-point Laplacian)
    data: List[float] = []
    rows: List[int] = []
    cols: List[int] = []

    # RHS for cos and sin
    bu = np.zeros(num_unknown, dtype=np.float64)
    bv = np.zeros(num_unknown, dtype=np.float64)

    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Iterate unknown pixels and assemble
    ys, xs = np.nonzero(unknown)
    for k, (y, x) in enumerate(zip(ys, xs)):
        # Diagonal
        rows.append(k); cols.append(k); data.append(4.0)

        # Neighbors
        for dy, dx in nbrs:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue  # outside image (should not happen for interior nodes)
            if not inside[ny, nx]:
                # neighbor outside domain implies current is boundary; but current is unknown interior, so ignore
                continue
            if unknown[ny, nx]:
                j = index[ny, nx]
                rows.append(k); cols.append(j); data.append(-1.0)
            else:
                # Neighbor is boundary: move known value to RHS
                bu[k] += cos2_b[ny, nx]
                bv[k] += sin2_b[ny, nx]

    A = sparse.csr_matrix((np.asarray(data, dtype=np.float64), (rows, cols)), shape=(num_unknown, num_unknown))

    # Solve A x = b for both U and V using CG
    # SciPy >=1.11 uses rtol/atol instead of tol
    try:
        u_sol, info_u = spla.cg(A, bu, maxiter=max_iter, tol=tol)
        v_sol, info_v = spla.cg(A, bv, maxiter=max_iter, tol=tol)
    except TypeError:
        u_sol, info_u = spla.cg(A, bu, maxiter=max_iter, rtol=tol)
        v_sol, info_v = spla.cg(A, bv, maxiter=max_iter, rtol=tol)

    if info_u != 0 or info_v != 0:
        raise RuntimeError(f"CG did not converge: info_u={info_u}, info_v={info_v}")

    U = np.zeros((h, w), dtype=np.float32)
    V = np.zeros((h, w), dtype=np.float32)
    U[boundary_mask] = cos2_b[boundary_mask].astype(np.float32)
    V[boundary_mask] = sin2_b[boundary_mask].astype(np.float32)
    U[unknown] = u_sol.astype(np.float32)
    V[unknown] = v_sol.astype(np.float32)
    return U, V


def _angles_from_double_angle(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    theta = 0.5 * np.arctan2(V, U).astype(np.float32)
    # Map to [0, π)
    theta = np.mod(theta, ANGLE_PI)
    return theta.astype(np.float32)


def _harmonic_line_field(mask: np.ndarray) -> np.ndarray:
    # Compute boundary and distance field
    boundary = _compute_boundary(mask)
    if not np.any(boundary):
        raise ValueError("No boundary found in mask")
    d = _distance_field(mask)
    dx, dy = _sobel_grad(d)

    # Tangent as 90-degree rotation of normal (dx,dy) -> (-dy, dx)
    tx = -dy
    ty = dx

    # Normalize only where gradient is significant to avoid NaNs
    mag = np.sqrt(tx * tx + ty * ty) + 1e-8
    tx /= mag
    ty /= mag

    theta_std = _std_angle_from_tangent(tx, ty)
    theta_hif = _to_hif_angle(theta_std)

    # Boundary values in double-angle space
    cos2_b = np.zeros_like(theta_hif, dtype=np.float32)
    sin2_b = np.zeros_like(theta_hif, dtype=np.float32)
    c, s = _double_angle(theta_hif)
    cos2_b[boundary] = c[boundary]
    sin2_b[boundary] = s[boundary]

    # Harmonic extension
    U, V = _solve_harmonic(mask, boundary, cos2_b, sin2_b)
    theta = _angles_from_double_angle(U, V)
    return theta


def _gaussian_blur_safe(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    k = int(max(3, (int(sigma * 4) // 2) * 2 + 1))  # odd kernel size ~ 4*sigma
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)


def _edt_line_field(mask: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    d = _distance_field(mask)
    dx, dy = _sobel_grad(d)
    tx = -dy
    ty = dx
    mag = np.sqrt(tx * tx + ty * ty) + 1e-8
    tx /= mag
    ty /= mag
    theta_std = _std_angle_from_tangent(tx, ty)
    theta_hif = _to_hif_angle(theta_std)

    # Smooth in double-angle space to reduce medial-axis artifacts
    cos2 = np.cos(2.0 * theta_hif, dtype=np.float32)
    sin2 = np.sin(2.0 * theta_hif, dtype=np.float32)
    cos2 = _gaussian_blur_safe(cos2, sigma)
    sin2 = _gaussian_blur_safe(sin2, sigma)
    theta = _angles_from_double_angle(cos2, sin2)
    return theta


def _angles_to_hsv_image(theta: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Map angles to HSV for visualization: H ∈ [0,179] proportional to θ/π, S=255, V=255. Background black."""
    h, w = theta.shape
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    valid = (mask > 0)
    h_val = (theta[valid] / ANGLE_PI * 179.0).astype(np.uint8)
    hsv[valid, 0] = h_val
    hsv[valid, 1] = 255
    hsv[valid, 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def _draw_vectors(img_bgr: np.ndarray, theta: np.ndarray, mask: np.ndarray,
                  step: int = 12, half_len: int = 6, style: str = "arrow",
                  thickness: int = 2, outline: bool = True) -> None:
    """Overlay vector visualization onto BGR image in-place.
    - style 'segment': draw centered line segments (orientation only).
    - style 'arrow'  : draw a single arrow from center to endpoint (direction picked arbitrarily).
    """
    h, w = mask.shape
    # Convert to standard image-space angle for drawing (0=East, pi/2=South)
    theta_std = (ANGLE_PI - theta)
    cos_t = np.cos(theta_std)
    sin_t = np.sin(theta_std)
    # Grid sampling
    y0 = step // 2
    x0 = step // 2
    for y in range(y0, h, step):
        for x in range(x0, w, step):
            if mask[y, x] == 0:
                continue
            dx = float(cos_t[y, x])
            dy = float(sin_t[y, x])
            if style == "segment":
                x1 = int(round(x - dx * half_len))
                y1 = int(round(y - dy * half_len))
                x2 = int(round(x + dx * half_len))
                y2 = int(round(y + dy * half_len))
                if outline:
                    cv2.line(img_bgr, (x1, y1), (x2, y2), (0, 0, 0), max(1, thickness + 1), cv2.LINE_AA)
                cv2.line(img_bgr, (x1, y1), (x2, y2), (255, 255, 255), thickness, cv2.LINE_AA)
            else:
                x2 = int(round(x + dx * half_len))
                y2 = int(round(y + dy * half_len))
                if outline:
                    cv2.arrowedLine(img_bgr, (x, y), (x2, y2), (0, 0, 0), max(1, thickness + 1), cv2.LINE_AA, tipLength=0.4)
                cv2.arrowedLine(img_bgr, (x, y), (x2, y2), (255, 255, 255), thickness, cv2.LINE_AA, tipLength=0.4)


def _save_hif_and_image(theta: np.ndarray, mask: np.ndarray, npy_path: Path, png_path: Path,
                        draw_vectors: bool = False, vector_style: str = "arrow",
                        vector_step: int = 12, vector_half_len: int = 6,
                        vector_thickness: int = 2, vector_outline: bool = True) -> None:
    # Compose HIF map with background = -1.0
    hif = np.full(theta.shape, -1.0, dtype=np.float32)
    hif[mask > 0] = theta[mask > 0].astype(np.float32)
    np.save(str(npy_path), hif)

    # Visualization
    bgr = _angles_to_hsv_image(theta, mask)
    # Optional: overlay boundary
    boundary = _compute_boundary(mask)
    overlay = bgr.copy()
    overlay[boundary] = (0, 255, 0)  # green boundary
    # Optional: overlay vectors
    if draw_vectors:
        _draw_vectors(
            overlay, theta, mask,
            step=vector_step, half_len=vector_half_len, style=vector_style,
            thickness=vector_thickness, outline=vector_outline,
        )
    cv2.imwrite(str(png_path), overlay)


def process_one(field_path: Path, opts: Options, hif_dir: Path, image_dir: Path) -> None:
    fid = _extract_id(field_path)
    if fid is None:
        print(f"[skip] Unrecognized file name: {field_path.name}")
        return

    npy_path = hif_dir / f"human_intent_field_{fid}.npy"
    png_path = image_dir / f"orientation_field_{fid}.png"

    if not opts.overwrite and npy_path.exists():
        print(f"[skip] Exists: {npy_path}")
        return

    mask = _load_mask(field_path)
    if opts.dry_run:
        print(f"[dry-run] Would generate: {npy_path} and {png_path} (shape={mask.shape})")
        return

    try:
        if opts.method == "harmonic":
            theta = _harmonic_line_field(mask)
        elif opts.method == "edt":
            theta = _edt_line_field(mask, sigma=opts.sigma)
        else:
            raise ValueError(f"Unknown method: {opts.method}")
    except Exception as e:
        # Fallback to EDT method if harmonic fails
        print(f"[warn] Harmonic method failed on {field_path.name}: {e}. Falling back to EDT.")
        theta = _edt_line_field(mask, sigma=max(1.0, opts.sigma))

    _save_hif_and_image(
        theta, mask, npy_path, png_path,
        draw_vectors=opts.vector_visualize,
        vector_style=opts.vector_style,
        vector_step=opts.vector_step,
        vector_half_len=opts.vector_half_len,
        vector_thickness=opts.vector_thickness,
        vector_outline=opts.vector_outline,
    )
    # Quick content check
    valid = (mask > 0)
    if np.any(valid):
        v = theta[valid]
        print(f"[ok] {field_path.name} -> HIF saved. θ[min,max]=[{float(v.min()):.4f}, {float(v.max()):.4f}] rad, valid={int(valid.sum())}")
    else:
        print(f"[ok] {field_path.name} -> Empty mask? Saved background-only HIF.")


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Generate HIF from field masks (weed_coverage only)")
    p.add_argument("--map-dir", type=Path, default=Path("envs_new/maps/weed_coverage"), help="Parent dir containing field/ and hif/")
    p.add_argument("--method", type=str, default="harmonic", choices=["harmonic", "edt"], help="HIF generation method")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing HIF files")
    p.add_argument("--sigma", type=float, default=1.0, help="Smoothing sigma for EDT method")
    p.add_argument("--max-files", type=int, default=None, help="Limit number of files")
    p.add_argument("--dry-run", action="store_true", help="List planned actions only")
    # Vector visualization options
    p.add_argument("--with-vectors", dest="vector_visualize", action="store_true", help="Overlay orientation vectors on visualization")
    p.add_argument("--no-vectors", dest="vector_visualize", action="store_false", help="Disable vector overlay on visualization")
    p.add_argument("--vector-style", type=str, default="arrow", choices=["arrow", "segment"], help="Vector visualization style")
    p.add_argument("--vector-step", type=int, default=12, help="Grid stride for vectors (pixels)")
    p.add_argument("--vector-length", type=int, default=7, help="Half-length for segments; length for arrows (pixels)")
    p.add_argument("--vector-thickness", type=int, default=1, help="Vector line/arrow thickness (pixels)")
    p.add_argument("--vector-outline", dest="vector_outline", action="store_true", help="Draw black outline for contrast")
    p.add_argument("--no-vector-outline", dest="vector_outline", action="store_false", help="Disable vector outline")
    # Set defaults BEFORE parsing so argparse applies them when flags are absent
    p.set_defaults(vector_visualize=True)
    p.set_defaults(vector_outline=False)
    args = p.parse_args(argv)

    # Safety: if using both --with-vectors/--no-vectors not provided and parser produced None, fallback to True
    if getattr(args, "vector_visualize", None) is None:
        args.vector_visualize = True
    if getattr(args, "vector_outline", None) is None:
        args.vector_outline = True

    opts = Options(
        map_dir=args.map_dir,
        method=args.method,
        overwrite=args.overwrite,
        sigma=args.sigma,
        max_files=args.max_files,
        dry_run=args.dry_run,
        vector_visualize=args.vector_visualize,
        vector_style=args.vector_style,
        vector_step=args.vector_step,
        vector_half_len=args.vector_length,
        vector_thickness=args.vector_thickness,
        vector_outline=args.vector_outline,
    )

    _guard_map_dir(opts.map_dir)

    field_dir = opts.map_dir / "field"
    hif_dir = opts.map_dir / "hif"
    if not field_dir.exists():
        raise FileNotFoundError(f"Field directory not found: {field_dir}")

    _ensure_dirs(hif_dir)
    _, image_dir = _ensure_dirs(hif_dir)

    files = _list_field_masks(field_dir, opts.max_files)
    if not files:
        print(f"No field_*.png found in {field_dir}")
        return 0

    print(f"Found {len(files)} field masks in {field_dir}")
    for fp in files:
        process_one(fp, opts, hif_dir, image_dir)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
