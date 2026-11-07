# NewCppRL Track A: Verification Infrastructure & Test Philosophy (TEST_INFRA.md)

> **Document Type**: Architecture & Test Infrastructure Specification  
> **Target Scope**: NewCppRL Track A (Bug Fixes, Performance Engineering & Documentation Canon)  
> **Authors**: Test Writer E2E & Independent Verification Team  
> **Last Updated**: 2026-11-06T16:55:00Z  
> **Integrity Mode**: Development / Non-regression Gate  

---

## 1. Executive Summary & Test Philosophy

The NewCppRL verification infrastructure is built upon the **Scientist Mindset** defined in the project's aesthetic standards (*"Code should behave like a scientific instrument: accurate error reporting is far more valuable than elegant masking"*). Rather than writing defensive, facade-based unit tests that always pass or mask silent failures with generic fallback values, our verification suite enforces rigorous, deterministic, and fail-fast invariants across every algorithmic component and bug fix.

### Core Testing Pillars

1. **Evidence-Based Ground Truth (No Facades)**:
   Every test establishes an authoritative oracle—either a closed-form mathematical derivation, an unconstrained graph search baseline, or an exhaustive numerical comparator. No mock objects or dummy pass-throughs are permitted to simulate core logic.
2. **Deterministic & Bit-Level Precision**:
   Wherever discrete indices, tensor shapes, or graph traversals are involved (e.g., APF 2D boundary wrap, Steps95 boolean reset masks, multiscale pooling views), tests demand **100% bitwise identity (0 ULP difference)**. Continuous geometric transformations (closed-form affine inverse projection) enforce tight mathematical bounds ($\text{MAE} < 10^{-3}$, 99%+ pixels within subpixel bilinear rounding).
3. **Ultra-Low Latency (<2.0s Hard Budget)**:
   The entire micro-kernel test suite executes in **$< 1.0\text{s}$ total** (measured at $\approx 0.82\text{s}$ in-process), enabling instantaneous developer feedback loops without heavy WSL/CUDA/Ray multi-node dependencies.
4. **Falsification-First Architecture**:
   Before verifying that a fix works, tests explicitly execute the flawed baseline to **prove and reproduce the failure mode** (e.g., reproducing cylinder wrap in 1D APF, update starvation in UTD division, diagonal truncation in naive bounded BFS, and false overlap on virgin fields).

---

## 2. Four-Tier Testing Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Tier 4: Zero-Diff Read-Only Integrity Audit          │
│                       (git status verification outside docs/, 08 & 09)      │
├─────────────────────────────────────────────────────────────────────────────┤
│                       Tier 3: Macro-Component Interface Contracts          │
│                       (MDP invariance, tensor shape (*B, 1), 25-ch pyramid) │
├─────────────────────────────────────────────────────────────────────────────┤
│                       Tier 2: Micro-Kernel Numerical Equivalence Suite      │
│                       (8 Standalone Tests in docs/.../standalone_tests/)    │
├─────────────────────────────────────────────────────────────────────────────┤
│                       Tier 1: Static AST Syntax Verification Harness        │
│                       (check_docs_ast.py: 100+ Markdown code blocks)        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Tier 1: Static AST Syntax Verification (`check_docs_ast.py`)
- **Objective**: Verify that every single Python snippet embedded in documentation Markdown files (`00`~`07`, `10`, `README.md`) is syntactically valid Python 3 code.
- **Mechanism**: Extracts all ` ```python ... ``` ` blocks, dedents code snippets, and runs `ast.parse()`. Captures block numbers, markdown file line offsets, syntax error diagnostic messages, and code previews.
- **Fail-Fast Boundary**: Any trailing syntax errors (such as incomplete code snippets, invalid indentation, or misplaced keyword arguments) trigger an immediate non-zero exit code.

### Tier 2: Micro-Kernel Numerical & Bitwise Equivalence Suite (`standalone_tests/`)
- **Objective**: Standalone validation of core mathematical, geometric, and state-machine algorithms underlying Bug Fixes (Volume 05) and Performance Optimizations (Volumes 01–04).
- **Execution Target**: Sequential execution in $< 2.0$s with 100% pass rate.
- **Dependencies**: Restricted strictly to Python 3 standard library, NumPy, and OpenCV (`cv2`). Zero heavy distributed, GPU, or simulation dependencies.

### Tier 3: Macro-Component Interface Contracts
- **Objective**: Ensure that refactored components preserve Markov Decision Process (MDP) state transitions, coordinate systems, and downstream model shapes.
- **Contract Rules**:
  - Coordinate system: Float continuous $(x, y) \in [0, W] \times [0, H]$ and grid discrete $(r, c) \in [0, H-1] \times [0, W-1]$.
  - Heading angle: Continuous $\theta \in [-\pi, \pi]$ radians, counter-clockwise positive from x-axis.
  - Multi-scale observation: Exact output shape $(C \cdot N_{\text{scales}} + C, H_f, W_f) = (25, 16, 16)$ matching ResNet-FPN feature backbone expectations.
  - Transform output: Strict `(*batch_size, 1)` tensor shape compliance.

### Tier 4: Zero-Diff Read-Only Integrity Audit
- **Objective**: Maintain complete safety and isolation of the production codebase during documentation and verification engineering.
- **Boundary**: `git status` must confirm zero modifications outside `docs/performance_and_fixes/`. Files `08_code_aesthetics_and_less_is_more_refactoring.md` and `09_directory_topology_and_naming_architecture.md` remain strictly read-only.

---

## 3. Comprehensive Micro-Kernel Test Inventory

The Tier 2 test harness consists of 8 dedicated micro-kernel test scripts located in `docs/performance_and_fixes/standalone_tests/`:

| # | Test Script | Target Feature | Milestone | Pathology / Bug Mechanism | First-Principles Fix & Invariant |
|---|-------------|----------------|-----------|---------------------------|----------------------------------|
| **1** | `test_01_b01_apf_boundary_wrap.py` | B01 APF 2D Boundary Isolation | M1 | Flattened 1D index `0 <= aim < H*W` caused cylinder wrap from `(r, W-1)` to `(r+1, 0)`. | Explicit 2D check `0 <= nx < H && 0 <= ny < W`. Proves zero boundary leakage across East/West/North/South edges. |
| **2** | `test_02_b02_steps95_tensor_mask.py` | B02 Steps95 Tensor Mask Reset | M1 | Single environment reset wiped all concurrent parallel worker step counts to zero. | Vectorized per-env tensor mask `has_reached_95[mask] = False`. Proves non-resetting envs preserve exact counts. |
| **3** | `test_03_b04_utd_accumulator.py` | B04 UTD Frame Debt Accumulator | M1 | `ceil(20/2048 * utd)` deadlocked at 1; doc 05 division by batch_size starved updates by 102.4x. | Frame-based UTD accumulator: `debt += frames * utd; updates = floor(debt)`. Bit-level determinism across 4M frames. |
| **4** | `test_04_bounded_bfs_apf_equivalence.py` | Bounded BFS APF $\lceil D_{\max}\sqrt{2} \rceil$ | M2 | Bounded BFS cutting off at $step \ge D_{\max}$ truncated diagonal cells (error > 0.35). | Manhattan cutoff $R_{L1} = \lceil D_{\max}\sqrt{2} \rceil$ + 2D LUT. Proves bitwise exact $0.00e+00$ diff vs full-grid BFS. |
| **5** | `test_05_closed_form_affine_ego_patch.py` | Closed-Form 2x3 Affine Ego Patch | M2 | 3-stage padding/cropping/rotating allocated 79.80MB intermediate heap buffers per step. | Single 2x3 rigid affine matrix $M$. Direct sampling eliminates transient memory (0.0MB), $\text{MAE} < 10^{-3}$, $28\times$ faster. |
| **6** | `test_06_multiscale_stride_pooling.py` | Multiscale Stride Pooling View | M2 | 4th round dead pooling discarded; PyTorch/NumPy roundtrip boxing overhead. | Pure NumPy 5D stride view `reshape(...).max(axis=(2, 4))` + global layer. Bitwise identical output, shape $(25, 16, 16)$. |
| **7** | `test_07_aabb_collision_detection.py` | AABB Collision Early Exit | M2 | Full-canvas $512 \times 512$ zero allocation and rasterization for tiny $6 \times 4$ vehicle. | AABB bounding box check short-circuits free space with 0 rasterization. 1,000 poses tested with 100% agreement. |
| **8** | `test_08_quad_sweep_seam_erasure.py` | B07 Quadrilateral Seam Erasure | M1 | Swept quads $Q_t$ and $Q_{t+1}$ double-rasterized shared edge $F_{t+1}$, causing false overlap. | Seam erasure `cv2.line(footprint, F_{t+1}, 0, 1)` + stationary check. Virgin field straight drive has strictly 0 overlap. |

---

## 4. Test Runner Specifications (`run_all_tests.py`)

### Execution Mechanics
- **Default Mode (Fast In-Process)**:
  Dynamically imports all 8 modules and executes test functions sequentially within a shared Python process. This completely eliminates operating system process fork and Python/OpenCV import overhead (~240ms per process), reducing total execution time to **$< 0.9$ seconds**.
- **Subprocess Mode (`--subprocess`)**:
  Spawns an isolated `subprocess.run([sys.executable, ...])` for each script, verifying standalone process viability and zero global state leakage across modules.

### Acceptance Criteria
- **Pass Criterion**: Exactly $8 / 8$ tests return status `PASS`.
- **Latency Budget**: Total execution time must not exceed $2.0$ seconds.
- **Exit Code**: Returns `0` on 100% pass and latency compliance; non-zero (`1`) otherwise.

---

## 5. Automated Verification Workflow

To execute the complete E2E verification suite locally:

```bash
# 1. Run all 8 Standalone Numerical & Bit-Level Tests (< 1.0s)
python docs/performance_and_fixes/standalone_tests/run_all_tests.py

# 2. Run AST Syntax Validation across all Canonical Documentation Volumes
python docs/performance_and_fixes/check_docs_ast.py

# 3. Optional: Run AST check across all Markdown files in the directory
python docs/performance_and_fixes/check_docs_ast.py --all
```
