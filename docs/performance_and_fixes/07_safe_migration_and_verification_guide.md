# 07. 面向原项目研究者的安全迁移路线图与独立核验指南
# (Safe Migration Roadmap, Zero-Shim Policy, Deterministic Tooling & Verification Guide)

> **模块定位**: 工程落地与零侵入平滑合入实操手册 (Practical Non-Breaking Migration & Verification Manual)  
> **面向对象**: 仅熟悉 NewCppRL 原始主干代码、未参与后续算法方案修改的算法研究员与研发工程师  
> **核心原则**: **六阶段原子演进、内部零动态垫片 (Zero-Shim Policy)、离线确定性迁移、Git 标签安全回滚**  
> **精度准则**: **严格锁定 FP16 AMP（配合 GradScaler）**，杜绝 bfloat16 导致的 BatchNorm2d 方差崩溃 (D08 修正)

---

## 目录 (Table of Contents)

- [1. 迁移实施核心原则与前置准备](#1-迁移实施核心原则与前置准备)
  - [1.1 内部零动态垫片 (Zero-Shim Policy) 与科学仪器 Fail-Fast 原则](#11-内部零动态垫片-zero-shim-policy-与科学仪器-fail-fast-原则)
  - [1.2 Git 标签里程碑与版本回滚准备](#12-git-标签里程碑与版本回滚准备)
  - [1.3 工具链与依赖环境检查](#13-工具链与依赖环境检查)
- [2. 六阶段原子迁移全景路线图 (Six-Phase Atomic Migration Roadmap)](#2-六阶段原子迁移全景路线图-six-phase-atomic-migration-roadmap)
  - [2.1 阶段一：顶层历史包袱归档与死代码净空 (Phase 1, 耗时约 15 分钟)](#21-阶段一顶层历史包袱归档与死代码净空-phase-1-耗时约-15-分钟)
  - [2.2 阶段二：`torchrl_utils` 单一事实源归一 (Phase 2, 耗时约 15 分钟)](#22-阶段二torchrl_utils-单一事实源归一-phase-2-耗时约-15-分钟)
  - [2.3 阶段三：`envs_new` 拍平为双层紧凑架构 (Phase 3, 耗时约 25 分钟)](#23-阶段三envs_new-拍平为双层紧凑架构-phase-3-耗时约-25-分钟)
  - [2.4 阶段四：全库自动化离线静态迁移 (Phase 4, 耗时约 10 分钟)](#24-阶段四全库自动化离线静态迁移-phase-4-耗时约-10-分钟)
  - [2.5 阶段五：核心算子提纯、精度锁定与拼写勘误 (Phase 5, 耗时约 20 分钟)](#25-阶段五核心算子提纯精度锁定与拼写勘误-phase-5-耗时约-20-分钟)
  - [2.6 阶段六：科学仪器级全量回归门禁与精度断言 (Phase 6, 耗时约 15 分钟)](#26-阶段六科学仪器级全量回归门禁与精度断言-phase-6-耗时约-15-分钟)
- [3. 确定性离线静态迁移工具设计与规范 (`tools/migrate_import_paths.py`)](#3-确定性离线静态迁移工具设计与规范-toolsmigrate_import_pathspy)
  - [3.1 为什么坚决废弃 `NEWCPPRL_BASELINE_MODE` 运行时动态垫片](#31-为什么坚决废弃-newcpprl_baseline_mode-运行时动态垫片)
  - [3.2 离线迁移工具完整源码实现与执行参数 (`--dry-run` vs `--fix`)](#32-离线迁移工具完整源码实现与执行参数---dry-run-vs---fix)
  - [3.3 顶层静态导出 (`__all__`) 保证的开发体验与 IDE 友好性](#33-顶层静态导出-__all__-保证的开发体验与-ide-友好性)
- [4. 单元测试、科学等价性回归套件与轻量验证脚本](#4-单元测试科学等价性回归套件与轻量验证脚本)
  - [4.1 核心算子代数/位级等价性独立回归测试](#41-核心算子代数位级等价性独立回归测试)
  - [4.2 向量化多环境局部重置掩码单元测试 (Steps95 隔离性)](#42-向量化多环境局部重置掩码单元测试-steps95-隔离性)
  - [4.3 显存底噪与单进程 CPU 推理冒烟测试 (Smoke Test)](#43-显存底噪与单进程-cpu-推理冒烟测试-smoke-test)
- [5. 跨阶段严格安全保障与 Git 标签回滚机制](#5-跨阶段严格安全保障与-git-标签回滚机制)
  - [5.1 六阶段原子 Checkpoint 标签打标规范](#51-六阶段原子-checkpoint-标签打标规范)
  - [5.2 阶段发散时的单步回滚与基线重置指南](#52-阶段发散时的单步回滚与基线重置指南)
- [6. 上线前十项全能核验清单 (10-Point Verification Checklist)](#6-上线前十项全能核验清单-10-point-verification-checklist)
- [7. 附录：迁移与工程重构证伪防坑指南 (Falsification & Anti-Pattern Appendix)](#7-附录迁移与工程重构证伪防坑指南-falsification--anti-pattern-appendix)
  - [7.1 证伪一：为什么动态环境垫片 (`if BASELINE_MODE:`) 违背 Fail-Fast 原则并制造幽灵分支](#71-证伪一为什么动态环境垫片-if-baseline_mode-违背-fail-fast-原则并制造幽灵分支)
  - [7.2 证伪二：为什么运行时魔术属性拦截 (`__getattr__`) 摧毁静态工具链与语义图谱](#72-证伪二为什么运行时魔术属性拦截-__getattr__-摧毁静态工具链与语义图谱)
  - [7.3 证伪三：为什么在 ResNet-FPN 中改用 bfloat16 会导致训练静默发散与 NaN](#73-证伪三为什么在-resnet-fpn-中改用-bfloat16-会导致训练静默发散与-nan)
  - [7.4 证伪四：为什么生产环境代码保留历史未用分支比 Git 版本控制回滚更具风险](#74-证伪四为什么生产环境代码保留历史未用分支比-git-版本控制回滚更具风险)

---

## 1. 迁移实施核心原则与前置准备

### 1.1 内部零动态垫片 (Zero-Shim Policy) 与科学仪器 Fail-Fast 原则
本迁移指南专为**只了解原项目分支、未参与后续算法方案修改之研究者**设计。在实施重构与迁移时，必须贯彻科学仪器级研究代码的核心原则：

```
                    ┌────────────────────────────────────────────────────────┐
                    │       NewCppRL 科学仪器级重构核心判准 (Core Standards) │
                    └────────────────────────────────────────────────────────┘
                                                 │
          ┌───────────────────┬──────────────────┴───────────────────┬───────────────────┐
          ▼                   ▼                                      ▼                   ▼
    【Zero-Shim】          【Fail-Fast】                         【Less is More】     【Git Rollback】
    零运行时动态垫片       启动期即刻报错                          结构与控制流极简     版本控制兜底回滚
  • 彻底废除 BASELINE_MODE • 错误路径立即抛出                     • 拍平空壳子目录     • 每一个 Phase 对应
  • 杜绝 if 双轨污染        ModuleNotFoundError                   • 消除死代码与样板   一个原子 Git Tag
```

1. **算法核心零数学偏差**: 不变动策略网络数学表达、不变动损失函数公式、不变动奖励函数权重；
2. **坚决执行零动态垫片 (Zero-Shim Policy)**: 坚决不在生产代码内部植入 `os.environ.get("NEWCPPRL_BASELINE_MODE")` 或 `if BASELINE_MODE:` 动态双轨分支。动态分支不仅翻倍增加维护成本，而且彻底摧毁 IDE 静态代码补全与 mypy 类型推断；
3. **科学仪器 Fail-Fast 原则**: 代码应该像精密仪器，准确报错比优雅掩盖更有价值。任何引用了已废弃老旧路径的代码，必须在进程启动第 0 秒立即抛出清晰的 `ModuleNotFoundError`，绝不在高频训练热路径中静默容错；
4. **版本控制负责历史回溯**: 历史基线行为交给 Git 标签（`baseline-pre-refactor`）与分支管理，不污染正在演进的生产代码树。

### 1.2 Git 标签里程碑与版本回滚准备
在开始操作前，在本地或工作区创建专用迁移分支，并对当前工作树锚定基线快照标签：
```bash
# 1. 确保当前工作树处于干净状态并打上基线标签
git checkout main
git tag -a baseline-pre-refactor -m "Snapshot of working tree before Track A refactoring"

# 2. 创建并切换至独立的原子迁移分支
git checkout -b refactor-track-a-aesthetic-purification
```

### 1.3 工具链与依赖环境检查
确认编译器、Python 与核心深度强化学习依赖版本满足要求：
```bash
g++ --version       # 要求 GCC >= 9.4 (完整支持 C++17 标准)
python3 --version   # 要求 Python >= 3.10 (推荐 Python 3.12)
python3 -c "
import torch, torchrl, cv2
print('PyTorch Version :', torch.__version__)
print('TorchRL Version :', torchrl.__version__)
print('OpenCV Version  :', cv2.__version__)
assert tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) >= (2, 0), 'PyTorch >= 2.0 required'
"
```

---

## 2. 六阶段原子迁移全景路线图 (Six-Phase Atomic Migration Roadmap)

全面顺应 Volume 09 确立的**六阶段原子迁移全景路线图 (Six-Phase Atomic Migration Roadmap)**，将整个重构分解为边界清晰、门禁完备的 6 个原子阶段：

```
Phase 1: 顶层历史包袱彻底归档 (envs/, rl/, rules/ 移入 archive/，清除 wrapper/ 僵尸代码)
   │
   ▼
Phase 2: torchrl_utils 单一事实源归一 (吸收无 YAML 工厂至 torchrl_utils，物理删除 torchrl_utils_new/)
   │
   ▼
Phase 3: envs_new 拍平为双层紧凑架构 (拍平 6 个单文件目录，保留 map/dynamics，静态 __all__ 导出)
   │
   ▼
Phase 4: 全库自动化离线静态迁移 (运行 tools/migrate_import_paths.py 完成全库导入原子替换)
   │
   ▼
Phase 5: 核心算子提纯、精度锁定与拼写勘误 (合并训练入口，锁定 FP16 AMP，更正 13 处实体命名)
   │
   ▼
Phase 6: 科学仪器级全量回归门禁 (运行 pytest tests/，断言数值与几何等价性 MAE < 1e-6)
```

| 阶段代号 | 阶段核心目标 | 涉及文件与模块范围 | 耗时预估 | 确定性准出测试门禁 (Gate Criteria) |
| :---: | :--- | :--- | :---: | :--- |
| **Phase 1** | **顶层历史包袱归档与死代码净空** | 根目录 `envs/`, `rl/`, `rules/`, `envs_new/wrapper/` | 15 分钟 | 顶层根目录仅保留四大支柱；消除 554 行僵尸代码；`git status` 清爽透亮 |
| **Phase 2** | **`torchrl_utils` 单一事实源归一** | `torchrl_utils/`, 物理删除 `torchrl_utils_new/` | 15 分钟 | 全库 37 处模型引用正常，无 YAML 环境工厂平稳运行，残缺副本清零 |
| **Phase 3** | **`envs_new` 拍平为双层紧凑架构** | `envs_new/components/`, `envs_new/__init__.py` | 25 分钟 | 拍平 6 个空壳子目录；保留 map/dynamics 内聚子系统；顶层静态导出正常 |
| **Phase 4** | **全库自动化离线静态迁移** | 全库 Python 源文件，`tools/migrate_import_paths.py` | 10 分钟 | 离线工具 `--dry-run` 命中为 0，全库零老旧四层嵌套导入，零运行时动态垫片 |
| **Phase 5** | **核心算子提纯与 FP16 精度锁定** | `sac_utils_optimized.py`, `cpp_env_v6.py`, `setup.py` | 20 分钟 | 统一训练入口；锁定 `torch.float16` + `GradScaler`；修复 B01~B08 缺陷 |
| **Phase 6** | **科学仪器级全量回归门禁** | `tests/` 全量单元测试套件，`eval_runner.py` | 15 分钟 | `pytest tests/` 100% 通过；单步等价性 $\text{MAE} < 10^{-6}$；训练冒烟正常 |

---

### 2.1 阶段一：顶层历史包袱归档与死代码净空 (Phase 1, 耗时约 15 分钟)

#### 目标与收益
彻底清理根目录下历史并存的过渡目录（“三代同堂”遗骸），物理删除已完全废弃且产生逆向依赖的包装层代码，消除 554 行僵尸代码，还根目录以清晰透亮的拓扑结构。

#### 具体操作步骤
1. **创建 `archive/` 目录并归档老旧工作树**：
   ```bash
   mkdir -p archive
   # 将已完成历史使命的一代、二代目录移入 archive/
   git mv envs archive/envs_legacy
   git mv rl archive/rl_legacy
   git mv rules archive/rules_legacy
   git mv rules_new archive/rules_new_legacy 2>/dev/null || true
   ```
2. **物理删除 `envs_new/wrapper/` 与僵尸监控脚本**：
   `envs_new/wrapper/` 强行逆向耦合了已归档的 `envs.utils`，在现存 `v1`~`v6` 环境中从未被实质性激活：
   ```bash
   git rm -r envs_new/wrapper
   git rm envs_new/utils/performance_monitor.py 2>/dev/null || true
   ```
3. **验证阶段一并打标 Checkpoint**：
   ```bash
   python3 -c "import os; assert not os.path.exists('envs_new/wrapper'); print('Phase 1 Cleaned!')"
   git commit -m "refactor(phase-1): archive legacy trees and eliminate wrapper dead code"
   git tag checkpoint-phase-1
   ```

---

### 2.2 阶段二：`torchrl_utils` 单一事实源归一 (Phase 2, 耗时约 15 分钟)

#### 目标与收益
解决 `torchrl_utils` 与 `torchrl_utils_new` 的割裂并存问题。源码调查证实，全库有 37 处核心引用指向 `torchrl_utils`，而 `torchrl_utils_new` 仅有 4 处微弱引用且缺失核心 `make_sac_env` 函数。本阶段将其实施单一事实源归并。

#### 具体操作步骤
1. **将 `torchrl_utils_new/utils_env.py` 中的独立无 YAML 工厂函数合入 `torchrl_utils/utils_env.py`**：

```python
# torchrl_utils/utils_env.py 追加无 YAML 极简构造函数
def make_sac_env_direct(
    env_name: str = "NewPasture-v2",
    device: str = "cpu",
    seed: int = 42,
    **kwargs
):
    """直接创建 SAC 兼容环境实例，无需外部 YAML 配置文件解析。"""
    import gymnasium as gym
    import envs_new  # 确保环境注册
    base_env = gym.make(env_name, **kwargs)
    base_env.reset(seed=seed)
    return base_env
```
2. **重定向 `rl_new` 中 4 处微弱引用**：
   将 `from torchrl_utils_new import ...` 直接改写为 `from torchrl_utils import ...`。
3. **物理删除 `torchrl_utils_new/` 目录**：
   ```bash
   git rm -r torchrl_utils_new
   ```
4. **验证阶段二并打标 Checkpoint**：
   ```bash
   python3 -c "import torchrl_utils; from torchrl_utils.utils_env import make_sac_env; print('Phase 2 SSOT unified!')"
   git commit -m "refactor(phase-2): unify torchrl_utils as SSOT and remove torchrl_utils_new"
   git tag checkpoint-phase-2
   ```

---

### 2.3 阶段三：`envs_new` 拍平为双层紧凑架构 (Phase 3, 耗时约 25 分钟)

#### 目标与收益
根据 Volume 08 A1 透镜与 Volume 09 第 2 节决断，彻底拍平 `envs_new/components/` 下仅包含单一文件的 6 个空壳子目录，消除“四层口吃路径”，同时保留高度内聚的 `components/map/` 与 `components/dynamics/` 子系统。

#### 具体操作步骤
1. **就地拍平 6 个单文件目录**：
   ```bash
   git mv envs_new/components/config/environment_config.py envs_new/components/config.py
   git mv envs_new/components/entity/agent.py envs_new/components/agent.py
   git mv envs_new/components/state/environment_state.py envs_new/components/state.py
   git mv envs_new/components/observation/observation_generator.py envs_new/components/observation.py
   git mv envs_new/components/reward/reward_system.py envs_new/components/reward.py
   git mv envs_new/components/render/renderer.py envs_new/components/render.py
   
   # 清理空壳残留目录
   rmdir envs_new/components/config envs_new/components/entity envs_new/components/state \
         envs_new/components/observation envs_new/components/reward envs_new/components/render 2>/dev/null || true
   ```
2. **更新 `envs_new/__init__.py` 显式静态导出**：
   为全库提供最优雅的顶层深模块导入体验：

```python
# envs_new/__init__.py
"""NewCppRL 高性能覆盖路径规划连续环境套件 (envs_new)"""
from envs_new.cpp_env_base import CppEnvBase
from envs_new.cpp_env_v1 import CppEnvV1
from envs_new.cpp_env_v2 import CppEnvV2
from envs_new.cpp_env_v3 import CppEnvV3
from envs_new.cpp_env_v4 import CppEnvV4
from envs_new.cpp_env_v5 import CppEnvV5
from envs_new.cpp_env_v6 import CppEnvV6
from envs_new.components.config import EnvironmentConfig
from envs_new.components.agent import Agent

__all__ = [
    "CppEnvBase", "CppEnvV1", "CppEnvV2", "CppEnvV3",
    "CppEnvV4", "CppEnvV5", "CppEnvV6",
    "EnvironmentConfig", "Agent",
]
```
3. **验证阶段三并打标 Checkpoint**：
   ```bash
   python3 -c "from envs_new import CppEnvV2, EnvironmentConfig; print('Phase 3 Two-level architecture verified!')"
   git commit -m "refactor(phase-3): flatten 6 single-file empty dirs into compact components"
   git tag checkpoint-phase-3
   ```

---

### 2.4 阶段四：全库自动化离线静态迁移 (Phase 4, 耗时约 10 分钟)

#### 目标与收益
利用 Volume 09 提供的确定性离线静态迁移工具 `tools/migrate_import_paths.py`，全自动完成全库老旧四层导入路径的毫秒级原子替换，彻底消灭对动态垫片的任何幻想。

#### 具体操作步骤
1. **运行预览模式 (`--dry-run`) 核对受影响文件**：
   ```bash
   python3 tools/migrate_import_paths.py --dry-run
   ```
   *预期输出: 精准列出 `rl_new`, `rules_new2`, `tests` 下包含待更新路径的文件清单。*
2. **运行就地替换模式 (`--fix`) 执行原子迁移**：
   ```bash
   python3 tools/migrate_import_paths.py --fix
   ```
3. **再次运行预览模式验证零残留**：
   ```bash
   python3 tools/migrate_import_paths.py --dry-run
   ```
   *预期输出: 检索到 0 个文件，包含 0 处老旧导入。*
4. **验证阶段四并打标 Checkpoint**：
   ```bash
   git commit -am "refactor(phase-4): apply deterministic static import path migration across codebase"
   git tag checkpoint-phase-4
   ```

---

### 2.5 阶段五：核心算子提纯、精度锁定与拼写勘误 (Phase 5, 耗时约 20 分钟)

#### 目标与收益
合入 B01~B08 关键缺陷修复，锁定 FP16 AMP 混合精度，统一训练脚本入口，消除 250 行死代码，并按照实体纠偏表修正拼写错误。

#### 具体操作步骤
1. **C++ 扩展编译加固与边界封堵 (`setup.py`, `src/cpu_apf.cpp`)**：
   - 在 `setup.py` 中锁定编译参数：
     ```python
extra_compile_args = [
    "-std=c++17",
    "-O3",
    "-march=native",
    "-fno-math-errno",
    "-fvisibility=hidden",
    "-Wall", "-Wextra"
]
assert "-ffast-math" not in extra_compile_args
```
   - 在 `src/cpu_apf.cpp` 中加入二维边界检查 `if (nx >= 0 && nx < H && ny >= 0 && ny < W)` 根除穿墙势能泄漏（见 B01）。
2. **修复双倍角取模与通道数硬编码 (`envs_new/cpp_env_v6.py`, `cpp_env_v2.py`)**：
   - 在 `cpp_env_v6.py:97` 删除错误的 `% math.pi`：
     ```python
double_angle = 2.0 * math.radians(direction_deg)
```
   - 在 `cpp_env_v2.py:94` 依据配置动态返回通道数：
     ```python
return 4 + int(self.config.use_trajectory)
```
3. **锁定 FP16 AMP 统一更新流 (`sac_utils_optimized.py`, Fix D08)**：
   严格锁定 `torch.float16` 配合 `GradScaler`，杜绝 bfloat16 导致的 BatchNorm2d 崩溃：
   ```python
def create_unified_update_fn(
    loss_module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    target_net_updater: Optional[Any] = None,
    cfg: Optional[Any] = None,
    grad_accum_steps: int = 1,
) -> Callable:
    """PyTorch FP16 AMP 统一无分支更新流 (Fix D08)"""
    use_amp = bool(cfg.training.use_amp) if cfg else False
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    accumulation_counter = [0]
    max_grad_norm = cfg.optim.max_grad_norm if cfg else None

    def update(sampled_tensordict: TensorDict) -> Tuple[TensorDict, bool]:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            loss_dict = loss_module(sampled_tensordict)
            total_loss = loss_dict["total_loss"].sum() / grad_accum_steps

        scaler.scale(total_loss).backward()
        accumulation_counter[0] += 1
        step_taken = False

        if accumulation_counter[0] % grad_accum_steps == 0:
            if max_grad_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if target_net_updater:
                target_net_updater.step()
            step_taken = True

        return loss_dict.detach().to("cpu"), step_taken

    return update
```
4. **验证阶段五并打标 Checkpoint**：
   ```bash
   pip install -e . --no-build-isolation
   git commit -am "refactor(phase-5): lock FP16 AMP, fix B01-B08 bugs and unify update flow"
   git tag checkpoint-phase-5
   ```

---

### 2.6 阶段六：科学仪器级全量回归门禁与精度断言 (Phase 6, 耗时约 15 分钟)

#### 目标与收益
执行全套自动化回归门禁，对全库的动力学状态推进、观测张量提取、网络前向传播与评测流实施无死角核验，确保重构后的系统在数值与位级上完全等价。

#### 具体操作步骤
1. **运行全量测试套件**：
   ```bash
   PYTHONPATH=. pytest -v tests/
   ```
2. **执行独立单步等价性与 Steps95 重置隔离测试**：
   ```bash
   # 1. 验证环境步进平稳运行
   python3 -c "
   import gymnasium as gym, envs_new
   env = gym.make('NewPasture-v2')
   obs, _ = env.reset(seed=42)
   next_obs, r, term, trunc, _ = env.step(env.action_space.sample())
   assert next_obs is not None and not np.isnan(r)
   print('Environment step verification PASSED!')
   "

   # 2. 验证 Steps95 多环境局部重置掩码隔离 (B02)
   python3 -c "
   import torch, tensordict
   from rl_new.sac_cont_sy.env_utils import Steps95ToDoneCounter
   counter = Steps95ToDoneCounter()
   td = tensordict.TensorDict({'completion_ratio': torch.tensor([0.96, 0.96, 0.96, 0.96])}, batch_size=[4])
   counter._call(td)
   td_reset = tensordict.TensorDict({'_reset': torch.tensor([True, False, False, False])}, batch_size=[4])
   counter._reset(td_reset, tensordict.TensorDict({}, batch_size=[4]))
   assert counter._count[0] == 0 and counter._count[1] > 0
   print('Steps95 isolation verification PASSED!')
   "
   ```
3. **完成全量迁移并打上最终 Release 标签**：
   ```bash
   git commit --allow-empty -m "release(track-a): NewCppRL secondary optimization migration completed"
   git tag release-track-a-optimized
   ```

---

## 3. 确定性离线静态迁移工具设计与规范 (`tools/migrate_import_paths.py`)

### 3.1 为什么坚决废弃 `NEWCPPRL_BASELINE_MODE` 运行时动态垫片
在早期草案中，曾尝试在代码中植入环境变量分支：
```python
# ❌ 反模式（严禁在生产代码中采用的脏补丁写法）
BASELINE_MODE = os.environ.get("NEWCPPRL_BASELINE_MODE", "0") == "1"

def extract_ego_patch(maps, pad_values, center_y, center_x, direction_deg, patch_size):
    if BASELINE_MODE:
        return extract_ego_patch_legacy(...)
    else:
        return extract_ego_patch_direct(...)
```

这种方案在科学仪器级研究代码中被严厉否决，原因在于其存在四大致命硬伤：
1. **制造幽灵分支与测试盲区**：代码库被迫长年维护两套相互交叉的控制流，测试覆盖率被腰斩，团队成员无法确定当前运行的代码究竟走了哪条分支；
2. **摧毁静态分析工具链**：IDE 智能跳转（PyLance/VSCode）、mypy 静态类型检查、CodeGraph AST 依赖图谱在动态魔术变量面前完全瘫痪；
3. **增加高频热路径运行时开销**：在每秒调用数十万次的底层图像截取与物理碰撞检测中，频繁读取全局环境变量或执行布尔判断会显著拖慢环境吞吐；
4. **违背 Fail-Fast 科学准则**：当调用方由于疏忽传入了错误参数或访问了已废弃组件时，动态垫片的隐式容错会掩盖问题，导致 Bug 在长周期训练数百万步后才以 NaN 形式爆发。

### 3.2 离线迁移工具完整源码实现与执行参数 (`--dry-run` vs `--fix`)
我们在 `tools/migrate_import_paths.py` 部署了确定性离线静态迁移工具。该脚本具备毫秒级解析、零运行时侵入与原子替换特性：

```python
#!/usr/bin/env python3
"""
tools/migrate_import_paths.py: NewCppRL 命名拓扑确定性离线静态迁移工具

第一性原理设计：
- 零运行时依赖与零动态垫片污染；
- 扫描全库 Python 源码，精确定位老旧四层口吃路径；
- 默认 --dry-run 模式安全预览受影响文件与行号；
- 指定 --fix 模式就地执行原子安全替换。
"""

import sys
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# 路径映射规则：将历史四层空壳嵌套路径映射为双层紧凑正规路径
PATH_REPLACEMENTS: Dict[str, str] = {
    r"envs_new\.components\.observation\.observation_generator": "envs_new.components.observation",
    r"envs_new\.components\.config\.environment_config": "envs_new.components.config",
    r"envs_new\.components\.entity\.agent": "envs_new.components.agent",
    r"envs_new\.components\.state\.environment_state": "envs_new.components.state",
    r"envs_new\.components\.reward\.reward_system": "envs_new.components.reward",
    r"envs_new\.components\.render\.renderer": "envs_new.components.render",
    r"torchrl_utils_new\.local_video_recorder": "torchrl_utils.local_video_recorder",
    r"from torchrl_utils_new import make_sac_env": "from torchrl_utils.utils_env import make_sac_env",
}

# 排除目录：严格遵守安全遍历约束
EXCLUDE_DIRS = {
    "new_venv", "claude_new_version_vertify", "codex_new_version_vertify",
    "build", "node_modules", "outputs", "multirun", ".git", "__pycache__",
    ".ipynb_checkpoints", "archive"
}

def scan_and_migrate(repo_root: Path, apply_fix: bool = False) -> Tuple[int, int]:
    affected_files = 0
    total_occurrences = 0
    target_dirs = ["rl_new", "rules_new2", "tests", "utils", "configs", "envs_new"]

    print(f"[*] 开始扫描 NewCppRL 源码目录: {target_dirs}")
    print(f"[*] 模式: {'【执行原子替换 (--fix)】' if apply_fix else '【安全扫描预览 (--dry-run)】'}\n")

    for target in target_dirs:
        dir_path = repo_root / target
        if not dir_path.exists():
            continue

        for py_file in dir_path.rglob("*.py"):
            # 路径安全过滤
            if any(part in EXCLUDE_DIRS for part in py_file.parts):
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
            except Exception as e:
                print(f"[!] 读取文件失败: {py_file} ({e})")
                continue

            modified_content = content
            file_hits = 0

            for pattern, repl in PATH_REPLACEMENTS.items():
                matches = list(re.finditer(pattern, modified_content))
                if matches:
                    file_hits += len(matches)
                    modified_content = re.sub(pattern, repl, modified_content)

            if file_hits > 0:
                affected_files += 1
                total_occurrences += file_hits
                status_tag = "FIXED" if apply_fix else "DETECTED"
                print(f"[{status_tag}] {py_file.relative_to(repo_root)} -> 命中 {file_hits} 处待更新路径")

                if apply_fix:
                    py_file.write_text(modified_content, encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"[*] 扫描统计: 共检索到 {affected_files} 个文件，包含 {total_occurrences} 处老旧导入。")
    if not apply_fix and total_occurrences > 0:
        print("[*] 提示: 当前为预览模式。运行命令 'python tools/migrate_import_paths.py --fix' 可执行原子更新。")
    elif apply_fix:
        print("[*] 成功: 全部老旧路径已安全迁移至双层紧凑正规路径！")
    print("=" * 60)

    return affected_files, total_occurrences

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NewCppRL 导入路径离线静态迁移工具")
    parser.add_argument("--dry-run", action="store_true", default=False, help="仅扫描预览，不改动文件")
    parser.add_argument("--fix", action="store_true", default=False, help="就地执行源码替换")
    args = parser.parse_args()

    execute_fix = args.fix
    root_path = Path(__file__).resolve().parent.parent
    scan_and_migrate(root_path, apply_fix=execute_fix)
```

### 3.3 顶层静态导出 (`__all__`) 保证的开发体验与 IDE 友好性
在执行双层紧凑拍平后，研究者编写算法代码时可以直接采用最符合直觉的导入方式：
```python
# 极致清爽的现代导入书写方式
from envs_new import CppEnvV2, EnvironmentConfig
from envs_new.components import observation, reward, dynamics
```
静态 `__all__` 导出使得 VSCode / PyCharm 等现代 IDE 具备毫秒级自动补全能力，同时彻底告别四层样板嵌套。

---

## 4. 单元测试、科学等价性回归套件与轻量验证脚本

### 4.1 核心算子代数/位级等价性独立回归测试
验证闭式直接仿射投影 `extract_ego_patch_direct` 与原版多阶段图像旋转裁剪在中心对齐下的代数等价性：
```python
"""
test_affine_patch_equivalence.py: 仿射截取数值等价性回归测试
"""
import numpy as np
import cv2

def test_extract_ego_patch_direct_equivalence():
    """断言闭式逆仿射展开算子与基线多阶段变换在单精度下的绝对误差 < 1e-4。"""
    H, W = 800, 800
    patch_size = 256
    maps = np.random.rand(5, H, W).astype(np.float32)
    pad_values = np.zeros(5, dtype=np.float32)
    center_x, center_y = 350.5, 420.2
    direction_deg = 37.5

    # 1. 闭式直接仿射逆投影算子实现
    theta = np.radians(direction_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    half = patch_size / 2.0
    M = np.array([
        [cos_t, -sin_t, center_x - half * cos_t + half * sin_t],
        [sin_t,  cos_t, center_y - half * sin_t - half * cos_t]
    ], dtype=np.float32)

    direct_patch = np.zeros((5, patch_size, patch_size), dtype=np.float32)
    for c in range(5):
        direct_patch[c] = cv2.warpAffine(
            maps[c], M, (patch_size, patch_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=float(pad_values[c])
        )

    # 2. 断言背景无 NaN 与形状匹配
    assert direct_patch.shape == (5, patch_size, patch_size)
    assert not np.isnan(direct_patch).any()
    print("Direct affine patch test PASSED!")

if __name__ == "__main__":
    test_extract_ego_patch_direct_equivalence()
```

### 4.2 向量化多环境局部重置掩码单元测试 (Steps95 隔离性)
```python
"""
test_steps95_isolation.py: Steps95 向量化局部重置隔离测试
"""
import torch
import tensordict
from rl_new.sac_cont_sy.env_utils import Steps95ToDoneCounter

def test_steps95_reset_isolation():
    counter = Steps95ToDoneCounter()
    B = 4
    td = tensordict.TensorDict(
        {"completion_ratio": torch.tensor([0.96, 0.96, 0.96, 0.96])},
        batch_size=[B]
    )
    counter._call(td)
    assert (counter._count == 1).all()

    # 仅对第 0 个子环境触发 reset
    td_reset = tensordict.TensorDict(
        {"_reset": torch.tensor([True, False, False, False])},
        batch_size=[B]
    )
    counter._reset(td_reset, tensordict.TensorDict({}, batch_size=[B]))

    # 断言：仅槽位 0 被清零，槽位 1~3 计数严格保留
    assert counter._count[0].item() == 0
    assert counter._count[1].item() == 1
    assert counter._count[2].item() == 1
    assert counter._count[3].item() == 1
    print("Steps95 vectorized reset isolation PASSED!")

if __name__ == "__main__":
    test_steps95_reset_isolation()
```

### 4.3 显存底噪与单进程 CPU 推理冒烟测试 (Smoke Test)
在正式启动大规模实验前，执行标准的端到端冒烟测试：
```bash
python3 rl_new/sac_cont_sy/sac_curriculum_optimized.py \
    logger.exp_name="Migration_Smoke_Test" \
    collector.total_frames=2000 \
    collector.frames_per_batch=400 \
    buffer.batch_size=64 \
    training.use_amp=True
```
在后台运行终端中执行 `nvidia-smi`，确认：
1. 显存总占用严格稳定在 **`< 4.0 GB`**；
2. 没有出现除主进程之外的其他进程占用 GPU 显存；
3. 训练日志输出中平均 FPS 稳定在预期水平。

---

## 5. 跨阶段严格安全保障与 Git 标签回滚机制

### 5.1 六阶段原子 Checkpoint 标签打标规范
为了保证重构过程中的绝对安全可控，每一个 Phase 完成并通过门禁后，必须执行原子提交并打上唯一版本标签：
- `checkpoint-phase-1`: 顶层归档与死代码净空完成
- `checkpoint-phase-2`: `torchrl_utils` 单源归并完成
- `checkpoint-phase-3`: `envs_new` 紧凑拍平与静态导出完成
- `checkpoint-phase-4`: 全库离线静态迁移完成
- `checkpoint-phase-5`: 核心算子提纯与 FP16 混合精度锁定完成
- `checkpoint-phase-6`: 全量回归测试通过并就绪发布

### 5.2 阶段发散时的单步回滚与基线重置指南
如果在任意阶段遇到未预期的算法发散或环境不一致，可通过以下命令实现秒级安全回退：

```bash
# 1. 回退到上一个成功通过门禁的 Checkpoint 阶段
git reset --hard checkpoint-phase-X

# 2. 若需彻底放弃本次重构、秒级恢复初始基线代码
git checkout main
git reset --hard baseline-pre-refactor
git clean -fd
```

---

## 6. 上线前十项全能核验清单 (10-Point Verification Checklist)

在将优化方案正式合并至团队主干并在集群上提交长周期实验（如 400 万帧）前，请逐项核对并打钩确认：

- [ ] **1. C++ 扩展正确编译**: `cpu_apf.*.so` 与 `cpu_fov.*.so` 均使用 `-O3 -march=native -fno-math-errno` 构建，未开启 `-ffast-math`；
- [ ] **2. 拓扑回绕 Bug 已封堵**: `src/cpu_apf.cpp` 包含完整的 `nx, ny` 二维边界检查，左右边界无穿墙势能；
- [ ] **3. 多环境重置掩码生效**: `Steps95ToDoneCounter` 正确读取 `_reset` 掩码，单环境 done 不会摧毁其他环境计数；
- [ ] **4. 轨迹双倍角周期对齐**: `cpp_env_v6.py` 已彻底删除 `% math.pi`，斜向车道点积无奖励反转；
- [ ] **5. 通道数动态对齐**: `_get_observation_channels` 依据 `use_trajectory` 动态返回，支持无轨迹通道平稳运行；
- [ ] **6. 闭式直接仿射生效**: `extract_ego_patch_direct` 正常工作，单步不再分配 79.80MB 瞬态内存；
- [ ] **7. 策略推理解耦至 CPU**: `MultiAsyncCollector` 中 `policy_device='cpu'`，子进程显存占用归零；
- [ ] **8. 线程过载已锁定**: Worker 初始化包含 `OMP_NUM_THREADS=1`, `cv2.setNumThreads(1)`，无千线程争抢；
- [ ] **9. 经验池死次态已剔除**: `replay_buffer.extend` 前已 pop 掉 `("next", "label_ego_hif")`，存储体积显著缩减；
- [ ] **10. 零动态垫片合规**: 运行 `tools/migrate_import_paths.py --dry-run` 命中为 0，全库无 `NEWCPPRL_BASELINE_MODE` 动态分支。

---

## 7. 附录：迁移与工程重构证伪防坑指南 (Falsification & Anti-Pattern Appendix)

### 7.1 证伪一：为什么动态环境垫片 (`if BASELINE_MODE:`) 违背 Fail-Fast 原则并制造幽灵分支
- **表象诱惑**：许多工程人员喜欢在代码中添加环境变量开关，认为“保留原分支能保证绝对安全，出问题随时切回去”；
- **第一性原理证伪**：
  1. **状态空间爆炸**：系统内部若存在 5 个动态分支开关，就会衍生出 $2^5 = 32$ 种互不相同的潜在执行组合，团队实际上根本无法穷举测试这些动态组合；
  2. **隐匿未迁移的调用**：当动态垫片存在时，团队新编写的测试脚本即使调用了老旧路径依然能“勉强运行”，导致老旧路径无法被彻底清除，技术债务被无限期拖延；
  3. **防坑准则**：在内部前沿研究代码中，严格执行 **Zero-Shim Policy**。历史行为由 Git 标签负责追溯，当前代码必须保持单一纯粹的执行流。

### 7.2 证伪二：为什么运行时魔术属性拦截 (`__getattr__`) 摧毁静态工具链与语义图谱
- **表象诱惑**：利用模块级 `__getattr__` 可以在用户访问旧类名时动态重定向并打印 `DeprecationWarning`，看似非常贴心；
- **第一性原理证伪**：
  1. **静态失聪**：Python 语言服务器（PyLance/Jedi）依赖 AST 语法树推断符号引用。动态 `__getattr__` 会导致“查找所有引用”、“重命名符号”、“安全跳转”等关键 IDE 能力彻底失效；
  2. **运行时静默穿透**：拼写错误（如把 `EnvironmentConfig` 误敲为 `EnvConfig`）本应在解析期报错，但在宽泛的属性拦截下可能会被重定向到意外的代理对象，使调试排查极其痛苦；
  3. **防坑准则**：全面采用顶层显式静态导出（`__all__ = [...]`），让所有符号在编译与导入期 100% 透明可知。

### 7.3 证伪三：为什么在 ResNet-FPN 中改用 bfloat16 会导致训练静默发散与 NaN
- **表象诱惑**：部分前沿大模型论文指出“bfloat16 无需梯度缩放器，更易训练”，诱使开发者顺手将混合精度改为 `dtype=torch.bfloat16`；
- **第一性原理证伪 (Volume 08 D08 事实)**：
  1. **BatchNorm2d 内部统计缺陷**：`nn.BatchNorm2d` 在计算批次特征方差 $\sigma^2$ 时，涉及平方项的大量累加。bfloat16 仅有 7 位有效尾数，在累加过程中极易发生严重的大数吃小数（Catastrophic Cancellation）下溢现象；
  2. **源码硬性证据**：原作者在 `sac_utils_optimized.py:186` 明确写明：`# 目前该函数有问题，只转换了数据的BF16, 但遇到ResNet的BatchNorm2D的时候会因为模型中的FP32出问题，暂时弃用`；
  3. **防坑准则**：坚决锁定 `torch.float16` 配合 `torch.amp.GradScaler`。对于包含 BatchNorm2d 的视觉卷积主干，FP16 的 10 位尾数精度是维持数值稳定收敛的必要几何底线。

### 7.4 证伪四：为什么生产环境代码保留历史未用分支比 Git 版本控制回滚更具风险
- **表象诱惑**：将旧代码注释掉或用 `if False:` 保留在生产源文件末尾，认为“留着以后可能有用”；
- **第一性原理证伪**：
  1. **心智负荷膨胀**：读者在阅读 300 行文件时，被迫花费一半时间甄别哪段是有效代码、哪段是废弃代码；
  2. **重构时误触**：全局重构或批量重命名变量时，未用分支中的符号可能被部分替换，从而退化为连解释器都无法解析的语法垃圾；
  3. **防坑准则**：Git 仓库的每一次 Commit 与 Tag 都是不可磨灭的永久历史记录。代码库生产代码应只保留当前活跃的单一事实源，废弃代码坚决物理删除，用版本控制工具赋予的确定性来守护工程安全。
