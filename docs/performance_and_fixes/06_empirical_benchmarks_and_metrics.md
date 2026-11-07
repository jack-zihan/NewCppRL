# 06. 多硬件平台全梯队实测基准、评价指标体系与规则/强化学习博弈
# (Empirical Benchmarks, Multi-Tier Hardware Metrics, Axiomatic CPP Evaluation & Rules vs. RL Game Matrix)

> **模块定位**: 实测数据支撑、科学评价指标公理化与多硬件全景性能遥测 (Empirical Proof, Axiomatic Metrics & Hardware Telemetry)  
> **评测硬件**: 涵盖本地旗舰工作站（WSL2 i9-14900HX + RTX 4090）与 Google Colab 5 档主流云计算硬件（Tesla T4 Std/HM, L4, A100-SXM4, RTX PRO 6000 G4）  
> **数据来源**: `bench_v2_all_tiers.py`, `all_tiers_results.json`, `bench_real_e2e.py`, `rules_new2/metrics_io.py`, `eval_runner.py` (Commit `2b5d947ba`)  
> **核心结论**: 
> 1. 建立基于欧氏几何与非平稳遍历理论的公理化 CPP 评价指标体系（$\text{CR}, \text{OR}, L_{\tau}, L_{\tau}^{\text{norm}}, \rho_{95}, \Delta\Theta$），揭示收尾阶段路径比与转向平滑度的科学本质；
> 2. 深入解剖 `rules_new2/` 几何算法架构，构建专家启发式规则（BCP, Jump, Snake, R-Snake, React）与 SAC-ResNet 强化学习策略的多维度博弈与 Pareto 权衡矩阵；
> 3. 消除 CPU/IPC 瓶颈后，全系硬件实现 2.80× ~ 3.26× 端到端加速，本地 RTX 4090 工作站（3.8 小时）全面反超未优化前的顶级云端 A100 集群（11.3 小时）；
> 4. 纠正混合精度表述为 **FP16 AMP（配合 GradScaler，锁定 torch.float16）**，严格遵守 Volume 08 D08 裁决，规避 ResNet-FPN 中 BatchNorm2d 引起的方差发散与数值崩溃。

---

## 目录 (Table of Contents)

- [1. 覆盖路径规划（CPP）科学评价指标体系的第一性原理数学公理化](#1-覆盖路径规划cpp科学评价指标体系的第一性原理数学公理化)
  - [1.1 动态作业覆盖率 (Coverage Rate, $\text{CR}$) 与边界防御](#11-动态作业覆盖率-coverage-rate-textcr-与边界防御)
  - [1.2 重复覆盖重叠率 (Overlap Rate, $\text{OR}$) 与重叠计数 ($\text{OC}$)](#12-重复覆盖重叠率-overlap-rate-textor-与重叠计数-textoc)
  - [1.3 门限特征路径长度 (Threshold Path Length, $L_{90}, L_{95}, L_{98}$)](#13-门限特征路径长度-threshold-path-length-l_90-l_95-l_98)
  - [1.4 几何尺度面积归一化路径长度 (Area-Normalized Path Length, $L_{\tau}^{\text{norm}}$)](#14-几何尺度面积归一化路径长度-area-normalized-path-length-l_tautextnorm)
  - [1.5 收尾阶段路径消耗比 (Tail Path Ratio, $\rho_{90}, \rho_{95}$)](#15-收尾阶段路径消耗比-tail-path-ratio-rho_90-rho_95)
  - [1.6 转向平滑度与角速度变差 (Angular Roughness, $\Delta\Theta$)](#16-转向平滑度与角速度变差-angular-roughness-deltatheta)
  - [1.7 回合终态分类学体系 (Episode Termination Taxonomy)](#17-回合终态分类学体系-episode-termination-taxonomy)
- [2. `rules_new2/` 几何算法架构与启发式规则 vs 强化学习全景博弈](#2-rules_new2-几何算法架构与启发式规则-vs-强化学习全景博弈)
  - [2.1 `rules_new2/` 核心几何原语解剖 (`RectifiedFrame`, `WeedTracker`, `PathExecutor`)](#21-rules_new2-核心几何原语解剖-rectifiedframe-weedtracker-pathexecutor)
  - [2.2 五大经典基准规划算法特性分析 (BCP, Jump, Snake, R-Snake, React)](#22-五大经典基准规划算法特性分析-bcp-jump-snake-r-snake-react)
  - [2.3 专家启发式规则 vs 人工势场 (APF) vs SAC-ResNet 全景博弈矩阵](#23-专家启发式规则-vs-人工势场-apf-vs-sac-resnet-全景博弈矩阵)
  - [2.4 典型农田地图集实测性能与 Pareto 前沿消融对比](#24-典型农田地图集实测性能与-pareto-前沿消融对比)
- [3. 评测硬件矩阵与基准测试环境规范](#3-评测硬件矩阵与基准测试环境规范)
  - [3.1 六大硬件梯队物理规格表 (Local WSL2 至 Cloud G4)](#31-六大硬件梯队物理规格表-local-wsl2-至-cloud-g4)
  - [3.2 压测基准与严谨测量协议 (Methodology)](#32-压测基准与严谨测量协议-methodology)
- [4. 环境交互细粒度耗时分解与消融基准](#4-环境交互细粒度耗时分解与消融基准)
  - [4.1 v2, v4, v6 版本单步耗时全景对比 (表 6-1)](#41-v2-v4-v6-版本单步耗时全景对比-表-6-1)
  - [4.2 单步计算子模块微基准耗时消融 (9 大算子提速, 表 6-2)](#42-单步计算子模块微基准耗时消融-9-大算子提速-表-6-2)
- [5. 多进程并发与吞吐扩展性基准 (FPS Scaling)](#5-多进程并发与吞吐扩展性基准-fps-scaling)
  - [5.1 单环境 vs. 16/32 Worker 吞吐扩展对比 (表 6-3)](#51-单环境-vs-1632-worker-吞吐扩展对比-表-6-3)
  - [5.2 线程锁控 (`OMP_NUM_THREADS=1`) 吞吐提速曲线](#52-线程锁控-omp_num_threads1-吞吐提速曲线)
- [6. 内存层次、显存与存储削减指标](#6-内存层次显存与存储削减指标)
  - [6.1 显存底噪消除 (立省 7.65GB 显存，`policy_device='cpu'`)](#61-显存底噪消除-立省-765gb-显存policy_devicecpu)
  - [6.2 瞬态分配归零与评测 OOM 消除 (79.80MB $\to$ 0.00MB)](#62-瞬态分配归零与评测-oom-消除-7980mb-to-000mb)
  - [6.3 经验池存储削减与 PCIe H2D 传输加速 (51.5GB 净省，3.67× 加速)](#63-经验池存储削减与-pcie-h2d-传输加速-515gb-净省367-加速)
- [7. 跨硬件端到端全流程训练收益全景表](#7-跨硬件端到端全流程训练收益全景表)
  - [7.1 400 万帧全量训练物理时长对比 (表 6-4, FP16 AMP 修正)](#71-400-万帧全量训练物理时长对比-表-6-4-fp16-amp-修正)
  - [7.2 核心工程洞见：为什么本地 4090 能够反超未优化云端超算](#72-核心工程洞见为什么本地-4090-能够反超未优化云端超算)
- [8. 附录：评测指标与性能工程证伪防坑指南 (Falsification & Anti-Pattern Appendix)](#8-附录评测指标与性能工程证伪防坑指南-falsification--anti-pattern-appendix)
  - [8.1 证伪一：为什么朴素覆盖率与回合总奖励无法区分高效扫掠与无目的游荡](#81-证伪一为什么朴素覆盖率与回合总奖励无法区分高效扫掠与无目的游荡)
  - [8.2 证伪二：为什么 bfloat16 在 ResNet-FPN BatchNorm2d 下引发数值下溢与方差爆炸](#82-证伪二为什么-bfloat16-在-resnet-fpn-batchnorm2d-下引发数值下溢与方差爆炸)
  - [8.3 证伪三：为什么在环境微基准中忽略瞬态内存分配与 GC 压力会导致严重的系统级性能误判](#83-证伪三为什么在环境微基准中忽略瞬态内存分配与-gc-压力会导致严重的系统级性能误判)
  - [8.4 证伪四：为什么无面积归一化的原始路径长 $L_{\tau}$ 会破坏跨地图评价的统计有效性](#84-证伪四为什么无面积归一化的原始路径长-l_tau-会破坏跨地图评价的统计有效性)

---

## 1. 覆盖路径规划（CPP）科学评价指标体系的第一性原理数学公理化

在移动机器人连续农田全覆盖作业（Coverage Path Planning, CPP）领域，评价指标绝不能仅凭模糊的“最终覆盖率”或“单回合累积奖励”，必须建立在严谨的欧几里得几何、非平稳遍历理论与马尔可夫决策过程之上。

根据对 `rules_new2/metrics_io.py` 及 `envs_new` 动力学源码的底层数学提炼，完整的 CPP 科学评价体系由以下七大公理化指标构成：

### 1.1 动态作业覆盖率 (Coverage Rate, $\text{CR}$) 与边界防御
设农田有效作业区域为二维连通闭集 $\mathcal{F} \subset \mathbb{R}^2$，其离散像素总有效面积为 $A_{\text{field}} = |\mathcal{F}|$。割草机器人割刀底盘在时步 $t$ 的局部作业覆盖凸包为 $\mathcal{C}_t \subset \mathbb{R}^2$。则时步 $t$ 的累计覆盖点集为 $\mathcal{M}_t = \bigcup_{\tau=1}^t \mathcal{C}_\tau \cap \mathcal{F}$。
$$\text{CR}(t) = \frac{|\mathcal{M}_t|}{|\mathcal{F}|} = \frac{\sum_{(x, y) \in \mathcal{F}} \mathbb{I}\left(\text{Covered}(x, y, t)\right)}{A_{\text{field}}} \in [0.0, 1.0]$$
- **关键物理边界防御**：在障碍物挖空时，必须保证 $A_{\text{field}}$ 严格扣除障碍物占位像素（见 Bug 08），确保第 0 步 $\text{CR}(0) \equiv 0.000\%$；
- **离散网格量化误差**：当作业宽度 $W_c$ 与网格分辨率 $r_{\text{grid}}$ 处于同一数量级时，采用像素中心包含判定，避免浮点边缘抖动。

### 1.2 重复覆盖重叠率 (Overlap Rate, $\text{OR}$) 与重叠计数 ($\text{OC}$)
为了衡量作业能量消耗与刀盘磨损的无谓浪费，定义每个像素点的覆盖频次 $N_{\text{covered}}(x, y, t)$。重复覆盖率定义为重复遍历像素占有效农田总面积的比率：
$$\text{OR}(t) = \frac{\sum_{(x,y) \in \mathcal{F}} \max(0, N_{\text{covered}}(x, y, t) - 1)}{A_{\text{field}}}$$
- **重叠计数 ($\text{OC}$)**：在离散网格上，`overlap_count` 记录了小车扫掠四边形 $\text{swept\_quad}_t$ 落在已覆盖像素点上的增量累加值；
- **物理边界修复 (Bug 07)**：当机器人线速度 $v=0$ 原地静止或由于物理碰撞卡滞时，$\text{swept\_quad}_t$ 退化为线段或点（面积为零），此时重叠计数必须短路跳过，杜绝静止状态下重叠计数虚假发散。

### 1.3 门限特征路径长度 (Threshold Path Length, $L_{90}, L_{95}, L_{98}$)
定义小车自规范化起始位姿出发，作业覆盖率首次达到指定门限 $\tau \in \{0.90, 0.95, 0.98\}$ 时小车几何中心累计走过的欧几里得轨迹积分长度：
$$L_{\tau} = \int_{0}^{t_{\tau}} \|\mathbf{v}(t)\|_2 \, dt = \sum_{k=1}^{k_{\tau}} \|\mathbf{p}_k - \mathbf{p}_{k-1}\|_2$$
其中 $k_{\tau} = \min \{k \mid \text{CR}(k) \ge \tau\}$。
- **未达门限定义与统计清洗**：若回合在结束时未达到门限 $\tau$（如发生碰撞或超时），则标记 $L_{\tau} = \text{NaN}$（或 $-1.0$）。在计算数据集聚合统计均值时，必须将其剔除或单列“到达率（Success Rate at $\tau$）”，严禁将 $-1$ 混入均值计算。

### 1.4 几何尺度面积归一化路径长度 (Area-Normalized Path Length, $L_{\tau}^{\text{norm}}$)
由于不同测试农田（Level 1~3，Map 0~39）的有效耕地面积 $A_i$ 存在数倍差异，原始路径长 $L_{\tau}$ 不具备跨地图可比性。定义标准参考农田面积 $\bar{A}_{\text{ref}} = \frac{1}{|\mathcal{S}|} \sum_{i \in \mathcal{S}} A_i$（由场景集均值严格确定，单一事实源）：
$$L_{\tau}^{\text{norm}} = L_{\tau} \cdot \left( \frac{\bar{A}_{\text{ref}}}{A_i} \right)$$
- **第一性原理意义**：消除了大田与小田的固有尺度偏置，直观反映规划算法在单位面积耕地上的作业行进效率。在 $L_{\tau}^{\text{norm}}$ 标尺下，数值越小代表算法的几何覆盖效率越高。

### 1.5 收尾阶段路径消耗比 (Tail Path Ratio, $\rho_{90}, \rho_{95}$)
农田全覆盖任务中最考验算法智能性的是“收尾阶段（Tail Phase）”——当大面积主田块已作业完毕、仅剩边角零星碎块时，算法是陷入局部震荡还是能够精准穿梭收尾：
$$\rho_{95} = \frac{L_{\text{done}} - L_{95}}{L_{\text{done}}} = \frac{\Delta L_{95 \to \text{done}}}{L_{\text{done}}} \in [0.0, 1.0)$$
- **物理判准**：该指标仅在回合正常触发 `env_success`（如达到 98% 终止条件）时具有物理意义。$\rho_{95}$ 越小，说明算法从 95% 冲刺至完赛的路径开销占比越低，长尾收敛能力越强；若 $\rho_{95} > 0.40$，表明算法在收尾阶段出现了盲目搜寻或频繁无效掉头。

### 1.6 转向平滑度与角速度变差 (Angular Roughness, $\Delta\Theta$)
衡量非完整约束（Non-holonomic）车辆转向机构磨损、电机动态平顺度与能量消耗：
$$\Delta\Theta = \sum_{k=1}^T |\omega_k - \omega_{k-1}| \quad \text{或} \quad \Theta_{\text{total}} = \sum_{k=1}^T |\omega_k| \Delta t$$
- 启发式规则（如基于 Dubins 曲线的硬编码转弯）通常呈现阶跃式离散转向，在直线与弧线交界处加速度突变；而良好收敛的 SAC 连续策略则呈现出连续平滑的曲率变化。

### 1.7 回合终态分类学体系 (Episode Termination Taxonomy)
在长周期自动化评估中，回合终止触发器必须实施强正交的分类学管理（见 `rules_new2/metrics_io.py`）：
```
              ┌────────────────────────────────────────────────────────┐
              │             Episode 终态判定流 (Termination Flow)       │
              └────────────────────────────────────────────────────────┘
                                           │
                        ┌──────────────────┴──────────────────┐
                        ▼                                     ▼
                【物理碰撞发生】                      【物理碰撞未发生】
               crashed == True                        crashed == False
                        │                                     │
                        ▼                                     ▼
             done_type: "collision"                ┌──────────┴──────────┐
             done_trigger: "env"                   ▼                     ▼
                                            【任务目标达成】       【任务目标未达成】
                                            finished == True      finished == False
                                                   │                     │
                                                   ▼                     ▼
                                         done_type: "env_success" ┌──────┴──────┐
                                         done_trigger: "env"      ▼             ▼
                                                              【规则规划排空】 【步数/时间超限】
                                                              planner_idle   timeout == True
                                                                   │          或 hit_step_cap
                                                                   ▼                 │
                                                              done_type:             ▼
                                                            "planner_idle"       done_type:
                                                             done_trigger:       "timeout"
                                                               "planner"       done_trigger:
                                                                               "env"/"max_steps"
```

Python 3.12 规范的科学度量参考实现：
```python
"""
metrics_evaluator.py: 科学公理化 CPP 评估指标计算核心实现
"""
from typing import Dict, List, Optional, Tuple
import numpy as np

def compute_cpp_metrics(
    trajectory: np.ndarray,
    coverage_history: List[float],
    angular_velocities: np.ndarray,
    field_area: float,
    ref_field_area: float,
    done_type: str,
    thresholds: Tuple[float, ...] = (0.90, 0.95, 0.98),
) -> Dict[str, float]:
    """计算单回合欧几里得几何与动力学公理化评测指标。

    Args:
        trajectory: (N, 2) 形状的小车中心坐标序列。
        coverage_history: 长度为 N 的覆盖率历史序列。
        angular_velocities: (N,) 形状的角速度序列 (rad/s)。
        field_area: 当前农田有效面积 (像素或 m^2)。
        ref_field_area: 参考基准农田面积。
        done_type: 回合终态分类字符串。

    Returns:
        包含 CR, L_tau, L_tau_norm, rho_95, delta_theta 等的指标字典。
    """
    diffs = np.diff(trajectory, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cum_path = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cum_path[-1])

    results: Dict[str, float] = {
        "final_cr": float(coverage_history[-1]) if coverage_history else 0.0,
        "total_path_length": total_length,
        "area_norm_factor": ref_field_area / max(field_area, 1e-5),
        "angular_roughness": float(np.sum(np.abs(np.diff(angular_velocities)))),
        "total_rotation": float(np.sum(np.abs(angular_velocities))),
        "success": 1.0 if done_type == "env_success" else 0.0,
    }

    cr_array = np.array(coverage_history)
    for tau in thresholds:
        hit_indices = np.where(cr_array >= tau)[0]
        if len(hit_indices) > 0:
            first_idx = hit_indices[0]
            l_tau = float(cum_path[first_idx])
            results[f"L_{int(tau*100)}"] = l_tau
            results[f"L_{int(tau*100)}_norm"] = l_tau * results["area_norm_factor"]
        else:
            results[f"L_{int(tau*100)}"] = -1.0
            results[f"L_{int(tau*100)}_norm"] = -1.0

    if results["success"] > 0.5 and results["L_95"] > 0.0:
        results["rho_95"] = max(0.0, (total_length - results["L_95"]) / max(total_length, 1e-5))
    else:
        results["rho_95"] = -1.0

    return results
```

---

## 2. `rules_new2/` 几何算法架构与启发式规则 vs 强化学习全景博弈

### 2.1 `rules_new2/` 核心几何原语解剖 (`RectifiedFrame`, `WeedTracker`, `PathExecutor`)
查阅 `rules_new2/coverage_planners_v2.py`，其几何算法架构建立在三大轻量级几何原语之上：

1. **`RectifiedFrame`（几何主惯性对齐框架）**：
   - 从环境状态提取外接矩形包围盒，识别长度最大的长边作为本地 $+X$ 轴单位向量 $\mathbf{e}_x$，以正交垂直向量作为 $+Y$ 轴单位向量 $\mathbf{e}_y$，以矩形几何中心为原点 $\mathbf{O}$；
   - 将任意旋转倾斜的农田统一变换至标准对齐坐标系 $(X, Y) \in [X_{\min}, X_{\max}] \times [Y_{\min}, Y_{\max}]$，极大简化了往复割草行的等间距划分与转向点生成。
2. **`WeedTracker`（视锥动态杂草感知追踪器）**：
   - 每步调用底层的 C++ 视锥内核 `cpu_fov_bool`，检测当前进入小车视野视锥的杂草网格；
   - 维护内部存活杂草活跃点集 $\mathcal{W}$，并在割刀触达时动态注销已割除杂草，为上层规划器提供实时的目标吸引源。
3. **`PathExecutor`（纯追踪 Dubins 轨迹跟踪器）**：
   - 将上层规划算法生成的几何航路点序列转化为小车的连续动作 $(v, \omega)$；
   - 在直线行进段采用航向误差比例控制，在换行掉头时通过 `dubins.shortest_path` 生成符合机器人最小转弯半径 $R_{\min} = \frac{v_{\max}}{\omega_{\max}}$ 的时间最优无碰撞圆弧曲线。

### 2.2 五大经典基准规划算法特性分析 (BCP, Jump, Snake, R-Snake, React)
- **`BCPPlannerV2` (Boustrophedon Cellular Decomposition)**：经典牛耕式往复覆盖。沿农田长轴划分宽度为作业幅宽 $B$ 的平行条带，首尾通过 Dubins 曲线衔接。在规则凸多边形中具有极高的几何理论覆盖效率。
- **`JumpPlannerV2` (跳跃式搜索覆盖)**：条带间距放大为视锥弦长 $S_w$，快速粗扫农田；一旦 `WeedTracker` 探测到杂草集群，立即打断主扫描流，通过最短路转移跳入杂草点割草，割完后返回最近扫描行。
- **`SnakePlannerV2` (蛇形连续搜索覆盖)**：连续迂回前进，在行进中动态吸附视野内的零散杂草点。
- **`RestrictedSnakePlannerV2` (约束蛇形覆盖)**：在 `Snake` 基础上严格限制车体转弯包络，防止在边界处因转向半径不足撞击农田外沿。
- **`ReactPlannerV2` (反应式势场覆盖)**：纯局部规划器。利用 C++ APF 势场产生障碍物排斥向量，结合未覆盖区域吸引向量与杂草吸引向量，通过向量叠加实时解算角速度 $\omega$。

### 2.3 专家启发式规则 vs 人工势场 (APF) vs SAC-ResNet 全景博弈矩阵

| 对比维度 | 专家启发式规划器 (`BCP / OPCPP`) | 人工势场反应式规划器 (`React / APF`) | 深度强化学习策略 (`SAC + ResNet-FPN`) |
| :--- | :--- | :--- | :--- |
| **核心数学机制** | 全局凸分解 + 几何长轴投影 + 离散条带划分 + Dubins 曲线拼接 | 梯度下降法：环境人工排斥势场 $\nabla U_{\text{rep}}$ 与目标吸引势场 $\nabla U_{\text{att}}$ 向量叠加 | 深度马尔可夫决策过程：$Q(s, a)$ 贝尔曼最优化、Actor 高斯分布采样与时空特征抽取 |
| **感知输入形态** | 全局静态多边形顶点坐标、全局障碍物地图 | 局部/全局栅格距离场、当前位姿传感器向量 | 多尺度局部特征图（金字塔池化 25 通道）、时空车辙引导、连续位姿状态向量 |
| **对不规则农田适应性** | 差。在凹多边形、不规则狭长死角易产生碎片化子胞腔，衔接路径大幅增长 | 极差。极易陷入局部极小点（Local Minima）或在对称障碍物狭窄通道震荡徘徊 | **优**。通过多尺度 CNN 学习到非局部几何上下文，能自主探索不规则边界与死角 |
| **对动态杂草响应效率** | 僵化。全局规划固定路线，无法动态向视野内新出现的杂草集群机动 | 良好。杂草作为吸引势场可动态牵引车体，但容易因排斥力过大无法接近边缘杂草 | **极佳**。双头网络兼顾农田覆盖与杂草清除，学会在往复作业中顺路消灭杂草的 Pareto 最优动作 |
| **非完整动力学执行** | 严格保证。前向硬编码 Dubins 连续曲率转弯，转弯半径严格 $\ge R_{\min}$ | 弱。在势场突变处可能输出剧烈的角速度阶跃，导致电机物理打滑 | **平滑**。通过连续动作空间高斯策略与加速度平滑惩罚，输出高动态下柔顺控制量 |
| **首要指标：95% 覆盖路径长 ($L_{95}^{\text{norm}}$)** | 规则平原农田下达到理论几何下界；复杂地图下显著劣于 RL | 极长（因局部震荡与频繁绕弯，通常高出 40%~80%） | **复杂非规则地图下最优**（比规则算法缩短 15%~25% 的作业路程） |
| **单步推演耗时** | 规划期 $O(V \log V)$ 一次性解算，步进推演开销极低（$<0.1\text{ms}$） | 单步计算局部梯度，耗时 $<1.0\text{ms}$ | 神经网络前向推理（CPU 约 $2.3\text{ms}$，GPU 批处理 $<0.5\text{ms}$） |

### 2.4 典型农田地图集实测性能与 Pareto 前沿消融对比
在标准测试地图集（Level 1 简单规则凸多边形、Level 2 凹多边形含孤岛障碍、Level 3 复杂狭长农田多障碍死角，每组 40 张地图，随机种子固定）下，通过 `eval_runner.py` 采集的实测聚合基准：

| 算法策略 | Level 1 完赛率 / $L_{95}^{\text{norm}}$ | Level 2 完赛率 / $L_{95}^{\text{norm}}$ | Level 3 完赛率 / $L_{95}^{\text{norm}}$ | 全集平均重叠率 $\text{OR}$ | 平均角速度变差 $\Delta\Theta$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BCPPlannerV2** | **100.0%** / **`1,245.2 m`** | 92.5% / `1,890.4 m` | 75.0% / `2,650.8 m` | `14.2%` | `184.2 rad` (阶跃) |
| **JumpPlannerV2** | 97.5% / `1,310.8 m` | 87.5% / `1,945.0 m` | 70.0% / `2,710.2 m` | `18.5%` | `210.5 rad` (阶跃) |
| **SnakePlannerV2** | 95.0% / `1,380.5 m` | 85.0% / `2,020.1 m` | 67.5% / `2,840.6 m` | `21.2%` | `195.0 rad` (频繁) |
| **RestrictedSnakeV2**| 97.5% / `1,350.2 m` | 87.5% / `1,980.4 m` | 72.5% / `2,780.0 m` | `19.8%` | `176.4 rad` |
| **ReactPlannerV2** | 82.5% / `1,920.6 m` | 65.0% / `2,850.3 m` | 42.5% / `3,940.1 m` | `38.4%` | `342.1 rad` (剧烈) |
| **SAC-ResNet (Ours)** | 97.5% / `1,290.4 m` | **97.5%** / **`1,620.5 m`** | **95.0%** / **`2,045.2 m`** | **`12.6%`** | **`98.5 rad` (平滑)** |

> **博弈结论与系统定位**：
> 1. 在完全规则的凸多边形中（Level 1），`BCPPlannerV2` 能够逼近欧氏几何的理论覆盖下界；
> 2. 在包含凹角、狭长死角与内嵌障碍的复杂真实农田（Level 2 & 3）中，SAC-ResNet 凭借多尺度上下文感知与自主探索能力，完赛率大幅领先传统规则（95.0% vs 75.0%），且归一化路径缩短近 **22.8%**；
> 3. SAC 策略在全集上的重复覆盖率最低（`12.6%`），且转向平滑度显著优于几何折线规则，大幅减轻了物理车体的机械冲击。

---

## 3. 评测硬件矩阵与基准测试环境规范

### 3.1 六大硬件梯队物理规格表 (Local WSL2 至 Cloud G4)
为了全面验证系统优化在不同 CPU 微架构、GPU 算力水平与内存带宽下的鲁棒性，基准压测覆盖了从消费级旗舰工作站到云端顶级超算集群的 6 个硬件梯队：

| 梯队代码 | 平台名称 | CPU 处理器规格 | 物理核心 / 线程 | 宿主机 RAM | GPU 加速卡规格 | GPU 显存 | 存储介质 |
| :---: | :--- | :--- | :---: | :---: | :--- | :---: | :--- |
| **Tier 1** | **本地 WSL2** | Intel Core i9-14900HX | 24C / 32T (8P+16E) | 64 GB DDR5 | NVIDIA RTX 4090 Laptop | 16 GB GDDR6 | PCIe 4.0 NVMe SSD |
| **Tier 2** | **Colab T4 Std** | Intel Xeon @ 2.20GHz | 2 vCPU | 12.7 GB | NVIDIA Tesla T4 | 15 GB GDDR6 | 云端虚拟磁盘 |
| **Tier 3** | **Colab T4 HM** | Intel Xeon @ 2.00GHz | 8 vCPU | 51.0 GB (高配) | NVIDIA Tesla T4 | 15 GB GDDR6 | 云端虚拟磁盘 |
| **Tier 4** | **Colab L4** | Intel Xeon @ 2.20GHz | 12 vCPU | 53.0 GB | NVIDIA Ada L4 | 24 GB GDDR6 | 云端虚拟磁盘 |
| **Tier 5** | **Colab A100** | Intel Xeon @ 2.00GHz | 12 vCPU | 83.5 GB (高配) | NVIDIA A100-SXM4 | 40 GB HBM2 | 高性能云存储 |
| **Tier 6** | **Colab G4** | AMD EPYC 9654 (Zen 5)| 48 vCPU | 180 GB (高配) | NVIDIA RTX PRO 6000 | 96 GB GDDR6 | 高性能 NVMe 挂载 |

### 3.2 压测基准与严谨测量协议 (Methodology)
所有测试均遵循严格的工程基准测试协议：
1. **预热 (Warm-up)**: 每个环境与算子在正式计时前均执行 100 步前置交互，确保 JIT 编译完成、OpenMP 线程池就绪且 OS 缓存预热；
2. **重复测量与统计**: 耗时指标均采集至少 1,000 次步进，计算剔除前 1% 极值的均值与标准差；
3. **环境物理状态锁定**: 随机数种子固定（`seed=42`），地图统一加载标准测试集；
4. **隔离对照验证**: 基于物理独立环境副本与确定性基线分支进行消融对照，杜绝机器背景负载漂移与动态分支穿透干扰。

---

## 4. 环境交互细粒度耗时分解与消融基准

### 4.1 v2, v4, v6 版本单步耗时全景对比 (表 6-1)
表 6-1 记录了在本地 Tier 1 (i9-14900HX) 工作站上，各个版本环境在优化前后的单步 step 耗时与加速比：

*表 6-1: 各环境版本单步推演耗时与加速比对比*
| 环境版本 | 观测空间规格与特性 | 优化前基线耗时 | 优化后耗时 | 单步净节约耗时 | 算子净加速比 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`NewPasture-v2`** | $256 \times 256$ 补丁，基础 5 通道 | `22.53 ms` | **`6.42 ms`** | **`16.11 ms`** | **`3.51×`** |
| **`NewPasture-v4`** | $768 \times 768$ 补丁，多尺度池化 | `36.28 ms` | **`13.95 ms`** | **`22.33 ms`** | **`2.60×`** |
| **`NewPasture-v6`** | $768 \times 768$，轨迹扩散与高斯滤波 | `87.52 ms` | **`26.83 ms`** | **`60.69 ms`** | **`3.26×`** |

> **数据洞见**: 在复杂度最高的 `v6` 环境中，单步执行时间由接近 0.1 秒（`87.52 ms`）直接压减至 **`26.83 ms`**，单步净省长达 **`60.7 ms`**（耗时降低 69.3%），为上层强化学习算法提供了极高频的环境吞吐。

### 4.2 单步计算子模块微基准耗时消融 (9 大算子提速, 表 6-2)
表 6-2 详细展示了环境步进内部各个具体计算子模块的优化前后耗时及技术机理：

*表 6-2: 环境内部核心算子微基准耗时消融分解表*
| 算子/子模块名称 | 核心技术机理 | 基线耗时 (ms) | 优化耗时 (ms) | 净加速比 | 节省时间 (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1. 仿射补丁截取 (`extract_ego_patch`)**| 闭式代数展开，消除 1488×1488 填充 | `25.530` | **`6.470`** | **3.95×** | `19.060` |
| **2. 轨迹 1D 分离高斯滤波** | 1D 可分离卷积 + 局部稀疏 ROI | `15.970` | **`0.380`** | **42.03×** | `15.590` |
| **3. C++ 4 图层 APF 势场计算** | 2D 查找表 + 有界截断 BFS + 释放 GIL | `10.469` | **`1.383`** | **7.57×** | `9.086` |
| **4. 多尺度特征金字塔池化** | 消除第 4 层死池化，单一 PyTorch 视图 | `13.880` | **`5.620`** | **2.47×** | `8.260` |
| **5. 死组件注销 (`EgoHIFUpdater`)** | `v6` 注销未使用的多尺度 HIF 更新器 | `19.340` | **`0.000`** | **彻底消除** | `19.340` |
| **6. 覆盖重叠增量更新 (`Overlap`)** | AABB 包围盒局部切片 + 增量计数 | `0.138` | **`0.016`** | **8.63×** | `0.122` |
| **7. 剩余面积差分更新 (`Field`)** | 机身凸包局部 AABB 差分扣减 | `0.448` | **`0.076`** | **5.89×** | `0.372` |
| **8. 双视锥射线单趟投射 (`FOV`)** | 融合田块视野与迷雾视野为单次光线循环 | `0.138` | **`0.071`** | **1.94×** | `0.067` |
| **9. 障碍物碰撞快速短路 (`Collision`)** | AABB 包围盒空旷快速判定 | `0.034` | **`0.019`** | **1.79×** | `0.015` |
| **静态障碍物边缘 TV 缓存** | `reset()` 预计算，单步零拷贝视图 | `0.095` | **`0.010`** | **9.50×** | `0.085` |

---

## 5. 多进程并发与吞吐扩展性基准 (FPS Scaling)

### 5.1 单环境 vs. 16/32 Worker 吞吐扩展对比 (表 6-3)
在多进程数据收集器 `MultiAsyncCollector` 中，随着工作进程数量从 1 扩展至 32，采样吞吐量呈现出优异的近线性扩展能力。

*表 6-3: 不同 Worker 并发配置下的物理交互吞吐量 (FPS)*
| 并发 Worker 数量 | 基线吞吐量 (FPS) | 优化后吞吐量 (FPS) | 纯并发提速比 |
| :---: | :---: | :---: | :---: |
| **1 Worker (单环境串行)** | `44.4 FPS` | **`150.1 FPS`** | **3.38×** |
| **4 Workers (并发)** | `168.2 FPS` | **`562.4 FPS`** | **3.34×** |
| **8 Workers (并发)** | `312.5 FPS` | **`1,048.0 FPS`** | **3.35×** |
| **16 Workers (并发)** | `580.4 FPS` | **`1,892.3 FPS`** | **3.26×** |
| **32 Workers (全核并发)** | `842.1 FPS` | **`2,714.5 FPS`** | **3.22×** |

### 5.2 线程锁控 (`OMP_NUM_THREADS=1`) 吞吐提速曲线
在 16 个 Worker 的基准压力测试中：
- **未锁定线程池 (默认线程超额订阅)**: OpenCV 与 PyTorch 内部派生 16 个线程，总线程数达 256。由于 CPU 缓存颠簸与自旋锁争抢，系统总吞吐量仅为 **`128.4 FPS`**；
- **锁定单线程 (`OMP_NUM_THREADS=1, threads=1`)**: 消除全部线程上下文切换，总吞吐量跃升至 **`227.1 FPS`**（**净提升 `1.77×`**，且 CPU 占用率稳定平直）。

---

## 6. 内存层次、显存与存储削减指标

### 6.1 显存底噪消除 (立省 7.65GB 显存，`policy_device='cpu'`)
在启动 32 个异步采样 Worker 时，显存占用发生断崖式下降：
- **基线配置 (`policy_device='cuda:0'`)**:
  32 个独立 Python 进程各自初始化 CUDA Context，仅驱动底噪就吃掉 $32 \times 240\text{ MB} \approx 7.68\text{ GB}$。叠加主进程模型权重与训练缓存，显存总占用高达 **`11.52 GB`**；
- **优化配置 (`policy_device='cpu'`)**:
  子进程完全不触碰 CUDA 驱动，仅由主进程持有 GPU。显存总占用降至 **`3.87 GB`**；
- **实测净收益**: **立省 `7.65 GB` 显存（降幅达 `-66.4%`）**，释放出的显存允许研究人员将训练 Batch Size 提升至 4096 或容纳更深的网络骨架。

### 6.2 瞬态分配归零与评测 OOM 消除 (79.80MB $\to$ 0.00MB)
- **单步瞬态内存分配峰值**:
  - 基线三阶段图像处理: 每步申请释放 **`79.80 MB`** 内存；
  - 闭式直接仿射投影: 瞬态数组申请严格降为 **`0.00 MB`**（彻底消除内存锯齿波）；
- **全套评测套件内存占用 (192 回合)**:
  - 基线全量密集存储: 单回合 720MB，累计申请 **`138 GB` 内存**，频繁引发系统 OOM 崩溃；
  - 有界快照机制: 单回合内存降至 **`< 5 MB`（缩减 `>99%`）**，常驻内存稳定在 2GB 以内。

### 6.3 经验池存储削减与 PCIe H2D 传输加速 (51.5GB 净省，3.67× 加速)
针对容量为 50 万步（$500\text{k}$）的经验回放池（ReplayBuffer）：
- **剔除未读次态标签 (`("next", "label_ego_hif")`)**:
  经验池磁盘存储体积由 `274.72 GB` 降为 **`223.22 GB` (立省 `51.50 GB` 存储空间)**；
- **UINT8 归一化量化存储**:
  观测张量存储体积由 `102.4 GB`（FP32）暴跌至 **`25.6 GB` (直降 `75.0%`)**；
- **Host-to-Device (H2D) PCIe 传输速度**:
  结合锁页内存（Pinned Memory），抽取批大小 2048 的样本传输至 GPU 显存：
  - 基线可分页 FP32 传输耗时: `21.59 ms`；
  - 锁页 UINT8 异步传输 + GPU 原位反量化耗时: **`5.88 ms`**；
  - **PCIe 数据加载速度提升 `3.67×`**！

---

## 7. 跨硬件端到端全流程训练收益全景表

### 7.1 400 万帧全量训练物理时长对比 (表 6-4, FP16 AMP 修正)
表 6-4 汇总了在 6 大硬件平台上，完成 400 万帧（4M steps）标准 SAC 强化学习全量训练的实测耗时与吞吐对比。

> **重要修正说明 (D08 对齐)**：原稿此处误写作“BF16 AMP”。根据 Volume 08 决策 D08，ResNet-FPN 网络骨架中的 `nn.BatchNorm2d` 遇到 bfloat16 时存在底层的数值溢出与方差不稳定缺陷，因此本项目全流程严格锁定 **FP16 AMP（`torch.float16` 配合 `torch.amp.GradScaler`）**，在确保数值单调收敛的前提下释放 Tensor Core 算力。

*表 6-4: 六大硬件平台 400 万帧端到端训练耗时与加速指标全景表*
| 硬件平台梯队 | 平台硬件规格 | 基线单步更新耗时 (bs=2048) | 优化后单步更新 (FP16 AMP) | 基线端到端 E2E FPS | 优化后端到端 E2E FPS | 400万帧总耗时 (优化前 → **优化后**) | 全流程净加速比 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tier 1 (本地 WSL2)** | i9-14900HX + RTX 4090 | `196.7 ms` | **`58.0 ms`** | `92.3 FPS` | **`290.5 FPS`** | `12.0 小时` → **`3.8 小时`** | **`3.16×`** |
| **Tier 2 (Colab T4 Std)**| 2 vCPU + Tesla T4 | `3023.8 ms` | **`976.9 ms`** | `5.2 FPS` | **`16.8 FPS`** | `213.0 小时` → **`66.1 小时`** | **`3.23×`** |
| **Tier 3 (Colab T4 HM)** | 8 vCPU + Tesla T4 | `2819.7 ms` | **`814.3 ms`** | `38.6 FPS` | **`115.0 FPS`** | `28.8 小时` → **`9.7 小时`** | **`2.98×`** |
| **Tier 4 (Colab L4)** | 12 vCPU + NVIDIA L4 | `824.9 ms` | **`605.9 ms`** | `72.7 FPS` | **`215.0 FPS`** | `15.3 小时` → **`5.2 小时`** | **`2.96×`** |
| **Tier 5 (Colab A100)** | 12 vCPU + A100-40GB | `385.3 ms` | **`288.9 ms`** | `98.2 FPS` | **`275.0 FPS`** | `11.3 小时` → **`4.0 小时`** | **`2.80×`** |
| **Tier 6 (Colab G4)** | EPYC 48C + RTX PRO 6000| `150.4 ms` | **`111.5 ms`** | `219.1 FPS` | **`640.0 FPS`** | `5.1 小时` → **`1.7 小时`** | **`2.92×`** |

### 7.2 核心工程洞见：为什么本地 4090 能够反超未优化云端超算
在深入分析表 6-4 中的实测数据时，可以得出一个极具震撼力的**现代强化学习系统性能结论**：

> **核心结论**:  
> **未优化前**，盲目租用昂贵的顶级云计算集群（如 Colab A100 或 G4）的端到端吞吐率仅为 **`98.2 FPS`** 与 **`219.1 FPS`**，因为 **超过 75% 的训练时间卡在 CPU 单核图像 Padding 内存分配、C++ 无界全图 BFS、千线程争抢以及每 20 帧一次的跨进程管道锁死上**！顶级 GPU 长期处于“饥饿等待”状态。  
> 
> **在实施本套系统工程优化后**:  
> **本地常规工作站（i9-14900HX + RTX 4090）的端到端吞吐率即可达到 `290.5 FPS`，全量 400 万帧仅需 3.8 小时，其运行速度直接超越了未优化前顶级的云端 A100 集群（11.3 小时）！**  

这充分证明：在复杂的物理强化学习与具身智能系统中，**算法底层与系统架构的精准工程优化，比单纯盲目堆砌昂贵硬件具有高得多的生产力价值**。

---

## 8. 附录：评测指标与性能工程证伪防坑指南 (Falsification & Anti-Pattern Appendix)

### 8.1 证伪一：为什么朴素覆盖率与回合总奖励无法区分高效扫掠与无目的游荡
- **表象诱惑**：许多强化学习论文仅汇报 Episode 终止时的最终覆盖率 $\text{CR}_{\text{final}}$ 或标量累积总奖励 $R_{\text{total}}$，认为只要最终覆盖率达到 95% 就代表策略有效；
- **第一性原理证伪**：
  1. **盲目游荡与时间稀释**：一个缺乏全局几何视野的差劲策略，在农田中央反复徘徊、经历长达 3,000 步的剧烈重叠与布朗运动后，同样可能偶然使绝大部分像素变绿（达到 95%）；
  2. **累积奖励虚假膨胀**：如果奖励函数中包含存活奖励或未惩罚重复覆盖步数，徘徊 3,000 步的策略累积奖励甚至可能高于 800 步高效覆盖完毕并触发完赛的策略；
  3. **指标防坑判准**：必须引入门限路径长 $L_{95}^{\text{norm}}$ 与收尾路径比 $\rho_{95}$。高效策略的特征是 $L_{95}^{\text{norm}}$ 趋近理论几何下界，且 $\rho_{95} < 0.15$；而游荡策略的 $L_{95}^{\text{norm}}$ 会膨胀数倍，$\rho_{95}$ 往往超过 0.40。

### 8.2 证伪二：为什么 bfloat16 在 ResNet-FPN BatchNorm2d 下引发数值下溢与方差爆炸
- **表象诱惑**：部分前沿大模型研究者习惯性地认为“现代 GPU 推荐 bfloat16，动态范围大，无需 GradScaler，能省代码”；
- **第一性原理证伪 (Volume 08 D08 事实)**：
  1. **尾数精度劣势**：bfloat16 仅有 7 位尾数（Mantissa），而 float16 具有 10 位尾数。在深层 CNN 卷积与特征金字塔中，梯度的相对舍入误差高达 $2^{-7} \approx 7.8 \times 10^{-3}$；
  2. **BatchNorm2d 内部统计崩溃**：`torch.nn.BatchNorm2d` 在计算运行方差 $\sigma^2 = \frac{1}{M}\sum (x_i - \mu)^2$ 时，要求极高的尾数累加精度。原作者在 `sac_utils_optimized.py:186` 明确标注：`# 目前该函数有问题，只转换了数据的BF16, 但遇到ResNet的BatchNorm2D的时候会因为模型中的FP32出问题，暂时弃用`；
  3. **实测表现**：强制启用 bfloat16 会导致 BatchNorm2d 归一化后的特征方差剧烈震荡，Critic 网络的 Q 预测值在训练至 50k 步时出现 NaN 爆炸；
  4. **正确修法**：坚决锁定 `torch.float16` 并配合 `torch.amp.GradScaler`，实现无抖动平稳加速。

### 8.3 证伪三：为什么在环境微基准中忽略瞬态内存分配与 GC 压力会导致严重的系统级性能误判
- **表象诱惑**：在单步算子评测时，只用 `time.perf_counter()` 测算 10 次单次执行耗时，看到 `extract_ego_patch` 耗时约 25ms 便认为“可以接受”；
- **第一性原理证伪**：
  1. **单步 vs 并发累积**：单步申请 79.80MB 内存看似不大，但在 32 个 Worker 并发采样下，每秒总共在 Python 堆上申请释放超 **`6.4 GB`** 的瞬态数组；
  2. **垃圾回收（GC）Stop-the-World**：瞬态数组迅速填满 Gen 0/1/2 垃圾回收池，频繁触发 Python 的全局垃圾回收。在多进程 IPC 交互管道中，GC 停顿导致 Worker 无法及时向主进程推送数据，引发 `BrokenPipeError` 或严重的队列阻塞抖动；
  3. **防坑准则**：在性能工程中，必须将“单步瞬态堆分配（Transient Heap Allocation）”作为硬性度量指标，通过闭式直接仿射投影等技术手段实现单步瞬态分配严格归零（`0.00 MB`）。

### 8.4 证伪四：为什么无面积归一化的原始路径长 $L_{\tau}$ 会破坏跨地图评价的统计有效性
- **表象诱惑**：在评测脚本中直接计算所有测试地图的平均路径长 $\bar{L}_{95} = \frac{1}{M}\sum_{i=1}^M L_{95}^{(i)}$；
- **第一性原理证伪**：
  1. **大田对均值的绝对支配**：在 NewCppRL 场景集中，小地块面积约 $400\text{ m}^2$，大地块面积超 $2,500\text{ m}^2$（相差超 6 倍）。原始路径长与田块面积呈强正相关；
  2. **统计失真**：若算法在大田上稍有失误（多走了 100m），其对全局平均值的影响会彻底掩盖其在 5 个小田上的卓越表现，导致评估结论被大田主导；
  3. **防坑准则**：必须通过 $L_{\tau}^{\text{norm}} = L_{\tau} \cdot \left(\frac{\bar{A}_{\text{ref}}}{A_i}\right)$ 将所有地图的作业路径投影到同一基准尺度下，才能进行公平的算法横向对比。
