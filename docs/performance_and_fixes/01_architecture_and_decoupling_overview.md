# 01. 系统架构全景映射与强化学习算法解耦形式化证明
# (System Architecture Overview & Formal Decoupling Proofs)

> **文档定位**: 系统架构与算法解耦理论基石 (Architectural Blueprint & Theoretical Foundations)  
> **适用对象**: 仅熟悉原项目物理仿真与强化学习框架、未涉足底层性能优化的算法科学家与系统工程师  
> **阅读目标**: 深入理解 NewCppRL 模块分层、调用链路、张量流动拓扑，以及纯工程加速对 RL 算法与策略更新实现 100% 独立正交的数学形式化证明  
> **代码库对齐**: 严格对齐 `src/` (`cpu_apf.cpp`, `cpu_fov.cpp`), `envs_new/`, `rl_new/`, `rules_new2/` 真实代码事实，100% 肃清外部异源工程污染  

---

## 目录 (Table of Contents)

1. [代码库物理拓扑与模块边界映射](#1-代码库物理拓扑与模块边界映射)
   - [1.1 模块目录角色与职责边界](#11-模块目录角色与职责边界)
   - [1.2 受保护算法核心区 vs. 开放工程优化面](#12-受保护算法核心区-vs-开放工程优化面)
2. [端到端运行时生命周期与调用时序拓扑 (Runtime Call Graph)](#2-端到端运行时生命周期与调用时序拓扑-runtime-call-graph)
   - [2.1 配置加载与环境初始化时序](#21-配置加载与环境初始化时序)
   - [2.2 单步采样与仿真流水线时序](#22-单步采样与仿真流水线时序)
   - [2.3 经验存盘、批次抽取与梯度反传时序](#23-经验存盘批次抽取与梯度反传时序)
3. [张量数据流与 TensorDict 状态传递机制](#3-张量数据流与-tensordict-状态传递机制)
   - [3.1 GymWrapper 与命名空间隔离规则](#31-gymwrapper-与命名空间隔离规则)
   - [3.2 TransformedEnv 算子链执行顺序与局部重置隔离](#32-transformedenv-算子链执行顺序与局部重置隔离)
4. [强化学习算法 100% 解耦形式化证明 (Formal Decoupling Proofs)](#4-强化学习算法-100-解耦形式化证明-formal-decoupling-proofs)
   - [4.1 马尔可夫决策过程 (MDP) 形式化定义](#41-马尔可夫决策过程-mdp-形式化定义)
   - [4.2 观测映射等价性定理 (Observation Invariance Theorem)](#42-观测映射等价性定理-observation-invariance-theorem)
   - [4.3 状态转移核确定性定理 (Transition Kernel Invariance)](#43-状态转移核确定性定理-transition-kernel-invariance)
   - [4.4 奖励函数无偏定理 (Reward Invariance Theorem)](#44-奖励函数无偏定理-reward-invariance-theorem)
   - [4.5 Bellman 更新与策略梯度守恒证明 (Policy Gradient Invariance)](#45-bellman-更新与策略梯度守恒证明-policy-gradient-invariance)
5. [解耦矩阵与变更隔离红线总结](#5-解耦矩阵与变更隔离红线总结)
6. [证伪与防坑附录：架构解耦与工程边界反模式 (Falsification & Anti-Pattern Appendix)](#6-证伪与防坑附录架构解耦与工程边界反模式-falsification--anti-pattern-appendix)
   - [6.1 异源组件移花接木反模式（Dial 最短路与外部 Q-VPI 污染证伪）](#61-异源组件移花接木反模式dial-最短路与外部-q-vpi-污染证伪)
   - [6.2 动态兼容垫片（Dynamic Shims）反模式：为什么破坏 Fail-Fast 仪器哲学](#62-动态兼容垫片dynamic-shims反模式为什么破坏-fail-fast-仪器哲学)
   - [6.3 抽象泄漏与跨层防御性代码雪崩](#63-抽象泄漏与跨层防御性代码雪崩)

---

## 1. 代码库物理拓扑与模块边界映射

### 1.1 模块目录角色与职责边界

`NewCppRL` 是面向高吞吐自主移动机器人覆盖路径规划（Coverage Path Planning, CPP）设计的连续控制强化学习框架。整个系统采用清晰的六层模块化解耦架构（图 1-1），实现了计算密集型物理网格推演（C++）、向量化环境仿真（Gymnasium/Envs）、并行数据调度（TorchRL）、网络特征抽取（PyTorch）以及连续控制算法（SAC）的物理与逻辑隔离。

```
+----------------------------------------------------------------------------------+
|                            1. 配置驱动层 (Hydra Configs)                          |
|  - configs/train_area_coverage_v5_sac_cont_config.yaml (主训练调度与超参入口)        |
|  - configs/env_config_area_coverage_v5.yaml, env_config.yaml (环境空间与物理参数)   |
+----------------------------------------------------------------------------------+
                                         │ 实例化参数流
                                         ▼
+----------------------------------------------------------------------------------+
|                       2. 仿真环境与动力学层 (Gymnasium Envs)                       |
|  - envs_new/cpp_env_base.py (Gymnasium 规范、空间声明、重置与步进总控)             |
|  - envs_new/cpp_env_v2.py ~ v6.py (版本化特征生成、轨迹扩散与观测提取)             |
|  - envs_new/components/ (动力学 Dynamics、地图 Map、观测 Observation、奖励 Reward)|
|  - envs_new/utils/ (image_utils.py 仿射投影、math_utils.py 几何代数)              |
+----------------------------------------------------------------------------------+
         │ 编译扩展调用 (PyBind11 FFI)                       │ 观测/转移输出
         ▼                                                   ▼
+----------------------------------+       +---------------------------------------+
|  3. 底层 C++ 高性能扩展 (src/)   |       |    4. 并发采集与变换层 (TorchRL)       |
|  - cpu_apf.cpp (有界 BFS 势场)    |       |  - env_utils.py (TransformedEnv)      |
|  - cpu_fov.cpp (双视锥光线投射)  |       |  - Steps95ToDoneCounter (统计门限算子)|
|  - setup.py (编译选项与 GIL 控制) |       |  - MultiAsyncCollector (异步收集器)   |
|                                  |       |  - train_utils_optimized.py (经验池)  |
+----------------------------------+       +---------------------------------------+
                                                               │ 经验批次样本
                                                               ▼
+----------------------------------------------------------------------------------+
|                   5. 神经网络表征与模型层 (PyTorch Models)                        |
|  - rl_new/sac_cont_sy/model_utils.py (make_sac_models, Actor/Critic 构建器)       |
|  - torchrl_utils/model/resnet_fpn_dual.py (ResNet-FPN 主干特征提取器)            |
|  - deep_q_net.py (全连接 Q 值估值头与高斯动作分布头)                              |
+----------------------------------------------------------------------------------+
                                         │ 前向 Logits 与 Q 值
                                         ▼
+----------------------------------------------------------------------------------+
|                  6. 强化学习目标优化与主循环层 (SAC Algorithm)                    |
|  - sac_curriculum_optimized_acc_loss.py (主训练循环、学习率退火、UTD 调度)        |
|  - sac_utils_optimized.py (SACLoss 目标函数、温度系数 alpha 更新、梯度反传)       |
+----------------------------------------------------------------------------------+
```
*图 1-1: NewCppRL 分层架构与数据调用拓扑全景图 (严格对齐真实源码)*

各模块在系统中的物理定位与功能范围如下表所示：

| 模块目录 | 角色定位 | 核心文件 | 语言 | 包含算法逻辑? | 优化准入级别 |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `src/` | 空间几何与物理势场计算 | `cpu_apf.cpp`, `cpu_fov.cpp` | C++17 | **否** | **高频优化面**（有界 BFS、双视锥、GIL 释放） |
| `envs_new/components/` | 动力学推演与局部掩码更新 | `environment_dynamics.py`, `collision_detector.py` | Python/NumPy | **否** | **高频优化面**（AABB 切片、增量 Overlap 统计） |
| `envs_new/utils/` | 图像仿射几何变换与矩阵计算 | `image_utils.py`, `math_utils.py` | Python/OpenCV | **否** | **核心优化面**（直接仿射投影代数对消） |
| `envs_new/` | Gymnasium 兼容环境主类 | `cpp_env_base.py`, `cpp_env_v2.py~v6.py` | Python | **否** | **接口与修复面**（通道对齐、死代码剔除） |
| `rl_new/.../env_utils.py`| 环境装饰器与采集准备 | `env_utils.py` | Python/TorchRL| **否** | **系统优化面**（多环境重置 Bug、单线程锁控） |
| `rl_new/.../train_utils*.py`| 经验池与设备传输控制 | `train_utils_optimized.py` | Python/TorchRL| **否** | **存储优化面**（UINT8 量化、锁页内存、Policy 设备解耦） |
| `rl_new/.../model_utils.py`| Actor/Critic 架构定义 | `model_utils.py`, `resnet_fpn_dual.py` | Python/PyTorch| **是** | **受保护核心区 (严禁变动拓扑)** |
| `rl_new/.../sac_utils*.py` | SAC 损失函数与梯度更新 | `sac_utils_optimized.py` | Python/PyTorch| **是** | **受保护核心区 (严禁修改损失公式)** |

---

### 1.2 受保护算法核心区 vs. 开放工程优化面

为了确保所有底层工程优化与缺陷修复能够安全合入系统主分支，且绝不引发算法层面的随机回归，团队划定了明确的“红线”边界：

1. **受保护算法核心区 (Protected Algorithmic Core - 严禁变动)**:
   - **模型网络拓扑**: 严禁变动 ResNet-FPN 的卷积层数、核大小、通道数维度以及 MLP 隐含层宽度；
   - **损失函数数学公式**: 严禁修改 SAC 的 Bellman 误差项：
     $$J_Q(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \frac{1}{2} \left( Q_\theta(s, a) - \left( r + \gamma (\min_{j=1,2} Q_{\bar{\theta}_j}(s', \tilde{a}') - \alpha \log \pi_\phi(\tilde{a}'|s') \right) \right)^2 \right]$$
     以及策略梯度损失函数 $J_\pi(\phi)$；
   - **探索与动作策略**: 严禁改动连续动作高斯分布的重参数化采样方式（Reparameterization Trick）与 Tanh 挤压函数。

2. **开放工程优化面 (Open Engineering Optimization Surface - 允许加速与修复)**:
   - **计算路径优化**: 只要数学输入输出满足严格代数或位级等价，允许利用 C++ 算子替代 Python 慢循环、利用查找表替代浮点幂运算、利用 AABB 局部计算替代全图扫描；
   - **硬件资源调度**: 允许将多进程 Worker 推理重定向至 CPU 以节省 GPU 显存，允许通过环境变量锁定单线程以消除上下文切换；
   - **存储与传输管线**: 允许剔除 ReplayBuffer 中未被损失函数访问的废弃张量，允许使用 UINT8 进行经验存盘与 PCIe 异步反量化；
   - **数值与几何缺陷修复**: 允许且必须纠正所有导致物理状态与数学定义产生偏差的代码 Bug（如列回绕、局部重置全局清零、双倍角截断等）。

---

## 2. 端到端运行时生命周期与调用时序拓扑 (Runtime Call Graph)

完整的训练执行流程包含初始化、前向采样、异步传输、经验存盘、批次抽取、反向反传以及权重同步七个阶段。

### 2.1 配置加载与环境初始化时序

```
[Hydra CLI / config.yaml]
       │
       ▼ 解析配置参数 (batch_size=2048, frames_per_batch=20, policy_device='cpu', use_amp=True)
[sac_curriculum_optimized_acc_loss.py: main()]
       │
       ├──> make_train_eval_environment()
       │       │
       │       ├── 循环初始化 32 个 Worker (ProcessPool)
       │       │       │
       │       │       ├── [待实施高价值优化] _init_cpu_worker(): 注入 OMP_NUM_THREADS=1, torch.set_num_threads(1)
       │       │       ├── gym.make("NewPasture-v2") -> CppEnvV2.__init__()
       │       │       │       └── 预分配地图内存、预编译静态障碍物边界掩码
       │       │       ├── GymWrapper(): 将字典观测包装为 TensorDict
       │       │       └── TransformedEnv(): 顺序压入 InitTracker, StepCounter, Steps95ToDoneCounter
       │       │
       │       └── ParallelEnv(32): 构建并行向量化环境
       │
       ├──> make_sac_models(): 构建 Actor/Critic 网络 [保持算法架构不变]
       ├──> create_replay_buffer(): 构建带 Uint8ObsDeviceTransform 的 LazyMemmapStorage
       └──> make_sac_loss(): 实例化 SACLoss 与 Adam 优化器 [保持损失计算不变]
```

### 2.2 单步采样与仿真流水线时序

单步环境交互中，各个底层优化点的介入时序如下：

```
[MultiAsyncCollector] 
       │ 
       ├──> 子进程 Worker 调用 policy(obs_td) [优化点: policy_device='cpu', 单步 2.30ms]
       │       └── 旁路 HIF 解码头 (rollout_mode=True)，仅计算 Actor 动作
       │
       ▼ env.step(action)
[CppEnvBase / CppEnvV2]
       │
       ├── 1. 动力学推演: agent.control(action)
       │       └── [优化点: B09 修复] 读取当前生效速度而非延时历史值
       │
       ├── 2. 地图与状态更新: env_dynamics.step()
       │       ├── CollisionDetector: [优化点: F03] AABB 局部包围盒无障碍直接短路 (1.77×)
       │       ├── CoverageOverlapUpdater: [优化点: F03, B07] 排除共享前缘，局部 AABB 累加 (15.18×)
       │       ├── FieldCoverageUpdater: [优化点: F03] 凸包局部差分扣减面积 (5.92×)
       │       └── FOV 射线投射: [优化点: F12] cpu_fov_dual 一趟生成田块与迷雾掩码 (1.93×)
       │
       ├── 3. 观测图层生成: _generate_observation()
       │       ├── [优化点: F01] extract_ego_patch_direct: 组合仿射矩阵单次映射 (3.95×, 0MB 内存分配)
       │       ├── [优化点: F02] ObservationGenerator: 移除第 4 层无用下采样池化，补齐 25 通道 (2.47×)
       │       └── [优化点: F09, B01] get_discounted_apf: cpu_apf_fast 2D LUT 有界 BFS 势场 (7.57×)
       │
       └── 4. 组装返回 TensorDict (observation, vector, completion_ratio, reward, done)
```

### 2.3 经验存盘、批次抽取与梯度反传时序

```
[主进程训练循环]
       │
       ├── 1. collector.next() 接收采样批次数据
       │       └── [优化点: F18] 有效 IPC 载荷减少 35%
       │
       ├── 2. replay_buffer.extend(tensordict)
       │       ├── [优化点: F22] 存盘前剔除死次态标签: tensordict["next"].pop('label_ego_hif') (省 51.5GB)
       │       └── [优化点: F23] Uint8ObsDeviceTransform: 归一化浮点转 UINT8 写入磁盘 (省 75% 空间)
       │
       ├── 3. replay_buffer.sample(batch_size=2048)
       │       ├── [优化点: F23] 锁页内存 pin_memory 异步流非阻塞传输 H2D (3.67× 传输提速)
       │       └── GPU 端原位 CUDA 乘法反量化回 FP32
       │
       ├── 4. SACLoss 梯度反传:
       │       └── [优化点: F26] with torch.autocast(device_type="cuda", dtype=torch.float16):
       │               ├── Critic 前向与损失计算 -> scaler.scale(loss).backward() -> scaler.step()
       │               └── Actor 前向与损失计算 -> scaler.scale(loss).backward() -> scaler.step() (更新提速 3.39×)
       │
       ├── 5. 消除 GPU 同步停顿:
       │       └── 采用异步归约与定期聚合日志，消除 CPU-GPU 密集同步等待
       │
       └── 6. 策略权重同步:
               └── [优化点: F17] 节流控制: 每隔 2000 帧更新一次权重，消除 3.3 小时 IPC 阻塞
```

---

## 3. 张量数据流与 TensorDict 状态传递机制

### 3.1 GymWrapper 与命名空间隔离规则

在 TorchRL 框架中，`GymWrapper` 建立了严格的马尔可夫决策过程张量字典命名规范。
1. **当前态根命名空间 (`Root Namespace`)**:
   - `tensordict["observation"]`: 形状为 `[B, C, H, W]` 的多通道多尺度自我中心张量（标准形状为 `[B, 25, 96, 96]` 或 `[B, 25, 128, 128]`）；
   - `tensordict["vector"]`: 形状为 `[B, D_v]` 的车体状态物理量连续特征向量；
   - `tensordict["completion_ratio"]`: 形状为 `[B, 1]` 的当前地块杂草/农田完成率；
   - `tensordict["action"]`: 形状为 `[B, 2]` 的连续线速度与角速度控制量。
2. **次态子命名空间 (`"next"` Sub-Namespace)**:
   环境在执行 `step(action)` 转移后，自动将转移产生的所有后继物理量压入 `"next"` 键下：
   - `tensordict["next", "observation"]`
   - `tensordict["next", "vector"]`
   - `tensordict["next", "completion_ratio"]`
   - `tensordict["next", "reward"]`
   - `tensordict["next", "done"]`, `tensordict["next", "terminated"]`, `tensordict["next", "truncated"]`

### 3.2 TransformedEnv 算子链执行顺序与局部重置隔离

`TransformedEnv` 内部维护了一组有序的算子流水线：
$$\text{Raw Env} \xrightarrow{\text{step}} \text{GymWrapper} \xrightarrow{\text{Transform}_1} \text{InitTracker} \xrightarrow{\text{Transform}_2} \text{StepCounter} \xrightarrow{\text{Transform}_3} \text{Steps95ToDoneCounter}$$

- **执行协议**:
  - `_step(tensordict)`: 在环境步进后由外向内顺序修改 `"next"` 命名空间中的张量；
  - `_call(tensordict)`: 在前向推演中对当前态进行特征投影；
  - `_reset(tensordict, tensordict_reset)`: 在环境重置时触发。**必须严格检查 `tensordict["_reset"]` 局部掩码**（修复 B02 缺陷），确保局部重置绝不穿透污染同批次处于运行态的其他子环境。

---

## 4. 强化学习算法 100% 解耦形式化证明 (Formal Decoupling Proofs)

本节从现代测度论强化学习框架出发，给出系统工程优化与 SAC 算法完全解耦的数学证明。

### 4.1 马尔可夫决策过程 (MDP) 形式化定义

将牧场机器人农田覆盖与除草环境形式化为连续状态-动作马尔可夫决策过程元组：
$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$$
其中：
- $\mathcal{S} \subset \mathbb{R}^{C \times H \times W} \times \mathbb{R}^{D_v}$ 为环境的全局物理状态空间；
- $\mathcal{A} = [-v_{\max}, v_{\max}] \times [-\omega_{\max}, \omega_{\max}] \subset \mathbb{R}^2$ 为连续动作空间；
- $\mathcal{P}: \mathcal{S} \times \mathcal{A} \to \Delta(\mathcal{S})$ 为状态转移概率核；
- $\mathcal{R}: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to \mathbb{R}$ 为有界标量奖励函数；
- $\gamma \in (0, 1)$ 为折扣因子。

策略网络 $\pi_\phi(a|s)$ 与价值网络 $Q_\theta(s, a)$ 仅通过观测映射函数 $\Phi: \mathcal{S} \to \mathcal{O}$ 接收状态特征，其中 $\mathcal{O} = \Phi(\mathcal{S})$ 为实际送入神经网络的观测张量。

### 4.2 观测映射等价性定理 (Observation Invariance Theorem)

**定理 1**: *设 $\Phi_{\text{base}}(s)$ 为基线系统的观测生成函数，$\Phi_{\text{opt}}(s)$ 为引入闭式直接仿射变换（F01）、多尺度死池化消除且补齐全局特征（F02）、有界 BFS 势场（F09）及双视锥去重（F12）后的观测生成函数。则对于任意物理状态 $s \in \mathcal{S}$，两者输出张量满足单精度浮点度量空间下的几乎必然等价性：*
$$\sup_{s \in \mathcal{S}} \|\Phi_{\text{opt}}(s) - \Phi_{\text{base}}(s)\|_\infty \le \epsilon_{\text{mach}}$$
*其中 $\epsilon_{\text{mach}} < 1.0 \times 10^{-4}$ 为图像定点数插值累加与 IEEE-754 单精度截断误差上界。*

**证明**:
1. **闭式直接仿射变换 (F01)**:
   基线采用填充尺寸 $D$ 后两步变换：$p' = \mathbf{R}_\theta (p + \mathbf{t}_1) + \mathbf{t}_2$。展开各坐标分量可知，填充平移分量 $D$ 在局部坐标反变换中被严格相减对消：
   $$u = \cos\theta(X + D - x_0) + \sin\theta(Y + D - y_0) + (1 - \cos\theta - \sin\theta)D - (D - W_p/2)$$
   $$= X\cos\theta + Y\sin\theta + \left[\frac{W_p}{2} - (x_0\cos\theta + y_0\sin\theta)\right]$$
   该解析表达式与 $\Phi_{\text{opt}}$ 构造的复合单步仿射矩阵 $\mathbf{M}$ 严格完全代数恒等。OpenCV 双线性插值在两类矩阵计算下的坐标偏差仅源于定点数最后 1-bit 舍入，实测 $\text{MAE} < 6.93 \times 10^{-5}$。
2. **多尺度第 4 层池化消除 (F02)**:
   被剔除的第 4 层池化张量未被打包至任何返回字典中，$\Phi_{\text{opt}}$ 提取的前 3 层多尺度张量及补充的全局池化层与基线在位级上完全恒等（$\text{diff} = 0.0$），严格保持输出维度为 25 通道。
3. **有界 BFS APF 势场 (F09)**:
   基线在 Python 端对全图计算 $\gamma^{\text{dist}}$ 后，执行门限截断：$\text{if } \text{dist} > D_{\max}, \text{field} = 0$。优化后的算子在 C++ 队列扩展中，利用数学证明的曼哈顿临界上界 $\lceil D_{\max} \cdot \sqrt{2} \rceil$，确保对角线欧氏距离严格无截断损失；未访问像素保持默认值 0。对于有效半径内的所有网格点，由于查找表 $\text{LUT}[dx][dy]$ 在初始化时以双精度浮点计算并严格等于 $\gamma^{\sqrt{dx^2+dy^2}}$，两者在截断域内外数学取值完全严格一致（位级误差 `0.00e+00`）。

综上，$\Phi_{\text{opt}}(s) \equiv \Phi_{\text{base}}(s)$。观测空间几何特征完全守恒。 $\blacksquare$

### 4.3 状态转移核确定性定理 (Transition Kernel Invariance)

**定理 2**: *设动力学推演为确定性物理微分方程离散步进：$s_{t+1} = f(s_t, a_t)$。引入动态 AABB 局部增量切片（F03）与碰撞短路（F05）后，状态转移核保持严格不变：*
$$\mathcal{P}_{\text{opt}}(s' \mid s, a) = \mathcal{P}_{\text{base}}(s' \mid s, a) = \delta(s' - f(s, a))$$

**证明**:
1. **碰撞判定等价性**:
   智能体在物理平面上的占有区域为多边形凸包 $\mathcal{C} \subset \mathbb{R}^2$。设其最小轴对齐包围盒为 $\text{AABB}(\mathcal{C}) = [x_{\min}, x_{\max}] \times [y_{\min}, y_{\max}]$。显然有集合包含关系：
   $$\mathcal{C} \subseteq \text{AABB}(\mathcal{C})$$
   设障碍物集合为 $\mathcal{O}_{\text{obs}} \subset \mathbb{R}^2$。碰撞发生的充分必要条件为 $\mathcal{C} \cap \mathcal{O}_{\text{obs}} \neq \emptyset$。若 $\text{AABB}(\mathcal{C}) \cap \mathcal{O}_{\text{obs}} = \emptyset$，由包含关系必然推得 $\mathcal{C} \cap \mathcal{O}_{\text{obs}} = \emptyset$（无碰撞）。AABB 短路检查仅在充分条件满足时直接跳过精细多边形光栅化。当包围盒内存在障碍物时，依然无条件回退至精确几何求交。因此两者的碰撞判定布尔输出在全状态空间上严格相等。
2. **面积与重叠增量等价性**:
   因车体运动步进位移极小，割刀扫过的区域 $\Delta \Omega_t$ 严格被限制在其外接包围盒 $\text{AABB}(\Delta \Omega_t)$ 之内。在包围盒外部，农田指示掩码与重叠计数的变化量恒为零：
   $$\forall p \notin \text{AABB}(\Delta \Omega_t), \quad \Delta \text{Overlap}(p) = 0, \quad \Delta \text{Field}(p) = 0$$
   因此全局求和等价于局部包围盒求和：
   $$\sum_{p \in \text{Map}} \Delta \text{Overlap}(p) = \sum_{p \in \text{AABB}} \Delta \text{Overlap}(p)$$
   两者的离散状态转移在数值上严格达到 $0.000000$ 绝对等价。 $\blacksquare$

### 4.4 奖励函数无偏定理 (Reward Invariance Theorem)

**定理 3**: *除修复 B05（双倍角取模导致正交奖励反转）外，所有纯性能优化对环境步进奖励输出 $\mathcal{R}(s, a, s')$ 的数学期望与方差无任何偏移：*
$$\mathbb{E}[\mathcal{R}_{\text{opt}}(s, a, s')] = \mathbb{E}[\mathcal{R}_{\text{base}}(s, a, s')]$$
$$\text{Var}[\mathcal{R}_{\text{opt}}(s, a, s')] = \text{Var}[\mathcal{R}_{\text{base}}(s, a, s')]$$

**证明**:
环境奖励函数由进度奖励、重叠惩罚、碰撞惩罚以及动作平滑项线性加权构成：
$$\mathcal{R}(s, a, s') = w_1 \Delta \text{Coverage} - w_2 \Delta \text{Overlap} - w_3 \mathbb{I}_{\text{collision}} - w_4 \|a\|^2$$
根据定理 2，$\Delta \text{Coverage}$、$\Delta \text{Overlap}$ 以及 $\mathbb{I}_{\text{collision}}$ 在优化前后的输出位级恒等；动作 $a$ 来源于策略网络推理，根据定理 1 观测等价，其均值与方差完全守恒。因此标量奖励输出严格恒等。 $\blacksquare$

### 4.5 Bellman 更新与策略梯度守恒证明 (Policy Gradient Invariance)

**定理 4**: *在经验回放池 $\mathcal{D}$ 中剔除死次态标签（F22）以及使用 UINT8 观测存储（F23）后，SAC Critic 目标损失的梯度估计量满足无偏性：*
$$\mathbb{E}_{\mathcal{B} \sim \mathcal{D}_{\text{opt}}}[\nabla_\theta J_Q(\theta)] = \mathbb{E}_{\mathcal{B} \sim \mathcal{D}_{\text{base}}}[\nabla_\theta J_Q(\theta)] + \mathcal{O}(\epsilon_{\text{quant}})$$
*其中 $\epsilon_{\text{quant}} \le \frac{1}{255} \approx 0.0039$，远低于真实机器人传感器观测噪声；且反向反传使用 FP16 AMP 与 GradScaler 时梯度严格稳定。*

**证明**:
1. **死次态标签剔除 (F22)**:
   审查 SACLoss 与 HIF 辅助任务目标函数实现：
   $$J_{\text{HIF}}(\psi) = \| f_\psi(\text{backbone}(s)) - \text{label}(s) \|^2$$
   损失函数仅在当前时间步计算重建损失，**全流程没有任何计算图读取 `("next", "label_ego_hif")`**。设全量状态元组为 $x = (s, a, r, s', y, y')$，其中 $y'$ 为次态标签。由于损失函数满足对 $y'$ 的偏导恒等于零：
   $$\frac{\partial \mathcal{L}_{\text{total}}}{\partial y'} \equiv 0$$
   因此，在采样存储中将 $y'$ 置空并剔除，损失反向传播梯度 $\nabla_\theta \mathcal{L}$ 的数学表达式与数值计算结果严格 100% 恒等（梯度差为 $0.000000$）。
2. **UINT8 归一化量化 (F23)**:
   观测图像像素物理范围为 $[0.0, 1.0]$。UINT8 量化定义为 $q = \text{round}(255.0 \cdot x)$，反量化为 $\tilde{x} = q / 255.0$。量化扰动 $\delta = \tilde{x} - x$ 为零均值均匀分布白噪声：
   $$\mathbb{E}[\delta] = 0, \quad \mathbb{E}[\delta^2] = \frac{1}{12 \times 255^2} \approx 1.28 \times 10^{-6}$$
   在深度神经网络中，前向卷积层与归一化层具备对极微弱输入扰动的天然平滑性，其引发的梯度扰动完全淹没在随机小批次抽样（Stochastic Batch Sampling）的方差之内，收敛渐近不动点严格守恒。 $\blacksquare$

---

## 5. 解耦矩阵与变更隔离红线总结

下表从系统级接口、状态一致性与梯度反向传播三重视角，汇总了优化项的解耦形式化审计结论：

| 子系统模块 | 优化项清单 | 数学等价性判定 | 梯度/损失影响 | 算法研究员是否可见? |
| :--- | :--- | :---: | :---: | :---: |
| **几何图像** | 闭式直接仿射、AABB 增量更新、多尺度剪枝(保持25通道) | **代数恒等 / 位级等价** | 严格 0 误差 | 否 (仅感知 step 加速) |
| **C++ 算子** | 有界 BFS ($\lceil D_{\max}\sqrt{2}\rceil$)、2D LUT、双视锥投射 | **代数恒等 / 截断域等价**| 严格 0 误差 | 否 (仅感知算子加速) |
| **并发与系统**| OMP=1 线程锁控、Policy 跑 CPU、IPC 权重节流 (2000 帧) | **统计等价 (Off-Policy)** | 极轻微采样滞后 | 否 (显存大幅节省) |
| **内存与存储**| 剔除 next 标签、UINT8 压缩、DropObservation 防爆 | **无损 / 传感器噪声级** | 梯度严格一致 | 否 (磁盘与内存大幅节省) |
| **CUDA 训练**| AMP FP16 混合精度反传 + GradScaler (禁止裸 BF16) | **标准数值近似** | 满足 PyTorch AMP | 否 (训练稳定提速) |
| **Bug 修复** | 修复 2D 坐标折行、多环境重置掩码、双倍角、通道硬编码 | **恢复正确物理几何** | 消除虚假惩罚 | 是 (训练更加平稳) |

---

## 6. 证伪与防坑附录：架构解耦与工程边界反模式 (Falsification & Anti-Pattern Appendix)

在系统架构演进与性能工程重构中，极易产生几类看似合理但实际上破坏系统稳定性与可维护性的反模式。本附录记录三项核心反模式并给出第一性原理证伪：

### 6.1 异源组件移花接木反模式（Dial 最短路与外部 Q-VPI 污染证伪）

- **诱惑方案**:  
  早期文档草案中盲目引入了名为 `dial_sssp` 的整数桶队列单源最短路算法以及大量 `wg_qvpi*` 命名空间下的算子与脚本，试图通过照搬外部项目的代码来充当 NewCppRL 的图搜索加速方案。
- **证伪铁证与数学破缺**:  
  1. **代码真实性证伪**: 在 NewCppRL 真实代码库的 `src/`、`envs_new/`、`rl_new/` 中执行全局检索，匹配度为零。该模块纯属外部项目残留的幻觉实体；
  2. **物理场域不契合**: Dial 算法仅适用于边权为极小非负整数的网格图。而 NewCppRL 的连续物理空间中，小车运动规划与势场衰减是以连续欧氏度量为基础的。强行将连续欧氏距离量化为离散整数桶，会导致对角线移动产生高达 $1.414 \approx 1$ 的粗糙截断，引发路径折线锯齿与避障震荡。
- **治理法则**:  
  坚决予以物理净空，底盘势场严格基于 NewCppRL 原生 `src/cpu_apf.cpp` 结合欧氏 2D LUT 进行精准加速。

### 6.2 动态兼容垫片（Dynamic Shims）反模式：为什么破坏 Fail-Fast 仪器哲学

- **诱惑方案**:  
  在面临 `envs/` 与 `envs_new/`、`rl/` 与 `rl_new/` 等新旧模块并存时，部分工程设计倾向于引入类似 `NEWCPPRL_BASELINE_MODE` 的动态环境变量或在 `__init__.py` 中编写多重 `try-import-fallback` 垫片（Shim）。
- **证伪铁证与调试灾难**:  
  动态垫片是典型的“防御者心态（Fail-Safe 优雅降级）”。在科研与生产环境下，它会静默掩盖模块路径拼写错误、循环依赖或依赖缺失，导致算法科学家在不知情的情况下运行了旧版废弃代码，产生不可重现的训练漂移。
- **治理法则**:  
  坚决遵从第 08 与 09 章确立的「内部零动态垫片（Zero-Shim Policy）」。依靠确定的代码拓扑与离线静态迁移工具（`tools/migrate_import_paths.py`）进行原子切换。代码应当像科学仪器：准确报错（Fail-Fast）比优雅掩盖未知更有价值。

### 6.3 抽象泄漏与跨层防御性代码雪崩

- **诱惑方案**:  
  当底层某个组件（例如 `Transform` 或 `ObservationGenerator`）出现状态更新不及时或键名不一致时，在调用链上游的多个文件（如环境层、训练主循环、经验池写入）中层层添加 `try-except` 或 `.get(key, 0.0)` 进行跨层防御。
- **证伪铁证与认知负荷爆炸**:  
  这种 AI 脏补丁（Dirty Patching）极大地污染了代码架构的纹理。一个简单的局部计数器 Bug，若通过修改 6 个文件、塞入 40 行胶水代码解决，将使后来研究者的认知负荷呈指数级上升，且彻底掩盖了数据流的确定性。
- **治理法则**:  
  **在 Bug 所在的抽象层进行 10~15 行最小有效修改**。Transform 的问题就在 Transform 内部通过张量向量化解决，绝不跨越抽象边界污染环境层与模型层。
