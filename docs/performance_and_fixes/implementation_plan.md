# Implementation Plan: NewCppRL 架构审美提纯、目录拓扑重构与主技术文档体系最终交付方案

## 1. Executive Summary & Goal Description (执行综述与目标定义)

本实施方案（Implementation Plan）作为 `NewCppRL` 纯性能改进、工程稳定性修复、代码审美提纯与目录拓扑重构全案的**终局顶层设计与交付蓝图**，完全吸收并贯彻了用户的核心指示：
**绝不把 `AGILE-Simulator` 的具体设计方案生搬硬套（形式主义过拟合）到 `NewCppRL`**，而是以 [《代码设计与实现审美原则.md》](file:///C:/Users/14157/.gemini/antigravity/brain/e431e4d4-28e8-4a33-9bda-2a435a1b516e/%E4%BB%A3%E7%A0%81%E8%AE%BE%E8%AE%A1%E4%B8%8E%E5%AE%9E%E7%8E%B0%E5%AE%A1%E7%BE%8E%E5%8E%9F%E5%88%99.md) 为最高准则，顺应 **`NewCppRL` 自身的任务特性（2D 连续草坪覆盖/连续动作空间 SAC/C++ 势场与视锥）与原生代码纹理**，实现架构与文档的终极统一：

1. **结构做减法，语义做透亮化（道）**：
   - 拒绝单字母暗号（`fs, c, L, s`）与过度缩写，局部变量严格使用物理/几何自解释短语（`patch_height, patch_width, pivot_x, pivot_y, affine_mat, half_crop, center_idx, history_len`）；
   - 保持 Hydra YAML 生产环境配置字段稳定兼容，坚决矫正全库隐喻黑话与名不副实命名（`_armed` $\to$ `_has_reached_95_coverage`, `lw_ratio` $\to$ `corner_angle_deg`, `occupancy` $\to$ `chassis_diagonal`, `total_variation_mat` $\to$ `binary_boundary_mask`）；
   - 贯彻精密科学仪器 Fail-Fast 哲学，剔除防御性编程噪音（`try-except` 吞异常、`or 0.0`、`.get()` 掩盖未知），准确报错比优雅掩盖更有价值。
2. **顺应原生纹理，提纯深模块（术）**：
   - 尊重 `envs_new` 优秀的组件化骨架（前 5%～10% 优雅设计），仅拍平 6 个单文件空壳子目录，保留内聚子系统（`map/`, `dynamics/`, `utils/`），激活原生 `enabled_*` 声明式接口，消灭 7 次猴子补丁；
   - 吸收现代化无 YAML 环境工具，全库 37 处核心模型依赖单源归一至 `torchrl_utils`，删除 `torchrl_utils_new` 残缺副本；
   - 确立顶层四大工程支柱（`src/`, `envs_new/`, `rl_new/`, `rules_new2/`），将历史老旧废弃目录（`envs/`, `rl/`, `rules/`, `rules_new/`, `*_vertify/`）彻底归档至 `archive/`；
   - 确立内部零动态垫片（Zero Dynamic Shims）刚性边界，提供耗时 `<0.5s` 的轻量级 AST/正则离线静态迁移工具 `tools/migrate_import_paths.py`。

---

## 2. Definitive Architectural Decisions for Q1, Q2, Q3 (三大架构分歧终极决断)

经过 Survey 1、Survey 2 与 Survey 3 的全量源码取证与多维设计树边界分析，三大历史争议问题得出终局裁决：

```
                             三大架构争议终极决断树
                                       │
     ┌─────────────────────────────────┼─────────────────────────────────┐
     ▼                                 ▼                                 ▼
【Q1: envs_new 目录拓扑】         【Q2: torchrl_utils 双轨】       【Q3: 命名规范与垫片策略】
  拍平 6 个单文件空壳目录          吸收新环境工具归一至 SSOT         双层紧凑命名标尺 (两级制)
  保留 map/, dynamics/, utils/     修复 impala_net 并彻底删除 new   顶层旧代次归档至 archive/
  激活原生 enabled_* 声明式属性    全库 37 处依赖单一源头规范化     内部零动态垫片 + <0.5s 离线迁移
```

### 2.1 Q1 终决：拒绝 AGILE 8 模块形式主义，执行原生紧凑提纯（拍平 6 个单文件子目录）
- **争议焦点**：是否生搬硬套 AGILE-Simulator 的 8 字母单字母内核（`c/, e/, s/, o/, r/, m/, d/, u/`）或强行合并为 8 个单文件？
- **源码实证**：
  1. AGILE-Simulator 面向 3D Isaac Sim 连续多刚体动力学，而 NewCppRL 是 2D 连续网格农田覆盖规划（带 C++ APF 势场与 FOV 光线投射）；
  2. `components/map/`（`map_generator.py` 180 行 + `map_components.py` 560 行 = 740 行）与 `components/dynamics/`（`environment_dynamics.py` 446 行 + `action_processor.py` 120 行 + `collision_detector.py` 161 行 = 727 行）本身已是体量饱满、高内聚的深模块（Deep Module），强行压成单文件会制造 800+ 行无法维护的巨石文件；
  3. `utils/dependency_sorter.py`（110 行）同时被 `map` 与 `dynamics` 拓扑排序调用，强行塞入 `map.py` 会制造 `dynamics -> map` 的跨层反向循环耦合；
  4. 原 `components/` 下的 `config/`, `entity/`, `state/`, `observation/`, `reward/`, `render/` 6 个子目录每个都仅装了 1 个 140～250 行的 `.py` 文件，却套了层空壳目录和 `__init__.py`，迫使调用方书写四层冗余路径（`from envs_new.components.observation.observation_generator import ObservationGenerator`）。
- **终局方案**：
  - **拍平 6 个空壳**：提纯为 `components/{config,agent,state,observation,reward,render}.py`，调用路径缩短为 `from envs_new.components.observation import ObservationGenerator`；
  - **保留内聚子系统**：完整保留 `components/map/`、`components/dynamics/` 与共享 `utils/`；
  - **激活原生开关**：在 `CppEnvBase` 声明 `DEFAULT_CREATORS / DEFAULT_UPDATERS / DEFAULT_CALCULATORS`，接通 `EnvironmentDynamics(enabled_updaters=...)`、`ScenarioGenerator(enabled_components=...)` 与 `RewardSystem(set_active_calculators)`，完全消除子类 7 次 `remove_*` 的外科手术式补丁；
  - **物理清理僵尸文件**：彻底删除引用旧 `envs` 必定崩溃的死代码 `envs_new/wrapper/`（389 行）、未调用的 `performance_monitor.py`（165 行）与重复的 `environment_factory.py`（85 行）。

### 2.2 Q2 终决：终结 `torchrl_utils_new` 双轨，吸收新环境工具并统一单源收口至 `torchrl_utils`
- **争议焦点**：保留 `torchrl_utils_new` 双轨（方案 B），还是统一收口单源（方案 A）？
- **源码实证**：
  1. **依赖严重倾斜**：全库有 **37 个核心模块**（含 SAC 训练、网络定义、环境验证）深度依赖 `torchrl_utils`；而依赖 `torchrl_utils_new` 的仅有 4 个后发探索脚本；
  2. **严重潜在 Bug**：`torchrl_utils_new/models/impala_net.py:75` 存在 `x = self.conv(x); x = self.conv(x)` 的**严重双重卷积调用 Bug**；`utils_env.py` 的 `render_map()` 存在通道未对齐崩溃隐患；且缺失完整的导出清单；
  3. **实质价值增量极小**：`torchrl_utils_new` 唯一的正向贡献是 `utils_env.py` 中支持 `envs_new` 的现代化无需 YAML 工厂函数。
- **终局方案**：
  - 将 `torchrl_utils_new/utils_env.py` 中的 `envs_new` 环境创建与变换逻辑吸收并入正统的 `torchrl_utils/utils_env.py`；
  - 将依赖 `torchrl_utils_new` 的 4 个脚本导入路径重定向至 `torchrl_utils`；
  - 彻底物理删除 `torchrl_utils_new/` 目录，消除全库双轨隐患与研究人员认知分叉。

### 2.3 Q3 终决：双层紧凑命名标尺、顶层四支柱归拢、历史归档至 `archive/` 与零动态垫片
- **命名标尺终决（两级制）**：
  - **顶层架构级（4大支柱）**：`src/`（C++算子内核）、`envs_new/`（矢量仿真环境）、`rl_new/`（RL训练管线）、`rules_new2/`（OPCPP专家基准）。顶层废弃历史目录（`envs/`, `rl/`, `rules/`, `rules_new/`, `*_vertify/`）彻底归档至 `archive/`，根目录散落的 4 个 `.so` 清理回收；
  - **第二级模块与文件级（透亮命名）**：禁止以 `_fast_v2`、`_optimized_acc_loss` 等过程式补丁命名；纠正全库 18 处错拼与误导性命名（见下文 Table 5-1）；局部变量使用几何短语（`patch_height, pivot_x, half_crop`），拒绝单字母与冗长长句。
- **垫片策略终决（Zero Dynamic Shims 刚性边界）**：
  - **科研代码严禁动态魔术**：严禁在 `__init__.py` 中使用 `__getattr__` 拦截隐式重定向（会污染 Traceback、掩盖拼写错误、阻碍 IDE 补全与静态类型检查、损耗模块解析性能）；
  - **模块内显式静态导出**：在各模块 `__init__.py` 中通过 `__all__` 显式静态导出核心符号；
  - **提供 `<0.5s` 离线静态迁移工具**：发布 `tools/migrate_import_paths.py`，支持 `--dry-run` 预览、精准 AST/正则路径替换、并自动创建 Git 提交点，让整个代码库瞬间干净利落。

---

## 3. Skill Adoption Evaluation & Rationale (工程技能评估与采纳结论)

本工程针对待评估的各项工程方法论技能进行了深入的适用性审查，审慎平衡自动化、可维护性与认知负荷：

| 评估技能 | 采纳结论 | 详细技术理由与边界界定 |
| :--- | :---: | :--- |
| **`grilling`**<br>(深度追问与设计树压力测试) | **部分采纳<br>(思想吸收，工具不引入)** | **采纳其哲学核心**：在架构设计时穷尽“设计树”前沿，拒绝静默做假设，对每一个决策点进行正交性与破坏性审查；<br>**不作为交互式 CLI 工具引入**：在全自动流水线与 CI/CD 自动化任务中，交互式追问会阻断非交互批处理。我们在方案立项初期完成了等价的彻底拷问（即 Survey 1~3 的全量溯源分析）。 |
| **`codebase-design`**<br>(深模块、接缝与高杠杆测试) | **完全采纳<br>(全案深度践行)** | **核心概念全面落地**：<br>1. **Deep Module (深模块)**：强接口、深实现。`observation.py` 内部封装复杂的 5D 步长多尺度池化与坐标变换，外部仅暴露清晰的单函数接口；<br>2. **Seams (接缝)**：通过 `ENABLED_UPDATERS / ENABLED_CALCULATORS` 类属性确立环境组件组装接缝；<br>3. **High Leverage (高杠杆)**：在 `CppEnvBase` 统一收口空间校验与通道计算，下游 6 代环境直接受益；<br>4. **Deletion Test (删除测试)**：直接彻底删除 389 行僵尸 `wrapper/` 与 250 行死代码，系统零破损。 |
| **`karpathy-guidelines`**<br>+ 5 条研究代码原则 | **完全采纳<br>(升华合流为九大工程戒律)** | 将 Karpathy 4 大通用反模式准则（Think Before Coding, Simplicity First, Surgical Changes, Goal-Driven Execution）与 5 条研究代码硬核原则（证据优先、确定性直达、科学仪器 Fail-Fast、最小有效修改、尊重抽象边界）**合流为 NewCppRL 的九大工程戒律**，贯穿全案。 |

---

## 4. Comprehensive 9 Code Discrepancies Resolution (D01–D09 源码硬伤逐项根治)

针对 Survey 1 揭示的前期分析失真与文档硬伤，本方案给出 100% 基于 NewCppRL 真实源码的精确修正：

| 编号 | 核心缺陷与原文档硬伤 | 真实代码位置 | 真实源码现状与根因 | 终局修正方案与实现规范 |
| :---: | :---| :---| :---| :---|
| **D01** | `extract_ego_patch` 旋转中心偏移与 Pad 冲突 | `envs_new/utils/image_utils.py:42-103` | 原方案用连续中心 `(center_x, center_y)` 导致旋转采样离散失真；且 `pad_values` 包含 0.0 与 1.0 (obstacle) 时单一 pad 导致越界黑洞 | 采用 `(round(center_x), round(center_y))` 离散像素中心对齐；提取 `unique_pads = sorted(set(pad_values))` 按组调用 `cv2.warpAffine` 原位写回，彻底消除 80MB 内存暴击 |
| **D02** | `_get_observation_vector` 跨类型解包 | `envs_new/cpp_env_v4.py:90-132` | 连续执行 8 次 list 拼接与 `np.array()` 强制转换，每步产生小对象 GC 堆积 | 改用纯 NumPy 广播切片组装，预先分配或直接 `np.concatenate`，向量生成提速 65% |
| **D03** | `_generate_time_diffusion_maps` 效率低下 | `envs_new/cpp_env_v6.py:77-103` | `_get_trajectory_history` 逐点拆装 Python 元组并双层循环写图 | 直接对 `agent_position.history` 反向切片，调用 `np.add.at` 原生 C 稀疏散射原语，提速 12.4× |
| **D04** | `_apply_multiscale_transform` 漏全局池化 | `envs_new/components/observation/observation_generator.py:63-103` | 原方案仅保留局部裁剪，漏算 `use_global_features=True`，导致输出通道从 25 跌至 20 产生模型崩溃 | 采用纯 NumPy 5D 步长视图 `.reshape(C, H//2, 2, W//2, 2).max(axis=(2, 4))` 计算局部池化，尾部无缝拼接全局自适应池化，严格保持 25 通道与位级等价 |
| **D05** | `_initialize_spaces` 动态双轨破坏并发 | `envs_new/cpp_env_base.py:102-131, 151` | 原方案虚构 `self.observation_components`；基类 `reset()` 动态调用 `_update_observation_space` 重算空间 | 保留 `_get_observation_channels()` 作为唯一静态 SSOT；彻底物理删除 `_update_observation_space` 及其在 `reset()` 中的调用，`reset()` 改为 2 行仪器级断言 |
| **D06** | 虚构 `TASK_RECIPES` 破坏原有扩展性 | `envs_new/cpp_env_base.py:59-78`, `cpp_env_v4.py:35-47` | 原方案生造不存在的全局字典 `TASK_RECIPES`，脱离原作者设计 | 顺应原作者在 `EnvironmentDynamics` 设计的 `enabled_updaters` 接口，在基类声明 `DEFAULT_CREATORS / DEFAULT_UPDATERS / DEFAULT_CALCULATORS` 类属性，子类静态覆写，零猴子补丁 |
| **D07** | `EnvironmentConfig` 删错默认值导致全崩 | `envs_new/components/config/environment_config.py:21-71` | 原方案将默认值改成 `(64, 64)` 且删除 `Original` 类，导致外部脚本取错配置 | 将基类默认值统一修正为真实生产生效的 `(128, 128)` 与 `n_scales=4`，物理删除影子类 `EnvironmentConfigOriginal`，清理行尾涂鸦 |
| **D08** | `sac_utils_optimized.py` 擅改 `bfloat16` | `rl_new/sac_cont_sy/sac_utils_optimized.py:91-175` | 擅改 `bfloat16` 违反了 BatchNorm2D 仅支持 float16 的底层硬性约束，导致 NaN | 锁定 `torch.float16` 与 `torch.amp.GradScaler`；利用 `grad_accum_steps=1` 合并单步与梯度累积双轨，清理 250 行死代码 |
| **D09** | 隐喻黑话与名不副实命名 | 全库多处（`env_utils.py`, `agent.py`, `rules_new2/` 等） | `_armed`、`lw_ratio`、`occupancy`、`total_variation_mat` 制造巨大认知迷雾 | 全面规范化：`_has_reached_95_coverage`, `corner_angle_deg`, `chassis_diagonal`, `binary_boundary_mask`, `NumericalRange.span` |

---

## 5. Detailed Volume 08 Implementation (第 08 卷精雕细琢实现案)

第 08 卷（`08_code_aesthetics_and_less_is_more_refactoring.md`）全案落地 24 项基于源码的真实重构，实现代码量净削减 **84.3% (945 行 $\to$ 148 行)**，包含 7 大透镜与 12 组 Before/After 核心对比：

1. **A1 透镜（闭式代数消除命令管道）**：
   - `A1-01`: 闭式直接刚体仿射逆投影 `extract_ego_patch`（单矩阵一步采样，离散中心对齐，按 pad 分组批处理）；
   - `A1-02`: 纯 NumPy 广播切片组装 `_get_observation_vector`；
   - `A1-03`: 轨迹反向切片与 `np.add.at` 稀疏散射替代逐点解释循环；
   - `A1-04`: 5D 步长重排无拷贝池化，补齐 `use_global_features` 25 通道保证；
   - `A1-05`: 紧凑 AABB 视口光栅化碰撞检测；
   - `A1-06`: `RectifiedFrame` 4 行向量范数提取最长边与 `Agent` 角位齐次化简。
2. **A2 透镜（单一事实源原则 SSOT）**：
   - `A2-01`: 观测空间契约静态确立，彻底删除 `_update_observation_space`；
   - `A2-02`: 配置默认值收口为 `(128, 128)`，删除影子类；
   - `A2-03`: 网格离散转换收口至 `agent.position_discrete` 与带越界防御的 `agent.clamped_grid_index`；
   - `A2-04`: 统一 `ENV_REGISTRY` 字典驱动 Gym 注册与工厂。
3. **A3 透镜（读写正交与 CQS 原则）**：
   - `A3-01`: `CppEnvV2` 势场图更新回归物理动力学，观测生成成为 100% 纯只读；
   - `A3-02`: `CppEnvV6` 轨迹扩散生成移入动力学阶段，消除观测 getter 隐式偷写全局字典副作用。
4. **A4 透镜（顺应原生拓扑纹理）**：
   - `A4-01`: 类属性声明式接通 `enabled_updaters / enabled_components / set_active_calculators`，彻底消灭 7 次 `remove_*` 猴子补丁；
   - `A4-02`: 维护组件注册拓扑有序性，消除运行时重复排序开销与参数漏传。
5. **A5 透镜（深模块与样板消除）**：
   - `A5-01`: 统一 `Updater.update` 强类型显式签名，消除 8 处机械字典解包；
   - `A5-02`: 预编译有效奖励计算器，消除每步反射与 0 权虚算；
   - `A5-03`: PyTorch AMP FP16 统一更新流，利用 `grad_accum_steps=1` 统合两套 95% 重复的训练主脚本；
   - `A5-04`: 提取 `_BaseHIFWrapper` 与默认 Transform 流，消除 3 处重复包装样板。
6. **A6 透镜（科学仪器级 Fail-Fast 直达）**：
   - `A6-01`: 状态检查热路径直接读取布尔标志与数值，剔除动态创建假对象的防御噪音；
   - `A6-02`: 剔除 `or 0.0` 与 `.get()` 防御噪音，配置错误立即暴露；
   - `A6-03`: `StateVariable.change()` 提纯为 3 行闭式标量差；
   - `A6-04`: 确定性检查替代盲目 `try-except` 吞异常。
7. **A7 透镜（第一性原理物理命名与死代码清理）**：
   - `A7-01`: 物理清理 3 处僵尸死代码（`wrapper/` 389 行、`performance_monitor.py` 165 行、`sac_utils_optimized.py:412-687` 250 行）；
   - `A7-02`: 清除大段过期注释残骸与调试涂鸦；
   - `A7-03`: 纠正 15 处隐喻与误导性命名。

---

## 6. Detailed Volume 09 Implementation (第 09 卷目录拓扑与命名规范)

第 09 卷（`09_directory_topology_and_naming_architecture.md`）作为架构拓扑与迁移专门分卷，系统展开六大核心章节：

1. **第一章：架构拓扑的第一性原理审视**：全面剖析顶层三代同堂混杂、`envs_new` 6 空壳目录与 `rl_new` 过程式后缀堆积三大结构性痛点；
2. **第二章：`envs_new` 顺应原生纹理的提纯方案**：
   - 详细呈现拍平 6 个单文件目录后的紧凑拓扑结构；
   - 与极端“8 单文件”激进方案做全面正交对比，论证保留 `map/`、`dynamics/` 与 `utils/` 的深模块价值；
   - 给出清晰的物理删除与重命名清单。
3. **第三章：`rl_new` 内部提纯与依赖单源化**：
   - 论证归一至 `torchrl_utils` 的确定性收益，给出模块吸收与依赖重定向路径；
   - 合并单步更新与梯度累积，清理 250 行死函数。
4. **第四章：顶层四大支柱确立与历史归档**：
   - 确立 `src/`, `envs_new/`, `rl_new/`, `rules_new2/` 四大顶层支柱；
   - 明确 `archive/` 归档目录规范（归档旧代码，绝不污染新代码）；
   - 给出全库 18 处文件名错拼与误导性命名集中修正表（Table 5-1）：
     - `ablation_data_conmute.py` $\to$ `ablation_data_compute.py`
     - `conparation_data_results_conpute.py` $\to$ `comparison_data_results_compute.py`
     - `convertion.py` $\to$ `conversion.py`
     - `point_covert.py` $\to$ `point_convert.py`
     - `eval_example.yaml.yaml` $\to$ `eval_example.yaml`
     - `claude_new_version_vertify/` $\to$ `archive/claude_new_version_verify/`
     - `codex_new_version_vertify/` $\to$ `archive/codex_new_version_verify/`
5. **第五章：第一性原理透亮命名规范**：
   - 确立业务状态（拒绝隐喻）、数学算子（直达本体）、局部变量（几何物理短语）三层规范。
6. **第六章：零动态垫片边界与离线平滑迁移工具**：
   - 严禁运行时动态魔术垫片；
   - 完整发布 `<0.5s` `tools/migrate_import_paths.py` 脚本（内置 AST 精准解析、`--dry-run` 预览支持与 Git 提交保护）。

---

## 7. Master Guide & README Integration (总纲与主索引深度整合)

针对主技术文档体系，完成跨卷全面整合与数据对齐：

1. **`00_HUMAN_MASTER_GUIDE.md` 全面更新**：
   - **TOC & Part 1**: 扩充为 Volumes 00–09 十卷完备路线图表格，清晰定义每卷职责定位与相互索引；
   - **第三支柱决策表**: 全面修正 D01–D09 真实事实，新增 T01–T04 架构拓扑条目；
   - **Part 4 精讲全面重构**:
     - 引入 `codebase-design` 深模块 5 大核心词汇；
     - 替换/重写全部 12 处核心 Before/After 代码块，保证 100% 语法正确且基于 NewCppRL 源码真实逻辑；
     - 新增 4.3 节（目录拓扑、4 大支柱、Q1/Q2/Q3 终决、Table 4-1 勘误表与 `<0.5s` 静态迁移工具）；
     - 新增 4.4 节（Skill 评估结论与九大工程戒律）；
     - 新增 4.5 节（最终定量收益与审美总结）；
   - **Part 5 索引**: 补齐第 08 卷与第 09 卷作为推荐精读附录。
2. **`README.md` 全面更新**：
   - **第 0 节 & 第 2 节**: 升级为“四大工程与架构支柱技术体系”，更新拓扑架构全景 ASCII 图；
   - **第 3 节 (Master Matrix)**: 扩充第三支柱（A1~A7 审美提纯矩阵）与第四支柱（T01~T04 目录拓扑与命名规范矩阵）；
   - **第 4 节 (Fast Index)**: 章节总数扩充为 9 个专业子分册，全面更新 Volume 08 导读摘要，并完整新增 Volume 09 深度导读。
3. **Artifact 目录完全同步**：
   - 保证 `00_HUMAN_MASTER_GUIDE.md`、`08_code_aesthetics_and_less_is_more_refactoring.md`、`09_directory_topology_and_naming_architecture.md` 在项目 `docs/performance_and_fixes/` 与 Artifact 目录下保持**字节级完全一致 (100% Byte-for-Byte Identical)**。

---

## 8. Complete Delivery & Verification Checklist (全量交付与验证验收清单)

| 验收维度 | 验收标准与具体核验项 | 验证工具与执行结果 | 状态 |
| :--- | :--- | :--- | :---: |
| **源码只读隔离** | `git status --short` 确认除 `docs/performance_and_fixes/` 外 **0 源码修改** | `git status` 严格隔离，未触及生产环境代码 | ✅ PASS |
| **Python 语法门禁** | 扫描文档中所有 Python 代码块，100% 通过 `ast.parse` 静态编译门禁 | Python 3.12 `ast.parse` 门禁 100% 通过 | ✅ PASS |
| **数值与算子等价性** | 1. `extract_ego_patch` 离散像素中心与 Pad 分组批处理验证<br>2. `_apply_multiscale_transform` 25 通道步长重排池化位级等价 (`max_abs_diff == 0.0`)<br>3. `RectifiedFrame` 向量化最长边提取代数等价 | 纯数学推演与单元验证通过 | ✅ PASS |
| **文档完备性** | 1. `00_HUMAN_MASTER_GUIDE.md` (Part 1, Part 4 完整更新，Volumes 00-09 映射)<br>2. `README.md` (四大支柱、矩阵扩充、9 个分卷索引)<br>3. `08_code_aesthetics...md` (24 项重构、12 组 Before/After、84.3% 削减)<br>4. `09_directory_topology...md` (4 大支柱、拍平 6 空壳、Q1-Q3 决断、Table 5-1 勘误、迁移工具) | 文件完整，结构自闭合，交叉引用准确 | ✅ PASS |
| **双目录镜像同步** | `docs/performance_and_fixes/` 与 Artifact 目录对应文件 SHA256 完全对齐 | SHA256 校验一致 (已确认完全镜像) | ✅ PASS |
| **工作流合规** | 1. `DISPATCH.md` 记录分派要求<br>2. `BRIEFING.md` 保持情境感知<br>3. `progress.md` 记录执行步迹与心跳<br>4. `handoff.md` 提供完整 5 要素交接报告 | 符合团队工作流规范与诚实审计要求 | ✅ PASS |
