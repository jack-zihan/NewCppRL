# NewCppRL 当前项目说明

核查日期：2026-09-07。对象：本仓库当前工作树，Git HEAD `a0e82157ba6d2c0fbfc3fcb5a261531fdfc12561`，含原有未提交修改。

本说明服务于后续重构：说明能力在哪里、数据如何流动、哪些边界已经核验、哪些仍需实验。它不是生产修复记录，也不是训练或论文结果认证。

## 核查范围与结论

对用户指定的七个目录进行了 Python 文件、符号和显式导入的全量静态盘点，并沿环境、连续 SAC、规则评估及跨目录依赖阅读关键实现；补查 C++ 扩展、配置、地图工具、可视化、部署兼容层及测试。运行验证限定为 CPU 小配置、已有有限测试、配置解析和短规则评估。

| 目录 | Python 文件数 | 源文件行数 | 当前定位 |
| --- | ---: | ---: | --- |
| `envs/` | 15 | 2,824 | 历史环境、实体、地图脚本、旧接口包装 |
| `envs_new/` | 48 | 6,952 | 组件化环境、六个变体、配置、地图与 HIF 工具 |
| `rl/` | 25 | 3,663 | 历史算法、覆盖训练、评估、实测入口 |
| `rl_new/` | 28 | 5,489 | 迁移中的算法入口与连续 SAC 主线 |
| `rules_new2/` | 8 | 3,621 | 五类规则策略、学习策略接入及评估产物 |
| `torchrl_utils/` | 15 | 2,348 | 新旧代码共用的网络和旧环境工具 |
| `torchrl_utils_new/` | 12 | 1,316 | 部分迁移后的工具，仍有旧包依赖 |
| 合计 | 151 | 26,213 | 盘点范围，不代表每一行均经过缺陷审计 |

计数包含上述目录内的 `.py`，排除 `doc/`、`team_reports/`，不包含地图二进制、测试目录、第三方环境或运行产物。静态导入统计不覆盖动态导入、pickle 类路径和所有数据依赖，因此不能用“无静态 import”单独批准删除。

**核心判断：功能已经形成多条可辨认的数据流，但尚未收敛成完全独立、可替换的新版。** 当前可继续开发；应按任务语义和能力闭环重构，而不是把 `_new` 目录覆盖旧目录。历史审计只作线索，本说明中的现状来自当前源码和下面列出的运行检查；未重新验收历史报告的全部问题。

## 依赖与部署边界

### 当前依赖来源

| 层 | 当前事实 | 开发含义 |
| --- | --- | --- |
| Python | `pyproject.toml` 要求 ≥3.14；本机 `.venv` 为 3.14.5 | 旧 Python 3.12 / conda 命令不能作为当前默认安装说明 |
| lock 内 | torch 2.12.0、torchvision 0.27.0、torchcodec 0.13.0、triton 3.7.0；默认 cu130 索引 | 这是仓库当前配置，不是所有机器的通用兼容声明 |
| lock 外 | TorchRL、TensorDict 源码安装 | `uv.lock` 不能独自复原全部运行环境 |
| TorchRL 实装 | `0.14.0+gb295f5b011`；源码 commit `b295f5b011c6e8e3d895d4cf2828c8864a135d93`（2026-10-11 前进 +27 commits：MultiAsyncCollector 每个 batch 检查 worker 存活（#4505，子进程崩溃立即报错而不是等到超时）、DQN/DiscreteSAC 取动作 Q 值改走共享 helper（#4367）、整数 Bounded.rand 能取到上界（#4497/#4498）、reset/rollout 的 set_state 默认改为 False（#4501 [Deprecation]；本项目 env `_supports_set_state=False`，走不到）、新增 TD-MPC2 全套与 `rb.as_dataset()`；本项目 SAC loss / grad-norm / PRB 与分桶回放 / Collector / 2-worker MultiAsyncCollector 新旧双跑逐位一致，10 个真实入口模块导入通过；`torchrl/csrc` 未变动，故 `.so` 仅重链接未重编，editable 元数据已刷新） | 本机 editable 指向 `/home/lzh/rl_remote_uv314`；删除该目录会影响本项目。回退锚点 `b8b6f0cbdec82d2b47967de6bbafcd02439b9d9a` |
| TensorDict 实装 | `0.14.3+g6fdce72`；commit `6fdce72af5a1a2f52af68b5c6ae77da6c5dd978a`（tag `v0.14.3`，源码装，`_C` 对 torch 2.12 现编；27 个纯 Python 回补：索引规则对齐 torch、memmap 与 to_module 修复） | 升级顺序：先 TensorDict 后 TorchRL（TorchRL 下限仍为 `tensordict>=0.14.2,<0.15.0`；上游 0.14.1 正式版与 tensordict 0.14.3 配对）。回退锚点 `f2b44ecd1ed86b34aad43a075a04e9f45e22c044`（tag `v0.14.2`） |
| dubins | 实装 1.0.1，来源 `rdesc/pydubins` commit `58c4b68d9ddf972dfd32afea295c3195da648cf2` | 用于规则策略的 Dubins 路径；uv 构建配置注入 Cython / NumPy |
| 本地扩展 | `setup.py` 构建 `src/cpu_apf.cpp` 和 `src/cpu_fov.cpp` | 改 C++ 后需重建扩展；Python import 成功不能验证核函数数学正确性 |
| 旧模型兼容 | `deploy/legacy_ckpt_compat/` 提供 spec 类名与 InteractionType 的反序列化 shim | 本机环境已安装启动 shim；本次运行包含此前提，不等于干净环境验证 |

日常运行用 `.venv/bin/python` 或 `uv run --no-sync python`。本次 `uv sync --locked --offline --dry-run` 虽退出 0，却计划卸载 8 个 lock 外包；加 `--inexact` 后显示无变更。此行为由实际预演确认，不再沿用旧 README 的“默认同步不会动源码包”说法。

`uv pip check` 当前退出 1：TorchRL 要求 `pyvers>=0.2.3`，本机为 0.2.2；缺 `hoptorch>=0.1.4`。本次没有安装或升级它们。使用当前环境做的局部检查不等于依赖完全健康。

### 构建与迁移的真实边界

- 开发工作树通过 editable 安装能导入，不代表 wheel 完整。`[tool.setuptools.packages.find]` 当前仅包含 `envs*`、`rl*`，没有覆盖 `torchrl_utils*`、`rules_new2`、`utils`；地图资源的独立分发也未验收。
- 全新部署需要先安装 lock 中依赖并构建项目扩展，再构建与 torch ABI 匹配的 TorchRL / TensorDict 源码版本，并检查其依赖。上表记录可追溯版本，不是重新安装成功的证明。
- 本次没有重建干净虚拟环境、独立 wheel、CUDA 12 环境或跨机器运行；不继续承诺旧 README 中未经本次核验的 CUDA 平台兼容性。
- 主 SAC 还使用 TorchRL / TensorDict 私有接口，如 `torchrl._utils`、`_exclude_td_from_pytree()` 和 replay `storage._len`。升级时应同时验证接口、数值、采集、优先级更新和检查点，不能只看 import。

## 新旧实现之间的实际依赖

以下是当前源码里的真实依赖，不是计划中的结构：

```text
rl_new/dqn/dqn_train.py ──→ rl/dqn/dqn_utils.py ──→ torchrl_utils/utils_env.py ──→ envs/
rl_new/sac/sac_train.py ──→ rl/sac/sac_utils.py ───→ torchrl_utils/utils_env.py ──→ envs/
rl_new/sac_cont_sy/model_utils.py ──→ torchrl_utils/model/{deep_q_net,resnet_fpn_dual}.py
rl_new/sac_cont_sy/env_utils.py ────→ envs_new/
rules_new2/policies_learning.py ───→ rl_new/sac_cont_sy/{env_utils,model_utils}.py
torchrl_utils_new/{model,utils_eval,custom_evaluator} ──→ torchrl_utils/ 或 envs/
envs_new/wrapper/ ──→ 旧 envs.utils 和旧环境属性
```

三个同名环境工厂也不能混用：

| 工厂 | 环境来源 | 额外行为 |
| --- | --- | --- |
| `torchrl_utils/utils_env.py` | import 时读 `configs/env_config.yaml`，注册旧 `envs` | 旧模型和 DQN / 离散 SAC 仍走此处 |
| `torchrl_utils_new/utils_env.py` | 显式 `env_id`、`env_kwargs`，注册 `envs_new` | 基础 transforms；没有主 SAC 的 Steps95 / HIF 链 |
| `rl_new/sac_cont_sy/env_utils.py` | 主 SAC 配置中的 `env.env_id` 与 `env.env_kwargs` | overlap info 注册、Steps95、HIF 标签 / 预测包装、专用评估视频链 |

无 `_new` 后缀的 `torchrl_utils/model/` 还包含 CNN 双头、ResNet18/34 + FPN、HIF decoder、critic 和重建 loss。`torchrl_utils_new/model/` 并不是它的超集。清理目录前需同时处理训练、评估、pickle 路径和脚本入口。

## 环境内部数据流

主流程来源：[cpp_env_base.py](../envs_new/cpp_env_base.py)、[environment_dynamics.py](../envs_new/components/dynamics/environment_dynamics.py)、[map_generator.py](../envs_new/components/map/map_generator.py)。

```text
reset(seed, options)
  → 将同一 np_random 交给场景与观测组件
  → ScenarioGenerator 按依赖生成 field / agent / obstacle / weed / 可选地图
  → 创建 agent、maps_dict、env_state
  → dynamics.reset：必要时修正初始位置，setup_state，再执行一次 update
  → 根据真实地图刷新 observation_space → 生成初始观测

step(action)
  → ActionProcessor 解析 → MowerAgent 更新朝向与位置
  → 碰撞检测；碰撞时回退位置与方向
  → 按当前 updater 容器顺序更新地图、轨迹、状态
  → 生成观测（v2 写 APF，v6 写轨迹权重等派生地图）
  → RewardSystem 聚合奖励
  → terminated = crashed or finished；truncated = timeout
  → observation, reward, terminated, truncated, info
```

`env_state` 用 `StateVariable` 的 deque 保存状态历史，用静态信息保存地图尺寸、总目标量等。`weed_coverage_ratio = 1 - weed_count / total_weed_count`，field 同理；总目标为零时当前返回 1.0。`field_area` 在运行时是剩余量，不是累计覆盖量。

`CppEnvBase` 实际实例化 `EnvironmentConfigOriginal`：默认 128×128 视野、4 层多尺度、16×16 特征和全局特征。不能用其父类 `EnvironmentConfig` 的 512×512 / 6 层配置解释默认行为。v4 又覆盖为 768×768 / 96×96 特征、关闭全局特征并启用历史向量；v6 默认历史长度为 16。

### 版本、观测与奖励

| 版本 | 地图与任务更新 | 观测 / 附加能力 | 代码入口 |
| --- | --- | --- | --- |
| v1 | FOV 探索 field，车体清除 weed | 原始地图，无 mist / APF | `envs_new/cpp_env_v1.py` |
| v2 | 同为除草任务，含 mist / overlap 更新 | APF 势场和 APF 增量奖励；overlap 热图观测被注释 | `envs_new/cpp_env_v2.py` |
| v3 | 除草和迷雾探索 | 可见区域掩码，无 APF | `envs_new/cpp_env_v3.py` |
| v4 | 移除 weed 组件、updater、奖励；以扩展车体多边形覆盖 field | field、obstacle、overlap、覆盖顺序、可选 trajectory；历史 vector | `envs_new/cpp_env_v4.py` |
| v5 | 继承 v4，加 HIF creator / updater / reward | GT HIF 的 cos2 / sin2 / 可选 confidence 观测，另产出监督标签 | `envs_new/cpp_env_v5.py` |
| v6 | 继承 v5，替换 HIF reward | 显式取 v4 观测并加 trajectory_weights；GT HIF 不在策略图像输入中 | `envs_new/cpp_env_v6.py` |

多尺度普通地图使用中心裁剪与 max pooling；HIF 方向通道使用方向向量处理与加权池化。空间尺寸、通道数与 vector 必须从实际 spec / reset 输出读取，不用一个固定 shape 覆盖所有任务。`include_confidence`、`use_trajectory`、历史长度、多尺度和全局特征都会改变接口。

HIF 有三个不同角色：环境奖励的方向目标、v5 的观测输入、训练监督的 `label_ego_hif`。v6 虽不把 GT HIF 当观测，仍必须加载它。`set_pred_hif()` 只把预测送到可视化地图，不等于策略在用预测场驱动控制。

奖励由 `AVAILABLE_CALCULATORS` 遍历，读取 `reward_<name>`，再乘 `field` / `turning` 组系数。overlap 使用新增重复面积的负惩罚；APF / HIF 计算还需要 `map_dict`。基类 `get_reward_breakdown()` 没有传地图，不能直接声称它与实际 APF / HIF 总奖励一致。

### 地图与坐标契约

来源：[map_components.py](../envs_new/components/map/map_components.py)、[environment_config.py](../envs_new/components/config/environment_config.py)、[agent.py](../envs_new/components/entity/agent.py)。

- 位姿 `(x, y)`、地图数组 `[y, x]`，静态 `dimensions` 为 `(width, height)`。
- 机器人 direction / steer 使用度；动作是每步位移量与角度增量，运动先改变朝向再前进。源码没有把它定义成经过标定的米 / 秒控制接口。
- `map_dir` 相对路径由配置解析到仓库根。普通地图读取 `field/` 中 PNG 的**字典序排序列表**，`map_id` 是列表下标；从所选文件名解析出的数字才是 `field_id`。
- `field_id` 用于匹配 `obstacle/obstacle_<id>.png` 和 `hif/human_intent_field_<id>.npy`。不能把 map_id 直接当文件名数字。
- `specific_scenario_dir` 从目录读取 `map_field.png`、`map_obstacle.png`；除草环境还需 `map_weed.png`，v5/v6 还需 `map_hif.npy`。
- 随机障碍和已有障碍文件是不同加载分支；`num_obstacles_range=(0,0)` 关闭随机障碍生成，不会覆盖已存在的障碍文件或关闭边界。
- `weed_count` 是生成请求，随后障碍排除等操作可能减少实际目标数；实际分母取生成后的 `total_weed_count`。
- HIF 磁盘为二维弧度场，`-1` 无效；加载后转换到图像坐标轴向向量。图像旋转、HIF 旋转、标签与噪声位姿必须保持同一约定。

### 环境重构前的已核验边界

| 当前行为 / 缺口 | 对开发或研究的影响 | 当前证据 |
| --- | --- | --- |
| `Pasture-real` 注册到模块内导入的普通 v2 `CppEnv`，实际类名为 `CppEnvReal` | 注册成功不能证明真机 position / direction 接口已接入；新版 RealAgent 也未由默认场景工厂选择 | `envs/__init__.py`、`envs/cpp_env_real.py`、`AgentCreator.generate` |
| CPU APF 只检查展平索引边界，邻居可能跨行 | CPU 核不能作为精确欧氏距离真值，不能假定与 CuPy EDT 等价 | `src/cpu_apf.cpp`；2×3 探针得到 1.0，而 EDT 为 √5 |
| v6 双倍角使用 `2 * radians(direction) % pi` | 0° 和 90° 的编码相同，方向奖励存在语义缺陷 | `_generate_time_diffusion_maps`；正交角探针相同 |
| APF 在动作后观测阶段重建，再用于比较前后位置 | 改观测调用时机可能改变 reward；目标图使用哪个时刻需明确 | v2 `_get_observation_maps`、`APFCalculator.calculate` |
| field / weed 使用当前车体多边形，overlap 使用前缘扫掠四边形 | 覆盖与重复覆盖统计的几何区域不同，不能视作同一面积口径 | `FieldCoverageUpdater`、`WeedUpdater`、`CoverageOverlapUpdater` |
| `total_field_area` / `original_field` 先保存，随机障碍分支随后改 field | 初始分母与可覆盖区域需要重新校验 | `FieldCreator.generate`、`ObstacleCreator._generate_obstacles` |
| `update_config()` 仅赋值，ActionProcessor 预计算速度表 / range | 动态修改动作参数不自动重建全部相关状态与空间 | `CppEnvBase.update_config`、`ActionProcessor.__init__` |
| 动力学只在构造时排序；`add_updater` 直接插入 | 新增或替换 updater 不自动按声明依赖重排 | `EnvironmentDynamics.__init__/add_updater` |
| 新 wrapper 仍访问 `map_weed`、`map_frontier`、`get_reward`、`render_map` 等旧属性 | 文件放在 `envs_new/wrapper` 不代表兼容新版环境 | `reward_tracker.py`、`feature_tracker.py` |

这些是本次核验的具体边界，不是全部环境缺陷清单。修复动作、覆盖几何、APF、HIF 或任务分母会改变实验定义，应同时更新环境 / 配置版本与对照基线。

## 训练内部数据流

### 入口选择与旧算法

根 README 列出了三个连续 SAC 入口与默认配置。它们不是同一实现的别名：

- `sac.py` 直接创建普通 CNN actor / critic、PRB、MultiAsyncCollector 和 SACLoss，没有执行 Phase 日程；YAML 中存在 `curriculum` / `hif` 字段不表示该入口消费了它们。
- `sac_curriculum_optimized.py` 使用日程与普通更新函数，默认 HIF 预训练配置。其阶段切换仍向 `ScheduleState` 传 `consec_completion`、`consec_stable`、`last_ratio95`，与当前 dataclass 不匹配。
- `sac_curriculum_optimized_acc_loss.py` 使用同一日程、梯度累积函数和正确的 `pass_streak` / `prev_metric` 字段；默认配置是 v2、CNN，关闭课程、HIF 和分桶。`config-v2-refine.yaml` 才是另一份启用课程的 v2 配置。
- `rl_new/dqn/dqn_train.py` 和 `rl_new/sac/sac_train.py` 直接读普通 YAML，导入旧模型与旧环境工厂；DQN 使用 QValueActor / EGreedy 和可选 value rescale，离散 SAC 使用 OneHotCategorical / DiscreteSACLoss。当前 `configs/env_config.yaml` 是连续动作，启动前必须重新匹配算法动作空间。
- `rl/sac_cont/` 仍包含普通连续 SAC、v4/v5 覆盖任务、评估和 `sac_cont_reaLtest.py`。`preview/sac-async.py` 属另一实验入口，不是已统一的主循环。

### 主线职责与执行顺序

| 环节 | 当前文件 / 符号 | 契约 |
| --- | --- | --- |
| 环境 → TensorDict | `env_utils.make_env_lambda` | `gym.make → GymWrapper`；尝试注册 overlap info，注册失败只警告并继续 |
| 批量环境 | `_create_parallel_env`、`make_train_environment` | `ParallelEnv(serial_for_single=True)`，再加 InitTracker / StepCounter / Steps95 / DoubleToFloat / RewardSum |
| HIF 监督 | `HIFBidirectionalEnv` | v5/v6 且 `cfg.hif.enabled` 才提取三通道 `label_ego_hif` |
| 策略与 Q 网络 | `model_utils` | 从环境 spec 推导输入 / 动作维度、把 action bounds 移到目标设备、懒初始化并关闭用于建模的环境 |
| 采集器 | `train_utils_optimized.create_collector` | MultiAsyncCollector；CPU 环境与存储，policy 在传入 device，每个 collector 再由 env_per_collector 决定环境数 |
| 回放 | `create_replay_buffer`、`bucketed_replay.py` | CPU memmap 存储；可选 n-step；sample 转训练设备 |
| 更新 | `sac_utils_optimized` | SACLoss + 可选 HIF loss，backward / optimizer / SoftUpdate，返回日志与优先级所需结果 |
| 调度 | `Phase`、`ScheduleState`、`apply_phase` | 阶段控制环境参数、采样比例、loss 权重与学习率，切换时清空 replay 并重建 collector |
| 评估 | `AsyncEvaluator`、`evaluate_policy_standalone` | 保存模型快照，线程池执行独立 CPU 评估，返回指标与阶段名 |
| 收尾 | 各入口主函数 | 回收 collector / evaluator，记录末尾评估；异常路径和完整续训尚未统一 |

```text
collector batch → 保留 transition 的 next / done / completion → replay.extend
  → 按更新次数 sample → loss → backward → optimizer / target update → 更新优先级
  → 记录回合指标 → 保存并提交评估 → 收取结果 → 可能切换 Phase
  → collector.update_policy_weights_()
```

`env_device` / `policy_device` 与原始 Gym 环境属性不是同一件事。v2 APF 后端读取的是原始环境的 `self.device`；主工厂只把 device 传给 GymWrapper，没有给原始环境设置它。不能据 `policy_device=cuda` 宣称 APF 已在 GPU 上运行。

### Phase、HIF 与回放

`build_training_schedule()` 生成可选 PRETRAIN 加 STAGE 列表。关闭 `curriculum.enabled` 时仍从配置最后一个 stage 构造日程，不是完全不读课程配置；其中 `min_completion` / `consecutive_k` 为必需字段。PRETRAIN 按更新次数推进，STAGE 过滤匹配 `phase_name` 的评估结果，检查完成率、连续次数和可选稳定性；最后一阶段不再推进。

HIF 预训练 loss 运行 actor、计算方向重建并写 `td_error`；联合阶段计算 SAC 加加权 HIF。标签是 `[cos2, sin2, confidence]`，预测为两通道；`HIFReconstructionLoss` 计算置信度加权余弦误差与可选 TV。环境 HIF 奖励系数和网络 HIF loss 权重是两套参数，不可混同。

梯度累积函数以 `grad_accum_steps` 决定 optimizer / target update 时机；不是每次 sample 都完成一次参数更新。acc_loss 的预训练计数只在 `step_taken` 后增加。跨 Phase 的残余梯度、AMP / compile / CudaGraph 及 HIF 联合阶段均需独立实测，当前配置解析通过不能代替这些验收。

分桶回放按 transition 分类：

- SUCCESS：`next.done` 且 completion 达阈值且不是 truncated；没有额外检查 `finished` / `crashed`。
- NEAR_END：completion 达近终点阈值且不属于 SUCCESS；其余 MID。
- 容量固定约 1:1:2；配置里的 `bucket_capacity_ratio` 没有被该实现读取。
- 抽样异常时 SUCCESS / NEAR_END 回退 MID；MID 为空或失败仍会报错，不存在“任何桶状态都能安全抽样”的保证。
- 返回 `bucket_id` + `index`，优先级必须回写原桶；阶段切换会删除并重建三个子存储目录。
- n-step 在展平、按桶分流后的数据上如何保持轨迹连续性，需要针对实际路径验证，不能只验证 transform 能构造。

### 评估、视频与 checkpoint

训练内评估在 CPU 上运行确定性 rollout。视频链为 `KeepLastPixels → VideoRecorder → DropPixels`，另含回合统计 transforms。v5/v6 的评估包装按版本注入 `pred_ego_hif`，与标签提取包装是不同路径；`cfg.hif.enabled=false` 不会自动取消按版本安装的预测注入，需核查策略输出。

`AsyncEvaluator` 用 ThreadPoolExecutor，按提交顺序释放结果。失败会打印并跳过，队列没有容量上限；因此“没有返回失败结果”不能当作所有评估成功。当前它以 5 个位置参数调用回调，而 `sac.py` 导入的非 optimized `evaluate_policy_standalone` 只接受 4 个位置参数，是可由当前签名直接确认的接线问题。

| 保存入口 | 当前保存内容 | 能说明什么 |
| --- | --- | --- |
| `sac.py` | `torch.save(actor_critic, ...)`，包含 Python 模型对象 | 依赖类路径与反序列化兼容；不等于完整训练快照 |
| 两个课程入口 | `{'actor': actor.state_dict(), 'critic': critic.state_dict()}` | 可供匹配架构重建权重；没有 optimizer、target、alpha、replay、RNG、Phase 等完整续训状态 |
| `rules_new2/policies_learning.py` | 读取多种 actor 权重容器，用训练工厂建模 | 明确匹配 architecture / env spec；不是任意历史 checkpoint 的转换器 |

Hydra 配置开启 `job.chdir: true`。连续 SAC 的 `checkpoints/` 位于本次 Hydra 输出目录内，不总是根目录的 `ckpt/`。课程入口文件名使用 `model_step...`，基础入口使用 `model_s...`；watcher 只匹配前者。仅处理可信来源的 pickle / `.pt` / 对象 NPY。

`sac.py` 在使用非空 `buffer.temp_dir` 时会递归删除已有目录；清理工具 `rl_new/utils/buffer_cleanup.py` 默认也是真实删除。运行与重构时必须确认具体目录所有权，不能把公共数据目录当临时目录。

## 规则与学习策略评估

来源：[coverage_planners_v2.py](../rules_new2/coverage_planners_v2.py)、[eval_runner.py](../rules_new2/eval_runner.py)、[metrics_io.py](../rules_new2/metrics_io.py)、[policies_learning.py](../rules_new2/policies_learning.py)。

| 对象 | 业务含义 |
| --- | --- |
| RectifiedFrame | 以 field 最小包围矩形长边建立坐标系；由车宽、速度 / 转角上限推导条带宽、转弯半径和 FOV 间距 |
| PathExecutor / Pose | 用航点序列及 pure-pursuit 风格跟踪输出 `(v, w)`；Dubins 段与普通条带的重新锚定规则不同 |
| WeedTracker | 记录进入 FOV 且尚未清除的 weed 集合 |
| BCP | 固定宽度的往复条带覆盖 |
| JUMP | 从条带跳转到 weed，再返回条带 |
| SNAKE | 跳到 weed 后平移后续条带继续扫 |
| R_SNAKE | SNAKE 增加检测目标的 y 范围约束 |
| REACT | 随机探索，按检测 FIFO 访问 weed；有路径长度预算 |

这些是当前实现的操作描述，不表示已验证与某篇论文完全等价。共同使用的车体、视野、转弯半径和覆盖几何仍应统一标定。

评估先 reset 获取包围框，再以相同 seed / 场景参数和统一初始位姿进行第二次 reset；规则 planner 使用 `reset(place_agent=False)` 避免再次瞬移。YAML 的 `scenes.levels[].weed_num` 和 `scenes.weed_dists` 列表经任务展开后，分别转换为环境接收的 `weed_count` 和 `weed_distribution`。

运行记录保存 `meta`、奖励、完成率、overlap、累计路径长度、位置序列、终止原因及阈值路径长度。`Lxx` 是第一次达到阈值时的路径长度，未达到存 `-1`；汇总将负值转换为 NaN。因此 Lxx 均值与“所有回合的成功率”是不同统计口径。

`eval_watch_learning.py` 扫描 `model_step*.pt`，等待文件稳定后评估，并以 `step_meta.json` 是否存在决定是否已处理。`merge_eval_outputs.py` 合并配置与 NPY / PNG，优先硬链接再复制，重算汇总而非拼旧 CSV。`visualize_eval_scenes.py` 输出初始场景概览。

### 当前评估边界

- **短评估已复现终止归类缺口。** 当 `gym.make(max_episode_steps=8)` 的外层 TimeLimit 先截断时，内部 `env_state.timeout` 尚未变真；collector 在 truncated 处退出，未到下一轮设置 `hit_step_cap`，最终可能写 `unknown`。本次五种策略均出现此结果，不能把它计作成功或丢掉。
- 增量缓存签名包含方法配置、环境配置、场景等，但不包含代码、地图内容、checkpoint 文件内容的哈希；同路径替换内容可能复用旧结果。
- 学习 actor 的 LRU 缓存也按路径和配置缓存；同路径替换 checkpoint 不等于自动重新加载。
- `load_runs()` 会从当前默认 weed_coverage 地图重读面积；汇总又计算配置地图集合的参考面积。旧 NPY 搭配修改后的地图，或使用自定义 map_dir，不能视为原实验几何的忠实重放。
- watcher 即使某批任务出错也会写 step 元信息，而 processed 检查只看文件存在；已处理不代表任务完整成功。
- 合并配置对同名 method 保留先出现者，没有完整比对 checkpoint / 模型配置；硬链接也不是隔离的备份副本。
- `max_steps: -1` 取消评估步数上限；当前 `opcpp_eval.yaml` 不是轻量测试配置。运行前显式固定范围和输出目录。

## 地图、可视化与真实实验工具

| 范围 | 当前入口 | 使用边界 |
| --- | --- | --- |
| HIF 生成 | `envs_new/tools/generate_hif_from_field_shape.py` | harmonic / EDT 方向场；有 dry-run 与路径限制，生成参数和失败回退需随数据保存 |
| 室内地图 | `convert_houseexpo_to_indoor_coverage.py`、`convert_pgm_to_indoor_coverage.py` | 输出配对 field / obstacle；脚本内存在机器绝对路径，不能直接当通用导入 CLI |
| 地图展示 | `visualize_weed_field_gallery.py`、`rules_new2/visualize_eval_scenes.py` | 展示地图或初始场景，不证明动态任务可完成 |
| 交互可视化 | `utils/visualizer/{feature,reward,video}_visualizer.py` | GUI 与旧 wrapper 契约需单独适配；不可由 env.render() 冒烟推断整个 GUI 可用 |
| 实测资料处理 | `utils/real_world_video_process/` | 视频追踪、地图拼接、轨迹与标注处理；不等于机器人控制链验证 |
| 真机状态输入 | `envs/cpp_env_real.py:CppEnvReal`、新实体 `RealAgent` | position / direction 适配仍需保留；注册与工厂接线存在上述问题，未做硬件验收 |

## 本次运行核验

除 rules CLI 的 forkserver 需要宿主机进程通信外，检查使用项目 `.venv`、CPU、禁用在线 W&B 和图形界面；没有修改生产代码、锁文件或依赖，没有跑完整训练或全量测试。

| 检查 | 结果 | 证明范围 |
| --- | --- | --- |
| CodeGraph 本地 status / query / explore | 通过；索引 complete，pending refs / changes 为 0 | 索引和本地查询可用 |
| 实际 MCP `codegraph_explore` | 修复后通过 | 网关实际请求可用，不只是 CLI |
| 六个新环境 reset / 2 step / render | 6/6 通过 | 下述小配置、单 seed 的观测空间和有限奖励 |
| v2 观测范围 + transforms | 18 passed | `test_observation_ranges_v2.py`、`test_env_transforms.py` 的已有用例 |
| v6 默认观测空间 | 1 passed | `test_gym_space_contains_v6.py`，seed 0..9，每次 reset 后 step 一次 |
| 旧训练日程测试 | 10 failed | fixture 缺 stage `min_completion`，并使用旧 Phase 接口；尚未修测试或调度 |
| 三入口 Hydra 配置解析 | acc_loss 直接通过；另两个加已有兼容函数后通过 | 没有进入训练函数 |
| 五策略 8 步 CLI | 退出 0；5 NPY + summary / meta | 构造、执行、落盘与汇总；全部 completion=0，done_type=unknown |
| watch / merge / visualize `--help` | 均退出 0 | 模块与 CLI 参数入口，未执行对应业务 |
| CPU APF / v6 方向微型探针 | 复现上述两处数学问题 | 不能将环境冒烟通过当作任务数学正确 |
| `uv pip check` | 退出 1 | 两项依赖缺口仍存在 |

### 环境冒烟的准确配置

每个 v1–v6：`reset(seed=0)`，`action_space.seed(0)`，随机 step 两次，再 render；检查 reset / step 的 `observation_space.contains()`、reward 有限和渲染数组。

```python
kwargs = dict(
    num_obstacles_range=(0, 0),
    state_size=(64, 64), state_downsize=(32, 32),
    use_multiscale=False, use_global_features=False,
    render_mode="rgb_array",
)
```

| v1–v6 顺序 | v1 | v2 | v3 | v4 | v5 | v6 |
| --- | --- | --- | --- | --- | --- | --- |
| 图像通道数 | 4 | 5 | 5 | 5 | 8 | 6 |
| vector 长度 | 4 | 4 | 4 | 14 | 14 | 98 |
| 渲染尺寸 | 800×800×3 | 800×800×3 | 800×800×3 | 800×800×3 | 800×1600×3 | 800×1600×3 |

图像观测空间为 `(通道数,32,32)`；各版本还包含 `(1,)` completion_ratio。v5/v6 双面板是 GT / Pred 可视化；没有预测时右侧是占位面板。

### 重跑已有检查

根 README 给出 19 个通过用例的命令。检查仍未修复的旧日程测试：

```bash
CUDA_VISIBLE_DEVICES='' WANDB_MODE=disabled SDL_VIDEODRIVER=dummy MPLBACKEND=Agg \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/test_training_schedule.py
```

该命令当前应暴露失败；不要为了让文档显得完整而隐藏它。全量 `tests/` 中还有 import 时运行循环、引用旧 rules 实现的脚本；本次未执行全量 pytest 或全量 collect-only。

五策略 smoke 的可复建 YAML 如下，另存为临时文件；运行命令为 `.venv/bin/python -m rules_new2.eval_runner --config <临时YAML>`。使用一个全新的 output_root；下面的 `overwrite` 会覆盖同名 smoke 结果。

```yaml
env:
  id: NewPasture-v2
  base_kwargs:
    action_type: continuous
    use_trajectory: true
    use_mist: true
    use_apf: true
    boundary_source: box
    state_size: [64, 64]
    state_downsize: [32, 32]
    use_multiscale: false
    use_global_features: false
scenes:
  weed_dists: [uniform]
  seeds: [0]
  levels:
    - {name: smoke, num_obstacles_range: [0, 0], weed_num: 50, map_ids: [0]}
methods:
  learning: []
  rules: [{name: BCP}, {name: JUMP}, {name: SNAKE}, {name: R_SNAKE}, {name: REACT}]
eval:
  mode: overwrite
  max_steps: 8
  L_thresholds: [0.9, 0.95, 0.98]
  workers_rules: 1
  workers_net: 1
  render_last_frame: false
  output_root: /tmp/newcpprl-doc-rules-smoke
```

初次沙箱运行因 forkserver 创建本地 socket 被拒绝而失败；同命令在宿主机完成。该差异属于执行环境权限，不能误诊为策略或地图问题。

### CodeGraph 故障与恢复

本次 MCP 调用最初两次返回 `unhandled errors in a TaskGroup (1 sub-exception)`。将实际网关 BackendInstance 调用中的嵌套异常展开后，原因为 `CatalogMismatch: codegraph: backend serverInfo changed; refresh the catalog`：本机 CodeGraph 1.6.0，但共享网关缓存为 1.5.0。

仅刷新共享 supervisor 的 CodeGraph catalog 条目，确认工具 schema 未变、没有活跃后端，再重启网关。原 MCP `CppEnvBase` 查询随后成功。没有升级包、重建项目索引或修改项目代码。以后出现同类错误先查真实异常和缓存版本，不能从外层 TaskGroup 错误推断数据库损坏。

## 后续重构顺序与验收条件

以下是基于本次现状的建议，不表示已经执行或批准删除：

1. **固定任务契约。** 明确 weed / field 完成率、APF 时序、HIF 坐标、车体 / 扫掠几何、地图分母与零目标行为；用小型确定性场景证明修复前后的意义。
2. **收敛可运行入口。** 以实际使用的 acc_loss 路径为主检查配置 → collector → replay → update → async eval → phase 切换 → shutdown，修复 sibling 入口接线与测试接口漂移。
3. **闭合评估证据。** 统一内部 timeout 与 Gym 截断，保留失败任务；把地图 / 配置 / 代码 / checkpoint 身份与原始结果绑定，再验收增量、watch 和 merge。
4. **分离推理与续训。** 给 checkpoint 明确格式和能力；完整续训需恢复 optimizer、目标网络、熵系数、回放、随机状态、计数和课程阶段，并通过中断对照。
5. **最后迁移依赖与删旧。** 逐条解除 `rl_new → rl`、新工具 → 旧工具 / envs 的依赖；保留 DQN、离散与连续 SAC、规则、HIF、地图、可视化、真机和历史模型能力，之后再做独立安装和删除验证。

未完成的验收包括：真实采集与训练反传、长训、多 seed 策略效果、PRETRAIN / 联合 HIF / 阶段切换、AMP / compile / GPU、完整视频编码、历史 checkpoint 矩阵、断点续训、完整规则基准和真机硬件。有限 smoke 不替代这些条件。

## 文档如何保持有效

- `AGENTS.md` 保存短工作规则与关键契约；根 README 保存使用入口；本页保存架构、边界和带日期的核验快照。
- `rl_new/README.md` 只作导航，避免再维护第二份冲突的安装和入口说明。
- 源码、配置和实测结果优先；`CLAUDE.md`、`envs/doc/`、`team_reports/` 等旧材料只作线索，涉及现状时重新核验。本次没有改写这些历史材料。
- 改注册、配置、观测 / reward / done、checkpoint 或入口时更新相应段落；替换过期结论，不追加升级流水账。
- 报告源码阅读、单元检查、真实训练和研究结论时分别注明证据范围；不能用“目录已迁移”“测试通过”概括尚未完成的能力闭环。
