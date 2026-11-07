# NewCppRL 代理工作指南

本文件约束代理如何工作；项目介绍与使用说明见 [README.md](README.md)。
当前架构、入口、依赖和已核验限制见 [项目现状](docs/project-current-state.md)。
源码与实际运行结果是事实依据。文档与源码冲突时，先核验并纠正文档，不按旧描述改坏代码。

## 工作边界

- 先理解任务对应的数据流、调用者与实际入口，再选择最小、清晰、可验证的修改。
- 新环境开发优先放在 `envs_new/`；连续 SAC 主线优先查看 `rl_new/sac_cont_sy/`。
- 项目仍包含 DQN、离散 SAC、规则策略、真机适配和历史 checkpoint 使用路径；不能缩减为单一 SAC 项目。
- `envs/`、`rl/`、`torchrl_utils/` 仍被新代码引用。“旧目录”不代表可以删除。
- 删除或替换旧实现前，核对导入、运行入口、配置、checkpoint 格式和行为等价性。
- 保留用户未提交的修改，不顺手重构、清理数据或更改无关配置。
- 记录“已验证 / 进行中 / 未完成阻塞”；导入成功、测试通过、训练成功是不同结论。
- 不把历史报告、旧命令、单次 smoke 或静态源码阅读当作当前端到端可用的证明。

## 目录导航

| 目录 | 阅读目的 |
| --- | --- |
| `envs/` | 历史 Gym 环境及仍被引用的环境、真机相关实现 |
| `envs_new/` | 组件化环境、v1–v6 变体、注册与环境工厂 |
| `envs_new/components/` | 地图、动力学、观测、奖励、状态的实现 |
| `rl/` | 旧训练入口及新栈仍复用的算法工具 |
| `rl_new/` | DQN、离散 SAC、连续 SAC 等训练实现 |
| `rl_new/sac_cont_sy/` | 连续 SAC、Phase 调度、回放、评估、配置与模型 |
| `rules_new2/` | 规则策略及其执行、评估路径 |
| `torchrl_utils/` | 新旧栈共同引用的模型及 TorchRL 工具 |
| `torchrl_utils_new/` | 新工具实现；仍依赖旧工具和旧环境 |
| `src/` | `cpu_apf`、`cpu_fov` 两个 C++ / pybind11 扩展 |
| `configs/` | 部分运行配置；连续 SAC 配置也在其算法目录 |
| `tests/` | pytest 与历史实验脚本混合；先检查再选择测试 |

目录名不能替代依赖分析。例如 `rl_new/dqn/dqn_train.py` 仍导入 `rl.dqn.dqn_utils`，
`rl_new/sac/sac_train.py` 仍导入 `rl.sac.sac_utils`，连续 SAC 模型也复用 `torchrl_utils.model`。

## 使用 CodeGraph

- 结构问题优先使用 CodeGraph：符号位置、调用关系、影响范围、模块上下文。
- 当前连接主要暴露 `codegraph_explore`；按实际工具 schema 使用，每次显式传入当前仓库绝对路径 `projectPath`。
- 其他 `codegraph_*` 工具只有在当前会话实际可调用时才使用，不能照抄不存在的工具名。
- 用有范围、有输出上限的查询取得相关符号和源码；已返回的源码不再重复读取，只补齐未展示的关键段落。
- `rg` 用于字符串、配置、日志、文档等文本搜索，以及 CodeGraph 不可用时的必要排查。
- 服务报错时保留真实错误，检查服务连接、项目路径、索引状态与日志；不要直接推断索引损坏。
- 不以重复初始化或删除 `.codegraph/` 代替诊断。确认索引缺失后，再说明原因并请求初始化。
- 编辑后给文件监听器更新的时间；查询结果与文件不一致时说明索引可能滞后。

## 环境契约

- `envs/__init__.py` 管理 `Pasture-*`；`envs_new/__init__.py` 注册 `NewPasture-v1` 至 `NewPasture-v6`。
- Gym 注册与 `EnvironmentFactory` 支持范围不同；不能根据注册 ID 推断工厂也支持同一版本。
- `CppEnvBase` 当前使用 `EnvironmentConfigOriginal`；修改配置时确认实际实例化的配置类。
- 状态由 `env_state` 管理，地图在 `maps_dict`，机器人位姿在 `agent`；变更时保持三者一致。
- `step()` 的关键顺序是动力学更新 → 观测生成 → 奖励计算；观测生成还会更新 APF / 轨迹地图。
- 因此，观测代码并非全部为无副作用的读取；调整调用顺序前检查奖励和历史状态依赖。
- 机器人角度使用度；磁盘 HIF 使用弧度；位姿坐标为 `(x, y)`，数组索引为 `[y, x]`。

| 新环境 | 任务与关键差异 |
| --- | --- |
| v1 | 除草；无 mist、无 APF |
| v2 | 除草；有 mist、APF，也提供 `overlap_count` |
| v3 | 除草；有 mist、无 APF |
| v4 | 机器人本体覆盖；完成率按 field 计算 |
| v5 | v4 加 GT HIF 观测、奖励和标签 |
| v6 | 观测使用 v4 通道加 `trajectory_weights`；GT HIF 仍用于标签、奖励和渲染 |

- v1–v3 的完成率基于 weed，v4–v6 基于 field；不要把二者当作相同任务指标比较。
- v6 观测不输入 GT HIF；不要因为继承 v5 就断言其策略能看见 GT HIF。
- v2 有 `overlap_count` 不代表存在 overlap 热图观测；指标与观测通道分别核对。
- 当前新环境 reset 场景选项为 `map_id`、`specific_scenario_dir`、`weed_count`、
  `weed_distribution`、`initial_position`、`initial_direction`；旧键 `weed_num` / `weed_dist` 会被忽略。
- 地图、HIF、设备和真机适配要求以实际创建路径为准；仅有注册 ID 不证明可运行。

## 训练、评估与 checkpoint

- 连续 SAC 入口位于 `rl_new/sac_cont_sy/`：`sac.py`、`sac_curriculum_optimized.py`、
  `sac_curriculum_optimized_acc_loss.py`；不能恢复引用已删除的 `sac_curriculum.py` / `train_utils.py`。
- 三个入口的默认配置不同。开始运行前确认实际加载的 YAML、`env_id`、设备、日志和输出目录。
- 涉及启动兼容性时分别验证入口；一个入口具备 Python / Hydra 兼容处理，不代表其余入口也具备。
- 当前调度以 `PRETRAIN` 和 `STAGE` Phase 组织；阶段阈值包括 `min_completion`、
  `consecutive_k` 及可选 `stability_tolerance`，不能按旧 S1 / S2 / S3 描述推断当前行为。
- `ScheduleState` 的字段是 `idx`、`pretrain_updates`、`pass_streak`、`prev_metric`；
  改调度时检查各训练入口的调用者，不只检查状态类。
- 追踪真实训练路径：环境 → transforms → collector → replay → loss / optimizer → 权重同步 → 评估。
- 修改 info、终止信号、完成率或观测时，同时检查 TensorDict 键、时间维、batch 维与评估聚合。
- 评估、视频、分桶回放、预训练及阶段切换分别验收；不能用主循环启动代替这些能力的验证。
- checkpoint 的“可加载推理”和“可精确续训”分开描述；检查保存的是模型还是完整训练状态。
- `acc_loss` 入口目前保存 actor / critic 的 `state_dict`，不含 optimizer、replay、RNG 与 Phase 状态。
- 训练结论必须能回溯到入口、配置、随机种子、checkpoint、原始指标和实际评估过程。

## 修改与验证

- 优先复用现有逻辑、标准库和已安装依赖；不增加仅为未来扩展服务的包装层或配置。
- 修共享逻辑前检查调用者；在根因处修复，避免给每个入口叠加不同的补丁。
- Python 使用四空格、清晰命名；注释解释原因、约束和非显然关系，不重复代码。
- 非平凡逻辑改动保留最小可运行回归检查，重点覆盖真实行为边界。
- 环境改动至少检查受影响版本的 reset / step、观测 shape / dtype、奖励和终止语义。
- 训练改动先验证配置加载，再跑小规模真实采集与更新；涉及阶段切换或评估时覆盖对应路径。
- 优先运行与改动相关的测试；执行大范围测试、训练或 GPU 作业前检查其资源需求和现有任务。
- 缺依赖、设备、地图、HIF 或外部服务时明确阻塞；不得把跳过或降级冒充测试通过。
- 文档改动校验文件路径、符号、配置和命令来源；不为纯排版修改新增测试。
- 不提交 checkpoint、视频、日志、本地虚拟环境、凭证或密钥；大体积产物使用配置指定目录。

## 协作与文档维护

- Issues 使用 GitHub Issues，流程见 [issue-tracker.md](docs/agents/issue-tracker.md)。
- 分流标签固定为 `needs-triage`、`needs-info`、`ready-for-agent`、`ready-for-human`、`wontfix`；
  使用规范见 [triage-labels.md](docs/agents/triage-labels.md)。
- 本仓库为单一领域上下文，术语与领域规则见 [domain.md](docs/agents/domain.md)。
- Commit 使用聚焦的 Conventional Commits；PR 说明具体行为变化、复现或验证结果及剩余限制。
- AGENTS 只保留工作规则、导航与关键契约；README 放使用入口，项目现状页放架构证据与限制。
- 不把会话日志、瞬时机器版本、过往故障清单或大段源码复制进 AGENTS。
- 改注册、入口、配置键或关键语义时同步相应文档；更新过期段落，不在其后追加相互矛盾的新说明。
- 每个“支持”“可用”“已修复”的结论都应有当前源码或运行证据；未运行的命令标记为未验证。
