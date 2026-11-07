# NewCppRL

面向割草机器人、杂草清除与区域覆盖路径规划的强化学习研究项目。环境使用 Gymnasium，训练基于 PyTorch / TorchRL，APF 距离传播和视野计算包含 pybind11 C++ 扩展。

**当前是新旧实现并存的研究工作树。** 连续 SAC、课程学习与 HIF 辅助学习集中在 `rl_new/sac_cont_sy/`；DQN、离散 SAC、规则基线、真机适配、历史模型和可视化仍是项目能力的一部分。`_new` 表示代码组织的演进，不表示迁移已完成，也不表示旧代码可以删除。

- 开发代理先读 [AGENTS.md](AGENTS.md)。
- 架构、调用关系、已核验问题、验证结果和重构顺序见 [当前项目说明](docs/project-current-state.md)。
- 本页依据 **2026-09-07 的当前工作树**更新；旧迁移文档和升级日志不能替代当前源码。

## 从哪里开始

| 范围 | 当前职责与入口 |
| --- | --- |
| [envs_new/](envs_new/) | 组件化环境；注册表 `__init__.py`，主流程 `cpp_env_base.py`，变体 `cpp_env_v1.py` 至 `cpp_env_v6.py` |
| [envs/](envs/) | 历史 `Pasture-*` 环境、旧地图工具和真机适配；与新环境的接口、任务语义不同 |
| [rl_new/sac_cont_sy/](rl_new/sac_cont_sy/) | 连续 SAC 的三个入口、模型工厂、Phase 日程、回放、异步评估和 YAML 配置 |
| [rl_new/dqn/](rl_new/dqn/)、[rl_new/sac/](rl_new/sac/) | DQN / 离散 SAC；仍引用 `rl/` 和 `torchrl_utils/`，不是独立的新环境训练栈 |
| [rl/](rl/) | 历史 DQN、离散 SAC、连续 SAC、覆盖任务和实测脚本；仍有被新栈调用的实现 |
| [rules_new2/](rules_new2/) | BCP、JUMP、SNAKE、R_SNAKE、REACT；与学习策略共用场景评估、原始结果和汇总流程 |
| [torchrl_utils/](torchrl_utils/)、[torchrl_utils_new/](torchrl_utils_new/) | 模型、环境包装、DQN loss、评估与视频工具；主线 CNN / ResNet-FPN 仍在无 `_new` 的目录 |
| [src/](src/)、[envs_new/tools/](envs_new/tools/)、[utils/](utils/) | C++ 扩展，地图 / HIF 生成，特征 / 奖励 / 视频可视化及实测视频处理 |

## 环境与任务

先 `import envs_new` 再使用 `gym.make("NewPasture-vN", ...)`。当前六个版本均已注册。

| ID | 任务完成率 | 默认特征 |
| --- | --- | --- |
| `NewPasture-v1` | 已清除 weed 比例 | 无迷雾、无 APF |
| `NewPasture-v2` | 已清除 weed 比例 | 迷雾探索、APF 观测与奖励；提供 `overlap_count` |
| `NewPasture-v3` | 已清除 weed 比例 | 迷雾探索，无 APF |
| `NewPasture-v4` | 已覆盖 field 比例 | 车体覆盖、重复覆盖热图、覆盖顺序、位姿与动作历史 |
| `NewPasture-v5` | 已覆盖 field 比例 | v4 加 HIF 方向场观测、奖励、监督标签与预测可视化 |
| `NewPasture-v6` | 已覆盖 field 比例 | v4 观测加时空轨迹权重；不直接输入 GT HIF，仍用 HIF 计算奖励、标签和渲染 |

HIF 是轴向方向场，磁盘数据为弧度，`-1` 表示无引导。v5/v6 需要匹配地图的 HIF 文件。机器人动作仍是速度与转角，v6 并未实现“轨迹动作”。

新环境返回 `observation`、`vector`、`completion_ratio`；shape 取决于版本和观测配置。`completion_ratio` 是观测键，不是所有版本都放在 info 中的通用字段。v2 的 overlap 指标已经存在，但 overlap 热图尚未输入其观测。

`Pasture-vN` 与 `NewPasture-vN` 不能互换。同一版本号不保证相同任务、观测或模型兼容性。`envs_new.make_env()` 使用另一套工厂注册表，默认只支持 base / v1 / v2 / v3 及其别名；v4–v6 使用 Gym ID。

## Python 环境

从仓库根目录工作。当前 `pyproject.toml` 要求 **Python ≥ 3.14**，用 `uv.lock` 管理大部分依赖，PyTorch 默认索引为 cu130。`setup.py` 构建 `cpu_apf` 和 `cpu_fov` 两个扩展。

本机核验快照：Python 3.14.5，torch 2.12.0+cu130，TorchRL 0.14.0+gb295f5b011，TensorDict 0.14.3+g6fdce72。**TorchRL / TensorDict 不在 lock 中**，本机 TorchRL 还是仓库外源码的 editable 安装；版本来源和部署限制见[依赖说明](docs/project-current-state.md#依赖与部署边界)。

已有 `.venv` 时，日常运行直接使用它，避免隐式同步改动环境：

```bash
.venv/bin/python --version
.venv/bin/python -c "import torch, torchrl, tensordict, cpu_apf, cpu_fov; print(torch.__version__, torchrl.__version__, tensordict.__version__)"
# 等价的 uv 运行形式：uv run --no-sync python ...
```

需要同步 lock 中依赖时，先检查预演，再决定是否执行：

```bash
uv sync --locked --inexact --dry-run
# 确认预演后：uv sync --locked --inexact
uv pip check
```

不要直接用默认 `uv sync` 维护这个源码依赖环境：本次预演会卸载 TorchRL、TensorDict 等 lock 外包。`--inexact` 保留它们，但不会补齐它们的依赖。本次 `uv pip check` 仍报告 `pyvers` 版本不足及缺少 `hoptorch`。

新机器部署需要两步：安装 lock 中依赖并构建项目扩展，再按记录的源码版本构建 TorchRL / TensorDict 并核验依赖。**本次未验证干净机器安装、独立 wheel 或其他 CUDA 平台**；当前 `.venv` 能运行不代表打包部署完整。旧 `environment.yaml` 仅作历史参考。

## 最小环境检查

以下小配置用于确认 v2 的 reset / step 通路，不是训练或性能基准；依赖仓库自带地图。

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python - <<'PY'
import gymnasium as gym
import numpy as np
import envs_new

env = gym.make(
    "NewPasture-v2", render_mode=None,
    num_obstacles_range=(0, 0),
    state_size=(64, 64), state_downsize=(32, 32),
    use_multiscale=False, use_global_features=False,
)
try:
    obs, _ = env.reset(seed=0)
    env.action_space.seed(0)
    assert env.observation_space.contains(obs)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs) and np.isfinite(reward)
    print({k: v.shape for k, v in obs.items()}, reward, sorted(info))
finally:
    env.close()
PY
```

指定场景使用 `env.reset(options={...})`：当前接受 `map_id`、`specific_scenario_dir`、`weed_count`、`weed_distribution`、`initial_position`、`initial_direction`。旧的 `weed_num` / `weed_dist` 键会被忽略。普通地图布局和 `map_id` 含义见[地图契约](docs/project-current-state.md#地图与坐标契约)。

## 连续 SAC 入口

以下文件都在 `rl_new/sac_cont_sy/`。旧文档中的 `sac_curriculum.py` 和 `train_utils.py` 已不存在。

| 入口 | 默认 YAML / 环境 | 当前用途与边界 |
| --- | --- | --- |
| `sac.py` | `config-sync-server` / v2 | 基础连续 SAC；没有接入 Phase 调度，且当前异步评估回调签名不匹配 |
| `sac_curriculum_optimized.py` | `config-sync-server-hif-pretrain` / v6 | HIF 预训练与课程实验；切换代码仍使用已移除的状态字段 |
| `sac_curriculum_optimized_acc_loss.py` | `config-sync-server-hif-v2` / v2 | Phase 调度、CNN / ResNet 双头和梯度累积；默认 HIF、课程、分桶均关闭 |

当前优先从第三个入口理解日程实现，但它仍需完整训练、阶段切换和评估验收。先查看配置，不启动训练：

```bash
.venv/bin/python rl_new/sac_cont_sy/sac_curriculum_optimized_acc_loss.py --cfg job
.venv/bin/python rl_new/sac_cont_sy/sac_curriculum_optimized_acc_loss.py \
  --config-name config-v2-refine --cfg job
```

正式启动采用同一入口并去掉 `--cfg job`。运行前必须核对 `training.device`、`collector.*`、`buffer.temp_dir`、`logger.*` 和 `hydra.run.dir`；现有配置含服务器绝对路径、较大训练预算和在线 W&B 设置，不宜直接照默认值启动。本次仅验证配置解析，没有执行长训。

Python 3.14 + Hydra 1.3.2 下，前两个入口直接解析配置会报 `badly formed help string`；第三个已内置兼容函数。需要检查前两个时可使用现有函数，示例：

```bash
.venv/bin/python -c "from rl_new.hydra_compat import install_hydra_py314_compat; install_hydra_py314_compat(); import runpy; runpy.run_module('rl_new.sac_cont_sy.sac', run_name='__main__')" --cfg job
```

DQN / 离散 SAC 使用 `configs/train_dqn_config.yaml` / `configs/train_sac_config.yaml` 与 `configs/env_config.yaml`，并非上述 Hydra 参数体系；当前旧环境配置是连续动作，不能直接当作 DQN 可运行配置。

## 规则与学习策略评估

[rules_new2/eval_runner.py](rules_new2/eval_runner.py) 读取普通 YAML，组合 method × level × map × weed distribution × seed，执行评估并写出逐回合 NPY、可选 PNG 和汇总。

```bash
.venv/bin/python -m rules_new2.eval_runner --help
.venv/bin/python -m rules_new2.eval_watch_learning --help
.venv/bin/python -m rules_new2.merge_eval_outputs --help
.venv/bin/python -m rules_new2.visualize_eval_scenes --help
```

准备独立配置后，用 `.venv/bin/python -m rules_new2.eval_runner --config <配置文件>` 运行。现有模板是 [opcpp_eval.yaml](rules_new2/configs/opcpp_eval.yaml)，其 `max_steps: -1` 取消步数上限；先缩小场景集、设置有限步数并使用新输出目录。学习策略还需提供 checkpoint、匹配的 architecture / env 配置及设备。

`L90/L95/L98` 是首次达到对应完成率的累计路径长度，不是成功率。`planner_idle` 不等于任务完成。增量缓存、终止分类和当前地图面积重算仍有研究有效性边界，见[评估说明](docs/project-current-state.md#规则与学习策略评估)。

## 验证与开发

当前可用的有限 CPU 回归检查：

```bash
CUDA_VISIBLE_DEVICES='' WANDB_MODE=disabled SDL_VIDEODRIVER=dummy MPLBACKEND=Agg \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  tests/test_observation_ranges_v2.py tests/test_env_transforms.py \
  tests/test_gym_space_contains_v6.py
```

本次这三份测试合计 **19 passed**；六个新环境的小配置 reset / step / render 均通过。另有 **10 个旧课程测试失败**。五种规则策略的 8 步评估已成功落盘，但均未完成除草目标，终止分类返回 `unknown`。完整验证配置及解释见[核验记录](docs/project-current-state.md#本次运行核验)。

`pytest` 默认收集 `tests/test_*.py`；部分历史测试在 import 时就执行环境循环，不能把全量收集当作无副作用操作。`test_env_consistency/` 当前不存在。

重构应先固定任务与数据契约，再收敛入口、消除反向依赖。模型权重可加载不等于完整续训；旧目录删除、历史模型兼容、真机能力和论文级结果都需要独立验收。请勿提交训练产物、视频、虚拟环境或密钥。
