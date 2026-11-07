# SAC 课程学习（线性日程表）— 组件测试报告（中文）

日期：2025-11-08

范围：仅覆盖优化分支（`rl_new/sac_cont_sy/sac_curriculum_optimized.py` 及其依赖）。

本地运行命令（规避第三方 pytest 插件干扰）：

```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 new_venv/bin/python -m pytest -q tests_2
```

## 组件与“正确职责”定义

- 日程表构建（`build_training_schedule`）
  - 启动时一次性生成线性 `Phase` 列表，作为唯一真源。
  - PRETRAIN 复制“下一阶段”的 `env_params/sampling_ratio` 以避免切换抖动。
  - 四种场景映射到固定序列。

- 阶段推进判定（`maybe_advance_by_updates`，`maybe_advance_by_eval`）
  - PRETRAIN：按 `max_updates` 基于更新步数推进。
  - S1：完成度 ≥ 阈值，连续 K 次评估即推进。
  - S2：完成度 ≥ 最小阈值，且 ratio95 相对变化 < 阈值，连续 K 次评估即推进；首个样本仅建立基线。

- 损失模块（`HIFAssistedSACLoss`）
  - PRETRAIN：仅 HIF 重建；为了日志一致性，`alpha/entropy=0`；并将 HIF 的 `td_error` 写回输入 TD，供 PRB 使用。
  - STAGE：SAC 三项 + 可选 `hif_weight*HIF`；不覆盖输入 TD 的 `td_error`（由 SAC 计算）。

- 分桶优先回放（`BucketedTensorDictPrioritizedReplayBuffer`）
  - 按 `next.done/truncated/completion_ratio` 路由到 SUCCESS/NEAR_END/MID。
  - 采样按比例，桶不足时回退到 MID；采样结果包含 `bucket_id`，优先级更新按 `bucket_id` 路由。

- 异步评估器（`AsyncEvaluator`）
  - 可并行评估，但保证返回顺序与提交顺序一致。

## 测试结果摘要（tests_2）

- schedule_build.py
  - 通过：四种场景的阶段序列与预期完全一致。
  - 通过：PRETRAIN 的 `env_params/sampling_ratio` 等于下一阶段。

- transition_logic.py
  - 通过：S1 连续完成度判定逻辑。
  - 通过：S2 “完成度 + 稳定性(ratio95)” 逻辑与基线处理（首次不推进）。
  - 通过：PRETRAIN 的更新步数门槛推进逻辑。

- loss_module.py
  - 通过：PRETRAIN 输出的 SAC 相关项为 0，并写入 `td_error`。
  - 通过：STAGE 合成 SAC 与（可选）HIF；不改写输入 TD 的 `td_error`（返回的 out 中可含 HIF 指标，不影响 PRB）。

- bucketed_replay.py
  - 通过：路由计数与采样比对（SUCCESS/NEAR_END/MID）。
  - 通过：采样 `bucket_id` 与按桶优先级更新。
  - 说明：在部分沙箱禁用 `/dev/shm` 时该用例跳过，属环境限制，非逻辑问题。

- static_gate_assignment.py
  - XFAIL（信息性护栏）：检测到 `rl_new/sac_cont_sy/sac_curriculum_optimized.py` 中存在
    `replace(state, ..., update_gate_frames=...)` 的模式。绝对“热身门”应设置在下一阶段的 `Phase` 上：
    `schedule[new_idx] = replace(schedule[new_idx], update_gate_frames=...)`。
    原因：`ScheduleState` 并不包含 `update_gate_frames` 字段；该门槛属于阶段属性。

## 说明与建议

- 阶段切换后的“绝对热身门”
  - 当前代码在切换时对 `state` 写入 `update_gate_frames`，这不是其字段；推荐改为更新下一阶段 `Phase`：
    `schedule[new_idx] = replace(schedule[new_idx], update_gate_frames=collected_frames + cfg.collector.init_random_frames)`。
  - 这样可保持“门槛属于 Phase”的语义一致，并使循环内判断更直观。

- 评估处理
  - 循环顶部仅消费一次 `async_evaluator.get_evaluate_results()` 并统一记录日志；仅在启用课程学习时驱动推进，设计正确。

- 优先级来源
  - PRETRAIN：`td_error` 源自 HIF；
  - STAGE：`td_error` 由 SACLoss 写入输入 TD；测试确认未被 HIF 分支覆盖。

在最小 CPU 测试架构下，其余组件行为均与预期一致。
