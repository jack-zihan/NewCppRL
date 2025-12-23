# sac_curriculum_optimized_acc_loss.py 策略评估与指标体系深度分析

> 目标：搞清楚 **新环境 + 新模型 + 新训练方法** 下，“策略好不好”究竟是怎么被度量和驱动课程切换的，从训练脚本、异步评估器到评估环境和指标的整条链路一眼看懂。

---

## 1. 评估在整体训练框架中的位置

### 1.1 核心参与模块

围绕 `rl_new/sac_cont_sy/sac_curriculum_optimized_acc_loss.py`，策略评估链路涉及：

- 主训练脚本：`sac_curriculum_optimized_acc_loss.py`
  - 负责训练循环、课程阶段调度、何时触发评估、如何消费评估结果。
- 异步评估管理：`async_evaluator.AsyncEvaluator`
  - 管理评估任务提交与并发执行，保证结果按 `step` 有序返回。
- 评估实现：`sac_utils_optimized.evaluate_policy_standalone` / `evaluate_policy`
  - 建立与训练一致的 Gym/TorchRL 环境；
  - 执行确定性 rollout，计算覆盖相关指标；
  - 如开启视频，则生成覆盖过程 mp4 并挂到 W&B。
- 课程调度：`train_utils_optimized.maybe_advance_by_eval`
  - 解析评估 metrics（特别是 `completion_ratio` 与 `steps_95_to_done`），决定是否从 S1→S2→S3。

总体结构（简化）如下：

```text
sac_curriculum_optimized_acc_loss.py
  ├─ while collected_frames < total_frames:
  │     ├─ 收集数据 + 更新模型
  │     ├─ if is_time_to_evaluate(...):
  │     │      ├─ 保存 {actor, critic} → model_stepXXXX_eval_pending.pt
  │     │      └─ AsyncEvaluator.submit_eval(evaluate_policy_standalone, ...)
  │     └─ eval_results = AsyncEvaluator.get_evaluate_results()
  │            ├─ log_evaluate_results(eval_results, checkpoint_dir, logger)
  │            └─ state, adv_eval = maybe_advance_by_eval(..., eval_results, ...)
  └─ 训练结束: AsyncEvaluator.shutdown(wait=True) + log_evaluate_results(remaining)
```

评估在这里承担两件事：

1. 对当前策略生成稳定的指标（奖励/完成率/重叠/steps_95_to_done）。
2. 提供课程学习的阶段切换信号（例如“完成率持续≥阈值且路径稳定”）。

---

## 2. 训练脚本中的评估逻辑（主线视角）

### 2.1 Logger 与 AsyncEvaluator 初始化

位置：`sac_curriculum_optimized_acc_loss.py` 中 main() 的前半段。

```python
# Logger（可选 W&B）
if cfg.logger.backend:
    logger = get_logger(..., wandb_kwargs={"mode": cfg.logger.mode, ...})

# 定义评估指标用独立 x 轴
if logger is not None:
    logger.experiment.define_metric("eval_step")
    logger.experiment.define_metric("eval/*", step_metric="eval_step")

# 异步评估器
async_evaluator = AsyncEvaluator(max_workers=cfg.logger.eval_worker)
```

要点：

- 所有 `eval/*` 指标使用 **独立的 `eval_step` 轴**，避免与训练 `global_step` 搅在一起，W&B 上会有两套清晰的曲线。
- `AsyncEvaluator` 由 `cfg.logger.eval_worker` 控制并发度，默认 1–2 线程即可。

### 2.2 何时触发评估：`is_time_to_evaluate`

在每个训练批次更新结束后，脚本检查是否该进行一次策略评估：

```python
if is_time_to_evaluate(current_frames, collected_frames, cfg):
    model_path = checkpoint_dir / f"model_step{collected_frames:08d}_eval_pending.pt"
    torch.save({'actor': actor.state_dict(),'critic': critic.state_dict()}, model_path)
    async_evaluator.submit_eval(
        evaluate_policy_standalone,
        str(model_path.absolute()),
        copy.deepcopy(cfg),
        collected_frames,
        schedule[state.idx].name,
    )
    torchrl_logger.info(f"提交评估任务: {collected_frames} (阶段: {schedule[state.idx].name})")
```

`is_time_to_evaluate`（定义在 `sac_utils_optimized.py`）通常基于：

- `cfg.logger.eval_interval`（帧/steps 间隔）；
- 是否到达初始 warmup 之后；
- 是否处于启用课程学习的阶段等。

这保证：

- 评估任务 **不会过于频繁**（避免压垮 CPU）；
- 评估间隔与课程的“阶段决策窗口”对齐。

### 2.3 如何消费评估结果与驱动课程切换

在每轮训练循环顶部：

```python
eval_results = async_evaluator.get_evaluate_results()
if eval_results:
    log_evaluate_results(eval_results, checkpoint_dir, logger)
    state, adv_eval = maybe_advance_by_eval(
        state,
        schedule[state.idx],
        eval_results,
        cfg,
        cfg.curriculum.enabled,
    )
    if adv_eval: should_transition = True
```

这里有三个关键动作：

1. `get_evaluate_results()`：
   - 从后台线程池拉取已完成的评估；
   - 保证按 `step` 有序返回（详细见第 3 节）。
2. `log_evaluate_results(...)`：
   - 把 `eval_metrics` 写到 W&B，并重命名 checkpoint 文件名，附加指标摘要；
   - 后期直接从文件名就能看出某一步的 reward / completion / steps95 / overlap。
3. `maybe_advance_by_eval(...)`：
   - 在 `train_utils_optimized` 中定义，核心逻辑：
     - 统计最近若干次评估的 `completion_ratio`；
     - 若连续 K 次 ≥ 阈值（`s1_min_completion` / `s2_min_completion`），则考虑从 S1→S2 或 S2→S3；
     - S2→S3 还会查看 `eval/ratio_95_to_done_mean` 的变化，只有“完成率高且路径稳定”才推进到更难阶段。

> 评估结论不是立即单次起效，而是通过 **“连续满足条件”** 驱动课程阶段，以过滤噪声与偶然性。

### 2.4 训练结束时的收尾评估

在主循环结束后：

```python
remaining_results = async_evaluator.shutdown(wait=True)
if remaining_results:
    log_evaluate_results(remaining_results, checkpoint_dir, logger)
```

- `shutdown(wait=True)` 会等待所有未完成的评估任务完成，并批量返回剩余结果；
- 保证训练退出前，所有 `_eval_pending` 的模型都完成评估与重命名，避免留下“半成品 checkpoint”。

---

## 3. AsyncEvaluator：线程池 + 有序结果释放

文件：`rl_new/sac_cont_sy/async_evaluator.py`

### 3.1 设计目标

- 使用 `ThreadPoolExecutor` 避免 `daemon` 进程不能再开子进程的问题（评估内部会用 `ParallelEnv`）；
- 支持任意多的评估任务排队；
- 即使 **后提交的评估先完成**，也要保证对外返回的顺序与 step 一致；
- 对错误进行 fail-fast 记录，不阻塞后续训练与评估。

### 3.2 核心数据结构

```python
self.executor = ThreadPoolExecutor(max_workers=max_workers)
self.submitted_steps: List[int]        # 提交顺序 [100000, 200000, ...]
self.pending_results: Dict[int,Future] # 未完成 {step: future}
self.completed_cache: Dict[int,dict]   # 已完成但尚未返回 {step: result}
self.next_return_index: int            # 下一个应返回的索引（在 submitted_steps 中）
```

### 3.3 提交评估任务：`submit_eval`

```python
def submit_eval(self, eval_func, model_path, cfg, step, phase_name=None):
    position = ...  # 分配 tqdm 进度条位置
    future = self.executor.submit(eval_func, model_path, cfg, step, position, phase_name)
    self.submitted_steps.append(step)
    self.pending_results[step] = future
    return future
```

要点：

- `eval_func` 通常是 `evaluate_policy_standalone`；
- `cfg` 是 `copy.deepcopy(cfg)` 的快照，避免训练线程中的配置被后续修改影响评估；
- `position` 为 tqdm 进度条在 stderr 中的行号，多评估任务并发时不会互相覆写；
- 不做排队限制：`ThreadPoolExecutor` 自带的无限队列保证每个提交都能执行，最多 `max_workers` 并行。

### 3.4 有序结果获取：`get_evaluate_results`

```python
def get_evaluate_results(self) -> List[Dict[str, Any]]:
    # 1) 把已完成的 future 移入 completed_cache
    for step, future in list(self.pending_results.items()):
        if future.done():
            try:
                result = future.result(timeout=0)
            except Exception as e:
                result = {'error': str(e), 'metrics': None, 'video_path': None, 'step': step}
            self.completed_cache[step] = result
            del self.pending_results[step]

    # 2) 按提交顺序释放连续完成的结果
    ordered_results = []
    while self.next_return_index < len(self.submitted_steps):
        next_step = self.submitted_steps[self.next_return_index]
        if next_step in self.completed_cache:
            result = self.completed_cache.pop(next_step)
            if 'error' not in result or result.get('metrics') is not None:
                ordered_results.append(result)
            self.next_return_index += 1
        else:
            break
    return ordered_results
```

- 通过 `submitted_steps` + `next_return_index` 保证：
  - step=200k 的结果绝不会在 step=100k 之前被返回；
  - 即使 200k 评估先完成，也会在缓存中等待 100k 完成。
- 错误（例如环境崩溃、模型不兼容）会被记录为包含 `'error'` 键的 result，随后在日志中打印并被跳过。

### 3.5 训练结束时的 shutdown

`shutdown(wait=True)` 的关键逻辑：

- 对所有 `pending_results` 调用 `future.result()`，将其全部放入 `completed_cache`；
- 关闭线程池；
- 按照 `submitted_steps` 剩余顺序返回所有成功的结果。

这一步配合主脚本中的 `log_evaluate_results`，保证不会泄漏任何一个 `_eval_pending` checkpoint。

---

## 4. 评估环境与 rollout：与训练环境完全对齐

评估逻辑主入口是 `sac_utils_optimized.evaluate_policy_standalone`：

```python
def evaluate_policy_standalone(model_path, cfg, step, position=1, phase_name=None):
    model_path, working_dir = Path(model_path), Path(model_path).parent.parent

    # 1) 加载 actor
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if cfg.model.architecture == "resnet":
        actor, _ = make_sac_resnet_dual_models(env=make_single_environment(cfg, device="cpu"), ...)
    else:
        actor, _ = make_sac_models(env=make_single_environment(cfg, device="cpu"), device="cpu")
    actor.load_state_dict(checkpoint['actor']); actor.eval()

    # 2) CSVLogger + rollout
    csv_logger = CSVLogger(...) if cfg.logger.eval_video else None
    metrics = evaluate_policy(actor, cfg, logger=csv_logger, step=step, position=position)

    # 3) 处理视频路径与清理
    ...
    return { 'step': step, 'phase_name': phase_name, 'metrics': metrics,
             'reward_mean': reward_mean, 'completion_rate': completion_rate,
             'video_path': str(video_path) or None }
```

### 4.1 环境构造：`make_single_environment` / `make_drop_pixels_eval_environment`

- 训练环境：由 `env_utils.make_train_environment(cfg, device=...)` 创建，带有采集专用 transforms、统计字段、HIF 通道等。
- 评估环境：由 `make_drop_pixels_eval_environment` 创建（内部基于 `make_single_environment`），特点：
  - 在 CPU 上跑评估，避免占用 GPU；
  - Transform 链：

    ```text
    KeepLastPixels → VideoRecorder → DropPixels → InitTracker/StepCounter/Steps95ToDone/RewardSum/DoubleToFloat
    ```

  - `Steps95ToDoneCounter` 会在 episode 中一旦 `completion_ratio ≥ 0.95` 开始计数，最终输出 `steps_95_to_done`；
  - `auto_register_info_dict(default_info_dict_reader(keys=['overlap_count']))` 会把 env info 中的 `overlap_count` 安全映射成张量字段，若不存在则不会报错。

- 这样，训练和评估在：
  - 状态空间（HIF 通道、trajectory weights 等）；
  - 信息字段（`completion_ratio` / `overlap_count` / `steps_95_to_done`）；
  都保持一致，评估指标可以直接用于课程逻辑，不再需要手搓 `env.map_weed`。

### 4.2 rollout 实现：`evaluate_policy`

```python
def evaluate_policy(actor_critic, cfg, logger, step, position=1):
    actor = actor_critic[0] if isinstance(actor_critic, (tuple, list)) else actor_critic
    _, eval_env = make_drop_pixels_eval_environment(cfg, logger, eval_device=torch.device('cpu'))

    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        pbar = tqdm(...)
        eval_rollout = eval_env.rollout(
            max_steps=cfg.logger.eval_max_steps + 2,
            policy=actor,
            auto_cast_to_device=True,
            break_when_all_done=True,
            callback=lambda env, td: (... 更新进度条 ...),
        )
        pbar.close()

    if cfg.logger.eval_video:
        eval_env.apply(partial(dump_video, step=step))

    episode_end = eval_rollout['next', 'done'] if eval_rollout['next', 'done'].any() else eval_rollout['next', 'truncated']
    episode_rewards = eval_rollout['next', 'episode_reward'][episode_end].cpu().numpy()
    episode_lengths = eval_rollout['next', 'step_count'][episode_end].cpu().numpy()
    completion_ratios = eval_rollout['next', 'completion_ratio'][episode_end].cpu().numpy()

    eval_metrics = {
        'eval/reward_mean': float(np.mean(episode_rewards)),
        'eval/reward_min': float(np.min(episode_rewards)),
        'eval/reward_max': float(np.max(episode_rewards)),
        'eval/episode_length': float(np.mean(episode_lengths)),
        'eval/completion_ratio': float(np.mean(completion_ratios)),
        'eval/completion_ratio_max': float(np.max(completion_ratios)),
    }

    # 可选指标: steps_95_to_done
    if 'steps_95_to_done' in eval_rollout['next'].keys():
        steps_95 = eval_rollout['next', 'steps_95_to_done'][episode_end].cpu().numpy()
        eval_metrics['eval/steps_95_to_done_mean'] = float(np.mean(steps_95))
        eval_metrics['eval/ratio_95_to_done_mean'] = float(np.mean(steps_95 / np.clip(episode_lengths, 1, None)))

    # 可选指标: overlap_count
    if 'overlap_count' in eval_rollout['next'].keys():
        overlap = eval_rollout['next', 'overlap_count'].unsqueeze(-1)[episode_end].cpu().numpy()
        eval_metrics['eval/overlap_count_mean'] = float(np.mean(overlap))

    eval_env.close()
    return eval_metrics
```

注意几个关键点：

- `episode_end` 统一处理 `done` / `truncated`，保证早截断的 episode 也参与统计；
- 所有指标都基于 **episode 结束时刻** 的 `next` 字段：
  - `episode_reward`：累积奖励；
  - `step_count`：episode 长度；
  - `completion_ratio`：覆盖完成率；
  - `steps_95_to_done`：到 95% 后到 done 之间的步数；
  - `overlap_count`：覆盖重叠总次数。

> 与 `rules/` 旧脚本中手工从 `env.map_weed` 计算覆盖率相比，这里完全依托统一的 `EnvironmentState` 与 TorchRL transforms，指标计算更可靠也更容易迁移。

---

## 5. 指标体系与课程学习的关系

### 5.1 主要评估指标含义

- `eval/reward_mean`：
  - 每 episode 累积奖励的平均值。
  - 直接反映当前阶段奖励设计下，策略整体表现好坏。

- `eval/completion_ratio` & `eval/completion_ratio_max`：
  - 取 episode 结束时 `completion_ratio` 的均值 / 最大值；
  - 对 v4/v5/v6 环境，通常代表 **字段覆盖率**；
  - 课程阶段 S1/S2/S3 的切换门槛主要基于这个指标。

- `eval/steps_95_to_done_mean`：
  - 一旦 episode 过程中首次出现 `completion_ratio ≥ 0.95`，开始计时；
  - 直到 episode 结束之间所用步数的平均值；
  - 反映「在基本完成覆盖以后，是否能快速收尾而不是无效绕圈」。

- `eval/ratio_95_to_done_mean`：
  - `steps_95_to_done / episode_length`，一个标准化的稳定性指标；
  - 在课程 S2→S3 切换判断中，要求这个比值 **稳定下降或保持较低**，说明策略不仅能完成覆盖，还能高效结束。

- `eval/overlap_count_mean`：
  - 重复覆盖次数的平均值；
  - 对 v4+ 覆盖任务来说，是衡量路径冗余的重要指标，尤其在有 HIF 引导时，可作为上游损失/奖励设计的验证信号。

### 5.2 怎么驱动课程切换

`train_utils_optimized.maybe_advance_by_eval`（伪代码）：

```python
def maybe_advance_by_eval(state, phase: Phase, eval_results, cfg, curriculum_enabled):
    if not curriculum_enabled: return state, False

    # 汇总最近一次或几次评估结果
    metrics = aggregate(eval_results)

    if phase.type == 'S1':
        if metrics['eval/completion_ratio'] >= cfg.curriculum.s1_min_completion:
            state.consec_completion += 1
        else:
            state.consec_completion = 0
        if state.consec_completion >= cfg.curriculum.s1_consecutive_k:
            return state, True   # 进入 S2

    elif phase.type == 'S2':
        if metrics['eval/completion_ratio'] >= cfg.curriculum.s2_min_completion:
            # 判断 ratio_95_to_done 是否稳定下降
            ...
            if 状态稳定: return state, True  # 进入 S3

    return state, False
```

- S1 → S2：
  - 只关注 `completion_ratio`；
  - 要求在多次评估中连续达标（例如 ≥0.85）。
- S2 → S3：
  - 既看 `completion_ratio` 是否足够高；
  - 又看 `ratio_95_to_done_mean` 是否趋于稳定/下降；
  - 避免 S2 阶段出现“覆盖够了但一直乱跑”的不稳定行为。

> 评估指标不仅用于“好坏对比”，更直接参与了训练 curriculum 的调度，是训练 loop 的一部分，而不是事后分析。

---

## 6. 与旧版 rules/ 评估逻辑的对比

| 维度            | rules/ 下脚本（jump_path/sac_cont_test）            | 新栈 sac_curriculum_optimized_acc_loss + sac_utils_optimized |
|----------------|------------------------------------------------------|-------------------------------------------------------------|
| 环境           | `Pasture-v2`，手工访问 `env.map_weed`               | `NewPasture-v*` + `EnvironmentState` + TorchRL transforms   |
| 覆盖度         | `(init_weed - map_weed.sum()) / init_weed` 手算     | 直接用 `completion_ratio` 字段                              |
| 评估触发       | 通常单次脚本运行结束                                | 训练过程中，按 `eval_interval` 周期异步评估                |
| 执行方式       | 同步，在主进程/线程内                               | 线程池 + ParallelEnv，完全异步                              |
| 指标种类       | 覆盖率 & 距离（cover_90/95/98）                     | 奖励 / 完成率 / steps_95 / ratio_95_to_done / overlap 等   |
| 课程学习       | 无，评估仅用于对比                                  | 直接驱动 Phase 切换（S1→S2→S3）                            |
| 视频记录       | 基于手写脚本或无                                    | CSVLogger + VideoRecorder / LocalVideoRecorder              |

从设计理念上看：

- 旧版更像“实验脚本 + 手工记账”；
- 新版把评估变成 **一等公民**：
  - 环境、Transforms、指标都是统一的；
  - 与训练 loop 深度耦合但实现上又通过 AsyncEvaluator 解耦执行时序；
  - 指标自然地成为课程调度与模型选择的依据。

---

## 7. 实战使用建议

1. 想看“当前模型到底表现如何”：
   - 在训练中开启 W&B（`cfg.logger.backend=wandb`）与 `cfg.logger.eval_video=true`；
   - 关注 `eval/reward_mean`, `eval/completion_ratio`, `eval/steps_95_to_done_mean`, `eval/overlap_count_mean`；
   - 配合 checkpoint 文件名中的 reward/completion/steps95/overlap，快速定位最佳模型。

2. 想调整课程难度：
   - 在 `config-sync-server*.yaml` 中修改 `curriculum.s1_min_completion / s2_min_completion / s2s3_threshold` 等；
   - 注意这些阈值都是基于本文描述的评估指标，调节时心中要有数：
     - completion 越高，越难达到；
     - ratio_95_to_done 越低，意味着路径越“干净”。

3. 想调试评估环境是否与训练一致：
   - 查看 `env_utils.py` 中 `make_train_environment` 与 `make_drop_pixels_eval_environment` 的 Transform 链；
   - 确认 `completion_ratio` / `overlap_count` / `steps_95_to_done` 在两端都存在且含义一致。

4. 想扩展新指标（例如路径平滑度、转向次数）：
   - 在环境 State / Updater 中增加对应统计字段；
   - 在评估 rollout 中从 `eval_rollout['next', 'your_metric'][episode_end]` 读出并汇总；
   - 在 `log_evaluate_results` 中附上到 checkpoint 文件名或 W&B 指标中。

---

这份文档聚焦 `sac_curriculum_optimized_acc_loss.py` 及其依赖的评估模块，描述了从训练循环、异步调度，到 TorchRL 环境与指标的完整评估链路。结合 `rl_new/sac_cont_sy/doc/data_flow_analysis.md` 与本报告，可以在最短时间内建立对新栈“训练–评估–课程学习”闭环的完整理解。
