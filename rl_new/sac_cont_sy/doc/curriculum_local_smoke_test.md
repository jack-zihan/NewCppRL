# Curriculum SAC 本地烟雾测试记录（2025-10-06）

## 目标
- 验证课程学习代码在轻量参数下的执行流程：
  - 课程阶段初始化与切换；
  - 分桶回放的采样与重置；
  - 训练主循环与日志逻辑。
- 环境约束：仅允许在 `/dev/shm` 权限受限的沙箱内运行一次小规模训练，不修改 `rl_new/sac_cont_sy` 正式代码。

## 测试准备
1. 克隆正式代码到 `rl_new/sac_cont_sy_test`。
2. 调整 `config-sync-server.yaml` 为极小工作量配置：
   - `frames_per_batch=16`、`total_frames=240`；
   - `batch_size=32`、`buffer_size=512`；
   - 评估参数：`eval_interval=5`、`eval_episodes=1`、`eval_max_steps=20`；
   - 课程门槛：`s1_min_completion=0.0`、`s2_min_completion=0.0`、`s1_consecutive_k=1`、`s2_consecutive_k=1`。
3. 仅在 `sac_cont_sy_test` 中做兼容性修改：
   - 课程初始化使用 `cfg.env.env_kwargs = dict(...)`，避免 OmegaConf `struct` 限制；
   - `bucketed_replay` 改为纯 Python list 实现，去除共享内存依赖；
   - `create_replay_buffer` 改用轻量存储；
   - 训练脚本设置 `torch.multiprocessing.set_sharing_strategy("file_system")`；
   - 采集器改为 `SyncDataCollector` / 自定义轻量实现。

## 测试执行
- 命令：`python -m rl_new.sac_cont_sy_test.sac_curriculum`
- 环境：沙箱无 `/dev/shm` 写权限。

## 结果与问题
1. **共享内存限制**
   - TorchRL 的回放存储、优先级写指针、收集器（包括 `SyncDataCollector`）都会尝试在 `/dev/shm` 创建共享内存或信号量，触发 `PermissionError: [Errno 13]`。
   - 即使改用 `torch.multiprocessing.set_sharing_strategy("file_system")`，仍有若干路径依赖 `torch_shm_manager` 或 POSIX semaphore，导致进程初始化失败。
2. **替代方案尝试**
   - 自定义分桶回放（纯 Python list）成功规避了存储问题，但采集器与 Trajectory pool 仍要求共享内存。
   - 将采集器切换为 `SyncDataCollector` 仍失败，因为其内部也会构造共享 `Tensor`（`share_memory_`）。
   - 若继续深度改写采集与训练逻辑，可以模拟生成数据，但这样无法真实验证训练循环中的更新逻辑，失去测试意义。

## 结论
- 在当前受限沙箱（禁用 `/dev/shm` 与 POSIX semaphore）的条件下，TorchRL 官方采集器与回放缓冲均无法完成初始化，课程训练流程无法被真实运行。
- 所有修改均限定在 `rl_new/sac_cont_sy_test`，主目录 `rl_new/sac_cont_sy` 未被改动。

## 建议的本地验证步骤
如需在具有共享内存权限的本地 4090 笔电上完成真实烟雾测试，可按照以下步骤：
1. `cp -r rl_new/sac_cont_sy rl_new/sac_cont_sy_test_local`（确保正式代码独立）。
2. 按本文的轻量参数修改配置（或直接参考 `sac_cont_sy_test/config-sync-server.yaml`）。
3. 若无需完全禁用共享内存，可恢复官方回放实现，并运行：
   ```bash
   python -m rl_new.sac_cont_sy_test_local.sac_curriculum hydra.run.dir=outputs/smoke
   ```
4. 关注日志：
   - 阶段切换：`[Curriculum] ✨ 切换到阶段 ...`
   - 分桶重置：`分桶采样比例更新为 ...`
   - 课程指标：`curriculum/stage_idx`、`buffer/*_size` 等。
5. 若需可重复的离线测试，可在本地写入 `/dev/shm` 权限并确保 `torch_shm_manager` 可执行。

## 后续
- 本沙箱无法完成实际运行，但所有准备工作与兼容性修复已经在 `sac_cont_sy_test` 中给出，可直接在具备共享内存的环境里复用。
- 如需继续在沙箱内做功能验证，可考虑编写纯 Python 的训练数据模拟器，跳过 TorchRL 采集器；但这对“真实流程”验证价值有限。
