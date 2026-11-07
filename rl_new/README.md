# rl_new

当前包含 `dqn/`、`sac/`（离散 SAC）和 `sac_cont_sy/`（连续 SAC）。目录后缀不代表已完成独立迁移：DQN、离散 SAC 仍引用 `rl/` 与 `torchrl_utils/`。

连续 SAC 入口是 `sac_cont_sy/sac.py`、`sac_cont_sy/sac_curriculum_optimized.py` 和 `sac_cont_sy/sac_curriculum_optimized_acc_loss.py`，三者的配置和行为不同。

- 环境准备、入口选择与命令：[根 README](../README.md)。
- 调度、模型、回放、评估和已核验限制：[当前项目说明](../docs/project-current-state.md)。
- 修改规范：[AGENTS.md](../AGENTS.md)。

原本页的 TorchRL 0.9.2 迁移说明、`rl_new/sac_cont/` 路径及“全部兼容”结论已过期，不再作为当前使用说明。依赖升级记录不继续累积在本页。
