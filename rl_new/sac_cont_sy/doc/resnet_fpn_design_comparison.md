# ResNet‑FPN 编码器方案对比与推荐（覆盖任务 v4/v5/v6）

本文对两套候选方案进行面向决策的对比，并给出在本仓库（TorchRL + 新环境栈）中最稳健且性价比最高的落地建议。

## 背景与接口约束

- 训练脚本：`rl_new/sac_cont_sy/sac_curriculum.py` 通过 `make_sac_models(env, device)` 构建 Actor/ Critic；接口要求：
  - Actor：`TensorDictModule([...], in_keys=["observation","vector"], out_keys=["loc","scale"]) → ProbabilisticActor(TanhNormal)`。
  - Critic：`ValueOperator(in_keys=["observation","vector","action"])`，输出标量 Q。
- 观测：多尺度 BEV 语义图，默认 multiscale 输出 `(C_total, 96, 96)`；`vector` 在 v4=14，v6=98（历史更长）。
- 现网：IMPALA‑CNN 风格，早降采样、快速扩感受野；在几何细节/边界保真方面存在先天短板。

## 两个方案的核心差异

下文以“我的方案（A）”与“同事方案（B）”对照。

### 1) 主干与归一化

- A：ResNet18/34 + GroupNorm（或 LayerNorm），SiLU；去除 maxpool，首层 stride=1，把 96×96 保住。
- B：ResNet34 + BatchNorm，去 maxpool，首层 stride=1。

对比：
- RL 回放批次虽大（batch_size=2048），但数据分布时变且异质；BN 运行统计在探索期与收敛期会漂移，评估时（batch 小）不稳定的概率更高。GN 不依赖 batch 统计，鲁棒性更强。若坚持 BN，建议至少准备 fallback：周期性重置 BN running stats 或切到 GN。

### 2) FPN 宽度

- A：统一 128（或 160/192），重视效率与稳定性。
- B：统一 256，追求极限表达能力。

对比：
- 128→256 的收益主要体现在高层抽象与细粒度语义，但计算/显存翻倍。考虑我们已经由环境提供“多尺度裁剪 + 结构稀疏”的强先验，128/160 常已足够；若预算允许，256 也可接受。

### 3) 多尺度聚合策略（关键差异）

- A：每个 P2..P5 做“注意力加权全局池化”（Σ(A·F)/ΣA），得到每层 128/256 维，全拼成 512/1024 维，再 `Linear→512`。优点：
  - 显著放大“窄缝边界、未覆盖薄片、贴边信息”等稀疏细节（由注意力学习）。
  - 特征维度小，训练/显存极友好（见“资源估算”）。
- B：每层先 `AdaptiveAvgPool(8×8)`，再把 P2..P5 flatten 并拼接（4×256×8×8=65536 维），随后 `FC(65536→512)`。

对比与资源估算：
- B 的单层 FC 权重量 ≈ 65,536×512 = 33.6M 参数（≈128MB FP32 权重，不含动量）。AdamW 需 2 组一阶/二阶动量，单网络该层优化器状态额外 ≈256MB；Actor+Critic 两套复制，光这一层权重+动量就接近 ≈640MB。再叠加前向激活：batch=2048 时单次前向的该层激活 ≈ 65,536×2,048×4B ≈ 512MB，极易 OOM/降速。
- A 的聚合后维度 ≤ 1024；`FC(1024→512)` 权重 ≈ 0.5M（≈2MB），优化器状态量级小，几乎不占显存；前向激活也很小，整体吞吐更高。

结论：在当前 batch 与 SAC 训练形态下，A 的聚合设计显著更稳健、更高效；B 需把 pool_size 至少降到 4（特征维度 16,384）才勉强可控，但仍然较重。

### 4) 预训练与解冻策略

- A：可以从头训练；若要用预训练，建议仅迁移卷积权重并使用 GN（BN→GN 无法一键迁移统计量），并搭配“差异化 LR + 渐进解冻”。
- B：ImageNet 预训练 + 端到端统一 LR 训练；提供 BN 重置与若不稳定则渐进解冻的 fallback。

对比：
- 预训练对 BEV 几何/语义的域偏移较大，实际收益依赖于卷积低层边缘/角点过滤器的可迁移性；保守做法是“先小心微调（小 LR/冻结）再放开”。

## 任务贴合度（覆盖几何 + 多尺度）

共同点：两案都保留首层高分辨率并引入 FPN，均能同时感知局部（P2）与全局（P5）。

差异关键在“如何聚合多尺度信息”：
- A 的注意力加权池化对“局部细节很重要但占比很小”的 CPP 任务更友好（贴边、窄通道、重复覆盖热区等）。
- B 的 8×8 空间保留能携带一定布局信息，但在没有 Q‑map 头的前提下，最终仍被压成向量；其好处在于 MLP 可以更显式地区分多栅格，但代价是巨量参数与显存压力。

## 推荐配置（首选）

> 目标：在你现有训练栈零改动/极少改动前提下，最大化稳态、效率与几何表现。

- Backbone：ResNet34（性能优先；若资源紧则 ResNet18）。
- Norm：GroupNorm（32 组或按通道数自适应），激活 SiLU。
- FPN：统一 256 通道（预算许可时选 256；轻量可选 128/160）。
- 聚合：注意力加权全局池化（每层 256 维）→ 拼接为 1024 维 → `Linear(1024→512)`。
- 融合：`concat(g, vector[, action]) → MLP(512→512)`；Actor 接 `NormalParamExtractor` 输出 `loc/scale`，Critic 接 `Linear→1`。
- 训练：
  - 先与当前优化器/超参保持一致（方便替换 IMPALA 做 A/B 测）；
  - GradClip=1.0；WD 对卷积/线性生效（不对 norm/bias）。
- 预训练（可选）：若启用，建议“差异化 LR + 渐进解冻”，否则从头训练更稳。

为何首选该配置：
- 与现有装配完全同口（`make_sac_resnet_models` 可一键替换），对 v4/v5/v6 的 `in_ch/vec_dim` 自适配；
- 显著降低显存与计算峰值，适配 batch=2048 的回放训练；
- 在 v6 中能有效利用 `trajectory_weights` 的低频分布与 v5 的 HIF 通道，同时放大 P2 的边界细节。

## 可选折中（若需空间保留）

如果确实希望保留一定二维布局，可把聚合改成：
- 每层 `AdaptiveAvgPool(4×4)` → flatten → 拼接（维度 4×256×16=16,384），再 `Linear(16,384→512)`。
- 此时单层 FC ≈ 8.4M 参数（≈32MB 权重；AdamW 额外 ≈64MB），仍可控，但比注意力池化重很多。

## 与同事方案（B）的优劣权衡小结

- 性能上限：B（ResNet34 + FPN256）具备更强表征上限；A 也可采用相同宽度，因此“上限”更多由聚合与归一化决定。
- 训练稳定性：A（GN + 轻聚合）显著更稳，几乎无 BN 与超大 FC 的不确定性；B 需多个 fallback 才能稳住。
- 资源效率：A 远优，能与当前 batch=2048、UTD=1 配置和平共处；B 的 8×8 聚合在现配下大概率超内存或降速明显。
- 工程改动：两案对外接口一致；若要做差异化 LR/渐进解冻，训练脚本需轻微改造 param groups（后续可加开关）。

## 后续实现建议（不在本提交中改代码）

1) 新增 `rl_new/sac_cont_sy/resnet_fpn.py`：
- `ResNetFPNEncoder(in_ch, out_dim=512, width=256, norm='gn', act='silu')`：返回 `embed`。
- `FPNQNet`/`FPNPolicyNet`：封装 `Encoder + head`，保持 `DeepQNet` 的 forward 签名（`observation, vector[, action]`）。

2) `model_utils.py`：
- 新增 `make_sac_resnet_models(env, device=...)`，完全复用 `make_sac_models` 的打包逻辑（ProbabilisticActor/ValueOperator、in_keys/out_keys、`env.fake_tensordict()` 懒初始化）。

3) 训练与配置（可选增强）：
- 为 Actor/ Critic 提供可选 param groups（backbone 小 LR，FPN/头正常 LR）。
- 增加 `model.type={impala,resnet_fpn}` 开关，便于 A/B 对比。

---

### TL;DR（结论）

在当前任务与训练配置下，推荐采用：ResNet34 + FPN(256) + 注意力加权全局池化 + GroupNorm + SiLU，维持 512‑D 视觉嵌入与现有头部/接口不变。该方案在几何细节、稳定性与资源效率之间取得最优平衡，落地成本低、成功率高。若必须保留二维栅格信息，建议使用 4×4 池化的折中聚合，而非 8×8，以避免显存和吞吐瓶颈。

