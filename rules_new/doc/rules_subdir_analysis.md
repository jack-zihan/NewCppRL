# rules 子目录速查与功能分析（2025-11-23）

面向新同事的快速上手笔记，覆盖 `rules/` 下现存脚本、数据资产、运行路径与常见坑，便于在最短时间理解这块旧版评测/规划工具链。内容基于当前仓库快照（2025-11-23）。

## 目录速览
```
rules/
├── config.py                 # 全局常量（尺寸、视野、日志目录等）
├── env_make.py               # Pasture-v2 环境构造助手
├── jump_path.py              # 传统覆盖算法（JUMP/SNAKE/R_SNAKE/BCP/REACT）示例
├── sac_cont_test.py          # 连续动作 SAC 模型评估脚本
├── dqn_test.py               # 离散动作 DQN 模型评估脚本
├── script.py                 # 批量跑实验的“自修改”调度脚本
├── ckpt/                     # SAC 模型权重（*.pt）
├── logs/                     # 评测结果 CSV
└── 团队分析报告/              # 旧版中文分析文档（report1/2/3）
```

核心依赖：`gymnasium`, `torch`, `numpy`, `dubins`, `shapely`, `omegaconf`, `torchrl`，以及仓库内旧版环境 `envs.cpp_env_v2:CppEnv`（非 `envs_new`）。

## 核心脚本拆解
### 1) config.py
- 仅存放常量：画布宽高 `W/H=600`，小车宽度与视野尺寸，日志目录 `LOG_DIR='rules/logs'`，随机种子等。
- 供 `jump_path.py` 读取几何尺度；与新版 Hydra 配置无耦合。

### 2) env_make.py
- 单函数 `get_env()`：
  - 调用 `gym.make("Pasture-v2", action_type="continuous", state_size/downsize=(128,128), num_obstacles_range=(0,0), use_sgcnn/use_global_obs/use_apf/use_box_boundary/use_traj=True, noise_* = 0)`。
  - 默认 `render=True`，用 `HumanRendering` 包一层，reset 时传入 `options={'weed_dist': 'gaussian', 'map_id': 2, 'weed_num': 50}`。
  - 返回 `(env, obs)`。obs 可能是 dict（`observation`+`vector`）。
- 作用：提供固定配置的 Pasture-v2 环境供测试脚本直接使用；与 `rules/script.py` 联动（被文本替换）。

### 3) jump_path.py —— 传统覆盖路径示例
- 定位：以 Dubins 曲线+几何启发式生成覆盖轨迹，直接驱动环境，不依赖 RL 模型。
- 全局状态：`env` 与位姿、朝向 `rad`、覆盖度 `cover_*`、累计里程等均为全局变量；调用链依赖副作用。
- 主要流程：
  1. `env, _ = get_env()` 初始化环境与地图，计算车宽/视野、农场多边形 `farm_vertices`（来自 `env.min_area_rect`）。
  2. 依据最长边方向生成平行扫描线，或随机/跳跃策略，逐步调用 `env.step([...])` 执行 Dubins 轨迹。
  3. 每步更新：坐标→Dubins 角度→`env.step`→统计覆盖率 `(init_weed - env.map_weed.sum()) / init_weed`，记录 90/95/98% 里程。
  4. 终止/撞障后写入 CSV：`rules/logs/coverage_results_{task_type}_{difficulty}.csv`，字段包括覆盖阈值里程、覆盖率序列、累积路径等。
- 支持的策略（通过 `task_type` 常量选择，默认 `R_SNAKE`）：
  - **REACT**：随机目标点，Dubins 连接；遇到最近“weed”则追踪。
  - **BCP**：纯平行扫描（Back-and-forth）
  - **SNAKE**：按主方向蛇形前进，前方若发现 weed 则冲过去再回到扫描线。
  - **R_SNAKE**：在 SNAKE 基础上约束垂直偏移，减少回撤。
  - **JUMP**：检测到前方 weed 时跳出当前扫描线，采样后返回。
- 几何工具函数：
  - `find_longest_edge()` 取多边形最长边→决定主方向；`dubins_navigate()`/`navigate()` 分别生成 Dubins 轨迹与线性插值路点。
  - `find_nearest_point*` 系列按局部坐标/方向筛选最近目标。
  - `local_adjustment()` 在 Dubins 路径遇障时局部绕行。
- 保护逻辑：单轮运行超 300s 自动写日志并 `sys.exit()`；到达覆盖率阈值记录对应路径长度。
- 典型用法：直接运行 `python rules/jump_path.py`（会弹出渲染窗口；参数需改代码中的常量，如 `task_type`, `weed_dist`, `map_id`）。

### 4) sac_cont_test.py —— SAC 连续控制评估
- 用途：加载 SAC 模型（`.pt`，假定存储 `ModuleList`，索引 0 为 actor），对 Pasture-v2 进行指定回合数评测。
- 关键参数：
  - `render`（默认 True）、`act_randomly`（True 时 ExplorationType.RANDOM）。
  - 权重路径 `pt_path` 默认指向 `rules/ckpt/t[02600]_r[2731.41=2717.75~2750.74].pt`。
  - `noise_set`、`difficulty`、`map_id` 等写入日志。
- 主循环：
  1. `env, obs = get_env()` 取得环境与初始观测。
  2. 每步从 `actor(observation, vector)` 取 `logits[2][0]` 作为连续动作；`env.step(action)`。
  3. 统计覆盖率与 90/95/98% 里程，超时 300s 强制退出。
  4. 结束时写 CSV：`rules/logs/{rl_model}_{difficulty}.csv`（字段同上，含 `noise_set`）。

### 5) dqn_test.py —— DQN 离散控制评估
- 与 SAC 逻辑类似，但使用离散动作 argmax。
- 绝对路径默认值（LOG_DIR、pt_path）指向 `/Users/chuyuliu/...`，需要手动改为仓库内 ckpt/ 或自有路径才能运行。
- 同样记录覆盖率与路径长度到 CSV。

### 6) script.py —— 批量跑实验/自修改脚本
- 目标：批量覆盖 `seed × difficulty × map_id` 组合，调用 SAC 评估。
- 做法：
  - 通过文本重写方式修改 `env_make.py`（障碍范围、噪声、reset 选项）和 `sac_cont_test.py`（模型名、地图、噪声、动作读取行等）。
  - 然后 `subprocess.run(["python", sac_script])` 逐一执行。
  - 循环嵌套固定：seeds `[25,27,47,21,31]`; difficulties `easy/medium/hard`; map 列表随难度调整；障碍范围与 weed 数随难度变化。
- 注意：
  - 这是“自修改”脚本，会直接覆盖文件内容，建议配合 `git diff`/`git checkout` 管理变更。
  - 默认仅跑 SAC 连续模型（DQN 与其他 SAC 变体的写法被注释保留）。

## 数据与模型资产
- `ckpt/`：两份 SAC 权重（baseline 与奖励 2731.41）。加载顺序依赖 `torch.load` 的 `ModuleList` 布局。
- `logs/`：示例 CSV
  - `coverage_results_R_SNAKE_easy.csv`：传统算法运行结果
  - `t[02600]_r[...].pt_easy.csv`：SAC 评估结果
- `团队分析报告/`：三份旧版中文分析文档，可参考历史背景与差异，但当前代码已裁剪为 `jump_path.py` 单文件版本。

## 典型运行路径与示例
1) **传统覆盖算法**（默认 R_SNAKE）：
```bash
python rules/jump_path.py  # 若在无图形环境，需手动将 render 相关逻辑改为 False
```
常用开关：在文件顶部修改 `task_type`, `weed_dist`, `map_id`, `difficulty` 等常量。

2) **SAC 连续模型评估**：
```bash
python rules/sac_cont_test.py
```
前置：确认 `pt_path` 指向有效模型；如需无渲染，设 `render = False`；`act_randomly=False` 可切换为确定性策略。

3) **批量跑实验**：
```bash
python rules/script.py
```
注意它会改写 `env_make.py`/`sac_cont_test.py`；运行前后可用 `git diff` 观察变更，必要时还原。

## 依赖与外部接口
- 环境：`Pasture-v2` 来自旧目录 `envs/`，与新版 `envs_new`（NewPasture-*）互不兼容。
- 几何/路径：`dubins` 库用于最短曲线；`shapely` 和 `matplotlib.path.Path` 用于点-in-polygon 与多边形处理。
- 渲染：`gymnasium.wrappers.HumanRendering`，在无显示时需关闭。
- 设备：脚本默认 `cpu`，无 GPU 逻辑。

## 常见坑与维护建议
- **硬编码路径**：`dqn_test.py` 的 `LOG_DIR`、`pt_path` 指向 macOS 本地路径；执行前必须改为仓库内位置。
- **自修改副作用**：`script.py` 会覆盖文件，易污染工作区；使用前备份或在独立分支运行。
- **全局状态 + 早退**：`jump_path.py` 使用大量全局变量并在完成后 `sys.exit()`；想复用函数需重构为类/函数式接口。
- **渲染默认开启**：`env_make.py` 默认 `render=True`，在服务器上可能卡住；可切到 False 或移除 `HumanRendering` 包装。
- **依赖安装**：`dubins`、`shapely` 非常规依赖，确保在虚拟环境中安装；`envs.cpp_env_v2` 需要仓库根目录在 `PYTHONPATH`（脚本已用 `sys.path.insert` 处理）。
- **时间限制**：运行超 300 秒将直接退出并写 CSV；长图或慢模型时可适当增大上限。

## 与主线（envs_new + rl_new）关系
- `rules/` 属于旧版实验/基线脚本：绑定 Pasture-v2、手写几何路径或早期 SAC/DQN 模型。
- 新主线训练/评测请使用 `envs_new/` + `rl_new/sac_cont_sy/`；两者的观测/信息键、覆盖指标不同（旧版以 weed 覆盖为主）。
- 若需迁移：
  1) 用 `envs_new.__init__.py` 注册的 `NewPasture-v*` 替换 `Pasture-v2`；
  2) 调整 obs 解析（`observation_spec` 变更，可能无 `vector` 字段）；
  3) 更新覆盖指标：`completion_ratio`/`overlap_count` 取代 `map_weed`。

## 速查关系图（ASCII）
```
script.py
  └─(rewrite params)→ env_make.py → gym.make('Pasture-v2')
                       │
            ┌──────────┴──────────┐
            │                     │
    sac_cont_test.py       dqn_test.py
            │                     │
         CSV logs            CSV logs
               \
               └→ jump_path.py (无需模型，直接几何规划)
```

