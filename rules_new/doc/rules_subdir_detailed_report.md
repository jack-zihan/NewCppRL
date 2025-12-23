# rules 子目录深度分析报告（2025-11-23）

> 目标：让新同事在 **30 分钟内建立完整心智模型**——知道 `rules/` 是干什么的、各文件怎么串起来、怎样最快跑通和扩展，而不是一行行啃代码。

---

## 1. 模块总体定位：老版评测 & 基线工具箱

- `rules/` 是基于 **旧版 Pasture-v2 环境** 的一组脚本，用来：
  - 运行几何/启发式覆盖算法（JUMP/SNAKE/R_SNAKE/BCP/REACT）。
  - 评估早期 DQN / SAC 模型在 Pasture-v2 上的覆盖性能。
  - 批量扫参数（seed / 难度 / 地图 / 噪声）并落盘 CSV 结果。
- 它与当前主线栈：
  - 环境：`envs_new/*` + Gym ID `NewPasture-v*`
  - 训练：`rl_new/sac_cont_sy/*`
  完全独立，仅在 **历史对比、基线复现** 场景下仍然有价值。

可以把 `rules/` 理解为一个 **“旧版评测实验室”**：环境固定为 `Pasture-v2`，脚本直接操作 env，结果写入本地 CSV。

---

## 2. 目录与角色总览

### 2.1 目录树

```text
rules/
├── config.py                 # 几何/日志基础配置
├── env_make.py               # Pasture-v2 环境构造助手
├── jump_path.py              # 几何/启发式覆盖算法
├── sac_cont_test.py          # SAC 连续控制评估脚本
├── dqn_test.py               # DQN 离散控制评估脚本
├── script.py                 # 批量实验调度，自修改 env_make/sac_cont_test
├── ckpt/                     # SAC 模型权重（*.pt）
├── logs/                     # 覆盖评测 CSV
└── 团队分析报告/              # 旧版中文分析文档（历史背景）
```

### 2.2 角色速查表

| 文件 / 子目录             | 角色定位                                      | 典型入口命令                         |
|--------------------------|----------------------------------------------|--------------------------------------|
| `config.py`             | 车体 & 地图基础尺寸、日志目录等常量          | 被 `jump_path.py` 导入               |
| `env_make.py`           | 创建 `Pasture-v2` Gym 环境                   | `from rules.env_make import get_env` |
| `jump_path.py`          | 几何基线路径规划（JUMP/SNAKE/R_SNAKE/BCP/REACT） | `python rules/jump_path.py`          |
| `sac_cont_test.py`      | SAC 连续控制评估 → CSV                       | `python rules/sac_cont_test.py`      |
| `dqn_test.py`           | DQN 离散控制评估 → CSV                       | 需先改硬编码路径后运行               |
| `script.py`             | for-loop + 文本重写，批量跑 SAC 评估         | `python rules/script.py`             |
| `ckpt/`                 | SAC 权重（.pt）                               | `sac_cont_test.py` / `script.py`     |
| `logs/`                 | 覆盖结果 CSV                                 | 由脚本自动写入                       |
| `团队分析报告/`          | 老版本详细分析报告，含历史版本差异           | 只读参考                             |

---

## 3. 三条核心使用路径（架构视角）

### 3.1 几何基线算法路径（无 RL 模型）

```text
config.py  ──▶ jump_path.py ──▶ env_make.get_env() ──▶ gym.make('Pasture-v2')
                                              │
                                              └─▶ 直接 env.step([...]) + 覆盖统计 + CSV
```

- 典型场景：
  - 想要一个 **“无学习、纯几何”** 的覆盖基线（R_SNAKE / JUMP 等）。
  - 与 RL 模型对比路径长度、覆盖效率。

### 3.2 RL 模型评估路径（SAC / DQN）

```text
env_make.get_env() ──▶ Pasture-v2
        │
        ├─▶ sac_cont_test.py  ──▶ torch.load(SAC ckpt) ──▶ 连续动作 env.step(action)
        │
        └─▶ dqn_test.py       ──▶ torch.load(DQN ckpt) ──▶ 离散动作 env.step(action)
                             （都统计 weed 覆盖率 & 覆盖里程，落盘 CSV）
```

- 典型场景：
  - 验证「旧版 SAC 连续控制」在原始 Pasture-v2 上的表现。
  - 对比 DQN / SAC / 几何基线的覆盖效率。

### 3.3 批量实验路径（脚本自动写参数）

```text
script.py
  └─ for seed × difficulty × map_id × weed_dist × noise_set:
         ├─ rewrite env_make.py    （障碍范围、seed、weed_dist、map_id、weed_num）
         ├─ rewrite sac_cont_test.py（模型 ckpt、难度标签、seed、map_id、noise_set）
         └─ subprocess.run('python rules/sac_cont_test.py')
                 └─ logs/*.csv 累积输出
```

- 典型场景：
  - 想一次性跑完多地图、多难度、多随机种子，生成一批 CSV 做统计。
  - 接受“自修改脚本”的方式（需用 git 管理好变更）。

---

## 4. 单文件深度拆解

### 4.1 `config.py`：基础几何 & 日志配置

功能非常简单：

```python
class Config:
    W = 600
    H = 600
    CAR_WIDTH = 5
    SIGHT_WIDTH = 24
    SIGHT_LENGTH = 24
    RETURN_MAP = True
    NUM_OBSTACLE_MIN = 0
    NUM_OBSTACLE_MAX = 0
    LOG_DIR = 'rules/logs'
    SEED = 0
    DEBUG_MODE = True
```

- 只负责提供常量：
  - 画布尺寸：`W/H` 决定离散网格大小，对坐标变换、路径长度估计都有影响。
  - 小车/视野参数：`CAR_WIDTH/SIGHT_WIDTH/SIGHT_LENGTH` 被 `jump_path.py` 用来：
    - 估算蛇形扫描步长。
    - 估算覆盖带宽，以避免“漏扫”。
  - 障碍数量范围：这里只是默认值，真正的障碍范围在 `env_make.py` + `script.py` 中由 Gym 环境控制。

### 4.2 `env_make.py`：统一环境工厂

核心函数：

```python
def get_env():
    render = True
    env = gym.make(
        id="Pasture-v2",
        render_mode='rgb_array' if render else None,
        action_type="continuous",
        state_size=(128, 128),
        state_downsize=(128, 128),
        num_obstacles_range=(0, 0),
        use_sgcnn=True,
        use_global_obs=True,
        use_apf=True,
        use_box_boundary=True,
        use_traj=True,
        noise_position=0,
        noise_direction=0,
        noise_weed=0,
    )
    if render:
        env = HumanRendering(env)
        env.render()
    obs, info = env.reset(seed=25, options={'weed_dist': 'gaussian', 'map_id': 2, 'weed_num': 50})
    return env, obs
```

要点：

- 统一了所有脚本的 **环境创建入口**：不管是几何基线还是 RL 评估，都用 `get_env()`。
- 关键配置：
  - `Pasture-v2` + `action_type="continuous"`：旧版 C++ APF 环境，支持连续 `[distance, delta_angle]` 或 `[v, w]` 动作。
  - `num_obstacles_range=(0, 0)`：无障碍；在批量脚本中会被改写成 `[2,4]`/`[5,8]` 等。
  - `use_sgcnn / use_apf / use_traj` 等：开启旧版的图卷积、势场和轨迹通道。
- seed 与地图：
  - `seed=25`、`weed_dist='gaussian'`、`map_id=2`、`weed_num=50` 是默认“简单场景”。
  - `script.py` 会动态把这一行重写成不同组合。
- 渲染：
  - 默认启用 `HumanRendering`，在无图形环境（服务器）上运行时常见问题是卡在渲染，可以直接把 `render = False` 或去掉包装。

### 4.3 `jump_path.py`：几何/启发式覆盖算法

这是 `rules/` 中 **最“重”的文件**，承担：「不给任何 RL 模型，单靠几何和策略完成覆盖」。

#### 4.3.1 全局状态与环境信息

文件开头：

- 路径 & 依赖导入：
  - 把项目根目录加到 `sys.path`，以便 `rules.config`、`rules.env_make`、`envs.cpp_env_v2` 等能被导入。
- 创建环境并抽取关键属性：

```python
env, _ = get_env()

task_type = "R_SNAKE"  # 算法模式

agent_width = Config.CAR_WIDTH
sight_width = Config.SIGHT_WIDTH
sight_length = Config.SIGHT_LENGTH
agent_position = [env.agent.y, env.agent.x]  # 注意 [y, x]
W = Config.W
H = Config.H
w_max_rad = abs(env.w_range.max) * (math.pi / 180)
turning_radius = env.v_range.max / w_max_rad
farm_vertices = env.min_area_rect[0][:, 0, ::-1]  # farm 边界多边形
init_weed = env.map_weed.sum()
```

- 这里“读出”的环境信息是所有算法的基础：
  - `env.agent`：当前小车坐标（注意顺序是 `[y, x]`，与常规 `[x, y]` 相反）。
  - `env.min_area_rect`：农场最小外接矩形，用来估算主方向和最大对角线长度。
  - `env.map_weed`：布尔/计数矩阵，表示草/杂草分布；后续用于计算覆盖率。
  - `env.map_obstacle`：障碍物分布，用于 Dubins 路径的碰撞检查与局部绕行。
- 一些重要的「测度变量」：

```python
cover_90, cover_95, cover_98, cover, dist_list = -1, -1, -1, [], []
...
overall_length = 0  # 累计路径长度（在农田坐标系下）
```

这些在多文件里都是 **覆盖性能的评价指标**：

- `cover_rate = (init_weed - env.map_weed.sum()) / init_weed`
- 当覆盖率首次达到 90/95/98% 时，记录当前 `overall_length`，衡量“以里程为代价的覆盖效率”。

#### 4.3.2 核心驱动函数：`go(p2)`

```python
def go(p2):
    prev_position = agent_position
    radian = math.atan2(p2[1] - agent_position[1], p2[0] - agent_position[0])
    length = math.sqrt((p2[0] - agent_position[0]) ** 2 + (p2[1] - agent_position[1]) ** 2)
    delta_angle = - (radian - rad) % (2 * math.pi)
    delta_angle = delta_angle - 2 * math.pi if delta_angle > math.pi else delta_angle
    delta_angle = math.degrees(delta_angle)

    env.set_action_type("continuous")
    obs, reward, done, time_out, _ = env.step([length, delta_angle])

    agent_position = [env.agent.y, env.agent.x]
    distance = np.linalg.norm(np.array(agent_position) - np.array(prev_position))
    overall_length += distance
    rad = np.pi / 2 - math.radians(env.agent.direction)

    discovered = np.argwhere(np.logical_and(env.map_weed, np.logical_not(env.map_frontier)) == 1)
    discovered = [p for p in discovered if is_point_in_polygon(p, farm_vertices)]

    cover_rate = (init_weed - env.map_weed.sum()) / init_weed
    ...  # 更新 cover_90/95/98, cover, dist_list

    if done:
        if env.check_collision():
            save_data_to_csv(..., collapse=1, ...)
        else:
            save_data_to_csv(..., collapse=0, ...)
        exit()
```

可以把 `go(p2)` 看成一个 **“原子动作”**：给定下一个目标点 `p2`（农田坐标系），

1. 计算当前朝向到 `p2` 的角度差，转换成环境期望的动作 `[length, delta_angle]`。
2. 调用 `env.step(...)` 更新环境。
3. 基于新 `env.map_weed` 更新覆盖率 & 统计路径长度。
4. 若 episode 结束（完成或碰撞），写 CSV 并退出。

所有高层算法（JUMP / SNAKE / R_SNAKE / BCP / REACT）最终都降解为对 `go(p2)` 的一系列调用。

#### 4.3.3 几何辅助函数族

关键几类：

- 多边形 & 坐标系：
  - `is_point_in_polygon(point, vertices)`：基于 `matplotlib.path.Path` 判断点是否在农田内部。
  - `find_longest_edge(farm_vertices)`：
    - 扫描多边形边，找出最长那条边；
    - 用来确定农田“主方向”，即蛇形/平行扫描的基准方向。
  - `find_offset(start, end, point, real_radians=None)`：
    - 计算点到给定“扫线”（或按某个角度构造的直线）的有符号垂直距离；
    - 用于决定下一条平行扫描线应该相对当前线偏移多少。

- 局部目标筛选：
  - `get_forward_jump(...)` / `get_forward_snake(...)` / `get_forward_rsnake(...)`：
    - 在当前朝向坐标系中筛选“前方的”且满足一定垂直限制的候选 weed 点。
  - `find_nearest_point(p, coordinates, r)`：找到距离大于安全距离 `2r` 的最近点，用于避免过紧的转向。
  - `find_nearest_point_jump(...)`：在旋转坐标系下找到投影最近的点，适合“沿行进方向纵向跳跃”。

- Dubins 路径 & 避障：

```python
def dubins_navigate(p2, r):
    path = dubins.shortest_path((agent_position[0], agent_position[1], rad), (p2[0], p2[1], p2[2]), r)
    configurations, _ = path.sample_many(0.5)
    for point in configurations[1:]:
        navigate(list(point[:2]))
```

- `dubins_navigate_obstacle(...)` + `local_adjustment(...)` 进一步在采样点上检查 `env.map_obstacle` 是否阻挡，必要时对路径点做随机扰动绕障。

整体来看，这些辅助函数共同提供了一套 **“几何坐标系 + Dubins 曲线 + 点选策略”** 的基础设施，高层的 JUMP/SNAKE/R_SNAKE 等只是“如何挑选下一个目标点”的不同策略实现。

#### 4.3.4 算法主循环：按 task_type 分支

文件末尾根据 `task_type` 分为两大分支：`REACT` 与「基于平行扫线的族（JUMP/SNAKE/R_SNAKE/BCP）」。

- `REACT` 模式（更偏“追踪型”）：
  - 随机采样目标点 `rand_goal`，插值形成直线路径；
  - 途中每个点检查附近是否有 weed，如果发现则优先 `dubins_navigate` 跳到 weed；
  - 重复若干次，直到时间上限或覆盖终止。

- 其他模式共用一套“平行扫描线”主结构：

  1. 通过 `find_longest_edge` 确定农田主方向 `real_radians`，构造一条长扫线 `[start, end]` 及其法向。
  2. 从 `y_offset = -diag_length + agent_width / 2` 开始，沿法向逐步平移扫线：

     ```python
     new_start = start + y_offset * normal - diag_length * main_dir
     new_end   = end   + y_offset * normal + diag_length * main_dir
     line = LineString([new_start, new_end])
     # 等步长采样所有点，筛选在农田内的 valid_points
     ```

  3. `turn` 标志决定当前扫线的行进方向（“来回蛇形”），并据此修正期望朝向 `rad_n`。
  4. 根据 `task_type` 决定如何在当前 `valid_points` 上行走：
     - **BCP**：全部 `navigate(valid_points[p_i])`，不看 weed，只做规则扫线。
     - **SNAKE**：
       - 在 `valid_points` 之间走到一半，检查前方是否有 weed（`get_forward_snake`）。
       - 若有，则 `dubins_navigate` 冲过去，再生成一条新的“等距平移扫线”继续扫。
     - **R_SNAKE**：类似 SNAKE，但约束 weed 在“向上的扇形区域”，避免大范围横跳。
     - **JUMP**：
       - 对前方 weed 使用 `find_nearest_point_jump` 进行 **投影匹配**；
       - 确保在扫线上的“起跳点”和“落地点”之间留够空间（与 `turning_radius` 成比例）；
       - 通过 Dubins 来回接入，形成“行进—跳跃—回归”结构。

  5. 每完成一条扫线后，根据扫描结果和 `find_offset` 推进下一条扫线的 `y_offset`：

     - 如果在本条扫线中发现 new weed，则基于其垂直偏移量 `find_offset(...)` 决定下一条扫线更贴近 weed 区域；
     - 否则按 `sight_width`/`agent_width` 的固定步长前进，保证不留死角。

  6. 整个 while 循环有时间上限和空扫线计数双重保护，避免无意义循环：

     - 运行时间 > 300s：直接写 CSV 并退出。
     - 连续多条平移后没有任何有效 `valid_points`：认为区域已覆盖完，大致结束。

最终无论哪种模式，结束时都会调用 `save_data_to_csv(...)` 将 `weed_dist / random_seed / map_id / collapse / cover_90/95/98 / cover 序列 / dist_list` 写入 `rules/logs/coverage_results_{task_type}_{difficulty}.csv`。

### 4.4 `sac_cont_test.py`：SAC 连续控制评估

核心职责：给出一个 **已训练好的 SAC 连续控制模型 + Pasture-v2 环境**，统计同样的覆盖指标。

结构概览：

1. 路径与工具：
   - 通过 `BASE_DIR` 把仓库根目录加入 `sys.path`。
   - `to_absolute_path` 工具保证 ckpt 路径、日志路径既可以写相对路径也可以写绝对路径。
2. 关键参数：

```python
episodes = 10
render = True
act_randomly = True  # True: ExplorationType.RANDOM, False: DETERMINISTIC

LOG_DIR = BASE_DIR / 'rules' / 'logs'
rl_model = "t[02600]_r[2731.41=2717.75~2750.74].pt"
pt_path = 'rules/ckpt/t[02600]_r[2731.41=2717.75~2750.74].pt'

noise_set = [0, 0, 0]
weed_dist = "gaussian"
random_seed = 25
map_id = 2
```

3. 模型加载：

```python
actor_critic = torch.load(to_absolute_path(pt_path), map_location='cpu').to(device)
actor = actor_critic[0].to(device)
```

- 假定 ckpt 是一个 `ModuleList` 或类似结构，`[0]` 是 actor；调用时传入 `observation` 和 `vector` 两个参数。

4. 主评估循环：

```python
env, obs = get_env()
init_weed = env.map_weed.sum()
...
with set_exploration_type(exploration_type), torch.no_grad():
    for i in range(episodes):
        done = False
        ret = 0.0
        t = 0
        start_time = time.time()
        while not done:
            if time.time() - start_time > 300: ... # 超时保护

            if isinstance(obs, dict):
                observation = obs['observation']
                vector = obs['vector']
            observation = torch.from_numpy(observation).float().unsqueeze(0)
            vector = torch.tensor(vector).float().unsqueeze(0)

            logits = actor(observation=observation, vector=vector)
            action = logits[2][0].tolist()  # 连续动作

            past_position = env.agent.position
            obs, reward, done, _, info = env.step(action)
            now_position = env.agent.position
            overall_length += np.linalg.norm(np.array(now_position) - np.array(past_position))

            cover_rate = (init_weed - env.map_weed.sum()) / init_weed
            ...  # 更新 cover_90/95/98, cover, dist_list

            if done:
                if env.check_collision():
                    collapse = 1
                else:
                    collapse = 0
                save_data_to_csv(..., collapse, ...)
                exit()

            t += 1
            ret += reward
            if render:
                env.render()
```

- 可以看到，**评价指标与 `jump_path.py` 完全一致**：
  - 都靠 `env.map_weed` 计算覆盖率。
  - 都记录 90/95/98% 覆盖时的路径长度。
  - 都在 episode 结束时写 CSV 并 `exit()`。

- 这意味着：
  - 在相同地图 + 障碍配置下，可以直接对比 CSV 中的 `cover_xx`/`dist_list`，评估 RL 与几何基线谁更高效。

### 4.5 `dqn_test.py`：DQN 离散控制评估

逻辑与 `sac_cont_test.py` 十分相似，但有几个关键差异：

- 环境动作是 **离散** 的：

```python
action = actor(observation=observation, vector=vector)
action = action[0].argmax().item()
obs, reward, done, time_out, _ = env.step(action)
```

- 观察/向量处理方式略有差异（`vector` 被包了 `[vector]` 再 unsqueeze）。
- **硬编码路径**：

```python
LOG_DIR = '/Users/chuyuliu/CppRL-main-chuyu/logs'
pt_path = '/Users/chuyuliu/CppRL-main-chuyu/ckpt/dqn_model_3_0907.pt'
```

  - 若在本仓库直接运行，需要手动改成例如：

    ```python
    LOG_DIR = str(Path(__file__).resolve().parents[1] / 'rules' / 'logs')
    pt_path = str(Path(__file__).resolve().parents[2] / 'ckpt' / 'dqn_model_3_0907.pt')
    ```

- 覆盖指标 & 超时逻辑与 SAC 版本完全一致，输出 CSV 字段也相同。

### 4.6 `script.py`：批量实验调度器（自修改脚本）

这是一个“**脚本写脚本**”的工具，用 for-loop 扫描参数空间，然后 **直接重写 `env_make.py` 与 `sac_cont_test.py` 源码**，再调用 Python 运行。

核心函数 `run_all(...)`：

```python
def run_all(seed, difficulty, map_id, obstacle_range, weed_num):
    for weed_dist in ['gaussian', 'uniform']:
        for noise_set in [[0,0,0]]:
            # 1) 重写 env_make.py
            with open(BASE_DIR / "rules" / "env_make.py", 'r+') as file:
                config_content = file.readlines()
                file.seek(0)
                for line in config_content:
                    if "env = gym.make(" in line:
                        line = f"    env = gym.make(id=\"Pasture-v2\", ..., num_obstacles_range={(obstacle_range[0], obstacle_range[1])}, ..., noise_position={noise_set[0]}, ...)\n"
                    if "obs, info = env.reset(" in line:
                        line = f"    obs, info = env.reset(seed={seed}, options={{'weed_dist': '{weed_dist}', 'map_id': {map_id}, 'weed_num': {weed_num}}})\n"
                    file.write(line)
                file.truncate()

            # 2) 重写 sac_cont_test.py
            for file_add in ["rules/ckpt/t[02600]_r[2731.41=2717.75~2750.74].pt"]:
                before_t = file_add.split('/')[-1].split('_t[')[0]
                with open(BASE_DIR / "rules" / "sac_cont_test.py", "r+") as file:
                    config_content = file.readlines()
                    file.seek(0)
                    for line in config_content:
                        if "difficulty = " in line:
                            line = f"difficulty = \"{difficulty}\"\n"
                        elif "rl_model =" in line:
                            line = f"rl_model = \"{before_t}\"\n"
                        elif "weed_dist = " in line:
                            line = f"weed_dist = \"{weed_dist}\"\n"
                        elif "random_seed = " in line:
                            line = f"random_seed = {seed}\n"
                        elif "map_id =" in line:
                            line = f"map_id = {map_id}\n"
                        elif "pt_path =" in line:
                            line = f"pt_path = '{file_add}'\n"
                        elif "noise_set =" in line:
                            line = f"noise_set = {noise_set}\n"
                        file.write(line)
                    file.truncate()

                print("running sac...")
                subprocess.run(["python", str(sac_script)])
```

底部是参数网格：

```python
for seed in [25,27,47,21,31]:
    for hard_degree in ["easy", "medium", "hard"]:
        if hard_degree == "easy":
            maps = [2, 3, 6, 16, 20]; obstacle_range = [0,0]; weed_num = 50
        elif hard_degree == "medium":
            maps = [4, 9, 21, 59, 80]; obstacle_range = [2,4]; weed_num = 100
        else:
            maps = [22, 29, 57, 63, 22]; obstacle_range = [5,8]; weed_num = 200
        for map_id in maps:
            run_all(seed, hard_degree, map_id, obstacle_range, weed_num)
```

使用建议：

- 把它看作一个「一次性实验脚本」，跑完后用 `git diff` 检查它对 `env_make.py` 和 `sac_cont_test.py` 做了哪些改动。
- 如果要改成“更现代”的方式，可以考虑用 Hydra 配置/命令行参数替代文本重写；但在当前仓库中保持原样即可。

---

## 5. 使用示例：从 0 到跑通

### 5.1 跑几何基线 R_SNAKE 覆盖

1. 关闭渲染（如果在服务器上）：
   - 编辑 `rules/env_make.py`，将 `render = True` 改为 `False`，并可去掉 `HumanRendering` 包装。
2. 在 `rules/jump_path.py` 顶部设定模式：

```python
task_type = "R_SNAKE"   # 或 "BCP" / "SNAKE" / "JUMP" / "REACT"
difficulty = "easy"
weed_dist = "gaussian"
random_seed = 25
map_id = 2
```

3. 运行：

```bash
python rules/jump_path.py
```

4. 查看结果：
   - 打开 `rules/logs/coverage_results_R_SNAKE_easy.csv`，里面有：
     - `cover_90 / cover_95 / cover_98`：到达阈值时的路径长度。
     - `cover`：每步的覆盖率轨迹。
     - `dist_list`：每步累计路径长度。

### 5.2 评估 SAC 连续控制模型

1. 确认权重存在：`rules/ckpt/t[02600]_r[2731.41=2717.75~2750.74].pt`。
2. 修改 `rules/sac_cont_test.py` 中参数：

```python
render = False           # 服务器无渲染
act_randomly = False     # 用确定性策略
pt_path = 'rules/ckpt/your_model.pt'
rl_model = 'your_model'  # 用于 CSV 文件名
```

3. 运行：

```bash
python rules/sac_cont_test.py
```

4. 查看结果：
   - CSV 文件：`rules/logs/your_model_easy.csv`（difficulty 默认为 "easy"）。
   - 字段与几何基线相同，可以直接拿来对比。

### 5.3 批量实验（小心使用）

如果需要复现实验论文中的整套结果，可以直接：

```bash
python rules/script.py
```

注意：

- 它会修改源文件；完成后建议：
  - `git diff rules/env_make.py rules/sac_cont_test.py` 看清改动。
  - 如有需要，用 `git checkout -- rules/env_make.py rules/sac_cont_test.py` 还原。

---

## 6. 与新栈（envs_new + rl_new）如何衔接

从设计视角看：

- `rules/`：
  - 环境：`Pasture-v2`；覆盖指标基于 `env.map_weed`；信息字段较简单。
  - 代码风格：脚本式、全局状态 + 早退 `exit()`；几何/策略逻辑直接糅在一个文件里。
- 新栈 `envs_new/` + `rl_new/sac_cont_sy/`：
  - 环境：`NewPasture-v*`；有统一的 `EnvironmentState`、观测/奖励组件化；
  - 覆盖指标：`completion_ratio`（weed or field）+ `overlap_count` + `steps_95_to_done` 等；
  - 训练/评估完全由 TorchRL + Hydra 驱动，避免了脚本级自修改。

如果你需要把 `rules/` 中的某个思路迁移到新栈：

1. 几何算法迁移：
   - 把 `jump_path.py` 中的路径生成逻辑（`find_longest_edge`、扫线 + JUMP/SNAKE 策略）抽成 **“规划器类”**，只负责给出目标 waypoint 序列；
   - 在 `envs_new` 环境中写一个小脚本：从规划器拿到 waypoint，用与 `go(p2)` 类似的转换得到动作并调用 `env.step()`。
2. 覆盖指标对齐：
   - 旧版：手动从 `env.map_weed` 计算覆盖率；
   - 新版：直接读 `info['completion_ratio']` 和 `info['overlap_count']`，再通过 TorchRL transforms 做统计。

---

## 7. 快速心智模型 & 小结

如果只记一张图，可以记这张：

```text
             Pasture-v2 环境（envs.cpp_env_v2:CppEnv）
                           ▲
           ┌───────────────┼────────────────┐
           │               │                │
     jump_path.py   sac_cont_test.py   dqn_test.py
      (几何基线)      (SAC 连续控制)    (DQN 离散控制)
           │               │                │
           └───────────▶ logs/*.csv ◀───────┘
                            ▲
                            │
                       script.py
              （批量改参数 + 调度评估）
```

- 想看覆盖算法细节 → 读 `jump_path.py`，重点关注 `go()`、扫线循环、JUMP/SNAKE/R_SNAKE 分支。
- 想看 RL 表现 → 读 `sac_cont_test.py` / `dqn_test.py`，关注如何从 `obs` 解析出 `observation/vector`，以及动作索引。
- 想批量扫参数 → 看 `script.py` 如何重写 `env_make.py` 和 `sac_cont_test.py`。

这就是 `rules/` 子目录的核心功能与框架。
