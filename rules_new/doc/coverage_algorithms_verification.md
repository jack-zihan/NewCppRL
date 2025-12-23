# 覆盖规划算法实现计划

## 目标
在v2环境中复现RAL 2022论文的5个算法：**BCP, REACT, JUMP, SNAKE, R-SNAKE**

## 确认信息
- **动作接口**：`env.step([v, w])`，语义等同于`[length, delta_angle]`（step=1，先转弯再前进）
- **文件位置**：新建`rules_new1/`目录
- **可视化**：暂不考虑，专注核心算法

---

## 文件结构

```
rules_new1/
├── __init__.py
├── coverage_planners.py     # 所有5个算法实现（约400行）
└── test_planners.py         # 简单测试脚本
```

---

## 核心设计

### 1. 关键参数提取
```python
B = env.config.agent_width  # 4.0，BCP覆盖间距
S_w = 2 * vision_length * sin(vision_angle/2)  # ~34.1，FOV弦宽
R = v_max / (w_max * π/180)  # ~7.0，最小转弯半径
```

### 2. 坐标系变换（参考rules/jump_path.py:261-268）
将旋转的bounding box变换为正坐标系：
- `world_to_local(x, y)` → 局部坐标（x沿长边，y垂直）
- `local_to_world(lx, ly)` → 世界坐标
- 变换基于最长边方向角`theta`

### 3. 统一接口
```python
class BaseCoveragePlanner:
    def __init__(self, env): ...
    def reset(self): ...
    def act(self) -> tuple[float, float] | None:
        """返回(v, w)动作，None表示完成"""
```

### 4. 各算法核心逻辑

| 算法 | pass间距 | 核心机制 |
|------|---------|---------|
| **BCP** | `B` | 牛耕式遍历 |
| **JUMP** | `min(S_w/2, weed+B/2)` | Spring动态间距 + Jump闭合环 |
| **SNAKE** | `S_w/2 + B/2` | 双向detour（不返回） |
| **R-SNAKE** | 同SNAKE | 限制detour：`y_weed >= y_p - 1.5*S_w` |
| **REACT** | N/A | 随机搜索 + 最近杂草追踪 |

### 5. Dubins路径（参考rules/jump_path.py:168-173）
```python
import dubins
path = dubins.shortest_path(start_pose, end_pose, R)
configs, _ = path.sample_many(step_size=0.5)
```

---

## 实现步骤

### Phase 1: 基础设施（~100行）
1. 创建`rules_new1/`目录
2. 坐标系变换函数（`setup_coordinate_system`, `world_to_local`, `local_to_world`）
3. Dubins路径规划函数
4. 路径跟踪函数（单步移动到目标点）
5. 杂草检测函数

### Phase 2: BCP基线（~50行）
1. `BCPPlanner`类实现
2. pass线生成
3. 换行逻辑

### Phase 3: JUMP算法（~80行）
1. `JUMPPlanner`类实现
2. Spring机制（动态y_offset计算）
3. Jump机制（LSR-RSL闭合环）
4. `get_forward_jump()`过滤函数

### Phase 4: SNAKE算法（~60行）
1. `SNAKEPlanner`类实现
2. `get_forward_snake()`过滤函数
3. 双向detour逻辑

### Phase 5: R-SNAKE和REACT（~60行）
1. `RSNAKEPlanner`继承SNAKE，添加y限制
2. `REACTPlanner`随机搜索实现

### Phase 6: 测试（~50行）
1. 简单测试脚本验证各算法

---

## 关键文件参考

| 文件 | 用途 |
|------|------|
| `rules/jump_path.py` | Dubins使用、坐标变换、算法原型 |
| `envs_new/cpp_env_v2.py` | 环境接口 |
| `envs_new/components/config/environment_config.py` | 配置参数 |
| `rules_new/doc/RAL_2022_coverage_planning_algorithms_analysis.md` | 论文算法分析 |

---

## 预估
- **代码量**：~400行（单文件）
- **时间**：2-3小时

---

## 注意事项
1. **坐标顺序**：`rules/jump_path.py`中`agent_position = [env.agent.y, env.agent.x]`需确认v2环境是否一致
2. **动力学模型**：先转弯(w)再前进(v)，单步完成
3. **Less is More**：避免过度抽象，直接实现核心逻辑

---

## 🔍 算法正确性验证（基于论文分析 + 现有代码审查）

### 1. JUMP算法 ✅ 结构正确，⚠️ 细节需修正

**论文要求**（RAL 2022, Section IV-A）：
- **Jump定义**：从当前位置(a)出发，访问杂草(b)，返回当前pass的闭合路径
- **Dubins类型**：
  - θ_p = 0（向右）：去程=LSR，返程=RSL
  - θ_p = π（向左）：去程=RSL，返程=LSR
- **三个航点航向角均固定为θ_p**
- **约束**：只访问当前pass**上方**的杂草

**现有代码（rules/jump_path.py:479-502）分析**：
```python
# Line 496-499 确实有去程+返程：
navigate(valid_points[int(i - (4 * turning_radius))])  # 入口点
dubins_navigate([weed[0], weed[1], rad_n], turning_radius)  # 去weed
dubins_navigate([valid_points[int(i + (4 * turning_radius))], ...], turning_radius)  # 回pass
```

**✅ 正确的部分**：
1. 有去程（to weed）和返程（back to pass）两段
2. `get_forward_jump()`正确过滤只取上方杂草
3. 入口/出口点计算：`i ± 4*R` 合理（留出转弯空间）

**⚠️ 需要修正的部分**：
1. **Dubins类型未指定**：`dubins.shortest_path()`返回任意最短路径，不保证LSR+RSL组合
   - dubins库支持指定类型：`dubins.path_type()`
   - 需要改为显式指定路径类型
2. **Spring机制公式**（Line 534）：
   ```python
   # 当前代码
   y_offset = min(y_offset + weed_offset + B/2, y_offset + S_w/2, max - B/2)
   # 论文公式
   y_p(i+1) = min(y_p(i) + S_w/2, min(y_i + B/2 for w_i in W), W - B/2)
   ```
   - 需要取所有未割杂草的最小y坐标，而不是当前发现的最低点

---

### 2. SNAKE算法 ✅ 正确

**论文要求**：
- 双向detour（上方用LSR，下方用RSL）
- **不返回当前pass**，从杂草位置继续前进
- 固定间距：`y_p(i+1) = y_p(i) + S_w/2 + B/2`

**现有代码（rules/jump_path.py:504-524）分析**：
```python
weed = find_nearest_point(agent_position, forward, turning_radius)
if weed is not None:
    dubins_navigate([weed[0], weed[1], rad_n], turning_radius)  # 只去，不回！
    # 从杂草位置重新生成路径点
    valid_points = [new points along current direction]
    p_i = 0  # 重置索引
```

**✅ 全部正确**：
1. 访问杂草后不返回pass
2. 从杂草位置重新生成沿当前方向的路径点
3. 间距公式正确（Line 545-547）：`y_offset + S_w/2 + B/2`

---

### 3. R-SNAKE算法 ✅ 正确

**论文要求**：
- 继承SNAKE行为
- 额外约束：只考虑 `y_i >= y_p - 1.5*S_w` 的杂草

**现有代码（rules/jump_path.py:296-304）**：
```python
def get_forward_rsnake(discovered, point, rad, real_radians):
    upward_vector = np.array([np.cos(real_radians + np.pi / 2), ...])
    final_points = [
        p for p in discovered
        if np.dot(p - point, forward_vector) > 0
        and np.dot(p - point, upward_vector) > -3/2 * sight_width  # -1.5*S_w
    ]
```

**✅ 正确**：`-3/2 * sight_width` 正是 `-1.5*S_w`

---

### 4. BCP算法 ✅ 正确

**论文要求**：
- 牛耕式覆盖，pass间距 = B（agent_width）

**现有代码（rules/jump_path.py:525-529, 549）**：
```python
while p_i < len(valid_points):
    navigate(valid_points[p_i])
    p_i += 1
# ...
y_offset += agent_width  # 间距 = B
```

**✅ 完全正确**

---

### 5. REACT算法 ⚠️ 需要修正

**论文要求**（RAL 2022, Section III-C）：
- 无杂草时：随机生成航点搜索
- 有杂草时：**FIFO顺序**访问
- 终止条件：路径长度达到BCP上界

**现有代码（rules/jump_path.py:381-418）**：
```python
while times < 50:  # ❌ 按迭代次数终止
    # 随机搜索...
    weed = find_nearest_point(agent_position, discovered, 0)  # ❌ 最近点，非FIFO
    while weed is not None:
        dubins_navigate(...)
        weed = find_nearest_point(...)  # 继续取最近点
```

**❌ 需要修正**：
1. 杂草访问顺序应为FIFO（先检测到的先访问），而非nearest-first
2. 终止条件应为 `path_length >= BCP_length`，而非迭代次数

---

## 📋 修正后的实现要点

### JUMP修正
```python
# 显式指定Dubins路径类型（已验证API可用）
import dubins
# dubins.LSL=0, dubins.LSR=1, dubins.RSL=2, dubins.RSR=3, dubins.RLR=4, dubins.LRL=5

def execute_jump(start_pose, weed_pose, return_pose, R, direction):
    """
    执行JUMP闭合环：去weed + 回pass

    Args:
        start_pose: (x, y, θ) 入口点（pass上）
        weed_pose: (x, y, θ) 杂草位置，θ=direction
        return_pose: (x, y, θ) 出口点（pass上）
        R: 最小转弯半径
        direction: 当前pass方向，0=向右，π=向左
    """
    if abs(direction) < 0.1:  # θ_p ≈ 0（向右）
        go_path = dubins.path(start_pose, weed_pose, R, dubins.LSR)  # Left-Straight-Right
        return_path = dubins.path(weed_pose, return_pose, R, dubins.RSL)  # Right-Straight-Left
    else:  # θ_p ≈ π（向左）
        go_path = dubins.path(start_pose, weed_pose, R, dubins.RSL)
        return_path = dubins.path(weed_pose, return_pose, R, dubins.LSR)

    # 采样路径点
    go_configs, _ = go_path.sample_many(0.5)
    return_configs, _ = return_path.sample_many(0.5)

    return go_configs + return_configs
```

### REACT修正
```python
class REACTPlanner:
    def __init__(self):
        self.weed_queue = deque()  # FIFO队列
        self.path_length = 0.0
        self.bcp_length = compute_bcp_length()  # 预计算上界

    def act(self):
        # 更新检测到的杂草（按检测顺序加入队列尾部）
        new_weeds = detect_weeds_in_fov()
        for w in new_weeds:
            if w not in self.weed_queue:
                self.weed_queue.append(w)  # FIFO入队

        if self.weed_queue:
            target = self.weed_queue.popleft()  # FIFO出队
            return navigate_to(target)
        else:
            # 随机搜索
            return random_waypoint()

        # 终止条件
        if self.path_length >= self.bcp_length:
            return None
```

---

## 🎯 最终实现计划（修订版）

### 核心修改点
1. **JUMP**：使用 `dubins.path()` 指定LSR/RSL类型，而非 `dubins.shortest_path()`
2. **REACT**：改用 `deque` 实现FIFO杂草队列
3. **Spring机制**：正确实现 `min(y_i + B/2 for all w_i in W)`

### 代码结构（约450行）
```python
# coverage_planners.py

# === 工具函数 ===
def setup_coordinate_system(bbox) -> CoordSystem
def world_to_local(x, y, cs) -> (lx, ly)
def local_to_world(lx, ly, cs) -> (x, y)
def dubins_path_typed(start, end, R, path_type) -> List[Point]  # 新增：指定类型
def detect_weeds_in_fov(env, field_mask) -> List[Point]

# === 基类 ===
class BaseCoveragePlanner:
    def __init__(self, env)
    def reset()
    def act() -> Optional[Tuple[float, float]]

# === BCP ===
class BCPPlanner(BaseCoveragePlanner):
    # pass间距 = B，简单牛耕

# === JUMP ===
class JUMPPlanner(BaseCoveragePlanner):
    # Spring机制 + LSR-RSL闭合Jump
    def _compute_next_y(self, W) -> float  # 论文公式(1)
    def _execute_jump(self, weed, direction)  # LSR+RSL

# === SNAKE ===
class SNAKEPlanner(BaseCoveragePlanner):
    # 双向detour，不返回pass
    def _compute_next_y(self) -> float  # 固定间距

# === R-SNAKE ===
class RSNAKEPlanner(SNAKEPlanner):
    # 继承SNAKE，添加y限制过滤

# === REACT ===
class REACTPlanner(BaseCoveragePlanner):
    # FIFO队列 + 随机搜索
    weed_queue: deque
    def _terminate(self) -> bool  # path_length >= BCP_length
```
