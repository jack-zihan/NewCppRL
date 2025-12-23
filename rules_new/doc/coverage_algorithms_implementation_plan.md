# 覆盖规划算法实现计划

## 目标
在v2环境中复现RAL 2022论文的5个算法：BCP, REACT, JUMP, SNAKE, R-SNAKE

## 设计原则
- **Less is More**：科学研究导向，不过度工程化
- **单文件实现**：所有算法放在一个文件中（约400-500行）
- **借鉴现有代码**：参考`rules/jump_path.py`的成熟逻辑

---

## 一、关键参数

### 从环境配置提取
```python
# 覆盖宽度 B（BCP间距）
B = env.config.agent_width  # 4.0

# FOV探索宽度 S_w（JUMP/SNAKE间距基准）
vision_length = env.config.agent_vision_length  # 28.0
vision_angle = env.config.agent_vision_angle    # 75.0
S_w = 2 * vision_length * math.sin(math.radians(vision_angle / 2))  # ≈ 34.1

# 最小转弯半径 R
v_max = env.config.v_max  # 3.5
w_max_rad = abs(env.config.w_max) * math.pi / 180  # 28.6° → 0.499 rad
R = v_max / w_max_rad  # ≈ 7.0
```

### 间距公式
| 算法 | pass间距 | 论文公式 |
|------|---------|---------|
| BCP | `B` | 固定车宽 |
| JUMP | `min(S_w/2, weed_offset + B/2)` | 公式(1)动态Spring |
| SNAKE | `S_w/2 + B/2` | 公式(3)固定 |
| R-SNAKE | 同SNAKE | 同上 |

---

## 二、坐标系变换

### 问题
v2环境的bounding box可能是旋转的，但算法假设正坐标系（x沿长边，y垂直）。

### 解决方案
参考`rules/jump_path.py`的实现：

```python
def setup_coordinate_system(env):
    """建立局部坐标系，使x轴沿长边方向"""
    # 获取bounding box顶点
    bbox = env.env_state.get_static_info('bounding_box')[0].reshape(-1, 2)

    # 找最长边作为x轴方向
    edges = [(bbox[i], bbox[(i+1)%4]) for i in range(4)]
    longest = max(edges, key=lambda e: np.linalg.norm(e[1] - e[0]))
    start, end = longest

    # 计算旋转角度
    direction = end - start
    theta = np.arctan2(direction[1], direction[0])  # 长边方向角

    # 返回变换参数
    return {
        'origin': start,           # 局部坐标系原点
        'theta': theta,            # 旋转角度
        'cos_t': np.cos(theta),
        'sin_t': np.sin(theta),
        'width': np.linalg.norm(direction),  # 长边长度（局部x范围）
        'height': ...              # 短边长度（局部y范围）
    }

def world_to_local(x, y, coord_sys):
    """世界坐标 → 局部坐标"""
    dx = x - coord_sys['origin'][0]
    dy = y - coord_sys['origin'][1]
    local_x = dx * coord_sys['cos_t'] + dy * coord_sys['sin_t']
    local_y = -dx * coord_sys['sin_t'] + dy * coord_sys['cos_t']
    return local_x, local_y

def local_to_world(lx, ly, coord_sys):
    """局部坐标 → 世界坐标"""
    x = coord_sys['origin'][0] + lx * coord_sys['cos_t'] - ly * coord_sys['sin_t']
    y = coord_sys['origin'][1] + lx * coord_sys['sin_t'] + ly * coord_sys['cos_t']
    return x, y
```

---

## 三、核心工具函数

### Dubins路径（参考rules/jump_path.py:168-173）
```python
import dubins

def plan_dubins_path(start_pose, end_pose, radius, step_size=0.5):
    """规划Dubins路径并采样点"""
    path = dubins.shortest_path(
        (start_pose[0], start_pose[1], start_pose[2]),
        (end_pose[0], end_pose[1], end_pose[2]),
        radius
    )
    configs, _ = path.sample_many(step_size)
    return [(c[0], c[1], c[2]) for c in configs]
```

### 路径跟踪（参考rules/jump_path.py:96-148）
```python
def step_to_point(env, target_x, target_y, current_rad):
    """单步移动到目标点，返回新的rad"""
    ax, ay = env.agent.y, env.agent.x  # 注意：rules中xy是反的

    # 计算方向和距离
    radian = math.atan2(target_y - ay, target_x - ax)
    length = math.hypot(target_x - ax, target_y - ay)

    # 计算相对转角
    delta_angle = (radian - current_rad)
    delta_angle = (delta_angle + math.pi) % (2 * math.pi) - math.pi
    delta_angle = math.degrees(delta_angle)

    # 执行动作
    obs, reward, done, truncated, info = env.step([length, -delta_angle])

    # 更新rad
    new_rad = math.pi / 2 - math.radians(env.agent.direction)
    return new_rad, done
```

### 杂草检测
```python
def get_visible_weeds(env, field_mask):
    """获取FOV内可见且在田地内的杂草"""
    weed_map = env.maps_dict['weed']
    frontier_map = env.maps_dict.get('mist', np.ones_like(weed_map))

    # 已发现但未清除的杂草
    discovered = np.argwhere(np.logical_and(weed_map, np.logical_not(frontier_map)))

    # 过滤：只保留田地内的点
    return [p for p in discovered if field_mask[p[0], p[1]]]
```

---

## 四、算法实现结构

### 统一接口
```python
class BaseCoveragePlanner:
    """基类：所有算法的共同行为"""

    def __init__(self, env):
        self.env = env
        self.coord_sys = setup_coordinate_system(env)
        self.B = env.config.agent_width
        self.S_w = 2 * env.config.agent_vision_length * \
                   math.sin(math.radians(env.config.agent_vision_angle / 2))
        self.R = env.config.v_max / (abs(env.config.w_max) * math.pi / 180)
        self.rad = ...  # 当前朝向（弧度）
        self.field_mask = self._compute_field_mask()

    def reset(self):
        """重置规划器状态"""
        pass

    def act(self) -> Tuple[float, float]:
        """返回下一步动作 (v, w)"""
        raise NotImplementedError

    def _navigate_dubins(self, goal_pose):
        """Dubins导航到目标"""
        pass
```

### 各算法实现（伪代码）

#### BCP
```python
class BCPPlanner(BaseCoveragePlanner):
    """牛耕式覆盖，pass间距 = B"""

    def reset(self):
        self.y_offset = self.B / 2  # 从底部开始
        self.pass_points = self._generate_pass_line(self.y_offset)
        self.point_idx = 0
        self.turn_dir = 1  # 1=正向，-1=反向

    def act(self):
        if self.point_idx >= len(self.pass_points):
            # 换行
            self.y_offset += self.B
            if self.y_offset > self.coord_sys['height'] - self.B/2:
                return None  # 完成
            self.pass_points = self._generate_pass_line(self.y_offset)
            self.turn_dir *= -1
            self.point_idx = 0
            # Dubins换行
            self._navigate_dubins(...)

        # 直行到下一点
        target = self.pass_points[self.point_idx]
        action = self._step_to_point(target)
        self.point_idx += 1
        return action
```

#### JUMP
```python
class JUMPPlanner(BaseCoveragePlanner):
    """JUMP算法：Spring + Jump机制"""

    def act(self):
        discovered = self.get_visible_weeds()

        # 检查是否有可跳跃的杂草（上方、前方无障碍）
        forward_weeds = self._get_forward_jump_weeds(discovered)
        if forward_weeds:
            nearest = self._find_nearest_in_direction(forward_weeds)
            if self._can_jump(nearest):
                # 执行Jump：LSR-RSL闭合环
                self._execute_jump(nearest)
                return self.act()  # 递归继续

        # 否则直行
        if self.point_idx < len(self.pass_points):
            target = self.pass_points[self.point_idx]
            self.point_idx += 1
            return self._step_to_point(target)

        # 换行：Spring机制计算下一pass
        self.y_offset = self._compute_next_y(discovered)
        ...

    def _compute_next_y(self, weeds):
        """公式(1)：y_next = min(y_p + S_w/2, min(y_weed + B/2), H - B/2)"""
        c1 = self.y_offset + self.S_w / 2
        c2 = min((self._world_to_local(w)[1] + self.B/2 for w in weeds), default=float('inf'))
        c3 = self.coord_sys['height'] - self.B / 2
        return min(c1, c2, c3)
```

#### SNAKE
```python
class SNAKEPlanner(BaseCoveragePlanner):
    """SNAKE算法：双向detour，固定间距"""

    def act(self):
        discovered = self.get_visible_weeds()
        forward_weeds = self._get_forward_snake_weeds(discovered)

        if forward_weeds:
            # 找最近杂草，执行Dubins子路径（不返回）
            nearest = self._find_nearest(forward_weeds)
            self._navigate_dubins((nearest[0], nearest[1], self.current_rad))
            # 从杂草位置继续沿当前方向生成新pass
            self._regenerate_pass_from_current()
            return self.act()

        # 直行
        ...

    def _compute_next_y(self):
        """公式(3)：y_next = y_p + S_w/2 + B/2"""
        return self.y_offset + self.S_w / 2 + self.B / 2
```

#### R-SNAKE
```python
class RSNAKEPlanner(SNAKEPlanner):
    """R-SNAKE：限制向下detour范围"""

    def _get_forward_snake_weeds(self, weeds):
        """公式(5)：只考虑 y_weed >= y_p - 1.5*S_w 的杂草"""
        forward = super()._get_forward_snake_weeds(weeds)
        threshold = self.y_offset - 1.5 * self.S_w
        return [w for w in forward if self._world_to_local(w)[1] >= threshold]
```

#### REACT
```python
class REACTPlanner(BaseCoveragePlanner):
    """REACT：随机搜索 + FIFO杂草访问"""

    def act(self):
        discovered = self.get_visible_weeds()

        if discovered:
            # 追踪最近杂草
            nearest = self._find_nearest(discovered)
            self._navigate_dubins((nearest[0], nearest[1], self.current_rad))
            return self.act()

        # 随机搜索
        if not self.random_goal or self._reached(self.random_goal):
            self.random_goal = self._generate_random_goal()
            self._navigate_dubins(self.random_goal)

        return self._step_towards(self.random_goal)
```

---

## 五、与Gym的集成

### 使用模式
```python
# 创建环境
env = gym.make('CppEnv-v2', ...)

# 创建规划器
planner = JUMPPlanner(env)
planner.reset()

# 主循环
obs, info = env.reset()
done = False
while not done:
    action = planner.act()
    if action is None:
        break  # 算法完成
    obs, reward, done, truncated, info = env.step(action)
```

### 关键适配
1. **动作格式**：`env.step([length, delta_angle])` 需要连续动作模式
2. **状态访问**：直接从`env.agent`, `env.maps_dict`获取
3. **坐标注意**：`rules/jump_path.py`中`agent_position = [env.agent.y, env.agent.x]`（xy反转）

---

## 六、文件结构

```
rules_new/
├── coverage_planners.py     # 所有5个算法实现（约400-500行）
├── test_coverage.py         # 测试脚本
└── doc/
    ├── RAL_2022_coverage_planning_algorithms_analysis.md  # 论文分析
    └── coverage_algorithms_implementation_plan.md          # 本文档
```

---

## 七、实现优先级

1. **Phase 1**：工具函数（坐标变换、Dubins、路径跟踪）
2. **Phase 2**：BCP基线实现
3. **Phase 3**：JUMP和SNAKE实现
4. **Phase 4**：R-SNAKE和REACT实现
5. **Phase 5**：测试验证

---

## 八、预估

- **代码量**：约400-500行（单文件）
- **复杂度**：中等（主要是坐标变换和算法逻辑）
- **时间**：2-3小时完成核心实现

---

## 九、待确认问题

1. **动作模式确认**：v2环境是否支持`env.step([length, delta_angle])`？还是需要`env.step([v, w])`？
   - `rules/jump_path.py`使用`env.set_action_type("continuous")`然后`env.step([length, delta_angle])`
   - 需要确认v2环境的动作接口

2. **xy坐标顺序**：`rules/jump_path.py`中`agent_position = [env.agent.y, env.agent.x]`是否正确？

3. **FOV弦长计算**：`S_w = 2 * vision_length * sin(vision_angle/2)` 是否正确？
