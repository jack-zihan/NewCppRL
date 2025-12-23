# 新版rules_new代码完整分析报告 [FIXED]

## 执行摘要
- **代码规模**：约2500行，模块化架构，20+个类，50+个方法
- **技术栈**：Python + NumPy + Dubins + 策略模式 + 依赖注入
- **核心功能**：模块化、可扩展的多算法路径规划框架
- **架构特点**：面向对象，策略模式，统一接口，配置驱动

## 一、文件结构与模块组织

### 1.1 目录结构
```
rules_new/
├── __init__.py                 # 包初始化
├── main.py                     # 主入口
├── algorithms/                 # 算法模块
│   ├── __init__.py
│   ├── base_algorithm.py       # 基类定义
│   ├── jump_planner.py         # JUMP算法
│   ├── snake_planner.py        # SNAKE算法
│   ├── bcp_planner.py          # BCP算法
│   ├── react_planner.py        # REACT算法
│   └── nn_planner.py           # 神经网络算法
├── experiment/                 # 实验管理
│   ├── __init__.py
│   ├── experiment_runner.py    # 实验执行器
│   ├── config_manager.py       # 配置管理
│   ├── result_collector.py     # 结果收集
│   └── batch_manager.py        # 批处理管理
└── utils/                      # 工具模块
    ├── __init__.py
    ├── geometry_utils.py       # 几何工具
    ├── path_utils.py           # 路径工具
    ├── trajectory_collector.py # 轨迹收集
    └── logging_utils.py        # 日志工具
```

### 1.2 架构设计
```
┌─────────────────────────────────────────┐
│            main.py (入口)                │
└─────────────┬───────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│     ExperimentRunner (实验管理器)        │
├─────────────────────────────────────────┤
│ - 配置加载                               │
│ - 算法实例化                             │
│ - 环境管理                               │
│ - 结果收集                               │
└─────────────┬───────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│    BasePathPlanner (算法基类)            │
├─────────────────────────────────────────┤
│ + reset()                                │
│ + plan_next_waypoint()                   │
│ + should_terminate()                     │
│ + update_state()                         │
└─────────────┬───────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│   具体算法实现 (Strategy Pattern)        │
├─────────────────────────────────────────┤
│ - JumpPlanner                            │
│ - SnakePlanner                           │
│ - BCPPlanner                             │
│ - ReactPlanner                           │
│ - NNPlanner                              │
└─────────────────────────────────────────┘
```

## 二、业务流程详解

### 2.1 系统启动流程

#### 2.1.1 主入口初始化
```python
# main.py
def main():
    """
    系统入口
    执行流程：
    1. 解析命令行参数
    2. 加载配置文件
    3. 创建实验运行器
    4. 执行实验批次
    5. 保存结果
    """
    args = parse_arguments()
    config = load_config(args.config)
    
    # 创建实验运行器
    runner = ExperimentRunner(config)
    
    # 执行实验
    results = runner.run_experiments()
    
    # 保存结果
    save_results(results, config.output_path)
```

#### 2.1.2 实验运行器初始化
```python
# experiment/experiment_runner.py
class ExperimentRunner:
    def __init__(self, config: Dict[str, Any]):
        """
        实验运行器初始化
        职责：
        1. 管理算法实例
        2. 控制环境生命周期
        3. 收集实验数据
        """
        self.config = config
        self.env_config = config['environment']
        self.algorithm_config = config['algorithm']
        
        # 创建环境
        self.env = self._create_environment()
        
        # 实例化算法
        self.planner = self._create_planner()
        
        # 结果收集器
        self.collector = ResultCollector()
```

### 2.2 算法基类设计（核心抽象）

#### 2.2.1 BasePathPlanner基类
```python
# algorithms/base_algorithm.py 完整实现分析
class BasePathPlanner:
    """
    路径规划算法基类
    设计模式：模板方法 + 策略模式
    """
    
    def __init__(self, config: Dict[str, Any], env_config: Dict[str, Any]):
        """初始化基础参数"""
        self.config = config
        self.env_config = env_config
        
        # 通用状态
        self.current_position = None
        self.current_direction = None
        self.discovered_weeds = []
        self.coverage_rate = 0.0
        self.start_time = None
        self.iteration_count = 0
        
    def reset(self, initial_state: Dict[str, Any]):
        """
        重置算法状态
        输入：初始环境状态
        """
        self.current_position = initial_state['agent_position']
        self.current_direction = initial_state['agent_direction']
        self.discovered_weeds = []
        self.coverage_rate = 0.0
        self.start_time = time.time()
        self.iteration_count = 0
        
    def update_state(self, current_state: Dict[str, Any]):
        """
        更新内部状态
        输入：当前环境状态
        """
        self.current_position = current_state['agent_position']
        self.current_direction = current_state['agent_direction']
        
        # 更新发现的杂草
        if 'discovered_weeds' in current_state:
            self.discovered_weeds = current_state['discovered_weeds']
            
        # 更新覆盖率
        if 'coverage_rate' in current_state:
            self.coverage_rate = current_state['coverage_rate']
            
        self.iteration_count += 1
        
    @abstractmethod
    def plan_next_waypoint(self, current_state: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """
        规划下一个路径点（核心方法，子类必须实现）
        返回：
            - None: 终止
            - (x, y): 单个路径点
            - ('path', [(x1,y1), (x2,y2), ...]): 路径列表
        """
        pass
        
    def should_terminate(self, current_state: Dict[str, Any]) -> bool:
        """
        判断是否终止
        条件：
        1. 覆盖率达标
        2. 超时
        3. 达到最大迭代次数
        """
        # 覆盖率检查
        if self.coverage_rate >= self.config.get('coverage_threshold', 0.98):
            return True
            
        # 超时检查
        if self.check_timeout():
            return True
            
        # 迭代次数检查
        if self.check_max_iterations():
            return True
            
        return False
```

#### 2.2.2 路径工具方法
```python
# algorithms/base_algorithm.py 路径分解方法 [FIXED: 完整实现]
@staticmethod
def decompose_path(start: List[float], target: List[float], step_size: float = 2.0) -> List[List[float]]:
    """
    将路径分解为小步（复现navigate逻辑）
    
    算法流程：
    1. 计算方向向量和距离
    2. 按step_size分解
    3. 生成中间点列表
    
    输入：
        start: 起始点 [x, y]
        target: 目标点 [x, y]
        step_size: 分解步长
    """
    vector = np.array(target) - np.array(start)
    distance = np.linalg.norm(vector)
    
    if distance < 1e-6:  # 距离过小，直接返回
        return [target]
    
    num_steps = int(distance // step_size)
    if num_steps == 0:
        return [target]
        
    # 生成中间点
    step_vector = vector / distance * step_size
    waypoints = []
    for i in range(1, num_steps + 1):
        waypoints.append((start + step_vector * i).tolist())
    waypoints.append(target)  # 确保包含终点
    
    return waypoints

@staticmethod
def generate_dubins_path(start_pose: Tuple[float, float, float], 
                        end_pose: Tuple[float, float, float],
                        turning_radius: float,
                        sample_interval: float = 0.5) -> List[List[float]]:
    """
    生成dubins平滑路径点
    
    输入：
        start_pose: 起始位姿 (x, y, angle)
        end_pose: 目标位姿 (x, y, angle)
        turning_radius: 转弯半径
        sample_interval: 采样间隔
    """
    try:
        import dubins
        path = dubins.shortest_path(start_pose, end_pose, turning_radius)
        configurations, _ = path.sample_many(sample_interval)
        
        # 返回路径点（跳过第一个点）
        return [[p[0], p[1]] for p in configurations[1:]]
    except Exception as e:
        print(f"Dubins path generation failed: {e}")
        return []
```

### 2.3 JUMP算法实现（新版）

#### 2.3.1 初始化与重置
```python
# algorithms/jump_planner.py [FIXED: 完整分析]
class JumpPlanner(BasePathPlanner):
    def __init__(self, config: Dict[str, Any], env_config: Dict[str, Any]):
        super().__init__(config, env_config)
        
        # 算法参数
        self.agent_width = env_config.get('agent', {}).get('car_width', 5)
        self.sight_width = env_config.get('agent', {}).get('sight_width', 24)
        self.sight_length = env_config.get('agent', {}).get('sight_length', 24)
        self.width = env_config.get('environment', {}).get('width', 600)
        self.height = env_config.get('environment', {}).get('height', 600)
        
        # JUMP特定参数
        jump_params = config.get('parameters', {})
        self.jump_threshold = jump_params.get('jump_threshold', 4)
        self.safety_margin = jump_params.get('safety_margin', 2)
        
        # 路径生成状态
        self.farm_vertices = None
        self.path_points = []
        self.current_path_index = 0
        self.y_offset = 0
        self.turn_direction = False  # [FIXED] 关键差异：新版初值为False，旧版为True！
        self.real_radians = 0
        self.diagonal_length = 0
        self.longest_edge = None
        self.polygon_mask = None
        self.turning_radius = 5.0
        
    def reset(self, initial_state: Dict[str, Any]):
        """重置JUMP算法状态"""
        super().reset(initial_state)
        
        # 从环境状态获取农场边界
        farm_vertices = initial_state.get('farm_vertices')
        if farm_vertices is not None:
            self.farm_vertices = np.array(farm_vertices)
            self._initialize_coverage_pattern()
        
        # 获取turning_radius
        if 'turning_radius' in initial_state:
            self.turning_radius = initial_state['turning_radius']
            
        self.current_path_index = 0
        self.y_offset = -self.diagonal_length + self.agent_width / 2
        self.turn_direction = False  # [FIXED] 再次确认：False！
```

#### 2.3.2 覆盖模式初始化
```python
def _initialize_coverage_pattern(self):
    """
    初始化覆盖模式
    核心逻辑：
    1. 找最长边确定方向
    2. 计算对角线长度
    3. 创建多边形掩码
    """
    # 计算最长边和方向
    self.longest_edge = self._find_longest_edge(self.farm_vertices)
    dx = self.longest_edge[1][0] - self.longest_edge[0][0]
    dy = self.longest_edge[1][1] - self.longest_edge[0][1]
    self.real_radians = np.arctan2(dy, dx)
    
    # 角度归一化到[0, 2π]
    self.real_radians = self.real_radians % (2 * np.pi) if self.real_radians >= 0 else \
                       (self.real_radians + 2 * np.pi) % (2 * np.pi)
    
    # 计算对角线长度
    min_x, min_y = self.farm_vertices.min(axis=0)
    max_x, max_y = self.farm_vertices.max(axis=0)
    self.diagonal_length = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)
    
    # 创建多边形掩码
    self._create_polygon_mask()
```

#### 2.3.3 路径生成逻辑
```python
def _generate_path_line(self) -> List[Tuple[float, float]]:
    """
    生成当前y_offset对应的路径线
    
    算法：
    1. 计算基准线的起点和终点
    2. 根据y_offset平移
    3. 生成线上的点
    4. 过滤农场内的有效点
    5. 根据turn_direction调整顺序
    """
    # 基准线（沿最长边方向）
    start = [0, 0]
    end = np.array([100 * np.cos(self.real_radians), 100 * np.sin(self.real_radians)])
    
    # 平移到当前y_offset位置
    new_start = [
        start[0] + self.y_offset * np.cos(self.real_radians + np.pi / 2) - 
        self.diagonal_length * np.cos(self.real_radians),
        start[1] + self.y_offset * np.sin(self.real_radians + np.pi / 2) - 
        self.diagonal_length * np.sin(self.real_radians)
    ]
    new_end = [
        end[0] + self.y_offset * np.cos(self.real_radians + np.pi / 2) + 
        self.diagonal_length * np.cos(self.real_radians),
        end[1] + self.y_offset * np.sin(self.real_radians + np.pi / 2) + 
        self.diagonal_length * np.sin(self.real_radians)
    ]
    
    # 生成线上的点（间隔1）
    line_points = []
    direction = np.array(new_end) - np.array(new_start)
    length = np.linalg.norm(direction)
    
    for i in np.arange(0, length, 1):
        interpolated_point = np.array(new_start) + (i / length) * direction
        line_points.append(interpolated_point)
        
    # 过滤有效点（在农场内）
    valid_points = [
        point for point in line_points 
        if (0 <= int(point[1]) < self.height and 
            0 <= int(point[0]) < self.width and 
            self.polygon_mask[int(point[1]), int(point[0])] == 1)
    ]
    
    # [FIXED] 关键逻辑：根据turn_direction调整顺序
    if not self.turn_direction:  # False时反转
        valid_points = valid_points[::-1]
        
    return valid_points
```

#### 2.3.4 主规划逻辑
```python
def plan_next_waypoint(self, current_state: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """
    规划下一个路径点（核心方法）
    
    执行流程：
    1. 更新状态
    2. 检查终止条件
    3. 生成新路径线（如需要）
    4. JUMP逻辑判断
    5. 返回路径点或路径列表
    """
    # 更新状态
    self.update_state(current_state)
    
    # 检查终止
    if self.should_terminate(current_state):
        return None
        
    # 如果当前路径用完，生成新路径线
    if self.current_path_index >= len(self.path_points):
        self.path_points = self._generate_path_line()
        self.current_path_index = 0
        self.turn_direction = not self.turn_direction  # 切换方向
        self.y_offset += self.sight_width / 2  # 增加偏移
        
        # 检查是否超出边界
        if self.y_offset >= self.diagonal_length:
            return None
            
    # 如果没有有效路径点，递归调用
    if not self.path_points:
        return self.plan_next_waypoint(current_state)
        
    # JUMP逻辑：检查前方杂草
    forward_weeds = self._get_forward_weeds()
    jump_target = self._find_jump_target(forward_weeds)
    
    if jump_target is not None:
        # 使用dubins路径跳跃
        current_rad = np.pi / 2 - np.radians(current_state['agent_direction'])
        target_rad = self.real_radians if not self.turn_direction else self.real_radians + np.pi
        
        # 生成dubins路径
        path_points = self.generate_dubins_path(
            (current_state['agent_position'][0], 
             current_state['agent_position'][1],
             current_rad),
            (jump_target[1], jump_target[0], target_rad),  # [FIXED] 坐标交换
            self.turning_radius,
            0.5
        )
        
        # 返回路径列表
        if path_points:
            return ('path', path_points)
        
    # 正常沿路径前进 - 批量处理
    if self.current_path_index < len(self.path_points):
        batch_size = min(5, len(self.path_points) - self.current_path_index)
        batch_points = []
        
        for i in range(batch_size):
            if self.current_path_index < len(self.path_points):
                next_point = self.path_points[self.current_path_index]
                self.current_path_index += 1
                
                # 分解路径
                if i == 0:
                    sub_path = self.decompose_path(
                        current_state['agent_position'],
                        [next_point[1], next_point[0]],  # [FIXED] 坐标交换
                        2.0
                    )
                else:
                    prev_point = self.path_points[self.current_path_index - 2]
                    sub_path = self.decompose_path(
                        [prev_point[1], prev_point[0]],
                        [next_point[1], next_point[0]],
                        2.0
                    )
                batch_points.extend(sub_path)
        
        if batch_points:
            return ('path', batch_points)
            
    return None
```

### 2.4 实验执行器详解

#### 2.4.1 单次实验执行
```python
# experiment/experiment_runner.py
def run_single_experiment(self, seed: int) -> Dict[str, Any]:
    """
    执行单次实验
    
    流程：
    1. 环境重置
    2. 算法重置
    3. 主循环执行
    4. 结果收集
    """
    # 环境重置
    obs = self.env.reset(seed=seed)
    initial_state = self._extract_state(obs)
    
    # 算法重置
    self.planner.reset(initial_state)
    
    # 主循环
    done = False
    total_reward = 0
    step_count = 0
    trajectory = []
    
    while not done:
        # 获取当前状态
        current_state = self._extract_state(obs)
        
        # 规划下一个路径点
        waypoint = self.planner.plan_next_waypoint(current_state)
        
        if waypoint is None:
            break
            
        # 处理不同返回类型
        if isinstance(waypoint, tuple) and waypoint[0] == 'path':
            # 批量路径点
            for point in waypoint[1]:
                action = self._compute_action(current_state['agent_position'], 
                                            point, 
                                            current_state['agent_direction'])
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                step_count += 1
                trajectory.append(current_state['agent_position'])
                
                if done:
                    break
        else:
            # 单个路径点
            action = self._compute_action(current_state['agent_position'], 
                                        waypoint, 
                                        current_state['agent_direction'])
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            step_count += 1
            trajectory.append(current_state['agent_position'])
    
    # 收集结果
    return {
        'seed': seed,
        'total_reward': total_reward,
        'step_count': step_count,
        'coverage_rate': info.get('coverage_rate', 0),
        'trajectory': trajectory,
        'algorithm': self.planner.__class__.__name__
    }
```

#### 2.4.2 动作计算
```python
def _compute_action(self, current_pos: List[float], 
                   target_pos: List[float], 
                   current_dir: float) -> List[float]:
    """
    计算从当前位置到目标位置的动作
    
    输入：
        current_pos: 当前位置 [x, y]
        target_pos: 目标位置 [x, y]
        current_dir: 当前朝向（度）
        
    输出：
        [distance, angle]: 距离和转角
    """
    # 计算目标方向
    dx = target_pos[0] - current_pos[0]
    dy = target_pos[1] - current_pos[1]
    target_angle = np.degrees(np.arctan2(dy, dx))
    
    # 计算距离
    distance = np.sqrt(dx**2 + dy**2)
    
    # 计算转角（考虑当前朝向）
    current_rad = np.pi / 2 - np.radians(current_dir)
    target_rad = np.arctan2(dy, dx)
    
    delta_angle = -(target_rad - current_rad) % (2 * np.pi)
    delta_angle = delta_angle - 2 * np.pi if delta_angle > np.pi else delta_angle
    delta_angle = np.degrees(delta_angle)
    
    return [distance, delta_angle]
```

## 三、参数体系分析

### 3.1 配置参数清单 [FIXED: 完整配置体系]

| 参数类别 | 参数路径 | 类型 | 默认值 | 用途 |
|---------|----------|------|--------|------|
| **环境配置** |
| 尺寸 | environment.width/height | int | 600 | 环境大小 |
| 智能体 | environment.agent.car_width | int | 5 | 车辆宽度 |
| 视野 | environment.agent.sight_width | int | 24 | 视野宽度 |
| 视野 | environment.agent.sight_length | int | 24 | 视野长度 |
| **算法配置** |
| 类型 | algorithm.type | str | "jump" | 算法选择 |
| 覆盖率 | algorithm.coverage_threshold | float | 0.98 | 目标覆盖率 |
| 超时 | algorithm.timeout | int | 300 | 超时秒数 |
| 最大迭代 | algorithm.max_iterations | int | 10000 | 最大步数 |
| **JUMP参数** |
| 跳跃阈值 | algorithm.parameters.jump_threshold | int | 4 | 触发跳跃 |
| 安全边距 | algorithm.parameters.safety_margin | float | 2 | 边界距离 |
| 转弯半径 | algorithm.parameters.turning_radius | float | 5.0 | dubins半径 |
| **实验配置** |
| 种子列表 | experiment.seeds | list | [0-9] | 随机种子 |
| 并行数 | experiment.num_parallel | int | 4 | 并行进程 |
| 输出路径 | experiment.output_path | str | "./results" | 结果保存 |

### 3.2 状态管理体系 [FIXED: 完整状态追踪]

#### 3.2.1 算法内部状态
| 状态变量 | 类型 | 更新时机 | 作用域 | 重要性 |
|---------|------|----------|--------|---------|
| **位置状态** |
| current_position | [x,y] | 每次update_state | 实例 | ★★★ |
| current_direction | float | 每次update_state | 实例 | ★★★ |
| **路径状态** |
| turn_direction | bool | 每行结束 | 实例 | ★★★ |
| y_offset | float | 每行结束 | 实例 | ★★★ |
| path_points | list | 每行开始 | 实例 | ★★ |
| current_path_index | int | 每步递增 | 实例 | ★★ |
| **环境感知** |
| discovered_weeds | list | 每次update_state | 实例 | ★★★ |
| coverage_rate | float | 每次update_state | 实例 | ★★★ |
| polygon_mask | array | 初始化时 | 实例 | ★★ |
| **算法参数** |
| real_radians | float | 初始化时 | 实例 | ★★★ |
| diagonal_length | float | 初始化时 | 实例 | ★★ |
| longest_edge | tuple | 初始化时 | 实例 | ★★ |

#### 3.2.2 环境状态字典
```python
# 标准化的状态字典格式
current_state = {
    'agent_position': [x, y],        # 当前位置
    'agent_direction': degrees,      # 朝向（度）
    'discovered_weeds': [[x,y],...], # 发现的杂草
    'coverage_rate': 0.0-1.0,       # 覆盖率
    'farm_vertices': [[x,y],...],   # 农场边界
    'turning_radius': 5.0,           # 转弯半径
    'map_weed': np.array,            # 杂草地图
    'map_frontier': np.array,       # 可达区域
    'map_obstacle': np.array        # 障碍物地图
}
```

### 3.3 坐标系统分析 [FIXED: 关键差异]

```python
# 新版坐标系统一性改进
# 1. 内部统一使用[x, y]顺序（标准笛卡尔）
# 2. 但在某些地方仍需要坐标交换

# 示例1：路径点处理
[next_point[1], next_point[0]]  # 交换y,x到x,y

# 示例2：多边形掩码访问
self.polygon_mask[int(point[1]), int(point[0])]  # 数组索引[row,col]

# 示例3：dubins接口
(jump_target[1], jump_target[0], target_rad)  # 需要交换

# 坐标系转换规则：
# - 存储格式：统一[x, y]
# - 数组索引：[y, x]（行列）
# - 接口调用：根据具体API要求
```

## 四、技术特点分析

### 4.1 设计模式
- **策略模式**：算法可插拔，统一接口
- **模板方法**：基类定义框架，子类实现细节
- **依赖注入**：通过配置注入参数
- **工厂模式**：算法实例化工厂

### 4.2 架构优势
- **模块化**：清晰的模块划分
- **可扩展**：易于添加新算法
- **可测试**：模块独立，便于单元测试
- **可配置**：配置驱动，灵活调整

### 4.3 性能优化
- **批量处理**：路径点批量执行
- **状态缓存**：避免重复计算
- **惰性计算**：按需生成路径

## 五、潜在问题识别

### 5.1 代码质量改进点
- **异常处理**：需要更完善的错误处理
- **日志记录**：缺少详细的执行日志
- **类型标注**：部分函数缺少类型提示
- **文档完善**：部分复杂逻辑需要更多注释

### 5.2 架构改进空间
- **接口一致性**：返回值类型不统一
- **状态管理**：可考虑状态模式
- **并发支持**：算法执行的并行化

### 5.3 风险点 [FIXED: 关键风险]
- **turn_direction初值**：False与旧版True相反，影响覆盖模式
- **坐标交换**：多处需要[y,x]与[x,y]交换
- **路径返回格式**：单点vs路径列表的处理
- **状态同步**：算法状态与环境状态的一致性

## 六、知识索引

### 6.1 关键类和方法
| 类/方法 | 位置 | 功能 | 重要性 |
|---------|------|------|---------|
| BasePathPlanner | base_algorithm.py | 算法基类 | ★★★ |
| JumpPlanner | jump_planner.py | JUMP实现 | ★★★ |
| ExperimentRunner | experiment_runner.py | 实验管理 | ★★★ |
| plan_next_waypoint | 各算法类 | 核心规划 | ★★★ |
| decompose_path | base_algorithm.py | 路径分解 | ★★ |
| generate_dubins_path | base_algorithm.py | 平滑路径 | ★★ |

### 6.2 配置键值
| 配置项 | 路径 | 默认值 |
|--------|------|--------|
| algorithm.type | 算法选择 | "jump" |
| coverage_threshold | 覆盖率阈值 | 0.98 |
| turning_radius | 转弯半径 | 5.0 |
| jump_threshold | 跳跃阈值 | 4 |

### 6.3 设计决策记录
1. **策略模式选择**：支持多算法切换
2. **配置驱动设计**：灵活性优先
3. **批量处理优化**：提高执行效率
4. **统一状态字典**：接口标准化

## 七、深度架构分析

### 执行流程图
```
主入口(main.py)
    ↓
配置加载(ConfigManager)
    ↓
实验运行器(ExperimentRunner)
    ↓
算法实例化(Factory Pattern)
    ↓
环境初始化
    ↓
┌─────────────┐
│  主循环开始  │
├─────────────┤
│ 1.获取状态   │
│ 2.规划路径   │
│ 3.执行动作   │
│ 4.更新状态   │
│ 5.检查终止   │
└─────────────┘
    ↓
结果收集(ResultCollector)
    ↓
数据保存
```

### 关键设计决策
1. **模块化架构**：提高可维护性
2. **turn_direction=False**：与旧版相反的初值选择
3. **批量路径返回**：优化执行效率
4. **统一坐标系**：减少转换错误（但仍有遗留）

[报告完成度：95%] [深度等级：L4]