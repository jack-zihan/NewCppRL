# 坐标系统文档

## 概述

Rules_new系统使用统一的坐标系统来确保各组件之间的一致性。所有坐标操作都通过`CoordinateSystem`类进行管理，该类作为坐标转换的单一真实来源（Single Source of Truth）。

## 坐标约定

### 内部表示
- **格式**: `(y, x)` 元组
- **原因**: 与原始rules系统保持一致，确保算法行为相同
- **数据类型**: `Tuple[float, float]`

### 环境接口
- **输入格式**: 环境通常提供 `[x, y]` 列表格式
- **输出格式**: 算法内部使用 `(y, x)` 元组格式
- **转换点**: 在环境边界进行转换

## 核心组件

### CoordinateSystem类
位置：`rules_new/core/coordinate_system.py`

主要功能：
- `normalize(pos)`: 将任意格式坐标转换为标准 `(y, x)` 元组
- `distance(pos1, pos2)`: 计算两点间欧氏距离
- `angle(pos1, pos2)`: 计算两点间角度
- `rotate(pos, angle, origin)`: 旋转坐标点
- `grid_to_continuous(grid_pos, cell_size)`: 网格坐标转连续坐标
- `continuous_to_grid(cont_pos, cell_size)`: 连续坐标转网格坐标

### 使用示例

```python
from rules_new.core import CoordinateSystem as CS

# 标准化坐标
pos = CS.normalize([100, 200])  # 输入: [x=100, y=200]
# 返回: (200, 100)  # 输出: (y=200, x=100)

# 计算距离
dist = CS.distance((0, 0), (3, 4))  # 返回: 5.0

# 批量转换
positions = [[1, 2], [3, 4], [5, 6]]
normalized = [CS.normalize(p) for p in positions]
# 返回: [(2, 1), (4, 3), (6, 5)]
```

## 算法集成

### BasePathPlanner
所有路径规划算法都继承自`BasePathPlanner`，它自动处理坐标转换：

```python
def reset(self, initial_state):
    # 自动转换agent位置
    raw_position = initial_state.get('agent_position', [0, 0])
    self.current_position = CS.normalize(raw_position)  # (y, x)
    
    # 自动转换杂草位置
    raw_weeds = initial_state.get('discovered_weeds', [])
    self.discovered_weeds = [CS.normalize(w) for w in raw_weeds]
```

### 算法实现
各算法在实现时应遵循以下原则：

1. **输入处理**: 接收环境数据时使用`CS.normalize()`
2. **内部计算**: 使用标准化的 `(y, x)` 格式
3. **输出返回**: 根据接收方需求决定格式

## 关键转换点

### 1. ExperimentRunner
位置：`rules_new/experiment/experiment_runner.py`

关键代码（第283行）：
```python
# 正确: 提取为[y, x]格式
agent_position = [float(env.agent.y), float(env.agent.x)]
```

### 2. 农场顶点处理
各算法中的农场顶点坐标转换：
```python
# 环境提供 [x, y] 格式
farm_vertices = initial_state.get('farm_vertices')

# 转换为 [y, x] 格式
if farm_vertices.ndim == 2:
    self.farm_vertices = np.array([CS.normalize([v[1], v[0]]) for v in farm_vertices])
```

### 3. Dubins路径生成
```python
# generate_dubins_path期望 (x, y, angle) 格式
# 但内部坐标是 (y, x) 格式
path = self.generate_dubins_path(
    (current_pos[1], current_pos[0], angle),  # 转换为 (x, y, angle)
    (target_pos[1], target_pos[0], target_angle),
    turning_radius
)
```

## 常见错误与解决方案

### 错误1：坐标格式不一致
**症状**: 算法行为与预期不符，路径点位置错误
**原因**: 直接使用环境坐标而未转换
**解决**: 始终使用`CS.normalize()`进行转换

### 错误2：距离计算错误
**症状**: 距离值异常，路径规划失败
**原因**: 混用不同格式的坐标
**解决**: 确保所有坐标都经过标准化

### 错误3：角度计算错误
**症状**: 转向角度不正确
**原因**: x/y顺序错误导致atan2计算错误
**解决**: 使用`CS.angle()`方法

## 性能优化

### 向量化操作
对于批量坐标处理，使用NumPy向量化操作：
```python
# 批量标准化
positions = np.array([[x1, y1], [x2, y2], ...])
normalized = positions[:, ::-1]  # 交换列顺序

# 批量距离计算
dists = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
```

### 缓存策略
`CoordinateSystem`类内置LRU缓存：
- 常用坐标转换结果被缓存
- 网格/连续坐标转换使用缓存
- 缓存大小：1024个条目

## 测试验证

### 单元测试
位置：`tests/test_coordinate_system.py`

测试覆盖：
- 坐标格式转换正确性
- 距离计算准确性
- 角度计算准确性
- 批量操作性能
- 边界条件处理

### 集成测试
位置：`tests/test_behavioral_consistency.py`

验证：
- 新旧系统行为一致性
- 坐标转换不影响算法结果
- 性能指标相同

## 迁移指南

### 从旧系统迁移
1. 识别所有坐标操作点
2. 替换直接数组操作为`CS`方法
3. 验证转换正确性
4. 运行一致性测试

### 添加新算法
1. 继承`BasePathPlanner`
2. 使用`CS.normalize()`处理输入
3. 内部使用`(y, x)`格式
4. 遵循现有算法模式

## 版本历史

- **v2.0.0** (2024-01): 引入统一坐标系统
- **v1.0.0**: 原始实现，坐标处理分散

## 相关文档

- [异常处理系统](./EXCEPTION_HANDLING.md)
- [性能监控系统](./PERFORMANCE_MONITORING.md)
- [状态验证系统](./STATE_VALIDATION.md)