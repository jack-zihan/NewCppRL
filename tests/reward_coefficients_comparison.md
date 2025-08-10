# 奖励系数对比分析

## 原版 vs 新版奖励系数对比

### 原版 (cpp_env_base_copy.py) 奖励系数

```python
# 原版硬编码的奖励系数
reward_const = -0.1
reward_turn = 0.0 * (...)  # 总权重为0
  - reward_turn_gap = -0.5 * abs(steer_tp1 - self.steer_t) / self.w_range.max
  - reward_turn_direction = -0.30 * (...)
  - reward_turn_self = 0.25 * (0.4 - abs(steer_tp1 / self.w_range.max) ** 0.5)

reward_frontier = 0.125 * (...)
  - reward_frontier_coverage = (self.frontier_area_t - frontier_area_tp1) / (2 * MowerAgent.width * self.v_range.max)
  - reward_frontier_tv = 0.5 * (self.frontier_tv_t - frontier_tv_tp1) / self.v_range.max

reward_weed = 20.0 * (self.weed_num_t - weed_num_tp1)

# Step函数中的额外奖励
collision_penalty = -399.0  # if crashed
completion_bonus = 500.0    # if finished
```

### 新版 (RewardConfig) 默认奖励系数

```python
coefficients = {
    'base_penalty': -0.1,                    # ✓ 匹配
    'weed_removal_coef': 20.0,              # ✓ 匹配
    'frontier_total_coef': 0.125,           # ✓ 匹配
    'frontier_coverage_coef': 1.0,          # ✓ 匹配 (内部使用)
    'frontier_tv_coef': 0.5,                # ✓ 匹配
    'turn_total_coef': 0.0,                 # ✓ 匹配
    'turn_gap_coef': -0.5,                  # ✓ 匹配
    'turn_direction_coef': -0.30,           # ✓ 匹配
    'turn_self_coef': 0.25,                 # ✓ 匹配
    'collision_penalty': -399.0,            # ✓ 匹配
    'completion_bonus': 500.0               # ✓ 匹配
}
```

## 系数一致性验证结果

**✅ 完全一致** - 所有奖励系数都与原版完全匹配。

### 详细对比：

1. **基础惩罚**: `-0.1` ✓
2. **杂草移除奖励**: `20.0` ✓
3. **边界总权重**: `0.125` ✓
4. **边界覆盖系数**: `1.0` (新版内部使用) ✓
5. **边界变化系数**: `0.5` ✓
6. **转向总权重**: `0.0` ✓
7. **转向变化惩罚**: `-0.5` ✓
8. **方向改变惩罚**: `-0.30` ✓
9. **转向平滑奖励**: `0.25` ✓
10. **碰撞惩罚**: `-399.0` ✓
11. **完成奖励**: `500.0` ✓

所有系数都与原版实现完全一致。