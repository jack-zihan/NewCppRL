# 奖励计算详细示例对比

本文档通过具体数值示例，展示新旧环境在相同状态变化下的奖励计算过程和结果。

---

## 示例场景设定

### 环境参数
```python
# 共同参数
agent_width = 4.0
agent_length = 6.0
v_max = 3.5
w_max = 28.6

# 归一化因子
normalization = 2 * agent_width * v_max = 2 * 4.0 * 3.5 = 28.0
```

### 状态变化示例
```python
# 时刻t到t+1的状态变化
frontier_area: 1000 → 950  (减少50，表示探索了50单位面积)
frontier_tv: 200 → 190     (减少10，表示前沿边界更平滑)
weed_count: 100 → 95        (减少5，表示清除了5个杂草)
steer: 10.0 → 15.0          (增加5.0，表示转向变化)
crashed: False
finished: False
```

---

## 1. 前沿覆盖奖励计算

### 旧版本计算过程

```python
# 代码位置: envs/cpp_env_base_copy.py:250-251
# 步骤1: 计算前沿面积变化
frontier_area_change = frontier_area_t - frontier_area_tp1
                     = 1000 - 950 
                     = 50

# 步骤2: 归一化
reward_frontier_coverage = frontier_area_change / (2 * agent_width * v_max)
                        = 50 / 28.0
                        = 1.7857

# 步骤3: 应用组系数（注意：coverage本身没有个体系数）
reward_frontier_coverage_final = 0.125 * 1.7857
                               = 0.2232
```

### 新版本计算过程

```python
# 代码位置: envs_new/components/reward/reward_system.py:43-62
# 步骤1: StateVariable计算变化
frontier_info.change() = current - last = 950 - 1000 = -50
frontier_covered = -(-50) = 50

# 步骤2: 归一化并应用个体系数
# 注意：cpp_env_v2设置了reward_frontier_coverage_coef = 0.5
individual_reward = 50 / 28.0 * 0.5  # 个体系数
                  = 1.7857 * 0.5
                  = 0.8929

# 步骤3: 应用组系数
# 代码位置: envs_new/components/reward/reward_system.py:235-241
group_coefficient = 0.125  # reward_frontier_total_coef
final_reward = 0.8929 * 0.125
             = 0.1116
```

### 🔴 差异对比
```
旧版本: 0.2232
新版本: 0.1116
差异: -50%
```

---

## 2. 前沿变化(TV)奖励计算

### 旧版本计算过程

```python
# 代码位置: envs/cpp_env_base_copy.py:252
# 步骤1: 计算TV变化
tv_change = frontier_tv_t - frontier_tv_tp1
          = 200 - 190
          = 10

# 步骤2: 归一化并应用内部系数0.5
reward_frontier_tv = 0.5 * tv_change / v_max
                   = 0.5 * 10 / 3.5
                   = 1.4286

# 步骤3: 应用组系数
reward_frontier_tv_final = 0.125 * 1.4286
                         = 0.1786
```

### 新版本计算过程

```python
# 代码位置: envs_new/components/reward/reward_system.py:65-82
# 步骤1: StateVariable计算变化
variation_reduction = -(190 - 200) = 10

# 步骤2: 归一化并应用个体系数
individual_reward = 10 / 3.5 * 0.5  # coefficient = 0.5
                  = 1.4286

# 步骤3: 应用组系数
final_reward = 1.4286 * 0.125
             = 0.1786
```

### ✅ 差异对比
```
旧版本: 0.1786
新版本: 0.1786
差异: 0% (一致)
```

---

## 3. 前沿总奖励

### 旧版本
```python
# 代码位置: envs/cpp_env_base_copy.py:253-255
reward_frontier = 0.125 * (reward_frontier_coverage + reward_frontier_tv)
                = 0.125 * (1.7857 + 1.4286)
                = 0.125 * 3.2143
                = 0.4018
```

### 新版本
```python
# 各组件独立计算后相加
reward_frontier = frontier_coverage_final + frontier_tv_final
                = 0.1116 + 0.1786
                = 0.2902
```

### 🔴 总差异
```
旧版本: 0.4018
新版本: 0.2902
差异: -27.7%
```

---

## 4. 杂草清除奖励计算

### 旧版本
```python
# 代码位置: envs/cpp_env_base_copy.py:257
weed_removed = weed_num_t - weed_num_tp1
             = 100 - 95
             = 5
reward_weed = 20.0 * 5 = 100.0
```

### 新版本
```python
# 代码位置: envs_new/components/reward/reward_system.py:29-40
weed_info.change() = 95 - 100 = -5
weed_removed = -(-5) = 5
reward_weed = 20.0 * 5 = 100.0
```

### ✅ 差异对比
```
旧版本: 100.0
新版本: 100.0
差异: 0% (一致)
```

---

## 5. 转向相关奖励计算

### 旧版本
```python
# 代码位置: envs/cpp_env_base_copy.py:240-248
# 转向变化惩罚
reward_turn_gap = -0.5 * abs(15.0 - 10.0) / 28.6
                = -0.5 * 5.0 / 28.6
                = -0.0875

# 方向改变惩罚（同向，无惩罚）
reward_turn_direction = -0.30 * 0 = 0

# 转向平滑奖励
normalized_steer = abs(15.0 / 28.6) = 0.5245
reward_turn_self = 0.25 * (0.4 - 0.5245^0.5)
                 = 0.25 * (0.4 - 0.7242)
                 = 0.25 * (-0.3242)
                 = -0.0811

# 总转向奖励（注意：乘以0.0）
reward_turn = 0.0 * (-0.0875 + 0 + (-0.0811))
            = 0.0
```

### 新版本
```python
# 所有转向Calculator的组系数都是0.0
reward_turn = (各项计算) * 0.0 = 0.0
```

### ✅ 差异对比
```
旧版本: 0.0
新版本: 0.0
差异: 0% (一致，都被禁用)
```

---

## 6. 总奖励汇总

### 旧版本总奖励
```python
reward_total = reward_const + reward_frontier + reward_weed + reward_turn
             = -0.1 + 0.4018 + 100.0 + 0.0
             = 100.3018
```

### 新版本总奖励
```python
reward_total = base + frontier_coverage + frontier_tv + weed + turning
             = -0.1 + 0.1116 + 0.1786 + 100.0 + 0.0
             = 100.1902
```

### 🔴 总差异
```
旧版本: 100.3018
新版本: 100.1902
差异: -0.1116 (-0.11%)

虽然总差异看似很小（因为杂草奖励占主导），
但前沿奖励部分的差异高达27.7%，
这会严重影响探索行为！
```

---

## 7. APF奖励计算（仅cpp_env_v2）

### 示例APF场景
```python
# APF地图在位置变化后的值
obs_apf[0][y_t, x_t] = 0.8      # frontier APF at t
obs_apf[0][y_tp1, x_tp1] = 0.9   # frontier APF at t+1
obs_apf[2][y_t, x_t] = 0.3      # obstacle APF at t  
obs_apf[2][y_tp1, x_tp1] = 0.2   # obstacle APF at t+1
obs_apf[3][y_t, x_t] = 0.5      # weed APF at t
obs_apf[3][y_tp1, x_tp1] = 0.7   # weed APF at t+1
```

### 旧版本v2计算
```python
# 代码位置: envs/cpp_env_v2.py:86-100
reward_apf_frontier = 0.0 * (0.9 - 0.8) = 0.0
reward_apf_obstacle = 0.3 * (0.2 - 0.3) = -0.03
reward_apf_obstacle = min(0., -0.03) = -0.03
reward_apf_weed = 5.0 * (0.7 - 0.5) = 1.0
reward_apf = 1.0 * (0.0 + (-0.03) + 1.0) = 0.97
```

### 新版本v2计算
```python
# 代码位置: envs_new/cpp_env_v2.py:23-61
# 计算逻辑完全相同
reward_apf = 0.97
```

### ✅ APF奖励一致

---

## 关键发现总结

### 1. 前沿覆盖奖励差异的根源

**配置层级混淆**：
```
旧版本: 组系数 × (coverage + tv×0.5)
       = 0.125 × (1.0×coverage + 0.5×tv)
       
新版本: (coverage×个体系数×组系数) + (tv×个体系数×组系数)
       = (coverage×0.5×0.125) + (tv×0.5×0.125)
       
问题：cpp_env_v2错误地设置了coverage的个体系数为0.5
```

### 2. 影响分析

即使在杂草奖励占主导的情况下（100 vs 0.4），前沿奖励的27.7%差异仍然会：
1. 影响探索策略的学习
2. 改变局部决策行为
3. 延长训练收敛时间
4. 降低最终性能

### 3. 修复优先级

🔴 **必须立即修复**：将`reward_frontier_coverage_coef`从0.5改为1.0

这种"细微但致命"的差异正是导致训练失败的典型原因！