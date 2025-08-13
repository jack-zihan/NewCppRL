# 🔧 故障排除增强指南 (Enhanced Troubleshooting Guide)

本文档为 envs_new 环境系统提供详细的故障排除指南，帮助开发者快速定位和解决问题。

## 调试理念

遇到问题时，先理解错误的本质，再寻找解决方案。大多数问题都有明确的模式和标准解决方法。

## 常见问题及解决方案

### 1. 导入错误 (Import Errors)

**问题表现**: `ImportError: cannot import name 'cpu_apf_bool'`

**根本原因**: 
C++扩展模块未编译或编译版本不匹配。cpu_apf是用C++实现的高性能APF计算模块。

**解决方案**:
```bash
# 重新编译C++扩展
cd /home/lzh/NewCppRL
python setup.py build_ext --inplace

# 验证编译成功
ls -la *.so  # 应该看到 cpu_apf.cpython-*.so 文件

# 测试导入
python -c "from cpu_apf import cpu_apf_bool; print('Import successful!')"
```

**预防措施**:
- Python版本更新后需要重新编译
- 系统库更新后可能需要重新编译
- 建议在requirements.txt中记录编译依赖

### 2. 观察形状不匹配 (Observation Shape Mismatch)

**问题表现**: Neural network expects different observation shape

**根本原因**: 
新架构使用动态观察空间，实际通道数在reset后才确定。这是两阶段初始化模式的特点。

**解决方案**:
```python
# 检查reset后的实际观察形状
env = CppEnv()
obs, _ = env.reset()
print(f"Observation shape: {obs['observation'].shape}")
# 典型输出：(5, 16, 16) 或 (20, 16, 16) 取决于配置

# 根据实际形状调整网络
if env.config.use_multiscale:
    # 多尺度：通道数 = base_channels × (4 + use_global)
    in_channels = obs['observation'].shape[0]
else:
    # 单尺度：通道数 = 地图类型数
    in_channels = obs['observation'].shape[0]

# 更新网络定义
model = YourNetwork(in_channels=in_channels, ...)
```

### 3. 重置时无限循环 (Infinite Loop During Reset)

**问题表现**: Environment hangs during reset

**这是老版本的严重BUG - 新版本已完全修复！**

**老版本问题代码**:
```python
# 危险：可能永远找不到有效位置！
while True:
    pos = random_position()
    if is_valid(pos):
        break
```

**新版本解决方案**:
```python
# 安全：批量生成，保证终止
valid_positions = np.argwhere(is_valid_map)
if len(valid_positions) > 0:
    selected = np.random.choice(valid_positions, size=count)
else:
    # 优雅处理无有效位置的情况
    raise ValueError("No valid positions available")
```

### 4. 内存泄漏 (Memory Leak)

**问题表现**: Memory usage grows over episodes

**诊断方法**:
```python
import tracemalloc
import gc

# 开始内存追踪
tracemalloc.start()

# 运行多个回合
for i in range(100):
    env = CppEnv()
    obs = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        if done:
            break
    env.close()
    
    # 每10回合检查内存
    if i % 10 == 0:
        current, peak = tracemalloc.get_traced_memory()
        print(f"Episode {i}: Current={current/1024/1024:.1f}MB, Peak={peak/1024/1024:.1f}MB")
        gc.collect()  # 强制垃圾回收
```

**解决方案**:
```python
# 确保正确清理
env.close()  # 释放环境资源

# 批量环境的特殊处理
if hasattr(env, 'envs'):  # 批量环境
    for e in env.envs:
        e.close()

# 强制垃圾回收
import gc
gc.collect()
```

### 5. 奖励计算差异 (Reward Calculation Differences)

**问题表现**: Different rewards between old and new versions

**诊断和解决**:
```python
# 获取详细奖励分解
breakdown = env.get_reward_breakdown()
print("奖励组成：")
for component, value in breakdown.items():
    print(f"  {component}: {value:.4f}")

# 输出示例：
# 奖励组成：
#   total: -0.3500
#   base: -0.1000
#   weed_removal: 0.0000
#   frontier_coverage: 0.2500
#   turning_penalty: -0.5000
#   collision_penalty: 0.0000

# 比较配置系数
old_config = {...}  # 老版本配置
new_config = env.config.get_reward_coefficients()

print("系数对比：")
for key in new_config:
    if key in old_config:
        diff = new_config[key] - old_config[key]
        if abs(diff) > 0.001:
            print(f"  {key}: {old_config[key]} → {new_config[key]} (差异: {diff:+.3f})")
```

## 性能诊断工具

### 性能监控器

```python
class PerformanceMonitor:
    """环境性能监控器"""
    
    def __init__(self, env):
        self.env = env
        self.timings = defaultdict(list)
    
    def profile_step(self, action):
        """分析step操作的各组件耗时"""
        import time
        
        timestamps = {}
        
        # 记录各阶段时间戳
        timestamps['start'] = time.perf_counter()
        
        # 执行step
        obs, reward, done, truncated, info = self.env.step(action)
        
        timestamps['end'] = time.perf_counter()
        
        # 计算耗时
        total_time = timestamps['end'] - timestamps['start']
        
        # 更详细的分析需要修改环境代码添加时间戳
        self.timings['total'].append(total_time * 1000)  # 转换为毫秒
        
        return obs, reward, done, truncated, info
    
    def report(self):
        """生成性能报告"""
        import numpy as np
        
        print("\n性能分析报告:")
        print("-" * 50)
        
        for component, times in self.timings.items():
            times_array = np.array(times)
            print(f"{component}:")
            print(f"  平均: {np.mean(times_array):.2f} ms")
            print(f"  最小: {np.min(times_array):.2f} ms")
            print(f"  最大: {np.max(times_array):.2f} ms")
            print(f"  标准差: {np.std(times_array):.2f} ms")
            print(f"  P95: {np.percentile(times_array, 95):.2f} ms")
```

### 瓶颈定位

```python
import cProfile
import pstats
from pstats import SortKey

def profile_environment(env, num_steps=1000):
    """
    详细性能分析，找出瓶颈
    """
    profiler = cProfile.Profile()
    
    # 开始分析
    profiler.enable()
    
    obs = env.reset()
    for _ in range(num_steps):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            obs = env.reset()
    
    profiler.disable()
    
    # 生成报告
    stats = pstats.Stats(profiler)
    
    print("\n最耗时的20个函数:")
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(20)
    
    print("\n调用次数最多的20个函数:")
    stats.sort_stats(SortKey.CALLS)
    stats.print_stats(20)
    
    return stats
```

## 调试技巧

### 1. 使用断言进行早期错误检测

```python
class DebugWrapper:
    """调试包装器，添加断言检查"""
    
    def __init__(self, env):
        self.env = env
    
    def step(self, action):
        # 动作有效性检查
        assert self.env.action_space.contains(action), f"Invalid action: {action}"
        
        # 执行step
        obs, reward, done, truncated, info = self.env.step(action)
        
        # 观察有效性检查
        assert self.env.observation_space.contains(obs['observation']), \
            f"Invalid observation shape: {obs['observation'].shape}"
        
        # 奖励范围检查
        assert -1000 < reward < 1000, f"Unusual reward: {reward}"
        
        return obs, reward, done, truncated, info
```

### 2. 可视化调试

```python
def visualize_apf_fields(env):
    """可视化APF场，帮助理解奖励计算"""
    import matplotlib.pyplot as plt
    
    if hasattr(env, 'obs_apf') and env.obs_apf is not None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        titles = ['Frontier APF', 'Mist Inv', 'Obstacle APF', 
                 'Weed APF', 'Trajectory APF', 'Combined']
        
        for i, (ax, title) in enumerate(zip(axes.flat, titles)):
            if i < env.obs_apf.shape[0]:
                im = ax.imshow(env.obs_apf[i], cmap='hot')
                ax.set_title(title)
                plt.colorbar(im, ax=ax)
            elif i == 5:  # Combined view
                combined = np.sum(env.obs_apf, axis=0)
                im = ax.imshow(combined, cmap='hot')
                ax.set_title(title)
                plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.show()
```

### 3. 日志记录最佳实践

```python
import logging

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('env_debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('EnvDebug')

class LoggingWrapper:
    """添加详细日志的环境包装器"""
    
    def __init__(self, env):
        self.env = env
        self.episode_count = 0
        self.step_count = 0
    
    def reset(self, **kwargs):
        self.episode_count += 1
        self.step_count = 0
        logger.info(f"Episode {self.episode_count} started")
        
        obs, info = self.env.reset(**kwargs)
        
        logger.debug(f"Reset complete. Initial obs shape: {obs['observation'].shape}")
        logger.debug(f"Initial info: {info}")
        
        return obs, info
    
    def step(self, action):
        self.step_count += 1
        
        logger.debug(f"Step {self.step_count}: action={action}")
        
        obs, reward, done, truncated, info = self.env.step(action)
        
        logger.debug(f"Step {self.step_count} result: reward={reward:.4f}, done={done}")
        
        if done or truncated:
            logger.info(f"Episode {self.episode_count} ended after {self.step_count} steps")
        
        return obs, reward, done, truncated, info
```

## 常见配置错误

### 1. 动作空间配置错误

```python
# 错误：动作离散化数量不匹配
config = {
    'action_nvec': (7, 21),  # 期望7×21=147个动作
    'action_type': 'discrete'
}

# 但神经网络输出了错误的动作维度
model_output = torch.randn(batch_size, 100)  # 错误：应该是147

# 解决：确保网络输出维度正确
num_actions = config['action_nvec'][0] * config['action_nvec'][1]
model = DQN(output_dim=num_actions)
```

### 2. 多尺度观察配置错误

```python
# 错误：忘记考虑多尺度的通道数倍增
config = {
    'use_multiscale': True,
    'n_scales': 4,
    'use_global_features': True
}

# 基础通道数是5（5种地图）
base_channels = 5

# 错误：假设通道数还是5
model = CNN(in_channels=5)  # 错误！

# 正确：计算实际通道数
actual_channels = base_channels * (4 + (1 if config['use_global_features'] else 0))
# actual_channels = 5 * 5 = 25
model = CNN(in_channels=25)  # 正确！
```

## 性能优化建议

### 1. 禁用不必要的功能

```python
# 如果不需要APF奖励，可以禁用APF计算
env = CppEnv(
    use_apf=False,  # 禁用APF计算，节省约30%计算时间
    use_mist=False,  # 如果不需要雾效，也可以禁用
    render_mode=None  # 训练时禁用渲染
)
```

### 2. 使用批量环境

```python
# 单环境训练
single_env = CppEnv()
# 1000 steps耗时: ~8秒

# 批量环境训练（16个并行）
from envs_new.utils.batch_wrapper import BatchEnvWrapper
batch_env = BatchEnvWrapper(CppEnv, num_envs=16)
# 16×1000 steps耗时: ~15秒（而不是128秒）
```

### 3. 缓存配置

```python
# 启用观察缓存（如果机器人经常重复访问相同位置）
env = CppEnv()
env.observation_generator.enable_caching = True
env.observation_generator.cache_size = 1000
```

## 与其他组件的集成

### 与TorchRL集成

```python
from torchrl.envs import GymWrapper, TransformedEnv
from torchrl.envs.transforms import ToTensorImage, Resize

# 包装为TorchRL环境
env = CppEnv(render_mode=None)
torchrl_env = GymWrapper(env)

# 添加转换
torchrl_env = TransformedEnv(
    torchrl_env,
    transforms=[
        ToTensorImage(),  # 转换为张量
        Resize(64, 64)    # 调整大小
    ]
)
```

### 与Stable-Baselines3集成

```python
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

# 创建环境
env = CppEnv(action_type='continuous')

# 检查兼容性
check_env(env)

# 训练模型
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

## 总结

envs_new 系统的模块化设计使得调试和优化变得更加容易。通过理解各组件的职责和交互方式，大多数问题都可以快速定位和解决。记住以下关键点：

1. **C++扩展需要编译** - 遇到导入错误先检查编译
2. **观察空间是动态的** - reset后才能确定实际形状  
3. **使用调试工具** - 性能监控器、日志记录、可视化
4. **批量处理提升性能** - 并行环境可大幅加速训练
5. **配置要匹配** - 网络架构必须与环境配置一致

如果遇到本文档未涵盖的问题，建议：
1. 查看单元测试了解正确用法
2. 使用调试包装器定位问题
3. 查看源代码理解实现细节
4. 在GitHub Issues中寻找或提出问题

---

*故障排除指南版本: 1.0*
*适用系统: envs_new (重构版)*
*最后更新: 基于当前代码库分析*