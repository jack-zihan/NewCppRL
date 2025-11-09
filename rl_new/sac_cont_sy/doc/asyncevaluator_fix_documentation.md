# AsyncEvaluator 多进程问题修复文档

## 问题描述

使用 `MultiAsyncDataCollector` 后，异步评估时报错：
```
跳过失败的评估: step=100000, error=daemonic processes are not allowed to have children
```

## 问题根因

1. **AsyncEvaluator** 使用 `mp.Pool` 创建进程池
2. **mp.Pool 的 worker 进程默认是 daemon=True**（守护进程）
3. **evaluate_policy_standalone** 在 daemon worker 中执行
4. 该函数会创建 **ParallelEnv**，而 ParallelEnv 需要创建子进程
5. **Python 不允许 daemon 进程创建子进程** → 错误发生

## 解决方案：使用线程池替代进程池

### 修改内容

将 `AsyncEvaluator` 从使用 `multiprocessing.Pool` 改为使用 `concurrent.futures.ThreadPoolExecutor`。

### 主要改动

1. **导入变更**：
```python
# 旧代码
import multiprocessing as mp

# 新代码
from concurrent.futures import ThreadPoolExecutor, Future
```

2. **初始化变更**：
```python
# 旧代码
self.pool = mp.Pool(processes=max_workers, maxtasksperchild=1)

# 新代码
self.executor = ThreadPoolExecutor(max_workers=max_workers)
```

3. **任务提交变更**：
```python
# 旧代码
async_result = self.pool.apply_async(eval_func, args=(model_path, cfg, step, position))

# 新代码
future = self.executor.submit(eval_func, model_path, cfg, step, position)
```

4. **结果检查变更**：
```python
# 旧代码
if async_result.ready():
    result = async_result.get(timeout=0)

# 新代码
if future.done():
    result = future.result(timeout=0)
```

5. **关闭变更**：
```python
# 旧代码
self.pool.close()
self.pool.join()

# 新代码
self.executor.shutdown(wait=wait)
```

## 为什么这个方案有效

1. **线程可以创建子进程**：线程不受 daemon 进程的限制
2. **性能影响最小**：
   - 评估主要是 I/O 密集型（等待环境 step）
   - 真正的并行计算发生在 ParallelEnv 的子进程中
   - GIL 对这种场景影响很小
3. **代码改动最小**：只需修改 5-10 行代码
4. **接口保持兼容**：外部调用方式完全不变

## 性能分析

### CPU 评估场景
- 即使模型在 CPU 上推理，性能影响仍然很小
- 原因：通常只有一个评估任务在运行（不会并发多个）
- 主要的并行发生在 ParallelEnv 层（多个环境并行）

### 对训练的影响
- **基本无影响**：训练使用 GPU，评估使用 CPU，资源不冲突
- **进程隔离**：评估在独立进程中（ParallelEnv 的子进程）
- **异步执行**：评估不阻塞训练循环

## 使用建议

1. **保持现有配置**：`max_workers=2` 足够应对大多数场景
2. **内存管理**：继续在 `evaluate_policy_standalone` 中使用 `torch.cuda.empty_cache()`
3. **未来扩展**：如果需要真正的并发评估多个模型，可以考虑实现非 daemon 进程池

## 测试验证

创建了测试脚本 `tests/test_async_evaluator_thread.py`，验证：
- ✅ 可以成功创建 ParallelEnv
- ✅ 没有 daemon 进程限制错误
- ✅ 异步评估功能正常

## 总结

这是一个简洁、有效的解决方案，以最小的代码改动彻底解决了多进程兼容性问题。