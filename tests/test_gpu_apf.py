"""测试GPU APF性能"""
import time
import numpy as np
from cpu_apf import cpu_apf_bool
from envs_new.utils.gpu_apf import gpu_apf_bool

# 400x400地图
size = 400
iterations = 100

# 创建测试地图
np.random.seed(42)
binary_map = (np.random.random((size, size)) > 0.8).astype(np.uint8)

print(f"测试 {size}x{size} 地图，{iterations} 次迭代")

# 预热GPU
print("预热GPU...")
for _ in range(10):
    gpu_apf_bool(binary_map)

# CPU测试
print("测试CPU APF...")
start = time.time()
for _ in range(iterations):
    cpu_result, _ = cpu_apf_bool(binary_map)
cpu_time = time.time() - start

# GPU测试
print("测试GPU APF...")
start = time.time()
for _ in range(iterations):
    gpu_result, _ = gpu_apf_bool(binary_map)
gpu_time = time.time() - start

# 结果
print(f"\n结果:")
print(f"CPU: {cpu_time:.3f}s ({iterations/cpu_time:.1f} FPS)")
print(f"GPU: {gpu_time:.3f}s ({iterations/gpu_time:.1f} FPS)")
print(f"加速: {cpu_time/gpu_time:.2f}x")

# 验证结果一致性
max_diff = np.max(np.abs(cpu_result - gpu_result))
print(f"\n结果差异: {max_diff:.2e}")
if max_diff < 1e-5:
    print("✅ 结果一致性验证通过")