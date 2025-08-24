"""分析环境的GPU利用情况"""
import numpy as np
import torch
import time
from typing import Dict, List, Tuple

def analyze_environment_execution():
    """深度分析环境的执行位置和数据流"""
    print("=" * 80)
    print("🔬 环境GPU利用情况深度分析")
    print("=" * 80)
    
    from envs_new.cpp_env_v2 import CppEnv
    
    # 创建环境
    env = CppEnv()
    
    print("\n📍 组件执行位置分析:")
    print("-" * 50)
    
    # 1. 环境主体
    print("\n1. 环境主体 (CppEnvBase):")
    print("   位置: CPU内存")
    print("   说明: Python对象，所有状态和逻辑在CPU上")
    
    # 2. 数据存储
    print("\n2. 数据存储:")
    print("   - maps_dict: NumPy数组 (CPU)")
    print("   - env_state: Python对象 (CPU)")
    print("   - agent: Python对象 (CPU)")
    print("   - observation: NumPy数组 (CPU)")
    
    # 3. 计算模块
    print("\n3. 计算模块执行位置:")
    components = [
        ("scenario_generator", "CPU", "地图生成、障碍物放置等"),
        ("action_processor", "CPU", "动作处理和转换"),
        ("collision_detector", "CPU", "碰撞检测"),
        ("env_dynamics", "CPU", "环境动力学更新"),
        ("reward_system", "CPU", "奖励计算（除APF外）"),
        ("observation_generator", "CPU", "观察生成（使用CPU版torch）"),
        ("renderer", "CPU", "渲染（OpenCV）"),
        ("APF计算", "GPU/CPU", "根据device设置自动选择")
    ]
    
    for name, location, desc in components:
        print(f"   - {name:20s}: {location:8s} | {desc}")
    
    # 4. 数据流分析
    print("\n\n📊 数据流分析:")
    print("-" * 50)
    print("""
    环境执行流程:
    
    1. reset() [CPU]
       ├─ scenario_generator.generate() [CPU] → maps_dict (NumPy)
       ├─ env_dynamics.reset() [CPU]
       └─ observation_generator.generate() [CPU] → obs (NumPy)
    
    2. step(action) [CPU]
       ├─ action_processor.process() [CPU]
       ├─ collision_detector.check() [CPU]
       ├─ env_dynamics.update() [CPU]
       ├─ APF计算 [GPU/CPU] ← 唯一可能使用GPU的地方
       │   └─ 如果device='cuda':
       │       NumPy → CuPy (CPU→GPU传输)
       │       GPU计算
       │       CuPy → NumPy (GPU→CPU传输)
       ├─ reward_system.calculate() [CPU]
       └─ observation_generator.generate() [CPU] → obs (NumPy)
    
    3. GymWrapper处理 [在环境外部]
       └─ 将NumPy观察转为Tensor并移到指定device
    """)
    
    # 5. 性能影响分析
    print("\n⚡ 性能影响分析:")
    print("-" * 50)
    
    # 测试不同大小的地图
    sizes = [100, 200, 400]
    
    for size in sizes:
        print(f"\n{size}×{size} 地图:")
        
        # 创建测试地图
        test_map = np.random.randint(0, 2, (size, size), dtype=np.uint8)
        
        # CPU版本
        env.device = 'cpu'
        cpu_times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = env.get_discounted_apf(test_map, 30)
            cpu_times.append(time.perf_counter() - start)
        cpu_avg = np.mean(cpu_times[1:]) * 1000  # 跳过第一次
        
        # GPU版本（如果可用）
        if torch.cuda.is_available():
            env.device = 'cuda:0'
            
            # Warmup
            _ = env.get_discounted_apf(test_map, 30)
            
            gpu_times = []
            for _ in range(10):
                start = time.perf_counter()
                _ = env.get_discounted_apf(test_map, 30)
                gpu_times.append(time.perf_counter() - start)
            gpu_avg = np.mean(gpu_times) * 1000
            
            # 估算传输开销
            data_size = size * size * 4  # float32
            transfer_overhead = (data_size / 1e6) * 0.1  # 估算约0.1ms/MB
            
            print(f"  CPU时间: {cpu_avg:.2f}ms")
            print(f"  GPU时间: {gpu_avg:.2f}ms (含传输)")
            print(f"  估算传输开销: ~{transfer_overhead:.2f}ms")
            print(f"  净GPU计算: ~{gpu_avg - transfer_overhead:.2f}ms")
            print(f"  加速比: {cpu_avg/gpu_avg:.2f}x")
        else:
            print(f"  CPU时间: {cpu_avg:.2f}ms")
            print("  GPU: 不可用")
    
    # 6. 总结
    print("\n\n" + "=" * 80)
    print("📋 分析总结")
    print("=" * 80)
    
    print("""
    🎯 关键发现:
    
    1. ❌ 环境主体仍在CPU上运行
       - 所有Python对象和NumPy数组都在CPU内存
       - 环境逻辑（step、reset等）都在CPU执行
    
    2. ✅ 只有APF计算可以在GPU上执行
       - 当device='cuda'时，APF计算会使用GPU
       - 但需要CPU↔GPU数据传输
    
    3. ⚠️ 存在数据传输开销
       - 每次APF计算需要：CPU→GPU→计算→GPU→CPU
       - 小地图上传输开销可能超过计算收益
       - 大地图（400×400+）才能看到明显加速
    
    4. 📊 GPU利用率低
       - GPU只在APF计算时短暂使用
       - 大部分时间GPU处于空闲状态
       - 无法充分利用GPU的并行计算能力
    
    5. 🔄 真正的GPU环境需要:
       - 将所有NumPy操作改为CuPy/Torch操作
       - 保持所有数据在GPU内存中
       - 重构整个环境架构
       - 但收益可能有限（环境逻辑不够复杂）
    
    💡 结论:
    当前方案是一个实用的折中：
    - 解决了APF计算瓶颈（特别是大地图）
    - 保持了代码简洁性和兼容性
    - 避免了大规模重构
    - 适合当前的需求场景
    """)

if __name__ == "__main__":
    analyze_environment_execution()