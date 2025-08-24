"""对比CPU和GPU APF在v2环境中的帧率"""
import time
import numpy as np

def test_env_fps(use_gpu=False, num_steps=500):
    """测试环境帧率"""
    # 动态导入和替换
    import envs_new.cpp_env_v2 as v2_module
    
    if use_gpu:
        from envs_new.utils.gpu_apf import gpu_apf_bool
        original = v2_module.cpu_apf_bool
        v2_module.cpu_apf_bool = gpu_apf_bool
    
    from envs_new.cpp_env_v2 import CppEnv
    
    env = CppEnv()
    obs, info = env.reset(seed=42)
    
    # 运行测试
    start = time.time()
    for _ in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    
    elapsed = time.time() - start
    fps = num_steps / elapsed
    
    env.close()
    
    # 恢复原函数
    if use_gpu:
        v2_module.cpu_apf_bool = original
    
    return fps, elapsed

print("=" * 50)
print("对比CPU vs GPU APF性能")
print("=" * 50)

# 测试参数
num_steps = 500
num_runs = 3

# CPU测试
print(f"\n测试CPU APF ({num_runs}轮，每轮{num_steps}步)...")
cpu_fps_list = []
for i in range(num_runs):
    fps, elapsed = test_env_fps(use_gpu=False, num_steps=num_steps)
    cpu_fps_list.append(fps)
    print(f"  第{i+1}轮: {fps:.1f} FPS")

cpu_avg_fps = np.mean(cpu_fps_list)
print(f"CPU平均: {cpu_avg_fps:.1f} FPS")

# GPU测试  
print(f"\n测试GPU APF ({num_runs}轮，每轮{num_steps}步)...")
gpu_fps_list = []
for i in range(num_runs):
    fps, elapsed = test_env_fps(use_gpu=True, num_steps=num_steps)
    gpu_fps_list.append(fps)
    print(f"  第{i+1}轮: {fps:.1f} FPS")

gpu_avg_fps = np.mean(gpu_fps_list)
print(f"GPU平均: {gpu_avg_fps:.1f} FPS")

# 结果汇总
print("\n" + "=" * 50)
print("性能对比结果")
print("=" * 50)
print(f"CPU APF: {cpu_avg_fps:.1f} FPS")
print(f"GPU APF: {gpu_avg_fps:.1f} FPS")
print(f"提升: {gpu_avg_fps/cpu_avg_fps:.2f}x ({(gpu_avg_fps-cpu_avg_fps)/cpu_avg_fps*100:.1f}%)")
print(f"每步节省: {1000/cpu_avg_fps - 1000/gpu_avg_fps:.2f}ms")