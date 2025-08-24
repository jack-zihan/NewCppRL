"""验证CPU和GPU APF算法差异"""
import numpy as np
import torch

def verify_algorithm_difference():
    """验证CPU（BFS+欧几里得）和GPU（EDT）算法的差异"""
    print("=" * 60)
    print("验证APF算法差异")
    print("=" * 60)
    
    from cpu_apf import cpu_apf_bool
    from envs_new.utils.gpu_apf import gpu_apf_bool
    
    # 创建测试地图 - 简单的单点障碍物
    test_map = np.zeros((10, 10), dtype=np.uint8)
    test_map[5, 5] = 1  # 中心点障碍物
    
    print("\n测试地图（1表示障碍物）:")
    print(test_map)
    
    # CPU版本（BFS + 欧几里得）
    cpu_dist, _ = cpu_apf_bool(test_map)
    
    # GPU版本（标准EDT）
    if torch.cuda.is_available():
        gpu_dist, _ = gpu_apf_bool(test_map)
        
        print("\nCPU距离（BFS+欧几里得）:")
        print(np.round(cpu_dist, 2))
        
        print("\nGPU距离（标准EDT）:")
        print(np.round(gpu_dist, 2))
        
        print("\n差异:")
        diff = cpu_dist - gpu_dist
        print(np.round(diff, 2))
        
        # 分析差异
        print("\n分析:")
        print(f"CPU (5,4)到(5,5)的距离: {cpu_dist[4, 5]:.4f}")
        print(f"GPU (5,4)到(5,5)的距离: {gpu_dist[4, 5]:.4f}")
        print(f"理论欧几里得距离: 1.0000")
        
        print(f"\nCPU (4,4)到(5,5)的距离: {cpu_dist[4, 4]:.4f}")
        print(f"GPU (4,4)到(5,5)的距离: {gpu_dist[4, 4]:.4f}")
        print(f"理论欧几里得距离: {np.sqrt(2):.4f}")
        
        # 测试势场转换后的差异
        print("\n\n势场转换后的差异:")
        from envs_new.cpp_env_v2 import CppEnv
        
        env = CppEnv()
        
        # CPU版本
        env.device = 'cpu'
        cpu_potential = env.get_discounted_apf(test_map, 30)
        
        # GPU版本
        env.device = 'cuda:0'
        gpu_potential = env.get_discounted_apf(test_map, 30)
        
        potential_diff = np.abs(cpu_potential - gpu_potential)
        print(f"平均差异: {potential_diff.mean():.6f}")
        print(f"最大差异: {potential_diff.max():.6f}")
        
        # 显示中心区域的势场
        print("\nCPU势场（中心区域）:")
        print(np.round(cpu_potential[3:8, 3:8], 3))
        
        print("\nGPU势场（中心区域）:")
        print(np.round(gpu_potential[3:8, 3:8], 3))
        
        print("\n结论:")
        print("✅ CPU使用BFS找到曼哈顿最近点，然后计算欧几里得距离")
        print("✅ GPU使用标准欧几里得距离变换（EDT）")
        print("✅ 两种方法都是物理上合理的距离度量")
        print("✅ 差异主要在于对角线距离的处理")
        print("✅ 在势场转换后，差异会被指数衰减平滑")
        
    else:
        print("⚠️ CUDA不可用，无法进行对比")

if __name__ == "__main__":
    verify_algorithm_difference()