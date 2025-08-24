"""检查GPU和CuPy设置"""
import sys

# 检查CUDA
try:
    import torch
    if torch.cuda.is_available():
        print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA版本: {torch.version.cuda}")
    else:
        print("❌ CUDA不可用")
except ImportError:
    print("⚠️ PyTorch未安装，无法检查CUDA")

# 检查CuPy
try:
    import cupy as cp
    print(f"✅ CuPy已安装: 版本 {cp.__version__}")
    
    # 测试基本功能
    test_array = cp.array([1, 2, 3])
    result = cp.sum(test_array)
    print(f"   CuPy测试: {result.get()} (预期6)")
    
except ImportError:
    print("❌ CuPy未安装")
    print("   请运行: pip install cupy-cuda12x")
    sys.exit(1)

# 检查gpu_apf模块
try:
    from envs_new.utils.gpu_apf import gpu_apf_bool
    print("✅ gpu_apf模块可以导入")
    
    # 简单测试
    import numpy as np
    test_map = np.ones((10, 10), dtype=np.uint8)
    result, is_empty = gpu_apf_bool(test_map)
    print(f"   gpu_apf测试: 输出形状 {result.shape}")
    
except ImportError as e:
    print(f"❌ 无法导入gpu_apf: {e}")
    sys.exit(1)

print("\n✨ 所有检查通过！可以运行测试。")