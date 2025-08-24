# GPU APF测试指南

## 文件说明

### 核心实现
- `envs_new/utils/gpu_apf.py` - GPU APF实现（30行代码）

### 测试脚本
- `check_gpu_setup.py` - 检查环境设置
- `test_gpu_apf.py` - APF算法性能对比
- `compare_apf_fps.py` - 环境帧率对比（推荐）
- `test_v2_cpu_fps.py` - 单独测试CPU版本
- `test_v2_gpu_fps.py` - 单独测试GPU版本

## 使用步骤

### 1. 安装CuPy
```bash
# 根据你的CUDA版本选择
pip install cupy-cuda12x  # CUDA 12.x
# 或
pip install cupy-cuda11x  # CUDA 11.x
```

### 2. 检查环境
```bash
cd /home/lzh/NewCppRL
python tests/check_gpu_setup.py
```

### 3. 运行测试

#### 快速对比测试（推荐）
```bash
python tests/compare_apf_fps.py
```

#### APF算法性能测试
```bash
python tests/test_gpu_apf.py
```

#### 分别测试CPU和GPU版本
```bash
python tests/test_v2_cpu_fps.py
python tests/test_v2_gpu_fps.py
```

## 永久启用GPU APF

如果测试效果良好，可以永久修改`envs_new/cpp_env_v2.py`：

```python
# 原代码
from cpu_apf import cpu_apf_bool

# 改为
from envs_new.utils.gpu_apf import gpu_apf_bool as cpu_apf_bool
```

## 预期结果

- **400×400地图**: 5-7倍APF计算加速
- **环境帧率**: 提升30-50%
- **每步节省**: 15-20ms

## 注意事项

- 第一次运行GPU版本会有初始化开销
- 建议运行多轮取平均值
- GPU内存占用很小（<100MB）