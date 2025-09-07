# 基于形状的HIF方向场生成工具

## 概述

本工具从农田形状掩码自动生成高质量的方向场（Human Intention Field），使机器人能够沿着农田边界形状自然地进行覆盖路径规划。

## 核心技术

**算法：Signed Distance Field + Gradient Orthogonalization**
- 使用距离变换计算每个内部像素到边界的最短距离
- 通过梯度正交化得到沿边界流动的切线方向
- 产生类似等高线的自然流动效果

## 使用方法

### 基本用法
```bash
# 激活虚拟环境
source new_venv/bin/activate

# 生成方向场
python utils/generate_shape_based_hif.py /path/to/map/dir

# 示例：处理field_coverage地图
python utils/generate_shape_based_hif.py /home/lzh/NewCppRL/envs_new/maps/field_coverage
```

### 命令行参数
- `map_dir`: 地图目录路径（必需）
- `--smooth N`: 平滑迭代次数（默认: 1）
- `--test`: 测试模式，只处理前3个文件

### 高级用法
```bash
# 增强平滑效果
python utils/generate_shape_based_hif.py /path/to/map/dir --smooth 2

# 测试模式
python utils/generate_shape_based_hif.py /path/to/map/dir --test
```

## 文件组织

### 输入文件
```
map_dir/
└── field/
    ├── field_1.png    # 农田掩码（二值图像）
    ├── field_2.png
    └── ...
```

### 输出文件
```
map_dir/
└── hif/
    ├── human_intent_field_1.npy    # 方向场数据
    ├── human_intent_field_2.npy
    └── image/
        ├── orientation_field_1.png         # HSV可视化
        ├── orientation_field_1_vector.png  # 矢量场可视化
        └── ...
```

## 数据格式

### HIF文件格式
- **类型**: NumPy数组 (.npy)
- **形状**: (H, W) 与对应的field掩码相同
- **数据类型**: float32
- **值域**: 
  - 有效方向: [0, π) 弧度（无向场）
  - 无效区域: -1

### 坐标系定义
- **HIF系统**: 0=西(9点钟方向), π/2=南(6点钟方向)
- **与Agent系统映射**: 180°相位差

## 可视化说明

### HSV颜色编码
- **色相(H)**: angle/π 映射到[0,1]
- **饱和度(S)**: 1.0（最大饱和度）
- **亮度(V)**: 1.0（最大亮度）
- **颜色对应**: 红色(0°) → 黄色(60°) → 绿色(120°) → 青色(180°)

### 矢量场显示
- 箭头方向表示局部方向场
- 箭头颜色使用HSV编码
- 背景显示原始掩码形状

## 技术特点

1. **简洁高效**: 核心算法仅20行代码，总代码量<300行
2. **数学基础扎实**: 基于成熟的计算机图形学技术
3. **完全兼容**: 生成的HIF可直接用于v5环境
4. **可视化丰富**: 提供HSV和矢量场两种可视化方式
5. **批处理支持**: 自动处理整个目录，带进度显示

## 测试验证

运行测试脚本验证生成效果：
```bash
python tests/test_generated_hif.py
```

测试内容：
- HIF文件格式验证
- v5环境加载兼容性
- 角度差计算正确性

## 算法优势

- **自然流畅**: 方向场沿边界自然流动，无突变
- **计算高效**: O(n)复杂度，无需迭代优化
- **结果稳定**: 基于距离场的确定性算法
- **易于调节**: 通过平滑参数控制场的平滑度

## 注意事项

1. 输入的field掩码应为二值图像（非零值为农田区域）
2. 生成的方向场为无向场（最大角度差90°）
3. 中文字体警告不影响功能，可忽略
4. 建议使用--smooth 1或2获得平滑的方向场

## 相关文件

- 生成脚本: `utils/generate_shape_based_hif.py`
- 测试脚本: `tests/test_generated_hif.py`
- v5环境: `envs_new/cpp_env_v5.py`
- HIF系统参考: `/home/lzh/HIFS/orientation_field_ui.py`