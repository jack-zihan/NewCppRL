# V4/V5 环境切换测试指南

## 直接运行测试

在PyCharm中，您可以直接运行 `envs/cpp_env_v4.py` 文件来测试环境。

## 切换测试环境

### 测试 Pasture-v4（默认）
```python
# 测试 Pasture-v4 (无多尺度)
env = CppEnvV4(
    render_mode='rgb_array' if if_render else None,
    state_pixels=False,
)

# 测试 Pasture-v5 (有多尺度) - 需要时取消注释
# env = CppEnvV5(
#     render_mode='rgb_array' if if_render else None,
#     state_pixels=False,
# )
```

### 测试 Pasture-v5
将代码修改为：
```python
# 测试 Pasture-v4 (无多尺度)
# env = CppEnvV4(
#     render_mode='rgb_array' if if_render else None,
#     state_pixels=False,
# )

# 测试 Pasture-v5 (有多尺度) - 需要时取消注释
env = CppEnvV5(
    render_mode='rgb_array' if if_render else None,
    state_pixels=False,
)
```

## 运行输出示例

### V4 运行输出
- 观测形状: (4, 128, 128) - 直接像素观测
- 覆盖率逐步增加
- 碰撞或覆盖完成时结束

### V5 运行输出  
- 观测形状: (20, 16, 16) - 多尺度SGCNN特征
- 其他行为与V4相同

## 快捷键（PyCharm）
- **运行**: Shift+F10 或 右键 → Run
- **切换注释**: Ctrl+/ (Windows/Linux) 或 Cmd+/ (Mac)