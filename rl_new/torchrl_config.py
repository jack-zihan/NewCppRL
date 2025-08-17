"""
TorchRL 0.9.2配置
"""
import tempfile
import os

def get_replay_buffer_dir():
    """获取回放缓冲区的临时目录"""
    return tempfile.mkdtemp(prefix="torchrl_replay_")

# 建议使用的配置
USE_SYNC_COLLECTOR = True  # 使用同步收集器避免多进程问题
USE_TEMP_DIR = True  # 使用独立的临时目录避免文件冲突
