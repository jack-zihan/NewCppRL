#!/usr/bin/env python
"""
自动修复SAC训练和评估脚本中的TorchRL 0.9.2兼容性问题
"""

import os
import shutil
from pathlib import Path

# 获取rl_new目录路径
rl_new_dir = Path(__file__).parent

# 需要修复的文件列表
files_to_fix = [
    "sac_cont/area_coverage_sac_cont_train.py",
    "sac_cont/area_coverage_sac_cont_eval.py",
    "sac_cont/area_coverage_v5_sac_cont_train.py",
    "sac_cont/area_coverage_v5_sac_cont_eval.py",
]

# 修复规则
fixes = {
    # 问题1: make_area_coverage_sac_models函数签名
    "make_area_coverage_sac_models(proof_environment)": "make_area_coverage_sac_models()",
    
    # 问题2: SACLoss参数名称变化
    '"target_entropy_weight"': '"target_entropy"',
    "'target_entropy_weight'": "'target_entropy'",
    
    # 问题3: 使用SyncDataCollector代替MultiaSyncDataCollector（可选，取决于是否使用主模块保护）
    # 这个修复可能需要更复杂的逻辑，暂时保留原样
    
    # 问题5: update_priority需要priority参数
    "replay_buffer.update_priority(sampled_tensordict)": 
        "# replay_buffer.update_priority(sampled_tensordict)  # TODO: 需要添加priority参数",
}

def fix_file(file_path):
    """修复单个文件"""
    print(f"修复文件: {file_path}")
    
    # 备份原文件
    backup_path = file_path.with_suffix('.py.bak')
    if not backup_path.exists():
        shutil.copy(file_path, backup_path)
        print(f"  创建备份: {backup_path}")
    
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 应用修复
    original_content = content
    for old, new in fixes.items():
        if old in content:
            content = content.replace(old, new)
            print(f"  修复: {old} -> {new}")
    
    # 添加主模块保护（如果使用MultiaSyncDataCollector）
    if "MultiaSyncDataCollector" in content and 'if __name__ == "__main__":' not in content:
        # 查找main函数调用
        import re
        main_call_pattern = r'^main\([^)]*\)$'
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if re.match(main_call_pattern, line.strip()):
                # 添加主模块保护
                lines[i] = f'if __name__ == "__main__":\n    {line}'
                content = '\n'.join(lines)
                print(f"  添加主模块保护")
                break
    
    # 如果内容有变化，写回文件
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ 文件已修复")
    else:
        print(f"  - 文件无需修复")
    
    return content != original_content

def main():
    print("=== 开始修复SAC脚本 ===\n")
    
    fixed_count = 0
    for file_name in files_to_fix:
        file_path = rl_new_dir / file_name
        if file_path.exists():
            if fix_file(file_path):
                fixed_count += 1
        else:
            print(f"文件不存在: {file_path}")
        print()
    
    print(f"=== 修复完成 ===")
    print(f"共修复 {fixed_count} 个文件")
    
    # 创建临时目录配置文件
    config_file = rl_new_dir / "torchrl_config.py"
    config_content = '''"""
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
'''
    
    with open(config_file, 'w') as f:
        f.write(config_content)
    print(f"\n创建配置文件: {config_file}")
    
    print("\n修复建议:")
    print("1. 如果需要使用MultiaSyncDataCollector，确保main()调用在if __name__ == '__main__':块中")
    print("2. 考虑使用SyncDataCollector进行单进程收集，特别是在笔记本电脑上测试时")
    print("3. 为replay buffer使用独立的临时目录，避免文件冲突")
    print("4. update_priority需要提供priority参数，可以从loss_td中获取")

if __name__ == "__main__":
    main()