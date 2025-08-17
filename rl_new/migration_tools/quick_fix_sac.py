#!/usr/bin/env python
"""
快速修复SAC脚本以兼容TorchRL 0.9.2
直接应用必要的修复，让训练脚本可以运行
"""

import os
import sys
from pathlib import Path

def apply_fixes_to_train_script(script_path):
    """应用修复到训练脚本"""
    
    print(f"修复文件: {script_path}")
    
    # 读取文件
    with open(script_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    modified = False
    
    for i, line in enumerate(lines):
        # 修复1：在SACLoss之后添加alpha_init和target_entropy
        if 'loss_module = SACLoss(' in line and i+1 < len(lines):
            # 检查是否已经修复
            remaining_text = ''.join(lines[i:min(i+15, len(lines))])
            if 'alpha_init' not in remaining_text:
                # 找到闭括号的位置
                paren_count = 1
                end_idx = i + 1
                while end_idx < len(lines) and paren_count > 0:
                    for char in lines[end_idx]:
                        if char == '(':
                            paren_count += 1
                        elif char == ')':
                            paren_count -= 1
                            if paren_count == 0:
                                break
                    if paren_count > 0:
                        end_idx += 1
                
                # 在闭括号前添加参数
                if end_idx < len(lines):
                    # 插入新参数
                    insert_line = end_idx
                    lines[insert_line] = lines[insert_line].replace(
                        ')',
                        '        alpha_init=1.0,  # TorchRL 0.9.2需要\n' +
                        '        target_entropy=-2,  # 动作空间2维\n    )',
                        1
                    )
                    modified = True
                    print("  ✓ 添加alpha_init和target_entropy参数")
        
        # 修复2：添加临时目录
        if 'scratch_dir="/tmp"' in line:
            # 在replay_buffer创建前添加tempfile导入
            if 'import tempfile' not in ''.join(new_lines[-10:]):
                new_lines.append('    import tempfile\n')
                new_lines.append('    temp_dir = tempfile.mkdtemp(prefix="sac_")\n')
            line = line.replace('"/tmp"', 'temp_dir')
            modified = True
            print("  ✓ 使用独立临时目录")
        
        # 修复3：注释掉update_priority（如果没有priority参数）
        if 'replay_buffer.update_priority(' in line and 'priority' not in line:
            line = '    # ' + line.lstrip() + '    # TODO: 需要添加priority参数\n'
            modified = True
            print("  ✓ 注释update_priority调用")
        
        new_lines.append(line)
    
    # 修复4：添加主模块保护（如果使用MultiaSyncDataCollector）
    content = ''.join(new_lines)
    if 'MultiaSyncDataCollector' in content:
        # 查找main调用
        if 'if __name__ == "__main__":' not in content:
            # 在文件末尾查找main调用
            for i in range(len(new_lines)-1, -1, -1):
                if new_lines[i].strip().startswith('main('):
                    new_lines[i] = 'if __name__ == "__main__":\n    ' + new_lines[i]
                    modified = True
                    print("  ✓ 添加主模块保护")
                    break
    
    if modified:
        # 备份原文件
        backup_path = script_path + '.backup'
        if not os.path.exists(backup_path):
            with open(backup_path, 'w') as f:
                with open(script_path, 'r') as orig:
                    f.write(orig.read())
            print(f"  备份保存到: {backup_path}")
        
        # 写入修复后的内容
        with open(script_path, 'w') as f:
            f.writelines(new_lines)
        print(f"  ✅ 文件已修复！")
        return True
    else:
        print(f"  - 文件无需修复或已修复")
        return False

def main():
    """主函数"""
    print("=== TorchRL 0.9.2 快速修复工具 ===\n")
    
    # 要修复的文件
    scripts = [
        'rl_new/sac_cont/area_coverage_sac_cont_train.py',
        'rl_new/sac_cont/area_coverage_sac_cont_eval.py',
        'rl_new/sac_cont/area_coverage_v5_sac_cont_train.py',
        'rl_new/sac_cont/area_coverage_v5_sac_cont_eval.py',
    ]
    
    fixed_count = 0
    for script in scripts:
        if os.path.exists(script):
            if apply_fixes_to_train_script(script):
                fixed_count += 1
        else:
            print(f"文件不存在: {script}")
        print()
    
    print(f"=== 修复完成 ===")
    print(f"共修复 {fixed_count} 个文件\n")
    
    if fixed_count > 0:
        print("建议的测试步骤：")
        print("1. 使用小参数测试训练脚本：")
        print("   python rl_new/sac_cont/area_coverage_sac_cont_train.py \\")
        print("       collector.frames_per_batch=100 \\")
        print("       collector.total_frames=500 \\")
        print("       collector.num_envs=1")
        print("\n2. 如果使用MultiaSyncDataCollector遇到问题，考虑：")
        print("   - 改用SyncDataCollector")
        print("   - 或确保在if __name__ == '__main__':中运行")
        print("\n3. 监控训练是否正常：")
        print("   - 损失是否在更新")
        print("   - 是否有奖励改进")

if __name__ == "__main__":
    main()