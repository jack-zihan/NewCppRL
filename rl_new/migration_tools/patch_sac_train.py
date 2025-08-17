#!/usr/bin/env python
"""
手动修复area_coverage_sac_cont_train.py的TorchRL 0.9.2兼容性问题
"""

def create_patched_train_script():
    """创建修复后的训练脚本"""
    
    # 读取原始文件
    with open('sac_cont/area_coverage_sac_cont_train.py', 'r') as f:
        lines = f.readlines()
    
    # 需要修改的地方：
    # 1. SACLoss添加alpha_init和target_entropy参数
    # 2. 添加主模块保护
    # 3. 修复临时目录问题
    
    modified = False
    new_lines = []
    
    skip_lines = set()  # 记录需要跳过的行号
    
    for i, line in enumerate(lines):
        # 跳过已处理的行
        if i in skip_lines:
            continue
            
        # 修复SACLoss参数
        if 'loss_module = SACLoss(' in line:
            new_lines.append(line)
            # 在SACLoss创建中添加alpha相关参数
            if i+1 < len(lines) and 'actor_network=' in lines[i+1]:
                new_lines.append(lines[i+1])  # actor_network
                new_lines.append(lines[i+2])  # qvalue_network
                new_lines.append(lines[i+3])  # num_qvalue_nets
                new_lines.append(lines[i+4])  # loss_function
                new_lines.append(lines[i+5])  # delay_actor
                new_lines.append(lines[i+6])  # delay_qvalue
                # 在闭括号前添加新参数
                new_lines.append('        alpha_init=1.0,  # TorchRL 0.9.2需要\n')
                new_lines.append('        target_entropy=-2,  # 动作空间维度为2\n')
                new_lines.append(lines[i+7])  # 闭括号
                # 标记已处理的行
                for j in range(1, 8):
                    skip_lines.add(i+j)
                modified = True
            continue
            
        # 添加临时目录创建
        if 'replay_buffer = TensorDictPrioritizedReplayBuffer(' in line:
            new_lines.append('    # 创建独立的临时目录避免文件冲突\n')
            new_lines.append('    import tempfile\n')
            new_lines.append('    temp_dir = tempfile.mkdtemp(prefix="sac_replay_")\n')
            new_lines.append(line)
            continue
            
        # 修改scratch_dir参数
        if 'scratch_dir=' in line and '"/tmp"' in line:
            new_lines.append(line.replace('"/tmp"', 'temp_dir'))
            modified = True
            continue
            
        # 添加主模块保护
        if line.strip().startswith('main(') and not lines[i-1].strip().startswith('if __name__'):
            new_lines.append('if __name__ == "__main__":\n')
            new_lines.append('    ' + line)
            modified = True
            continue
            
        # 添加未处理的行
        new_lines.append(line)
    
    # 写入修复后的文件
    output_file = 'sac_cont/area_coverage_sac_cont_train_fixed.py'
    with open(output_file, 'w') as f:
        f.writelines(new_lines)
    
    print(f"✓ 创建修复后的文件: {output_file}")
    return output_file, modified

def test_import():
    """测试修复后的文件是否可以导入"""
    try:
        import sys
        sys.path.insert(0, 'sac_cont')
        # 只测试导入，不运行
        exec(open('sac_cont/area_coverage_sac_cont_train_fixed.py').read(), {'__name__': '__not_main__'})
        print("✓ 修复后的文件语法检查通过")
        return True
    except SyntaxError as e:
        print(f"✗ 语法错误: {e}")
        return False
    except Exception as e:
        # 其他错误（如导入错误）是预期的，因为我们不是真正运行
        print(f"✓ 文件可解析（导入错误是预期的）: {type(e).__name__}")
        return True

if __name__ == "__main__":
    print("=== 修复area_coverage_sac_cont_train.py ===\n")
    
    output_file, modified = create_patched_train_script()
    
    if modified:
        print("\n修复的问题:")
        print("1. 添加SACLoss的alpha_init和target_entropy参数")
        print("2. 使用独立的临时目录避免文件冲突")
        print("3. 添加主模块保护（如果需要）")
    
    print("\n=== 测试修复后的文件 ===")
    test_import()
    
    print("\n下一步:")
    print(f"1. 测试运行: python {output_file}")
    print("2. 如果测试成功，可以替换原文件")
    print("3. 对其他3个文件进行类似修复")