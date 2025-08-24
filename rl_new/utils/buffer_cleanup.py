#!/usr/bin/env python3
"""
Replay Buffer临时文件清理工具
用于管理和清理/home/lzh/data/rl_buffers下的临时文件
"""

import os
import shutil
import time
from pathlib import Path
from datetime import datetime, timedelta
import argparse


def get_directory_size(path):
    """计算目录大小（MB）"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)  # 转换为MB


def get_directory_age(path):
    """获取目录的年龄（小时）"""
    stat = path.stat()
    creation_time = stat.st_mtime
    current_time = time.time()
    age_hours = (current_time - creation_time) / 3600
    return age_hours


def cleanup_old_buffers(base_dir="/home/lzh/data/rl_buffers", 
                        max_age_hours=24, 
                        keep_recent=5,
                        dry_run=False):
    """
    清理旧的缓冲区文件
    
    Args:
        base_dir: 缓冲区基础目录
        max_age_hours: 最大保留时间（小时）
        keep_recent: 保留最近的N个目录
        dry_run: 只显示将要删除的内容，不实际删除
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"⚠️ 目录不存在: {base_dir}")
        return
    
    print(f"🔍 扫描目录: {base_dir}")
    print("=" * 60)
    
    total_size_before = get_directory_size(base_dir)
    print(f"📊 当前总大小: {total_size_before:.2f} MB")
    
    # 收集所有buffer目录
    buffer_dirs = []
    
    # 遍历算法目录
    for algo_dir in base_path.iterdir():
        if not algo_dir.is_dir():
            continue
            
        # 遍历checkpoint目录
        for ckpt_dir in algo_dir.iterdir():
            if not ckpt_dir.is_dir():
                continue
                
            # 遍历buffer目录
            for buffer_dir in ckpt_dir.glob("buffer_*"):
                if buffer_dir.is_dir():
                    age_hours = get_directory_age(buffer_dir)
                    size_mb = get_directory_size(buffer_dir)
                    buffer_dirs.append({
                        'path': buffer_dir,
                        'age_hours': age_hours,
                        'size_mb': size_mb,
                        'mtime': buffer_dir.stat().st_mtime
                    })
    
    if not buffer_dirs:
        print("✅ 没有找到buffer目录")
        return
    
    # 按修改时间排序（最新的在前）
    buffer_dirs.sort(key=lambda x: x['mtime'], reverse=True)
    
    print(f"\n📋 找到 {len(buffer_dirs)} 个buffer目录:")
    print("-" * 60)
    
    # 决定要删除的目录
    dirs_to_delete = []
    dirs_to_keep = []
    
    for i, buffer_info in enumerate(buffer_dirs):
        buffer_dir = buffer_info['path']
        age_hours = buffer_info['age_hours']
        size_mb = buffer_info['size_mb']
        
        # 保留最近的N个目录
        if i < keep_recent:
            dirs_to_keep.append(buffer_info)
            status = "保留（最近）"
        # 删除超过时间限制的目录
        elif age_hours > max_age_hours:
            dirs_to_delete.append(buffer_info)
            status = f"删除（{age_hours:.1f}小时）"
        else:
            dirs_to_keep.append(buffer_info)
            status = "保留"
        
        print(f"  {status:15} | {size_mb:8.2f} MB | {buffer_dir.relative_to(base_path)}")
    
    # 执行删除
    if dirs_to_delete:
        print(f"\n🗑️ 准备删除 {len(dirs_to_delete)} 个目录:")
        total_size_to_delete = sum(d['size_mb'] for d in dirs_to_delete)
        print(f"   将释放空间: {total_size_to_delete:.2f} MB")
        
        if not dry_run:
            for buffer_info in dirs_to_delete:
                buffer_dir = buffer_info['path']
                try:
                    shutil.rmtree(buffer_dir)
                    print(f"   ✅ 已删除: {buffer_dir.name}")
                except Exception as e:
                    print(f"   ❌ 删除失败 {buffer_dir.name}: {e}")
            
            # 清理空目录
            cleanup_empty_dirs(base_path)
            
            # 计算清理后的大小
            total_size_after = get_directory_size(base_dir)
            space_freed = total_size_before - total_size_after
            print(f"\n✅ 清理完成！释放空间: {space_freed:.2f} MB")
        else:
            print("\n⚠️ DRY RUN模式 - 未实际删除任何文件")
    else:
        print("\n✅ 没有需要清理的目录")
    
    # 显示保留的目录
    if dirs_to_keep:
        print(f"\n📁 保留 {len(dirs_to_keep)} 个目录:")
        total_size_kept = sum(d['size_mb'] for d in dirs_to_keep)
        print(f"   占用空间: {total_size_kept:.2f} MB")


def cleanup_empty_dirs(base_path):
    """清理空的目录"""
    for dirpath, dirnames, filenames in os.walk(base_path, topdown=False):
        if not dirnames and not filenames:
            try:
                os.rmdir(dirpath)
                print(f"   🧹 清理空目录: {Path(dirpath).relative_to(base_path)}")
            except:
                pass


def monitor_usage(base_dir="/home/lzh/data/rl_buffers"):
    """监控缓冲区目录的使用情况"""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"⚠️ 目录不存在: {base_dir}")
        return
    
    print(f"📊 缓冲区使用情况监控")
    print("=" * 60)
    
    # 统计信息
    total_size = 0
    buffer_count = 0
    algo_stats = {}
    
    # 遍历目录
    for algo_dir in base_path.iterdir():
        if not algo_dir.is_dir():
            continue
        
        algo_name = algo_dir.name
        algo_size = 0
        algo_buffers = 0
        
        for ckpt_dir in algo_dir.iterdir():
            if not ckpt_dir.is_dir():
                continue
                
            for buffer_dir in ckpt_dir.glob("buffer_*"):
                if buffer_dir.is_dir():
                    size_mb = get_directory_size(buffer_dir)
                    algo_size += size_mb
                    algo_buffers += 1
                    buffer_count += 1
                    total_size += size_mb
        
        if algo_buffers > 0:
            algo_stats[algo_name] = {
                'size': algo_size,
                'count': algo_buffers
            }
    
    # 显示统计
    print(f"📁 基础目录: {base_dir}")
    print(f"📈 总大小: {total_size:.2f} MB")
    print(f"🔢 Buffer数量: {buffer_count}")
    
    if algo_stats:
        print(f"\n📊 按算法分类:")
        for algo_name, stats in sorted(algo_stats.items()):
            print(f"  - {algo_name:30} | {stats['count']:3} buffers | {stats['size']:8.2f} MB")
    
    # 显示最大的buffer
    print(f"\n🏆 最大的5个buffer:")
    all_buffers = []
    for algo_dir in base_path.iterdir():
        if not algo_dir.is_dir():
            continue
        for ckpt_dir in algo_dir.iterdir():
            if not ckpt_dir.is_dir():
                continue
            for buffer_dir in ckpt_dir.glob("buffer_*"):
                if buffer_dir.is_dir():
                    size_mb = get_directory_size(buffer_dir)
                    all_buffers.append((buffer_dir, size_mb))
    
    all_buffers.sort(key=lambda x: x[1], reverse=True)
    for buffer_dir, size_mb in all_buffers[:5]:
        rel_path = buffer_dir.relative_to(base_path)
        print(f"  {size_mb:8.2f} MB | {rel_path}")


def main():
    parser = argparse.ArgumentParser(description='Replay Buffer临时文件清理工具')
    parser.add_argument('--base-dir', default='/home/lzh/data/rl_buffers',
                       help='缓冲区基础目录')
    parser.add_argument('--max-age', type=int, default=24,
                       help='最大保留时间（小时）')
    parser.add_argument('--keep-recent', type=int, default=5,
                       help='保留最近的N个目录')
    parser.add_argument('--dry-run', action='store_true',
                       help='只显示将要删除的内容，不实际删除')
    parser.add_argument('--monitor', action='store_true',
                       help='监控模式：显示使用情况统计')
    
    args = parser.parse_args()
    
    if args.monitor:
        monitor_usage(args.base_dir)
    else:
        cleanup_old_buffers(
            base_dir=args.base_dir,
            max_age_hours=args.max_age,
            keep_recent=args.keep_recent,
            dry_run=args.dry_run
        )


if __name__ == "__main__":
    main()