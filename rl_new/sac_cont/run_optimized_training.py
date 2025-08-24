#!/usr/bin/env python3
"""
运行优化的SAC训练脚本 - 支持6x RTX 3090多GPU配置
使用方法：
    # 使用默认配置
    python run_optimized_training.py
    
    # 使用自定义配置
    python run_optimized_training.py --config custom_config.yaml
    
    # 使用所有GPU
    python run_optimized_training.py --gpu_devices -1
    
    # 使用特定GPU
    python run_optimized_training.py --gpu_devices 0,1,2,3,4,5
"""

import argparse
import yaml
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

# 导入优化的训练器
from sac_cont_train_class import main


def parse_args():
    parser = argparse.ArgumentParser(description='Run optimized SAC training')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (default: configs/train_sac_cont_config.yaml)')
    parser.add_argument('--gpu_devices', type=str, default=None,
                        help='GPU devices to use: -1 for all, or comma-separated list (e.g., 0,1,2)')
    parser.add_argument('--processes_per_gpu', type=int, default=None,
                        help='Number of processes per GPU (default: 2 for RTX 3090)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training (default: 512)')
    parser.add_argument('--use_amp', action='store_true',
                        help='Enable mixed precision training')
    parser.add_argument('--checkpoint_interval', type=int, default=None,
                        help='Checkpoint saving interval in frames')
    parser.add_argument('--logger_backend', type=str, default=None,
                        choices=['wandb', 'tensorboard', None],
                        help='Logger backend to use')
    return parser.parse_args()


def update_config(cfg, args):
    """根据命令行参数更新配置"""
    
    # GPU配置
    if args.gpu_devices is not None:
        if args.gpu_devices == '-1':
            cfg.collector.gpu_devices = -1  # 使用所有GPU
        else:
            # 解析GPU列表
            gpu_list = [int(x.strip()) for x in args.gpu_devices.split(',')]
            cfg.collector.gpu_devices = gpu_list
    
    # 每GPU进程数
    if args.processes_per_gpu is not None:
        cfg.collector.processes_per_gpu = args.processes_per_gpu
    
    # 批量大小
    if args.batch_size is not None:
        cfg.buffer.batch_size = args.batch_size
    
    # 混合精度训练
    if args.use_amp:
        if 'training' not in cfg:
            cfg.training = {}
        cfg.training.use_amp = True
    
    # 检查点间隔
    if args.checkpoint_interval is not None:
        if 'training' not in cfg:
            cfg.training = {}
        cfg.training.checkpoint_interval = args.checkpoint_interval
    
    # 日志后端
    if args.logger_backend is not None:
        cfg.logger.backend = args.logger_backend
    
    return cfg


def print_training_info(cfg):
    """打印训练配置信息"""
    print("=" * 60)
    print("SAC训练配置（优化版）")
    print("=" * 60)
    
    # GPU配置
    gpu_devices = cfg.collector.get('gpu_devices', None)
    if gpu_devices == -1:
        import torch
        num_gpus = torch.cuda.device_count()
        print(f"GPU配置: 使用所有{num_gpus}个GPU")
    elif gpu_devices:
        print(f"GPU配置: 使用GPU {gpu_devices}")
    else:
        cpu_workers = cfg.collector.get('cpu_workers', None)
        if cpu_workers:
            print(f"CPU配置: 使用{cpu_workers}个CPU工作进程")
        else:
            print("使用默认CPU配置")
    
    # 其他重要参数
    print(f"批量大小: {cfg.buffer.batch_size}")
    print(f"总帧数: {cfg.collector.total_frames:,}")
    print(f"每批帧数: {cfg.collector.frames_per_batch:,}")
    
    # 训练特性
    use_amp = cfg.get('training', {}).get('use_amp', False)
    print(f"混合精度训练: {'启用' if use_amp else '禁用'}")
    
    logger_backend = cfg.logger.get('backend', None)
    if logger_backend:
        print(f"日志后端: {logger_backend}")
    
    print("=" * 60)
    print()


def main_wrapper():
    args = parse_args()
    
    # 加载配置
    if args.config:
        config_path = Path(args.config)
    else:
        base_dir = Path(__file__).parent.parent.parent
        config_path = base_dir / 'configs' / 'train_sac_cont_config.yaml'
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    cfg = DictConfig(cfg)
    
    # 更新配置
    cfg = update_config(cfg, args)
    
    # 打印配置信息
    print_training_info(cfg)
    
    # 运行训练
    print("开始训练...")
    main(cfg)


if __name__ == "__main__":
    main_wrapper()