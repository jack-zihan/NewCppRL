import os
import torch
from torchrl._utils import compile_with_warmup, logger as torchrl_logger
from tensordict.nn import CudaGraphModule

# ============ 设备配置============
def setup_devices(cfg):
    """设置训练和收集设备"""
    # 训练设备
    if cfg.training.device:
        train_device = torch.device(cfg.training.device)
    else:
        train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 提取训练GPU ID（用于排除）
    train_gpu_id = int(str(train_device).split(':')[1])

    # 收集设备配置
    collector_devices = []
    gpu_devices = cfg.collector.gpu_devices

    if gpu_devices is not None:
        # GPU收集器
        if gpu_devices == -1:  # 使用所有GPU（除了训练GPU）
            all_gpus = list(range(torch.cuda.device_count()))
            if train_gpu_id is not None and train_gpu_id in all_gpus and len(all_gpus) > 1:
                all_gpus.remove(train_gpu_id)  # 排除训练GPU（如果有多个GPU）
            gpu_devices = all_gpus
            if not gpu_devices:  # 如果没有可用GPU
                torchrl_logger.warning("没有可用的GPU用于收集，将使用CPU")
                gpu_devices = []
            elif len(gpu_devices) == 1 and gpu_devices[0] == train_gpu_id:
                torchrl_logger.warning(f"只有一个GPU，训练和收集将共享GPU {train_gpu_id}")
        elif isinstance(gpu_devices, list):
            # 验证GPU ID是否有效
            valid_gpus = []
            for gpu_id in gpu_devices:
                if gpu_id < torch.cuda.device_count():
                    valid_gpus.append(gpu_id)
                else:
                    torchrl_logger.warning(f"GPU {gpu_id} 不存在，跳过")
            gpu_devices = valid_gpus

        processes_per_gpu = cfg.collector.get('processes_per_gpu', 2)
        for gpu_id in gpu_devices:
            collector_devices.extend([f'cuda:{gpu_id}'] * processes_per_gpu)

    # CPU收集器
    cpu_workers = cfg.collector.get('cpu_workers', 0)
    if cpu_workers == -1:  # 最大化CPU使用
        cpu_workers = max(1, os.cpu_count() - 2)
    if cpu_workers > 0:
        collector_devices.extend(['cpu'] * cpu_workers)

    # 如果没有配置任何设备，使用默认配置
    if not collector_devices:
        collector_devices = ['cpu'] * cfg.collector.get('num_envs', 32)

    return train_device, collector_devices

# ============ 优化的更新函数============
def create_update_fn(loss_module, optimizer, target_net_updater, cfg, scaler=None):
    """创建优化的更新函数，支持编译和cudagraph"""

    def update(sampled_tensordict):
        # 计算损失
        if cfg.training.use_amp and scaler is not None:
            # 混合精度训练 - 使用autocast
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss_out = loss_module(sampled_tensordict)
                actor_loss = loss_out["loss_actor"]
                q_loss = loss_out["loss_qvalue"]
                alpha_loss = loss_out["loss_alpha"]
                total_loss = actor_loss + q_loss + alpha_loss

            # 使用GradScaler进行反向传播
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()

            # 梯度裁剪
            if cfg.optim.max_grad_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), cfg.optim.max_grad_norm
                )

            scaler.step(optimizer)
            scaler.update()
        else:
            # 标准训练
            loss_out = loss_module(sampled_tensordict)
            actor_loss = loss_out["loss_actor"]
            q_loss = loss_out["loss_qvalue"]
            alpha_loss = loss_out["loss_alpha"]
            total_loss = actor_loss + q_loss + alpha_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()

            if cfg.optim.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), cfg.optim.max_grad_norm
                )

            optimizer.step()

        # 更新目标网络
        target_net_updater.step()

        return loss_out.detach()

    # 编译优化（使用compile_with_warmup）
    if cfg.compile.enable:
        mode = cfg.compile.mode
        warmup = cfg.compile.warmup
        update = compile_with_warmup(update, mode=mode, warmup=warmup)
        torchrl_logger.info(f"启用编译加速，模式: {mode}, warmup: {warmup}")

    # CudaGraph优化（需要PyTorch 2.0+）
    if cfg.compile.cudagraph and torch.cuda.is_available():
        try:
            update = CudaGraphModule(update, in_keys=[], out_keys=[], warmup=10)
            torchrl_logger.info("启用CudaGraph优化")
        except Exception as e:
            torchrl_logger.warning(f"CudaGraph初始化失败: {e}")

    return update
