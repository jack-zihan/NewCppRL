import os
import torch
from torchrl._utils import compile_with_warmup, logger as torchrl_logger
from tensordict.nn import CudaGraphModule

# ============ 设备配置============
def setup_devices(cfg):
    """设置训练和收集设备（单进程版本）"""
    # 训练设备
    if cfg.training.device:
        train_device = torch.device(cfg.training.device)
    else:
        train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 收集设备（单个设备）
    if cfg.collector.gpu_device and torch.cuda.is_available():
        # 验证GPU是否存在
        gpu_str = cfg.collector.gpu_device
        if ':' in gpu_str:
            gpu_id = int(gpu_str.split(':')[1])
        else:
            gpu_id = int(gpu_str)
            gpu_str = f"cuda:{gpu_id}"
        
        if gpu_id < torch.cuda.device_count():
            collector_device = torch.device(gpu_str)
            torchrl_logger.info(f"收集器使用GPU: {collector_device}")
        else:
            torchrl_logger.warning(f"GPU {gpu_str} 不存在，将使用CPU收集")
            collector_device = torch.device("cpu")
    else:
        collector_device = torch.device("cpu")
        torchrl_logger.info("收集器使用CPU")

    return train_device, collector_device

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
