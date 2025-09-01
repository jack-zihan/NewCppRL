"""
SAC同步训练脚本 (sync version)
基于数据收集批次的同步训练模式
"""
from __future__ import annotations

import os
import torch
# 设置缓存目录（确保有写权限）
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/root/.cache/torchinductor'  # 主缓存目录
os.environ['TORCHINDUCTOR_FX_GRAPH_CACHE'] = '1'  # 启用 FX 图缓存
os.environ['TORCHINDUCTOR_AUTOGRAD_CACHE'] = '1'  # 启用 AOTAutograd 缓存
os.environ['TRITON_CACHE_DIR'] = '/root/.cache/triton'  # Triton 缓存目录

# 创建缓存目录
os.makedirs('/root/.cache/torchinductor', exist_ok=True)
os.makedirs('/root/.cache/torchinductor/fxgraph', exist_ok=True)
os.makedirs('/root/.cache/torchinductor/aotautograd', exist_ok=True)
os.makedirs('/root/.cache/triton', exist_ok=True)

# 启用缓存
torch._inductor.config.fx_graph_cache = True  # 这个属性仍然有效
torch._inductor.config.force_disable_caches = False # 确保不禁用缓存
torch._dynamo.config.cache_size_limit = 256  # 缓存条目数限制
torch._dynamo.config.accumulated_cache_size_limit = 256  # 累积缓存限制

import sys
import time
import math
import tempfile
import warnings

from pathlib import Path
from functools import partial

import hydra
import numpy as np
import tensordict

import torch.cuda
import tqdm
import gymnasium as gym
from omegaconf import DictConfig

from tensordict import TensorDict
from tensordict.nn import CudaGraphModule
from torchrl._utils import compile_with_warmup, logger as torchrl_logger, timeit
from torchrl.collectors import MultiaSyncDataCollector, aSyncDataCollector
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import SoftUpdate, SACLoss, group_optimizers
from torchrl.data import LazyMemmapStorage, LazyTensorStorage,TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.record.loggers import get_logger

from rl_new.sac_cont_sy.model_utils import make_sac_models
from rl_new.sac_cont_sy.sac_utils import (setup_devices, create_update_fn, flatten, get_actor_actions,
                                          generate_exp_name, evaluate_policy_parallel as evaluate_policy, CheckpointManager)
from rl_new.sac_cont_sy.env_utils import make_train_environment, make_environment
from torchrl_utils_new.local_video_recorder import LocalVideoRecorder


torch.set_float32_matmul_precision("high")  # 提升矩阵乘法性能
tensordict.nn.functional_modules._exclude_td_from_pytree().set()


@hydra.main(version_base="1.3", config_path=".", config_name="config-async-server")
def main(cfg: DictConfig):
    # 处理临时目录路径
    temp_dir = cfg.buffer.temp_dir
    if temp_dir and temp_dir.startswith('~'):
        temp_dir = os.path.expanduser(temp_dir)

    with tempfile.TemporaryDirectory(dir=temp_dir) as tmpdir:
        # ============ 1. 创建实验目录和基础设置 ============
        exp_name = generate_exp_name(cfg.logger.model_name, cfg.logger.exp_name)
        # ckpt_path = Path.cwd() / 'ckpt' # 这是同步版本的遗留代码，等待同步版本上线解决
        # ckpt_path.mkdir(parents=True, exist_ok=True)

        # 设备配置
        # train_device, collector_devices = setup_devices(cfg) # 双缓冲才开启设备选择
        # torchrl_logger.info(f"训练设备: {train_device}")
        # torchrl_logger.info(f"收集设备: {collector_devices[:5]}... (共{len(collector_devices)}个)")
        train_device, collector_devices = (torch.device("cuda:0"), torch.device("cuda:1")) if cfg.in_server else (torch.device("cpu"), torch.device("cuda:0"))
        torchrl_logger.info(f"训练设备: {train_device}, 收集设备: {collector_devices}")

        # 设置随机种子
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        # 确定编译模式
        if cfg.compile.enable:
            compile_mode = (cfg.compile.mode or ("default" if cfg.compile.cudagraphs else "reduce-overhead"))
            compile_mode_collector = compile_mode
        else:
            compile_mode = None
            compile_mode_collector = False

        # ============ 2. 创建日志记录器和checkpoint管理器 ============
        logger = None
        if cfg.logger.backend:
            logger = get_logger(
                logger_type=cfg.logger.backend, experiment_name=exp_name, logger_name=exp_name, # logger_name在wandb不显示，主要影响本地存储名字
                wandb_kwargs={"mode": cfg.logger.mode, "config": dict(cfg),
                              "project": cfg.logger.project_name, "group": cfg.logger.group_name, "name": exp_name},
            )

        # 初始化checkpoint管理器 - 使用Hydra管理的工作目录，相对于Hydra输出目录
        checkpoint_dir = Path.cwd() / "checkpoints"
        checkpoint_manager = CheckpointManager(save_dir=checkpoint_dir, max_checkpoints=cfg.logger.test_ckpt_num)
        torchrl_logger.info(f"Checkpoint将保存到: {checkpoint_dir}")

        # ============ 3. 创建模型 ============
        # 使用配置中的环境ID创建模型
        if cfg.pretrained_model:
            torchrl_logger.info(f"加载预训练模型: {cfg.pretrained_model}")
            actor_critic = torch.load(cfg.pretrained_path, map_location=train_device)

            # 为探索策略创建副本
            exploration_actor_critic = torch.load(cfg.pretrained_path, map_location=collector_devices)
            exploration_policy = exploration_actor_critic[0]  # 提取 actor
        else:
            # 创建dummy环境用于模型初始化（与官方一致）
            torchrl_logger.info(f"创建环境: {cfg.env.env_id}")
            dummy_env = make_train_environment(cfg, device="cpu")

            actor_critic = make_sac_models(dummy_env, device=train_device) # 在正确设备上创建模型（关键：传递device参数）
            exploration_actor_critic = make_sac_models(dummy_env, device=collector_devices)
            exploration_actor_critic[0].load_state_dict(actor_critic[0].state_dict()) # 同步权重
            exploration_policy = exploration_actor_critic[0]  # 提取 actor 用于探索
            dummy_env.close() # 清理
            del dummy_env
        # ============ 4. 创建回放缓冲区 ============
        replay_buffer = TensorDictReplayBuffer(
            pin_memory=cfg.buffer.pin_memory, prefetch=cfg.buffer.prefetch, shared=False, #
            # storage=LazyMemmapStorage(max_size=cfg.buffer.buffer_size, scratch_dir=tmpdir),

            storage = LazyTensorStorage(max_size=cfg.buffer.buffer_size, device=train_device),
            # 是LazyMemmapStorage所以不需要replay_buffer.append_transform(lambda td: td.to(device))
            batch_size=cfg.buffer.batch_size)
        # 对TensorDictReplayBuffer进行懒初始化
        replay_buffer.extend(make_train_environment(cfg).rollout(1).view(-1))
        replay_buffer.empty()

        # 目前异步无法使用优先级回放，只能使用双缓存机制，这个以后再判断如何解决
        # replay_buffer = TensorDictPrioritizedReplayBuffer(
        #     alpha=0.7,
        #     beta=0.5,
        #     pin_memory=cfg.buffer.get('pin_memory', True),
        #     prefetch=cfg.buffer.get('prefetch', 3),
        #     storage=LazyMemmapStorage(
        #         max_size=cfg.buffer.buffer_size,
        #         scratch_dir=tmpdir,
        #     ),
        #     batch_size=cfg.buffer.batch_size,
        # )
        # replay_buffer.append_transform(lambda td: td.to(train_device))
        # replay_buffer.empty()

        # ============ 5. 创建收集器（同步模式，不传递replay_buffer） ============
        collector = aSyncDataCollector(
            partial(make_train_environment, cfg),
            exploration_policy,
            init_random_frames=0,  # Currently not supported, but accounted for in script: cfg.collector.init_random_frames,
            frames_per_batch=cfg.collector.frames_per_batch,
            total_frames=-1,
            device=collector_devices,
            env_device=torch.device("cpu"),
            compile_policy={"mode": compile_mode_collector, "warmup": 5} if compile_mode_collector else False,
            cudagraph_policy={"warmup": 20} if cfg.compile.cudagraphs else False,
            replay_buffer=replay_buffer,
            extend_buffer=True,
            postproc=flatten,
            no_cuda_sync=True,  # 放弃CPU对GPU的计算同步等待
            max_frames_per_traj=-1, # 不分割轨迹
        )
        collector.set_seed(cfg.seed)
        collector.start()
        torchrl_logger.info(f"创建{collector_devices}收集进程 (完全异步模式)")

        # 目前异步无法使用优先级回放，只能使用双缓存机制，这个以后再判断如何解决
        # env_kwargs = dict(cfg.env.env_kwargs) if hasattr(cfg.env, 'env_kwargs') and cfg.env.env_kwargs else {}
        # collector = MultiaSyncDataCollector(
        #     create_env_fn=[lambda d=dev: make_sac_env(
        #         env_id=cfg.env.env_id,
        #         num_envs=cfg.collector.processes_per_gpu if 'cuda' in str(dev) else 1,
        #         device=str(d),
        #         **env_kwargs
        #     ) for dev in collector_devices],
        #     policy=actor,
        #     policy_device='cpu',
        #     frames_per_batch=cfg.collector.frames_per_batch,
        #     total_frames=cfg.collector.total_frames,
        #     device=collector_devices,
        #     storing_device='cpu',
        #     max_frames_per_traj=-1,
        #     # 不传递 replay_buffer，使用同步收集模式
        #     postproc=flatten,
        # )
        # collector.set_seed(cfg.seed)
        # torchrl_logger.info(f"创建{len(collector_devices)}个收集进程 (同步模式)")

        # ============ 6. 创建损失和优化器 ============
        loss_module = SACLoss(actor_network=actor_critic[0], qvalue_network=actor_critic[1],  # actor and qvalue
                              num_qvalue_nets=2, loss_function=cfg.loss.loss_function, alpha_init=cfg.loss.alpha_init
                              , delay_actor=False, delay_qvalue=True)
        loss_module.make_value_estimator(gamma=cfg.loss.gamma)

        # 目标网络更新器
        target_net_updater = SoftUpdate(loss_module, eps=cfg.loss.target_update_polyak)

        # 创建优化器
        critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
        actor_params = list(loss_module.actor_network_params.flatten_keys().values())

        optimizer_actor = torch.optim.AdamW(
            actor_params, lr=cfg.optim.lr_actor, weight_decay=cfg.optim.weight_decay_actor, eps=cfg.optim.eps_actor)
        optimizer_critic = torch.optim.AdamW(
            critic_params, lr=cfg.optim.lr_critic, weight_decay=cfg.optim.weight_decay_critic, eps=cfg.optim.eps_critic)
        optimizer_alpha = torch.optim.AdamW(
            [loss_module.log_alpha], lr=cfg.optim.lr_alpha, weight_decay=cfg.optim.weight_decay_alpha)

        # 使用group_optimizers合并优化器
        optimizer = group_optimizers(optimizer_actor, optimizer_critic, optimizer_alpha)
        del optimizer_actor, optimizer_critic, optimizer_alpha

        # 创建GradScaler（如果使用混合精度）
        scaler = None
        if cfg.training.use_amp and torch.cuda.is_available():
            scaler = torch.amp.GradScaler('cuda')
            torchrl_logger.info("启用混合精度训练 (AMP)")

        # 创建优化函数
        update_fn = create_update_fn(loss_module, optimizer, target_net_updater, cfg, compile_mode, scaler)

        # ============ 7. 主训练循环（异步模式） ============
        start_time = time.time()

        # 核心参数
        init_random_frames = cfg.collector.init_random_frames  # 随机初始帧数
        update_freq = cfg.collector.update_freq  # 更新频率
        log_freq = cfg.logger.log_freq

        num_updates = 1000
        total_iter = 4000
        pbar = tqdm.tqdm(total=total_iter * num_updates)
        params = TensorDict.from_module(actor_critic[0]).data

        # Wait till we have enough data to start training
        while replay_buffer.write_count <= init_random_frames:
            time.sleep(0.01)

        losses = []
        for i in range(total_iter * num_updates):
            timeit.printevery(num_prints=total_iter * num_updates // log_freq, total_count=total_iter * num_updates,
                              erase=True) # num_prints打印总步数，total_count总步数
            if (i % update_freq) == 0:
                torchrl_logger.info("Updating weights")
                collector.update_policy_weights_(params) # Update weights of the inference policy
            pbar.update(1)

            # Optimization steps
            with timeit("train"):
                with timeit("train - rb - sample"):
                    sampled_tensordict = replay_buffer.sample() # Sample from replay buffer

                with timeit("train - update"):
                    torch.compiler.cudagraph_mark_step_begin()
                    loss_td = update_fn(sampled_tensordict).clone()
                losses.append(loss_td.select("loss_actor", "loss_qvalue", "loss_alpha"))

            # Logging
            if (i % log_freq) == (log_freq - 1):
                torchrl_logger.info("Logging")
                collected_frames = replay_buffer.write_count
                metrics_to_log = {}
                if collected_frames >= init_random_frames:
                    losses_m = torch.stack(losses).mean()
                    losses = []
                    metrics_to_log["train/q_loss"] = losses_m.get("loss_qvalue")
                    metrics_to_log["train/actor_loss"] = losses_m.get("loss_actor")
                    metrics_to_log["train/alpha_loss"] = losses_m.get("loss_alpha")
                    metrics_to_log["train/alpha"] = loss_td["alpha"]
                    metrics_to_log["train/entropy"] = loss_td["entropy"]
                    metrics_to_log["train/collected_frames"] = int(collected_frames)

                # Evaluation
                eval_interval = cfg.logger['eval_interval']
                if collected_frames >= init_random_frames and collected_frames % eval_interval == 0:
                    with timeit("eval"):
                        eval_metrics = evaluate_policy(actor_critic=actor_critic, train_device=train_device,
                                                       cfg=cfg, logger=logger, step=collected_frames)
                        metrics_to_log.update(eval_metrics)

                        # Checkpoint保存（基于评估奖励）
                        checkpoint_interval = cfg.logger['test_interval']
                        if collected_frames % checkpoint_interval == 0:
                            checkpoint_manager.save_if_best(model=actor_critic, reward=eval_metrics['eval/reward_mean'],
                                                            step=collected_frames)
                # 显示训练进度
                torchrl_logger.info(f"Collected frames: {collected_frames}")
                torchrl_logger.info(f"Eval Logs: {metrics_to_log}")

                if logger is not None:
                    metrics_to_log.update(timeit.todict(prefix="time"))
                    metrics_to_log["time/speed"] = pbar.format_dict["rate"]
                    log_metrics(logger, metrics_to_log, collected_frames)

        collector.shutdown()
        end_time = time.time()
        execution_time = end_time - start_time
        torchrl_logger.info(f"训练完成，耗时: {execution_time:.2f}秒")



################################################################同步代码#####################################
# 这些是同步的参数
# batch_size = cfg.buffer.batch_size
# frames_per_batch = cfg.collector.frames_per_batch
# num_updates = math.ceil(frames_per_batch / batch_size * cfg.loss.utd_ratio)
# test_interval = cfg.logger.test_interval


# # 初始化统计
# collected_frames = 0
# pbar = tqdm.tqdm(total=cfg.collector.total_frames, desc="收集数据")
#
# # 同步收集循环
# for i, data in enumerate(collector):
#     log_info = {}
#
#     # 处理收集到的数据
#     pbar.update(data.numel())
#     data = data.reshape(-1)  # 展平数据
#     current_frames = data.numel()
#     collected_frames += current_frames
#
#     # 提取episode统计信息
#     if ("next", "done") in data.keys(include_nested=True):
#         done_mask = data["next", "done"]
#         if done_mask.any():
#             # 收集episode奖励
#             if ("next", "episode_reward") in data.keys(include_nested=True):
#                 episode_rewards = data["next", "episode_reward"][done_mask]
#                 if len(episode_rewards) > 0:
#                     log_info["train/episode_reward"] = episode_rewards.mean().item()
#                     log_info["train/episode_reward_max"] = episode_rewards.max().item()
#                     log_info["train/episode_reward_min"] = episode_rewards.min().item()
#
#             # 收集episode长度
#             if ("next", "step_count") in data.keys(include_nested=True):
#                 episode_lengths = data["next", "step_count"][done_mask]
#                 if len(episode_lengths) > 0:
#                     log_info["train/episode_length"] = episode_lengths.float().mean().item()
#
#             # 收集weed_ratio（如果存在）
#             if ("next", "completion_ratio") in data.keys(include_nested=True):
#                 episode_completion_ratios = data["next", "completion_ratio"][done_mask]
#                 if len(episode_completion_ratios) > 0:
#                     log_info["train/episode_completion_ratio"] = episode_completion_ratios.mean().item()
#                     # 删除额外信息，避免replay buffer存储
#                     data.pop('completion_ratio', None)
#                     data.pop(('next', 'completion_ratio'), None)
#
#     # 手动添加到replay_buffer
#     replay_buffer.extend(data)
#
#     # 如果还在收集初始随机帧，跳过训练
#     if collected_frames < init_random_frames:
#         if logger and log_info:
#             for key, value in log_info.items():
#                 logger.log_scalar(key, value, step=collected_frames)
#         continue
#
#     # ============ 训练更新 ============
#     losses = []
#     for j in range(num_updates):
#         # 从回放缓冲区采样
#         sampled_tensordict = replay_buffer.sample()
#         if sampled_tensordict.device != train_device:
#             sampled_tensordict = sampled_tensordict.to(train_device, non_blocking=True)
#
#         # 执行更新
#         loss_out = update_fn(sampled_tensordict)
#         losses.append(loss_out.select("loss_actor", "loss_qvalue", "loss_alpha"))
#
#         # 更新优先级
#         td_error = (loss_out["loss_qvalue"] + loss_out["loss_actor"]).abs()
#         priority = td_error.expand(batch_size).detach()
#         replay_buffer.update_priority(sampled_tensordict["index"], priority)
#
#     # ============ 日志记录 ============
#     if i % 10 == 0 and len(losses) > 0:  # 每10个批次记录一次
#         # 计算平均损失
#         losses_tensor = torch.stack(losses)
#         log_info.update({
#             "train/q_loss": losses_tensor.get("loss_qvalue").mean().item(),
#             "train/a_loss": losses_tensor.get("loss_actor").mean().item(),
#             "train/alpha_loss": losses_tensor.get("loss_alpha").mean().item(),
#             "train/alpha": loss_out.get("alpha", 0),
#             "train/entropy": loss_out.get("entropy", 0),
#         })
#
#         # 记录收集信息
#         elapsed_time = time.time() - start_time
#         log_info.update({
#             "train/collected_frames": collected_frames,
#             "train/frames_per_sec": collected_frames / elapsed_time,
#             "train/batches": i + 1,
#         })
#
#         # 写入日志
#         if logger:
#             for key, value in log_info.items():
#                 if isinstance(value, torch.Tensor):
#                     value = value.item()
#                 logger.log_scalar(key, value, step=collected_frames)
#
#     # ============ 保存模型 ============
#     if collected_frames % test_interval == 0:
#         reward_str = f"{log_info.get('train/episode_reward', 0):.2f}" if 'train/episode_reward' in log_info else "0"
#         model_name = f"f[{collected_frames // 1000:05d}]_r[{reward_str}].pt"
#         torch.save(
#             actor_critic,
#             ckpt_path / model_name
#         )
#         torchrl_logger.info(f"保存模型: {model_name}")
#
#     # ============ 评估 ============
#     eval_interval = cfg.logger.get('eval_interval', 25000)
#     if collected_frames % eval_interval == 0 and collected_frames > 0:
#         torchrl_logger.info(f"开始评估 (frames: {collected_frames})")
#
#         # 执行评估
#         eval_metrics = evaluate_policy(
#             actor=actor,
#             env_id=cfg.env.env_id,
#             env_kwargs=env_kwargs,
#             device=train_device,
#             logger=logger,
#             step=collected_frames,
#             cfg=cfg
#         )
#
#         # 记录评估结果
#         torchrl_logger.info(
#             f"评估完成 - 奖励: {eval_metrics['eval/reward']:.2f} ± {eval_metrics['eval/reward_std']:.2f} "
#             f"[{eval_metrics['eval/reward_min']:.2f}, {eval_metrics['eval/reward_max']:.2f}]"
#         )
#
#     # 检查是否达到总帧数
#     if collected_frames >= cfg.collector.total_frames:
#         break
#
# # 训练结束
# collector.shutdown()
# end_time = time.time()
# execution_time = end_time - start_time
# torchrl_logger.info(f"训练完成，耗时: {execution_time:.2f}秒")
# torchrl_logger.info(f"总帧数: {collected_frames}")
# torchrl_logger.info(f"平均FPS: {collected_frames / execution_time:.2f}")


if __name__ == "__main__":
    main()
