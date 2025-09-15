# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""SAC Example.

This is a simple self-contained example of a SAC training script.

It supports state environments like MuJoCo.

The helper functions are coded in the utils.py associated with this script.
"""
from __future__ import annotations

import warnings

import os
import sys
import time
import math
import tqdm
import hydra
import shutil
import tempfile
import warnings
import tensordict
import torch
import torch.cuda
import numpy as np
import gymnasium as gym
from threading import RLock

from pathlib import Path
from functools import partial

from omegaconf import DictConfig
from tensordict import TensorDict
from tensordict.nn import CudaGraphModule
from torchrl._utils import compile_with_warmup, logger as torchrl_logger, timeit
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import SoftUpdate, SACLoss, group_optimizers
from torchrl.data import LazyMemmapStorage, LazyTensorStorage, TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.collectors import SyncDataCollector, MultiaSyncDataCollector, aSyncDataCollector
from torchrl.record.loggers import get_logger

from rl_new.sac_cont_sy.model_utils import make_sac_models
from rl_new.sac_cont_sy.sac_utils import (setup_devices, create_update_fn, flatten, get_actor_actions,
                                          generate_exp_name, evaluate_policy_parallel, CheckpointManager, log_metrics,
                                          evaluate_policy, evaluate_policy_parallel, setup_torch_cache,
                                          evaluate_policy_standalone, log_evaluate_results, is_time_to_evaluate)
from rl_new.sac_cont_sy.env_utils import make_train_environment, make_environment
from rl_new.sac_cont_sy.async_evaluator import AsyncEvaluator

torch.set_float32_matmul_precision("high")  # 提升矩阵乘法性能
tensordict.nn.functional_modules._exclude_td_from_pytree().set()


@hydra.main(version_base="1.1", config_path="", config_name="config-sync-server")
def main(cfg: DictConfig):  # noqa: F821
    # 处理临时目录路径
    temp_dir = cfg.buffer.temp_dir
    if temp_dir:
        if temp_dir.startswith('~'): temp_dir = os.path.expanduser(temp_dir) # ~转换为实际路径
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir); # 清空路径
        os.makedirs(temp_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=temp_dir) as tmpdir:
        # ============ 1. 创建实验目录和基础设置 ============
        exp_name = generate_exp_name(cfg.logger.model_name, cfg.logger.exp_name)

        # 设置多线程安全的tqdm锁
        tqdm.tqdm.set_lock(RLock())

        # 设备配置
        train_device, collector_devices = (torch.device("cuda:0"), torch.device("cuda:0")) if cfg.in_server else (
            torch.device("cpu"), torch.device("cpu"))
        torchrl_logger.info(f"训练设备: {train_device}, 收集设备: {collector_devices}")

        # 设置随机种子
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

        # 确定编译模式
        if cfg.compile.enable:
            if cfg.in_server: setup_torch_cache()
            compile_mode = (cfg.compile.mode or ("default" if cfg.compile.cudagraphs else "reduce-overhead"))
        else:
            compile_mode = None

        # 设置checkpoint目录
        checkpoint_dir = Path.cwd() / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torchrl_logger.info(f"Checkpoint将保存到: {checkpoint_dir}")

        # ============ 2. 创建日志记录器和异步评估器 ============
        logger = None
        if cfg.logger.backend:
            logger = get_logger(
                logger_type=cfg.logger.backend, experiment_name=exp_name, logger_name=exp_name,
                # logger_name在wandb不显示，主要影响本地存储名字
                wandb_kwargs={"mode": cfg.logger.mode, "config": dict(cfg),
                              "project": cfg.logger.project_name, "group": cfg.logger.group_name, "name": exp_name},
            )
        
        # 定义评估指标使用独立的x轴，避免异步评估的step冲突
        if logger is not None:
            logger.experiment.define_metric("eval_step")
            logger.experiment.define_metric("eval/*", step_metric="eval_step")
            torchrl_logger.info("已配置WandB评估指标使用独立的eval_step轴")

        # 初始化异步评估器
        async_evaluator = AsyncEvaluator(max_workers=cfg.logger.eval_worker)

        # ============ 3. 创建模型 ============
        # 使用配置中的环境ID创建模型
        if cfg.pretrained_model:
            torchrl_logger.info(f"加载预训练模型: {cfg.pretrained_model}")
            actor_critic = torch.load(cfg.pretrained_path, map_location=train_device, weights_only=False)
        else:
            torchrl_logger.info(f"使用dummy环境创建模型: {cfg.env.env_id}")
            actor_critic = make_sac_models(env=make_train_environment(cfg, device="cpu"),
                                           device=train_device)  # 在正确设备上创建模型（关键：传递device参数）

        # ============ 4. 创建回放缓冲区和采集器 ============
        # 同步缓冲区一版较大，需要memmap磁盘映射存储
        replay_buffer = TensorDictPrioritizedReplayBuffer(
            alpha=0.7, beta=0.5, batch_size=cfg.buffer.batch_size,
            pin_memory=cfg.buffer.pin_memory, prefetch=cfg.buffer.prefetch,
            storage=LazyMemmapStorage(max_size=cfg.buffer.buffer_size, scratch_dir=tmpdir),
        ).append_transform(lambda td: td.to(train_device))  # 采样后传输到训练设备

        # Create off-policy collector
        # collector = SyncDataCollector(
        #     create_env_fn=partial(make_train_environment, cfg), policy=actor_critic[0],  # 提取 actor 用于探索
        #     init_random_frames=cfg.collector.init_random_frames, total_frames=cfg.collector.total_frames,
        #     frames_per_batch=cfg.collector.frames_per_batch, max_frames_per_traj=-1,
        #     device=None, policy_device=train_device, storing_device="cpu", env_device="cpu",
        #     compile_policy={"mode": compile_mode} if compile_mode else False,
        #     cudagraph_policy={"warmup": 10} if cfg.compile.cudagraphs else False,
        # )

        # Create the collector
        collector = MultiaSyncDataCollector(
            create_env_fn=[lambda: make_train_environment(cfg, device='cpu')] * cfg.collector.num_collectors,
            frames_per_batch=cfg.collector.frames_per_batch, total_frames=cfg.collector.total_frames,
             device=None, policy_device=train_device, storing_device="cpu", env_device="cpu",
            policy=actor_critic[0], max_frames_per_traj=-1,
        )
        #
        # collector = aSyncDataCollector(
        #     partial(make_train_environment, cfg),
        #     exploration_policy,
        #     init_random_frames=0,
        #     # Currently not supported, but accounted for in script: cfg.collector.init_random_frames,
        #     frames_per_batch=cfg.collector.frames_per_batch,
        #     total_frames=-1,
        #     device=collector_devices,
        #     env_device=torch.device("cpu"),
        #     compile_policy={"mode": compile_mode_collector, "warmup": 5} if compile_mode_collector else False,
        #     cudagraph_policy={"warmup": 20} if cfg.compile.cudagraphs else False,
        #     replay_buffer=replay_buffer,
        #     extend_buffer=True,
        #     postproc=flatten,
        #     no_cuda_sync=True,  # 放弃CPU对GPU的计算同步等待
        #     max_frames_per_traj=-1,  # 不分割轨迹
        # )

        collector.set_seed(cfg.seed)
        torchrl_logger.info(f"创建{collector_devices}收集进程 (同步模式)")

        # ============ 5. 创建损失和优化器 ============
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

        # ============ 7. 主训练循环（同步模式） ============

        # Main loop
        collected_frames = 0
        pbar = tqdm.tqdm(total=cfg.collector.total_frames, desc="Training", position=0,leave=True, dynamic_ncols=True) # 固定在顶部 训练结束后保留 适应终端宽度

        init_random_frames = cfg.collector.init_random_frames
        frames_per_batch = cfg.collector.frames_per_batch
        num_updates = math.ceil(frames_per_batch / cfg.buffer.batch_size * cfg.loss.utd_ratio)

        collector_iter, total_iter = iter(collector), len(collector)
        start_time = time.time()

        for step in range(total_iter):
            timeit.printevery(num_prints=1000, total_count=total_iter, erase=True)

            with timeit("collect"):
                tensordict = next(collector_iter)

            current_frames = tensordict.numel()
            pbar.update(current_frames)

            with timeit("rb - extend"):
                # Add to replay buffer
                tensordict = tensordict.reshape(-1)
                replay_buffer.extend(tensordict)

            collected_frames += current_frames

            # Optimization steps
            with timeit("train"):
                if collected_frames >= init_random_frames:
                    losses = TensorDict(batch_size=[num_updates])
                    for i in range(num_updates):
                        with timeit("rb - sample"):
                            sampled_tensordict = replay_buffer.sample()  # Sample from replay buffer

                        with timeit("update"):
                            torch.compiler.cudagraph_mark_step_begin()
                            loss_td = update_fn(sampled_tensordict)
                        losses[i] = loss_td.select("loss_actor", "loss_qvalue", "loss_alpha")
                        replay_buffer.update_tensordict_priority(sampled_tensordict)  # Update priority

            episode_end = (tensordict["next", "done"] if tensordict["next", "done"].any()
                           else tensordict["next", "truncated"])
            episode_rewards = tensordict["next", "episode_reward"][episode_end]

            # Logging
            metrics_to_log = {}
            if len(episode_rewards) > 0:
                episode_length = tensordict["next", "step_count"][episode_end]
                completion_ratio = tensordict["next", "completion_ratio"][episode_end]
                metrics_to_log["train/reward"] = episode_rewards.mean().item()
                metrics_to_log["train/episode_length"] = (episode_length.sum() / len(episode_length)).item()
                metrics_to_log["train/completion_ratio"] = completion_ratio.mean().item()

            if collected_frames >= init_random_frames:
                losses = losses.mean()
                metrics_to_log["train/q_loss"] = losses["loss_qvalue"].item()
                metrics_to_log["train/actor_loss"] = losses["loss_actor"].item()
                metrics_to_log["train/alpha_loss"] = losses["loss_alpha"].item()
                metrics_to_log["train/alpha"] = loss_td["alpha"].item()
                metrics_to_log["train/entropy"] = loss_td["entropy"].item()

            # Evaluation
            if is_time_to_evaluate(current_frames, collected_frames, cfg):
                model_path = checkpoint_dir / f"model_s{collected_frames:08d}_eval_pending.pt" # pending表示等待评估
                torch.save(actor_critic, model_path)
                # 提交异步评估
                async_evaluator.submit_eval(evaluate_policy_standalone, str(model_path.absolute()), cfg, collected_frames)
                torchrl_logger.info(f"提交评估任务: collected_frames {collected_frames}")

            evaluate_results = async_evaluator.get_evaluate_results()
            if evaluate_results:
                log_evaluate_results(evaluate_results, checkpoint_dir, logger)


            if logger is not None:
                metrics_to_log.update(timeit.todict(prefix="time"))
                metrics_to_log["time/speed"] = pbar.format_dict["rate"]
                log_metrics(logger, metrics_to_log, collected_frames)

            # Update weights of the inference policy
            collector.update_policy_weights_()

        torchrl_logger.info(f"Training took {time.time() - start_time:.2f} seconds to finish")

        collector.shutdown()
        torchrl_logger.info("等待异步评估任务完成...")
        remaining_results = async_evaluator.shutdown(wait=True)
        if remaining_results:
            torchrl_logger.info(f"处理剩余的 {len(remaining_results)} 个评估结果")
            log_evaluate_results(remaining_results, checkpoint_dir, logger)

        
        # 确保WandB数据完整保存
        if logger is not None:
           logger.experiment.finish()
           torchrl_logger.info("WandB运行已完成")
        
        time.sleep(3)  # 休眠3s使得日志上传完成


if __name__ == "__main__":
    main()
