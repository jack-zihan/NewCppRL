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
from dataclasses import dataclass
from typing import Optional, Union, Iterator

from pathlib import Path
from functools import partial

from omegaconf import DictConfig
from tensordict import TensorDict
from tensordict.nn import CudaGraphModule
from torchrl._utils import compile_with_warmup, logger as torchrl_logger, timeit
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.envs.transforms import MultiStepTransform
from torchrl.objectives import SoftUpdate, SACLoss, group_optimizers
from torchrl.data import LazyMemmapStorage, LazyTensorStorage, TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.collectors import SyncDataCollector, MultiaSyncDataCollector
from torchrl.record.loggers import get_logger

from rl_new.sac_cont_sy.model_utils import make_sac_models, make_sac_resnet_dual_models
from rl_new.sac_cont_sy.sac_utils import (setup_devices, create_update_fn, flatten, get_actor_actions,
                                          LossMode, HIFAssistedSACLoss, generate_exp_name, evaluate_policy_parallel,
                                          CheckpointManager, log_metrics, evaluate_policy, evaluate_policy_parallel,
                                          setup_torch_cache, evaluate_policy_standalone, log_evaluate_results,
                                          is_time_to_evaluate)
from torchrl_utils.model.resnet_fpn_dual import HIFReconstructionLoss
from rl_new.sac_cont_sy.env_utils import make_train_environment
from rl_new.sac_cont_sy.async_evaluator import AsyncEvaluator
from rl_new.sac_cont_sy.bucketed_replay import BucketedTensorDictPrioritizedReplayBuffer
from rl_new.sac_cont_sy.train_utils import CurriculumState, create_replay_buffer, create_collector, \
    execute_stage_transition, load_curriculum_config, TrainingPhaseManager

torch.set_float32_matmul_precision("high")  # 提升矩阵乘法性能
tensordict.nn.functional_modules._exclude_td_from_pytree().set()

@hydra.main(version_base="1.1", config_path="", config_name="config-sync-server")
def main(cfg: DictConfig):  # noqa: F821
    # 清空临时数据路径
    temp_dir = cfg.buffer.temp_dir
    if temp_dir:
        if temp_dir.startswith('~'): temp_dir = os.path.expanduser(temp_dir)  # ~转换为实际路径
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir);  # 清空路径
        os.makedirs(temp_dir, exist_ok=True)

    with (tempfile.TemporaryDirectory(dir=temp_dir) as tmpdir):
        # ============ 1. 创建实验目录和基础设置 ============
        # 生成实验名称
        exp_name = generate_exp_name(cfg.logger.model_name, cfg.logger.exp_name)

        # 设置多线程安全的tqdm锁
        tqdm.tqdm.set_lock(RLock())

        # 设备配置
        train_device = torch.device(cfg.training.device)
        collector_device = train_device
        torchrl_logger.info(f"训练设备: {train_device}, 收集设备: {collector_device}")

        # 设置随机种子
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

        # 确定编译模式
        compile_mode = None
        if cfg.compile.enable:
            if cfg.in_server: setup_torch_cache()
            compile_mode = (cfg.compile.mode or ("default" if cfg.compile.cudagraphs else "reduce-overhead"))

        # 设置checkpoint目录
        checkpoint_dir = Path.cwd() / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torchrl_logger.info(f"Checkpoint将保存到: {checkpoint_dir}")

        # 设置课程学习参数
        curriculum_config = load_curriculum_config(cfg)
        curriculum_state, initial_stage = None, None
        if curriculum_config:
            # 初始化课程状态
            curriculum_state = CurriculumState()
            initial_stage = curriculum_config['stages'][0]
            cfg.env.env_kwargs.update(initial_stage['env_params'])
            torchrl_logger.info(f"[Curriculum] 课程学习启动，初始阶段: {initial_stage['name']}, {curriculum_config}")

        # ============ 2. 创建logger和evaluator ============
        logger = None
        if cfg.logger.backend:
            logger = get_logger(logger_type=cfg.logger.backend, experiment_name=exp_name, logger_name=exp_name, # logger_name在wandb不显示，主要影响本地存储名字
                                wandb_kwargs={"mode": cfg.logger.mode, "config": dict(cfg), "name": exp_name,
                                              "project": cfg.logger.project_name, "group": cfg.logger.group_name, })

        # 定义评估指标使用独立的x轴，避免异步评估的step冲突
        if logger is not None:
            logger.experiment.define_metric("eval_step")
            logger.experiment.define_metric("eval/*", step_metric="eval_step")
            torchrl_logger.info("已配置WandB评估指标使用独立的eval_step轴")

        # 初始化异步评估器
        async_evaluator = AsyncEvaluator(max_workers=cfg.logger.eval_worker)

        # ============ 3. 创建模型（统一逻辑） ============

        if cfg.hif.enabled:
            # Use ResNet-FPN with dual-head actor (action + HIF output)
            torchrl_logger.info(f"创建ResNet-FPN双头模型 (HIF辅助任务集成): {cfg.env.env_id}")
            actor, critic = make_sac_resnet_dual_models(
                env=make_train_environment(cfg, device="cpu"),
                device=train_device,
                hif_decoder_type=cfg.hif.decoder_type
            )
            # HIF输出已集成在actor中，无需单独模块和优化器
        else:
            torchrl_logger.info(f"使用标准SAC模型: {cfg.env.env_id}")
            actor, critic = make_sac_models(env=make_train_environment(cfg, device="cpu"), device=train_device)

        # 加载预训练参数（如果提供）
        if cfg.pretrained_model:
            torchrl_logger.info(f"加载预训练模型: {cfg.pretrained_model}")
            checkpoint = torch.load(cfg.pretrained_path, map_location=train_device, weights_only=False)
            actor.load_state_dict(checkpoint['actor'])
            critic.load_state_dict(checkpoint['critic'])

        # collector_policy直接使用actor（已包含双输出）
        collector_policy = actor
        torchrl_logger.info("Collector使用Actor策略")

        # ============ 4. 创建回放缓冲区和采集器 ============
        replay_buffer = create_replay_buffer(cfg, tmpdir, train_device, cfg.buffer.bucketed, initial_stage)
        collector = create_collector(cfg, collector_policy, train_device)

        torchrl_logger.info(f"创建{collector_device}收集进程 (同步模式)")

        # ============ 5. 创建损失和优化器 ============
        loss_module = SACLoss(actor_network=actor, qvalue_network=critic, num_qvalue_nets=2, delay_actor=False,
                              loss_function=cfg.loss.loss_function, alpha_init=cfg.loss.alpha_init, delay_qvalue=True)

        loss_module.make_value_estimator(gamma=cfg.loss.gamma)

        # 目标网络更新器
        target_net_updater = SoftUpdate(loss_module, eps=cfg.loss.target_update_polyak)

        # 创建优化器（原子模块设计保证参数分离，无需手动过滤）
        # Actor params: 原子模块结构保证不含 decoder，SACLoss 自动提取正确参数
        actor_params = list(loss_module.actor_network_params.flatten_keys().values())
        critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())

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

        # ============ 6. 统一Loss与更新函数（Step A） ============
        if cfg.hif.enabled:
            hif_loss = HIFReconstructionLoss(lambda_tv=cfg.hif.tv_weight, use_tv=cfg.hif.use_tv)
            if curriculum_config:
                initial_weight = float(cfg.hif.weights[initial_stage['name']])  # S1 对应权重
            else:
                initial_weight = float(cfg.hif.initial_weight)
            unified_loss = HIFAssistedSACLoss(actor=actor, sac_loss=loss_module, hif_loss=hif_loss,
                                              mode=LossMode.JOINT, hif_weight=initial_weight)
        else:
            unified_loss = HIFAssistedSACLoss(actor=actor, sac_loss=loss_module, hif_loss=None,
                                              mode=LossMode.SAC_ONLY, hif_weight=0.0)

        update_fn = create_update_fn(unified_loss, optimizer, target_net_updater, cfg, compile_mode, scaler)

        # ============ 6.1 阶段管理器（Step B） ============
        phase_manager = TrainingPhaseManager(cfg)

        # ============ 7. 主训练循环（同步模式） ============
        # Main loop
        collected_frames = 0
        pbar = tqdm.tqdm(total=cfg.collector.total_frames, desc="Training", position=0, leave=True, dynamic_ncols=True)  # 固定在顶部 训练结束后保留 适应终端宽度

        init_random_frames = cfg.collector.init_random_frames
        frames_per_batch = cfg.collector.frames_per_batch
        num_updates = math.ceil(frames_per_batch / cfg.buffer.batch_size * cfg.loss.utd_ratio)

        collector_iter = iter(collector)
        start_time = time.time()
        step = 0

        while collected_frames < cfg.collector.total_frames:
            timeit.printevery(num_prints=1000, total_count=cfg.collector.total_frames, erase=True)

            with timeit("collect"):
                tensordict = next(collector_iter)

            current_frames = tensordict.numel() # 当前采集的帧数
            pbar.update(current_frames)

            # ============ 首次从强监督切换到正则化 ============
            if collected_frames >= init_random_frames and hif_weight == 1.0 and curriculum_state:
                stage_name = curriculum_config['stages'][curriculum_state.stage_idx]['name']
                hif_weight = cfg.hif.weights[stage_name]
                cfg.hif_weight = hif_weight  # 更新cfg以便update函数访问
                joint_loss_module.hif_weight = hif_weight  # 同步更新loss模块权重
                torchrl_logger.info(
                    f"[HIF] 从强监督(1.0)切换到正则化({hif_weight:.3f}), "
                    f"阶段={stage_name}, frames={collected_frames}"
                )

            # HIF训练已集成在update_fn中，无需单独处理

            with timeit("rb - extend"):
                # Add to replay buffer (without HIF labels if they were removed)
                tensordict.pop('pred_ego_hif', None)
                if "next" in tensordict.keys(): tensordict["next"].pop('pred_ego_hif', None)
                tensordict = tensordict.reshape(-1)
                replay_buffer.extend(tensordict)

            collected_frames += current_frames

            # Optimization steps
            with timeit("train"):
                # 统一更新门槛（Step C）
                update_gate = phase_manager.get_update_gate()
                if collected_frames >= update_gate:
                    losses = TensorDict(batch_size=[num_updates])
                    for i in range(num_updates):
                        with timeit("rb - sample"):
                            sampled_tensordict = replay_buffer.sample()  # Sample from replay buffer

                        with timeit("update"):
                            torch.compiler.cudagraph_mark_step_begin()
                            loss_td = update_fn(sampled_tensordict)
                        losses[i] = loss_td.select("loss_actor", "loss_qvalue", "loss_alpha")
                        tensordict.pop('pred_ego_hif', None)
                        if "next" in tensordict.keys(): tensordict["next"].pop('pred_ego_hif', None)
                        replay_buffer.update_tensordict_priority(sampled_tensordict)  # Update priority


            episode_end = (tensordict["next", "done"] if tensordict["next", "done"].any()
                           else tensordict["next", "truncated"])
            episode_end = episode_end.squeeze(-1)
            episode_rewards = tensordict["next", "episode_reward"][episode_end]

            # Logging
            metrics_to_log = {}
            if len(episode_rewards) > 0:
                episode_length = tensordict["next", "step_count"][episode_end]
                completion_ratio = tensordict["next", "completion_ratio"][episode_end]
                metrics_to_log["train/reward"] = episode_rewards.mean().item()
                metrics_to_log["train/episode_length"] = (episode_length.sum() / len(episode_length)).item()
                metrics_to_log["train/completion_ratio"] = completion_ratio.mean().item()
                # steps_95_to_done 与 overlap
                if ("steps_95_to_done" in tensordict["next"].keys()):
                    steps_95_to_done = tensordict["next", "steps_95_to_done"][episode_end]
                    ratio_95_to_done = (steps_95_to_done.float() / (episode_length.float().clamp_min(1)))  # per-episode ratio of tail steps
                    metrics_to_log["train/steps_95_to_done_mean"] = steps_95_to_done.float().mean().item()
                    metrics_to_log["train/ratio_95_to_done_mean"] = ratio_95_to_done.mean().item()
                if ("overlap_count" in tensordict["next"].keys()):
                    overlap_count = tensordict["next", "overlap_count"][episode_end]
                    metrics_to_log["train/overlap_count_mean"] = overlap_count.float().mean().item()

            if collected_frames >= phase_manager.get_update_gate():
                losses = losses.mean()
                metrics_to_log["train/q_loss"] = losses["loss_qvalue"].item()
                metrics_to_log["train/actor_loss"] = losses["loss_actor"].item()
                metrics_to_log["train/alpha_loss"] = losses["loss_alpha"].item()
                metrics_to_log["train/alpha"] = loss_td["alpha"].item()
                metrics_to_log["train/entropy"] = loss_td["entropy"].item()

                # HIF metrics（统一）
                if cfg.hif.enabled and "loss_hif" in loss_td.keys():
                    metrics_to_log["hif/total_loss"] = loss_td["loss_hif"].item()
                    metrics_to_log["hif/effective_weight"] = float(unified_loss.hif_weight)

            # Evaluation
            if is_time_to_evaluate(current_frames, collected_frames, cfg):
                model_path = checkpoint_dir / f"model_s{collected_frames:08d}_eval_pending.pt"  # pending表示等待评估
                torch.save({
                    'actor': actor.state_dict(),
                    'critic': critic.state_dict(),
                    # HIF已集成在actor中，无需单独保存
                    'metadata': {'stage': (curriculum_state.stage_idx if curriculum_state is not None else -1),
                                 'frames': collected_frames}}, model_path)
                # 提交异步评估
                async_evaluator.submit_eval(evaluate_policy_standalone, str(model_path.absolute()), cfg,
                                            collected_frames)
                torchrl_logger.info(f"提交评估任务: collected_frames {collected_frames}")

            evaluate_results = async_evaluator.get_evaluate_results()
            if evaluate_results:
                # 日志落盘
                log_evaluate_results(evaluate_results, checkpoint_dir, logger)

                # 课程切换：仅更新状态，由PhaseManager在下一轮循环顶部执行切换
                if curriculum_state is not None:
                    for result in evaluate_results:
                        phase_manager.update_after_eval(result['metrics'])

            if logger is not None:
                metrics_to_log.update(timeit.todict(prefix="time"))
                metrics_to_log["time/speed"] = pbar.format_dict["rate"]
                # 额外记录分桶占比与课程阶段
                if cfg.buffer.bucketed:
                    sizes = replay_buffer.bucket_sizes
                    for k, v in sizes.items():
                        metrics_to_log[f"buffer/{k}_size"] = v
                # 统一阶段日志
                metrics_to_log.update({"curriculum/stage_idx": float(phase_manager.current_phase)})
                log_metrics(logger, metrics_to_log, collected_frames)

            # Update weights of the inference policy
            collector.update_policy_weights_()

            # 循环顶部执行可能的阶段切换（Step B）
            if phase_manager.should_transition():
                collector, replay_buffer, collector_iter = phase_manager.execute_stage_transition(
                    optimizer=optimizer,
                    collector=collector,
                    replay_buffer=replay_buffer,
                    tmpdir=tmpdir,
                    train_device=train_device,
                    actor=actor,
                    loss_module=unified_loss,
                )

            step += 1

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
