"""
SAC 课程学习带HIF正则化和预训练脚本
"""
from __future__ import annotations

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
import numpy as np
from pathlib import Path
from threading import RLock
from dataclasses import dataclass

from typing import Optional, Union, Iterator

from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from torchrl._utils import compile_with_warmup, logger as torchrl_logger, timeit
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import SoftUpdate, SACLoss, group_optimizers
from torchrl.data import LazyMemmapStorage, LazyTensorStorage, TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.collectors import SyncDataCollector, MultiaSyncDataCollector
from torchrl.record.loggers import get_logger

from rl_new.sac_cont_sy.model_utils import make_sac_models, make_sac_resnet_dual_models
from rl_new.sac_cont_sy.sac_utils import (setup_devices, create_update_fn, flatten, get_actor_actions,
                                          set_optimizer_group_lrs, LossMode, HIFAssistedSACLoss,
                                          generate_exp_name, evaluate_policy_parallel, CheckpointManager, log_metrics,
                                          evaluate_policy, evaluate_policy_parallel, setup_torch_cache,
                                          evaluate_policy_standalone, log_evaluate_results, is_time_to_evaluate)
from torchrl_utils.model.resnet_fpn_dual import HIFReconstructionLoss
from rl_new.sac_cont_sy.env_utils import make_train_environment
from rl_new.sac_cont_sy.async_evaluator import AsyncEvaluator
from rl_new.sac_cont_sy.bucketed_replay import BucketedTensorDictPrioritizedReplayBuffer
from rl_new.sac_cont_sy.train_utils import CurriculumState, create_replay_buffer, create_collector, \
    update_curriculum_state, execute_stage_transition, load_curriculum_config, TrainingPhaseManager

torch.set_float32_matmul_precision("high")
tensordict.nn.functional_modules._exclude_td_from_pytree().set()


@hydra.main(version_base="1.1", config_path="", config_name="config-sync-server-hif-pretrain")
def main(cfg: DictConfig):
    # 清空临时数据路径
    temp_dir = cfg.buffer.temp_dir
    if temp_dir:
        if temp_dir.startswith('~'): temp_dir = os.path.expanduser(temp_dir)  # ~转换为实际路径
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir);  # 清空路径
        os.makedirs(temp_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=temp_dir) as tmpdir:
        # ============ 1. 创建实验目录和基础设置 ============
        # 生成实验名称
        exp_name = generate_exp_name(cfg.logger.model_name, cfg.logger.exp_name)

        # 设置多线程安全的tqdm锁
        tqdm.tqdm.set_lock(RLock())

        # 设置采集和训练设备，目前使用单gpu实现
        collector_device = train_device = torch.device(cfg.training.device)
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

        # ============ 2. 创建Logger & 异步评估器 ============
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

        # 设置异步评估器
        async_evaluator = AsyncEvaluator(max_workers=cfg.logger.eval_worker)

        # ============ 3. 创建模型 ============
        if cfg.hif.enabled:
            actor, critic = make_sac_resnet_dual_models(env=make_train_environment(cfg, device="cpu"),
                                                        device=train_device, hif_decoder_type=cfg.hif.decoder_type)
        else:
            actor, critic = make_sac_models(env=make_train_environment(cfg, device="cpu"), device=train_device)

        # 加载预训练参数（键名兼容：优先使用 pretrained_path，其次 pretrained_model）
        _pretrained_path = getattr(cfg, 'pretrained_path', None) or getattr(cfg, 'pretrained_model', None)
        if _pretrained_path:
            torchrl_logger.info(f"加载预训练模型: {_pretrained_path}")
            checkpoint = torch.load(_pretrained_path, map_location=train_device, weights_only=False)
            actor.load_state_dict(checkpoint['actor']); critic.load_state_dict(checkpoint['critic'])

        # ============ 4. 创建回放缓冲区和采集器 ============
        replay_buffer = create_replay_buffer(cfg, tmpdir, train_device, cfg.buffer.bucketed, initial_stage)
        collector = create_collector(cfg, actor, train_device)

        torchrl_logger.info(f"创建{collector_device}收集进程 (预训练课程学习脚本)")

        # ============ 5. 创建损失和优化器 ============
        loss_module = SACLoss(actor_network=actor, qvalue_network=critic, num_qvalue_nets=2, delay_actor=False,
                              loss_function=cfg.loss.loss_function, alpha_init=cfg.loss.alpha_init, delay_qvalue=True)
        loss_module.make_value_estimator(gamma=cfg.loss.gamma)
        target_net_updater = SoftUpdate(loss_module, eps=cfg.loss.target_update_polyak) # 目标网络更新器使用软更新

        # 获取参数并创建优化器
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
        scaler = torch.amp.GradScaler('cuda') if (cfg.training.use_amp and torch.cuda.is_available()) else None

        # ============ 5. 统一Loss & 更新函数（Layer 1） ============
        hif_loss = HIFReconstructionLoss(lambda_tv=cfg.hif.tv_weight, use_tv=cfg.hif.use_tv) if cfg.hif.enabled else None

        if cfg.hif.enabled and cfg.hif.pretrain.enabled:
            initial_mode = LossMode.PRETRAIN
            initial_weight = 1.0
        elif cfg.hif.enabled:
            initial_mode = LossMode.JOINT
            if curriculum_config:
                initial_weight = float(cfg.hif.weights[initial_stage['name']])
            else:
                initial_weight = float(cfg.hif.initial_weight)
        else:
            initial_mode = LossMode.SAC_ONLY
            initial_weight = 0.0

        unified_loss = HIFAssistedSACLoss(actor=actor, sac_loss=loss_module, hif_loss=hif_loss,
                                          mode=initial_mode, hif_weight=initial_weight)
        update_fn = create_update_fn(unified_loss, optimizer, target_net_updater, cfg, compile_mode, scaler)

        # 阶段管理器（Layer 2）
        phase_manager = TrainingPhaseManager(cfg)

        # 预训练阶段学习率（只在初始化时设置一次）
        if initial_mode == LossMode.PRETRAIN:
            set_optimizer_group_lrs(optimizer, all_groups_lr=float(cfg.hif.pretrain.actor_lr))
            torchrl_logger.info(f"[Init] PRETRAIN lr set: {cfg.hif.pretrain.actor_lr}")

        # ============ 7) 主训练循环 ============
        collected_frames = 0
        pbar = tqdm.tqdm(total=cfg.collector.total_frames, desc="Training", position=0, leave=True, dynamic_ncols=True)

        init_random_frames = cfg.collector.init_random_frames
        frames_per_batch = cfg.collector.frames_per_batch
        num_updates = math.ceil(frames_per_batch / cfg.buffer.batch_size * cfg.loss.utd_ratio)

        collector_iter = iter(collector)
        start_time = time.time()

        while collected_frames < cfg.collector.total_frames:
            timeit.printevery(num_prints=1000, total_count=cfg.collector.total_frames, erase=True)

            # 阶段切换（在循环开始判断）
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

            with timeit("collect"):
                tensordict = next(collector_iter)

            current_frames = tensordict.numel()
            pbar.update(current_frames)

            with timeit("rb - extend"):
                tensordict.pop('pred_ego_hif', None)
                if "next" in tensordict.keys():
                    tensordict["next"].pop('pred_ego_hif', None)
                replay_buffer.extend(tensordict.reshape(-1))

            collected_frames += current_frames
            metrics_to_log = {}

            # 统一的更新门槛（补强点1）
            update_gate = phase_manager.get_update_gate()
            if collected_frames >= update_gate:
                with timeit("train"):
                    losses = TensorDict(batch_size=[num_updates])
                    for i in range(num_updates):
                        with timeit("rb - sample"):
                            sampled_td = replay_buffer.sample()
                        with timeit("update"):
                            torch.compiler.cudagraph_mark_step_begin()
                            loss_td = update_fn(sampled_td)
                        losses[i] = loss_td.select("loss_actor", "loss_qvalue", "loss_alpha")
                        replay_buffer.update_tensordict_priority(sampled_td)
                        if unified_loss.mode == LossMode.PRETRAIN:
                            phase_manager.increment_pretrain_updates()

                # 统计与日志
                episode_end = (tensordict["next", "done"] if tensordict["next", "done"].any() else tensordict["next", "truncated"]).squeeze(-1)
                if episode_end.any():
                    episode_rewards = tensordict["next", "episode_reward"][episode_end]
                    episode_length = tensordict["next", "step_count"][episode_end]
                    completion_ratio = tensordict["next", "completion_ratio"][episode_end]
                    metrics_to_log.update({
                        "train/reward": episode_rewards.mean().item(),
                        "train/episode_length": (episode_length.sum() / episode_length.numel()).item(),
                        "train/completion_ratio": completion_ratio.mean().item(),
                    })

                # 损失日志
                mean_losses = losses.mean()
                metrics_to_log.update({
                    "train/q_loss": mean_losses["loss_qvalue"].item(),
                    "train/actor_loss": mean_losses["loss_actor"].item(),
                    "train/alpha_loss": mean_losses["loss_alpha"].item(),
                    **({"train/alpha": loss_td["alpha"].item()} if "alpha" in loss_td.keys() else {}),
                    **({"train/entropy": loss_td["entropy"].item()} if "entropy" in loss_td.keys() else {}),
                })
                if cfg.hif.enabled and "loss_hif" in loss_td.keys():
                    metrics_to_log["train/hif_total_loss"] = loss_td["loss_hif"].item()
                    metrics_to_log["train/hif_weight"] = float(unified_loss.hif_weight)

                if is_time_to_evaluate(current_frames, collected_frames, cfg):
                    model_path = checkpoint_dir / f"model_s{collected_frames:08d}_eval_pending.pt"
                    # 保存可直接用于评估的模块列表，避免评估端改动
                    torch.save(torch.nn.ModuleList([actor, critic]), model_path)
                    async_evaluator.submit_eval(evaluate_policy_standalone, str(model_path.absolute()), cfg, collected_frames)
                    torchrl_logger.info(f"提交评估任务: {collected_frames}")

                eval_results = async_evaluator.get_evaluate_results()
                if eval_results:
                    log_evaluate_results(eval_results, checkpoint_dir, logger)
                    # 课程学习：仅更新状态，由PhaseManager在下轮切换
                    for result in eval_results:
                        phase_manager.update_after_eval(result['metrics'])

                if logger is not None:
                    metrics_to_log.update(timeit.todict(prefix="time"))
                    metrics_to_log["time/speed"] = pbar.format_dict["rate"]
                    if cfg.buffer.bucketed:
                        sizes = replay_buffer.bucket_sizes
                        for k, v in sizes.items():
                            metrics_to_log[f"buffer/{k}_size"] = v
                    # 统一阶段日志
                    metrics_to_log.update({"train/stage_idx": float(phase_manager.current_phase)})
                    log_metrics(logger, metrics_to_log, collected_frames)

                collector.update_policy_weights_()

        torchrl_logger.info(f"Training finished in {time.time() - start_time:.2f}s")
        collector.shutdown()
        remaining = async_evaluator.shutdown(wait=True)
        if remaining:
            log_evaluate_results(remaining, checkpoint_dir, logger)
        if logger is not None:
            logger.experiment.finish()
            torchrl_logger.info("WandB运行已完成")
        time.sleep(2)


if __name__ == "__main__":
    main()
