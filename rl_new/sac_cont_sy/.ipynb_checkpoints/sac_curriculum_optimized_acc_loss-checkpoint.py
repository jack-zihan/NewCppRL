"""
优化版SAC课程学习训练脚本 - 集成所有Less is More优化
现在其实进一步分析出了课程学习的本质，设计了优雅的简洁方案：不论是预训练、课程学习过程，主要影响的在于环境参数的切换和优化目标的切换，
因此用统一的阶段概念描述每一轮的环境参数和优化目标，形成一个阶段课程表Schedule，按照课程表的环境参数、优化目标往前走即可。Less is More!
sac_curricuum.py 课程学习，但是课程hif切换的逻辑错误 -> sac_curriculum_optimized.py 课程逻辑正确，但非常混乱复杂 ->
sac_curriculum_with_pretrain.py 逻辑正确，并且加入了预训练，优化了整体的混乱逻辑 -> sac_utils_optimized_acc_loss.py Less is More的完成了累计梯度的问题
"""
import os
import sys
import time
import math
import copy
import shutil
import tempfile
import warnings
import hydra
import tqdm
import torch
import numpy as np
import tensordict
from pathlib import Path
from threading import RLock
from dataclasses import replace

from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from torchrl._utils import compile_with_warmup, logger as torchrl_logger, timeit
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import SoftUpdate, SACLoss, group_optimizers
from torchrl.record.loggers import get_logger

# 导入优化后的组件（梯度累积版本）
from rl_new.sac_cont_sy.sac_utils_optimized import (HIFAssistedSACLoss, create_update_fn, create_grad_accum_update_fn,
                                                    evaluate_policy_standalone, generate_exp_name, log_metrics,
                                                    set_optimizer_group_lrs, setup_torch_cache,
                                                    log_evaluate_results, is_time_to_evaluate)
from rl_new.sac_cont_sy.train_utils_optimized import (create_replay_buffer, create_collector,
                                                      build_training_schedule, ScheduleState, Phase,
                                                      maybe_advance_by_eval, maybe_advance_by_updates, apply_phase)

from rl_new.sac_cont_sy.model_utils import make_sac_models, make_sac_resnet_dual_models, make_sac_cnn_dual_models
from rl_new.sac_cont_sy.env_utils import make_train_environment
from rl_new.sac_cont_sy.async_evaluator import AsyncEvaluator
from torchrl_utils.model.resnet_fpn_dual import HIFReconstructionLoss

torch.set_float32_matmul_precision("high")
tensordict.nn.functional_modules._exclude_td_from_pytree().set()


@hydra.main(version_base="1.1", config_path="", config_name="config-sync-server-hif-v2") # "config-sync-server-hif-v2" "config-v2-refine"
def main(cfg: DictConfig):
    # ============ 1. 临时实验目录和基础设置 ============
    temp_dir = cfg.buffer.temp_dir
    if temp_dir:
        if temp_dir.startswith('~'): temp_dir = os.path.expanduser(temp_dir)  # ~转换为实际路径
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)  # 清空路径
        os.makedirs(temp_dir, exist_ok=True)

    with (tempfile.TemporaryDirectory(dir=temp_dir) as tmpdir):
        # 生成实验名称
        exp_name = generate_exp_name(cfg.logger.model_name, cfg.logger.exp_name)

        # 设置多线程安全的tqdm锁
        tqdm.tqdm.set_lock(RLock())

        # 设置采集和训练设备，目前使用单gpu实现
        collector_device = train_device = torch.device(cfg.training.device)
        torchrl_logger.info(f"训练设备: {train_device}, 收集设备: {collector_device}")

        # 设置随机种子
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

        # 设置checkpoint目录
        checkpoint_dir = Path.cwd() / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torchrl_logger.info(f"Checkpoint将保存到: {checkpoint_dir}")

        # 确定编译模式
        compile_mode = None
        if cfg.compile.enable:
            if cfg.in_server: setup_torch_cache()
            compile_mode = (cfg.compile.mode or ("default" if cfg.compile.cudagraphs else "reduce-overhead"))

        # 创建GradScaler（如果使用混合精度）
        scaler = torch.amp.GradScaler('cuda') if (cfg.training.use_amp and torch.cuda.is_available()) else None

        # ============ 1.1 构建训练日程（Schedule） ============
        schedule = build_training_schedule(cfg)
        state = ScheduleState(idx=0)
        # 将首阶段的环境参数注入（PRETRAIN 已复制下一阶段参数）
        if schedule[0].env_params:
            cfg.env.env_kwargs.update(schedule[0].env_params)
            # torchrl_logger.info(f"学习环境数据: {cfg.env.env_kwargs.map_dir} , 障碍 {cfg.env.env_kwargs.num_obstacles_range}")

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
        if cfg.model.architecture == "resnet":
            # ResNet-FPN双头模型：始终使用dual-head结构，是否启用HIF由enable_hif控制
            actor, critic = make_sac_resnet_dual_models(
                env=make_train_environment(cfg, device="cpu"),
                device=train_device,
                enable_hif=cfg.hif.enabled,
                hif_decoder_type=cfg.hif.decoder_type,
                backbone_type=cfg.model.backbone,
            )
        elif cfg.model.architecture == "cnn":
            # CNN双头模型：与ResNet-FPN保持同样接口，enable_hif控制是否输出pred_ego_hif
            actor, critic = make_sac_cnn_dual_models(
                env=make_train_environment(cfg, device="cpu"),
                device=train_device,
                enable_hif=cfg.hif.enabled,
            )
        else:
            raise ValueError(f"Unsupported model.architecture='{cfg.model.architecture}', "
                             f"expected 'resnet' or 'cnn'.")

        # 加载预训练模型
        if cfg.pretrained_model:
            torchrl_logger.info(f"加载预训练模型: {cfg.pretrained_model}")
            checkpoint = torch.load(cfg.pretrained_model, map_location=train_device, weights_only=False)
            actor.load_state_dict(checkpoint['actor']);
            critic.load_state_dict(checkpoint['critic'])

        # ============ 4. 创建回放缓冲区和采集器 ============
        init_stage_hint = {'sampling_ratio': schedule[0].sampling_ratio} if cfg.buffer.bucketed else None
        replay_buffer = create_replay_buffer(cfg, tmpdir, train_device, cfg.buffer.bucketed, init_stage_hint)
        collector = create_collector(cfg, actor, train_device)
        torchrl_logger.info(f"创建{collector_device}收集进程 (预训练课程学习脚本)")

        # ============ 5. 创建损失、优化器和优化函数 ============
        # 创建SAC Loss
        loss_module = SACLoss(actor_network=actor, qvalue_network=critic, num_qvalue_nets=2, delay_actor=False,
                              loss_function=cfg.loss.loss_function, alpha_init=cfg.loss.alpha_init, delay_qvalue=True)
        loss_module.make_value_estimator(gamma=cfg.loss.gamma)
        target_net_updater = SoftUpdate(loss_module, eps=cfg.loss.target_update_polyak)  # 目标网络更新器使用软更新

        # 获取参数并创建优化器
        actor_params = list(loss_module.actor_network_params.flatten_keys().values())
        critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())

        optimizer_actor = torch.optim.AdamW(
            actor_params, lr=cfg.optim.lr_actor, weight_decay=cfg.optim.weight_decay_actor, eps=cfg.optim.eps_actor)
        optimizer_critic = torch.optim.AdamW(
            critic_params, lr=cfg.optim.lr_critic, weight_decay=cfg.optim.weight_decay_critic, eps=cfg.optim.eps_critic)
        optimizer_alpha = torch.optim.AdamW(
            [loss_module.log_alpha], lr=cfg.optim.lr_alpha, weight_decay=cfg.optim.weight_decay_alpha)

        # 合并优化器
        optimizer = group_optimizers(optimizer_actor, optimizer_critic, optimizer_alpha)
        del optimizer_actor, optimizer_critic, optimizer_alpha

        # 创建hif Loss
        hif_loss = HIFReconstructionLoss(lambda_tv=cfg.hif.tv_weight,
                                         use_tv=cfg.hif.use_tv) if cfg.hif.enabled else None

        # 绑定SACLoss和HIFLoss为HIFAssistedSACLoss（以schedule驱动阶段/权重）
        phase_provider = lambda: 0 if schedule[state.idx].type == 'PRETRAIN' else 1
        weight_provider = lambda: schedule[state.idx].hif_weight
        unified_loss = HIFAssistedSACLoss(actor=actor, sac_loss=loss_module, hif_loss=hif_loss,cfg=cfg,
                                          phase_provider=phase_provider, weight_provider=weight_provider)
        # 创建梯度累积更新函数
        update_fn = create_grad_accum_update_fn(unified_loss, optimizer, target_net_updater, cfg, compile_mode, scaler)

        # 根据首阶段设置学习率（PRETRAIN 统一lr，否则恢复三组lr）
        if schedule[0].type == 'PRETRAIN':
            set_optimizer_group_lrs(optimizer, all_groups_lr=float(cfg.hif.pretrain.actor_lr))
        else:
            set_optimizer_group_lrs(optimizer, actor_lr=float(cfg.optim.lr_actor),
                                    critic_lr=float(cfg.optim.lr_critic), alpha_lr=float(cfg.optim.lr_alpha))

        torchrl_logger.info(f"[Training] Starting - Phase: {schedule[state.idx].name}, "
                            f"Curriculum: {'Enabled' if cfg.curriculum.enabled else 'Disabled'}")

        # ============ 6. 主训练循环 ============
        collected_frames = 0 # 帧参数部分
        frames_per_batch = cfg.collector.frames_per_batch
        num_updates = math.ceil(frames_per_batch / cfg.buffer.batch_size * cfg.loss.utd_ratio) * cfg.training.grad_accum_steps

        collector_iter = iter(collector) # 迭代器部分
        pbar = tqdm.tqdm(total=cfg.collector.total_frames, desc="Training", position=0, leave=True, dynamic_ncols=True)
        start_time = time.time()

        while collected_frames < cfg.collector.total_frames:
            timeit.printevery(num_prints=1000, total_count=cfg.collector.total_frames, erase=True)
            # ========== 阶段转换检查（基于Schedule） ==========
            should_transition = False

            # 预训练切换判断
            if schedule[state.idx].type == 'PRETRAIN':
                state, adv_upd = maybe_advance_by_updates(state, schedule[state.idx], cfg)
                if adv_upd: should_transition = True

            # 课程阶段切换判断（终态由位置决定）
            eval_results = async_evaluator.get_evaluate_results()
            if eval_results:
                log_evaluate_results(eval_results, checkpoint_dir, logger)
                state, adv_eval = maybe_advance_by_eval(state, schedule, eval_results)
                if adv_eval: should_transition = True

            # 执行phase切换
            if should_transition and state.idx + 1 < len(schedule):
                old_idx = state.idx
                state = replace(state, idx=state.idx + 1, pass_streak=0, prev_metric=None)  # 切换到下一阶段状态
                schedule[state.idx] = replace(schedule[state.idx],
                                              update_gate_frames=collected_frames + cfg.collector.init_random_frames) # 确定该阶段绝对帧数门控
                collector, collector_iter, replay_buffer = apply_phase(schedule[state.idx], cfg, actor, train_device,
                                                                       optimizer, collector, replay_buffer)
                torchrl_logger.info(f"[Transition] {schedule[old_idx].name} → {schedule[state.idx].name}")

            # ========== 数据收集 ==========
            with timeit("collect"):
                tensordict = next(collector_iter)

            current_frames = tensordict.numel()
            collected_frames += current_frames
            pbar.update(current_frames)

            # ========== 回放缓冲区更新 ==========
            with timeit("rb - extend"):
                tensordict.pop('pred_ego_hif', None) # 清理pred_ego_hif（HIF预测结果，不需要存储）
                if "next" in tensordict.keys():
                    tensordict["next"].pop('pred_ego_hif', None)
                replay_buffer.extend(tensordict.reshape(-1))

            # ========== 模型更新 ==========
            if collected_frames >= int(schedule[state.idx].update_gate_frames):
                with timeit("train"):
                    losses = TensorDict(batch_size=[num_updates])
                    for i in range(num_updates):
                        with timeit("rb - sample"):
                            sampled_td = replay_buffer.sample()
                        with timeit("update"):
                            torch.compiler.cudagraph_mark_step_begin()
                            loss_td, step_taken = update_fn(sampled_td)
                        losses[i] = loss_td.select("loss_actor", "loss_qvalue", "loss_alpha").detach().to("cpu")
                        priority_td = sampled_td.select("index", replay_buffer.priority_key).detach().to("cpu")
                        replay_buffer.update_tensordict_priority(priority_td)
                        # replay_buffer.update_tensordict_priority(sampled_td)
                        if schedule[state.idx].type == 'PRETRAIN' and step_taken:
                            state = replace(state, pretrain_updates=state.pretrain_updates + 1)

                # Logging
                metrics_to_log = {}
                # episode指标
                episode_end = (tensordict["next", "done"] if tensordict["next", "done"].any() else tensordict["next", "truncated"]).squeeze(-1)
                if episode_end.any():
                    metrics_to_log["train/reward"] = tensordict["next", "episode_reward"][episode_end].mean().item()
                    metrics_to_log["train/episode_length"] = float(tensordict["next", "step_count"][episode_end].float().mean().item())
                    metrics_to_log["train/completion_ratio"] = tensordict["next", "completion_ratio"][episode_end].mean().item()

                # 损失指标
                losses = losses.mean()
                metrics_to_log["train/q_loss"] = losses["loss_qvalue"].item()
                metrics_to_log["train/actor_loss"] = losses["loss_actor"].item()
                metrics_to_log["train/alpha_loss"] = losses["loss_alpha"].item()
                metrics_to_log["train/alpha"] = loss_td["alpha"].item()
                metrics_to_log["train/entropy"] = loss_td["entropy"].item()

                if "loss_hif" in loss_td.keys():
                    metrics_to_log["train/hif_total_loss"] = loss_td["loss_hif"].item()
                    metrics_to_log["train/hif_weight"] = unified_loss.current_hif_weight

                # 分桶缓冲池与课程训练指标
                metrics_to_log["train/stage_idx"] = float(state.idx)
                metrics_to_log["train/stage_name"] = schedule[state.idx].name
                if cfg.buffer.bucketed:
                    for k, v in replay_buffer.bucket_sizes.items():
                        metrics_to_log[f"buffer/{k}_size"] = v

                # 提交异步评估任务
                if is_time_to_evaluate(current_frames, collected_frames, cfg):
                    model_path = (checkpoint_dir / f"model_step{collected_frames:08d}_eval_pending.pt")
                    torch.save({'actor': actor.state_dict(),'critic': critic.state_dict()}, model_path) # 保存为dict格式（与评估端一致）
                    async_evaluator.submit_eval(evaluate_policy_standalone,str(model_path.absolute()),copy.deepcopy(cfg),collected_frames,schedule[state.idx].name)
                    torchrl_logger.info(f"提交评估任务: {collected_frames} (阶段: {schedule[state.idx].name})")

                # ========== 日志记录 ==========
                if logger is not None:
                    metrics_to_log.update(timeit.todict(prefix="time"))
                    metrics_to_log["time/speed"] = pbar.format_dict["rate"]
                    log_metrics(logger, metrics_to_log, collected_frames)

                collector.update_policy_weights_() # collector的policy权重同步
                del sampled_td, loss_td # 清理内存
            del tensordict # 清理内存
        # ============ 9. 训练结束 ============
        torchrl_logger.info(f"Training took {time.time() - start_time:.2f} seconds to finish")
        pbar.close()
        collector.shutdown()

        # 等待所有评估完成
        remaining_results = async_evaluator.shutdown(wait=True)
        torchrl_logger.info(f"处理剩余的 {len(remaining_results)} 个评估结果")
        if remaining_results: log_evaluate_results(remaining_results, checkpoint_dir, logger)

        if logger is not None: # 确保WandB数据完整保存
            logger.experiment.finish()
            torchrl_logger.info("WandB运行已完成")
        time.sleep(2)

if __name__ == "__main__":
    main()
