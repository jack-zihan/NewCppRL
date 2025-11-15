"""训练工具：线性日程 + 课程学习 + 分桶回放"""
from __future__ import annotations

import os
import time
import shutil
import tensordict
import torch
import torch.cuda
from dataclasses import dataclass, replace
from typing import Optional, Union, Iterator, Tuple, List, Dict
from tensordict import TensorDict
from omegaconf import DictConfig
from torchrl._utils import logger as torchrl_logger, timeit
from torchrl.envs.transforms import MultiStepTransform
from torchrl.data import LazyMemmapStorage, LazyTensorStorage, TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.collectors import SyncDataCollector, MultiaSyncDataCollector

from rl_new.sac_cont_sy.env_utils import make_train_environment
from rl_new.sac_cont_sy.sac_utils_optimized import set_optimizer_group_lrs
from rl_new.sac_cont_sy.bucketed_replay import BucketedTensorDictPrioritizedReplayBuffer

torch.set_float32_matmul_precision("high")
tensordict.nn.functional_modules._exclude_td_from_pytree().set()


@dataclass(frozen=True)
class Phase:
    """训练阶段配置（PRETRAIN 或课程 STAGE）"""
    type: str  # "PRETRAIN" | "STAGE"
    name: str  # "PRETRAIN" | "S1" | "S2" | "S3"
    env_params: Optional[Dict] = None
    sampling_ratio: Optional[Tuple[float, ...]] = None
    hif_weight: float = 0.0
    update_gate_frames: int = 0


@dataclass
class ScheduleState:
    """日程运行时状态（课程推进跟踪）"""
    idx: int = 0
    pretrain_updates: int = 0
    consec_completion: int = 0
    consec_stable: int = 0
    last_ratio95: Optional[float] = None


def build_training_schedule(cfg: DictConfig) -> List[Phase]:
    """构建线性训练课程表：[PRETRAIN?] + 课程阶段序列"""
    def _env_params(stage_cfg):
        return {
            'reward_field_group_coef': float(stage_cfg.reward_field_group_coef),
            'reward_turning_group_coef': float(stage_cfg.reward_turning_group_coef),
            'reward_overlap_penalty': float(stage_cfg.reward_overlap_penalty),
            'overlap_tolerance': float(stage_cfg.overlap_tolerance),
            'field_scale_enabled': bool(stage_cfg.field_scale_enabled),
            'field_scale_range': (float(stage_cfg.field_scale_range[0]), float(stage_cfg.field_scale_range[1])),}

    def _create_phase(stage_cfg) -> Phase:
        name = str(stage_cfg.name)
        return Phase(type='STAGE', name=name, env_params=_env_params(stage_cfg),
                     sampling_ratio=tuple(stage_cfg.sampling_ratio), hif_weight=float(getattr(cfg.hif.weights, name)),
                     update_gate_frames=int(cfg.collector.init_random_frames),)
    # 课程学习列表
    curriculum_phases = [_create_phase(s) for s in cfg.curriculum.stages] if cfg.curriculum.enabled else \
                        [_create_phase(cfg.curriculum.stages[-1])] # 不开启课程学习则只用最终阶段参数
    # 判断是否预训练
    if cfg.hif.enabled and cfg.hif.pretrain.enabled:
        pretrain_phase = Phase(type='PRETRAIN', name='PRETRAIN', env_params=curriculum_phases[0].env_params,
                               hif_weight=1.0, sampling_ratio=curriculum_phases[0].sampling_ratio,
                               update_gate_frames=int(cfg.hif.pretrain.min_buffer_frames))
        schedule = [pretrain_phase] + curriculum_phases
    else:
        schedule = curriculum_phases
    torchrl_logger.info(f"[Schedule] {[p.name for p in schedule]}")
    return schedule


def maybe_advance_by_eval(state: ScheduleState, phase: Phase, metrics_list: list, cfg: DictConfig,
                          curriculum_enabled: bool) -> Tuple[ScheduleState, bool]:
    """根据评估指标判断是否推进课程（仅 S1→S2, S2→S3）"""
    # 不启动课程学习则不推进
    if not curriculum_enabled: return state, False

    # 只处理匹配当前阶段的评估结果（避免阶段切换后延迟返回的评估污染课程推进）
    valid_metrics = [m for m in metrics_list if m.get('phase_name') == phase.name]
    if not valid_metrics: return state, False  # 所有评估都不属于当前阶段，不推进
    metrics_list = valid_metrics  # 用过滤后的结果继续处理

    # S1阶段：连续N次评估达到最低完成度即可推进
    if phase.name == 'S1':
        consecutive_success_count = state.consec_completion
        for metric in metrics_list:
            consecutive_success_count = consecutive_success_count + 1 \
                if metric['metrics']['eval/completion_ratio'] >= cfg.curriculum.s1_min_completion else 0
        new_state = replace(state, consec_completion=consecutive_success_count)
        return new_state, consecutive_success_count >= cfg.curriculum.s1_consecutive_k

    if phase.name == 'S2': # S2阶段：连续N次评估完成度和稳定性均达标即可推进
        consecutive_stable_count, previous_ratio95 = state.consec_stable, state.last_ratio95
        for metric in metrics_list:
            if metric['metrics']['eval/completion_ratio'] >= cfg.curriculum.s2_min_completion and (previous_ratio95 is not None):
                relative_change = abs(metric['metrics']['eval/ratio_95_to_done_mean'] - previous_ratio95) / max(previous_ratio95, 1e-6)
                stable = relative_change< float(cfg.curriculum.s2s3_threshold)
            else:
                stable = False
            consecutive_stable_count = consecutive_stable_count + 1 if stable else 0
        new_state = replace(state, consec_stable=consecutive_stable_count, last_ratio95=metric['metrics']['eval/ratio_95_to_done_mean'])
        return new_state, consecutive_stable_count>= int(cfg.curriculum.s2_consecutive_k)
    return state, False # 预训练和S3阶段不推进，但其实只有课程阶段，因为预训练不会评估

def maybe_advance_by_updates(state: ScheduleState, phase: Phase, cfg: DictConfig) -> Tuple[ScheduleState, bool]:
    """根据更新步数判断 PRETRAIN 是否推进"""
    if phase.type != 'PRETRAIN':
        return state, False
    return state, (state.pretrain_updates >= int(cfg.hif.pretrain.max_updates))

def apply_phase(phase: Phase, cfg: DictConfig, actor_model: torch.nn.Module, device, optimizer,
                collector, replay_buffer):
    """应用阶段配置：环境参数、采样比例、学习率、重建采集器 Returns: (new_collector, new_iter, replay_buffer)"""

    if phase.type == 'STAGE' and phase.env_params: # 更新环境参数
        cfg.env.env_kwargs.update(phase.env_params)

    if replay_buffer is not None and cfg.buffer.bucketed: # 如果分桶更新回放缓冲区采样比例
        replay_buffer.set_sampling_ratio(phase.sampling_ratio)
        replay_buffer.reset_buckets()

    if phase.type == 'PRETRAIN': # 更新学习率
        set_optimizer_group_lrs(optimizer, all_groups_lr=float(cfg.hif.pretrain.actor_lr))
    else:
        set_optimizer_group_lrs(optimizer, actor_lr=float(cfg.optim.lr_actor), critic_lr=float(cfg.optim.lr_critic),
                                           alpha_lr=float(cfg.optim.lr_alpha))

    if collector is not None: collector.shutdown(timeout=3.0); time.sleep(2.0)
    new_collector = create_collector(cfg, actor_model, device) # 重新创建采集器

    torchrl_logger.info(f"[Phase Applied] {phase.name} | "f"Env params: {bool(phase.env_params)} | "f"Sampling: {phase.sampling_ratio}")
    return new_collector, iter(new_collector), replay_buffer


def create_collector(cfg, actor_model, device):
    """创建并行数据采集器（MultiaSyncDataCollector）"""
    collector = MultiaSyncDataCollector(
        create_env_fn=[lambda: make_train_environment(cfg, device='cpu') for _ in range(cfg.collector.num_collectors)],
        policy=actor_model, frames_per_batch=cfg.collector.frames_per_batch, total_frames=cfg.collector.total_frames,
        device=None, policy_device=device, storing_device='cpu', env_device='cpu', max_frames_per_traj=-1)
    collector.set_seed(cfg.seed)
    return collector


def create_replay_buffer(cfg, tmpdir, device, bucketed=False, initial_stage=None):
    """创建回放缓冲区（含 n-step 和设备转换，增加了混合精度支持，帮助有效降低缓存，快速验证算法"""
    storage = LazyMemmapStorage(max_size=cfg.buffer.buffer_size,scratch_dir=tmpdir)

    if bucketed: # 使用分桶优先级回放缓冲区
        buffer = BucketedTensorDictPrioritizedReplayBuffer(
            alpha=cfg.buffer.alpha, beta=cfg.buffer.beta, prefetch=0, # 分桶优先级预取禁用，避免提前采样不匹配的batch问题
            batch_size=cfg.buffer.batch_size, pin_memory=cfg.buffer.pin_memory, storage=storage,
            success_threshold=cfg.buffer.success_threshold, near_end_threshold=cfg.buffer.near_end_threshold,)
        if initial_stage: buffer.set_sampling_ratio(initial_stage['sampling_ratio']) # 设置分桶采样比例
    else: # 使用标准优先级回放缓冲区
        buffer = TensorDictPrioritizedReplayBuffer(
            storage=storage,alpha=cfg.buffer.alpha, beta=cfg.buffer.beta,
            prefetch=cfg.buffer.prefetch,
            batch_size=cfg.buffer.batch_size, pin_memory=cfg.buffer.pin_memory,)

    if cfg.loss.n_steps > 1: # 添加多步转换
        buffer.append_transform(MultiStepTransform(n_steps=cfg.loss.n_steps, gamma=cfg.loss.gamma,
                                                   reward_keys=["reward"], done_keys=["done", "truncated", "terminated"]))
    # device_transform = to_device_with_bf16_for_floats if cfg.training.use_amp else to_device
    # buffer.append_transform(lambda td: device_transform(td, device)) # 设备转换

    buffer.append_transform(lambda td: td.to(device))  # 设备转换

    torchrl_logger.info(f"[Buffer] Created - {'Bucketed' if bucketed else 'Standard'} PRB | "
                        f"Size: {cfg.buffer.buffer_size} | Batch: {cfg.buffer.batch_size}")
    return buffer


# def to_device(td: TensorDict, device: torch.device) -> TensorDict:
#     return td.to(device)
#
# def to_device_with_bf16_for_floats(td: TensorDict, device: torch.device) -> TensorDict:
#     """
#     - 对指定的大浮点键：CPU → GPU + 转为 bfloat16
#     - 其他键：正常 .to(device)（如果已经在同一 device，会是轻量操作）
#     """
#     float_keys = [("observation",), ("next", "observation"), ("label_ego_hif",), ("next", "label_ego_hif"),]
#
#     for key in float_keys: # 1) 先对这些“大头浮点键”做一步到位的 device + bf16
#         if td.get(key, None) is not None:
#             x = td.get(key)
#             if torch.is_floating_point(x):
#                 td.set(key, x.to(device=device, dtype=torch.bfloat16))
#
#     # 2) 再把整个 TensorDict 搬到 device：
#     #    - 对还在 CPU 的键（非浮点 / 未列入 float_keys 的小浮点）做 .to(device)
#     #    - 对已经在 device 上的键（上面刚 to(device, bf16) 过的）因为 device 相同，基本是 no-op
#     td = td.to(device)
#
#     return td