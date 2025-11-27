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
    name: str  # "PRETRAIN" | "S1" | "S2" | "S3" | ...
    env_params: Optional[Dict] = None
    sampling_ratio: Optional[Tuple[float, ...]] = None
    hif_weight: float = 0.0
    update_gate_frames: int = 0
    # 课程切换条件（终态由位置判断，非Phase属性）
    min_completion: float = 0.9            # 完成率阈值
    consecutive_k: int = 3                  # 连续达标次数
    stability_tolerance: Optional[float] = None  # 稳定性容差（None=不检查）


@dataclass
class ScheduleState:
    """日程运行时状态（课程推进跟踪）"""
    idx: int = 0
    pretrain_updates: int = 0
    pass_streak: int = 0                   # 统一：连续达标次数
    prev_metric: Optional[float] = None    # 统一：上次评估值（用于稳定性检查）


def build_training_schedule(cfg: DictConfig) -> List[Phase]:
    """构建线性训练课程表：[PRETRAIN?] + 课程阶段序列"""
    def _env_params(stage_cfg):
        env_params = {
            'reward_field_group_coef': float(stage_cfg.reward_field_group_coef),
            'reward_turning_group_coef': float(stage_cfg.reward_turning_group_coef),
            'reward_overlap_penalty': float(stage_cfg.reward_overlap_penalty),
            'overlap_tolerance': float(stage_cfg.overlap_tolerance),
            'field_scale_enabled': bool(stage_cfg.field_scale_enabled),
            'field_scale_range': (float(stage_cfg.field_scale_range[0]), float(stage_cfg.field_scale_range[1])),
        }
        # 可选课程参数：仅当在stage配置中显式给出时才覆盖环境默认值
        #   - 保持向后兼容：旧配置未声明时使用 EnvironmentConfig 默认值
        if "reward_weed_removal" in stage_cfg:
            env_params["reward_weed_removal"] = float(stage_cfg.reward_weed_removal)
        if "reward_base_penalty" in stage_cfg:
            env_params["reward_base_penalty"] = float(stage_cfg.reward_base_penalty)
        if "reward_completion_bonus" in stage_cfg:
            env_params["reward_completion_bonus"] = float(stage_cfg.reward_completion_bonus)
        if "reward_apf" in stage_cfg:
            env_params["reward_apf"] = float(stage_cfg.reward_apf)

        if "num_obstacles_range" in stage_cfg:
            env_params["num_obstacles_range"] = (int(stage_cfg.num_obstacles_range[0]), int(stage_cfg.num_obstacles_range[1]))
        if "weed_count" in stage_cfg:
            env_params["weed_count"] = int(stage_cfg.weed_count)
        return env_params

    def _create_phase(stage_cfg) -> Phase:
        """创建阶段配置（终态判断由位置决定，不在此处理）"""
        name = str(stage_cfg.name)
        return Phase(
            type='STAGE', name=name,
            env_params=_env_params(stage_cfg),
            sampling_ratio=tuple(stage_cfg.sampling_ratio),
            hif_weight=float(getattr(cfg.hif.weights, name, 0.0)),
            update_gate_frames=int(cfg.collector.init_random_frames),
            min_completion=float(stage_cfg.min_completion),      # 必须配置，缺失则报错
            consecutive_k=int(stage_cfg.consecutive_k),          # 必须配置，缺失则报错
            stability_tolerance=getattr(stage_cfg, 'stability_tolerance', None),  # 可选配置
        )

    # 课程学习列表
    stages = list(cfg.curriculum.stages) if cfg.curriculum.enabled else [cfg.curriculum.stages[-1]]
    curriculum_phases = [_create_phase(s) for s in stages]

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


def maybe_advance_by_eval(state: ScheduleState, schedule: List[Phase],
                          metrics_list: list) -> Tuple[ScheduleState, bool]:
    """统一的课程推进判断 - 终态由位置决定"""
    # 终态检查：已到最后一个阶段 → 不推进
    if state.idx >= len(schedule) - 1:
        return state, False

    phase = schedule[state.idx]

    # 过滤当前阶段的有效评估（避免延迟返回的评估污染课程推进）
    valid = [m for m in metrics_list if m.get('phase_name') == phase.name]
    if not valid:
        return state, False

    streak, prev = state.pass_streak, state.prev_metric

    for m in valid:
        completion = m['metrics']['eval/completion_ratio']
        current = m['metrics'].get('eval/ratio_95_to_done_mean', completion)

        # 统一判断逻辑：completion达标 + 可选的稳定性检查
        passed = completion >= phase.min_completion
        if passed and phase.stability_tolerance is not None and prev is not None:
            passed = abs(current - prev) / max(prev, 1e-6) < phase.stability_tolerance

        streak = streak + 1 if passed else 0
        prev = current

    return replace(state, pass_streak=streak, prev_metric=prev), streak >= phase.consecutive_k

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

    if replay_buffer is not None:
        if cfg.buffer.bucketed: # 如果分桶更新回放缓冲区采样比例
            replay_buffer.set_sampling_ratio(phase.sampling_ratio)
            replay_buffer.reset_buckets()
        else:
            replay_buffer.empty()  # 清空标准回放缓冲区

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
