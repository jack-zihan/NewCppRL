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

import os
import shutil
import tensordict
import torch.cuda
from dataclasses import dataclass
from typing import Optional, Union, Iterator

from omegaconf import DictConfig
from torchrl._utils import logger as torchrl_logger, timeit
from torchrl.envs.transforms import MultiStepTransform
from torchrl.data import LazyMemmapStorage, LazyTensorStorage, TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.collectors import SyncDataCollector, MultiaSyncDataCollector

from rl_new.sac_cont_sy.env_utils import make_train_environment
from rl_new.sac_cont_sy.sac_utils import LossMode, set_optimizer_group_lrs
from rl_new.sac_cont_sy.bucketed_replay import BucketedTensorDictPrioritizedReplayBuffer

torch.set_float32_matmul_precision("high")  # 提升矩阵乘法性能
tensordict.nn.functional_modules._exclude_td_from_pytree().set()


# ============ Curriculum Learning State ============
@dataclass
class CurriculumState:
    """Tracks curriculum learning state across training stages."""
    stage_idx: int = 0
    consecutive_completion_count: int = 0  # For S1→S2
    consecutive_stable_count: int = 0  # For S2→S3
    last_ratio_95_to_done: Optional[float] = None  # step count for 95% to done


# ================= Layer 2: Minimal Training Phase Manager =================
from dataclasses import dataclass
from typing import Tuple


@dataclass
class PhaseState:
    """Top-level training phase state.

    Phase mapping:
        0 = PRETRAIN (HIF pretrain)
        1 = S1
        2 = S2
        3 = S3
    """

    current_phase: int = 0
    pretrain_updates: int = 0
    curriculum_state: Optional[CurriculumState] = None


class TrainingPhaseManager:
    """Orchestrates phase transitions and exposes update gates.

    Responsibilities (minimal):
    - Maintain PhaseState
    - Provide update gate threshold per mode
    - Track curriculum state after evaluation (update_after_eval)
    - Decide whether to transition (should_transition)
    - Execute transition side effects: env/collector/replay, LR, HIF weight via loss.set_mode
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.curriculum_config = load_curriculum_config(cfg)

        pretrain_enabled = bool(getattr(cfg, 'hif', None) and cfg.hif.enabled and cfg.hif.pretrain.enabled)
        initial_phase = 0 if pretrain_enabled else 1
        self.state = PhaseState(
            current_phase=initial_phase,
            pretrain_updates=0,
            curriculum_state=CurriculumState() if self.curriculum_config else None,
        )

        # Held flag set via update_after_eval; consumed by should_transition
        self._curriculum_should_switch = False

        torchrl_logger.info(
            f"[PhaseManager] init: phase={self.state.current_phase}, pretrain={pretrain_enabled}, curriculum={self.curriculum_config is not None}"
        )

    # -------- Read-only helpers --------
    @property
    def current_phase(self) -> int:
        return self.state.current_phase

    @property
    def current_mode(self) -> LossMode:
        if self.state.current_phase == 0:
            return LossMode.PRETRAIN
        return LossMode.JOINT if (getattr(self.cfg, 'hif', None) and self.cfg.hif.enabled) else LossMode.SAC_ONLY

    def get_update_gate(self) -> int:
        """Return number of frames required before allowing updates in current mode."""
        if self.current_mode == LossMode.PRETRAIN:
            return int(self.cfg.hif.pretrain.min_buffer_frames)
        return int(self.cfg.collector.init_random_frames)

    # -------- Mutations from training loop --------
    def increment_pretrain_updates(self):
        self.state = PhaseState(
            current_phase=self.state.current_phase,
            pretrain_updates=self.state.pretrain_updates + 1,
            curriculum_state=self.state.curriculum_state,
        )

    def update_after_eval(self, metrics: dict):
        """Update curriculum state using evaluation metrics.

        Only meaningful for phases >= 1 (S1/S2/S3). Sets an internal flag if should transition.
        """
        if self.state.curriculum_state is None or self.curriculum_config is None:
            return
        new_state, should_transition = update_curriculum_state(self.state.curriculum_state, self.curriculum_config, metrics)
        # Update state regardless
        self.state.curriculum_state = new_state

        # Do not move past final stage
        final_reached = (new_state.stage_idx >= len(self.curriculum_config['stages']) - 1)
        self._curriculum_should_switch = bool(should_transition and not final_reached)

    def should_transition(self) -> bool:
        if self.state.current_phase == 0:
            return self.state.pretrain_updates >= int(self.cfg.hif.pretrain.max_updates)
        if self.state.curriculum_state is None:
            return False
        if self._curriculum_should_switch:
            # consume the flag
            self._curriculum_should_switch = False
            return True
        return False

    # -------- Execute side effects --------
    def execute_stage_transition(self,
                                 optimizer: torch.optim.Optimizer,
                                 collector: MultiaSyncDataCollector,
                                 replay_buffer: Union[BucketedTensorDictPrioritizedReplayBuffer, TensorDictPrioritizedReplayBuffer],
                                 tmpdir: str,
                                 train_device: torch.device,
                                 actor: torch.nn.Module,
                                 loss_module: torch.nn.Module,
                                 ) -> Tuple[MultiaSyncDataCollector, Union[BucketedTensorDictPrioritizedReplayBuffer, TensorDictPrioritizedReplayBuffer], Iterator]:
        """Apply transition and return new collector/buffer/iter.

        Internally sets LR and HIF weight (loss_module.set_mode) according to new phase.
        """
        old_phase = self.state.current_phase
        new_phase = old_phase + 1

        # PRETRAIN -> S1
        if old_phase == 0:
            # 1) 恢复SAC学习率
            set_optimizer_group_lrs(
                optimizer,
                actor_lr=float(self.cfg.optim.lr_actor),
                critic_lr=float(self.cfg.optim.lr_critic),
                alpha_lr=float(self.cfg.optim.lr_alpha),
            )

            # 2) 选取S1阶段配置
            if self.curriculum_config is None:
                raise RuntimeError("Curriculum config required to transition into S1")
            s1_stage = self.curriculum_config['stages'][0]

            torchrl_logger.info(
                f"[Curriculum] ✨ 切换到阶段: {s1_stage['name']} | 采样比例={s1_stage['sampling_ratio']} | 环境参数={s1_stage['env_params']}"
            )

            # 3) 关闭旧采集器
            collector.shutdown()

            # 4) 回放：分桶则更新比例并清桶；否则重建标准PRB
            use_bucketed = bool(self.cfg.buffer.bucketed)
            if use_bucketed:
                replay_buffer.set_sampling_ratio(s1_stage['sampling_ratio'])
                replay_buffer.reset_buckets()
                torchrl_logger.info(f"[Curriculum] 分桶采样比例更新为: {s1_stage['sampling_ratio']}")
                new_buffer = replay_buffer
            else:
                del replay_buffer
                shutil.rmtree(tmpdir)
                os.makedirs(tmpdir, exist_ok=True)
                new_buffer = create_replay_buffer(self.cfg, tmpdir, train_device, use_bucketed=False)

            # 5) 更新env参数
            if self.cfg.env.get('env_kwargs', None) is None:
                self.cfg.env.env_kwargs = {}
            self.cfg.env.env_kwargs.update(s1_stage['env_params'])
            torchrl_logger.info(f"[Curriculum] 环境参数更新已应用: {s1_stage['env_params']}")

            # 6) 重建采集器
            new_collector = create_collector(self.cfg, actor, train_device)
            new_iter = iter(new_collector)

            # 7) 设置Loss模式与HIF权重
            stage_name = s1_stage['name']
            if getattr(self.cfg, 'hif', None) and self.cfg.hif.enabled:
                weight = float(self.cfg.hif.weights[stage_name])
                loss_module.set_mode(LossMode.JOINT, hif_weight=weight)
            else:
                loss_module.set_mode(LossMode.SAC_ONLY)

            # 8) 推进阶段
            self.state = PhaseState(
                current_phase=new_phase,
                pretrain_updates=self.state.pretrain_updates,
                curriculum_state=self.state.curriculum_state,
            )

            return new_collector, new_buffer, new_iter

        # S1/S2 -> S2/S3
        if self.curriculum_config is None or self.state.curriculum_state is None:
            raise RuntimeError("Curriculum state required for stage transition")

        self.state.curriculum_state.stage_idx += 1
        next_stage = self.curriculum_config['stages'][self.state.curriculum_state.stage_idx]

        # 记录日志
        torchrl_logger.info(
            f"[Curriculum] ✨ 切换到阶段: {next_stage['name']} | 采样比例={next_stage['sampling_ratio']} | 环境参数={next_stage['env_params']}"
        )

        # 关闭旧采集器
        collector.shutdown()

        # 回放处理
        use_bucketed = bool(self.cfg.buffer.bucketed)
        if use_bucketed:
            replay_buffer.set_sampling_ratio(next_stage['sampling_ratio'])
            replay_buffer.reset_buckets()
            torchrl_logger.info(f"[Curriculum] 分桶采样比例更新为: {next_stage['sampling_ratio']}")
            new_buffer = replay_buffer
        else:
            del replay_buffer
            shutil.rmtree(tmpdir)
            os.makedirs(tmpdir, exist_ok=True)
            new_buffer = create_replay_buffer(self.cfg, tmpdir, train_device, use_bucketed=False)

        # 更新环境参数
        if self.cfg.env.get('env_kwargs', None) is None:
            self.cfg.env.env_kwargs = {}
        self.cfg.env.env_kwargs.update(next_stage['env_params'])
        torchrl_logger.info(f"[Curriculum] 环境参数更新已应用: {next_stage['env_params']}")

        # 重建采集器
        new_collector = create_collector(self.cfg, actor, train_device)
        new_iter = iter(new_collector)

        # Reset curriculum counters
        self.state.curriculum_state.consecutive_completion_count = 0
        self.state.curriculum_state.consecutive_stable_count = 0
        self.state.curriculum_state.last_ratio_95_to_done = None

        # Keep LR as SAC values, only adjust HIF weight if enabled
        stage_name = next_stage['name']
        if getattr(self.cfg, 'hif', None) and self.cfg.hif.enabled:
            weight = float(self.cfg.hif.weights[stage_name])
            loss_module.set_mode(self.current_mode, hif_weight=weight)
        else:
            loss_module.set_mode(LossMode.SAC_ONLY)

        self.state = PhaseState(
            current_phase=new_phase,
            pretrain_updates=self.state.pretrain_updates,
            curriculum_state=self.state.curriculum_state,
        )

        return new_collector, new_buffer, new_iter


# ============ Factory Functions ============
def create_replay_buffer(cfg: DictConfig, tmpdir: str, train_device: torch.device, use_bucketed: bool = False,
                         stage_config: Optional[dict] = None, ) -> Union[
    BucketedTensorDictPrioritizedReplayBuffer, TensorDictPrioritizedReplayBuffer]:
    """Create and configure replay buffer with all transforms.

    Args:
        cfg: Hydra configuration
        tmpdir: Temporary directory for storage
        train_device: Device for training (samples will be transferred here)
        use_bucketed: Whether to use bucketed prioritized replay buffer
        stage_config: Optional curriculum stage config (for sampling ratio)

    Returns:
        Configured replay buffer with n-step and device transforms applied
    """

    # Create buffer based on type
    if use_bucketed:
        buffer = BucketedTensorDictPrioritizedReplayBuffer(alpha=cfg.buffer.alpha, beta=cfg.buffer.beta,
                                                           batch_size=cfg.buffer.batch_size,
                                                           pin_memory=cfg.buffer.pin_memory, prefetch=0,
                                                           # 分桶优先级预取禁用，避免提前采样不匹配的batch问题
                                                           storage=LazyMemmapStorage(max_size=cfg.buffer.buffer_size,
                                                                                     scratch_dir=tmpdir),
                                                           success_threshold=cfg.buffer.success_threshold,
                                                           near_end_threshold=cfg.buffer.near_end_threshold)
        # Set initial sampling ratio if provided (for curriculum learning)
        if stage_config:
            buffer.set_sampling_ratio(stage_config['sampling_ratio'])
    else:
        buffer = TensorDictPrioritizedReplayBuffer(alpha=cfg.buffer.alpha, beta=cfg.buffer.beta,
                                                   batch_size=cfg.buffer.batch_size, pin_memory=cfg.buffer.pin_memory,
                                                   prefetch=cfg.buffer.prefetch,
                                                   storage=LazyMemmapStorage(max_size=cfg.buffer.buffer_size,
                                                                             scratch_dir=tmpdir))
    # Apply n-step transform if configured
    if cfg.loss.n_steps > 1:
        buffer.append_transform(MultiStepTransform(n_steps=cfg.loss.n_steps, gamma=cfg.loss.gamma,
                                                   reward_keys=["reward"], done_keys=["done", "truncated", "terminated"]
                                                   ))
        torchrl_logger.info(f"配置n-step SAC: n_steps={cfg.loss.n_steps}, gamma={cfg.loss.gamma}")
    # Apply device transfer transform
    buffer.append_transform(lambda td: td.to(train_device))
    return buffer


def create_collector(cfg: DictConfig, actor_policy: torch.nn.Module, train_device: torch.device,
                     ) -> MultiaSyncDataCollector:
    """Create and configure multi-async data collector.

    Args:
        cfg: Hydra configuration
        actor_policy: Actor network for policy
        train_device: Device for policy execution

    Returns:
        Configured collector with seed set
    """
    collector = MultiaSyncDataCollector(
        create_env_fn=[lambda: make_train_environment(cfg, device='cpu') for _ in range(cfg.collector.num_collectors)],
        frames_per_batch=cfg.collector.frames_per_batch, total_frames=cfg.collector.total_frames, device=None,
        policy_device=train_device, storing_device="cpu", env_device="cpu", policy=actor_policy, max_frames_per_traj=-1)
    collector.set_seed(cfg.seed)
    return collector

# ============ Curriculum Learning Logic ============
def update_curriculum_state(curriculum_state: CurriculumState, curriculum_config: dict, metrics: dict,
                            ) -> tuple[CurriculumState, bool]:
    """Update curriculum state and determine if stage transition is needed.

    Pure function - no side effects, returns new state and transition decision.

    Args:
        curriculum_state: Current curriculum state
        curriculum_config: Curriculum configuration
        metrics: Evaluation metrics dict

    Returns:
        (updated_state, should_transition): New state and whether to advance stage
    """
    # Extract metrics
    completion = float(metrics['eval/completion_ratio'])
    ratio_95_to_done = float(metrics.get('eval/ratio_95_to_done_mean', 0.0))

    # Copy state for immutability
    new_state = CurriculumState(
        stage_idx=curriculum_state.stage_idx,
        consecutive_completion_count=curriculum_state.consecutive_completion_count,
        consecutive_stable_count=curriculum_state.consecutive_stable_count,
        last_ratio_95_to_done=curriculum_state.last_ratio_95_to_done)

    should_transition = False

    if new_state.stage_idx == 0:  # S1 Stage: Learning to scan
        # S1→S2: Use consecutive counting (K times), not single judgment
        new_state.consecutive_completion_count = (
            new_state.consecutive_completion_count + 1 if completion >= curriculum_config['s1_min_completion'] else 0
        )
        should_transition = (new_state.consecutive_completion_count >= curriculum_config['s1_consecutive_k'])

        if new_state.consecutive_completion_count > 0:
            torchrl_logger.info(f"[Curriculum S1] 连续达标: {new_state.consecutive_completion_count}/"
                                f"{curriculum_config['s1_consecutive_k']} (完成率={completion:.3f})")

    elif new_state.stage_idx == 1:  # S2 Stage: Reducing overlap
        # S2→S3: completion & ratio95 pass absolute gates AND relative change < threshold (consecutive K times)
        if new_state.last_ratio_95_to_done is not None:
            relative_ratio_change = (abs(ratio_95_to_done - new_state.last_ratio_95_to_done)
                                     / max(new_state.last_ratio_95_to_done, 1e-6))

            # Check both completion threshold and ratio95 stability
            is_stable = (completion >= curriculum_config['s2_min_completion']
                         and relative_ratio_change < curriculum_config['s2_threshold'])
            new_state.consecutive_stable_count = (new_state.consecutive_stable_count + 1 if is_stable else 0)

            if new_state.consecutive_stable_count > 0:
                torchrl_logger.info(
                    f"[Curriculum S2] 连续稳定: {new_state.consecutive_stable_count}/"
                    f"{curriculum_config['s2_consecutive_k']} "
                    f"(completion={completion:.3f}"
                    f"{'✓' if completion >= curriculum_config['s2_min_completion'] else '✗'}, "
                    f"tail_ratio={ratio_95_to_done:.3f}, "
                    f"变化={relative_ratio_change:.4f})"
                )

        new_state.last_ratio_95_to_done = ratio_95_to_done
        should_transition = (new_state.consecutive_stable_count >= curriculum_config['s2_consecutive_k'])

    # Only transition if not at final stage
    if should_transition and new_state.stage_idx >= len(curriculum_config['stages']) - 1:
        should_transition = False

    return new_state, should_transition


def execute_stage_transition(*args, **kwargs):
    """[deprecated] 请使用 TrainingPhaseManager.execute_stage_transition。

    该自由函数已废弃，切换逻辑已合并到 Manager 方法中，避免重复实现与不一致。
    """
    raise RuntimeError(
        "execute_stage_transition 已废弃，请改用 TrainingPhaseManager.execute_stage_transition(...)"
    )


def load_curriculum_config(cfg):
    """从配置加载课程学习参数
    Args:
        cfg: Hydra配置对象
    Returns:
        dict: 包含stages列表和转换参数的字典，如果不启用课程学习则返回None
            - stages: 阶段配置列表
            - s1_consecutive_k: S1→S2连续达标次数
            - s2_consecutive_k: S2→S3连续稳定次数
            - s2_threshold: S2→S3相对变化阈值
    """
    if not getattr(cfg, 'curriculum', None) or not cfg.curriculum.enabled:
        # Curriculum 禁用，检查是否需要单阶段 PRETRAIN → S3
        if cfg.hif.pretrain.enabled:
            # 直接访问最后一个阶段（S3）- fail-fast 如果配置错误
            s3_cfg = cfg.curriculum.stages[-1]

            # 复用完整课程学习的构建逻辑
            s3_stage = {
                'name': str(s3_cfg.name),
                'env_params': {
                    'reward_field_group_coef': float(s3_cfg.reward_field_group_coef),
                    'reward_turning_group_coef': float(s3_cfg.reward_turning_group_coef),
                    'reward_overlap_penalty': float(s3_cfg.reward_overlap_penalty),
                    'field_scale_enabled': bool(s3_cfg.field_scale_enabled),
                    'field_scale_range': (float(s3_cfg.field_scale_range[0]),
                                         float(s3_cfg.field_scale_range[1])),
                },
                'sampling_ratio': tuple(s3_cfg.sampling_ratio),
            }

            torchrl_logger.info(
                f"[Curriculum] Single-stage: {s3_stage['name']} "
                f"(PRETRAIN → sparse reward)"
            )

            return {
                'stages': [s3_stage],
                's1_consecutive_k': 999,
                's2_consecutive_k': 999,
                's2_threshold': 999.0,
                's1_min_completion': 0.0,
                's2_min_completion': 0.0,
            }

        return None

    stages = []
    for stage_cfg in cfg.curriculum.stages:

        # 统一构建环境参数（env_params）：奖励参数与几何缩放参数统一注入 env.env_kwargs
        field_scale_enabled_value = bool(stage_cfg.field_scale_enabled)
        field_scale_range_values_list = list(stage_cfg.field_scale_range)
        field_scale_range_values = (float(field_scale_range_values_list[0]), float(field_scale_range_values_list[1]))

        env_params = {
            'reward_field_group_coef': float(stage_cfg.reward_field_group_coef),
            'reward_turning_group_coef': float(stage_cfg.reward_turning_group_coef),
            'reward_overlap_penalty': float(stage_cfg.reward_overlap_penalty),
            'overlap_tolerance': float(stage_cfg.overlap_tolerance),
            'field_scale_enabled': field_scale_enabled_value,
            'field_scale_range': field_scale_range_values,
        }

        stage = {
            'name': str(stage_cfg.name),
            'env_params': env_params,
            'sampling_ratio': tuple(stage_cfg.sampling_ratio),
        }
        stages.append(stage)
    return {
        'stages': stages, 's2_threshold': float(cfg.curriculum.s2s3_threshold),
        's1_consecutive_k': int(cfg.curriculum.s1_consecutive_k),
        's2_consecutive_k': int(cfg.curriculum.s2_consecutive_k),
        's1_min_completion': float(cfg.curriculum.s1_min_completion),
        's2_min_completion': float(cfg.curriculum.s2_min_completion),
    }
