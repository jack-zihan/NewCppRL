"""
测试阶段推进系统

验证maybe_advance_by_eval和maybe_advance_by_updates的推进条件：
- S1: completion >= 0.90连续K次
- S2: completion >= 0.95 AND ratio95稳定(<5%变化) 连续K次
- S3: 不推进
- PRETRAIN: 更新步数 >= max_updates
"""
import pytest
from dataclasses import replace
from rl_new.sac_cont_sy.train_utils_optimized import (
    maybe_advance_by_eval,
    maybe_advance_by_updates,
    Phase,
    ScheduleState,
)


class TestPhaseAdvancementByEval:
    """测试基于评估结果的阶段推进（S1/S2）"""

    def test_s1_consecutive_advancement(self, minimal_cfg):
        """S1阶段连续达标推进

        验证点:
        - completion >= 0.90连续3次 → 推进
        - consec_completion正确累加
        - 中间值<0.90 → 重置计数
        """
        cfg = minimal_cfg
        cfg.curriculum.s1_consecutive_k = 3
        cfg.curriculum.s1_min_completion = 0.90

        # S1阶段
        phase = Phase(
            kind='STAGE',
            name='S1',
            env_params={},
            sampling_ratio=(0.4, 0.3, 0.3),
            hif_weight=0.5,
            update_gate_frames=100,
        )

        state = ScheduleState(idx=0)

        # 第1次达标
        metrics = [{'metrics': {'eval/completion_ratio': 0.92}}]
        state, should_advance = maybe_advance_by_eval(state, phase, metrics, cfg, curriculum_enabled=True)
        assert state.consec_completion == 1
        assert not should_advance

        # 第2次达标
        metrics = [{'metrics': {'eval/completion_ratio': 0.95}}]
        state, should_advance = maybe_advance_by_eval(state, phase, metrics, cfg, curriculum_enabled=True)
        assert state.consec_completion == 2
        assert not should_advance

        # 第3次达标 → 推进
        metrics = [{'metrics': {'eval/completion_ratio': 0.91}}]
        state, should_advance = maybe_advance_by_eval(state, phase, metrics, cfg, curriculum_enabled=True)
        assert state.consec_completion == 3
        assert should_advance

    def test_s1_reset_on_low_completion(self, minimal_cfg):
        """S1阶段completion不足时重置计数"""
        cfg = minimal_cfg
        cfg.curriculum.s1_consecutive_k = 3
        cfg.curriculum.s1_min_completion = 0.90

        phase = Phase(kind='STAGE', name='S1', env_params={}, sampling_ratio=(0.4, 0.3, 0.3),
                      hif_weight=0.5, update_gate_frames=100)

        state = ScheduleState(idx=0)

        # 达标2次
        metrics = [{'metrics': {'eval/completion_ratio': 0.92}}]
        state, _ = maybe_advance_by_eval(state, phase, metrics, cfg, curriculum_enabled=True)
        metrics = [{'metrics': {'eval/completion_ratio': 0.94}}]
        state, _ = maybe_advance_by_eval(state, phase, metrics, cfg, curriculum_enabled=True)
        assert state.consec_completion == 2

        # 第3次不达标 → 重置
        metrics = [{'metrics': {'eval/completion_ratio': 0.85}}]
        state, should_advance = maybe_advance_by_eval(state, phase, metrics, cfg, curriculum_enabled=True)
        assert state.consec_completion == 0  # 重置
        assert not should_advance

    def test_s2_stability_advancement(self, minimal_cfg):
        """S2阶段稳定性推进

        验证点:
        - completion >= 0.95 AND ratio95变化<5% 连续5次 → 推进
        - last_ratio95正确更新
        - consec_stable正确累加
        """
        cfg = minimal_cfg
        cfg.curriculum.s2_consecutive_k = 5
        cfg.curriculum.s2_min_completion = 0.95
        cfg.curriculum.s2s3_threshold = 0.05

        phase = Phase(kind='STAGE', name='S2', env_params={}, sampling_ratio=(0.3, 0.3, 0.4),
                      hif_weight=0.3, update_gate_frames=100)

        state = ScheduleState(idx=1)  # S2 index

        # 第1次评估（初始化last_ratio95）
        metrics = [{'metrics': {'eval/completion_ratio': 0.96, 'eval/ratio_95_to_done_mean': 0.15}}]
        state, should_advance = maybe_advance_by_eval(state, phase, metrics, cfg, curriculum_enabled=True)
        assert state.last_ratio95 == 0.15
        assert state.consec_stable == 0  # 第一次没有last，不算stable
        assert not should_advance

        # 第2-6次评估，ratio95稳定（变化<5%）
        base_ratio = 0.15
        for i in range(2, 7):  # 5次稳定评估
            # ratio95轻微波动但<5%
            ratio95 = base_ratio * (1.0 + 0.02 * ((-1) ** i))  # ±2%变化
            metrics = [{'metrics': {'eval/completion_ratio': 0.97, 'eval/ratio_95_to_done_mean': ratio95}}]
            state, should_advance = maybe_advance_by_eval(state, phase, metrics, cfg, curriculum_enabled=True)
            assert state.last_ratio95 == ratio95
            if i < 6:
                assert state.consec_stable == i - 1
                assert not should_advance
            else:
                # 第5次稳定 → 推进
                assert state.consec_stable == 5
                assert should_advance

    def test_s2_reset_on_instability(self, minimal_cfg):
        """S2稳定性中断时重置计数"""
        cfg = minimal_cfg
        cfg.curriculum.s2_consecutive_k = 5
        cfg.curriculum.s2_min_completion = 0.95
        cfg.curriculum.s2s3_threshold = 0.05

        phase = Phase(kind='STAGE', name='S2', env_params={}, sampling_ratio=(0.3, 0.3, 0.4),
                      hif_weight=0.3, update_gate_frames=100)

        state = ScheduleState(idx=1)

        # 初始化
        metrics = [{'metrics': {'eval/completion_ratio': 0.96, 'eval/ratio_95_to_done_mean': 0.15}}]
        state, _ = maybe_advance_by_eval(state, phase, metrics, cfg, curriculum_enabled=True)

        # 稳定2次
        metrics = [{'metrics': {'eval/completion_ratio': 0.97, 'eval/ratio_95_to_done_mean': 0.15}}]
        state, _ = maybe_advance_by_eval(state, phase, metrics, cfg, curriculum_enabled=True)
        metrics = [{'metrics': {'eval/completion_ratio': 0.96, 'eval/ratio_95_to_done_mean': 0.15}}]
        state, _ = maybe_advance_by_eval(state, phase, metrics, cfg, curriculum_enabled=True)
        assert state.consec_stable == 2

        # 第3次ratio95大幅变化（>5%） → 重置
        metrics = [{'metrics': {'eval/completion_ratio': 0.97, 'eval/ratio_95_to_done_mean': 0.20}}]  # 33%变化
        state, should_advance = maybe_advance_by_eval(state, phase, metrics, cfg, curriculum_enabled=True)
        assert state.consec_stable == 0  # 重置
        assert not should_advance

    def test_s2_reset_on_low_completion(self, minimal_cfg):
        """S2 completion不足时重置计数"""
        cfg = minimal_cfg
        cfg.curriculum.s2_consecutive_k = 5
        cfg.curriculum.s2_min_completion = 0.95
        cfg.curriculum.s2s3_threshold = 0.05

        phase = Phase(kind='STAGE', name='S2', env_params={}, sampling_ratio=(0.3, 0.3, 0.4),
                      hif_weight=0.3, update_gate_frames=100)

        state = ScheduleState(idx=1)

        # 初始化
        metrics = [{'metrics': {'eval/completion_ratio': 0.96, 'eval/ratio_95_to_done_mean': 0.15}}]
        state, _ = maybe_advance_by_eval(state, phase, metrics, cfg, curriculum_enabled=True)

        # 稳定1次
        metrics = [{'metrics': {'eval/completion_ratio': 0.96, 'eval/ratio_95_to_done_mean': 0.15}}]
        state, _ = maybe_advance_by_eval(state, phase, metrics, cfg, curriculum_enabled=True)
        assert state.consec_stable == 1

        # completion降低到<0.95 → 重置
        metrics = [{'metrics': {'eval/completion_ratio': 0.92, 'eval/ratio_95_to_done_mean': 0.15}}]
        state, should_advance = maybe_advance_by_eval(state, phase, metrics, cfg, curriculum_enabled=True)
        assert state.consec_stable == 0  # 重置
        assert not should_advance

    def test_s3_never_advances(self, minimal_cfg):
        """S3阶段永不推进"""
        cfg = minimal_cfg

        phase = Phase(kind='STAGE', name='S3', env_params={}, sampling_ratio=(0.2, 0.3, 0.5),
                      hif_weight=0.1, update_gate_frames=100)

        state = ScheduleState(idx=2)

        # 即使连续多次高completion也不推进
        for _ in range(10):
            metrics = [{'metrics': {'eval/completion_ratio': 1.0, 'eval/ratio_95_to_done_mean': 0.10}}]
            state, should_advance = maybe_advance_by_eval(state, phase, metrics, cfg, curriculum_enabled=True)
            assert not should_advance

    def test_no_advancement_without_curriculum(self, minimal_cfg):
        """课程未启用时不推进"""
        cfg = minimal_cfg

        phase = Phase(kind='STAGE', name='S1', env_params={}, sampling_ratio=(0.4, 0.3, 0.3),
                      hif_weight=0.5, update_gate_frames=100)

        state = ScheduleState(idx=0)

        # 即使达标也不推进
        metrics = [{'metrics': {'eval/completion_ratio': 0.95}}]
        state, should_advance = maybe_advance_by_eval(state, phase, metrics, cfg, curriculum_enabled=False)
        assert not should_advance
        assert state.consec_completion == 0  # 不累加

    def test_empty_metrics_list(self, minimal_cfg):
        """空评估结果列表不改变state"""
        cfg = minimal_cfg

        phase = Phase(kind='STAGE', name='S1', env_params={}, sampling_ratio=(0.4, 0.3, 0.3),
                      hif_weight=0.5, update_gate_frames=100)

        state = ScheduleState(idx=0, consec_completion=2)

        # 空metrics不改变state
        metrics = []
        new_state, should_advance = maybe_advance_by_eval(state, phase, metrics, cfg, curriculum_enabled=True)
        assert new_state.consec_completion == 2  # 不变
        assert not should_advance

    def test_multiple_metrics_in_list(self, minimal_cfg):
        """多个评估结果连续处理"""
        cfg = minimal_cfg
        cfg.curriculum.s1_consecutive_k = 3
        cfg.curriculum.s1_min_completion = 0.90

        phase = Phase(kind='STAGE', name='S1', env_params={}, sampling_ratio=(0.4, 0.3, 0.3),
                      hif_weight=0.5, update_gate_frames=100)

        state = ScheduleState(idx=0)

        # 一次传入3个达标的metrics → 应该推进
        metrics = [
            {'metrics': {'eval/completion_ratio': 0.91}},
            {'metrics': {'eval/completion_ratio': 0.92}},
            {'metrics': {'eval/completion_ratio': 0.93}},
        ]
        state, should_advance = maybe_advance_by_eval(state, phase, metrics, cfg, curriculum_enabled=True)
        assert state.consec_completion == 3
        assert should_advance


class TestPhaseAdvancementByUpdates:
    """测试基于更新步数的阶段推进（PRETRAIN）"""

    def test_pretrain_advancement_by_updates(self, minimal_cfg):
        """PRETRAIN基于更新步数推进

        验证点:
        - pretrain_updates >= max_updates → 推进
        - pretrain_updates < max_updates → 不推进
        """
        cfg = minimal_cfg
        cfg.hif.pretrain.max_updates = 100

        phase = Phase(kind='PRETRAIN', name='PRETRAIN', env_params={}, sampling_ratio=(0.4, 0.3, 0.3),
                      hif_weight=1.0, update_gate_frames=20)

        # 未达到max_updates
        state = ScheduleState(idx=0, pretrain_updates=50)
        new_state, should_advance = maybe_advance_by_updates(state, phase, cfg)
        assert new_state == state  # state不变
        assert not should_advance

        # 刚好达到max_updates
        state = ScheduleState(idx=0, pretrain_updates=100)
        new_state, should_advance = maybe_advance_by_updates(state, phase, cfg)
        assert new_state == state  # state不变
        assert should_advance

        # 超过max_updates
        state = ScheduleState(idx=0, pretrain_updates=150)
        new_state, should_advance = maybe_advance_by_updates(state, phase, cfg)
        assert new_state == state  # state不变
        assert should_advance

    def test_stage_never_advances_by_updates(self, minimal_cfg):
        """STAGE阶段不基于更新步数推进"""
        cfg = minimal_cfg

        phase = Phase(kind='STAGE', name='S1', env_params={}, sampling_ratio=(0.4, 0.3, 0.3),
                      hif_weight=0.5, update_gate_frames=100)

        # 即使pretrain_updates很大也不推进
        state = ScheduleState(idx=0, pretrain_updates=10000)
        new_state, should_advance = maybe_advance_by_updates(state, phase, cfg)
        assert new_state == state  # state不变
        assert not should_advance


class TestPhaseAdvancementEdgeCases:
    """测试边界条件和异常情况"""

    def test_s1_boundary_completion(self, minimal_cfg):
        """S1 completion刚好等于阈值"""
        cfg = minimal_cfg
        cfg.curriculum.s1_consecutive_k = 2
        cfg.curriculum.s1_min_completion = 0.90

        phase = Phase(kind='STAGE', name='S1', env_params={}, sampling_ratio=(0.4, 0.3, 0.3),
                      hif_weight=0.5, update_gate_frames=100)

        state = ScheduleState(idx=0)

        # completion刚好=0.90应该算达标
        metrics = [{'metrics': {'eval/completion_ratio': 0.90}}]
        state, _ = maybe_advance_by_eval(state, phase, metrics, cfg, curriculum_enabled=True)
        assert state.consec_completion == 1

    def test_s2_boundary_relative_change(self, minimal_cfg):
        """S2 ratio95相对变化刚好等于阈值"""
        cfg = minimal_cfg
        cfg.curriculum.s2_consecutive_k = 2
        cfg.curriculum.s2_min_completion = 0.95
        cfg.curriculum.s2s3_threshold = 0.05  # 5%

        phase = Phase(kind='STAGE', name='S2', env_params={}, sampling_ratio=(0.3, 0.3, 0.4),
                      hif_weight=0.3, update_gate_frames=100)

        state = ScheduleState(idx=1)

        # 初始化last_ratio95
        metrics = [{'metrics': {'eval/completion_ratio': 0.96, 'eval/ratio_95_to_done_mean': 0.20}}]
        state, _ = maybe_advance_by_eval(state, phase, metrics, cfg, curriculum_enabled=True)

        # 相对变化>5% → 应该算不稳定
        metrics = [{'metrics': {'eval/completion_ratio': 0.96, 'eval/ratio_95_to_done_mean': 0.211}}]  # 5.5%变化
        state, should_advance = maybe_advance_by_eval(state, phase, metrics, cfg, curriculum_enabled=True)
        assert state.consec_stable == 0  # 不稳定

    def test_s2_missing_ratio95_metric(self, minimal_cfg):
        """S2缺少ratio_95_to_done_mean指标"""
        cfg = minimal_cfg
        cfg.curriculum.s2_consecutive_k = 2
        cfg.curriculum.s2_min_completion = 0.95
        cfg.curriculum.s2s3_threshold = 0.05

        phase = Phase(kind='STAGE', name='S2', env_params={}, sampling_ratio=(0.3, 0.3, 0.4),
                      hif_weight=0.3, update_gate_frames=100)

        state = ScheduleState(idx=1)

        # 缺少ratio_95_to_done_mean → 默认为0.0
        metrics = [{'metrics': {'eval/completion_ratio': 0.96}}]
        state, should_advance = maybe_advance_by_eval(state, phase, metrics, cfg, curriculum_enabled=True)
        assert state.last_ratio95 == 0.0
        assert not should_advance
