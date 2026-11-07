"""
测试训练日程构建系统

验证build_training_schedule在四种场景组合下的行为：
1. PRETRAIN=ON, CURRICULUM=ON
2. PRETRAIN=OFF, CURRICULUM=ON
3. PRETRAIN=ON, CURRICULUM=OFF
4. PRETRAIN=OFF, CURRICULUM=OFF
"""
import pytest
from dataclasses import FrozenInstanceError
from rl_new.sac_cont_sy.train_utils_optimized import (
    build_training_schedule,
    Phase,
    ScheduleState,
)


class TestTrainingSchedule:
    """测试训练日程构建的各种场景"""

    def test_schedule_pretrain_enabled_curriculum_enabled(self, minimal_cfg):
        """场景1: PRETRAIN=ON, CURRICULUM=ON

        验证点:
        - schedule长度 = 1(PRETRAIN) + len(stages) = 4
        - PRETRAIN.env_params == S1.env_params
        - PRETRAIN.sampling_ratio == S1.sampling_ratio
        - PRETRAIN.hif_weight == 1.0
        - PRETRAIN.update_gate_frames == pretrain.min_buffer_frames
        - 后续stages的env_params/sampling_ratio正确
        """
        # 场景配置
        cfg = minimal_cfg
        cfg.curriculum.enabled = True
        cfg.hif.enabled = True
        cfg.hif.pretrain.enabled = True

        # 构建schedule
        schedule = build_training_schedule(cfg)

        # 验证长度
        assert len(schedule) == 4, f"Expected 4 phases, got {len(schedule)}"

        # 验证PRETRAIN阶段
        pretrain = schedule[0]
        assert pretrain.kind == 'PRETRAIN'
        assert pretrain.name == 'PRETRAIN'
        assert pretrain.hif_weight == 1.0
        assert pretrain.update_gate_frames == int(cfg.hif.pretrain.min_buffer_frames)

        # 验证PRETRAIN复制S1参数
        s1 = schedule[1]
        assert s1.kind == 'STAGE'
        assert s1.name == 'S1'

        # env_params应该相同
        assert pretrain.env_params == s1.env_params
        # sampling_ratio应该相同
        assert pretrain.sampling_ratio == s1.sampling_ratio

        # 验证S1/S2/S3的配置
        s2 = schedule[2]
        s3 = schedule[3]

        assert s1.name == 'S1'
        assert s2.name == 'S2'
        assert s3.name == 'S3'

        # S1配置
        assert s1.env_params['reward_field_group_coef'] == 1.0
        assert s1.env_params['reward_turning_group_coef'] == 0.5
        assert s1.sampling_ratio == (0.40, 0.30, 0.30)
        assert s1.hif_weight == float(cfg.hif.weights.S1)  # 0.5
        assert s1.update_gate_frames == int(cfg.collector.init_random_frames)

        # S2配置
        assert s2.env_params['reward_field_group_coef'] == 0.5
        assert s2.env_params['reward_turning_group_coef'] == 0.2
        assert s2.env_params['reward_overlap_penalty'] == -0.10
        assert s2.sampling_ratio == (0.30, 0.30, 0.40)
        assert s2.hif_weight == float(cfg.hif.weights.S2)  # 0.3

        # S3配置
        assert s3.env_params['reward_field_group_coef'] == 0.10
        assert s3.env_params['reward_turning_group_coef'] == 0.05
        assert s3.env_params['reward_overlap_penalty'] == -0.20
        assert s3.sampling_ratio == (0.20, 0.30, 0.50)
        assert s3.hif_weight == float(cfg.hif.weights.S3)  # 0.1

    def test_schedule_pretrain_disabled_curriculum_enabled(self, minimal_cfg):
        """场景2: PRETRAIN=OFF, CURRICULUM=ON

        验证点:
        - schedule长度 == len(stages) = 3
        - 第一个stage为S1
        - 所有stage的配置正确
        """
        cfg = minimal_cfg
        cfg.curriculum.enabled = True
        cfg.hif.enabled = False  # 禁用HIF → 禁用PRETRAIN

        schedule = build_training_schedule(cfg)

        # 验证长度
        assert len(schedule) == 3, f"Expected 3 phases, got {len(schedule)}"

        # 验证所有阶段都是STAGE类型
        assert all(p.kind == 'STAGE' for p in schedule)

        # 验证名称顺序
        assert schedule[0].name == 'S1'
        assert schedule[1].name == 'S2'
        assert schedule[2].name == 'S3'

        # 验证各阶段配置
        s1 = schedule[0]
        assert s1.env_params['reward_field_group_coef'] == 1.0
        assert s1.sampling_ratio == (0.40, 0.30, 0.30)
        assert s1.update_gate_frames == int(cfg.collector.init_random_frames)

    def test_schedule_pretrain_enabled_curriculum_disabled(self, minimal_cfg):
        """场景3: PRETRAIN=ON, CURRICULUM=OFF

        验证点:
        - schedule = [PRETRAIN, S3]
        - PRETRAIN复制S3参数
        - 只有两个阶段
        """
        cfg = minimal_cfg
        cfg.curriculum.enabled = False  # 禁用课程
        cfg.hif.enabled = True
        cfg.hif.pretrain.enabled = True

        schedule = build_training_schedule(cfg)

        # 验证长度
        assert len(schedule) == 2, f"Expected 2 phases, got {len(schedule)}"

        # 验证PRETRAIN阶段
        pretrain = schedule[0]
        assert pretrain.kind == 'PRETRAIN'
        assert pretrain.name == 'PRETRAIN'
        assert pretrain.hif_weight == 1.0

        # 验证LAST阶段（使用S3配置）
        last = schedule[1]
        assert last.kind == 'STAGE'
        assert last.name == 'S3'  # 名称为S3

        # PRETRAIN应复制LAST(S3)的参数
        assert pretrain.env_params == last.env_params
        assert pretrain.sampling_ratio == last.sampling_ratio

        # LAST应该有S3的配置
        assert last.env_params['reward_field_group_coef'] == 0.10
        assert last.sampling_ratio == (0.20, 0.30, 0.50)

    def test_schedule_pretrain_disabled_curriculum_disabled(self, minimal_cfg):
        """场景4: PRETRAIN=OFF, CURRICULUM=OFF

        验证点:
        - schedule = [S3]
        - 单阶段LAST配置
        """
        cfg = minimal_cfg
        cfg.curriculum.enabled = False
        cfg.hif.enabled = False  # 禁用HIF → 禁用PRETRAIN

        schedule = build_training_schedule(cfg)

        # 验证长度
        assert len(schedule) == 1, f"Expected 1 phase, got {len(schedule)}"

        # 验证单阶段
        only_stage = schedule[0]
        assert only_stage.kind == 'STAGE'
        assert only_stage.name == 'S3'

        # 验证使用S3配置
        assert only_stage.env_params['reward_field_group_coef'] == 0.10
        assert only_stage.env_params['reward_overlap_penalty'] == -0.20
        assert only_stage.sampling_ratio == (0.20, 0.30, 0.50)

    def test_phase_immutability(self):
        """验证Phase的frozen属性

        验证点:
        - Phase创建后不可修改
        - 尝试修改抛出FrozenInstanceError
        """
        phase = Phase(
            kind='STAGE',
            name='test',
            env_params={'test_param': 1.0},
            sampling_ratio=(0.3, 0.3, 0.4),
            hif_weight=0.5,
            update_gate_frames=100,
        )

        # 尝试修改应该抛出异常
        with pytest.raises(FrozenInstanceError):
            phase.hif_weight = 0.8

        with pytest.raises(FrozenInstanceError):
            phase.name = 'modified'

    def test_hif_weight_lookup(self, minimal_cfg):
        """验证HIF权重查找逻辑

        验证点:
        - 从cfg.hif.weights.{stage_name}正确读取
        - 缺失时返回0.0
        - PRETRAIN固定1.0
        """
        cfg = minimal_cfg
        cfg.curriculum.enabled = True
        cfg.hif.enabled = True
        cfg.hif.pretrain.enabled = True

        # 设置特定权重
        cfg.hif.weights.S1 = 0.7
        cfg.hif.weights.S2 = 0.4
        cfg.hif.weights.S3 = 0.1

        schedule = build_training_schedule(cfg)

        # PRETRAIN固定1.0
        assert schedule[0].hif_weight == 1.0

        # STAGE使用配置权重
        assert schedule[1].hif_weight == 0.7  # S1
        assert schedule[2].hif_weight == 0.4  # S2
        assert schedule[3].hif_weight == 0.1  # S3

    def test_hif_weight_missing_defaults_to_zero(self, minimal_cfg):
        """验证缺失HIF权重时默认为0.0"""
        cfg = minimal_cfg
        cfg.curriculum.enabled = True
        cfg.hif.enabled = True
        cfg.hif.pretrain.enabled = False

        # 删除权重配置
        if hasattr(cfg.hif, 'weights'):
            delattr(cfg.hif, 'weights')

        schedule = build_training_schedule(cfg)

        # 所有STAGE的hif_weight应该为0.0
        for phase in schedule:
            if phase.kind == 'STAGE':
                assert phase.hif_weight == 0.0

    def test_env_params_structure(self, minimal_cfg):
        """验证env_params字典的完整性"""
        cfg = minimal_cfg
        cfg.curriculum.enabled = True
        cfg.hif.enabled = False

        schedule = build_training_schedule(cfg)

        required_keys = [
            'reward_field_group_coef',
            'reward_turning_group_coef',
            'reward_overlap_penalty',
            'overlap_tolerance',
            'field_scale_enabled',
            'field_scale_range',
        ]

        # 所有阶段的env_params应包含必需键
        for phase in schedule:
            for key in required_keys:
                assert key in phase.env_params, f"Missing {key} in {phase.name}.env_params"

    def test_sampling_ratio_tuple_length(self, minimal_cfg):
        """验证sampling_ratio元组长度为3"""
        cfg = minimal_cfg
        cfg.curriculum.enabled = True
        cfg.hif.enabled = True
        cfg.hif.pretrain.enabled = True

        schedule = build_training_schedule(cfg)

        # 所有阶段的sampling_ratio应为长度3的元组
        for phase in schedule:
            assert isinstance(phase.sampling_ratio, tuple)
            assert len(phase.sampling_ratio) == 3
            # 总和应接近1.0
            assert abs(sum(phase.sampling_ratio) - 1.0) < 1e-6

    def test_update_gate_frames_values(self, minimal_cfg):
        """验证update_gate_frames的值"""
        cfg = minimal_cfg
        cfg.curriculum.enabled = True
        cfg.hif.enabled = True
        cfg.hif.pretrain.enabled = True
        cfg.hif.pretrain.min_buffer_frames = 50
        cfg.collector.init_random_frames = 100

        schedule = build_training_schedule(cfg)

        # PRETRAIN使用min_buffer_frames
        assert schedule[0].update_gate_frames == 50

        # STAGE使用init_random_frames
        for phase in schedule[1:]:
            assert phase.update_gate_frames == 100
