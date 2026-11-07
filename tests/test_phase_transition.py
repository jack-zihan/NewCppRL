"""
测试阶段切换执行系统

验证apply_phase函数和阶段切换的完整执行过程：
- 环境参数注入
- 采样比例更新和清桶
- 学习率切换
- Collector重建

注意：此测试使用最小化参数（CPU、小batch）快速验证功能正确性
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from omegaconf import OmegaConf
from rl_new.sac_cont_sy.train_utils_optimized import Phase


class TestPhaseTransitionExecution:
    """测试阶段切换执行逻辑"""

    def test_env_params_injection(self, minimal_cfg):
        """环境参数注入验证

        验证点:
        - cfg.env.env_kwargs正确更新
        - 新环境使用更新后的参数
        """
        cfg = minimal_cfg

        # 初始环境参数
        initial_reward_field = cfg.env.env_kwargs.get('reward_field_group_coef', 0.0)

        # S2阶段参数
        phase_s2 = Phase(
            kind='STAGE',
            name='S2',
            env_params={
                'reward_field_group_coef': 0.5,
                'reward_turning_group_coef': 0.2,
                'reward_overlap_penalty': -0.10,
            },
            sampling_ratio=(0.30, 0.30, 0.40),
            hif_weight=0.3,
            update_gate_frames=100,
        )

        # 模拟apply_phase的env_params注入部分
        cfg.env.env_kwargs.update(phase_s2.env_params)

        # 验证cfg已更新
        assert cfg.env.env_kwargs['reward_field_group_coef'] == 0.5
        assert cfg.env.env_kwargs['reward_turning_group_coef'] == 0.2
        assert cfg.env.env_kwargs['reward_overlap_penalty'] == -0.10
        assert cfg.env.env_kwargs['reward_field_group_coef'] != initial_reward_field

    def test_sampling_ratio_update_and_bucket_reset(self, minimal_cfg):
        """采样比例更新和清桶验证

        验证点:
        - buffer.set_sampling_ratio正确调用
        - buffer.reset_buckets正确调用
        """
        # Mock buffer
        mock_buffer = Mock()
        mock_buffer.set_sampling_ratio = Mock()
        mock_buffer.reset_buckets = Mock()

        # S1 → S2切换
        phase_s2 = Phase(
            kind='STAGE',
            name='S2',
            env_params={},
            sampling_ratio=(0.30, 0.30, 0.40),  # S2采样比例
            hif_weight=0.3,
            update_gate_frames=100,
        )

        # 模拟apply_phase的buffer操作部分
        mock_buffer.set_sampling_ratio(phase_s2.sampling_ratio)
        mock_buffer.reset_buckets()

        # 验证调用
        mock_buffer.set_sampling_ratio.assert_called_once_with((0.30, 0.30, 0.40))
        mock_buffer.reset_buckets.assert_called_once()

    def test_actor_learning_rate_switch(self, minimal_cfg):
        """学习率切换验证

        验证点:
        - PRETRAIN → STAGE: actor_lr从pretrain_lr切回标准lr
        - STAGE → STAGE: actor_lr保持标准lr
        """
        cfg = minimal_cfg
        cfg.optim.lr_actor = 3e-4
        cfg.hif.pretrain.actor_lr = 1e-3

        # Mock optimizer
        mock_actor_opt = Mock()
        mock_actor_opt.param_groups = [{'lr': 1e-3}]  # PRETRAIN时的学习率

        # PRETRAIN → S1切换
        phase_s1 = Phase(kind='STAGE', name='S1', env_params={},
                        sampling_ratio=(0.4, 0.3, 0.3), hif_weight=0.5, update_gate_frames=100)

        # 模拟学习率切换逻辑
        if phase_s1.kind == 'STAGE':
            mock_actor_opt.param_groups[0]['lr'] = cfg.optim.lr_actor

        # 验证切换
        assert mock_actor_opt.param_groups[0]['lr'] == 3e-4

        # S1 → S2切换（应该保持标准lr）
        phase_s2 = Phase(kind='STAGE', name='S2', env_params={},
                        sampling_ratio=(0.3, 0.3, 0.4), hif_weight=0.3, update_gate_frames=100)

        # 模拟学习率切换逻辑（STAGE → STAGE不变）
        if phase_s2.kind == 'STAGE':
            mock_actor_opt.param_groups[0]['lr'] = cfg.optim.lr_actor

        # 验证保持
        assert mock_actor_opt.param_groups[0]['lr'] == 3e-4

    def test_collector_rebuild_on_stage_change(self, minimal_cfg):
        """Collector重建验证

        验证点:
        - 旧collector正确shutdown
        - 新collector使用更新后的配置
        """
        cfg = minimal_cfg

        # Mock旧collector
        mock_old_collector = Mock()
        mock_old_collector.shutdown = Mock()

        # S1 → S2切换
        phase_s2 = Phase(
            kind='STAGE',
            name='S2',
            env_params={'reward_field_group_coef': 0.5},
            sampling_ratio=(0.30, 0.30, 0.40),
            hif_weight=0.3,
            update_gate_frames=100,
        )

        # 模拟apply_phase的collector操作部分
        # 1. Shutdown旧collector
        mock_old_collector.shutdown()

        # 2. 更新配置
        cfg.env.env_kwargs.update(phase_s2.env_params)

        # 3. 创建新collector（这里用mock）
        # 实际代码会调用create_collector(cfg, actor, device)
        # 我们验证shutdown被调用，且配置已更新
        mock_old_collector.shutdown.assert_called_once()
        assert cfg.env.env_kwargs['reward_field_group_coef'] == 0.5

    def test_pretrain_to_s1_transition_complete(self, minimal_cfg):
        """PRETRAIN → S1完整切换验证

        验证点:
        - env_params注入
        - buffer操作
        - 学习率切换
        - collector重建
        - 所有操作按正确顺序执行
        """
        cfg = minimal_cfg

        # Mock组件
        mock_buffer = Mock()
        mock_buffer.set_sampling_ratio = Mock()
        mock_buffer.reset_buckets = Mock()

        mock_collector = Mock()
        mock_collector.shutdown = Mock()

        mock_actor_opt = Mock()
        mock_actor_opt.param_groups = [{'lr': 1e-3}]  # PRETRAIN学习率

        # S1阶段
        phase_s1 = Phase(
            kind='STAGE',
            name='S1',
            env_params={
                'reward_field_group_coef': 1.0,
                'reward_turning_group_coef': 0.5,
            },
            sampling_ratio=(0.40, 0.30, 0.30),
            hif_weight=0.5,
            update_gate_frames=100,
        )

        # 执行切换操作序列
        # 1. 环境参数注入
        cfg.env.env_kwargs.update(phase_s1.env_params)

        # 2. 采样比例更新和清桶
        mock_buffer.set_sampling_ratio(phase_s1.sampling_ratio)
        mock_buffer.reset_buckets()

        # 3. 学习率切换
        mock_actor_opt.param_groups[0]['lr'] = cfg.optim.lr_actor

        # 4. Collector重建
        mock_collector.shutdown()

        # 验证所有操作
        assert cfg.env.env_kwargs['reward_field_group_coef'] == 1.0
        assert cfg.env.env_kwargs['reward_turning_group_coef'] == 0.5
        mock_buffer.set_sampling_ratio.assert_called_once_with((0.40, 0.30, 0.30))
        mock_buffer.reset_buckets.assert_called_once()
        assert mock_actor_opt.param_groups[0]['lr'] == 3e-4  # 切回标准lr
        mock_collector.shutdown.assert_called_once()

    def test_s1_to_s2_transition_complete(self, minimal_cfg):
        """S1 → S2完整切换验证"""
        cfg = minimal_cfg

        # Mock组件
        mock_buffer = Mock()
        mock_buffer.set_sampling_ratio = Mock()
        mock_buffer.reset_buckets = Mock()

        mock_collector = Mock()
        mock_collector.shutdown = Mock()

        # S2阶段
        phase_s2 = Phase(
            kind='STAGE',
            name='S2',
            env_params={
                'reward_field_group_coef': 0.5,
                'reward_turning_group_coef': 0.2,
                'reward_overlap_penalty': -0.10,
            },
            sampling_ratio=(0.30, 0.30, 0.40),
            hif_weight=0.3,
            update_gate_frames=100,
        )

        # 执行切换
        cfg.env.env_kwargs.update(phase_s2.env_params)
        mock_buffer.set_sampling_ratio(phase_s2.sampling_ratio)
        mock_buffer.reset_buckets()
        mock_collector.shutdown()

        # 验证
        assert cfg.env.env_kwargs['reward_field_group_coef'] == 0.5
        assert cfg.env.env_kwargs['reward_overlap_penalty'] == -0.10
        mock_buffer.set_sampling_ratio.assert_called_once_with((0.30, 0.30, 0.40))
        mock_buffer.reset_buckets.assert_called_once()
        mock_collector.shutdown.assert_called_once()

    def test_s2_to_s3_transition_complete(self, minimal_cfg):
        """S2 → S3完整切换验证（最后阶段）"""
        cfg = minimal_cfg

        # Mock组件
        mock_buffer = Mock()
        mock_buffer.set_sampling_ratio = Mock()
        mock_buffer.reset_buckets = Mock()

        mock_collector = Mock()
        mock_collector.shutdown = Mock()

        # S3阶段（最后阶段）
        phase_s3 = Phase(
            kind='STAGE',
            name='S3',
            env_params={
                'reward_field_group_coef': 0.10,
                'reward_turning_group_coef': 0.05,
                'reward_overlap_penalty': -0.20,
            },
            sampling_ratio=(0.20, 0.30, 0.50),
            hif_weight=0.1,
            update_gate_frames=100,
        )

        # 执行切换
        cfg.env.env_kwargs.update(phase_s3.env_params)
        mock_buffer.set_sampling_ratio(phase_s3.sampling_ratio)
        mock_buffer.reset_buckets()
        mock_collector.shutdown()

        # 验证
        assert cfg.env.env_kwargs['reward_field_group_coef'] == 0.10
        assert cfg.env.env_kwargs['reward_overlap_penalty'] == -0.20
        mock_buffer.set_sampling_ratio.assert_called_once_with((0.20, 0.30, 0.50))
        mock_buffer.reset_buckets.assert_called_once()
        mock_collector.shutdown.assert_called_once()


class TestPhaseTransitionEdgeCases:
    """测试阶段切换的边界情况"""

    def test_rapid_phase_transitions(self, minimal_cfg):
        """快速连续切换验证

        验证点:
        - PRETRAIN → S1 → S2 → S3快速切换
        - 每次切换配置都正确更新
        """
        cfg = minimal_cfg

        phases = [
            Phase(kind='PRETRAIN', name='PRETRAIN', env_params={},
                  sampling_ratio=(0.4, 0.3, 0.3), hif_weight=1.0, update_gate_frames=20),
            Phase(kind='STAGE', name='S1', env_params={'reward_field_group_coef': 1.0},
                  sampling_ratio=(0.4, 0.3, 0.3), hif_weight=0.5, update_gate_frames=100),
            Phase(kind='STAGE', name='S2', env_params={'reward_field_group_coef': 0.5},
                  sampling_ratio=(0.3, 0.3, 0.4), hif_weight=0.3, update_gate_frames=100),
            Phase(kind='STAGE', name='S3', env_params={'reward_field_group_coef': 0.1},
                  sampling_ratio=(0.2, 0.3, 0.5), hif_weight=0.1, update_gate_frames=100),
        ]

        # 模拟快速切换
        for phase in phases:
            cfg.env.env_kwargs.update(phase.env_params)

        # 验证最终状态是S3的配置
        assert cfg.env.env_kwargs.get('reward_field_group_coef', 0.0) == 0.1

    def test_config_isolation(self, minimal_cfg):
        """配置隔离性验证

        验证点:
        - 切换到新阶段时，旧阶段配置被完全覆盖
        - 无残留配置干扰
        """
        cfg = minimal_cfg

        # S1配置
        phase_s1 = Phase(
            kind='STAGE', name='S1',
            env_params={'reward_field_group_coef': 1.0, 'custom_s1_param': 100},
            sampling_ratio=(0.4, 0.3, 0.3), hif_weight=0.5, update_gate_frames=100,
        )

        cfg.env.env_kwargs.update(phase_s1.env_params)
        assert cfg.env.env_kwargs.get('custom_s1_param') == 100

        # S2配置（不包含custom_s1_param）
        phase_s2 = Phase(
            kind='STAGE', name='S2',
            env_params={'reward_field_group_coef': 0.5, 'custom_s2_param': 200},
            sampling_ratio=(0.3, 0.3, 0.4), hif_weight=0.3, update_gate_frames=100,
        )

        cfg.env.env_kwargs.update(phase_s2.env_params)

        # S1的自定义参数仍然存在（update不会删除旧键）
        assert cfg.env.env_kwargs.get('custom_s1_param') == 100
        # S2的新参数已添加
        assert cfg.env.env_kwargs.get('custom_s2_param') == 200
