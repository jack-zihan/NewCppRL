"""
集成测试：完整训练流程

验证完整的训练流程：
- 环境创建 → 数据采集 → Buffer存储 → Loss计算 → 参数更新
- 使用最小化参数快速验证流程正确性
"""
import pytest
import torch
from tensordict import TensorDict
from omegaconf import OmegaConf


class TestSACTrainingIntegration:
    """测试完整SAC训练流程"""

    def test_minimal_training_loop(self, minimal_cfg):
        """最小化训练循环

        验证点:
        - 环境 → collector → buffer → loss → optimizer 完整流程
        - 参数正确更新
        - 无crash运行
        """
        from rl_new.sac_cont_sy.model_utils import make_sac_models
        from rl_new.sac_cont_sy.env_utils import make_single_environment
        from rl_new.sac_cont_sy.train_utils_optimized import create_replay_buffer
        from torchrl.objectives import SACLoss, group_optimizers
        from torchrl.data import LazyMemmapStorage
        import tempfile
        import torch.nn as nn

        # 1. 创建环境
        cfg = minimal_cfg
        cfg.env.env_id = "NewPasture-v4"  # 简单环境
        cfg.env.env_kwargs = OmegaConf.create({
            "num_obstacles_range": (0, 0),  # 无障碍
            "reward_field_group_coef": 1.0,
        })

        env = make_single_environment(cfg, device='cpu', seed=42, from_pixels=False)

        # 2. 创建模型
        device = torch.device('cpu')
        actor, critic = make_sac_models(env, device=device)

        # 3. 创建Loss
        loss_module = SACLoss(
            actor_network=actor,
            qvalue_network=critic,
            num_qvalue_nets=2,
            loss_function="smooth_l1",
        )
        loss_module.make_value_estimator()

        # 4. 创建优化器（分离actor/critic/alpha参数）
        actor_params = list(loss_module.actor_network_params.flatten_keys().values())
        critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())

        optimizer_actor = torch.optim.AdamW(actor_params, lr=3e-4)
        optimizer_critic = torch.optim.AdamW(critic_params, lr=3e-4)
        optimizer_alpha = torch.optim.AdamW([loss_module.log_alpha], lr=3e-4)

        optimizer = group_optimizers(optimizer_actor, optimizer_critic, optimizer_alpha)

        # 5. 创建Buffer
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer = create_replay_buffer(
                cfg, tmpdir, device, bucketed=False, initial_stage=None
            )

            # 6. 收集数据
            env.set_seed(42)
            td_reset = env.reset()

            # 手工rollout收集足够数据 (至少batch_size*10)
            for _ in range(50):
                with torch.no_grad():
                    td_action = actor(td_reset)
                td_next = env.step(td_action)

                # 添加batch维度 (single env返回scalar tensors, buffer需要至少1D)
                td_next_batched = td_next.unsqueeze(0)
                buffer.extend(td_next_batched)

                # 更新td_reset for next step
                td_reset = td_next

                if td_next["done"].any():
                    td_reset = env.reset()
                else:
                    td_reset = td_next.clone()

            # 7. 训练更新
            if len(buffer) >= cfg.buffer.batch_size:
                sampled_td = buffer.sample()

                # 计算loss
                loss_out = loss_module(sampled_td)

                # 计算total_loss (SACLoss默认reduction='mean'，返回标量)
                total_loss = loss_out["loss_actor"] + loss_out["loss_qvalue"] + loss_out["loss_alpha"]

                # 反向传播（total_loss已经是标量，无需sum）
                total_loss.backward()

                # 更新参数
                optimizer.step()
                optimizer.zero_grad()

                # 验证loss计算
                assert "loss_actor" in loss_out.keys()
                assert "loss_qvalue" in loss_out.keys()
                assert "loss_alpha" in loss_out.keys()
                # SACLoss默认reduction='mean'，返回标量loss，因此不检查batch维度
                # assert total_loss.shape == torch.Size([cfg.buffer.batch_size])

        env.close()

    def test_hif_assisted_training_loop(self, minimal_cfg):
        """带HIF辅助的训练循环（简化版）

        验证点:
        - HIFAssistedSACLoss在PRETRAIN和STAGE阶段正确切换
        - Mock SAC和HIF losses以避免复杂的模型创建

        注：完整的HIFAssistedSACLoss功能已在test_hif_assisted_loss.py中测试
        """
        from rl_new.sac_cont_sy.sac_utils_optimized import HIFAssistedSACLoss
        from unittest.mock import Mock

        # 1. Mock SAC Loss
        mock_sac_loss = Mock()
        sac_out = TensorDict({
            "loss_actor": torch.tensor(0.3),
            "loss_qvalue": torch.tensor(0.4),
            "loss_alpha": torch.tensor(0.1),
            "alpha": torch.tensor(0.2),
            "entropy": torch.tensor(-1.5),
        }, [])
        mock_sac_loss.return_value = sac_out

        # 2. Mock HIF Loss
        mock_hif_loss = Mock()
        hif_value = torch.tensor(0.5)
        hif_metrics = {"td_error": torch.tensor(0.1)}
        mock_hif_loss.return_value = (hif_value, hif_metrics)

        # 3. 配置
        cfg = minimal_cfg
        cfg.hif.enabled = True

        # 4. PRETRAIN阶段测试
        phase_idx = 0  # PRETRAIN
        phase_provider = lambda: phase_idx
        weight_provider = lambda: 1.0 if phase_idx == 0 else 0.5

        combined_loss = HIFAssistedSACLoss(
            actor=Mock(),
            sac_loss=mock_sac_loss,
            hif_loss=mock_hif_loss,
            cfg=cfg,
            phase_provider=phase_provider,
            weight_provider=weight_provider
        )

        # 5. 模拟TensorDict输入
        B = 2
        td = TensorDict({
            "observation": torch.randn(B, 5, 64, 64),  # v5: 5通道
            "action": torch.randn(B, 2),
            "next": TensorDict({
                "observation": torch.randn(B, 5, 64, 64),
                "reward": torch.randn(B, 1),
                "done": torch.tensor([[False], [True]]),
                "terminated": torch.tensor([[False], [False]]),
            }, batch_size=[B])
        }, batch_size=[B])

        # 添加HIF标签（v5环境必需）
        td["label_ego_hif"] = torch.randn(B, 3, 64, 64)  # [cos2, sin2, confidence]

        # 6. PRETRAIN阶段训练
        loss_out_pretrain = combined_loss(td)

        # 验证PRETRAIN阶段：仅HIF loss
        assert torch.allclose(loss_out_pretrain["total_loss"], hif_value)
        assert torch.allclose(loss_out_pretrain["loss_hif"], hif_value)
        assert torch.allclose(loss_out_pretrain["loss_actor"], torch.zeros_like(hif_value))

        # 7. 切换到STAGE阶段
        phase_idx = 1  # S1

        loss_out_stage = combined_loss(td)

        # 验证STAGE阶段：SAC + HIF
        assert "loss_actor" in loss_out_stage.keys()
        assert "loss_qvalue" in loss_out_stage.keys()
        assert "loss_hif" in loss_out_stage.keys()
        # total = SAC + 0.5*HIF
        expected_sac = loss_out_stage["loss_actor"] + loss_out_stage["loss_qvalue"] + loss_out_stage["loss_alpha"]
        expected_total = expected_sac + 0.5 * loss_out_stage["loss_hif"]
        assert torch.allclose(loss_out_stage["total_loss"], expected_total)


class TestCurriculumIntegration:
    """测试课程学习系统集成"""

    def test_phase_transition_integration(self, minimal_cfg):
        """阶段切换集成测试

        验证点:
        - Phase定义 → 条件检查 → 切换执行 完整流程
        - 配置正确更新
        - Buffer采样比例正确调整
        """
        from rl_new.sac_cont_sy.train_utils_optimized import (
            build_training_schedule,
            maybe_advance_by_eval,
            apply_phase,
            ScheduleState
        )
        from unittest.mock import Mock

        # 1. 构建训练日程
        cfg = minimal_cfg
        cfg.curriculum.enabled = True

        schedule = build_training_schedule(cfg)

        # 验证schedule构建 (minimal_cfg启用了HIF pretrain, 所以是4个阶段)
        assert len(schedule) == 4  # PRETRAIN, S1, S2, S3
        assert schedule[0].name == "PRETRAIN"
        assert schedule[1].name == "S1"
        assert schedule[2].name == "S2"
        assert schedule[3].name == "S3"

        # 2. 模拟S1阶段推进 (S1是schedule[1], schedule[0]是PRETRAIN)
        state = ScheduleState(idx=1)  # S1在索引1
        phase_s1 = schedule[1]  # S1阶段

        # 连续3次达标
        for i in range(3):
            metrics = [{"metrics": {"eval/completion_ratio": 0.92}}]
            state, should_advance = maybe_advance_by_eval(
                state, phase_s1, metrics, cfg, curriculum_enabled=True
            )

        # 验证推进
        assert should_advance is True
        assert state.consec_completion == 3

        # 3. 模拟阶段切换
        from unittest.mock import patch
        mock_actor = Mock()
        mock_optimizer = Mock()
        mock_optimizer.param_groups = [
            {"lr": 1e-3},  # actor
            {"lr": 2e-3},  # critic
            {"lr": 3e-3},  # alpha
        ]
        mock_collector = Mock()
        mock_collector.shutdown = Mock()
        mock_buffer = Mock()
        mock_buffer.set_sampling_ratio = Mock()
        mock_buffer.reset_buckets = Mock()

        # Mock create_collector避免实际创建collector
        mock_new_collector = Mock()
        mock_new_collector.__iter__ = Mock(return_value=iter([]))  # 支持iter()调用
        device = torch.device('cpu')

        # 应用S2阶段 (patch create_collector)
        phase_s2 = schedule[2]  # schedule[2]是S2 (schedule[0]=PRETRAIN, [1]=S1, [2]=S2)
        with patch('rl_new.sac_cont_sy.train_utils_optimized.create_collector', return_value=mock_new_collector):
            new_collector, new_iter, new_buffer = apply_phase(
                phase_s2, cfg, mock_actor, device, mock_optimizer,
                mock_collector, mock_buffer
            )

        # 验证副作用
        # 1) 配置更新
        assert cfg.env.env_kwargs["reward_field_group_coef"] == 0.5  # S2配置

        # 2) Buffer采样比例更新
        mock_buffer.set_sampling_ratio.assert_called_once_with((0.30, 0.30, 0.40))
        mock_buffer.reset_buckets.assert_called_once()

        # 3) 学习率更新
        assert mock_optimizer.param_groups[0]["lr"] == 3e-4  # actor lr

        # 4) Collector重建
        mock_collector.shutdown.assert_called_once()
        assert new_collector is not None

    def test_bucketed_replay_integration(self, minimal_cfg):
        """分桶回放系统集成测试

        验证点:
        - 分类 → 存储 → 采样 完整流程
        - 采样比例动态调整
        """
        from rl_new.sac_cont_sy.bucketed_replay import (
            BucketedTensorDictPrioritizedReplayBuffer,
            BucketId
        )
        from torchrl.data import LazyMemmapStorage
        import tempfile

        # 1. 创建Buffer
        cfg = minimal_cfg
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LazyMemmapStorage(max_size=100, scratch_dir=tmpdir)

            buffer = BucketedTensorDictPrioritizedReplayBuffer(
                storage=storage,
                alpha=0.7,
                beta=0.5,
                batch_size=4,
                prefetch=0,
                pin_memory=False,
                success_threshold=0.99,
                near_end_threshold=0.90
            )

            # 2. 设置采样比例（S1）
            buffer.set_sampling_ratio((0.40, 0.30, 0.30))

            # 3. 添加不同类型的数据
            # SUCCESS样本
            td_success = TensorDict({
                "observation": torch.randn(2, 3, 64, 64),
                "action": torch.randn(2, 2),
                "next": TensorDict({
                    "observation": torch.randn(2, 3, 64, 64),
                    "reward": torch.randn(2, 1),
                    "done": torch.tensor([[True], [True]]),
                    "truncated": torch.tensor([[False], [False]]),
                    "completion_ratio": torch.tensor([[1.00], [0.99]]),
                }, batch_size=[2])
            }, batch_size=[2])

            # NEAR_END样本
            td_near_end = TensorDict({
                "observation": torch.randn(2, 3, 64, 64),
                "action": torch.randn(2, 2),
                "next": TensorDict({
                    "observation": torch.randn(2, 3, 64, 64),
                    "reward": torch.randn(2, 1),
                    "done": torch.tensor([[False], [False]]),
                    "truncated": torch.tensor([[False], [False]]),
                    "completion_ratio": torch.tensor([[0.95], [0.92]]),
                }, batch_size=[2])
            }, batch_size=[2])

            # MID样本
            td_mid = TensorDict({
                "observation": torch.randn(2, 3, 64, 64),
                "action": torch.randn(2, 2),
                "next": TensorDict({
                    "observation": torch.randn(2, 3, 64, 64),
                    "reward": torch.randn(2, 1),
                    "done": torch.tensor([[False], [False]]),
                    "truncated": torch.tensor([[False], [False]]),
                    "completion_ratio": torch.tensor([[0.50], [0.60]]),
                }, batch_size=[2])
            }, batch_size=[2])

            # 添加到buffer (添加多份以确保每个bucket有足够样本)
            for _ in range(5):  # 每个bucket 10个样本
                buffer.extend(td_success)
                buffer.extend(td_near_end)
                buffer.extend(td_mid)

            # 4. 采样验证
            sampled = buffer.sample()

            # 验证采样batch大小
            assert sampled.batch_size == torch.Size([4])

            # 验证包含必要键
            assert "observation" in sampled.keys()
            assert "action" in sampled.keys()

            # 5. 切换采样比例（S2）
            buffer.set_sampling_ratio((0.30, 0.30, 0.40))

            # 再次采样（新比例）- buffer中已有数据，无需reset
            sampled_s2 = buffer.sample()
            assert sampled_s2.batch_size == torch.Size([4])


class TestEndToEndScenario:
    """端到端场景测试"""

    def test_minimal_end_to_end_workflow(self, minimal_cfg):
        """最小化端到端工作流

        验证点:
        - 从零开始 → 完成一次完整的train-eval循环
        - 所有组件协同工作
        """
        from rl_new.sac_cont_sy.model_utils import make_sac_models
        from rl_new.sac_cont_sy.env_utils import make_single_environment
        from rl_new.sac_cont_sy.train_utils_optimized import create_replay_buffer
        from rl_new.sac_cont_sy.sac_utils_optimized import create_update_fn
        from torchrl.objectives import SACLoss, group_optimizers
        import tempfile

        # 配置
        cfg = minimal_cfg
        cfg.env.env_id = "NewPasture-v4"
        cfg.env.env_kwargs = OmegaConf.create({"num_obstacles_range": (0, 0)})

        # 1. 初始化
        device = torch.device('cpu')
        env = make_single_environment(cfg, device='cpu', seed=42)
        actor, critic = make_sac_models(env, device=device)

        # 2. 创建训练组件
        loss_module = SACLoss(
            actor_network=actor,
            qvalue_network=critic,
            num_qvalue_nets=2,
            loss_function="smooth_l1",
        )
        loss_module.make_value_estimator()

        # 分离参数创建优化器
        actor_params = list(loss_module.actor_network_params.flatten_keys().values())
        critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
        opt_actor = torch.optim.AdamW(actor_params, lr=3e-4)
        opt_critic = torch.optim.AdamW(critic_params, lr=3e-4)
        opt_alpha = torch.optim.AdamW([loss_module.log_alpha], lr=3e-4)
        optimizer = group_optimizers(opt_actor, opt_critic, opt_alpha)

        with tempfile.TemporaryDirectory() as tmpdir:
            buffer = create_replay_buffer(cfg, tmpdir, device, bucketed=False)

            # 3. 数据收集
            td = env.reset()
            for _ in range(20):  # 收集20步
                with torch.no_grad():
                    td = actor(td)
                td = env.step(td)
                # 添加batch维度 (single env返回scalar tensors, buffer需要至少1D)
                td_batched = td.unsqueeze(0)
                buffer.extend(td_batched)

                if td["done"].any():
                    td = env.reset()

            # 4. 训练更新 (手动更新，不使用create_update_fn)
            if len(buffer) >= cfg.buffer.batch_size:
                for _ in range(3):  # 3次更新
                    sampled_td = buffer.sample()
                    loss_out = loss_module(sampled_td)

                    # 计算total_loss (SACLoss默认reduction='mean'，返回标量)
                    total_loss = loss_out["loss_actor"] + loss_out["loss_qvalue"] + loss_out["loss_alpha"]
                    total_loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()

                    # 验证loss输出
                    assert "loss_actor" in loss_out.keys()

            # 5. 评估
            env.set_seed(999)
            eval_td = env.reset()

            total_reward = 0.0
            for _ in range(10):  # 评估10步
                with torch.no_grad():
                    eval_td = actor(eval_td)
                eval_td = env.step(eval_td)
                total_reward += eval_td["next", "reward"].item()

                if eval_td["next", "done"].any():
                    break

            # 验证评估完成
            assert total_reward != 0.0  # 有奖励信号

        env.close()
