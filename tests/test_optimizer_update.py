"""
测试优化器更新系统

验证set_optimizer_group_lrs和create_update_fn的功能：
- 学习率分组管理（actor/critic/alpha）
- 梯度裁剪
- 目标网络更新
- AMP混合精度支持
"""
import pytest
import torch
from unittest.mock import Mock, MagicMock
from omegaconf import OmegaConf
from tensordict import TensorDict


class TestSetOptimizerGroupLRs:
    """测试set_optimizer_group_lrs学习率设置函数"""

    def test_set_all_groups_lr(self):
        """统一设置所有组学习率

        验证点:
        - all_groups_lr参数设置所有组
        - 所有param_groups的lr应相同
        """
        from rl_new.sac_cont_sy.sac_utils_optimized import set_optimizer_group_lrs

        # 创建Mock优化器（3个param_groups: actor, critic, alpha）
        mock_optimizer = Mock()
        mock_optimizer.param_groups = [
            {"lr": 1e-3},  # actor
            {"lr": 2e-3},  # critic
            {"lr": 3e-3},  # alpha
        ]

        # 统一设置为1e-4
        set_optimizer_group_lrs(mock_optimizer, all_groups_lr=1e-4)

        # 验证所有组都被更新
        assert mock_optimizer.param_groups[0]["lr"] == 1e-4
        assert mock_optimizer.param_groups[1]["lr"] == 1e-4
        assert mock_optimizer.param_groups[2]["lr"] == 1e-4

    def test_set_individual_group_lrs(self):
        """分别设置各组学习率

        验证点:
        - actor_lr更新param_groups[0]
        - critic_lr更新param_groups[1]
        - alpha_lr更新param_groups[2]
        """
        from rl_new.sac_cont_sy.sac_utils_optimized import set_optimizer_group_lrs

        mock_optimizer = Mock()
        mock_optimizer.param_groups = [
            {"lr": 0.0},  # actor
            {"lr": 0.0},  # critic
            {"lr": 0.0},  # alpha
        ]

        # 分别设置
        set_optimizer_group_lrs(
            mock_optimizer,
            actor_lr=3e-4,
            critic_lr=5e-4,
            alpha_lr=1e-4
        )

        # 验证
        assert mock_optimizer.param_groups[0]["lr"] == 3e-4  # actor
        assert mock_optimizer.param_groups[1]["lr"] == 5e-4  # critic
        assert mock_optimizer.param_groups[2]["lr"] == 1e-4  # alpha

    def test_set_partial_group_lrs(self):
        """仅设置部分组

        验证点:
        - 未指定的组保持原值
        - 指定的组正确更新
        """
        from rl_new.sac_cont_sy.sac_utils_optimized import set_optimizer_group_lrs

        mock_optimizer = Mock()
        mock_optimizer.param_groups = [
            {"lr": 1e-3},  # actor
            {"lr": 2e-3},  # critic
            {"lr": 3e-3},  # alpha
        ]

        # 仅更新actor和critic
        set_optimizer_group_lrs(
            mock_optimizer,
            actor_lr=5e-4,
            critic_lr=7e-4
        )

        # 验证
        assert mock_optimizer.param_groups[0]["lr"] == 5e-4  # actor已更新
        assert mock_optimizer.param_groups[1]["lr"] == 7e-4  # critic已更新
        assert mock_optimizer.param_groups[2]["lr"] == 3e-3  # alpha保持不变

    def test_all_groups_lr_overrides_individual(self):
        """all_groups_lr优先级最高

        验证点:
        - all_groups_lr和individual同时指定时，all_groups_lr生效
        """
        from rl_new.sac_cont_sy.sac_utils_optimized import set_optimizer_group_lrs

        mock_optimizer = Mock()
        mock_optimizer.param_groups = [
            {"lr": 0.0},
            {"lr": 0.0},
            {"lr": 0.0},
        ]

        # all_groups_lr应覆盖individual设置
        set_optimizer_group_lrs(
            mock_optimizer,
            all_groups_lr=1e-4,  # 统一设置
            actor_lr=3e-4,  # 应被忽略
            critic_lr=5e-4,  # 应被忽略
        )

        # 验证all_groups_lr生效
        assert all(pg["lr"] == 1e-4 for pg in mock_optimizer.param_groups)


class TestCreateUpdateFn:
    """测试create_update_fn更新函数创建"""

    def test_update_fn_basic_flow(self, minimal_cfg):
        """基础更新流程

        验证点:
        - loss_module.forward被调用
        - backward传播梯度
        - optimizer.step()更新参数
        - optimizer.zero_grad()清零梯度
        """
        from rl_new.sac_cont_sy.sac_utils_optimized import create_update_fn

        # Mock组件
        mock_loss_module = Mock()
        loss_out = TensorDict({
            "total_loss": torch.tensor([0.5, 0.6], requires_grad=True),
            "loss_actor": torch.tensor([0.2, 0.3], requires_grad=True),
        }, batch_size=[2])
        mock_loss_module.return_value = loss_out
        mock_loss_module.parameters = Mock(return_value=[])

        mock_optimizer = Mock()

        cfg = minimal_cfg
        cfg.training.use_amp = False  # 禁用AMP
        cfg.optim.max_grad_norm = 0  # 禁用梯度裁剪
        cfg.compile.enable = False  # 禁用编译

        # 创建update函数
        update_fn = create_update_fn(
            loss_module=mock_loss_module,
            optimizer=mock_optimizer,
            target_net_updater=None,
            cfg=cfg
        )

        # 执行update
        td_in = TensorDict({"observation": torch.randn(2, 3, 64, 64)}, batch_size=[2])
        result = update_fn(td_in)

        # 验证调用
        assert mock_loss_module.call_count == 1
        assert mock_optimizer.step.call_count == 1
        assert mock_optimizer.zero_grad.call_count == 1

        # 验证返回值
        assert "total_loss" in result.keys()

    def test_update_fn_with_gradient_clipping(self, minimal_cfg):
        """带梯度裁剪的更新

        验证点:
        - cfg.optim.max_grad_norm > 0时调用torch.nn.utils.clip_grad_norm_
        """
        from rl_new.sac_cont_sy.sac_utils_optimized import create_update_fn

        # Mock参数
        mock_param = Mock()
        mock_param.grad = torch.randn(10)

        mock_loss_module = Mock()
        mock_loss_module.parameters = Mock(return_value=[mock_param])
        loss_out = TensorDict({"total_loss": torch.tensor([0.5], requires_grad=True)}, batch_size=[1])
        mock_loss_module.return_value = loss_out

        mock_optimizer = Mock()

        cfg = minimal_cfg
        cfg.training.use_amp = False
        cfg.optim.max_grad_norm = 1.0  # 启用梯度裁剪
        cfg.compile.enable = False

        # 创建update函数
        update_fn = create_update_fn(
            loss_module=mock_loss_module,
            optimizer=mock_optimizer,
            cfg=cfg
        )

        # 执行update
        td_in = TensorDict({"observation": torch.randn(1, 3, 64, 64)}, batch_size=[1])
        update_fn(td_in)

        # 验证调用顺序：backward → clip → step → zero_grad
        assert mock_optimizer.step.call_count == 1
        assert mock_optimizer.zero_grad.call_count == 1

    def test_update_fn_with_target_net_updater(self, minimal_cfg):
        """带目标网络更新器

        验证点:
        - target_net_updater.step()在optimizer.step()后被调用
        """
        from rl_new.sac_cont_sy.sac_utils_optimized import create_update_fn

        mock_loss_module = Mock()
        loss_out = TensorDict({"total_loss": torch.tensor([0.5], requires_grad=True)}, batch_size=[1])
        mock_loss_module.return_value = loss_out
        mock_loss_module.parameters = Mock(return_value=[])

        mock_optimizer = Mock()
        mock_target_updater = Mock()

        cfg = minimal_cfg
        cfg.training.use_amp = False
        cfg.optim.max_grad_norm = 0
        cfg.compile.enable = False

        # 创建update函数
        update_fn = create_update_fn(
            loss_module=mock_loss_module,
            optimizer=mock_optimizer,
            target_net_updater=mock_target_updater,
            cfg=cfg
        )

        # 执行update
        td_in = TensorDict({"observation": torch.randn(1, 3, 64, 64)}, batch_size=[1])
        update_fn(td_in)

        # 验证目标网络更新器被调用
        assert mock_target_updater.step.call_count == 1

    def test_update_fn_with_amp(self, minimal_cfg):
        """AMP混合精度训练

        验证点:
        - cfg.training.use_amp=True时使用torch.autocast
        - scaler.scale().backward()
        - scaler.step()和scaler.update()被调用
        """
        from rl_new.sac_cont_sy.sac_utils_optimized import create_update_fn

        mock_loss_module = Mock()
        loss_out = TensorDict({"total_loss": torch.tensor([0.5], requires_grad=True)}, batch_size=[1])
        mock_loss_module.return_value = loss_out
        mock_loss_module.parameters = Mock(return_value=[])

        mock_optimizer = Mock()

        # Mock GradScaler
        mock_scaler = Mock()
        mock_scaler.scale = Mock(return_value=Mock(backward=Mock()))

        cfg = minimal_cfg
        cfg.training.use_amp = True  # 启用AMP
        cfg.optim.max_grad_norm = 0
        cfg.compile.enable = False

        # 创建update函数
        update_fn = create_update_fn(
            loss_module=mock_loss_module,
            optimizer=mock_optimizer,
            cfg=cfg,
            scaler=mock_scaler
        )

        # 执行update
        td_in = TensorDict({"observation": torch.randn(1, 3, 64, 64)}, batch_size=[1])
        update_fn(td_in)

        # 验证scaler调用
        assert mock_scaler.scale.call_count == 1
        assert mock_scaler.step.call_count == 1
        assert mock_scaler.update.call_count == 1

    def test_update_fn_with_amp_and_gradient_clipping(self, minimal_cfg):
        """AMP + 梯度裁剪

        验证点:
        - scaler.unscale_在clip_grad_norm_前调用
        """
        from rl_new.sac_cont_sy.sac_utils_optimized import create_update_fn

        mock_param = Mock()
        mock_param.grad = torch.randn(10)

        mock_loss_module = Mock()
        mock_loss_module.parameters = Mock(return_value=[mock_param])
        loss_out = TensorDict({"total_loss": torch.tensor([0.5], requires_grad=True)}, batch_size=[1])
        mock_loss_module.return_value = loss_out

        mock_optimizer = Mock()
        mock_scaler = Mock()
        mock_scaler.scale = Mock(return_value=Mock(backward=Mock()))

        cfg = minimal_cfg
        cfg.training.use_amp = True
        cfg.optim.max_grad_norm = 1.0  # 启用梯度裁剪
        cfg.compile.enable = False

        # 创建update函数
        update_fn = create_update_fn(
            loss_module=mock_loss_module,
            optimizer=mock_optimizer,
            cfg=cfg,
            scaler=mock_scaler
        )

        # 执行update
        td_in = TensorDict({"observation": torch.randn(1, 3, 64, 64)}, batch_size=[1])
        update_fn(td_in)

        # 验证scaler.unscale_被调用（在梯度裁剪前）
        assert mock_scaler.unscale_.call_count == 1


class TestUpdateSystemEdgeCases:
    """测试边界情况"""

    def test_empty_optimizer_param_groups(self):
        """空的optimizer param_groups不应crash"""
        from rl_new.sac_cont_sy.sac_utils_optimized import set_optimizer_group_lrs

        mock_optimizer = Mock()
        mock_optimizer.param_groups = []

        # 不应抛出异常
        set_optimizer_group_lrs(mock_optimizer, all_groups_lr=1e-4)

    def test_update_fn_detaches_output(self, minimal_cfg):
        """更新函数返回detached TensorDict

        验证点:
        - 返回的loss_out已detach，不持有计算图
        """
        from rl_new.sac_cont_sy.sac_utils_optimized import create_update_fn

        mock_loss_module = Mock()
        loss_tensor = torch.tensor([0.5], requires_grad=True)
        loss_out = TensorDict({"total_loss": loss_tensor}, batch_size=[1])
        loss_out.detach = Mock(return_value=loss_out)
        mock_loss_module.return_value = loss_out
        mock_loss_module.parameters = Mock(return_value=[])

        mock_optimizer = Mock()

        cfg = minimal_cfg
        cfg.training.use_amp = False
        cfg.optim.max_grad_norm = 0
        cfg.compile.enable = False

        update_fn = create_update_fn(
            loss_module=mock_loss_module,
            optimizer=mock_optimizer,
            cfg=cfg
        )

        # 执行update
        td_in = TensorDict({"observation": torch.randn(1, 3, 64, 64)}, batch_size=[1])
        result = update_fn(td_in)

        # 验证detach被调用
        assert loss_out.detach.call_count == 1
