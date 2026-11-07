"""
测试HIFAssistedSACLoss损失函数

验证阶段驱动的损失计算逻辑：
- PRETRAIN阶段（phase=0）：仅HIF损失
- 训练阶段（phase>0）：SAC损失 + 可选HIF辅助损失
- 权重提供器回调系统
- HIF权重解析和映射
"""
import pytest
import torch
from unittest.mock import Mock
from omegaconf import OmegaConf
from tensordict import TensorDict


class TestHIFAssistedLossPhaseLogic:
    """测试阶段驱动的损失计算逻辑"""

    def test_pretrain_phase_hif_only(self, minimal_cfg):
        """PRETRAIN阶段（phase=0）：仅计算HIF损失

        验证点:
        - phase_provider() == 0 → 仅运行HIF损失
        - SAC损失项全部为0
        - total_loss == loss_hif
        """
        from rl_new.sac_cont_sy.sac_utils_optimized import HIFAssistedSACLoss

        # Mock组件
        mock_actor = Mock()
        mock_sac_loss = Mock()
        mock_hif_loss = Mock()

        # phase_provider返回0（PRETRAIN）
        phase_provider = Mock(return_value=0)

        # HIF损失返回值
        hif_value = torch.tensor(0.5)
        hif_metrics = {"td_error": torch.tensor(0.1)}
        mock_hif_loss.return_value = (hif_value, hif_metrics)

        # 创建HIFAssistedLoss
        cfg = minimal_cfg
        cfg.hif.enabled = True
        loss_module = HIFAssistedSACLoss(
            actor=mock_actor,
            sac_loss=mock_sac_loss,
            hif_loss=mock_hif_loss,
            cfg=cfg,
            phase_provider=phase_provider,
            weight_provider=None
        )

        # 创建输入TensorDict
        td = TensorDict({
            "observation": torch.randn(2, 3, 64, 64),
            "action": torch.randn(2, 2),
        }, batch_size=[2])

        # 执行forward
        out = loss_module(td)

        # 验证PRETRAIN阶段逻辑
        assert phase_provider.call_count == 1
        assert mock_hif_loss.call_count == 1
        assert mock_sac_loss.call_count == 0  # SAC loss不应被调用

        # 验证输出
        assert "total_loss" in out.keys()
        assert torch.allclose(out["total_loss"], hif_value)
        assert torch.allclose(out["loss_hif"], hif_value)

        # SAC损失项应为0
        assert torch.allclose(out["loss_actor"], torch.zeros_like(hif_value))
        assert torch.allclose(out["loss_qvalue"], torch.zeros_like(hif_value))
        assert torch.allclose(out["loss_alpha"], torch.zeros_like(hif_value))

        # HIF metrics应被传递
        assert "td_error" in out.keys()
        assert torch.allclose(out["td_error"], hif_metrics["td_error"])

    def test_stage_phase_sac_only_without_hif(self, minimal_cfg):
        """训练阶段（phase>0）无HIF：仅SAC损失

        验证点:
        - phase_provider() > 0 且 cfg.hif.enabled=False → 仅SAC
        - total_loss = loss_actor + loss_qvalue + loss_alpha
        - 无loss_hif字段
        """
        from rl_new.sac_cont_sy.sac_utils_optimized import HIFAssistedSACLoss

        # Mock SAC损失
        mock_sac_loss = Mock()
        sac_out = TensorDict({
            "loss_actor": torch.tensor(0.3),
            "loss_qvalue": torch.tensor(0.4),
            "loss_alpha": torch.tensor(0.1),
            "alpha": torch.tensor(0.2),
            "entropy": torch.tensor(-1.5),
        }, [])
        mock_sac_loss.return_value = sac_out

        # phase=1（S1阶段），HIF disabled
        phase_provider = Mock(return_value=1)

        cfg = minimal_cfg
        cfg.hif.enabled = False  # 禁用HIF

        loss_module = HIFAssistedSACLoss(
            actor=Mock(),
            sac_loss=mock_sac_loss,
            hif_loss=None,  # 无HIF损失
            cfg=cfg,
            phase_provider=phase_provider,
            weight_provider=None
        )

        # 执行forward
        td = TensorDict({"observation": torch.randn(2, 3, 64, 64)}, batch_size=[2])
        out = loss_module(td)

        # 验证SAC损失
        assert mock_sac_loss.call_count == 1
        expected_total = sac_out["loss_actor"] + sac_out["loss_qvalue"] + sac_out["loss_alpha"]
        assert torch.allclose(out["total_loss"], expected_total)

        # 无HIF损失
        assert "loss_hif" not in out.keys()

    def test_stage_phase_sac_with_hif_assistance(self, minimal_cfg):
        """训练阶段（phase>0）有HIF：SAC + weight*HIF

        验证点:
        - total_loss = SAC_total + weight * loss_hif
        - weight从weight_provider获取
        - 所有损失项都正确传递
        """
        from rl_new.sac_cont_sy.sac_utils_optimized import HIFAssistedSACLoss

        # Mock SAC损失
        mock_sac_loss = Mock()
        sac_out = TensorDict({
            "loss_actor": torch.tensor(0.3),
            "loss_qvalue": torch.tensor(0.4),
            "loss_alpha": torch.tensor(0.1),
            "alpha": torch.tensor(0.2),
            "entropy": torch.tensor(-1.5),
        }, [])
        mock_sac_loss.return_value = sac_out

        # Mock HIF损失
        mock_hif_loss = Mock()
        hif_value = torch.tensor(0.6)
        hif_metrics = {"hif_metric1": torch.tensor(0.05)}
        mock_hif_loss.return_value = (hif_value, hif_metrics)

        # weight_provider返回0.5（S1阶段权重）
        weight_provider = Mock(return_value=0.5)
        phase_provider = Mock(return_value=1)  # S1阶段

        cfg = minimal_cfg
        cfg.hif.enabled = True

        loss_module = HIFAssistedSACLoss(
            actor=Mock(),
            sac_loss=mock_sac_loss,
            hif_loss=mock_hif_loss,
            cfg=cfg,
            phase_provider=phase_provider,
            weight_provider=weight_provider
        )

        # 执行forward
        td = TensorDict({"observation": torch.randn(2, 3, 64, 64)}, batch_size=[2])
        out = loss_module(td)

        # 验证损失计算
        sac_total = sac_out["loss_actor"] + sac_out["loss_qvalue"] + sac_out["loss_alpha"]
        expected_total = sac_total + 0.5 * hif_value

        assert torch.allclose(out["total_loss"], expected_total)
        assert torch.allclose(out["loss_hif"], hif_value)
        assert "hif_metric1" in out.keys()


class TestHIFWeightResolution:
    """测试HIF权重解析逻辑"""

    def test_hif_weight_from_provider_pretrain(self, minimal_cfg):
        """PRETRAIN阶段：current_hif_weight固定为1.0"""
        from rl_new.sac_cont_sy.sac_utils_optimized import HIFAssistedSACLoss

        phase_provider = Mock(return_value=0)  # PRETRAIN

        cfg = minimal_cfg
        cfg.hif.enabled = True

        loss_module = HIFAssistedSACLoss(
            actor=Mock(),
            sac_loss=Mock(),
            hif_loss=Mock(),
            cfg=cfg,
            phase_provider=phase_provider,
            weight_provider=None  # 无provider
        )

        # PRETRAIN阶段固定为1.0
        assert loss_module.current_hif_weight == 1.0

    def test_hif_weight_from_provider_stage(self, minimal_cfg):
        """训练阶段：优先从weight_provider读取"""
        from rl_new.sac_cont_sy.sac_utils_optimized import HIFAssistedSACLoss

        phase_provider = Mock(return_value=1)  # S1
        weight_provider = Mock(return_value=0.5)

        cfg = minimal_cfg
        cfg.hif.enabled = True
        cfg.hif.weights = OmegaConf.create({"S1": 0.8})  # 配置中是0.8

        loss_module = HIFAssistedSACLoss(
            actor=Mock(),
            sac_loss=Mock(),
            hif_loss=Mock(),
            cfg=cfg,
            phase_provider=phase_provider,
            weight_provider=weight_provider
        )

        # 优先使用weight_provider的0.5，而非配置的0.8
        assert loss_module.current_hif_weight == 0.5

    def test_hif_weight_from_config_mapping(self, minimal_cfg):
        """训练阶段：无provider时从配置映射读取"""
        from rl_new.sac_cont_sy.sac_utils_optimized import HIFAssistedSACLoss

        cfg = minimal_cfg
        cfg.hif.enabled = True
        cfg.hif.weights = OmegaConf.create({
            "S1": 0.5,
            "S2": 0.3,
            "S3": 0.1
        })

        # S1阶段
        phase_provider_s1 = Mock(return_value=1)
        loss_module_s1 = HIFAssistedSACLoss(
            actor=Mock(), sac_loss=Mock(), hif_loss=Mock(),
            cfg=cfg, phase_provider=phase_provider_s1, weight_provider=None
        )
        assert loss_module_s1.current_hif_weight == 0.5

        # S2阶段
        phase_provider_s2 = Mock(return_value=2)
        loss_module_s2 = HIFAssistedSACLoss(
            actor=Mock(), sac_loss=Mock(), hif_loss=Mock(),
            cfg=cfg, phase_provider=phase_provider_s2, weight_provider=None
        )
        assert loss_module_s2.current_hif_weight == 0.3

        # S3阶段
        phase_provider_s3 = Mock(return_value=3)
        loss_module_s3 = HIFAssistedSACLoss(
            actor=Mock(), sac_loss=Mock(), hif_loss=Mock(),
            cfg=cfg, phase_provider=phase_provider_s3, weight_provider=None
        )
        assert loss_module_s3.current_hif_weight == 0.1

    def test_hif_weight_defaults_to_zero(self, minimal_cfg):
        """无provider且无配置映射：默认为0.0"""
        from rl_new.sac_cont_sy.sac_utils_optimized import HIFAssistedSACLoss

        cfg = minimal_cfg
        cfg.hif.enabled = True
        # 删除weights配置以测试默认行为
        if 'weights' in cfg.hif:
            del cfg.hif['weights']

        phase_provider = Mock(return_value=1)  # S1

        loss_module = HIFAssistedSACLoss(
            actor=Mock(), sac_loss=Mock(), hif_loss=Mock(),
            cfg=cfg, phase_provider=phase_provider, weight_provider=None
        )

        # 无配置时默认为0.0
        assert loss_module.current_hif_weight == 0.0


class TestHIFAssistedLossEdgeCases:
    """测试边界情况和异常处理"""

    def test_pretrain_without_hif_module_raises(self, minimal_cfg):
        """PRETRAIN阶段缺少HIF模块应抛出异常"""
        from rl_new.sac_cont_sy.sac_utils_optimized import HIFAssistedSACLoss

        phase_provider = Mock(return_value=0)  # PRETRAIN

        cfg = minimal_cfg
        cfg.hif.enabled = True

        loss_module = HIFAssistedSACLoss(
            actor=Mock(),
            sac_loss=Mock(),
            hif_loss=None,  # 缺少HIF模块！
            cfg=cfg,
            phase_provider=phase_provider,
            weight_provider=None
        )

        # PRETRAIN需要HIF模块
        td = TensorDict({"observation": torch.randn(2, 3, 64, 64)}, batch_size=[2])

        with pytest.raises(ValueError, match="PRETRAIN phase requires HIF loss module"):
            loss_module(td)

    def test_stage_phase_zero_hif_weight(self, minimal_cfg):
        """训练阶段HIF权重为0时不调用HIF损失"""
        from rl_new.sac_cont_sy.sac_utils_optimized import HIFAssistedSACLoss

        mock_sac_loss = Mock()
        sac_out = TensorDict({
            "loss_actor": torch.tensor(0.3),
            "loss_qvalue": torch.tensor(0.4),
            "loss_alpha": torch.tensor(0.1),
            "alpha": torch.tensor(0.2),
            "entropy": torch.tensor(-1.5),
        }, [])
        mock_sac_loss.return_value = sac_out

        mock_hif_loss = Mock()
        weight_provider = Mock(return_value=0.0)  # 权重为0
        phase_provider = Mock(return_value=1)

        cfg = minimal_cfg
        cfg.hif.enabled = True

        loss_module = HIFAssistedSACLoss(
            actor=Mock(),
            sac_loss=mock_sac_loss,
            hif_loss=mock_hif_loss,
            cfg=cfg,
            phase_provider=phase_provider,
            weight_provider=weight_provider
        )

        # 执行forward
        td = TensorDict({"observation": torch.randn(2, 3, 64, 64)}, batch_size=[2])
        out = loss_module(td)

        # HIF权重为0，不应调用HIF损失
        assert mock_hif_loss.call_count == 0
        assert "loss_hif" not in out.keys()

        # total_loss仅为SAC损失
        sac_total = sac_out["loss_actor"] + sac_out["loss_qvalue"] + sac_out["loss_alpha"]
        assert torch.allclose(out["total_loss"], sac_total)
