#!/usr/bin/env python3
"""验证HIF解耦功能的正确性

测试要点：
1. enable_hif=False时不创建decoder，节省显存
2. enable_hif=True时正常创建decoder
3. ActorWrapper根据enable_hif正确输出不同的键
4. Fail-fast机制正常工作
"""

import torch
import sys
sys.path.insert(0, '/home/lzh/NewCppRL')

from torchrl_utils.model.resnet_fpn_dual import ResNetFPNDualHeadActor
from rl_new.sac_cont_sy.model_utils import ActorWrapper


def test_decoder_conditional_init():
    """测试decoder条件初始化"""
    print("=" * 60)
    print("测试1：Decoder条件初始化")
    print("=" * 60)

    # 测试enable_hif=False：不创建decoder
    print("\n[1.1] enable_hif=False 测试")
    actor_no_hif = ResNetFPNDualHeadActor(
        in_channels=5, vec_dim=14, action_dim=2,
        enable_hif=False, hif_decoder_type='two_stage'
    )
    assert actor_no_hif.hif_decoder is None, "❌ decoder应该为None"
    print("✅ decoder未创建（hif_decoder=None）")

    # 测试enable_hif=True：创建decoder
    print("\n[1.2] enable_hif=True 测试")
    actor_with_hif = ResNetFPNDualHeadActor(
        in_channels=5, vec_dim=14, action_dim=2,
        enable_hif=True, hif_decoder_type='two_stage'
    )
    assert actor_with_hif.hif_decoder is not None, "❌ decoder应该被创建"
    print(f"✅ decoder已创建（类型：{type(actor_with_hif.hif_decoder).__name__}）")


def test_actor_wrapper_output_keys():
    """测试ActorWrapper输出键的正确性"""
    print("\n" + "=" * 60)
    print("测试2：ActorWrapper输出键")
    print("=" * 60)

    # 创建两个actor实例
    actor_no_hif = ResNetFPNDualHeadActor(
        in_channels=5, vec_dim=14, action_dim=2, enable_hif=False
    )
    actor_with_hif = ResNetFPNDualHeadActor(
        in_channels=5, vec_dim=14, action_dim=2, enable_hif=True
    )

    # 创建wrapper
    wrapper_no_hif = ActorWrapper(actor_no_hif, enable_hif=False)
    wrapper_with_hif = ActorWrapper(actor_with_hif, enable_hif=True)

    # 准备测试数据
    observation = torch.randn(2, 5, 96, 96)
    vector = torch.randn(2, 14)

    # 测试enable_hif=False：只返回action_params
    print("\n[2.1] enable_hif=False 输出测试")
    output_no_hif = wrapper_no_hif(observation, vector)
    assert "action_params" in output_no_hif, "❌ 缺少action_params键"
    assert "pred_ego_hif" not in output_no_hif, "❌ 不应包含pred_ego_hif键"
    print(f"✅ 输出键正确：{list(output_no_hif.keys())}")

    # 测试enable_hif=True：返回两个键
    print("\n[2.2] enable_hif=True 输出测试")
    output_with_hif = wrapper_with_hif(observation, vector)
    assert "action_params" in output_with_hif, "❌ 缺少action_params键"
    assert "pred_ego_hif" in output_with_hif, "❌ 缺少pred_ego_hif键"
    print(f"✅ 输出键正确：{list(output_with_hif.keys())}")
    print(f"✅ action_params形状：{output_with_hif['action_params'].shape}")
    print(f"✅ pred_ego_hif形状：{output_with_hif['pred_ego_hif'].shape}")


def test_fail_fast_mechanism():
    """测试fail-fast机制"""
    print("\n" + "=" * 60)
    print("测试3：Fail-Fast机制")
    print("=" * 60)

    # 创建enable_hif=False的actor
    actor_no_hif = ResNetFPNDualHeadActor(
        in_channels=5, vec_dim=14, action_dim=2, enable_hif=False
    )

    observation = torch.randn(2, 5, 96, 96)
    vector = torch.randn(2, 14)

    # 尝试在未初始化decoder时请求HIF输出
    print("\n[3.1] 测试decoder未初始化时的fail-fast")
    try:
        _, _ = actor_no_hif(observation, vector, return_hif=True)
        print("❌ 应该抛出RuntimeError")
        assert False, "应该抛出异常"
    except RuntimeError as e:
        assert "HIF decoder未初始化" in str(e), "❌ 错误消息不正确"
        print(f"✅ 正确抛出RuntimeError：{e}")

    # 验证不请求HIF时可以正常运行
    print("\n[3.2] 测试不请求HIF时正常运行")
    action_params, hif_pred = actor_no_hif(observation, vector, return_hif=False)
    assert hif_pred is None, "❌ hif_pred应该为None"
    print("✅ 正常运行，hif_pred=None")


def test_memory_savings():
    """测试显存节省效果"""
    print("\n" + "=" * 60)
    print("测试4：显存使用对比")
    print("=" * 60)

    import gc

    # 清理显存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # 测试enable_hif=False的参数量
    print("\n[4.1] enable_hif=False 参数统计")
    actor_no_hif = ResNetFPNDualHeadActor(
        in_channels=5, vec_dim=14, action_dim=2, enable_hif=False
    )
    params_no_hif = sum(p.numel() for p in actor_no_hif.parameters())
    print(f"✅ 总参数量：{params_no_hif:,}")

    # 测试enable_hif=True的参数量
    print("\n[4.2] enable_hif=True 参数统计")
    actor_with_hif = ResNetFPNDualHeadActor(
        in_channels=5, vec_dim=14, action_dim=2,
        enable_hif=True, hif_decoder_type='two_stage'
    )
    params_with_hif = sum(p.numel() for p in actor_with_hif.parameters())
    decoder_params = params_with_hif - params_no_hif
    print(f"✅ 总参数量：{params_with_hif:,}")
    print(f"✅ Decoder参数量：{decoder_params:,}")

    # 估算显存节省（float32 = 4 bytes）
    memory_saved_mb = (decoder_params * 4) / (1024 * 1024)
    print(f"\n💾 显存节省估算：{memory_saved_mb:.2f} MB")


def main():
    print("\n" + "🔬" * 30)
    print("HIF解耦功能验证测试")
    print("🔬" * 30 + "\n")

    try:
        test_decoder_conditional_init()
        test_actor_wrapper_output_keys()
        test_fail_fast_mechanism()
        test_memory_savings()

        print("\n" + "=" * 60)
        print("✅ 所有测试通过！HIF解耦功能正常工作")
        print("=" * 60)

        print("\n📋 验证总结：")
        print("  1. ✅ Decoder条件初始化正确")
        print("  2. ✅ ActorWrapper输出键正确")
        print("  3. ✅ Fail-fast机制正常工作")
        print("  4. ✅ 显存节省效果明显")

        return True

    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
