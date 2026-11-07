#!/usr/bin/env python3
"""
测试 torchvision 与 PyAV 的兼容性问题

测试调用链：
VideoRecorder.dump() → logger.log_video() → experiment.add_video()
→ torchvision.write_video() → frame.pict_type = "NONE"
→ PyAV.VideoFrame.pict_type.__set__() → TypeError
"""

import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

def test_video_recording_compatibility():
    """测试完整的视频记录和导出流程"""

    print("=" * 60)
    print("TorchRL + torchvision + PyAV 兼容性测试")
    print("=" * 60)

    # 1. 打印版本信息
    print("\n📦 依赖版本:")
    try:
        import torchrl
        print(f"  TorchRL: {torchrl.__version__}")
    except:
        print("  TorchRL: 未安装")

    try:
        import torchvision
        print(f"  torchvision: {torchvision.__version__}")
    except:
        print("  torchvision: 未安装")

    try:
        import av
        print(f"  PyAV: {av.__version__}")
    except:
        print("  PyAV: 未安装")

    print(f"  PyTorch: {torch.__version__}")

    # 2. 创建临时目录
    temp_dir = Path(tempfile.mkdtemp())
    print(f"\n📁 临时目录: {temp_dir}")

    try:
        # 3. 创建 CSVLogger（最简单的 logger）
        from torchrl.record.loggers import CSVLogger
        logger = CSVLogger(exp_name="video_test", log_dir=str(temp_dir))
        print("\n✅ CSVLogger 创建成功")

        # 4. 创建 VideoRecorder
        from torchrl.record import VideoRecorder

        recorder = VideoRecorder(
            logger=logger,
            tag="test_video",
            in_keys=["pixels"],
            skip=1  # 每帧都记录
        )
        print("✅ VideoRecorder 创建成功")

        # 5. 模拟记录几帧视频
        print("\n🎬 开始记录视频帧...")
        num_frames = 10
        height, width = 64, 64  # 小尺寸测试

        for i in range(num_frames):
            # 创建简单的彩色渐变帧（RGB格式）
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = (i * 25) % 256  # R 通道渐变
            frame[:, :, 1] = 128  # G 通道固定
            frame[:, :, 2] = 255 - (i * 25) % 256  # B 通道反向渐变

            # 转换为 torch tensor
            # VideoRecorder 期望的格式：(C, H, W) 或 (H, W, C)
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)  # (C, H, W)

            # 创建 TensorDict（模拟环境输出）
            from tensordict import TensorDict
            td = TensorDict({
                "pixels": frame_tensor.unsqueeze(0),  # 添加 batch 维度
            }, batch_size=[1])

            # 调用 VideoRecorder（会内部累积帧）
            recorder._call(td)

            if (i + 1) % 5 == 0:
                print(f"  已记录 {i + 1}/{num_frames} 帧")

        print(f"✅ 成功记录 {num_frames} 帧")

        # 6. 🔥 关键测试：调用 dump() 触发视频写入
        print("\n🔥 测试关键步骤：调用 dump() 写入视频...")
        print("   这会触发调用链:")
        print("   recorder.dump() → logger.log_video()")
        print("   → experiment.add_video() → torchvision.write_video()")
        print("   → frame.pict_type = 'NONE' → PyAV 报错")
        print()

        try:
            recorder.dump()  # dump() 不需要参数
            print("✅ dump() 成功！视频写入正常")
            print(f"   视频应该保存在: {temp_dir}")

            # 检查是否真的生成了视频文件
            video_files = list(temp_dir.rglob("*.mp4"))
            if video_files:
                print(f"✅ 找到视频文件: {video_files[0].name}")
                print(f"   文件大小: {video_files[0].stat().st_size} bytes")
            else:
                print("⚠️  没有找到 .mp4 文件，但 dump() 没报错")

            return True

        except TypeError as e:
            if "integer is required" in str(e):
                print("❌ 发现兼容性问题！")
                print(f"   错误信息: {e}")
                print("\n📋 完整异常堆栈:")
                import traceback
                traceback.print_exc()
                print("\n💡 这证实了 torchvision 与 PyAV 的不兼容问题")
                return False
            else:
                print(f"❌ 其他 TypeError: {e}")
                raise

        except Exception as e:
            print(f"❌ 未预期的错误: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False

    finally:
        # 7. 清理临时目录
        print(f"\n🧹 清理临时目录: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        print()


def test_direct_torchvision_write():
    """直接测试 torchvision.write_video 是否有问题"""

    print("=" * 60)
    print("直接测试 torchvision.write_video()")
    print("=" * 60)

    temp_dir = Path(tempfile.mkdtemp())
    video_path = temp_dir / "direct_test.mp4"

    try:
        import torchvision

        print("\n🎬 创建测试视频数据...")
        # 创建简单的视频：10帧，64x64，RGB
        num_frames = 10
        height, width = 64, 64

        # torchvision.write_video 期望的格式：(T, H, W, C)，值范围 0-255
        video_data = torch.randint(0, 256, (num_frames, height, width, 3), dtype=torch.uint8)
        print(f"  视频数据形状: {video_data.shape}")

        print(f"\n🔥 调用 torchvision.io.write_video()...")
        print(f"   输出路径: {video_path}")

        try:
            torchvision.io.write_video(
                str(video_path),
                video_data,
                fps=30
            )
            print("✅ write_video() 成功！")

            if video_path.exists():
                print(f"✅ 视频文件已生成: {video_path.stat().st_size} bytes")

            return True

        except TypeError as e:
            if "integer is required" in str(e):
                print("❌ 发现兼容性问题！")
                print(f"   错误: {e}")
                print("\n💡 这证实了 torchvision.write_video() 与 PyAV 不兼容")
                import traceback
                traceback.print_exc()
                return False
            else:
                raise

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("开始兼容性测试")
    print("=" * 60 + "\n")

    # 测试1：直接测试 torchvision
    print("\n【测试 1/2】直接测试 torchvision.write_video()\n")
    result1 = test_direct_torchvision_write()

    # 测试2：完整的 TorchRL VideoRecorder 流程
    print("\n【测试 2/2】完整 TorchRL VideoRecorder 流程\n")
    result2 = test_video_recording_compatibility()

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"\n直接 torchvision 测试: {'✅ 通过' if result1 else '❌ 失败'}")
    print(f"TorchRL 完整流程测试: {'✅ 通过' if result2 else '❌ 失败'}")

    if not result1 or not result2:
        print("\n🔧 修复建议:")
        print("  1. 尝试降级 PyAV:")
        print("     pip install av==10.0.0")
        print()
        print("  2. 尝试升级 torchvision:")
        print("     pip install --upgrade torchvision")
        print()
        print("  3. 临时禁用视频记录:")
        print("     logger.eval_video=false")
    else:
        print("\n✅ 所有测试通过！当前环境兼容性良好。")

    print("\n" + "=" * 60)
