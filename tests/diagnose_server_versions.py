#!/usr/bin/env python3
"""
服务器版本诊断脚本

在服务器上运行这个脚本，检查：
1. 所有相关库的版本
2. torchvision.write_video() 是否正常
3. 具体在哪一步失败
"""

import sys
from pathlib import Path

def main():
    print("=" * 70)
    print("服务器环境诊断")
    print("=" * 70)

    # 1. Python 版本
    print(f"\n🐍 Python: {sys.version}")
    print(f"   可执行文件: {sys.executable}")

    # 2. 关键库版本
    print("\n📦 关键库版本:")

    libs = [
        ("torch", "PyTorch"),
        ("torchvision", "torchvision"),
        ("av", "PyAV"),
        ("torchrl", "TorchRL"),
        ("tensordict", "TensorDict"),
        ("numpy", "NumPy"),
    ]

    versions = {}
    for module_name, display_name in libs:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "未知版本")
            versions[module_name] = version
            print(f"  ✅ {display_name}: {version}")
        except ImportError as e:
            print(f"  ❌ {display_name}: 未安装 ({e})")
            versions[module_name] = None

    # 3. 检查 torchvision 是否支持视频
    print("\n🎬 torchvision 视频功能检查:")
    try:
        import torchvision.io
        if hasattr(torchvision.io, 'write_video'):
            print("  ✅ write_video 函数存在")
        else:
            print("  ❌ write_video 函数不存在")
            return

        # 尝试写入一个测试视频
        import torch
        import tempfile
        import shutil

        temp_dir = Path(tempfile.mkdtemp())
        try:
            test_video = temp_dir / "test.mp4"
            video_data = torch.randint(0, 256, (3, 16, 16, 3), dtype=torch.uint8)

            print(f"\n  测试写入: {test_video}")
            print(f"  视频数据: {video_data.shape}, dtype={video_data.dtype}")

            torchvision.io.write_video(str(test_video), video_data, fps=10)

            if test_video.exists():
                size = test_video.stat().st_size
                print(f"  ✅ 写入成功！文件大小: {size} bytes")
                print("\n💡 结论: torchvision + PyAV 基础功能正常")
            else:
                print("  ❌ 文件未生成")

        except TypeError as e:
            if "integer is required" in str(e):
                print(f"\n  ❌ 发现 'an integer is required' 错误！")
                print(f"     这确认了 torchvision 与 PyAV 不兼容")
                print(f"\n  📋 完整错误:")
                import traceback
                traceback.print_exc()
            else:
                print(f"\n  ❌ 其他 TypeError: {e}")
                import traceback
                traceback.print_exc()

        except Exception as e:
            print(f"\n  ❌ 写入失败: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        print(f"  ❌ 视频功能检查失败: {e}")

    # 4. 检查 TorchRL VideoRecorder
    print("\n📹 TorchRL VideoRecorder 检查:")
    try:
        from torchrl.record import VideoRecorder
        print("  ✅ VideoRecorder 可以导入")

        # 检查 dump 方法签名
        import inspect
        sig = inspect.signature(VideoRecorder.dump)
        print(f"  dump() 签名: {sig}")

    except Exception as e:
        print(f"  ❌ VideoRecorder 检查失败: {e}")

    # 5. 检查 PyAV 的 pict_type 要求
    print("\n🔍 PyAV VideoFrame.pict_type 类型检查:")
    try:
        import av

        # 尝试创建一个简单的视频帧
        print("  尝试设置 pict_type...")

        # 这需要一个真实的视频流，比较复杂
        # 我们只检查 PyAV 版本和可用性
        print(f"  PyAV 版本: {av.__version__}")

        # 检查是否有新的 API
        if hasattr(av.video.frame, 'VideoFrame'):
            print("  ✅ VideoFrame 类存在")
        else:
            print("  ❌ VideoFrame 类不存在")

    except Exception as e:
        print(f"  ⚠️  PyAV 检查失败: {e}")

    # 6. 诊断建议
    print("\n" + "=" * 70)
    print("诊断建议")
    print("=" * 70)

    if versions.get('torchvision'):
        tv_ver = versions['torchvision']
        av_ver = versions.get('av', 'unknown')

        print(f"\n当前版本组合:")
        print(f"  torchvision: {tv_ver}")
        print(f"  PyAV: {av_ver}")

        # 版本建议
        if av_ver and av_ver.startswith('15.'):
            print("\n⚠️  PyAV 15.x 版本可能与某些 torchvision 版本不兼容")
            print("建议尝试:")
            print("  1. 降级 PyAV: pip install 'av<12'")
            print("  2. 或升级到最新 torchvision: pip install --upgrade torchvision")

        if tv_ver and tv_ver.startswith('0.23'):
            print("\n✅ torchvision 0.23 是最新稳定版")
            print("如果还有问题，可能是 TorchRL 的使用方式问题")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
