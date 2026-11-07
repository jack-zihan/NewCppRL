#!/usr/bin/env python3
"""
最简单的视频兼容性测试
"""

import torch
import sys
from pathlib import Path
import tempfile
import shutil

def main():
    print("=" * 60)
    print("最简单的视频兼容性测试")
    print("=" * 60)

    # 1. 打印版本
    print("\n📦 版本信息:")
    try:
        import torchvision
        print(f"torchvision: {torchvision.__version__}")
    except Exception as e:
        print(f"torchvision: 导入失败 - {e}")
        return False

    try:
        import av
        print(f"PyAV: {av.__version__}")
    except Exception as e:
        print(f"PyAV: 导入失败 - {e}")
        return False

    print(f"PyTorch: {torch.__version__}")

    # 2. 创建临时文件
    temp_dir = Path(tempfile.mkdtemp())
    video_path = temp_dir / "test.mp4"

    try:
        print(f"\n🎬 测试写入视频: {video_path}")

        # 创建最简单的视频：5帧，32x32，RGB
        video_data = torch.randint(0, 256, (5, 32, 32, 3), dtype=torch.uint8)
        print(f"视频数据: {video_data.shape}, dtype={video_data.dtype}")

        # 尝试写入
        print("\n🔥 调用 torchvision.io.write_video()...")
        torchvision.io.write_video(
            str(video_path),
            video_data,
            fps=10
        )

        print("✅ 成功！")

        if video_path.exists():
            size = video_path.stat().st_size
            print(f"✅ 视频文件: {size} bytes")
            return True
        else:
            print("❌ 文件未生成")
            return False

    except Exception as e:
        print(f"\n❌ 失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
