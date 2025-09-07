#!/usr/bin/env python3
"""
测试将视频保存到本地文件，验证视频内容
"""
import sys
sys.path.append('/home/lzh/NewCppRL')

import torch
import numpy as np
from omegaconf import OmegaConf
from rl_new.sac_cont_sy.env_utils import make_single_environment
from torchrl_utils.local_video_recorder import LocalVideoRecorder


def test_save_video_locally():
    """测试保存视频到本地文件"""
    print("\n" + "="*80)
    print("测试保存视频到本地文件")
    print("="*80)
    
    cfg = OmegaConf.create({
        'seed': 42,
        'env': {
            'env_id': 'NewPasture-v5',
            'env_kwargs': {}
        },
        'collector': {
            'env_per_collector': 1
        }
    })
    
    # 创建4个环境
    print("\n1. 创建4个环境:")
    envs = []
    for i in range(4):
        env = make_single_environment(cfg, device="cpu", seed=42+i, from_pixels=True)
        envs.append(env)
        print(f"   环境{i}创建成功")
    
    # 创建recorder
    print("\n2. 创建recorder (make_grid=True):")
    recorder = LocalVideoRecorder(
        device="cpu",
        max_len=50,
        use_memmap=False,
        make_grid=True,
        nrow=2,
        skip=1,
        fps=6
    )
    print(f"   Recorder创建成功")
    
    # Reset环境并录制
    print("\n3. Reset环境并录制10帧:")
    tds = [env.reset() for env in envs]
    
    for frame_idx in range(10):
        # 收集pixels
        pixels = []
        for i in range(4):
            if "pixels" in tds[i]:
                pixels.append(tds[i]["pixels"])
                print(f"   帧{frame_idx}: 环境{i} pixels shape={tds[i]['pixels'].shape}")
            else:
                print(f"   帧{frame_idx}: 环境{i} 缺少pixels!")
                # 创建虚拟数据
                pixels.append(torch.zeros(800, 800, 3, dtype=torch.uint8))
        
        if len(pixels) == 4:
            stacked = torch.stack(pixels, 0)
            print(f"   帧{frame_idx}: stacked shape={stacked.shape}")
            recorder.apply(stacked)
            print(f"   帧{frame_idx}: recorder.idx={recorder.idx}")
        
        # 简单step
        for i in range(4):
            action = torch.zeros(2)  # 假设是2维连续动作
            tds[i]["action"] = action
            tds[i] = envs[i].step(tds[i])
    
    # 关闭环境
    for env in envs:
        env.close()
    
    # Dump视频
    print(f"\n4. Dump视频:")
    print(f"   recorder.idx={recorder.idx}")
    print(f"   recorder.obs shape={recorder.obs.shape if recorder.obs is not None else 'None'}")
    
    vid_tensor = recorder.dump()
    
    if vid_tensor is not None:
        print(f"\n✅ 成功生成视频tensor:")
        print(f"   shape: {vid_tensor.shape}")
        print(f"   dtype: {vid_tensor.dtype}")
        print(f"   min: {vid_tensor.min()}, max: {vid_tensor.max()}")
        
        # 尝试保存为numpy文件
        print("\n5. 保存视频数据:")
        np_video = vid_tensor.cpu().numpy()
        np.save("/tmp/test_video_data.npy", np_video)
        print(f"   保存到: /tmp/test_video_data.npy")
        print(f"   文件大小: {np_video.nbytes / 1024 / 1024:.2f} MB")
        
        # 尝试使用opencv保存
        try:
            import cv2
            print("\n6. 尝试使用OpenCV保存视频:")
            # vid_tensor shape: [batch, time, C, H, W]
            # 需要转换为 [time, H, W, C]
            if vid_tensor.ndim == 5:
                video = vid_tensor[0]  # 取第一个batch
                video = video.permute(0, 2, 3, 1)  # [T, H, W, C]
                video = video.cpu().numpy()
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                height, width = video.shape[1:3]
                out = cv2.VideoWriter('/tmp/test_video_opencv.mp4', fourcc, 6.0, (width, height))
                
                for frame in video:
                    # OpenCV使用BGR格式
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                
                out.release()
                print(f"   ✅ 保存到: /tmp/test_video_opencv.mp4")
        except ImportError:
            print("   ⚠️ OpenCV未安装，跳过")
        except Exception as e:
            print(f"   ❌ OpenCV保存失败: {e}")
        
        # 创建简单的测试图像验证内容
        print("\n7. 验证视频内容:")
        if vid_tensor.ndim == 5:
            first_frame = vid_tensor[0, 0]  # 第一帧
            print(f"   第一帧 shape: {first_frame.shape}")
            print(f"   第一帧 非零像素: {(first_frame > 0).sum().item()}/{first_frame.numel()}")
            
            last_frame = vid_tensor[0, -1]  # 最后一帧
            print(f"   最后一帧 shape: {last_frame.shape}")
            print(f"   最后一帧 非零像素: {(last_frame > 0).sum().item()}/{last_frame.numel()}")
            
            # 检查是否全黑
            if vid_tensor.max() == 0:
                print("   ⚠️ 警告：视频全黑！")
            else:
                print(f"   ✅ 视频有内容，像素值范围: {vid_tensor.min()}-{vid_tensor.max()}")
        
        return True
    else:
        print(f"\n❌ vid_tensor为None!")
        print(f"   recorder.idx={recorder.idx}")
        print(f"   这说明没有录制到任何帧")
        return False


if __name__ == "__main__":
    success = test_save_video_locally()
    
    print("\n" + "="*80)
    print("总结:")
    if success:
        print("✅ 视频生成成功")
        print("   检查文件: /tmp/test_video_data.npy")
        print("   可选文件: /tmp/test_video_opencv.mp4")
        print("\n下一步:")
        print("1. 检查生成的文件是否存在")
        print("2. 使用numpy加载.npy文件验证内容")
        print("3. 如果有.mp4文件，用播放器打开查看")
    else:
        print("❌ 视频生成失败")
        print("   需要检查recorder的实现")
    print("="*80)