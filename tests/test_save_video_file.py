"""
测试将recorder的输出保存为视频文件
"""
import sys
sys.path.append('/home/lzh/NewCppRL')

import envs_new
import torch
import torchvision
from torchrl_utils.local_video_recorder import LocalVideoRecorder
from rl_new.sac_cont_sy.env_utils import make_single_environment
from rl_new.sac_cont_sy.model_utils import make_sac_models
from omegaconf import OmegaConf


def test_save_video_to_file():
    """测试保存视频到文件"""
    print("\n" + "="*80)
    print("测试保存视频到文件")
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
    
    # 创建环境
    print("\n1. 创建4个环境:")
    envs = []
    for i in range(4):
        env = make_single_environment(cfg, device="cpu", seed=42+i, from_pixels=True)
        envs.append(env)
    
    # 创建recorder（使用make_grid）
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
    
    # Reset环境
    print("\n3. Reset所有环境:")
    tds = []
    for i, env in enumerate(envs):
        td = env.reset()
        tds.append(td)
        print(f"   环境{i}: 'pixels' in td = {'pixels' in td}")
    
    # 创建actor
    dummy_env = make_single_environment(cfg, device="cpu", from_pixels=False)
    actor_critic = make_sac_models(dummy_env, device="cpu")
    dummy_env.close()
    
    # 录制10帧
    print("\n4. 录制10帧:")
    for frame_idx in range(10):
        # 使用用户的简化代码（无防御）
        pixels = [tds[i]["pixels"] for i in range(4)]
        stacked = torch.stack(pixels, 0)
        print(f"   帧{frame_idx}: stacked shape={stacked.shape}")
        
        recorder.apply(stacked)
        print(f"   recorder.idx = {recorder.idx}")
        
        # Step环境
        for i in range(4):
            with torch.no_grad():
                tds[i] = actor_critic[0](tds[i])
            tds[i] = envs[i].step(tds[i])
    
    # 关闭环境
    for env in envs:
        env.close()
    
    # Dump视频
    print(f"\n5. Dump视频:")
    print(f"   dump前: recorder.idx = {recorder.idx}")
    vid_tensor = recorder.dump()
    print(f"   dump后: recorder.idx = {recorder.idx}")
    
    if vid_tensor is not None:
        print(f"   ✅ vid_tensor shape: {vid_tensor.shape}")
        print(f"   vid_tensor dtype: {vid_tensor.dtype}")
        
        # 保存到文件
        print("\n6. 保存到文件:")
        
        # 方法1: 使用dump的内置保存功能
        recorder2 = LocalVideoRecorder(
            device="cpu",
            max_len=50,
            use_memmap=False,
            make_grid=True,
            nrow=2,
            skip=1,
            fps=6
        )
        
        # 重新录制用于保存
        for i in range(10):
            pixels = torch.randn(4, 3, 100, 100).clamp(0, 255).to(torch.uint8)
            recorder2.apply(pixels)
        
        # 使用dump的filepath参数
        print("   尝试使用dump(filepath)保存...")
        vid_tensor2 = recorder2.dump(filepath="/tmp/test_video.mp4")
        print(f"   ✅ 视频保存到: /tmp/test_video.mp4")
        
        # 方法2: 手动保存
        print("\n   手动保存vid_tensor...")
        if vid_tensor.shape[-3] not in (3, 1):
            print(f"   ⚠️ 通道维度不对: {vid_tensor.shape[-3]}")
        
        # 调整格式 (batch, time, C, H, W) -> (time, H, W, C)
        if vid_tensor.ndim == 5:
            # 去掉batch维度
            vid_for_save = vid_tensor[0]  # (time, C, H, W)
            # 转换为 (time, H, W, C)
            vid_for_save = vid_for_save.permute(0, 2, 3, 1)
            # 确保是3通道
            if vid_for_save.shape[-1] == 1:
                vid_for_save = vid_for_save.expand(-1, -1, -1, 3)
            
            print(f"   调整后shape: {vid_for_save.shape}")
            
            # 保存
            torchvision.io.write_video("/tmp/test_video_manual.mp4", vid_for_save, fps=6)
            print(f"   ✅ 手动保存到: /tmp/test_video_manual.mp4")
        
        return True
    else:
        print("   ❌ vid_tensor为None")
        print(f"   recorder.idx = {recorder.idx}")
        return False


def verify_saved_videos():
    """验证保存的视频文件"""
    print("\n" + "="*80)
    print("验证保存的视频文件")
    print("="*80)
    
    import os
    
    files = ["/tmp/test_video.mp4", "/tmp/test_video_manual.mp4"]
    
    for filepath in files:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"✅ {filepath}: {size} bytes")
            
            # 尝试读取验证
            try:
                video, audio, info = torchvision.io.read_video(filepath)
                print(f"   视频shape: {video.shape}")
                print(f"   FPS: {info['video_fps']}")
            except Exception as e:
                print(f"   ⚠️ 读取失败: {e}")
        else:
            print(f"❌ {filepath}: 不存在")


if __name__ == "__main__":
    print("测试recorder输出是否能保存为视频文件...")
    print("="*80)
    
    # 测试保存
    success = test_save_video_to_file()
    
    # 验证文件
    verify_saved_videos()
    
    print("\n" + "="*80)
    print("结论")
    print("="*80)
    
    if success:
        print("✅ Recorder能正常工作并生成视频")
        print("   问题可能在wandb上传环节")
    else:
        print("❌ Recorder本身有问题")
        print("   需要检查recorder.apply()是否执行")
    
    print("\n下一步：")
    print("1. 检查/tmp/test_video*.mp4文件是否存在")
    print("2. 如果存在，用视频播放器打开看是否正常")
    print("3. 在sac_utils.py中添加调试代码确认recorder.idx的值")