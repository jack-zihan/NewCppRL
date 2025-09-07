"""
真正测试wandb视频上传 - 确保视频能在网站上看到
"""
import sys
sys.path.append('/home/lzh/NewCppRL')

import envs_new
import torch
import numpy as np
import time
from torchrl_utils.local_video_recorder import LocalVideoRecorder
from torchrl.record.loggers import get_logger
from rl_new.sac_cont_sy.env_utils import make_single_environment
from omegaconf import OmegaConf


def test_minimal_upload():
    """最小化测试 - 确保视频真的上传"""
    print("\n" + "="*80)
    print("最小化wandb视频上传测试")
    print("="*80)
    
    # 创建logger
    exp_name = f"minimal_test_{int(time.time())}"
    print(f"\n实验名称: {exp_name}")
    
    logger = get_logger(
        logger_type="wandb",
        logger_name="test_upload",
        experiment_name=exp_name,
        wandb_kwargs={
            "project": "sac-test",  # 使用您的实际项目
            "config": {"test": "video_upload"},
            "mode": "online",
            "name": exp_name
        }
    )
    print("Logger创建成功")
    
    # 创建recorder
    recorder = LocalVideoRecorder(
        device="cpu",
        max_len=10,
        use_memmap=False,
        make_grid=False,  # 不用grid，单个视频
        skip=1,
        fps=2  # 低帧率
    )
    
    # 创建一些明显的测试帧
    print("\n创建测试视频:")
    for i in range(5):
        # 创建一个有明显特征的帧 (3, 100, 100)
        frame = torch.zeros(3, 100, 100, dtype=torch.uint8)
        # 在不同位置画白色方块
        frame[:, i*20:(i+1)*20, i*20:(i+1)*20] = 255
        recorder.apply(frame)
        print(f"  帧{i}: 白色方块在位置({i*20}, {i*20})")
    
    print(f"\nrecorder.idx = {recorder.idx}")
    
    # Dump
    vid_tensor = recorder.dump()
    print(f"vid_tensor shape: {vid_tensor.shape if vid_tensor is not None else None}")
    
    if vid_tensor is not None:
        # 确保格式正确
        print(f"vid_tensor dtype: {vid_tensor.dtype}")
        print(f"vid_tensor device: {vid_tensor.device}")
        print(f"vid_tensor范围: [{vid_tensor.min()}, {vid_tensor.max()}]")
        
        # 上传
        print("\n上传视频:")
        logger.log_video('test/minimal_video', vid_tensor, step=0)
        print("log_video调用完成")
        
        # 记录一些标量以确保logger工作
        logger.log_scalar('test/value', 42.0, step=0)
        print("log_scalar调用完成")
        
        # 等待上传
        print("\n等待上传完成...")
        time.sleep(5)
        
        # 关闭logger
        print("关闭logger...")
        del logger
        time.sleep(2)
        
        print(f"\n✅ 请检查wandb项目: sac-test")
        print(f"   实验名称: {exp_name}")
        print("   应该看到test/minimal_video")
        
        return True
    else:
        print("❌ vid_tensor为None")
        return False


def test_with_actual_environment():
    """使用实际环境测试"""
    print("\n" + "="*80)
    print("实际环境视频上传测试")
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
    
    # 创建logger
    exp_name = f"env_test_{int(time.time())}"
    print(f"\n实验名称: {exp_name}")
    
    logger = get_logger(
        logger_type="wandb",
        logger_name="test_env_upload",
        experiment_name=exp_name,
        wandb_kwargs={
            "project": "sac-test",
            "config": {"test": "environment_video"},
            "mode": "online",
            "name": exp_name
        }
    )
    
    # 创建环境
    env = make_single_environment(cfg, device="cpu", seed=42, from_pixels=True)
    
    # 创建recorder（单个视频）
    recorder = LocalVideoRecorder(
        device="cpu",
        max_len=20,
        use_memmap=False,
        make_grid=False,
        skip=1,
        fps=4
    )
    
    # Reset
    td = env.reset()
    print(f"\nReset: 'pixels' in td = {'pixels' in td}")
    
    if 'pixels' in td:
        # 记录初始帧
        pixels = td["pixels"]  # (H, W, 3)
        print(f"pixels shape: {pixels.shape}")
        recorder.apply(pixels)
        
        # 运行几步
        from rl_new.sac_cont_sy.model_utils import make_sac_models
        dummy_env = make_single_environment(cfg, device="cpu", from_pixels=False)
        actor_critic = make_sac_models(dummy_env, device="cpu")
        dummy_env.close()
        
        print("\n运行环境:")
        for step in range(10):
            with torch.no_grad():
                td = actor_critic[0](td)
            
            next_td = env.step(td)
            
            if 'pixels' in next_td:
                recorder.apply(next_td["pixels"])
                print(f"  步{step}: 录制帧, recorder.idx={recorder.idx}")
            
            td = next_td
    
    env.close()
    
    # Dump并上传
    print(f"\nDump前: recorder.idx = {recorder.idx}")
    vid_tensor = recorder.dump()
    
    if vid_tensor is not None:
        print(f"vid_tensor shape: {vid_tensor.shape}")
        
        # 上传
        logger.log_video('test/env_video', vid_tensor, step=0)
        logger.log_scalar('test/env_steps', 10, step=0)
        
        print("\n等待上传...")
        time.sleep(5)
        
        # 正确关闭
        del logger
        time.sleep(2)
        
        print(f"\n✅ 请检查wandb项目: sac-test")
        print(f"   实验名称: {exp_name}")
        return True
    else:
        print("❌ vid_tensor为None")
        return False


def test_recorder_dump_details():
    """详细测试recorder.dump()的输出"""
    print("\n" + "="*80)
    print("详细测试recorder.dump()输出")
    print("="*80)
    
    recorder = LocalVideoRecorder(
        device="cpu",
        max_len=10,
        use_memmap=False,
        make_grid=False,
        skip=1,
        fps=2
    )
    
    # 添加一帧
    frame = torch.ones(3, 64, 64, dtype=torch.uint8) * 128
    recorder.apply(frame)
    
    print(f"\nrecorder内部状态:")
    print(f"  idx: {recorder.idx}")
    print(f"  obs shape: {recorder.obs.shape if recorder.obs is not None else None}")
    print(f"  obs dtype: {recorder.obs.dtype if recorder.obs is not None else None}")
    
    # Dump
    vid_tensor = recorder.dump()
    
    print(f"\ndump()返回:")
    print(f"  类型: {type(vid_tensor)}")
    print(f"  shape: {vid_tensor.shape if vid_tensor is not None else None}")
    print(f"  dtype: {vid_tensor.dtype if vid_tensor is not None else None}")
    
    # 检查dump()的代码逻辑
    print("\n根据LocalVideoRecorder.dump()第144-162行:")
    print("  如果self.idx > 0:")
    print("    返回self.obs[:, :self.idx]")
    print("    如果ndim > 4，会flatten和permute")
    print("    最后返回格式: (batch, time, height, width, channels)")


if __name__ == "__main__":
    print("真正的wandb视频上传测试")
    print("="*80)
    
    # 测试1: 最小化测试
    print("\n[测试1] 最小化上传测试...")
    result1 = test_minimal_upload()
    
    # 测试2: 实际环境
    print("\n[测试2] 实际环境测试...")
    result2 = test_with_actual_environment()
    
    # 测试3: dump细节
    print("\n[测试3] Dump细节...")
    test_recorder_dump_details()
    
    print("\n" + "="*80)
    print("测试完成")
    print("="*80)
    print(f"最小化测试: {'✅' if result1 else '❌'}")
    print(f"环境测试: {'✅' if result2 else '❌'}")
    print("\n⚠️ 重要：请登录wandb.ai查看sac-test项目")
    print("   如果没有视频，说明上传确实有问题")
    print("   如果有视频，说明机制正常，问题在evaluate_policy的具体执行")