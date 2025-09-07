#!/usr/bin/env python3
"""
测试evaluate_policy的视频录制修复
验证agents在视频中能正常移动
"""
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from rl_new.sac_cont_sy.sac_utils import evaluate_policy
from torchrl_utils.utils_env import make_single_environment
from omegaconf import OmegaConf
import tempfile

def test_video_recording():
    """测试视频录制是否正常工作"""
    
    # 加载配置
    cfg_path = Path(__file__).parent.parent / "rl_new/sac_cont_sy/config.yaml"
    cfg = OmegaConf.load(cfg_path)
    
    # 简化配置用于测试
    cfg.logger.eval_episodes = 4  # 使用4个环境测试
    cfg.logger.eval_max_steps = 20  # 减少步数加快测试
    cfg.logger.eval_video = True
    cfg.logger.eval_video_skip = 5  # 每5步录制一帧
    cfg.logger.show_progress = False
    
    # 创建假的actor模型（随机动作）
    class FakeActor:
        def __call__(self, td):
            # 生成随机动作
            batch_size = td.shape[0] if td.batch_size else 1
            action = torch.randn(batch_size, 2)  # 假设2维动作空间
            td["action"] = action
            return td
    
    actor_critic = [FakeActor(), None]
    
    # 创建简单的logger
    class SimpleLogger:
        def __init__(self):
            self.videos = []
        
        def log_video(self, key, video_tensor, step):
            print(f"视频录制成功: key={key}, shape={video_tensor.shape}, step={step}")
            self.videos.append(video_tensor)
            
            # 检查视频帧是否变化
            if video_tensor.dim() == 5:  # [T, B, C, H, W]
                # 计算相邻帧之间的差异
                frame_diffs = []
                for t in range(1, video_tensor.shape[0]):
                    diff = torch.abs(video_tensor[t] - video_tensor[t-1]).mean().item()
                    frame_diffs.append(diff)
                
                avg_diff = np.mean(frame_diffs) if frame_diffs else 0
                print(f"  平均帧间差异: {avg_diff:.4f}")
                
                if avg_diff < 0.001:
                    print("  ⚠️ 警告：视频帧几乎没有变化，可能agents还是静止的！")
                else:
                    print("  ✅ 视频帧有明显变化，agents在移动！")
                
                return avg_diff > 0.001
            return False
    
    logger = SimpleLogger()
    
    print("开始测试视频录制...")
    print("-" * 60)
    
    try:
        # 运行评估
        metrics = evaluate_policy(
            actor_critic=actor_critic,
            cfg=cfg,
            train_device="cpu",
            logger=logger,
            step=1000
        )
        
        print("-" * 60)
        print("评估指标:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.2f}")
        
        # 检查是否录制了视频
        if logger.videos:
            print(f"\n成功录制 {len(logger.videos)} 个视频")
            video = logger.videos[0]
            print(f"视频信息: shape={video.shape}, dtype={video.dtype}")
            
            # 分析第一个视频
            if video.dim() == 5:
                num_frames = video.shape[0]
                print(f"视频帧数: {num_frames}")
                
                # 检查帧变化
                frame_changes = []
                for t in range(1, min(5, num_frames)):
                    diff = torch.abs(video[t] - video[t-1]).mean().item()
                    frame_changes.append(diff)
                    print(f"  帧{t-1}→帧{t} 差异: {diff:.4f}")
                
                avg_change = np.mean(frame_changes)
                if avg_change > 0.001:
                    print(f"\n✅ 测试通过！平均帧变化: {avg_change:.4f}")
                    return True
                else:
                    print(f"\n❌ 测试失败！帧几乎没有变化: {avg_change:.4f}")
                    return False
        else:
            print("\n❌ 未录制任何视频")
            return False
            
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_video_recording()
    sys.exit(0 if success else 1)