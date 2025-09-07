#!/usr/bin/env python3
"""
完整测试评估流程，找出视频上传失败的原因
"""
import sys
sys.path.append('/home/lzh/NewCppRL')

from omegaconf import OmegaConf
import torch
from torchrl.record.loggers import get_logger
from rl_new.sac_cont_sy.env_utils import make_single_environment
from rl_new.sac_cont_sy.model_utils import make_sac_models


def test_eval_flow():
    """测试完整的评估流程"""
    print("\n" + "="*80)
    print("测试完整评估流程")
    print("="*80)
    
    # 1. 加载配置
    print("\n1. 加载配置:")
    config = OmegaConf.load('/home/lzh/NewCppRL/rl_new/sac_cont_sy/config-async-server.yaml')
    print(f"   配置加载成功")
    print(f"   logger.backend = {config.logger.backend}")
    print(f"   logger.eval_video = {config.logger.eval_video}")
    
    # 2. 创建logger（模拟sac-async.py的逻辑）
    print("\n2. 创建logger (模拟sac-async.py):")
    logger = None
    if config.logger.backend:
        print(f"   backend不为null，创建logger...")
        exp_name = "test_eval_flow"
        logger = get_logger(
            logger_type=config.logger.backend,
            experiment_name=exp_name,
            logger_name=exp_name,
            wandb_kwargs={
                "mode": config.logger.mode,
                "config": dict(config),
                "project": config.logger.project_name,
                "group": config.logger.group_name,
                "name": exp_name
            }
        )
        print(f"   ✅ logger创建成功: {logger}")
    else:
        print(f"   ❌ backend为null，logger未创建")
    
    # 3. 模拟evaluate_policy的逻辑
    print("\n3. 模拟evaluate_policy:")
    eval_cfg = config.logger
    print(f"   eval_cfg类型: {type(eval_cfg)}")
    
    # 3.1 测试recorder创建条件
    print("\n   3.1 测试recorder创建条件:")
    print(f"       eval_cfg.eval_video = {eval_cfg.eval_video}")
    print(f"       logger is not None = {logger is not None}")
    
    if eval_cfg.eval_video and logger is not None:
        print(f"       ✅ 条件满足，会创建recorder")
        
        # 3.2 创建recorder
        print("\n   3.2 创建recorder:")
        from torchrl_utils.local_video_recorder import LocalVideoRecorder
        
        recorder = LocalVideoRecorder(
            device="cpu",
            max_len=100,
            use_memmap=True,
            make_grid=True,
            nrow=2,
            skip=1,
            fps=6
        )
        print(f"       ✅ recorder创建成功")
        
        # 3.3 模拟录制
        print("\n   3.3 模拟录制视频:")
        # 创建假数据
        fake_pixels = torch.randn(4, 800, 800, 3).clamp(0, 255).to(torch.uint8)
        
        # 录制10帧
        for i in range(10):
            recorder.apply(fake_pixels)
            print(f"       帧{i}: recorder.idx = {recorder.idx}")
        
        # 3.4 dump视频
        print("\n   3.4 Dump视频:")
        vid_tensor = recorder.dump()
        
        if vid_tensor is not None:
            print(f"       ✅ dump成功, shape: {vid_tensor.shape}")
            
            # 3.5 上传到wandb
            print("\n   3.5 上传到wandb:")
            if logger is not None:
                try:
                    logger.log_video('eval/video', vid_tensor, step=1000)
                    print(f"       ✅ 上传成功")
                except Exception as e:
                    print(f"       ❌ 上传失败: {e}")
            else:
                print(f"       ❌ logger为None，无法上传")
        else:
            print(f"       ❌ dump返回None")
            print(f"       recorder.idx = {recorder.idx}")
    else:
        print(f"       ❌ 条件不满足，不会创建recorder")
        print(f"       原因: eval_video={eval_cfg.eval_video}, logger={logger}")
    
    # 清理
    if logger:
        try:
            logger.close()
        except:
            pass


def test_actual_eval_import():
    """测试实际的evaluate_policy函数"""
    print("\n" + "="*80)
    print("测试导入实际的evaluate_policy")
    print("="*80)
    
    try:
        from rl_new.sac_cont_sy.sac_utils import evaluate_policy
        print("   ✅ evaluate_policy导入成功")
        
        # 检查函数签名
        import inspect
        sig = inspect.signature(evaluate_policy)
        print(f"   函数签名: {sig}")
        
    except Exception as e:
        print(f"   ❌ 导入失败: {e}")


if __name__ == "__main__":
    test_eval_flow()
    test_actual_eval_import()
    
    print("\n" + "="*80)
    print("关键发现:")
    print("1. OmegaConf的DictConfig支持点号和方括号访问")
    print("2. 如果logger创建成功且eval_video=True，recorder应该会创建")
    print("3. 需要检查实际运行时logger是否创建成功")
    print("="*80)