"""
直接测试wandb视频上传，找出问题所在
"""
import sys
sys.path.append('/home/lzh/NewCppRL')

import envs_new
import torch
import numpy as np
from torchrl_utils.local_video_recorder import LocalVideoRecorder
from torchrl.record.loggers import get_logger
from rl_new.sac_cont_sy.env_utils import make_single_environment
from rl_new.sac_cont_sy.model_utils import make_sac_models
from omegaconf import OmegaConf
import time


def test_basic_wandb_upload():
    """测试最基本的wandb视频上传"""
    print("\n" + "="*80)
    print("测试1: 基本wandb视频上传")
    print("="*80)
    
    # 创建wandb logger
    print("\n创建wandb logger:")
    try:
        logger = get_logger(
            logger_type="wandb",
            logger_name="/tmp/test_video",
            experiment_name=f"test_video_{int(time.time())}",
            wandb_kwargs={
                "project": "test-video-upload",
                "config": {"test": "basic_upload"},
                "mode": "online"
            }
        )
        print("  ✅ Logger创建成功")
    except Exception as e:
        print(f"  ❌ Logger创建失败: {e}")
        return False
    
    # 创建recorder
    print("\n创建recorder:")
    recorder = LocalVideoRecorder(
        device="cpu",
        max_len=10,
        use_memmap=False,
        make_grid=True,
        nrow=2,
        skip=1,
        fps=6
    )
    print(f"  recorder创建成功")
    
    # 录制一些假数据
    print("\n录制测试帧:")
    for i in range(3):
        # 创建4个假的环境画面 (4, 3, 64, 64)
        fake_pixels = torch.randn(4, 3, 64, 64) * 255
        fake_pixels = fake_pixels.clamp(0, 255).to(torch.uint8)
        recorder.apply(fake_pixels)
        print(f"  帧{i}: recorder.idx = {recorder.idx}")
    
    # Dump视频
    print("\nDump视频:")
    print(f"  dump前: recorder.idx = {recorder.idx}")
    vid_tensor = recorder.dump()
    print(f"  dump后: recorder.idx = {recorder.idx}")
    
    if vid_tensor is not None:
        print(f"  ✅ vid_tensor shape: {vid_tensor.shape}")
        print(f"  vid_tensor dtype: {vid_tensor.dtype}")
        print(f"  vid_tensor device: {vid_tensor.device}")
        
        # 上传到wandb
        print("\n上传到wandb:")
        try:
            logger.log_video('test/basic_video', vid_tensor, step=0)
            print("  ✅ log_video调用成功")
            return True
        except Exception as e:
            print(f"  ❌ log_video失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("  ❌ vid_tensor为None")
        return False


def test_real_environment_upload():
    """测试真实环境的视频上传"""
    print("\n" + "="*80)
    print("测试2: 真实环境视频上传")
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
    print("\n创建wandb logger:")
    try:
        logger = get_logger(
            logger_type="wandb",
            logger_name="/tmp/test_real_video",
            experiment_name=f"test_real_video_{int(time.time())}",
            wandb_kwargs={
                "project": "test-video-upload",
                "config": {"test": "real_environment"},
                "mode": "online"
            }
        )
        print("  ✅ Logger创建成功")
    except Exception as e:
        print(f"  ❌ Logger创建失败: {e}")
        return False
    
    # 创建环境
    print("\n创建环境:")
    eval_episodes = 4
    envs = []
    for i in range(eval_episodes):
        env = make_single_environment(cfg, device="cpu", seed=42+i, from_pixels=True)
        envs.append(env)
    print(f"  创建了{eval_episodes}个环境")
    
    # 创建recorder
    print("\n创建recorder:")
    recorder = LocalVideoRecorder(
        device="cpu",
        max_len=100,
        use_memmap=False,
        make_grid=True,
        nrow=2,
        skip=1,
        fps=6
    )
    
    # Reset环境
    print("\nReset环境:")
    tds = []
    for i, env in enumerate(envs):
        td = env.reset()
        tds.append(td)
        print(f"  环境{i}: 'pixels' in td = {'pixels' in td}")
    
    # 录制几帧（使用用户的简化代码，无防御）
    print("\n录制帧（无防御代码）:")
    for frame_num in range(3):
        print(f"\n  帧{frame_num}:")
        # 直接使用简化代码，让错误暴露
        pixels = [tds[i]["pixels"] for i in range(min(4, eval_episodes))]
        print(f"    收集了{len(pixels)}个pixels")
        
        stacked = torch.stack(pixels, 0)
        print(f"    Stacked shape: {stacked.shape}")
        
        recorder.apply(stacked)
        print(f"    recorder.idx = {recorder.idx}")
    
    # Dump并上传
    print("\nDump视频:")
    vid_tensor = recorder.dump()
    
    if vid_tensor is not None:
        print(f"  ✅ vid_tensor shape: {vid_tensor.shape}")
        
        print("\n上传到wandb:")
        try:
            logger.log_video('test/real_env_video', vid_tensor, step=0)
            print("  ✅ 上传成功")
            return True
        except Exception as e:
            print(f"  ❌ 上传失败: {e}")
            return False
    else:
        print("  ❌ vid_tensor为None")
        return False


def test_evaluation_simulation():
    """完全模拟evaluate_policy的视频录制流程"""
    print("\n" + "="*80)
    print("测试3: 模拟evaluate_policy完整流程")
    print("="*80)
    
    cfg = OmegaConf.create({
        'seed': 42,
        'env': {
            'env_id': 'NewPasture-v5',
            'env_kwargs': {}
        },
        'collector': {
            'env_per_collector': 1
        },
        'logger': {
            'backend': 'wandb',
            'eval_video': True,
            'eval_episodes': 4,
            'eval_max_steps': 20,
            'eval_video_skip': 5
        }
    })
    
    eval_cfg = cfg.logger
    step = 1000  # 模拟训练步数
    
    # 创建logger（模拟sac-async.py中的logger）
    print("\n创建logger:")
    logger = get_logger(
        logger_type="wandb",
        logger_name="/tmp/test_eval",
        experiment_name=f"test_eval_{int(time.time())}",
        wandb_kwargs={
            "project": "test-video-upload",
            "config": dict(cfg),
            "mode": "online"
        }
    )
    print("  Logger创建成功")
    
    # 创建环境（模拟evaluate_policy第357-359行）
    print("\n创建评估环境:")
    eval_envs = []
    eval_episodes = eval_cfg.eval_episodes
    for i in range(eval_episodes):
        env = make_single_environment(cfg, device="cpu", seed=cfg.seed + i, from_pixels=eval_cfg['eval_video'])
        eval_envs.append(env)
    
    # 创建recorder（模拟第362-369行）
    print("\n创建recorder:")
    recorder = None
    if eval_cfg.eval_video and logger is not None:
        max_frames = min(4, eval_episodes)
        recorder = LocalVideoRecorder(
            device="cpu",
            max_len=(eval_cfg.eval_max_steps * max_frames) // eval_cfg.eval_video_skip + 2,
            use_memmap=True,
            make_grid=True,
            nrow=2,
            skip=1,
            fps=6
        )
        print(f"  Recorder创建成功，max_len={recorder.max_len}")
    
    # 创建actor
    dummy_env = make_single_environment(cfg, device="cpu", from_pixels=False)
    actor_critic = make_sac_models(dummy_env, device="cpu")
    dummy_env.close()
    
    # 初始化
    dones = [False] * eval_episodes
    tds = []
    for env in eval_envs:
        td = env.reset()
        tds.append(td)
    
    # 评估循环（简化版）
    print("\n运行评估循环:")
    for t in range(eval_cfg.eval_max_steps):
        # Actor推理和step（简化）
        active_tds = []
        active_indices = []
        for idx, (td, done) in enumerate(zip(tds, dones)):
            if not done:
                active_tds.append(td)
                active_indices.append(idx)
        
        if not active_tds:
            break
        
        batch_td = torch.stack(active_tds)
        with torch.no_grad():
            batch_td = actor_critic[0](batch_td)
        
        for i, (td, idx) in enumerate(zip(batch_td.unbind(0), active_indices)):
            next_td = eval_envs[idx].step(td)
            tds[idx] = next_td
            done = next_td["next"]["done"]
            if hasattr(done, 'item'):
                done = done.item()
            dones[idx] = done
        
        # 视频录制（模拟第453-455行，用户的简化代码）
        if recorder and (t + 1) % eval_cfg.eval_video_skip == 0:
            print(f"  t={t}: 录制帧")
            # 直接使用简化代码，无防御
            pixels = [tds[i]["pixels"] for i in range(min(4, eval_episodes))]
            recorder.apply(torch.stack(pixels, 0))
            print(f"    recorder.idx = {recorder.idx}")
    
    # 关闭环境
    for env in eval_envs:
        env.close()
    
    # Dump并上传（模拟第465-468行）
    print("\nDump并上传视频:")
    if recorder:
        print(f"  dump前: recorder.idx = {recorder.idx}")
        vid_tensor = recorder.dump()
        print(f"  dump后: vid_tensor is None = {vid_tensor is None}")
        
        if vid_tensor is not None and logger is not None:
            print(f"  vid_tensor shape: {vid_tensor.shape}")
            print(f"  调用logger.log_video('eval_grid/step_{step}', vid_tensor, step={step})")
            
            try:
                logger.log_video(f'eval_grid/step_{step}', vid_tensor, step=step)
                print("  ✅ 上传成功！")
                print("\n请检查wandb网站是否有视频")
                return True
            except Exception as e:
                print(f"  ❌ 上传失败: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print(f"  无法上传: vid_tensor={vid_tensor is not None}, logger={logger is not None}")
            return False
    else:
        print("  recorder为None")
        return False


if __name__ == "__main__":
    print("开始测试wandb视频上传...")
    print("="*80)
    
    results = []
    
    # 测试1: 基本上传
    print("\n执行测试1...")
    result1 = test_basic_wandb_upload()
    results.append(("基本上传", result1))
    
    # 测试2: 真实环境
    print("\n执行测试2...")
    result2 = test_real_environment_upload()
    results.append(("真实环境", result2))
    
    # 测试3: 完整流程
    print("\n执行测试3...")
    result3 = test_evaluation_simulation()
    results.append(("完整流程", result3))
    
    # 总结
    print("\n" + "="*80)
    print("测试结果总结")
    print("="*80)
    for name, success in results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{name}: {status}")
    
    print("\n关键发现：")
    print("1. recorder.idx的值表示已录制的帧数")
    print("2. dump()只在idx>0时返回视频")
    print("3. 如果所有测试都成功，说明视频上传机制本身没问题")
    print("4. 如果某个测试失败，检查具体错误信息")
    print("\n请登录wandb网站查看是否有视频上传")