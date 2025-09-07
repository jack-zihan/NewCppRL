"""
全面诊断视频上传问题
"""
import sys
sys.path.append('/home/lzh/NewCppRL')

import envs_new
import torch
import numpy as np
from torchrl_utils.local_video_recorder import LocalVideoRecorder
from rl_new.sac_cont_sy.env_utils import make_single_environment
from rl_new.sac_cont_sy.model_utils import make_sac_models
from omegaconf import OmegaConf


def test_pixels_existence_in_tensordict():
    """测试pixels是否真的存在于tensordict中"""
    print("\n" + "="*80)
    print("测试1: 验证pixels是否存在于TensorDict中")
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
    
    # 创建环境（关键：from_pixels=True）
    print("\n创建环境 (from_pixels=True):")
    env = make_single_environment(cfg, device="cpu", seed=42, from_pixels=True)
    
    # Reset获取初始tensordict
    td = env.reset()
    print(f"  Reset后的键: {list(td.keys())}")
    print(f"  包含pixels: {'pixels' in td}")
    
    if 'pixels' in td:
        print(f"  pixels shape: {td['pixels'].shape}")
        print(f"  pixels dtype: {td['pixels'].dtype}")
        print(f"  pixels设备: {td['pixels'].device}")
        
        # 检查pixels的值范围
        pixels_min = td['pixels'].min().item()
        pixels_max = td['pixels'].max().item()
        print(f"  pixels值范围: [{pixels_min}, {pixels_max}]")
        
        # 检查是否全为0（可能的问题）
        if pixels_max == 0:
            print("  ⚠️ 警告：pixels全为0！这可能是渲染问题")
    else:
        print("  ❌ 错误：pixels不存在！")
    
    return td


def test_recorder_apply_with_real_pixels():
    """测试recorder.apply()是否正确处理真实的pixels"""
    print("\n" + "="*80)
    print("测试2: recorder.apply()处理真实pixels")
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
    
    # 创建多个环境
    eval_episodes = 4
    envs = []
    for i in range(eval_episodes):
        env = make_single_environment(cfg, device="cpu", seed=42+i, from_pixels=True)
        envs.append(env)
    
    # 创建recorder
    recorder = LocalVideoRecorder(
        device="cpu",
        max_len=100,
        use_memmap=False,  # 为了测试用内存
        make_grid=True,
        nrow=2,
        skip=1,
        fps=6,
    )
    
    # Reset所有环境
    tds = []
    for env in envs:
        td = env.reset()
        tds.append(td)
    
    print("\n收集pixels并应用到recorder:")
    
    # 尝试用户的简化代码
    pixels = [tds[i]["pixels"] for i in range(min(4, eval_episodes))]
    print(f"  收集了{len(pixels)}个pixels")
    
    for i, p in enumerate(pixels):
        print(f"  pixels[{i}]: shape={p.shape}, dtype={p.dtype}, device={p.device}")
    
    # Stack并应用
    try:
        stacked_pixels = torch.stack(pixels, 0)
        print(f"\n  Stacked shape: {stacked_pixels.shape}")
        print(f"  Stacked dtype: {stacked_pixels.dtype}")
        
        recorder.apply(stacked_pixels)
        print(f"  ✅ recorder.apply()成功")
        print(f"  recorder.idx = {recorder.idx}")
        
        # 尝试dump
        vid_tensor = recorder.dump()
        if vid_tensor is not None:
            print(f"  ✅ dump()成功: shape={vid_tensor.shape}")
        else:
            print("  ❌ dump()返回None")
            
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    
    return recorder


def test_video_upload_conditions():
    """测试视频上传的所有条件"""
    print("\n" + "="*80)
    print("测试3: 视频上传条件检查")
    print("="*80)
    
    print("\n代码分析（sac_utils.py第465-468行）:")
    print("""
    if recorder:
        vid_tensor = recorder.dump()
        if vid_tensor is not None and logger is not None:
            logger.log_video(f'eval_grid/step_{step}', vid_tensor, step=step)
    """)
    
    print("\n需要满足的条件:")
    print("1. recorder不为None ✓ (第363行创建)")
    print("2. vid_tensor = recorder.dump()不为None")
    print("3. logger不为None")
    
    print("\nrecorder.dump()返回None的条件（第144-146行）:")
    print("""
    def dump(self, filepath: Optional[str] = None) -> None | torch.Tensor:
        vid_tensor = None
        if self.idx > 0:
            vid_tensor = self.obs[:, :self.idx]
    """)
    
    print("\n关键：self.idx > 0才会返回视频！")
    print("如果没有调用recorder.apply()或者apply失败，idx=0，dump()返回None")


def test_simplified_code_issue():
    """测试用户简化代码的潜在问题"""
    print("\n" + "="*80)
    print("测试4: 用户简化代码的潜在问题")
    print("="*80)
    
    print("\n用户的简化代码:")
    print("""
    if recorder and (t + 1) % eval_cfg.eval_video_skip == 0:
        pixels = [tds[i]["pixels"] for i in range(min(4, eval_episodes))]
        recorder.apply(torch.stack(pixels, 0))
    """)
    
    print("\n潜在问题:")
    print("1. 如果某个tds[i]没有'pixels'键，会抛出KeyError")
    print("2. 如果pixels形状不一致，torch.stack会失败")
    print("3. 如果环境已经done，tds[i]可能没有更新")
    
    print("\n原代码的防御性检查（已注释）:")
    print("""
    # 原代码有防御性检查
    # if "pixels" in tds[idx]:
    #     pixels.append(tds[idx]["pixels"])
    """)
    
    print("\n建议：添加错误处理")
    print("""
    pixels = []
    for i in range(min(4, eval_episodes)):
        if i < len(tds) and "pixels" in tds[i]:
            pixels.append(tds[i]["pixels"])
    if pixels:
        recorder.apply(torch.stack(pixels, 0))
    """)


def simulate_actual_evaluation_flow():
    """完整模拟实际评估流程，找出问题"""
    print("\n" + "="*80)
    print("测试5: 完整模拟实际评估流程")
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
            'eval_video': True,
            'eval_episodes': 4,
            'eval_max_steps': 20,  # 少一点方便调试
            'eval_video_skip': 5
        }
    })
    
    eval_cfg = cfg.logger
    eval_episodes = eval_cfg.eval_episodes
    
    # 创建环境
    print("\n1. 创建环境:")
    eval_envs = []
    for i in range(eval_episodes):
        env = make_single_environment(cfg, device="cpu", seed=cfg.seed + i, from_pixels=True)
        eval_envs.append(env)
    print(f"   创建了{len(eval_envs)}个环境")
    
    # 创建recorder
    print("\n2. 创建recorder:")
    max_frames = min(4, eval_episodes)
    recorder = LocalVideoRecorder(
        device="cpu",
        max_len=(eval_cfg.eval_max_steps * max_frames) // eval_cfg.eval_video_skip + 2,
        use_memmap=False,  # 测试用内存
        make_grid=True,
        nrow=2,
        skip=1,
        fps=6
    )
    print(f"   Recorder创建成功，max_len={recorder.max_len}")
    
    # 创建actor
    dummy_env = make_single_environment(cfg, device="cpu", from_pixels=False)
    actor_critic = make_sac_models(dummy_env, device="cpu")
    dummy_env.close()
    
    # Reset环境
    print("\n3. Reset所有环境:")
    tds = []
    for i, env in enumerate(eval_envs):
        td = env.reset()
        tds.append(td)
        has_pixels = "pixels" in td
        print(f"   环境{i}: pixels={'✓' if has_pixels else '✗'}")
    
    # 模拟评估循环
    print("\n4. 运行评估循环:")
    dones = [False] * eval_episodes
    frame_count = 0
    
    for t in range(eval_cfg.eval_max_steps):
        # 收集活跃环境
        active_tds = []
        active_indices = []
        for idx, (td, done) in enumerate(zip(tds, dones)):
            if not done:
                active_tds.append(td)
                active_indices.append(idx)
        
        if not active_tds:
            print(f"   t={t}: 所有环境已结束")
            break
        
        # 批处理和actor推理
        batch_td = torch.stack(active_tds)
        with torch.no_grad():
            batch_td = actor_critic[0](batch_td)
        
        # Step环境
        for i, (td, idx) in enumerate(zip(batch_td.unbind(0), active_indices)):
            next_td = eval_envs[idx].step(td)
            tds[idx] = next_td
            
            done = next_td["next"]["done"]
            if hasattr(done, 'item'):
                done = done.item()
            dones[idx] = done
        
        # 视频录制（用户的简化代码）
        if recorder and (t + 1) % eval_cfg.eval_video_skip == 0:
            try:
                pixels = [tds[i]["pixels"] for i in range(min(4, eval_episodes))]
                recorder.apply(torch.stack(pixels, 0))
                frame_count += 1
                print(f"   t={t}: 录制帧{frame_count}, recorder.idx={recorder.idx}")
            except KeyError as e:
                print(f"   t={t}: ❌ KeyError - {e}")
            except Exception as e:
                print(f"   t={t}: ❌ 其他错误 - {e}")
    
    # Dump视频
    print("\n5. Dump视频:")
    vid_tensor = None
    if recorder:
        print(f"   recorder.idx = {recorder.idx}")
        vid_tensor = recorder.dump()
        if vid_tensor is not None:
            print(f"   ✅ dump()成功: shape={vid_tensor.shape}")
        else:
            print(f"   ❌ dump()返回None（可能idx=0）")
    
    # 检查上传条件
    print("\n6. 检查上传条件:")
    print(f"   recorder不为None: {recorder is not None}")
    print(f"   vid_tensor不为None: {vid_tensor is not None}")
    print(f"   logger不为None: False（测试中没有logger）")
    
    if vid_tensor is not None:
        print(f"\n   如果有logger，会调用:")
        print(f"   logger.log_video('eval_grid/step_0', vid_tensor, step=0)")
    else:
        print(f"\n   ❌ 不会上传视频，因为vid_tensor=None")
    
    return vid_tensor


if __name__ == "__main__":
    print("全面诊断视频上传问题...")
    print("="*80)
    
    # 测试1: 基础pixels检查
    td = test_pixels_existence_in_tensordict()
    
    # 测试2: recorder处理
    recorder = test_recorder_apply_with_real_pixels()
    
    # 测试3: 上传条件分析
    test_video_upload_conditions()
    
    # 测试4: 简化代码问题
    test_simplified_code_issue()
    
    # 测试5: 完整流程模拟
    vid_tensor = simulate_actual_evaluation_flow()
    
    print("\n" + "="*80)
    print("诊断总结")
    print("="*80)
    print("\n可能的问题：")
    print("1. pixels可能全为0（渲染问题）")
    print("2. 某些tds[i]可能没有'pixels'键（特别是done的环境）")
    print("3. recorder.idx可能为0（没有成功apply任何帧）")
    print("4. logger可能为None（配置问题）")
    print("\n建议检查：")
    print("1. 添加防御性检查确保pixels存在")
    print("2. 记录recorder.idx的值，确认是否有帧被录制")
    print("3. 确认logger不为None")
    print("4. 检查pixels的值是否正常（非全0）")