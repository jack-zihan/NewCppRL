"""
诊断实际运行时的视频问题
"""
import sys
sys.path.append('/home/lzh/NewCppRL')

import envs_new
import torch


def diagnose_runtime_issues():
    """诊断实际运行时可能的问题"""
    print("\n" + "="*80)
    print("诊断实际运行时问题")
    print("="*80)
    
    print("\n1. Logger问题:")
    print("   检查点: sac_utils.py第467行")
    print("   条件: if vid_tensor is not None and logger is not None")
    print("   如果logger为None，即使有视频也不会上传")
    print("   建议: 添加日志确认logger状态")
    
    print("\n2. 用户简化代码的问题:")
    print("   原代码（有防御性检查）:")
    print("""
    pixels = []
    for idx in range(min(4, eval_episodes)):
        if not dones[idx] and "pixels" in tds[idx]:  # 防御性检查
            pixels.append(tds[idx]["pixels"])
    if pixels:  # 只有pixels非空才apply
        recorder.apply(torch.stack(pixels, 0))
    """)
    
    print("\n   用户简化代码（无防御）:")
    print("""
    pixels = [tds[i]["pixels"] for i in range(min(4, eval_episodes))]
    recorder.apply(torch.stack(pixels, 0))
    """)
    
    print("\n   潜在问题:")
    print("   - 如果环境done，step后的next_td可能结构不同")
    print("   - 直接访问tds[i]['pixels']可能抛出KeyError")
    print("   - 如果所有环境都done，pixels为空列表，torch.stack会失败")
    
    print("\n3. 循环时机问题:")
    print("   recorder.dump()在循环外（第466行）")
    print("   如果评估很快结束（所有环境early done），可能没录到足够帧")
    print("   recorder.idx可能为0，导致dump()返回None")
    
    print("\n4. 检查recorder状态的建议代码:")
    print("""
    # 在录制循环后，dump前添加调试信息
    if recorder:
        print(f"DEBUG: recorder.idx = {recorder.idx}")
        print(f"DEBUG: recorder.obs is None = {recorder.obs is None}")
        
        vid_tensor = recorder.dump()
        print(f"DEBUG: vid_tensor is None = {vid_tensor is None}")
        
        if vid_tensor is not None and logger is not None:
            print(f"DEBUG: Uploading video with shape {vid_tensor.shape}")
            logger.log_video(f'eval_grid/step_{step}', vid_tensor, step=step)
        else:
            print(f"DEBUG: Not uploading - vid_tensor={vid_tensor is not None}, logger={logger is not None}")
    """)
    
    print("\n5. 修复建议:")
    print("   方案A: 恢复防御性检查")
    print("""
    if recorder and (t + 1) % eval_cfg.eval_video_skip == 0:
        pixels = []
        for i in range(min(4, eval_episodes)):
            if i < len(tds) and not dones[i] and "pixels" in tds[i]:
                pixels.append(tds[i]["pixels"])
        if pixels:
            recorder.apply(torch.stack(pixels, 0))
    """)
    
    print("\n   方案B: 添加try-except")
    print("""
    if recorder and (t + 1) % eval_cfg.eval_video_skip == 0:
        try:
            pixels = [tds[i]["pixels"] for i in range(min(4, eval_episodes)) if not dones[i]]
            if pixels:
                recorder.apply(torch.stack(pixels, 0))
        except (KeyError, IndexError) as e:
            torchrl_logger.warning(f"Failed to record frame at t={t}: {e}")
    """)


def test_done_environment_pixels():
    """测试done环境的pixels情况"""
    print("\n" + "="*80)
    print("测试done环境的pixels")
    print("="*80)
    
    from rl_new.sac_cont_sy.env_utils import make_single_environment
    from omegaconf import OmegaConf
    
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
    
    print("\n创建环境并测试done后的行为:")
    env = make_single_environment(cfg, device="cpu", seed=42, from_pixels=True)
    
    # Reset
    td = env.reset()
    print(f"1. Reset后: 'pixels' in td = {'pixels' in td}")
    
    # 创建一个会导致done的action（假设）
    # 这里我们只是正常step
    from rl_new.sac_cont_sy.model_utils import make_sac_models
    dummy_env = make_single_environment(cfg, device="cpu", from_pixels=False)
    actor_critic = make_sac_models(dummy_env, device="cpu")
    dummy_env.close()
    
    # 运行直到done
    done = False
    step_count = 0
    max_steps = 1000
    
    print("\n2. 运行直到done:")
    while not done and step_count < max_steps:
        with torch.no_grad():
            td = actor_critic[0](td)
        
        next_td = env.step(td)
        
        # 检查done
        done = next_td["next"]["done"]
        if hasattr(done, 'item'):
            done = done.item()
        
        step_count += 1
        
        # 每100步报告一次
        if step_count % 100 == 0:
            has_pixels = "pixels" in next_td
            print(f"   Step {step_count}: done={done}, 'pixels' in next_td = {has_pixels}")
        
        if done:
            print(f"\n3. 环境在step {step_count}结束")
            print(f"   'pixels' in final next_td = {'pixels' in next_td}")
            print(f"   next_td.keys() = {list(next_td.keys())}")
            
            # 尝试再step一次（done后）
            print("\n4. 尝试在done后再step:")
            try:
                td_after_done = actor_critic[0](next_td)
                next_after_done = env.step(td_after_done)
                print(f"   成功step")
                print(f"   'pixels' in next_after_done = {'pixels' in next_after_done}")
            except Exception as e:
                print(f"   step失败: {e}")
        
        td = next_td
    
    if not done:
        print(f"\n未在{max_steps}步内结束")
    
    env.close()


if __name__ == "__main__":
    print("诊断实际运行时视频问题...")
    print("="*80)
    
    # 诊断分析
    diagnose_runtime_issues()
    
    # 测试done环境
    test_done_environment_pixels()
    
    print("\n" + "="*80)
    print("最可能的问题")
    print("="*80)
    print("\n根据分析，最可能的问题是：")
    print("1. 用户简化代码缺少防御性检查")
    print("2. 当环境done时，tds[i]可能没有'pixels'键或结构不同")
    print("3. 这导致KeyError或其他异常，recorder.apply()没有被调用")
    print("4. recorder.idx保持为0，dump()返回None")
    print("5. 即使logger不为None，也无法上传视频")
    
    print("\n建议的诊断步骤：")
    print("1. 在sac_utils.py中添加调试日志")
    print("2. 恢复防御性检查或添加try-except")
    print("3. 记录recorder.idx的值")
    print("4. 确认logger不为None")