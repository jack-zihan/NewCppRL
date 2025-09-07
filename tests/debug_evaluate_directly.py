"""
直接在evaluate_policy中添加调试代码
"""

debug_code = """
# 在sac_utils.py的evaluate_policy函数中添加以下调试代码：

# 1. 在第362行recorder创建后添加：
if recorder:
    torchrl_logger.info(f"✅ Recorder创建成功: max_len={recorder.max_len}")
else:
    torchrl_logger.error("❌ Recorder为None!")

# 2. 在第453-455行视频录制处改为：
if recorder and (t + 1) % eval_cfg.eval_video_skip == 0:
    torchrl_logger.info(f"📹 t={t}: 开始录制帧, recorder.idx={recorder.idx}")
    
    # 检查tds状态
    for i in range(min(4, eval_episodes)):
        has_pixels = "pixels" in tds[i] if i < len(tds) else False
        torchrl_logger.info(f"  tds[{i}]: has_pixels={has_pixels}, done={dones[i]}")
    
    try:
        pixels = [tds[i]["pixels"] for i in range(min(4, eval_episodes))]
        torchrl_logger.info(f"  收集了{len(pixels)}个pixels")
        
        if pixels:
            stacked = torch.stack(pixels, 0)
            torchrl_logger.info(f"  Stack成功: shape={stacked.shape}")
            recorder.apply(stacked)
            torchrl_logger.info(f"  ✅ Apply成功: recorder.idx={recorder.idx}")
        else:
            torchrl_logger.error("  ❌ pixels列表为空!")
            
    except Exception as e:
        torchrl_logger.error(f"  ❌ 录制失败: {e}")
        import traceback
        torchrl_logger.error(traceback.format_exc())

# 3. 在第465-468行dump处改为：
if recorder:
    torchrl_logger.info(f"📦 准备dump: recorder.idx={recorder.idx}")
    torchrl_logger.info(f"  recorder.obs is None: {recorder.obs is None}")
    
    vid_tensor = recorder.dump()
    
    if vid_tensor is not None:
        torchrl_logger.info(f"  ✅ Dump成功: shape={vid_tensor.shape}, dtype={vid_tensor.dtype}")
        
        if logger is not None:
            torchrl_logger.info(f"  📤 上传视频到wandb...")
            logger.log_video(f'eval_grid/step_{step}', vid_tensor, step=step)
            torchrl_logger.info(f"  ✅ log_video调用完成")
        else:
            torchrl_logger.error("  ❌ Logger为None，无法上传!")
    else:
        torchrl_logger.error(f"  ❌ vid_tensor为None! recorder.idx={recorder.idx}")
        torchrl_logger.error("  原因：recorder.idx=0，说明没有成功apply任何帧")
else:
    torchrl_logger.error("❌ Recorder为None，无法dump!")

# 4. 在函数结束前添加：
torchrl_logger.info("📊 评估完成，返回metrics")
"""

print("调试代码建议：")
print("="*80)
print(debug_code)
print("="*80)
print("\n将以上代码添加到sac_utils.py后，运行训练即可看到详细的调试信息")
print("这样可以精确定位问题在哪一步")