"""
测试VideoRecorder的行为，诊断视频上传问题
"""
import sys
sys.path.append('/home/lzh/NewCppRL')

import torch
import numpy as np
from torchrl_utils.local_video_recorder import LocalVideoRecorder


def test_recorder_dump_behavior():
    """测试recorder.dump()的行为"""
    print("\n" + "="*80)
    print("测试LocalVideoRecorder的dump()行为")
    print("="*80)
    
    # 创建recorder
    recorder = LocalVideoRecorder(
        max_len=100,
        skip=1,
        use_memmap=False,  # 为了简单测试用内存
        make_grid=True,
        nrow=2,
        fps=6,
    )
    
    print("\n1. 初始状态:")
    print(f"   recorder.idx = {recorder.idx}")
    print(f"   recorder.obs = {recorder.obs}")
    
    # 模拟第一次评估
    print("\n2. 第一次评估 - 添加一些帧:")
    for i in range(3):
        # 创建4个环境的假pixels (4, 3, 64, 64)
        fake_pixels = torch.randn(4, 3, 64, 64) * 255
        fake_pixels = fake_pixels.clamp(0, 255).to(torch.uint8)
        recorder.apply(fake_pixels)
        print(f"   After frame {i}: recorder.idx = {recorder.idx}")
    
    print("\n3. 调用dump():")
    vid_tensor1 = recorder.dump()
    print(f"   返回的tensor shape: {vid_tensor1.shape if vid_tensor1 is not None else None}")
    print(f"   dump()后: recorder.idx = {recorder.idx}")  # 关键：这里会重置为0！
    
    print("\n4. 第二次评估 - 再添加一些帧:")
    for i in range(2):
        fake_pixels = torch.randn(4, 3, 64, 64) * 255
        fake_pixels = fake_pixels.clamp(0, 255).to(torch.uint8)
        recorder.apply(fake_pixels)
        print(f"   After frame {i}: recorder.idx = {recorder.idx}")
    
    print("\n5. 不调用dump()，直接看状态:")
    print(f"   recorder.idx = {recorder.idx}")
    print(f"   recorder.obs.shape = {recorder.obs.shape if recorder.obs is not None else None}")
    
    print("\n6. 第三次评估 - 继续添加帧（模拟下一次训练的evaluate）:")
    for i in range(2):
        fake_pixels = torch.randn(4, 3, 64, 64) * 255
        fake_pixels = fake_pixels.clamp(0, 255).to(torch.uint8)
        recorder.apply(fake_pixels)
        print(f"   After frame {i}: recorder.idx = {recorder.idx}")
    
    print("\n7. 最后dump():")
    vid_tensor2 = recorder.dump()
    print(f"   返回的tensor shape: {vid_tensor2.shape if vid_tensor2 is not None else None}")
    print(f"   包含的帧数: {vid_tensor2.shape[1] if vid_tensor2 is not None else 0}")
    
    print("\n" + "="*80)
    print("关键发现")
    print("="*80)
    print("1. dump()会重置recorder.idx为0")
    print("2. dump()后继续添加的帧会从索引0开始覆盖")
    print("3. 如果不调用dump()，帧会累积在buffer中")
    print("4. 多次使用同一个recorder需要在合适的时机dump()")


def test_evaluation_loop_video_issue():
    """模拟sac_utils.py中的评估循环视频问题"""
    print("\n" + "="*80)
    print("模拟评估循环中的视频问题")
    print("="*80)
    
    # 模拟创建recorder（在evaluate函数开始）
    recorder = LocalVideoRecorder(
        max_len=100,
        skip=1,
        use_memmap=False,
        make_grid=True,
        nrow=2,
        fps=6,
    )
    
    # 模拟评估循环
    eval_episodes = 4
    max_steps = 50
    eval_video_skip = 10
    
    print("\n模拟评估循环:")
    frame_count = 0
    for t in range(max_steps):
        if (t + 1) % eval_video_skip == 0:
            # 模拟收集pixels
            pixels = [torch.randn(3, 64, 64) * 255 for _ in range(min(4, eval_episodes))]
            pixels_tensor = torch.stack(pixels, 0).clamp(0, 255).to(torch.uint8)
            recorder.apply(pixels_tensor)
            frame_count += 1
            print(f"   时间步 t={t}: 添加帧，recorder.idx = {recorder.idx}")
    
    print(f"\n循环结束后:")
    print(f"   共添加了 {frame_count} 帧")
    print(f"   recorder.idx = {recorder.idx}")
    
    # 模拟在循环外dump（这是sac_utils.py的做法）
    vid_tensor = recorder.dump()
    print(f"\n调用dump()后:")
    print(f"   返回的tensor shape: {vid_tensor.shape if vid_tensor is not None else None}")
    print(f"   recorder.idx = {recorder.idx}")
    
    # 关键问题：如果这个recorder被重用会怎样？
    print("\n如果recorder被重用（下一次evaluate调用）:")
    print(f"   recorder.idx从{recorder.idx}开始")
    print("   新的帧会覆盖之前的帧！")
    
    return vid_tensor


def compare_with_custom_evaluator():
    """对比CustomEvaluator的做法"""
    print("\n" + "="*80)
    print("对比CustomEvaluator的正确做法")
    print("="*80)
    
    print("\nCustomEvaluator的做法:")
    print("1. 在eval_actor函数内部创建和使用recorder")
    print("2. 每次eval_actor结束时立即dump()并上传")
    print("3. 这样每次评估都是独立的，不会有状态残留")
    
    print("\nsac_utils.py的问题:")
    print("1. recorder可能跨多次evaluate调用重用")
    print("2. dump()只在evaluate函数结束时调用一次")
    print("3. 如果recorder被重用，idx不是从0开始，可能导致问题")
    
    print("\n可能的解决方案:")
    print("1. 每次evaluate创建新的recorder（最简单）")
    print("2. 或者在evaluate开始时重置recorder状态")
    print("3. 或者像CustomEvaluator一样，在循环内定期dump()")


if __name__ == "__main__":
    print("开始诊断VideoRecorder问题...")
    print("="*80)
    
    # 测试1: recorder基本行为
    test_recorder_dump_behavior()
    
    # 测试2: 模拟评估循环问题
    vid_tensor = test_evaluation_loop_video_issue()
    
    # 测试3: 对比分析
    compare_with_custom_evaluator()
    
    print("\n" + "="*80)
    print("诊断完成")
    print("="*80)
    print("\n根本原因：")
    print("1. LocalVideoRecorder.dump()会重置内部索引self.idx=0")
    print("2. 如果recorder跨多次evaluate调用重用，状态可能不一致")
    print("3. sac_utils.py在循环外dump()，而CustomEvaluator在循环内dump()")
    print("\n建议：")
    print("- 确保每次evaluate()调用都创建新的recorder")
    print("- 或者在合适的时机调用dump()并上传视频")