"""
测试验证：评估代码中GPU推理流程是否导致pixels丢失
"""
import sys
sys.path.append('/home/lzh/NewCppRL')

# 先导入envs_new以注册环境
import envs_new

import torch
from torchrl.envs import GymWrapper
from tensordict import TensorDict
import gymnasium as gym


def test_gpu_inference_pixels_loss():
    """测试GPU推理流程对pixels的影响"""
    print("\n=== 测试GPU推理流程对pixels的影响 ===")
    
    # 1. 创建环境
    print("\n1. 创建环境（from_pixels=True）:")
    env = gym.make("NewPasture-v5", render_mode='rgb_array')
    wrapped_env = GymWrapper(env, device="cpu", from_pixels=True, pixels_only=False)
    print(f"  环境创建成功")
    
    # 2. Reset获取初始tensordict
    print("\n2. Reset环境:")
    td = wrapped_env.reset()
    print(f"  Reset后的tensordict键: {list(td.keys())}")
    has_pixels_initial = "pixels" in td.keys()
    print(f"  初始包含pixels: {has_pixels_initial}")
    if has_pixels_initial:
        print(f"  pixels shape: {td['pixels'].shape}")
    
    # 3. 模拟GPU推理流程（只返回action）
    print("\n3. 模拟GPU推理流程:")
    print("  原始td包含的键:", list(td.keys()))
    
    # 模拟actor_critic只返回action的情况
    # 这是实际代码第411行的行为
    action_only_td = TensorDict({
        "action": wrapped_env.action_spec.rand()  # 随机动作
    }, batch_size=td.batch_size)
    print("  模拟actor_critic返回的td键:", list(action_only_td.keys()))
    print("  ⚠️ 注意：只有action，没有pixels和其他观察数据!")
    
    # 4. 用只包含action的td调用step
    print("\n4. 用只包含action的td调用step:")
    next_td_action_only = wrapped_env.step(action_only_td)
    print(f"  Step后的tensordict键: {list(next_td_action_only.keys())}")
    
    # 检查主层级是否有pixels
    has_pixels_main = "pixels" in next_td_action_only.keys()
    print(f"  主层级包含pixels: {has_pixels_main}")
    
    # 检查next子字典
    if "next" in next_td_action_only.keys():
        print(f"  Next子字典键: {list(next_td_action_only['next'].keys())}")
        has_pixels_next = "pixels" in next_td_action_only['next'].keys()
        print(f"  Next中包含pixels: {has_pixels_next}")
    
    # 5. 对比：用完整td调用step
    print("\n5. 对比测试 - 用完整td调用step:")
    
    # 重新reset
    td_full = wrapped_env.reset()
    # 添加action但保留所有原始数据
    td_full["action"] = wrapped_env.action_spec.rand()
    print(f"  完整td包含的键: {list(td_full.keys())}")
    
    next_td_full = wrapped_env.step(td_full)
    print(f"  Step后的tensordict键: {list(next_td_full.keys())}")
    
    # 检查主层级是否有pixels
    has_pixels_full_main = "pixels" in next_td_full.keys()
    print(f"  主层级包含pixels: {has_pixels_full_main}")
    
    # 检查next子字典
    if "next" in next_td_full.keys():
        has_pixels_full_next = "pixels" in next_td_full['next'].keys()
        print(f"  Next中包含pixels: {has_pixels_full_next}")
    
    # 6. 结论
    print("\n" + "="*60)
    print("测试结论:")
    print("="*60)
    
    if not has_pixels_main and has_pixels_full_main:
        print("🔴 验证了问题:")
        print("   当step输入只有action时，返回的tensordict主层级没有pixels!")
        print("   当step输入包含完整观察时，返回的tensordict主层级保留pixels!")
        print("\n   这正是评估代码中的问题:")
        print("   - actor_critic只返回action")
        print("   - step(action_only_td)导致pixels丢失")
        print("   - 这就是为什么视频录制找不到pixels的原因!")
    else:
        print("✓ 测试结果与预期不符，需要进一步调查")
        print(f"  只有action时主层级pixels: {has_pixels_main}")
        print(f"  完整td时主层级pixels: {has_pixels_full_main}")
    
    return has_pixels_main, has_pixels_full_main


def test_evaluate_flow_simulation():
    """更精确地模拟评估代码流程"""
    print("\n\n=== 精确模拟评估代码流程 ===")
    
    # 创建多个环境（模拟批量评估）
    print("\n1. 创建2个评估环境:")
    envs = []
    for i in range(2):
        env = gym.make("NewPasture-v5", render_mode='rgb_array')
        wrapped = GymWrapper(env, device="cpu", from_pixels=True, pixels_only=False)
        envs.append(wrapped)
    
    # Reset所有环境
    print("\n2. Reset所有环境:")
    tds = []
    for env in envs:
        td = env.reset()
        tds.append(td)
    print(f"  环境0 reset后包含pixels: {'pixels' in tds[0]}")
    print(f"  环境1 reset后包含pixels: {'pixels' in tds[1]}")
    
    # 第一步：模拟评估循环
    print("\n3. 模拟评估循环第一步:")
    
    # 收集active tensordict（模拟第397-402行）
    active_tds = [tds[0], tds[1]]
    
    # 批处理（模拟第407行）
    batch_td = torch.stack(active_tds)
    print(f"  批处理后batch_td包含的键: {list(batch_td.keys())}")
    
    # 模拟GPU推理（第409-411行）
    # actor_critic只返回action
    print("\n  模拟actor_critic推理（只返回action）:")
    batch_action = TensorDict({
        "action": torch.stack([envs[0].action_spec.rand(), envs[1].action_spec.rand()])
    }, batch_size=batch_td.batch_size)
    print(f"  actor返回的td只包含: {list(batch_action.keys())}")
    
    # unbind并执行step（模拟第414-417行）
    print("\n4. 执行环境step:")
    for i, td_with_action in enumerate(batch_action.unbind(0)):
        print(f"\n  环境{i}:")
        print(f"    step输入的td键: {list(td_with_action.keys())}")
        next_td = envs[i].step(td_with_action)
        tds[i] = next_td  # 更新tds
        print(f"    step后td主层级键: {list(next_td.keys())}")
        print(f"    主层级包含pixels: {'pixels' in next_td}")
        if "next" in next_td:
            print(f"    next子字典包含pixels: {'pixels' in next_td['next']}")
    
    # 视频录制时的检查（模拟第457行）
    print("\n5. 视频录制时检查pixels:")
    for idx in range(2):
        has_pixels = "pixels" in tds[idx]
        print(f"  tds[{idx}]['pixels']存在: {has_pixels}")
        if not has_pixels:
            print(f"    ⚠️ 这就是为什么视频录制失败!")
    
    print("\n" + "="*60)
    print("精确模拟结论:")
    print("="*60)
    print("评估代码流程确实会导致pixels丢失:")
    print("1. actor_critic只返回action，丢弃了观察数据")
    print("2. step输入只有action，输出的主层级不包含pixels")
    print("3. 视频录制代码在主层级找不到pixels，导致录制失败")


if __name__ == "__main__":
    # 运行测试
    test_gpu_inference_pixels_loss()
    test_evaluate_flow_simulation()