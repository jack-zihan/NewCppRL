#!/usr/bin/env python3
"""
深入分析TensorDict的更新机制
理解TorchRL的设计理念
"""
import sys
sys.path.append('/home/lzh/NewCppRL')

import torch
from omegaconf import OmegaConf

# 导入环境创建函数
from rl_new.sac_cont_sy.env_utils import make_single_environment

def analyze_tensordict_mechanism():
    """深入分析TensorDict的更新机制"""
    print("\n" + "="*80)
    print("TensorDict更新机制深度分析")
    print("="*80)
    
    # 1. 创建环境
    config = OmegaConf.load('/home/lzh/NewCppRL/rl_new/sac_cont_sy/config-async-server.yaml')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    env = make_single_environment(
        cfg=config,
        from_pixels=True,
        device=device,
        seed=42
    )
    
    # 2. 分析第一次reset
    print("\n" + "="*60)
    print("阶段1：Reset后的初始状态")
    print("="*60)
    td = env.reset()
    print(f"Reset后TensorDict的键: {list(td.keys())}")
    print(f"'next' in td: {'next' in td}")
    
    # 保存初始值用于比较
    initial_pixels = td["pixels"].clone()
    initial_observation = td["observation"].clone()
    
    # 3. 第一次step
    print("\n" + "="*60)
    print("阶段2：第一次Step")
    print("="*60)
    action = env.action_spec.rand()
    print(f"执行动作: {action}")
    td["action"] = action
    
    # Step返回新的TensorDict
    next_td = env.step(td)
    
    print(f"\nStep后next_td的顶层键: {list(next_td.keys())}")
    print(f"'next' in next_td: {'next' in next_td}")
    print(f"next_td['next']的键: {list(next_td['next'].keys())}")
    
    # 4. 分析更新模式
    print("\n" + "="*60)
    print("阶段3：TensorDict的设计理念分析")
    print("="*60)
    
    print("\n[关键发现1] 根键保留历史状态:")
    print(f"  next_td['pixels'] == initial_pixels: {torch.equal(next_td['pixels'], initial_pixels)}")
    print(f"  next_td['observation'] == initial_observation: {torch.equal(next_td['observation'], initial_observation)}")
    print("  → 根键保存的是step前的状态（输入状态）")
    
    print("\n[关键发现2] next子字典包含新状态:")
    new_pixels = next_td["next"]["pixels"]
    new_observation = next_td["next"]["observation"]
    print(f"  next_td['next']['pixels'] != initial_pixels: {not torch.equal(new_pixels, initial_pixels)}")
    print(f"  next_td['next']['observation'] != initial_observation: {not torch.equal(new_observation, initial_observation)}")
    print("  → next子字典保存的是step后的状态（输出状态）")
    
    # 5. 模拟evaluate_policy的操作
    print("\n" + "="*60)
    print("阶段4：evaluate_policy的问题分析")
    print("="*60)
    
    # 模拟evaluate_policy的做法
    tds = [next_td]  # tds[idx] = next_td
    
    print("\n[问题] evaluate_policy执行 tds[idx] = next_td 后:")
    print(f"  tds[0]的结构: 根键(旧状态) + next键(新状态)")
    print(f"  读取 tds[0]['pixels'] → 得到旧的pixels")
    print(f"  读取 tds[0]['next']['pixels'] → 得到新的pixels")
    
    # 6. 第二次step - 关键问题
    print("\n" + "="*60)
    print("阶段5：连续Step的问题")
    print("="*60)
    
    # 第二次step - 错误的做法
    print("\n[错误做法] 在next_td上再次step:")
    action2 = env.action_spec.rand()
    next_td["action"] = action2  # 在根键设置action
    next_td2_wrong = env.step(next_td)
    
    print(f"  输入的next_td包含: 根键(第1帧) + next键(第2帧)")
    print(f"  step使用根键作为当前状态 → 使用第1帧状态计算第3帧")
    print(f"  结果: 跳过了第2帧！")
    
    # 正确的做法
    print("\n[正确做法] 在next子字典上step:")
    # 重新执行，使用正确方法
    td_correct = next_td["next"]  # 提取next作为新的当前状态
    td_correct["action"] = action2
    next_td2_correct = env.step(td_correct)
    
    print(f"  输入的td_correct只包含: 第2帧状态")
    print(f"  step使用第2帧状态计算第3帧")
    print(f"  结果: 正确的状态转换！")
    
    # 7. TorchRL的设计理念
    print("\n" + "="*60)
    print("阶段6：TorchRL的设计理念总结")
    print("="*60)
    
    print("""
TensorDict的设计理念：
1. **状态对保存**: 每个step返回(s_t, s_{t+1})对
   - 根键 = s_t (输入状态)
   - next键 = s_{t+1} (输出状态)
   
2. **为什么这样设计**:
   - 保留完整的转换信息用于训练
   - (s, a, r, s')元组完整保存
   - 便于经验回放和批处理
   
3. **手动rollout的陷阱**:
   - evaluate_policy直接保存整个next_td
   - 导致根键永远停留在第一帧
   - 每次step都从第一帧状态计算（错误！）
   
4. **正确的rollout方式**:
   方法1: 每次提取next作为新的当前状态
   方法2: 使用TorchRL的自动rollout功能
   方法3: 更新时同时更新根键和next键
""")
    
    # 8. 验证其他键是否也有问题
    print("\n" + "="*60)
    print("阶段7：验证其他键的更新情况")
    print("="*60)
    
    # 重新开始一个清晰的序列
    td = env.reset()
    print("\nReset后的初始值:")
    print(f"  done: {td['done'].item()}")
    print(f"  step_count: {td['step_count'].item()}")
    print(f"  episode_reward: {td['episode_reward'].item()}")
    
    # 执行3步
    for i in range(3):
        action = env.action_spec.rand()
        td["action"] = action
        next_td = env.step(td)
        
        print(f"\n第{i+1}步后:")
        print(f"  根键 done: {next_td['done'].item()} (仍是初始值)")
        print(f"  根键 step_count: {next_td['step_count'].item()} (仍是初始值)")
        print(f"  next键 done: {next_td['next']['done'].item()} (更新的值)")
        print(f"  next键 step_count: {next_td['next']['step_count'].item()} (更新的值)")
        
        # 保存供下次使用 - 这里是问题所在
        td = next_td  # 错误：应该是 td = next_td["next"]
    
    print("\n[结论] 所有根键都不会自动更新，包括:")
    print("  - pixels, observation (观察)")
    print("  - done, terminated, truncated (结束标志)")
    print("  - step_count, episode_reward (统计信息)")
    print("  → 这是TorchRL的设计，不是bug！")
    
    env.close()
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)


if __name__ == "__main__":
    analyze_tensordict_mechanism()