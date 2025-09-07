#!/usr/bin/env python3
"""
evaluate_policy的最佳修复方案示例
正确处理TensorDict的状态管理
"""
import torch
from typing import List, Dict, Any

def evaluate_policy_fixed(
    actor_critic,
    cfg,
    train_device,
    logger=None,
    step: int = 0
) -> Dict[str, Any]:
    """
    修复版evaluate_policy - 正确管理TensorDict状态
    
    核心改进：
    1. 添加has_stepped标记跟踪每个环境的状态
    2. 根据标记决定step的输入和pixels的读取位置
    3. 保持所有原有功能（异步管理、视频录制、指标收集）
    """
    eval_cfg = cfg.logger
    eval_episodes = eval_cfg.eval_episodes
    
    # ... 环境创建代码 ...
    
    # 关键改进1：添加状态跟踪
    has_stepped = [False] * eval_episodes  # 标记每个环境是否已经step过
    
    # 初始化
    tds = []
    for idx, env in enumerate(eval_envs):
        td = env.reset()
        tds.append(td)
    
    # 统计变量
    episode_rewards = [0.0] * eval_episodes
    episode_lengths = [0] * eval_episodes
    dones = [False] * eval_episodes
    
    # 主循环
    for t in range(eval_cfg.eval_max_steps):
        # 收集活跃环境
        active_indices = []
        for idx, done in enumerate(dones):
            if not done:
                active_indices.append(idx)
        
        if not active_indices:
            break  # 所有环境完成
        
        # 批处理获取动作
        batch_td = torch.stack([
            tds[idx]["next"] if has_stepped[idx] else tds[idx]  # 关键改进2：使用正确的输入
            for idx in active_indices
        ])
        
        with torch.no_grad():
            batch_td = actor_critic[0](batch_td)
        
        # 执行动作
        for i, (td, idx) in enumerate(zip(batch_td.unbind(0), active_indices)):
            # 关键改进3：根据状态选择正确的step输入
            if has_stepped[idx]:
                # 已经step过，使用next中的状态
                tds[idx]["next"]["action"] = td["action"]
                next_td = eval_envs[idx].step(tds[idx]["next"])
            else:
                # 第一次step，使用根状态
                tds[idx]["action"] = td["action"]
                next_td = eval_envs[idx].step(tds[idx])
                has_stepped[idx] = True  # 标记已step
            
            # 保存完整的next_td（包含状态对）
            tds[idx] = next_td
            
            # 更新统计（从next读取）
            reward = next_td["next"]["reward"]
            episode_rewards[idx] += reward.item() if hasattr(reward, 'item') else reward
            episode_lengths[idx] += 1
            
            done = next_td["next"]["done"]
            dones[idx] = done.item() if hasattr(done, 'item') else done
        
        # 视频录制
        if recorder and (t + 1) % eval_cfg.eval_video_skip == 0:
            pixels = []
            for i in range(min(4, eval_episodes)):
                # 关键改进4：根据状态读取正确位置的pixels
                if has_stepped[i]:
                    # 已step，最新pixels在next中
                    pixels.append(tds[i]["next"]["pixels"])
                else:
                    # 未step（不应该发生，但以防万一）
                    pixels.append(tds[i]["pixels"])
            
            stacked = torch.stack(pixels, 0)
            recorder.apply(stacked)
    
    # ... 统计和返回结果 ...
    
    return eval_metrics


# 方案对比分析
def compare_solutions():
    """
    比较不同修复方案的优缺点
    """
    solutions = {
        "方案A: 提取next作为新状态": {
            "优点": [
                "完全符合TorchRL设计理念",
                "状态管理最清晰",
                "不会有任何状态混淆"
            ],
            "缺点": [
                "需要重构整个循环逻辑",
                "改动较大，风险较高",
                "可能丢失完整的转换信息"
            ],
            "适用场景": "全新实现或大规模重构"
        },
        
        "方案B: 仅修改pixels读取位置": {
            "优点": [
                "改动最小",
                "快速解决视频问题",
                "风险最低"
            ],
            "缺点": [
                "只解决了表面问题",
                "其他状态管理仍有隐患",
                "不符合根本设计理念"
            ],
            "适用场景": "紧急修复，临时方案"
        },
        
        "方案C: 使用TorchRL自动rollout": {
            "优点": [
                "最标准的实现",
                "避免手动状态管理",
                "代码最简洁"
            ],
            "缺点": [
                "需要大幅重构",
                "可能失去异步控制",
                "学习成本高"
            ],
            "适用场景": "新项目或完全重写"
        },
        
        "最佳方案: 混合方案（has_stepped标记）": {
            "优点": [
                "保持现有结构",
                "正确管理状态",
                "最小化改动",
                "保留所有功能",
                "易于理解和维护"
            ],
            "缺点": [
                "需要额外的状态跟踪",
                "代码略微复杂"
            ],
            "适用场景": "当前项目的最佳选择"
        }
    }
    
    return solutions


# 实施步骤
def implementation_steps():
    """
    修复方案的实施步骤
    """
    steps = [
        {
            "步骤": 1,
            "任务": "添加has_stepped标记数组",
            "改动": "has_stepped = [False] * eval_episodes",
            "位置": "初始化部分"
        },
        {
            "步骤": 2,
            "任务": "修改批处理输入选择",
            "改动": "tds[idx]['next'] if has_stepped[idx] else tds[idx]",
            "位置": "动作生成部分"
        },
        {
            "步骤": 3,
            "任务": "修改step调用逻辑",
            "改动": "根据has_stepped选择正确的输入",
            "位置": "环境step部分"
        },
        {
            "步骤": 4,
            "任务": "修改pixels读取位置",
            "改动": "根据has_stepped选择pixels位置",
            "位置": "视频录制部分"
        },
        {
            "步骤": 5,
            "任务": "测试验证",
            "改动": "确保视频正确、指标准确",
            "位置": "完整测试"
        }
    ]
    
    return steps


if __name__ == "__main__":
    print("evaluate_policy最佳修复方案")
    print("="*60)
    
    # 打印方案对比
    solutions = compare_solutions()
    for name, details in solutions.items():
        print(f"\n{name}:")
        print(f"  优点: {', '.join(details['优点'])}")
        print(f"  缺点: {', '.join(details['缺点'])}")
        print(f"  适用: {details['适用场景']}")
    
    print("\n" + "="*60)
    print("实施步骤:")
    for step_info in implementation_steps():
        print(f"\n步骤{step_info['步骤']}: {step_info['任务']}")
        print(f"  改动: {step_info['改动']}")
        print(f"  位置: {step_info['位置']}")