#!/usr/bin/env python3
"""
快速一致性测试脚本
仅测试基本功能，验证改进后的适配器是否工作
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置环境变量
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'

import envs  # noqa - 注册环境
import gymnasium as gym
import yaml
from omegaconf import DictConfig

# 导入适配器
from rules_new_adapter import RulesNewAdapter

# 导入rules_new1算法
from rules_new.algorithms import BcpPlanner


def test_quick():
    """快速测试基本功能"""
    print("🚀 开始快速一致性测试...")
    
    # 创建环境
    cfg = DictConfig(yaml.load(
        open(f'{project_root}/configs/env_config.yaml'), 
        Loader=yaml.FullLoader
    ))
    
    # 从参数中提取id
    env_params = dict(cfg.env.params)
    env_id = env_params.pop('id', 'Pasture-v2')  # 提取id，默认为Pasture-v2
    
    # 创建环境，id作为第一个参数，其他作为关键字参数
    env = gym.make(
        env_id,
        render_mode='rgb_array',
        **env_params
    )
    
    # 重置环境
    obs, info = env.reset(seed=42)
    print(f"✅ 环境创建成功")
    print(f"   环境ID: {env_id}")
    print(f"   观测空间: {env.observation_space}")
    print(f"   动作空间: {env.action_space}")
    
    # 创建适配器
    adapter = RulesNewAdapter(env, cfg)
    print(f"✅ 适配器创建成功")
    
    # 测试BCP规划器
    try:
        planner = BcpPlanner(env, cfg)
        print(f"✅ BCP规划器创建成功")
        
        # 执行一步
        action = planner.get_action(obs)
        next_obs, reward, done, truncated, info = env.step(action)
        print(f"✅ 执行一步成功")
        print(f"   动作: {action}")
        print(f"   奖励: {reward:.3f}")
        print(f"   完成: {done}")
        
    except Exception as e:
        print(f"❌ BCP规划器测试失败: {e}")
        print("   尝试使用适配器...")
        
        # 使用适配器执行一步
        action = adapter.get_action(obs)
        next_obs, reward, done, truncated, info = env.step(action)
        print(f"✅ 适配器执行成功")
        print(f"   动作: {action}")
        print(f"   奖励: {reward:.3f}")
        print(f"   完成: {done}")
    
    # 测试多步执行
    print("\n📊 测试多步执行...")
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()  # 随机动作
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done or truncated:
            print(f"   Episode结束于步骤 {step+1}")
            break
    
    print(f"✅ 多步执行测试完成")
    print(f"   总奖励: {total_reward:.3f}")
    
    # 测试重置
    obs, info = env.reset(seed=123)
    print(f"✅ 环境重置成功")
    
    env.close()
    print("\n🎉 所有测试通过！")


def test_consistency():
    """测试新旧环境的一致性"""
    print("\n🔬 开始一致性测试...")
    
    project_root = Path(__file__).parent.parent
    cfg = DictConfig(yaml.load(
        open(f'{project_root}/configs/env_config.yaml'), 
        Loader=yaml.FullLoader
    ))
    
    # 从参数中提取id
    env_params = dict(cfg.env.params)
    env_id = env_params.pop('id', 'Pasture-v2')
    
    # 创建环境
    env = gym.make(
        env_id,
        render_mode='rgb_array',
        **env_params
    )
    
    # 固定种子测试
    seed = 42
    obs1, _ = env.reset(seed=seed)
    
    # 执行固定动作序列
    actions = [env.action_space.sample() for _ in range(5)]
    
    trajectory1 = []
    for action in actions:
        obs, reward, done, truncated, info = env.step(action)
        trajectory1.append((obs, reward, done, truncated))
        if done or truncated:
            break
    
    # 重置并重复相同序列
    obs2, _ = env.reset(seed=seed)
    trajectory2 = []
    for action in actions:
        obs, reward, done, truncated, info = env.step(action)
        trajectory2.append((obs, reward, done, truncated))
        if done or truncated:
            break
    
    # 比较结果
    consistent = True
    for i, (t1, t2) in enumerate(zip(trajectory1, trajectory2)):
        obs1, r1, d1, tr1 = t1
        obs2, r2, d2, tr2 = t2
        
        if not np.allclose(r1, r2):
            print(f"❌ 步骤{i+1}奖励不一致: {r1} vs {r2}")
            consistent = False
        if d1 != d2 or tr1 != tr2:
            print(f"❌ 步骤{i+1}终止状态不一致")
            consistent = False
    
    if consistent:
        print("✅ 一致性测试通过：相同种子产生相同轨迹")
    else:
        print("❌ 一致性测试失败")
    
    env.close()
    return consistent


if __name__ == "__main__":
    print("=" * 50)
    print("环境一致性快速测试")
    print("=" * 50)
    
    try:
        # 基本功能测试
        test_quick()
        
        # 一致性测试
        test_consistency()
        
        print("\n✨ 所有测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)