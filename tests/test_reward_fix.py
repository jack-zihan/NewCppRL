#!/usr/bin/env python
"""
测试evaluate_policy奖励获取修复
验证从next_td["next"]["reward"]正确获取奖励值
"""

import sys
import os
import torch
import yaml
from omegaconf import OmegaConf

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_new.sac_cont_sy.sac_utils import evaluate_policy
from rl_new.sac_cont_sy.env_utils import make_train_environment
from rl_new.sac_cont_sy.model_utils import make_sac_models

def test_evaluate_fix():
    """测试修复后的evaluate_policy"""
    print("=" * 60)
    print("测试evaluate_policy奖励获取修复")
    print("=" * 60)
    
    # 加载配置
    config_path = "/home/lzh/NewCppRL/rl_new/sac_cont_sy/config-async.yaml"
    with open(config_path, 'r') as f:
        cfg = OmegaConf.create(yaml.safe_load(f))
    
    # 快速测试配置
    cfg.logger.eval_episodes = 2
    cfg.logger.eval_max_steps = 50
    cfg.logger.eval_video = False
    cfg.logger.show_progress = False
    
    # 创建环境和模型
    print("创建测试环境和模型...")
    env = make_train_environment(cfg, device="cpu")
    actor_critic = make_sac_models(env, device="cpu")
    env.close()
    
    try:
        print("\n执行评估...")
        metrics = evaluate_policy(
            actor_critic=actor_critic,
            cfg=cfg,
            train_device="cpu",
            logger=None,
            step=0
        )
        
        print("\n评估结果:")
        print(f"  平均奖励: {metrics['eval/reward_mean']:.4f}")
        print(f"  最小奖励: {metrics['eval/reward_min']:.4f}")
        print(f"  最大奖励: {metrics['eval/reward_max']:.4f}")
        print(f"  平均长度: {metrics['eval/episode_length']:.1f}")
        
        # 检查奖励是否合理（不应该都是0）
        if abs(metrics['eval/reward_mean']) < 0.001 and \
           abs(metrics['eval/reward_max']) < 0.001:
            print("\n⚠️ 警告：所有奖励都接近0，可能仍有问题！")
            return False
        else:
            print("\n✅ 奖励值非零，修复成功！")
            return True
            
    except KeyError as e:
        print(f"\n❌ KeyError: {e}")
        print("这表明数据访问路径不正确")
        return False
    except Exception as e:
        print(f"\n❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tensordict_access():
    """测试TensorDict访问路径"""
    print("\n" + "=" * 60)
    print("测试TensorDict结构访问")
    print("=" * 60)
    
    # 模拟step返回的结构
    from tensordict import TensorDict
    
    # 创建模拟数据
    next_td = TensorDict({
        "observation": torch.randn(25, 16, 16),
        "done": torch.tensor([False]),  # 根级别done
        "next": TensorDict({
            "observation": torch.randn(25, 16, 16),
            "reward": torch.tensor([1.234]),
            "done": torch.tensor([True]),  # next中的done
            "completion_ratio": torch.tensor([0.85])
        }, batch_size=[])
    }, batch_size=[])
    
    try:
        # 测试正确的访问方式
        reward = next_td["next"]["reward"]
        done = next_td["next"]["done"]
        completion_ratio = next_td["next"]["completion_ratio"]
        
        print(f"✅ 成功访问:")
        print(f"  reward = {reward.item():.3f}")
        print(f"  done = {done.item()}")
        print(f"  completion_ratio = {completion_ratio.item():.3f}")
        
        # 测试错误的访问方式（应该失败）
        try:
            bad_reward = next_td.get("next_reward", next_td.get("reward", 0))
            print(f"\n⚠️ 错误方式返回: {bad_reward} (应该是0)")
        except:
            print("\n错误方式已正确失败")
            
        return True
        
    except Exception as e:
        print(f"❌ 访问失败: {e}")
        return False

if __name__ == "__main__":
    # 运行测试
    test1_pass = test_tensordict_access()
    test2_pass = test_evaluate_fix()
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"TensorDict访问测试: {'✅ 通过' if test1_pass else '❌ 失败'}")
    print(f"evaluate_policy测试: {'✅ 通过' if test2_pass else '❌ 失败'}")
    
    if test1_pass and test2_pass:
        print("\n🎉 所有测试通过！修复成功。")
        sys.exit(0)
    else:
        print("\n⚠️ 有测试失败，需要进一步检查。")
        sys.exit(1)