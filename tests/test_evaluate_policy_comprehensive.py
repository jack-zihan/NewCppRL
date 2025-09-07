#!/usr/bin/env python3
"""
综合测试evaluate_policy函数
验证视频录制、指标收集和状态管理的正确性
"""
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch import optim
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchrl.modules.tensordict_module import ProbabilisticActor, SafeModule
from torchrl.modules.distributions import TanhNormal
# from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
import numpy as np
from typing import Dict, Any
import tempfile
import shutil

# Import our modules
from rl_new.sac_cont_sy.env_utils import make_single_environment
from rl_new.sac_cont_sy.sac_utils import log_metrics, evaluate_policy
import envs_new  # 触发环境注册

class DummyActorNetwork(nn.Module):
    """简单的Actor网络用于测试"""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2),  # mean and std
        )
        self.action_dim = action_dim
        
    def forward(self, observation):
        """Forward pass returning mean and log_std"""
        out = self.net(observation)
        mean = out[..., :self.action_dim]
        log_std = out[..., self.action_dim:]
        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

class DummyCriticNetwork(nn.Module):
    """简单的Critic网络用于测试"""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(self, observation, action):
        """Forward pass returning Q1 and Q2 values"""
        x = torch.cat([observation, action], dim=-1)
        return self.q1(x), self.q2(x)

def create_dummy_actor_critic(obs_dim: int, action_dim: int, device: str = "cpu"):
    """创建用于测试的actor_critic模块"""
    # Create actor network
    actor_net = DummyActorNetwork(obs_dim, action_dim).to(device)
    
    # Wrap as TensorDict module
    actor_module = SafeModule(
        module=actor_net,
        in_keys=["observation"],
        out_keys=["loc", "scale"],
    )
    
    # Create probabilistic actor
    actor = ProbabilisticActor(
        module=actor_module,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        return_log_prob=False,
    )
    
    # Create critic networks
    qvalue_net1 = DummyCriticNetwork(obs_dim, action_dim).to(device)
    qvalue_net2 = DummyCriticNetwork(obs_dim, action_dim).to(device)
    
    # Combine into a simple structure
    class ActorCriticModule(nn.Module):
        def __init__(self, actor_module, actor, qvalue_net1, qvalue_net2):
            super().__init__()
            self.actor_module = actor_module
            self.actor = actor
            self.qvalue_net1 = qvalue_net1
            self.qvalue_net2 = qvalue_net2
            
        def get_policy_operator(self):
            """Return policy operator for evaluation"""
            def policy_op(td):
                # Apply actor module to get mean and std
                td = self.actor_module(td)
                # Apply distribution module to sample action
                td = self.actor(td)
                return td
            return policy_op
    
    actor_critic = ActorCriticModule(actor_module, actor, qvalue_net1, qvalue_net2)
    return actor_critic

def test_evaluate_policy():
    """主测试函数"""
    print("=" * 80)
    print("综合测试 evaluate_policy 函数")
    print("=" * 80)
    
    # Load configuration
    cfg_path = Path(__file__).parent.parent / "rl_new/sac_cont_sy/config-async.yaml"
    cfg = OmegaConf.load(cfg_path)
    
    # Modify config for testing
    cfg.env.env_name = "CppEnvParallel-v0"
    cfg.env.frame_skip = 1
    cfg.env.from_pixels = True  # 确保录制视频
    cfg.env.pixels_only = False
    cfg.env.num_envs = 3  # 测试多个环境
    cfg.env.device = "cpu"
    
    # 设置视频录制参数
    cfg.logger.video = True
    cfg.logger.eval_envs = 3  # 录制3个环境的视频
    
    # 创建临时目录用于保存视频
    temp_dir = tempfile.mkdtemp(prefix="test_eval_policy_")
    video_dir = Path(temp_dir) / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n临时视频目录: {video_dir}")
    
    eval_env = None
    try:
        # 1. 创建环境
        print("\n1. 创建测试环境...")
        eval_env = make_single_environment(
            cfg, 
            device="cpu", 
            seed=42, 
            from_pixels=True
        )
        
        # 获取环境规格
        td_test = eval_env.reset()
        obs_dim = td_test["observation"].shape[-1]
        action_spec = eval_env.action_spec
        
        if hasattr(action_spec, 'shape'):
            action_dim = action_spec.shape[-1]
        else:
            # For composite specs, extract action dimension
            action_dim = 2  # Default for our environment
        
        print(f"   观察空间维度: {obs_dim}")
        print(f"   动作空间维度: {action_dim}")
        print(f"   像素形状: {td_test['pixels'].shape if 'pixels' in td_test else 'N/A'}")
        
        # 2. 创建随机初始化的actor_critic
        print("\n2. 创建随机初始化的 actor_critic...")
        actor_critic = create_dummy_actor_critic(obs_dim, action_dim, device="cpu")
        print(f"   Actor-Critic 模块创建成功")
        
        # 3. 测试策略操作符
        print("\n3. 测试策略操作符...")
        policy = actor_critic.get_policy_operator()
        
        # 测试单步
        td_test = eval_env.reset()
        td_with_action = policy(td_test)
        print(f"   生成的动作: {td_with_action['action'].detach().cpu().numpy()}")
        
        # 4. 调用evaluate_policy
        print("\n4. 调用 evaluate_policy 函数...")
        
        eval_metrics = evaluate_policy(
            eval_env=eval_env,
            actor=actor_critic,
            step=1000,  # 模拟训练步数
            cfg=cfg,
            save_video=True,
            save_dir=str(video_dir),
            device="cpu"
        )
        
        # 5. 打印收集到的指标
        print("\n5. 收集到的评估指标:")
        print("-" * 40)
        
        for key, value in eval_metrics.items():
            if isinstance(value, (int, float)):
                print(f"   {key:30s}: {value:>10.4f}")
            elif isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    print(f"   {key:30s}: {value.item():>10.4f}")
                else:
                    print(f"   {key:30s}: shape={value.shape}, mean={value.mean().item():.4f}")
            else:
                print(f"   {key:30s}: {type(value).__name__}")
        
        # 6. 验证关键指标
        print("\n6. 验证关键指标:")
        print("-" * 40)
        
        # 检查必须存在的指标
        required_metrics = [
            "eval/episode_reward",
            "eval/episode_reward_mean", 
            "eval/episode_length",
            "eval/episode_length_mean",
            "eval/completion_ratio",
            "eval/completion_ratio_mean"
        ]
        
        missing_metrics = []
        for metric in required_metrics:
            if metric in eval_metrics:
                print(f"   ✓ {metric} 存在")
            else:
                print(f"   ✗ {metric} 缺失")
                missing_metrics.append(metric)
        
        # 7. 检查视频文件
        print("\n7. 检查视频文件生成:")
        print("-" * 40)
        
        video_files = list(video_dir.glob("*.mp4"))
        if video_files:
            print(f"   ✓ 找到 {len(video_files)} 个视频文件:")
            for vf in video_files:
                file_size = vf.stat().st_size / 1024  # KB
                print(f"      - {vf.name} ({file_size:.1f} KB)")
                
                # 检查文件大小是否合理（应该大于几KB）
                if file_size > 5:
                    print(f"        ✓ 文件大小正常")
                else:
                    print(f"        ✗ 文件可能为空或损坏")
        else:
            print(f"   ✗ 未找到视频文件")
        
        # 8. 测试多步执行以验证状态管理
        print("\n8. 测试多步执行（验证状态管理）:")
        print("-" * 40)
        
        td = eval_env.reset()
        pixel_changes = []
        
        for step in range(5):
            # 获取当前像素
            if "pixels" in td:
                current_pixels = td["pixels"].clone()
            
            # 执行动作
            td = policy(td)
            transition = eval_env.step(td)
            
            # 使用step_mdp获取下一状态
            td = eval_env.step_mdp(transition)
            
            # 检查像素是否变化
            if "pixels" in td and step > 0:
                pixel_diff = (td["pixels"] - current_pixels).abs().sum().item()
                pixel_changes.append(pixel_diff)
                print(f"   步骤 {step+1}: 像素变化量 = {pixel_diff:.2f}")
        
        if pixel_changes:
            avg_change = np.mean(pixel_changes)
            if avg_change > 0:
                print(f"   ✓ 平均像素变化量: {avg_change:.2f} (画面正在更新)")
            else:
                print(f"   ✗ 平均像素变化量: {avg_change:.2f} (画面可能静止)")
        
        # 9. 总结
        print("\n" + "=" * 80)
        print("测试总结:")
        print("-" * 40)
        
        if not missing_metrics and video_files and (not pixel_changes or avg_change > 0):
            print("✅ 所有测试通过！")
            print("   - 所有必需指标都已正确收集")
            print("   - 视频文件已成功生成")
            print("   - 画面正确更新（非静止）")
        else:
            print("⚠️  发现一些问题:")
            if missing_metrics:
                print(f"   - 缺失指标: {missing_metrics}")
            if not video_files:
                print("   - 未生成视频文件")
            if pixel_changes and avg_change == 0:
                print("   - 画面未更新（静止）")
        
        return eval_metrics
        
    finally:
        # 清理临时目录
        print(f"\n清理临时目录: {temp_dir}")
        if eval_env is not None:
            eval_env.close()
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    metrics = test_evaluate_policy()
    print("\n测试完成！")