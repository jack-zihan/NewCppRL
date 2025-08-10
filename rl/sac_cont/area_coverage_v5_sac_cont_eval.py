"""
Area Coverage V5 SAC评估脚本
用于评估V5环境（20通道，SGCNN）训练的模型
"""
import sys
import time
from pathlib import Path

import torch
import yaml
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from torchrl_utils.custom_evaluator import CustomEvaluator
from rl.sac_cont.area_coverage_v5_utils import make_area_coverage_v5_env

base_dir = Path(__file__).parent.parent.parent


class AreaCoverageV5SacEvaluator(CustomEvaluator):
    algo_name = 'area_coverage_v5_sac_cont'
    
    # 使用V5配置文件
    env_cfg = DictConfig(yaml.load(
        open(f'{base_dir}/configs/env_config_area_coverage_v5.yaml'), 
        Loader=yaml.FullLoader
    ))
    
    def __init__(self, *args, **kwargs):
        # 获取环境ID
        env_id = self.env_cfg.env.params.id
        
        # V5环境的特殊处理
        self.is_v5_env = env_id == "Pasture-v5"
        
        # V5环境中，weed_ratio键实际存储的是coverage_rate值
        # 为了日志一致性，我们在显示时使用正确的名称
        self.metric_name = "coverage_rate" if self.is_v5_env else "weed_ratio"
        
        super().__init__(*args, **kwargs)
        
        print(f"使用环境: {env_id}")
        print(f"观察空间: 20通道 (SGCNN多尺度)")
        print(f"评估指标: {self.metric_name}")
        print(f"方向场奖励: 启用")
    
    def make_env(self, from_pixels=False):
        """创建V5评估环境"""
        return make_area_coverage_v5_env(
            num_envs=1,
            device=self.device,
            from_pixels=from_pixels
        )
    
    def log_evaluation_info(self, episode_rewards, episode_lengths, episode_metrics):
        """记录评估信息，使用正确的指标名称"""
        if self.logger:
            log_info = {
                "eval/episode_reward": episode_rewards.mean().item(),
                "eval/episode_length": episode_lengths.mean().item(),
                f"eval/{self.metric_name}": episode_metrics.mean().item() if episode_metrics.numel() > 0 else 0.0,
            }
            
            for key, value in log_info.items():
                self.logger.log_scalar(key, value, step=self.global_steps)
        
        # 打印评估结果
        print(f"\n评估结果 (V5环境):")
        print(f"  平均奖励: {episode_rewards.mean().item():.2f}")
        print(f"  平均长度: {episode_lengths.mean().item():.2f}")
        if episode_metrics.numel() > 0:
            print(f"  平均{self.metric_name}: {episode_metrics.mean().item():.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='评估V5 Area Coverage SAC模型')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='模型checkpoint路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='运行设备')
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='评估回合数')
    parser.add_argument('--render', action='store_true',
                        help='是否渲染环境')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = AreaCoverageV5SacEvaluator(
        ckpt_path=args.ckpt_path,
        device=args.device,
        num_episodes=args.num_episodes,
        render=args.render
    )
    
    # 运行评估
    print("\n开始评估V5模型...")
    evaluator.evaluate()
    
    print("\nV5模型评估完成！")


if __name__ == "__main__":
    main()