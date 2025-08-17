"""
Area Coverage V5 DQN评估脚本
用于评估V5环境（20通道，SGCNN）训练的DQN模型
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


class AreaCoverageV5DqnEvaluator(CustomEvaluator):
    algo_name = 'area_coverage_v5_dqn'
    
    # 使用V5配置文件
    env_cfg = DictConfig(yaml.load(
        open(f'{base_dir}/configs/env_config_area_coverage_v5.yaml'), 
        Loader=yaml.FullLoader
    ))
    
    def __init__(self, *args, **kwargs):
        # 处理ckpt_path，避免父类在没有checkpoint目录时报错
        if 'ckpt_path' in kwargs and kwargs['ckpt_path'] is not None:
            # 如果提供了ckpt_path，先保存它
            self.ckpt_path = kwargs['ckpt_path']
            self.ckpt_dir = str(self.ckpt_path).split('/')[-1]

        if 'num_episodes' in kwargs:
            kwargs['episodes'] = kwargs.pop('num_episodes')

        if 'render' in kwargs:
            kwargs['video'] = kwargs.pop('render')

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
        print(f"模型类型: DQN (离散动作)")

    def make_env(self, from_pixels=False):
        """创建V5评估环境"""
        return make_area_coverage_v5_env(
            num_envs=1,
            device=self.device,
            from_pixels=from_pixels
        )
    
    # 注意：DQN使用父类的get_actions和get_actor方法即可，它们已经处理了离散动作


def main():
    import argparse
    parser = argparse.ArgumentParser(description='评估V5 Area Coverage DQN模型')
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
    evaluator = AreaCoverageV5DqnEvaluator(
        ckpt_path=args.ckpt_path,
        device=args.device,
        episodes=args.num_episodes,
        video=args.render
    )

    # 运行评估
    print("\n开始评估V5 DQN模型...")
    evaluator.run()
    
    print("\nV5 DQN模型评估完成！")


if __name__ == "__main__":
    main()