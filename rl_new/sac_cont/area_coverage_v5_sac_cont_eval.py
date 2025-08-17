"""
Area Coverage V5 SAC评估脚本
用于评估V5环境（20通道，SGCNN）训练的模型
"""
import sys
import time
from pathlib import Path
from typing import Any

import numpy
import numpy as np
import torch
import yaml
from omegaconf import DictConfig
from torchrl.envs import ExplorationType, set_exploration_type

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
        print(f"方向场奖励: 启用")

    def make_env(self, from_pixels=False):
        """创建V5评估环境"""
        return make_area_coverage_v5_env(
            num_envs=1,
            device=self.device,
            from_pixels=from_pixels
        )
    
    def get_actions(self,
                    actor: torch.nn.Module,
                    obss: list[Any]) -> list[float]:
        """获取连续动作（SAC使用连续动作空间）"""
        from tensordict import TensorDict
        
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            observation = []
            vector = []
            for obs in obss:
                if isinstance(obs, dict):
                    observation.append(obs['observation'])
                    # 处理vector，确保是标量
                    v = obs['vector']
                    if isinstance(v, np.ndarray):
                        v = v.item() if v.size == 1 else v[0]
                    vector.append([v])
            
            observation = torch.from_numpy(np.stack(observation, axis=0)).float().to(self.device)
            vector = torch.tensor(numpy.array(vector)).float().to(self.device)
            
            # 使用TensorDict调用actor（关键修改）
            td = TensorDict({
                'observation': observation,
                'vector': vector
            }, batch_size=[len(obss)])
            
            # 调用actor
            td_out = actor(td)
            
            # 从TensorDict输出中提取动作
            if 'action' in td_out:
                actions = td_out['action'].tolist()
            else:
                # 如果没有action键，尝试其他处理方式
                raise ValueError("Actor输出中没有'action'键")
                
        return actions
    
    def get_actor(self,
                  pt_path: str) -> torch.nn.Module:
        """加载SAC actor模型"""
        model = torch.load(pt_path)
        actor = model[0].to(self.device)
        return actor


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
        episodes=args.num_episodes,
        video=args.render
    )

    # 运行评估
    print("\n开始评估V5模型...")
    evaluator.run()
    
    print("\nV5模型评估完成！")


if __name__ == "__main__":
    main()