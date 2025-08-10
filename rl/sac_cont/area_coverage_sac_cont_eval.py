"""
Area Coverage V4 SAC评估脚本
用于评估V4环境（4通道，无SGCNN）训练的模型

重要说明：
V4环境为了兼容性保留了'weed_ratio'键名，但实际存储的是coverage_rate值。
本脚本在内部使用正确的语义（coverage），但保持与环境的接口兼容。
"""

from pathlib import Path
from typing import Any

import numpy
import numpy as np
import torch
import yaml
from omegaconf import DictConfig
from torchrl.envs import ExplorationType, set_exploration_type

from torchrl_utils.custom_evaluator import CustomEvaluator


class AreaCoverageSacEvaluator(CustomEvaluator):
    """Area Coverage版本的SAC评估器"""
    
    algo_name = 'area_coverage_sac_cont'
    base_dir = Path(__file__).parent.parent.parent
    env_cfg = DictConfig(yaml.load(open(f'{base_dir}/configs/env_config_area_coverage.yaml'), Loader=yaml.FullLoader))
    
    def __init__(self, *args, **kwargs):
        # 处理ckpt_path，避免父类在没有checkpoint目录时报错
        if 'ckpt_path' in kwargs and kwargs['ckpt_path'] is not None:
            # 如果提供了ckpt_path，先保存它
            self.ckpt_path = kwargs['ckpt_path']
            self.ckpt_dir = str(self.ckpt_path).split('/')[-1]
        
        super().__init__(*args, **kwargs)
        
        # 检测环境类型以确定正确的度量名称
        env_id = self.env_cfg.env.params.id
        self.is_coverage_env = env_id in ["Pasture-v4", "Pasture-v5"]
        
        # 设置正确的度量名称（用于显示和日志）
        if self.is_coverage_env:
            self.metric_name = "coverage_rate"
            self.metric_display = "Coverage"
            print(f"检测到覆盖率环境 ({env_id})，使用coverage_rate作为评估指标")
        else:
            self.metric_name = "weed_ratio"
            self.metric_display = "Weed Ratio"
            print(f"检测到标准环境 ({env_id})，使用weed_ratio作为评估指标")
    
    def get_actions(self,
                    actor: torch.nn.Module,
                    obss: list[Any]) -> list[float]:
        """获取连续动作（SAC使用连续动作空间）"""
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            observation = []
            vector = []
            for obs in obss:
                if isinstance(obs, dict):
                    observation.append(obs['observation'])
                    vector.append([obs['vector']])
            observation = torch.from_numpy(np.stack(observation, axis=0)).float().to(self.device)
            vector = torch.tensor(numpy.array(vector)).float().to(self.device)
            # SAC actor返回的是连续动作，使用索引[2]获取确定性动作
            actions = actor(observation=observation, vector=vector)[2].tolist()
        return actions
    
    def get_actor(self,
                  pt_path: str) -> torch.nn.Module:
        """加载SAC actor模型"""
        model = torch.load(pt_path).to(self.device)
        actor = model[0]
        return actor
    
    def eval_actor(self,
                   envs: list,
                   actor: torch.nn.Module,
                   logger,
                   collected_frames: int,
                   recorder):
        """重写评估方法以适配coverage环境"""
        import time
        import tqdm
        from torchrl.envs.utils import set_exploration_type, ExplorationType
        
        eval_start = time.time()
        with set_exploration_type(ExplorationType.MODE), torch.no_grad():
            obss = []
            for env in envs:
                obs, _ = env.reset()
                obss.append(obs)
            
            rets = [0.] * self.episodes
            # 使用更通用的名称，但仍从'weed_ratio'键读取
            metric_values = [0.] * self.episodes  
            dones = [False] * self.episodes
            
            # Render
            if self.video:
                pixels = []
                for idx, env in enumerate(envs):
                    if idx >= self.max_frames:
                        break
                    pixels.append(env.render_map())
                recorder.apply(torch.from_numpy(np.stack(pixels, 0)))
            
            pbar = tqdm.tqdm(total=self.max_step)
            for t in range(self.max_step):
                pbar.update(1)
                if (t + 1) % self.skip_frames == 0:
                    rewards_mean = np.mean(rets)
                    rewards_min = np.min(rets)
                    rewards_max = np.max(rets)
                    rewards_std = rewards_max - rewards_min
                    # 使用正确的显示名称
                    pbar.set_postfix({
                        "reward": f'{rewards_mean:.2f} ± {rewards_std:.2f}, {rewards_min} ~ {rewards_max}',
                        self.metric_display: f"{np.mean(metric_values):.3f}",
                        'agents_alive': f'{dones.count(False)} / {self.episodes}'
                    })
                
                actions = self.get_actions(actor, obss)
                act_idx = 0
                obss = []
                for idx, env in enumerate(envs):
                    if not dones[idx]:
                        obs, reward, done, _, _ = env.step(actions[act_idx])
                        obss.append(obs)
                        rets[idx] += reward
                        dones[idx] |= done
                        # 仍从'weed_ratio'键读取，但我们知道V4/V5中这实际是coverage_rate
                        metric_values[idx] = obs["weed_ratio"]
                        act_idx += 1
                
                # Render
                if self.video and (t + 1) % self.skip_frames == 0:
                    for idx, env in enumerate(envs):
                        if idx >= self.max_frames:
                            break
                        if not dones[idx]:
                            pixels[idx] = env.render_map()
                    recorder.apply(torch.from_numpy(np.stack(pixels, 0)))
                
                if all(dones):
                    break
        
        eval_time = time.time() - eval_start
        rewards_mean = np.mean(rets)
        rewards_min = np.min(rets)
        rewards_max = np.max(rets)
        rewards_std = rewards_max - rewards_min
        metric_mean = np.mean(metric_values)
        
        # 构建日志信息，使用正确的名称
        log_info = {
            "eval/reward": rewards_mean,
            "eval/reward_min": rewards_min,
            "eval/reward_max": rewards_max,
            "eval/reward_std": rewards_std,
            f"eval/{self.metric_name}": metric_mean,  # 使用正确的度量名称
            "eval/eval_time": eval_time,
        }
        
        # 为了向后兼容，如果是coverage环境，同时记录两个名称
        if self.is_coverage_env:
            log_info["eval/weed_ratio"] = metric_mean  # 兼容旧的日志查看器
        
        if logger:
            for key, value in log_info.items():
                logger.log_scalar(key, value, step=collected_frames * 2)
            if self.video:
                video_tensor = recorder.dump()
                logger.log_video('eval/video', video_tensor, step=collected_frames * 2)
        
        print(f'\tEvaluation finished, cost {eval_time:.2f} seconds.')
        print(f'\tReward = {rewards_mean:.2f} ± {rewards_std:.2f}, {rewards_min} ~ {rewards_max}')
        print(f'\t{self.metric_display} = {metric_mean:.4f}')
        
        return rewards_mean, rewards_min, rewards_max


if __name__ == '__main__':
    # 使用示例
    evaluator = AreaCoverageSacEvaluator(
        episodes=4,          # 评估的episode数量
        max_frames=4,        # 视频记录的最大帧数
        max_step=1500,       # 每个episode的最大步数
        skip_frames=30,      # 视频跳帧
        video=True,          # 是否录制视频
        device='cpu',        # 设备
        start_idx=0,         # 起始索引
        ckpt_path=None,      # checkpoint路径，None则自动查找最新
        # ckpt_path='../../ckpt/area_coverage_sac_cont/2024xxxx_xxxxxx',  # 指定路径示例
    )
    evaluator.run()