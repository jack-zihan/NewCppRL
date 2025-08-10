"""
环境创建模块 - 为rules_new提供环境接口
使用与sac_cont_test.py相同的方法
"""
import sys
from pathlib import Path
import yaml
from omegaconf import DictConfig

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import envs  # noqa - 自动注册所有环境
import gymnasium as gym
from gymnasium.wrappers import HumanRendering


def get_env(seed=42, render=False, weed_dist='gaussian', map_id=4, weed_num=100):
    """
    创建环境实例，使用与sac_cont_test相同的方法
    
    Args:
        seed: 随机种子
        render: 是否渲染
        weed_dist: 杂草分布类型
        map_id: 地图ID
        weed_num: 杂草数量
    
    Returns:
        env: 环境实例
        obs: 初始观察
    """
    # 加载环境配置（与sac_cont_test相同）
    cfg = DictConfig(yaml.load(
        open(f'{project_root}/configs/env_config.yaml'), 
        Loader=yaml.FullLoader
    ))
    
    # 创建环境（使用配置文件中的参数）
    env = gym.make(
        render_mode='rgb_array' if render else None,
        **cfg.env.params,
    )

    if render:
        env = HumanRendering(env)
        env.render()

    # 重置环境
    obs, info = env.reset(
        seed=seed, 
        options={
            'weed_dist': weed_dist, 
            'map_id': map_id, 
            'weed_num': weed_num
        }
    )
    
    return env, obs


# 用于测试的简化版本
def get_test_env(seed=42):
    """获取测试环境的简化接口"""
    return get_env(seed=seed, render=False, weed_dist='gaussian', map_id=4, weed_num=50)