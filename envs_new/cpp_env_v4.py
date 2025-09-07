"""
CppEnv v4 - 移除杂草相关逻辑，专注于田地覆盖任务， 保留所有配置灵活性（mist、trajectory、多尺度等）
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Any
from gymnasium.wrappers import HumanRendering

from envs_new.cpp_env_base import CppEnvBase
from envs_new.components.dynamics import FieldCoverageUpdater, FieldTaskStatusUpdater

class CppEnv(CppEnvBase):
    """
    v4环境 - 田地覆盖任务
    """
    def __init__(self, render_mode="rgb_array", **kwargs):
        v4_defaults = {'use_mist': False, 'use_apf': False, 'use_trajectory': True, 'render_mist': False,
                       "num_obstacles_range" : (0,0), "reward_completion_bonus": 5000, # "reward_field_group_coef": 10, "reward_completion_bonus": 10000,
                       # 'map_dir': "envs_new/maps/field_coverage"
                       }
        final_kwargs = {**v4_defaults, **kwargs}
        super().__init__(render_mode=render_mode, **final_kwargs)
        
        # 移除weed相关组件
        self.scenario_generator.remove_component('weed')
        self.reward_system.remove_calculator('weed_removal')
        self.env_dynamics.remove_updater('weed')
        
        # 替换field相关updater： Field探索模式转为覆盖模式, 任务统计变为field任务
        self.env_dynamics.remove_updater('field')
        self.env_dynamics.add_updater('field', FieldCoverageUpdater())
        self.env_dynamics.remove_updater('status')
        self.env_dynamics.add_updater('status', FieldTaskStatusUpdater())
    
    def _get_observation_channels(self) -> int:
        """v4环境的观察通道数：field, obstacle, (trajectory)"""
        return 2 + int(self.config.use_trajectory)
    
    def _get_observation_maps(self) -> Dict[str, Dict[str, Any]]:
        """
        field覆盖观测：field, obstacle, (optional) trajectory，不包含weed and mist
        """
        obs_maps = {'field': {'map': self.maps_dict['field'], 'pad': 0.0},
            'obstacle': {'map': self.maps_dict['obstacle'], 'pad': 1.0}} # 边界视为障碍物

        if self.config.use_trajectory and 'trajectory' in self.maps_dict: # 可选轨迹地图
            obs_maps['trajectory'] = {'map': self.maps_dict['trajectory'], 'pad': 0.0}
        
        return obs_maps
    
    def _get_completion_ratio(self) -> np.ndarray:
        return np.array([self.env_state.field_coverage_ratio], dtype=np.float32) # 田地覆盖完成率 [0, 1]
    
    def _get_step_info(self) -> Dict[str, Any]:
        return {'crashed': self.env_state.crashed, 'finished': self.env_state.finished, 'timeout': self.env_state.timeout,
            'field_area': self.env_state.field_area, 'field_ratio': self.env_state.field_coverage_ratio}

if __name__ == "__main__":
    # 测试v4环境
    if_render = True
    episodes = 3
    
    # 创建环境，展示配置灵活性
    env = CppEnv(
        use_mist=False,          # 可选：使用迷雾
        render_mist=False,       # 可选：渲染迷雾
        use_trajectory=True,    # 可选：记录轨迹
        use_apf=False,          # 可选：不使用APF（v4默认不需要）
        use_multiscale=True,   # 可选：不使用多尺度
        num_obstacles_range=[3, 5],  # 可配置障碍物数量
    )
    
    if if_render:
        env = HumanRendering(env)
    
    for episode in range(episodes):
        print(f"\n--- Episode {episode + 1} ---")
        obs, info = env.reset(seed=120 + episode, options={
            'map_id': None,  # 使用随机地图
            'initial_position': None,
            'initial_direction': None,
        })
        
        print(f"Observation shape: {obs['observation'].shape}")
        print(f"Initial field coverage: {obs['completion_ratio'][0]:.2%}")
        
        env.action_space.seed(66)
        done = False
        step_count = 0
        total_reward = 0
        
        while not done:  # 限制步数用于测试
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            step_count += 1
            total_reward += reward
            
            if step_count % 20 == 0:
                print(f"  Step {step_count}: reward={reward:.4f}, "
                      f"coverage={obs['completion_ratio'][0]:.2%}")
            
            if if_render:
                env.render()
            
            if done or truncated:
                print(f"\nEpisode finished after {step_count} steps!")
                print(f"  Total reward: {total_reward:.2f}")
                print(f"  Final coverage: {obs['completion_ratio'][0]:.2%}")
                if info.get('crashed'):
                    print("  Termination: Crashed")
                elif info.get('finished'):
                    print("  Termination: Field fully covered!")
                elif info.get('timeout'):
                    print("  Termination: Timeout")
                break
    
    env.close()
    print("\n✅ v4 Environment test completed successfully!")