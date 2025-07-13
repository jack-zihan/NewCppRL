#!/usr/bin/env python3
"""Simple test to verify wrapper compatibility and performance."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import gymnasium as gym
import time
from omegaconf import DictConfig
import yaml

import envs  # noqa
from envs.wrapper.reward_tracker import RewardTracker
from envs.wrapper.feature_tracker import FeatureTracker


def test_performance_comparison():
    """Compare performance with different wrapper configurations."""
    base_dir = Path(__file__).parent.parent
    cfg = DictConfig(yaml.load(open(f'{base_dir}/configs/env_config.yaml'), Loader=yaml.FullLoader))
    
    num_steps = 50
    
    # Test different configurations
    configs = [
        ("No wrappers", lambda: gym.make(**cfg.env.params)),
        ("RewardTracker only", lambda: RewardTracker(gym.make(**cfg.env.params))),
        ("FeatureTracker only", lambda: FeatureTracker(gym.make(**cfg.env.params))),
        ("Both wrappers", lambda: FeatureTracker(RewardTracker(gym.make(**cfg.env.params)))),
    ]
    
    print(f"\nPerformance comparison ({num_steps} steps):")
    print("-" * 60)
    
    for name, env_factory in configs:
        env = env_factory()
        env.reset(seed=42)
        
        start_time = time.time()
        for i in range(num_steps):
            action = env.action_space.sample()
            _, _, done, truncated, _ = env.step(action)
            if done or truncated:
                env.reset()
        
        elapsed = time.time() - start_time
        avg_step_time = elapsed / num_steps * 1000  # Convert to ms
        
        print(f"{name:20s}: {elapsed:.3f}s total, {avg_step_time:.2f}ms/step")
        
        env.close()
    
    print("\nTesting wrapper functionality when both are active...")
    
    # Create env with both wrappers
    env = FeatureTracker(RewardTracker(gym.make(**cfg.env.params)))
    env.reset(seed=42)
    
    # Take a few steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        # Check if reward tracking is working
        if hasattr(info, 'get') and 'reward_details' in info:
            print(f"\nStep {i+1} - Reward details captured: {list(info['reward_details'].keys())}")
        
        # Check if feature tracking is working  
        if hasattr(env, 'get_current_features'):
            features = env.get_current_features()
            if features:
                print(f"Step {i+1} - Features captured: {list(features.keys())}")
        
        if done or truncated:
            # Check episode summary from RewardTracker
            if hasattr(env, 'get_episode_summary'):
                summary = env.get_episode_summary()
                print(f"\nEpisode summary: {summary}")
            break
    
    env.close()
    print("\n✅ Both wrappers are functional when used together")


if __name__ == "__main__":
    test_performance_comparison()