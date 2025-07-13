#!/usr/bin/env python3
"""Test wrapper compatibility and state consistency between RewardTracker and FeatureTracker."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig
import yaml

import envs  # noqa
from envs.wrapper.reward_tracker import RewardTracker
from envs.wrapper.feature_tracker import FeatureTracker


def test_wrapper_state_consistency():
    """Test that wrappers don't cause state inconsistencies."""
    # Load environment config
    base_dir = Path(__file__).parent.parent
    cfg = DictConfig(yaml.load(open(f'{base_dir}/configs/env_config.yaml'), Loader=yaml.FullLoader))
    
    # Create three environments for comparison
    # 1. Base environment (no wrappers)
    env_base = gym.make(**cfg.env.params)
    
    # 2. Environment with only RewardTracker
    env_reward = gym.make(**cfg.env.params)
    env_reward = RewardTracker(env_reward)
    
    # 3. Environment with both wrappers
    env_both = gym.make(**cfg.env.params)
    env_both = RewardTracker(env_both)
    env_both = FeatureTracker(env_both)
    
    # Reset all environments with same seed
    seed = 42
    obs_base, _ = env_base.reset(seed=seed)
    obs_reward, _ = env_reward.reset(seed=seed)
    obs_both, _ = env_both.reset(seed=seed)
    
    # Convert observations to numpy arrays if needed
    if not isinstance(obs_base, np.ndarray):
        obs_base = np.array(obs_base)
    if not isinstance(obs_reward, np.ndarray):
        obs_reward = np.array(obs_reward)
    if not isinstance(obs_both, np.ndarray):
        obs_both = np.array(obs_both)
    
    # Verify initial observations are the same
    assert np.allclose(obs_base, obs_reward), "RewardTracker changed initial observation"
    assert np.allclose(obs_base, obs_both), "Combined wrappers changed initial observation"
    
    print("✓ Initial observations are consistent across all wrapper configurations")
    
    # Test multiple steps
    actions = [0, 1, 2, 3, 4]  # Test sequence of actions
    
    for i, action in enumerate(actions):
        print(f"\nStep {i+1}: Action {action}")
        
        # Step all environments
        obs_base, reward_base, done_base, trunc_base, info_base = env_base.step(action)
        obs_reward, reward_reward, done_reward, trunc_reward, info_reward = env_reward.step(action)
        obs_both, reward_both, done_both, trunc_both, info_both = env_both.step(action)
        
        # Convert observations to numpy arrays if needed
        if not isinstance(obs_base, np.ndarray):
            obs_base = np.array(obs_base)
        if not isinstance(obs_reward, np.ndarray):
            obs_reward = np.array(obs_reward)
        if not isinstance(obs_both, np.ndarray):
            obs_both = np.array(obs_both)
        
        # Check observations
        obs_diff_reward = np.mean(np.abs(obs_base - obs_reward))
        obs_diff_both = np.mean(np.abs(obs_base - obs_both))
        
        print(f"  Observation diff (RewardTracker): {obs_diff_reward:.6f}")
        print(f"  Observation diff (Both wrappers): {obs_diff_both:.6f}")
        
        # Check rewards
        print(f"  Reward base: {reward_base:.4f}")
        print(f"  Reward with RewardTracker: {reward_reward:.4f}")
        print(f"  Reward with both wrappers: {reward_both:.4f}")
        
        # Rewards should be identical
        assert abs(reward_base - reward_reward) < 1e-6, f"RewardTracker changed reward: {reward_base} vs {reward_reward}"
        assert abs(reward_base - reward_both) < 1e-6, f"Combined wrappers changed reward: {reward_base} vs {reward_both}"
        
        # Check done/truncated flags
        assert done_base == done_reward == done_both, "Done flags inconsistent"
        assert trunc_base == trunc_reward == trunc_both, "Truncated flags inconsistent"
        
        # Test APF state consistency
        if hasattr(env_base.unwrapped, 'use_apf'):
            base_apf = env_base.unwrapped.use_apf
            reward_apf = env_reward.unwrapped.use_apf
            both_apf = env_both.unwrapped.use_apf
            
            print(f"  APF state - Base: {base_apf}, RewardTracker: {reward_apf}, Both: {both_apf}")
            
            # All should have the same APF state after step
            assert base_apf == reward_apf == both_apf, "APF state inconsistent across wrappers"
    
    print("\n✅ All consistency tests passed!")
    
    # Test feature capture doesn't affect subsequent steps
    if hasattr(env_both, 'get_current_features'):
        print("\nTesting feature capture side effects...")
        
        # Get current state using the base environment's method
        actual_env = env_both.unwrapped
        obs_before = actual_env._get_observation() if hasattr(actual_env, '_get_observation') else None
        apf_before = actual_env.use_apf if hasattr(actual_env, 'use_apf') else None
        
        # Capture features
        features = env_both.get_current_features()
        
        # Check state after feature capture
        obs_after = actual_env._get_observation() if hasattr(actual_env, '_get_observation') else None
        apf_after = actual_env.use_apf if hasattr(actual_env, 'use_apf') else None
        
        if obs_before is not None and obs_after is not None:
            assert np.allclose(obs_before, obs_after), "Feature capture changed observation"
        
        if apf_before is not None and apf_after is not None:
            assert apf_before == apf_after, f"Feature capture didn't restore APF state: {apf_before} -> {apf_after}"
        
        print("✓ Feature capture correctly restores environment state")
    
    # Cleanup
    env_base.close()
    env_reward.close()
    env_both.close()


def test_wrapper_method_availability():
    """Test that wrapper methods are properly exposed."""
    base_dir = Path(__file__).parent.parent
    cfg = DictConfig(yaml.load(open(f'{base_dir}/configs/env_config.yaml'), Loader=yaml.FullLoader))
    
    # Create environment with both wrappers
    env = gym.make(**cfg.env.params)
    env = RewardTracker(env)
    env = FeatureTracker(env)
    
    # Check RewardTracker methods
    assert hasattr(env, 'get_episode_summary'), "Missing get_episode_summary from RewardTracker"
    
    # Check FeatureTracker methods
    assert hasattr(env, 'get_current_features'), "Missing get_current_features from FeatureTracker"
    
    print("✅ All wrapper methods are properly exposed")
    
    env.close()


def test_performance_comparison():
    """Compare performance with different wrapper configurations."""
    import time
    
    base_dir = Path(__file__).parent.parent
    cfg = DictConfig(yaml.load(open(f'{base_dir}/configs/env_config.yaml'), Loader=yaml.FullLoader))
    
    num_steps = 100
    
    # Test different configurations
    configs = [
        ("No wrappers", lambda: gym.make(**cfg.env.params)),
        ("RewardTracker only", lambda: RewardTracker(gym.make(**cfg.env.params))),
        ("FeatureTracker only", lambda: FeatureTracker(gym.make(**cfg.env.params))),
        ("Both wrappers", lambda: FeatureTracker(RewardTracker(gym.make(**cfg.env.params)))),
    ]
    
    print(f"\nPerformance comparison ({num_steps} steps):")
    print("-" * 50)
    
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


if __name__ == "__main__":
    print("Testing wrapper compatibility and state consistency...")
    print("=" * 60)
    
    test_wrapper_method_availability()
    print("\n" + "=" * 60)
    
    test_wrapper_state_consistency()
    print("\n" + "=" * 60)
    
    test_performance_comparison()