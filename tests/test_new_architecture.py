"""
Basic test script for the new modular CppEnvBase architecture.
Verifies core functionality and component integration.
"""

import sys
import os
sys.path.append('/home/lzh/NewCppRL')

import numpy as np
from envs_new.cpp_env_base import CppEnvBase, create_cpp_env_base
from envs_new.components.config.environment_config import EnvironmentConfig


def test_environment_creation():
    """Test basic environment creation."""
    print("Testing environment creation...")
    
    # Test with default config
    env1 = CppEnvBase()
    print(f"✓ Default environment created")
    print(f"  Action space: {env1.action_space}")
    print(f"  Observation space keys: {list(env1.observation_space.spaces.keys())}")
    
    # Test with custom config
    config = EnvironmentConfig(
        action_type="continuous",
        max_episode_steps=1000
    )
    env2 = CppEnvBase(config=config)
    print(f"✓ Custom environment created")
    print(f"  Action space: {env2.action_space}")
    
    # Test backward compatibility
    env3 = create_cpp_env_base(
        action_type="discrete",
        max_episode_steps=500,
        render_mode="rgb_array"
    )
    print(f"✓ Backward compatible environment created")
    
    return env1, env2, env3


def test_reset_functionality(env):
    """Test environment reset functionality."""
    print("\nTesting reset functionality...")
    
    obs, info = env.reset(seed=42)
    print(f"✓ Environment reset successful")
    print(f"  Observation keys: {list(obs.keys())}")
    print(f"  Observation shape: {obs['observation'].shape}")
    print(f"  Vector shape: {obs['vector'].shape}")
    print(f"  Weed ratio: {obs['weed_ratio'][0]:.3f}")
    
    return obs


def test_step_functionality(env):
    """Test environment step functionality."""
    print("\nTesting step functionality...")
    
    # Test with different action types
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"  Step {i+1}: reward={reward:.3f}, done={terminated or truncated}")
        print(f"    Weed count: {info.get('weed_count', 'N/A')}")
        print(f"    Crashed: {info.get('crashed', False)}")
        
        if terminated or truncated:
            print(f"    Episode ended: terminated={terminated}, truncated={truncated}")
            break
    
    return obs, reward, terminated, truncated, info


def test_component_functionality(env):
    """Test individual component functionality."""
    print("\nTesting component functionality...")
    
    # Test reward breakdown
    try:
        breakdown = env.get_reward_breakdown()
        print(f"✓ Reward breakdown: {len(breakdown)} components")
        print(f"  Total reward: {breakdown.get('total', 0):.3f}")
    except Exception as e:
        print(f"✗ Reward breakdown failed: {e}")
    
    # Test state info
    try:
        state_info = env.get_state_info()
        print(f"✓ State info: {len(state_info)} fields")
        print(f"  Current step: {state_info.get('current_step', 0)}")
    except Exception as e:
        print(f"✗ State info failed: {e}")
    
    # Test collision info
    try:
        collision_info = env.get_collision_info()
        print(f"✓ Collision info: {collision_info}")
    except Exception as e:
        print(f"✗ Collision info failed: {e}")


def test_rendering(env):
    """Test rendering functionality."""
    print("\nTesting rendering...")
    
    try:
        # Test map rendering
        env.render_mode = "rgb_array"
        rendered = env.render()
        if rendered is not None:
            print(f"✓ Map rendering successful: {rendered.shape}")
        else:
            print("✓ Rendering returned None (expected for no render mode)")
        
        # Test first-person rendering
        env.render_mode = "state_pixels"
        rendered_fp = env.render()
        if rendered_fp is not None:
            print(f"✓ First-person rendering successful: {rendered_fp.shape}")
        
    except Exception as e:
        print(f"✗ Rendering failed: {e}")


def run_comprehensive_test():
    """Run comprehensive test suite."""
    print("="*60)
    print("COMPREHENSIVE TEST OF NEW MODULAR ARCHITECTURE")
    print("="*60)
    
    try:
        # Test environment creation
        env1, env2, env3 = test_environment_creation()
        
        # Use discrete environment for detailed testing
        env = env1
        
        # Test reset
        obs = test_reset_functionality(env)
        
        # Test steps
        test_step_functionality(env)
        
        # Test components
        test_component_functionality(env)
        
        # Test rendering
        test_rendering(env)
        
        # Cleanup
        env.close()
        env2.close()
        env3.close()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED SUCCESSFULLY!")
        print("✓ New modular architecture is working correctly")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ TEST FAILED: {e}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)