"""
Test the new CppEnvBase environment step by step.
"""
import sys
import os
sys.path.append('/home/lzh/NewCppRL')

def test_env_creation():
    """Test creating the environment."""
    print("Testing environment creation...")
    try:
        from envs_new.cpp_env_base import CppEnvBase
        print("✓ CppEnvBase import successful")
        
        env = CppEnvBase()
        print("✓ Environment creation successful")
        return env
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_env_reset(env):
    """Test environment reset."""
    print("Testing environment reset...")
    try:
        obs, info = env.reset(seed=42)
        print("✓ Environment reset successful")
        print(f"  Observation keys: {list(obs.keys())}")
        for key, value in obs.items():
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        return obs, info
    except Exception as e:
        print(f"✗ Environment reset failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_env_step(env):
    """Test environment step."""
    print("Testing environment step...")
    try:
        action = env.action_space.sample()
        print(f"  Sampled action: {action}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        print("✓ Environment step successful")
        print(f"  Reward: {reward}")
        print(f"  Terminated: {terminated}")
        print(f"  Truncated: {truncated}")
        return True
    except Exception as e:
        print(f"✗ Environment step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run comprehensive tests."""
    print("=" * 60)
    print("COMPREHENSIVE TEST OF NEW CppEnvBase")
    print("=" * 60)
    
    # Test 1: Environment creation
    env = test_env_creation()
    if env is None:
        return
    
    print()
    
    # Test 2: Environment reset
    obs, info = test_env_reset(env)
    if obs is None:
        return
    
    print()
    
    # Test 3: Environment step
    if not test_env_step(env):
        return
    
    print()
    print("✓ All tests passed successfully!")

if __name__ == "__main__":
    main()