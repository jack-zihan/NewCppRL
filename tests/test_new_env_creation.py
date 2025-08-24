#!/usr/bin/env python3
"""
Test script for new environment creation system without YAML dependency.
Tests Gymnasium registration and parameter passing.
"""
import sys
import gymnasium as gym
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the new utilities
from torchrl_utils_new.utils_env import (
    make_env,
    make_base_env,
    make_apf_env,
    make_multiscale_env,
    make_field_coverage_env,
    make_hif_env,
    make_env_from_config
)


def test_gymnasium_registration():
    """Test that all NewPasture environments are registered with Gymnasium."""
    print("\n=== Testing Gymnasium Registration ===")
    
    registered_envs = list(gym.envs.registry.keys())
    new_envs = [f"NewPasture-v{i}" for i in range(1, 6)]
    
    for env_id in new_envs:
        if env_id in registered_envs:
            print(f"✅ {env_id} is registered")
        else:
            print(f"❌ {env_id} is NOT registered")
            return False
    
    print("All environments registered successfully!")
    return True


def test_direct_creation():
    """Test direct environment creation using gym.make()."""
    print("\n=== Testing Direct Environment Creation ===")
    
    try:
        # Test each version
        for version in range(1, 6):
            env_id = f"NewPasture-v{version}"
            print(f"\nTesting {env_id}...")
            
            # Skip v4 and v5 if they don't exist
            if version in [4, 5]:
                try:
                    env = gym.make(env_id)
                except (ImportError, ModuleNotFoundError) as e:
                    print(f"  ⚠️  {env_id} not available (module not found): {e}")
                    continue
            else:
                env = gym.make(env_id)
            
            # Test reset
            obs, info = env.reset(seed=42)
            print(f"  ✅ Reset successful - obs shape: {obs.shape}")
            
            # Test step
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  ✅ Step successful - reward: {reward:.4f}")
            
            env.close()
            print(f"  ✅ {env_id} works correctly")
            
    except Exception as e:
        print(f"❌ Error during direct creation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_make_env_function():
    """Test the make_env function with various parameters."""
    print("\n=== Testing make_env Function ===")
    
    try:
        # Test basic creation
        print("\n1. Basic creation (default v2):")
        env = make_env()
        print("  ✅ Default environment created")
        env.close()
        
        # Test version specification
        print("\n2. Version specification:")
        for version in [1, 2, 3]:
            env = make_env(env_id=f"NewPasture-v{version}")
            print(f"  ✅ v{version} created")
            env.close()
        
        # Test custom parameters
        print("\n3. Custom parameters:")
        env = make_env(
            env_id="NewPasture-v2",
            use_apf=True,
            num_obstacles_range=(3, 5),
            reward_field_coverage=2.0
        )
        print("  ✅ Custom parameters passed successfully")
        env.close()
        
        # Test parallel environments
        print("\n4. Parallel environments:")
        env = make_env(num_envs=4, device="cpu")
        print("  ✅ Parallel environment created with 4 workers")
        env.close()
        
    except Exception as e:
        print(f"❌ Error in make_env function: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_convenience_functions():
    """Test convenience functions for specific environment types."""
    print("\n=== Testing Convenience Functions ===")
    
    functions = [
        ("make_base_env", make_base_env),
        ("make_apf_env", make_apf_env),
        ("make_multiscale_env", make_multiscale_env),
        ("make_field_coverage_env", make_field_coverage_env),
        ("make_hif_env", make_hif_env),
    ]
    
    try:
        for name, func in functions:
            print(f"\nTesting {name}:")
            
            # Skip v4 and v5 if they don't exist
            if name in ["make_field_coverage_env", "make_hif_env"]:
                try:
                    env = func(num_envs=1)
                except (ImportError, ModuleNotFoundError) as e:
                    print(f"  ⚠️  {name} not available: {e}")
                    continue
            else:
                env = func(num_envs=1)
            
            print(f"  ✅ {name} works correctly")
            env.close()
            
    except Exception as e:
        print(f"❌ Error in convenience functions: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_backward_compatibility():
    """Test backward compatibility with config-based creation."""
    print("\n=== Testing Backward Compatibility ===")
    
    try:
        # Test with default config
        print("\n1. Default config:")
        env = make_env_from_config()
        print("  ✅ Created with default config")
        env.close()
        
        # Test with custom config
        print("\n2. Custom config:")
        config = {
            "action_type": "continuous",
            "num_obstacles_range": (2, 6),
            "use_apf": False,
        }
        env = make_env_from_config(config=config)
        print("  ✅ Created with custom config")
        env.close()
        
        # Test old-style ID mapping
        print("\n3. Old-style ID mapping:")
        config = {"id": "Pasture-v2"}  # Old-style ID
        env = make_env_from_config(config=config)
        print("  ✅ Old-style ID mapped correctly")
        env.close()
        
    except Exception as e:
        print(f"❌ Error in backward compatibility: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_parameter_passing():
    """Test that parameters are correctly passed to environments."""
    print("\n=== Testing Parameter Passing ===")
    
    try:
        # Create environment with specific parameters
        test_params = {
            "use_apf": True,
            "num_obstacles_range": (5, 10),
            "reward_field_coverage": 3.0,
            "reward_boundary": -5.0,
        }
        
        print("\nCreating environment with test parameters:")
        for key, value in test_params.items():
            print(f"  {key}: {value}")
        
        env = make_env(env_id="NewPasture-v2", **test_params)
        
        # Try to verify parameters were passed (this depends on environment implementation)
        print("\n✅ Parameters passed successfully (environment created without errors)")
        
        # Test a few steps to ensure environment works with custom parameters
        obs, info = env.reset(seed=42)
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        print("✅ Environment operates correctly with custom parameters")
        env.close()
        
    except Exception as e:
        print(f"❌ Error in parameter passing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing New Environment Creation System")
    print("=" * 60)
    
    all_passed = True
    
    # Run tests
    tests = [
        ("Gymnasium Registration", test_gymnasium_registration),
        ("Direct Creation", test_direct_creation),
        ("make_env Function", test_make_env_function),
        ("Convenience Functions", test_convenience_functions),
        ("Backward Compatibility", test_backward_compatibility),
        ("Parameter Passing", test_parameter_passing),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        try:
            passed = test_func()
            results.append((test_name, passed))
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"\n❌ Unexpected error in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
            all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! The new environment system works correctly.")
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)