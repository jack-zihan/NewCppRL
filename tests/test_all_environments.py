#!/usr/bin/env python3
"""
Comprehensive test for all environment versions (v1-v5)
Ensures all environments work correctly after refactoring
"""

import sys
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_env_v1():
    """Test v1 environment (base environment with weed)"""
    print("\n" + "="*60)
    print("Testing v1 Environment (Base with Weed)")
    print("="*60)
    
    try:
        from envs_new.cpp_env_base import CppEnvBase
        
        # v1 expects weed_coverage maps
        env = CppEnvBase(map_dir="envs_new/maps/weed_coverage/1-400")
        obs, info = env.reset(seed=42)
        
        # Run a few steps
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        env.close()
        print("✅ v1 Environment test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ v1 Environment test FAILED: {e}")
        traceback.print_exc()
        return False


def test_env_v2():
    """Test v2 environment (improved base)"""
    print("\n" + "="*60)
    print("Testing v2 Environment (Improved Base)")
    print("="*60)
    
    try:
        from envs_new.cpp_env_v2 import CppEnv
        
        # v2 uses weed_coverage maps
        env = CppEnv(map_dir="envs_new/maps/weed_coverage/1-400")
        obs, info = env.reset(seed=42)
        
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        env.close()
        print("✅ v2 Environment test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ v2 Environment test FAILED: {e}")
        traceback.print_exc()
        return False


def test_env_v3():
    """Test v3 environment (with multiscale features)"""
    print("\n" + "="*60)
    print("Testing v3 Environment (Multiscale Features)")
    print("="*60)
    
    try:
        from envs_new.cpp_env_v3 import CppEnv
        
        # v3 uses weed_coverage maps
        env = CppEnv(
            map_dir="envs_new/maps/weed_coverage/1-400",
            use_multiscale=True,
            use_global_features=True
        )
        obs, info = env.reset(seed=42)
        
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        env.close()
        print("✅ v3 Environment test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ v3 Environment test FAILED: {e}")
        traceback.print_exc()
        return False


def test_env_v4():
    """Test v4 environment (field coverage without weed)"""
    print("\n" + "="*60)
    print("Testing v4 Environment (Field Coverage)")
    print("="*60)
    
    try:
        from envs_new.cpp_env_v4 import CppEnv
        
        # v4 uses field_coverage maps (default)
        env = CppEnv()  # Uses default map_dir="envs_new/maps/field_coverage"
        obs, info = env.reset(seed=42)
        
        print(f"  Observation shape: {obs['observation'].shape}")
        print(f"  Initial field coverage: {obs['completion_ratio'][0]:.2%}")
        
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        env.close()
        print("✅ v4 Environment test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ v4 Environment test FAILED: {e}")
        traceback.print_exc()
        return False


def test_env_v5():
    """Test v5 environment (field coverage with HIF)"""
    print("\n" + "="*60)
    print("Testing v5 Environment (Field Coverage + HIF)")
    print("="*60)
    
    try:
        from envs_new.cpp_env_v5 import CppEnv
        
        # v5 uses field_coverage maps with HIF support (default)
        env = CppEnv(
            reward_hif=0.01,
            use_multiscale=True,
            use_global_features=True
        )
        obs, info = env.reset(seed=42)
        
        print(f"  Observation shape: {obs['observation'].shape}")
        print(f"  Initial field coverage: {obs['completion_ratio'][0]:.2%}")
        
        # Check if HIF is loaded
        if 'hif' in env.maps_dict:
            print("  ✓ HIF map loaded successfully")
        else:
            print("  ℹ HIF map not available (optional)")
        
        # Test reward breakdown
        reward_breakdown = env.reward_system.get_reward_breakdown(
            env.env_state, map_dict=env.maps_dict
        )
        if 'hif' in reward_breakdown['breakdown']:
            print("  ✓ HIF reward calculator active")
        
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        env.close()
        print("✅ v5 Environment test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ v5 Environment test FAILED: {e}")
        traceback.print_exc()
        return False


def test_scenario_mode():
    """Test scenario mode loading"""
    print("\n" + "="*60)
    print("Testing Scenario Mode")
    print("="*60)
    
    # Check if real scenario directory exists
    scenario_dir = Path("envs_new/maps/weed_coverage/real")
    if not scenario_dir.exists():
        print("ℹ Skipping scenario test - no scenario directory found")
        return True
    
    try:
        from envs_new.cpp_env_v2 import CppEnv
        
        env = CppEnv()
        obs, info = env.reset(
            seed=42,
            options={'scenario_directory': str(scenario_dir)}
        )
        
        print("  ✓ Scenario loaded successfully")
        
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        env.close()
        print("✅ Scenario mode test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Scenario mode test FAILED: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all environment tests"""
    print("\n" + "🧪"*30)
    print("COMPREHENSIVE ENVIRONMENT TEST SUITE")
    print("Testing all environments v1-v5 after refactoring")
    print("🧪"*30)
    
    results = {
        "v1": test_env_v1(),
        "v2": test_env_v2(),
        "v3": test_env_v3(),
        "v4": test_env_v4(),
        "v5": test_env_v5(),
        "scenario": test_scenario_mode(),
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name:10} : {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("All environments (v1-v5) are working correctly after refactoring.")
    else:
        print("\n⚠️ SOME TESTS FAILED")
        print("Please review the errors above and fix the issues.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)