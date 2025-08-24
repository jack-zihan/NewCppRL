#!/usr/bin/env python3
"""
Simple test for Gymnasium registration without full environment import.
Tests that environments are properly registered.
"""
import gymnasium as gym
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_registration_only():
    """Test that NewPasture environments are registered."""
    print("=" * 60)
    print("Testing Gymnasium Registration")
    print("=" * 60)
    
    # Import just the registration part
    import envs_new  # This should trigger registration
    
    # Check registered environments
    registered_envs = list(gym.envs.registry.keys())
    new_envs = [f"NewPasture-v{i}" for i in range(1, 6)]
    
    print("\nChecking environment registration:")
    all_registered = True
    for env_id in new_envs:
        if env_id in registered_envs:
            print(f"  ✅ {env_id} is registered")
        else:
            print(f"  ❌ {env_id} is NOT registered")
            all_registered = False
    
    if all_registered:
        print("\n🎉 All environments successfully registered with Gymnasium!")
    else:
        print("\n⚠️  Some environments are not registered")
    
    return all_registered


def test_utils_import():
    """Test that new utils can be imported."""
    print("\n" + "=" * 60)
    print("Testing Utils Import")
    print("=" * 60)
    
    try:
        # Try importing the utils module
        from torchrl_utils_new import utils_env
        print("✅ torchrl_utils_new.utils_env imported successfully")
        
        # Check for key functions
        functions = [
            'make_env',
            'make_env_lambda',
            'make_base_env',
            'make_apf_env',
            'make_multiscale_env',
            'make_field_coverage_env',
            'make_hif_env',
            'make_env_from_config'
        ]
        
        print("\nChecking exported functions:")
        for func_name in functions:
            if hasattr(utils_env, func_name):
                print(f"  ✅ {func_name} is available")
            else:
                print(f"  ❌ {func_name} is missing")
        
        return True
    except ImportError as e:
        print(f"❌ Failed to import utils: {e}")
        return False


def test_yaml_independence():
    """Verify that new utils don't depend on YAML config."""
    print("\n" + "=" * 60)
    print("Testing YAML Independence")
    print("=" * 60)
    
    # Read the utils file and check for YAML imports
    utils_path = Path(__file__).parent.parent / "torchrl_utils_new" / "utils_env.py"
    
    with open(utils_path, 'r') as f:
        content = f.read()
    
    yaml_indicators = ['import yaml', 'from yaml', 'env_config.yaml', 'yaml.load']
    found_yaml = []
    
    for indicator in yaml_indicators:
        if indicator in content:
            found_yaml.append(indicator)
    
    if found_yaml:
        print(f"❌ Found YAML dependencies: {found_yaml}")
        return False
    else:
        print("✅ No YAML dependencies found in torchrl_utils_new/utils_env.py")
        return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TESTING NEW ENVIRONMENT SYSTEM")
    print("=" * 60)
    
    results = []
    
    # Test YAML independence first (doesn't require imports)
    print("\n1. YAML Independence Test:")
    yaml_test = test_yaml_independence()
    results.append(("YAML Independence", yaml_test))
    
    # Test registration
    print("\n2. Registration Test:")
    try:
        reg_test = test_registration_only()
        results.append(("Gymnasium Registration", reg_test))
    except Exception as e:
        print(f"❌ Registration test failed: {e}")
        results.append(("Gymnasium Registration", False))
    
    # Test utils import
    print("\n3. Utils Import Test:")
    try:
        utils_test = test_utils_import()
        results.append(("Utils Import", utils_test))
    except Exception as e:
        print(f"❌ Utils import test failed: {e}")
        results.append(("Utils Import", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 Core functionality tests PASSED!")
        print("\nNote: Full environment creation tests require proper CUDA/CuPy setup.")
        print("The registration and YAML-free configuration are working correctly.")
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)