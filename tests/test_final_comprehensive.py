"""
Final comprehensive test for all three environment versions (v1, v2, v3).
Verifies that the new modular architecture works correctly for all variants.
"""
import sys
import os
sys.path.append('/home/lzh/NewCppRL')

def test_environment_version(env_version: str):
    """Test a specific environment version."""
    print(f"\n{'='*60}")
    print(f"TESTING CppEnv {env_version.upper()}")
    print(f"{'='*60}")
    
    try:
        # Import the specific environment
        if env_version == 'v1':
            from envs_new.cpp_env_v1 import CppEnv
        elif env_version == 'v2':
            from envs_new.cpp_env_v2 import CppEnv
        elif env_version == 'v3':
            from envs_new.cpp_env_v3 import CppEnv
        else:
            raise ValueError(f"Unknown environment version: {env_version}")
        
        print(f"✓ {env_version} import successful")
        
        # Create environment
        env = CppEnv(render_mode=None)
        print(f"✓ {env_version} creation successful")
        
        # Test reset
        obs, info = env.reset(seed=42)
        print(f"✓ {env_version} reset successful")
        print(f"  Observation keys: {list(obs.keys())}")
        
        for key, value in obs.items():
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        
        # Test multiple steps
        total_reward = 0
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if step == 0:
                print(f"  First step - Reward: {reward:.3f}")
                print(f"  Info keys: {list(info.keys())}")
        
        print(f"✓ {env_version} stepping successful")
        print(f"  Total reward over 5 steps: {total_reward:.3f}")
        
        # Test action space
        print(f"  Action space: {env.action_space}")
        print(f"  Action space type: {type(env.action_space)}")
        
        # Test observation space
        print(f"  Observation space: {env.observation_space}")
        
        env.close()
        print(f"✓ {env_version} all tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ {env_version} test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_base_environment():
    """Test the base CppEnvBase environment."""
    print(f"\n{'='*60}")
    print(f"TESTING BASE CppEnvBase")
    print(f"{'='*60}")
    
    try:
        from envs_new.cpp_env_base import CppEnvBase
        print("✓ CppEnvBase import successful")
        
        env = CppEnvBase()
        print("✓ CppEnvBase creation successful")
        
        obs, info = env.reset(seed=42)
        print("✓ CppEnvBase reset successful")
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print("✓ CppEnvBase step successful")
        
        env.close()
        print("✓ CppEnvBase all tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ CppEnvBase test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_component_compatibility():
    """Test compatibility with original component interfaces."""
    print(f"\n{'='*60}")
    print(f"TESTING COMPONENT COMPATIBILITY")
    print(f"{'='*60}")
    
    try:
        # Test configuration system
        from envs_new.components.config.environment_config import EnvironmentConfig
        config = EnvironmentConfig()
        print("✓ Configuration system working")
        
        # Test map system
        from envs_new.components.map.map_generator import MapGenerator
        map_gen = MapGenerator(config.map_config, config.agent_config)
        print("✓ Map generation system working")
        
        # Test observation system
        from envs_new.components.observation.observation_strategy import ObservationManager
        obs_manager = ObservationManager(config.observation_config)
        print("✓ Observation system working")
        
        # Test reward system
        from envs_new.components.reward.reward_system import RewardManager
        reward_manager = RewardManager(config.reward_config, config.action_config, config.agent_config)
        print("✓ Reward system working")
        
        print("✓ All component systems compatible!")
        return True
        
    except Exception as e:
        print(f"✗ Component compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all comprehensive tests."""
    print("FINAL COMPREHENSIVE TEST SUITE")
    print("Testing new modular CppEnv architecture")
    
    # Test results
    results = {}
    
    # Test base environment
    results['base'] = test_base_environment()
    
    # Test component compatibility
    results['components'] = test_component_compatibility()
    
    # Test all environment versions
    for version in ['v1', 'v2', 'v3']:
        results[version] = test_environment_version(version)
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name.upper():<15}: {status}")
        if result:
            passed += 1
    
    print(f"\nOVERALL: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! New modular architecture is working correctly!")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)