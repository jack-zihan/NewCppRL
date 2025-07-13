"""
Quick test to verify envs_new environments can be directly executed.
"""
import sys
import os
sys.path.append('/home/lzh/NewCppRL')

def test_direct_execution():
    """Test environments can be directly imported and run."""
    print("Testing direct execution of environments...")
    
    # Test v1
    from envs_new.cpp_env_v1 import CppEnv as V1
    env1 = V1()
    obs1, _ = env1.reset()
    print("✓ v1 direct execution successful")
    
    # Test v2
    from envs_new.cpp_env_v2 import CppEnv as V2  
    env2 = V2()
    obs2, _ = env2.reset()
    print("✓ v2 direct execution successful")
    
    # Test v3
    from envs_new.cpp_env_v3 import CppEnv as V3
    env3 = V3()
    obs3, _ = env3.reset()
    print("✓ v3 direct execution successful")
    
    # Test factory
    from envs_new import make_env, EnvironmentFactory
    env_factory = make_env('v2')
    env_class = EnvironmentFactory.create('apf')
    print("✓ Factory methods working")
    
    print("\n🎉 All direct execution tests passed!")
    return True

if __name__ == "__main__":
    test_direct_execution()