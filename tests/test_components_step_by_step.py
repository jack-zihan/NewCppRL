"""
Step-by-step component testing to isolate segmentation fault.
"""
import sys
import os
sys.path.append('/home/lzh/NewCppRL')

def test_basic_imports():
    """Test basic imports and modules."""
    print("Testing basic imports...")
    try:
        import numpy as np
        import cv2
        import gymnasium as gym
        print("✓ Basic imports successful")
        return True
    except Exception as e:
        print(f"✗ Basic imports failed: {e}")
        return False

def test_config_layer():
    """Test configuration layer."""
    print("Testing configuration layer...")
    try:
        from envs_new.components.config.environment_config import EnvironmentConfig
        config = EnvironmentConfig()
        print("✓ Configuration layer successful")
        return True
    except Exception as e:
        print(f"✗ Configuration layer failed: {e}")
        return False

def test_map_loader():
    """Test map loading functionality."""
    print("Testing map loader...")
    try:
        from envs_new.components.map.map_loader import MapLoader
        from envs_new.components.config.environment_config import EnvironmentConfig
        
        config = EnvironmentConfig()
        loader = MapLoader(config.map_config)
        print("✓ Map loader creation successful")
        
        # Try to load a map
        frontier_map, dimensions = loader.load_frontier_map(0)
        print(f"✓ Map loading successful, shape: {frontier_map.shape}")
        return True
    except Exception as e:
        print(f"✗ Map loader failed: {e}")
        return False

def test_agent_creation():
    """Test agent creation."""
    print("Testing agent creation...")
    try:
        from envs_new.components.entity.agent import MowerAgent
        from envs_new.components.config.environment_config import EnvironmentConfig
        
        config = EnvironmentConfig()
        agent = MowerAgent(config.agent_config)
        print("✓ Agent creation successful")
        return True
    except Exception as e:
        print(f"✗ Agent creation failed: {e}")
        return False

def test_observation_strategy():
    """Test observation strategy."""
    print("Testing observation strategy...")
    try:
        from envs_new.components.observation.observation_strategy import FirstPersonObservation
        from envs_new.components.config.environment_config import EnvironmentConfig
        import numpy as np
        
        config = EnvironmentConfig()
        obs_strategy = FirstPersonObservation(config.observation_config)
        
        # Create dummy maps
        map_shape = (400, 400)
        dummy_maps = {
            'map_obstacle': np.zeros(map_shape, dtype=np.uint8),
            'map_frontier': np.ones(map_shape, dtype=np.uint8) * 255,
            'map_weed': np.zeros(map_shape, dtype=np.uint8),
            'map_trajectory': np.zeros(map_shape, dtype=np.uint8)
        }
        
        # Create dummy agent position
        agent_pos = (200.0, 200.0, 0.0)  # x, y, direction
        
        print("✓ Observation strategy creation successful")
        return True
    except Exception as e:
        print(f"✗ Observation strategy failed: {e}")
        return False

def main():
    """Run step-by-step tests."""
    print("=" * 60)
    print("STEP-BY-STEP COMPONENT TESTING")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_config_layer,
        test_map_loader,
        test_agent_creation,
        test_observation_strategy,
    ]
    
    for i, test in enumerate(tests, 1):
        print(f"\n{i}. {test.__doc__.strip()}")
        if not test():
            print(f"STOPPED AT TEST {i}")
            return False
        print()
    
    print("✓ All step-by-step tests passed!")
    return True

if __name__ == "__main__":
    main()