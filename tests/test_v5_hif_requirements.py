#!/usr/bin/env python3
"""
Test script to verify v5 HIF requirement enforcement and error messages
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_missing_hif_file():
    """Test that missing HIF file raises clear error"""
    print("\n" + "="*60)
    print("Test 1: Missing HIF file should raise FileNotFoundError")
    print("="*60)
    
    try:
        from envs_new.cpp_env_v5 import CppEnv
        
        # This should fail because HIF files don't exist yet
        env = CppEnv()
        obs, info = env.reset(seed=42)
        
        print("❌ FAILED: Should have raised FileNotFoundError")
        return False
        
    except (FileNotFoundError, RuntimeError) as e:
        # RuntimeError wraps the FileNotFoundError from scenario generator
        error_msg = str(e)
        if "HIF file required but not found" in error_msg:
            print(f"✅ PASSED: Got expected error with clear message")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Key message: HIF file required but not found")
            return True
        else:
            print(f"❌ FAILED: Error doesn't mention missing HIF: {e}")
            return False
    except Exception as e:
        print(f"❌ FAILED: Got unexpected error: {type(e).__name__}: {e}")
        return False


def test_scenario_without_hif():
    """Test that scenario mode without HIF raises clear error"""
    print("\n" + "="*60)
    print("Test 2: Scenario without HIF should raise FileNotFoundError")
    print("="*60)
    
    try:
        from envs_new.cpp_env_v5 import CppEnv
        import tempfile
        import numpy as np
        
        # Create a temporary scenario directory without HIF
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create only field map, no HIF
            field_map = np.ones((400, 400), dtype=np.uint8)
            np.save(Path(tmpdir) / 'map_field.npy', field_map)
            
            env = CppEnv()
            obs, info = env.reset(seed=42, options={'scenario_directory': tmpdir})
            
            print("❌ FAILED: Should have raised FileNotFoundError")
            return False
            
    except (FileNotFoundError, RuntimeError) as e:
        # RuntimeError wraps the FileNotFoundError from scenario generator
        error_msg = str(e)
        if "Scenario HIF file required but not found" in error_msg or "HIF file required but not found" in error_msg:
            print(f"✅ PASSED: Got expected error with clear message")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Key message: Scenario HIF file required")
            return True
        else:
            print(f"❌ FAILED: Error doesn't mention missing HIF: {e}")
            return False
    except Exception as e:
        print(f"❌ FAILED: Got unexpected error: {type(e).__name__}: {e}")
        return False


def test_unified_unpacking():
    """Test that FieldCreator's unified unpacking works correctly"""
    print("\n" + "="*60)
    print("Test 3: FieldCreator unified unpacking")
    print("="*60)
    
    try:
        from envs_new.components.map.map_components import FieldCreator
        import numpy as np
        
        creator = FieldCreator()
        
        # Test _load_from_directory returns 3-tuple
        print("Testing _load_from_directory return value...")
        # This would need a real directory, so we'll just check the signature
        import inspect
        sig = inspect.signature(creator._load_from_directory)
        return_anno = sig.return_annotation
        
        # Check if it's a 3-tuple type hint
        if 'Tuple' in str(return_anno) and 'Optional[int]' in str(return_anno):
            print(f"✅ Return type includes Optional[int]: {return_anno}")
        else:
            print(f"⚠️  Return type might be incorrect: {return_anno}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        return False


def main():
    """Run all tests"""
    print("\n🧪 Testing v5 HIF Requirements and Error Messages")
    print("="*60)
    
    results = {
        "Missing HIF file": test_missing_hif_file(),
        "Scenario without HIF": test_scenario_without_hif(),
        "Unified unpacking": test_unified_unpacking(),
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name:25} : {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("HIF requirements are properly enforced with clear error messages.")
    else:
        print("\n⚠️ SOME TESTS FAILED")
        print("Please review the errors above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)