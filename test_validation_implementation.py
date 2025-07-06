#!/usr/bin/env python3
"""
Quick test to verify empirical validation implementation
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all required modules can be imported."""
    
    print("Testing imports...")
    
    try:
        # Test existing validation imports
        from tests.test_empirical_validation import (
            CausalDataSimulator, ConfounderDetector, CausalDataTransformer
        )
        print("✓ Existing validation modules imported successfully")
    except ImportError as e:
        print(f"X Failed to import existing validation modules: {e}")
        return False
    
    try:
        # Test new performance claims imports
        from tests.test_liftos_performance_claims import (
            TestLiftOSExecutionTimeClaims,
            TestLiftOSSpeedupClaims,
            TestLiftOSAccuracyClaims,
            TestObservabilityOverhead,
            PerformanceClaimResult
        )
        print("✓ Performance claims test modules imported successfully")
    except ImportError as e:
        print(f"X Failed to import performance claims modules: {e}")
        return False
    
    try:
        # Test runner imports
        from run_empirical_validation import EmpiricalValidationRunner
        print("✓ Validation runner imported successfully")
    except ImportError as e:
        print(f"X Failed to import validation runner: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of validation components."""
    
    print("\nTesting basic functionality...")
    
    try:
        # Test data simulator
        from tests.test_empirical_validation import CausalDataSimulator
        simulator = CausalDataSimulator()
        data, ground_truth = simulator.generate_marketing_data_with_confounders(n_samples=10)
        
        if len(data) == 10:
            print("✓ Data simulator working correctly")
        else:
            print(f"X Data simulator returned {len(data)} samples instead of 10")
            return False
            
    except Exception as e:
        print(f"X Data simulator failed: {e}")
        return False
    
    try:
        # Test performance claim result
        from tests.test_liftos_performance_claims import PerformanceClaimResult
        result = PerformanceClaimResult(
            claim_name="test",
            target_value=1.0,
            measured_value=0.9,
            meets_target=False,
            confidence_interval=(0.8, 1.0),
            test_conditions={}
        )
        
        if result.claim_name == "test":
            print("✓ PerformanceClaimResult working correctly")
        else:
            print("X PerformanceClaimResult not working correctly")
            return False
            
    except Exception as e:
        print(f"X PerformanceClaimResult failed: {e}")
        return False
    
    return True

def test_config_loading():
    """Test configuration loading."""
    
    print("\nTesting configuration loading...")
    
    try:
        import json
        with open('empirical_validation_config.json', 'r') as f:
            config = json.load(f)
        
        # Check required config sections
        required_sections = ['validation_config', 'infrastructure_config']
        for section in required_sections:
            if section not in config:
                print(f"X Missing config section: {section}")
                return False
        
        print("✓ Configuration loaded successfully")
        return True
        
    except Exception as e:
        print(f"X Configuration loading failed: {e}")
        return False

def main():
    """Run all tests."""
    
    print("="*60)
    print("EMPIRICAL VALIDATION IMPLEMENTATION TEST")
    print("="*60)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_config_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            print(f"Test {test.__name__} failed!")
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Empirical validation implementation is ready.")
        return True
    else:
        print("X Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)