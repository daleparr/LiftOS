#!/usr/bin/env python3
"""
Simple test script for data quality engine functionality.

This script tests the core data quality functionality in isolation.
"""

import sys
import os
import asyncio
from datetime import datetime
from typing import Dict, Any, List
from enum import Enum
import pandas as pd
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import just the data quality engine directly
from core.data_quality_engine import DataQualityEngine, QualityLevel, QualityDimension


async def test_basic_functionality():
    """Test basic data quality engine functionality."""
    print("🔍 Testing Data Quality Engine - Basic Functionality")
    print("=" * 55)
    
    # Initialize the engine
    engine = DataQualityEngine()
    print("✅ Data Quality Engine initialized successfully")
    
    # Test with high-quality sample data
    print("\n📊 Testing with high-quality sample data...")
    high_quality_data = {
        "customer_id": [1, 2, 3, 4, 5],
        "name": ["Alice Johnson", "Bob Smith", "Carol Davis", "David Wilson", "Eva Brown"],
        "email": ["alice@email.com", "bob@email.com", "carol@email.com", "david@email.com", "eva@email.com"],
        "age": [25, 30, 35, 28, 32],
        "purchase_amount": [100.50, 250.75, 175.25, 320.00, 89.99]
    }
    
    # Evaluate data quality
    report = await engine.evaluate_data_quality(
        data=high_quality_data,
        dataset_id="test_high_quality_001",
        include_profiling=True
    )
    
    print(f"📈 Overall Score: {report.overall_score:.3f}")
    print(f"🏆 Quality Level: {report.overall_level.value}")
    print(f"📋 Dataset ID: {report.dataset_id}")
    
    # Verify all 8 dimensions are evaluated
    expected_dimensions = {
        QualityDimension.COMPLETENESS,
        QualityDimension.ACCURACY,
        QualityDimension.CONSISTENCY,
        QualityDimension.VALIDITY,
        QualityDimension.UNIQUENESS,
        QualityDimension.TIMELINESS,
        QualityDimension.RELEVANCE,
        QualityDimension.INTEGRITY
    }
    
    actual_dimensions = set(report.dimension_scores.keys())
    
    if expected_dimensions == actual_dimensions:
        print("✅ All 8 quality dimensions evaluated")
    else:
        print(f"❌ Missing dimensions: {expected_dimensions - actual_dimensions}")
        return False
    
    # Display dimension scores
    print("\n📊 Dimension Scores:")
    for dimension, metric in report.dimension_scores.items():
        print(f"  {dimension.value}: {metric.score:.3f} ({metric.level.value})")
    
    # Verify high quality data gets good scores
    if report.overall_score >= 0.85:
        print("✅ High quality data correctly identified")
    else:
        print(f"❌ Expected high score (>=0.85), got {report.overall_score:.3f}")
        return False
    
    return True


async def test_poor_quality_detection():
    """Test detection of poor quality data."""
    print("\n🔍 Testing Poor Quality Data Detection")
    print("=" * 40)
    
    engine = DataQualityEngine()
    
    # Test with poor-quality data
    poor_quality_data = {
        "customer_id": [1, 2, None, 4, 2],  # Missing value and duplicate
        "name": ["Alice", "", "Carol Davis", "David Wilson", None],  # Empty and missing values
        "email": ["alice@email", "invalid-email", "carol@email.com", "david@email.com", "eva@email.com"],  # Invalid formats
        "age": [25, -5, 150, 28, "thirty"],  # Invalid values
        "purchase_amount": [100.50, None, 175.25, -50.00, 89.99]  # Missing and negative values
    }
    
    report = await engine.evaluate_data_quality(
        data=poor_quality_data,
        dataset_id="test_poor_quality_001",
        include_profiling=True
    )
    
    print(f"📈 Overall Score: {report.overall_score:.3f}")
    print(f"🏆 Quality Level: {report.overall_level.value}")
    
    # Verify poor quality is detected
    if report.overall_score < 0.70:
        print("✅ Poor quality data correctly identified")
    else:
        print(f"❌ Expected low score (<0.70), got {report.overall_score:.3f}")
        return False
    
    # Check for critical issues
    if len(report.critical_issues) > 0:
        print(f"✅ Critical issues detected: {len(report.critical_issues)}")
        for issue in report.critical_issues[:3]:
            print(f"  - {issue}")
    else:
        print("❌ Expected critical issues but none found")
        return False
    
    return True


async def test_agent_testing_validation():
    """Test agent testing validation functionality."""
    print("\n🤖 Testing Agent Testing Validation")
    print("=" * 35)
    
    engine = DataQualityEngine()
    
    # Test with high quality data
    high_quality_data = {
        "campaign_id": [1, 2, 3, 4, 5],
        "target_audience": ["young_adults", "professionals", "seniors", "students", "families"],
        "budget": [1000, 2500, 1500, 800, 2000],
        "conversion_rate": [0.05, 0.08, 0.03, 0.06, 0.07]
    }
    
    is_valid, validation_report = await engine.validate_data_for_agent_testing(
        data=high_quality_data,
        test_type="marketing_campaign"
    )
    
    print(f"✅ High Quality Validation: {'PASS' if is_valid else 'FAIL'}")
    print(f"📈 Validation Score: {validation_report.overall_score:.3f}")
    
    if not is_valid:
        print("❌ High quality data should pass validation")
        return False
    
    # Test with poor quality data
    poor_quality_data = {
        "campaign_id": [1, 2, None, 4, 2],  # Missing and duplicate
        "target_audience": ["young_adults", "", None, "students", "families"],  # Missing values
        "budget": [1000, -500, 1500, None, 2000],  # Negative and missing
        "conversion_rate": [0.05, 1.5, 0.03, 0.06, -0.1]  # Invalid rates
    }
    
    is_valid_poor, _ = await engine.validate_data_for_agent_testing(
        data=poor_quality_data,
        test_type="marketing_campaign"
    )
    
    print(f"❌ Poor Quality Validation: {'PASS' if is_valid_poor else 'FAIL'}")
    
    if is_valid_poor:
        print("❌ Poor quality data should fail validation")
        return False
    
    print("✅ Agent testing validation working correctly")
    return True


async def test_performance():
    """Test performance with larger dataset."""
    print("\n⚡ Testing Performance")
    print("=" * 20)
    
    engine = DataQualityEngine()
    
    # Generate larger dataset
    size = 1000
    large_data = {
        "id": list(range(size)),
        "value": np.random.normal(100, 15, size).tolist(),
        "category": np.random.choice(["A", "B", "C", "D"], size).tolist(),
        "score": np.random.uniform(0, 1, size).tolist()
    }
    
    start_time = datetime.now()
    report = await engine.evaluate_data_quality(
        data=large_data,
        dataset_id="performance_test_001",
        include_profiling=True
    )
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds()
    print(f"⏱️  Evaluation Time: {duration:.2f} seconds")
    print(f"📊 Dataset Size: {size} rows")
    print(f"📈 Quality Score: {report.overall_score:.3f}")
    
    if duration < 10.0:
        print("✅ Performance acceptable (< 10 seconds)")
        return True
    else:
        print("⚠️  Performance slower than expected")
        return False


async def main():
    """Main test function."""
    print("🚀 LiftOS Agentic Data Quality Engine Tests")
    print("=" * 45)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Poor Quality Detection", test_poor_quality_detection),
        ("Agent Testing Validation", test_agent_testing_validation),
        ("Performance", test_performance)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running: {test_name}")
            result = await test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 45)
    print("🎯 TEST SUMMARY")
    print("=" * 45)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🏆 ALL TESTS PASSED!")
        print("Data Quality Engine is ready for production use.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)