#!/usr/bin/env python3
"""
LiftOS Causal AI Integration Test Suite
Comprehensive tests for the Causal AI module integration with LiftOS.
"""

import requests
import json
import time
import sys
from datetime import datetime, timedelta
import random

# Configuration
CAUSAL_SERVICE_URL = "http://localhost:3003"
CAUSAL_MODULE_URL = "http://localhost:8008"
AUTH_SERVICE_URL = "http://localhost:8001"
REGISTRY_URL = "http://localhost:8005"

# Test data
SAMPLE_CAMPAIGN_DATA = {
    "campaigns": [
        {
            "id": "camp_001",
            "name": "Summer Sale 2024",
            "platform": "google_ads",
            "spend": 15000,
            "impressions": 250000,
            "clicks": 12500,
            "conversions": 875,
            "revenue": 52500,
            "start_date": "2024-06-01",
            "end_date": "2024-08-31"
        },
        {
            "id": "camp_002",
            "name": "Meta Brand Awareness",
            "platform": "meta",
            "spend": 8000,
            "impressions": 180000,
            "clicks": 7200,
            "conversions": 432,
            "revenue": 21600,
            "start_date": "2024-06-15",
            "end_date": "2024-09-15"
        }
    ],
    "external_factors": [
        {"date": "2024-07-04", "factor": "holiday", "impact": 0.15},
        {"date": "2024-08-15", "factor": "competitor_launch", "impact": -0.08}
    ]
}

SAMPLE_MMM_DATA = {
    "date_range": {
        "start": "2024-01-01",
        "end": "2024-08-31"
    },
    "channels": ["google_ads", "meta", "tiktok", "email", "organic"],
    "spend_data": [
        {"date": "2024-01-01", "google_ads": 5000, "meta": 3000, "tiktok": 1000, "email": 500, "organic": 0},
        {"date": "2024-02-01", "google_ads": 5500, "meta": 3200, "tiktok": 1200, "email": 500, "organic": 0},
        {"date": "2024-03-01", "google_ads": 6000, "meta": 3500, "tiktok": 1500, "email": 600, "organic": 0}
    ],
    "revenue_data": [
        {"date": "2024-01-01", "revenue": 45000},
        {"date": "2024-02-01", "revenue": 48000},
        {"date": "2024-03-01", "revenue": 52000}
    ]
}

class CausalTestSuite:
    def __init__(self):
        self.auth_token = None
        self.test_results = []
        
    def log_test(self, test_name, success, message="", response_data=None):
        """Log test result"""
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
        if message:
            print(f"    {message}")
        if response_data and not success:
            print(f"    Response: {response_data}")
        
        self.test_results.append({
            "test": test_name,
            "success": success,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        print()

    def get_auth_token(self):
        """Get authentication token for testing"""
        try:
            # For testing, we'll use a mock token or skip auth
            # In production, this would authenticate with the auth service
            self.auth_token = "test_token_123"
            return True
        except Exception as e:
            self.log_test("Authentication", False, f"Failed to get auth token: {e}")
            return False

    def test_service_health(self):
        """Test health endpoints"""
        # Test Causal AI service health
        try:
            response = requests.get(f"{CAUSAL_SERVICE_URL}/health", timeout=5)
            success = response.status_code == 200
            self.log_test("Causal Service Health", success, 
                         f"Status: {response.status_code}" if not success else "Service is healthy")
        except Exception as e:
            self.log_test("Causal Service Health", False, f"Connection failed: {e}")

        # Test Causal AI module health
        try:
            response = requests.get(f"{CAUSAL_MODULE_URL}/health", timeout=5)
            success = response.status_code == 200
            self.log_test("Causal Module Health", success,
                         f"Status: {response.status_code}" if not success else "Module is healthy")
        except Exception as e:
            self.log_test("Causal Module Health", False, f"Connection failed: {e}")

    def test_module_registration(self):
        """Test module registration with registry"""
        try:
            response = requests.get(f"{REGISTRY_URL}/modules/causal", timeout=5)
            if response.status_code == 200:
                module_data = response.json()
                success = module_data.get("id") == "causal"
                self.log_test("Module Registration", success,
                             f"Module registered with status: {module_data.get('status', 'unknown')}")
            else:
                self.log_test("Module Registration", False, f"Module not found: {response.status_code}")
        except Exception as e:
            self.log_test("Module Registration", False, f"Registry connection failed: {e}")

    def test_attribution_analysis(self):
        """Test attribution analysis endpoint"""
        try:
            headers = {"Content-Type": "application/json"}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            
            response = requests.post(
                f"{CAUSAL_MODULE_URL}/attribution/analyze",
                json=SAMPLE_CAMPAIGN_DATA,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                has_attribution = "attribution_scores" in result
                has_insights = "insights" in result
                success = has_attribution and has_insights
                
                self.log_test("Attribution Analysis", success,
                             f"Returned attribution scores and insights" if success 
                             else f"Missing expected fields in response")
            else:
                self.log_test("Attribution Analysis", False, 
                             f"Request failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            self.log_test("Attribution Analysis", False, f"Request error: {e}")

    def test_mmm_analysis(self):
        """Test Media Mix Modeling endpoint"""
        try:
            headers = {"Content-Type": "application/json"}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            
            response = requests.post(
                f"{CAUSAL_MODULE_URL}/mmm/analyze",
                json=SAMPLE_MMM_DATA,
                headers=headers,
                timeout=45
            )
            
            if response.status_code == 200:
                result = response.json()
                has_contributions = "channel_contributions" in result
                has_saturation = "saturation_curves" in result
                has_recommendations = "recommendations" in result
                success = has_contributions and has_saturation and has_recommendations
                
                self.log_test("MMM Analysis", success,
                             f"Returned complete MMM analysis" if success 
                             else f"Missing expected fields in response")
            else:
                self.log_test("MMM Analysis", False,
                             f"Request failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            self.log_test("MMM Analysis", False, f"Request error: {e}")

    def test_lift_measurement(self):
        """Test lift measurement endpoint"""
        try:
            headers = {"Content-Type": "application/json"}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            
            lift_data = {
                "test_group": {
                    "size": 10000,
                    "conversions": 850,
                    "revenue": 42500
                },
                "control_group": {
                    "size": 10000,
                    "conversions": 720,
                    "revenue": 36000
                },
                "test_period": {
                    "start": "2024-07-01",
                    "end": "2024-07-31"
                }
            }
            
            response = requests.post(
                f"{CAUSAL_MODULE_URL}/lift/measure",
                json=lift_data,
                headers=headers,
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                has_lift = "lift_percentage" in result
                has_significance = "statistical_significance" in result
                success = has_lift and has_significance
                
                self.log_test("Lift Measurement", success,
                             f"Calculated lift and significance" if success 
                             else f"Missing expected fields in response")
            else:
                self.log_test("Lift Measurement", False,
                             f"Request failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            self.log_test("Lift Measurement", False, f"Request error: {e}")

    def test_budget_optimization(self):
        """Test budget optimization endpoint"""
        try:
            headers = {"Content-Type": "application/json"}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            
            budget_data = {
                "total_budget": 50000,
                "channels": ["google_ads", "meta", "tiktok", "email"],
                "constraints": {
                    "min_spend_per_channel": 1000,
                    "max_spend_per_channel": 20000
                },
                "objective": "maximize_revenue",
                "historical_performance": SAMPLE_MMM_DATA["spend_data"]
            }
            
            response = requests.post(
                f"{CAUSAL_MODULE_URL}/budget/optimize",
                json=budget_data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                has_allocation = "optimal_allocation" in result
                has_expected_return = "expected_return" in result
                success = has_allocation and has_expected_return
                
                self.log_test("Budget Optimization", success,
                             f"Generated optimal budget allocation" if success 
                             else f"Missing expected fields in response")
            else:
                self.log_test("Budget Optimization", False,
                             f"Request failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            self.log_test("Budget Optimization", False, f"Request error: {e}")

    def test_memory_integration(self):
        """Test memory service integration"""
        try:
            headers = {"Content-Type": "application/json"}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            
            # Test storing analysis results
            memory_data = {
                "analysis_type": "attribution",
                "results": {"test": "data"},
                "metadata": {"timestamp": datetime.now().isoformat()}
            }
            
            response = requests.post(
                f"{CAUSAL_MODULE_URL}/memory/store",
                json=memory_data,
                headers=headers,
                timeout=10
            )
            
            success = response.status_code in [200, 201]
            self.log_test("Memory Integration", success,
                         f"Stored analysis results in memory" if success 
                         else f"Failed to store: {response.status_code}")
                
        except Exception as e:
            self.log_test("Memory Integration", False, f"Memory integration error: {e}")

    def run_all_tests(self):
        """Run complete test suite"""
        print("========================================")
        print("LiftOS Causal AI Integration Test Suite")
        print("========================================")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Get authentication token
        if not self.get_auth_token():
            print("Skipping authenticated tests due to auth failure")
        
        # Run tests
        self.test_service_health()
        self.test_module_registration()
        self.test_attribution_analysis()
        self.test_mmm_analysis()
        self.test_lift_measurement()
        self.test_budget_optimization()
        self.test_memory_integration()
        
        # Summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        
        print("========================================")
        print("Test Results Summary")
        print("========================================")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print()
        
        if failed_tests > 0:
            print("Failed Tests:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"  - {result['test']}: {result['message']}")
            print()
        
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return failed_tests == 0

def main():
    """Main test execution"""
    test_suite = CausalTestSuite()
    success = test_suite.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()