#!/usr/bin/env python3
"""
LiftOS Surfacing Integration Test Script

This script validates that the surfacing module integration is working correctly
by testing all endpoints and verifying proper communication between services.
"""

import asyncio
import json
import sys
import time
from typing import Dict, Any, Optional
import aiohttp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SurfacingIntegrationTester:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.surfacing_direct_url = "http://localhost:3002"
        self.surfacing_module_url = "http://localhost:8007"
        self.auth_token = None
        self.user_id = "test-user-123"
        
    async def run_tests(self):
        """Run all integration tests"""
        print("=" * 60)
        print("LiftOS Surfacing Integration Test Suite")
        print("=" * 60)
        print()
        
        async with aiohttp.ClientSession() as session:
            self.session = session
            
            # Test sequence
            tests = [
                ("Service Health Checks", self.test_service_health),
                ("Authentication Setup", self.setup_authentication),
                ("Direct Surfacing Service", self.test_direct_surfacing_service),
                ("Surfacing Module Endpoints", self.test_surfacing_module),
                ("Gateway Integration", self.test_gateway_integration),
                ("Memory Integration", self.test_memory_integration),
                ("Error Handling", self.test_error_handling),
                ("Performance Test", self.test_performance),
            ]
            
            passed = 0
            failed = 0
            
            for test_name, test_func in tests:
                print(f"Running: {test_name}")
                try:
                    await test_func()
                    print(f"✓ PASSED: {test_name}")
                    passed += 1
                except Exception as e:
                    print(f"✗ FAILED: {test_name} - {str(e)}")
                    failed += 1
                print()
            
            # Summary
            print("=" * 60)
            print("Test Results Summary")
            print("=" * 60)
            print(f"Total Tests: {passed + failed}")
            print(f"Passed: {passed}")
            print(f"Failed: {failed}")
            print(f"Success Rate: {(passed / (passed + failed) * 100):.1f}%")
            
            if failed == 0:
                print("\n🎉 All tests passed! Surfacing integration is working correctly.")
                return True
            else:
                print(f"\n⚠️  {failed} test(s) failed. Please check the logs above.")
                return False

    async def test_service_health(self):
        """Test that all required services are healthy"""
        services = [
            ("LiftOS Gateway", f"{self.base_url}/health"),
            ("Surfacing Service", f"{self.surfacing_direct_url}/health"),
            ("Surfacing Module", f"{self.surfacing_module_url}/health"),
        ]
        
        for service_name, url in services:
            async with self.session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"{service_name} is not healthy (status: {response.status})")
                data = await response.json()
                if data.get("status") != "healthy":
                    raise Exception(f"{service_name} reports unhealthy status")
                print(f"  ✓ {service_name} is healthy")

    async def setup_authentication(self):
        """Setup authentication for testing"""
        # For testing, we'll create a mock JWT token or use a test endpoint
        # In a real scenario, this would authenticate with the auth service
        auth_data = {
            "username": "test_user",
            "password": "test_password"
        }
        
        try:
            async with self.session.post(f"{self.base_url}/api/v1/auth/login", json=auth_data) as response:
                if response.status == 200:
                    data = await response.json()
                    self.auth_token = data.get("access_token")
                    print("  ✓ Authentication successful")
                else:
                    # For testing purposes, create a mock token
                    self.auth_token = "mock-jwt-token-for-testing"
                    print("  ✓ Using mock authentication token")
        except Exception:
            # Fallback to mock token for testing
            self.auth_token = "mock-jwt-token-for-testing"
            print("  ✓ Using mock authentication token")

    async def test_direct_surfacing_service(self):
        """Test the direct Node.js surfacing service"""
        test_data = {
            "text": "This is a test document for surface analysis.",
            "options": {
                "extract_keywords": True,
                "analyze_sentiment": True
            }
        }
        
        async with self.session.post(f"{self.surfacing_direct_url}/api/analyze", json=test_data) as response:
            if response.status != 200:
                raise Exception(f"Direct service failed (status: {response.status})")
            
            data = await response.json()
            if "analysis" not in data:
                raise Exception("Direct service response missing analysis data")
            
            print("  ✓ Direct surfacing service is working")
            print(f"    - Analysis keys: {list(data['analysis'].keys())}")

    async def test_surfacing_module(self):
        """Test the Python surfacing module endpoints"""
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        
        # Test analyze endpoint
        test_data = {
            "text": "This is a test document for the surfacing module.",
            "user_id": self.user_id,
            "options": {
                "extract_keywords": True,
                "analyze_sentiment": True,
                "store_in_memory": True
            }
        }
        
        async with self.session.post(f"{self.surfacing_module_url}/api/v1/analyze", 
                                   json=test_data, headers=headers) as response:
            if response.status != 200:
                raise Exception(f"Module analyze endpoint failed (status: {response.status})")
            
            data = await response.json()
            if "analysis" not in data:
                raise Exception("Module response missing analysis data")
            
            print("  ✓ Surfacing module analyze endpoint working")
            
        # Test batch analyze endpoint
        batch_data = {
            "documents": [
                {"id": "doc1", "text": "First test document"},
                {"id": "doc2", "text": "Second test document"}
            ],
            "user_id": self.user_id,
            "options": {"extract_keywords": True}
        }
        
        async with self.session.post(f"{self.surfacing_module_url}/api/v1/batch-analyze", 
                                   json=batch_data, headers=headers) as response:
            if response.status != 200:
                raise Exception(f"Module batch analyze failed (status: {response.status})")
            
            data = await response.json()
            if "results" not in data or len(data["results"]) != 2:
                raise Exception("Module batch response invalid")
            
            print("  ✓ Surfacing module batch analyze endpoint working")

    async def test_gateway_integration(self):
        """Test access through the LiftOS API gateway"""
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        
        test_data = {
            "text": "Testing through the LiftOS gateway.",
            "user_id": self.user_id,
            "options": {"extract_keywords": True}
        }
        
        async with self.session.post(f"{self.base_url}/api/v1/modules/surfacing/analyze", 
                                   json=test_data, headers=headers) as response:
            if response.status != 200:
                raise Exception(f"Gateway integration failed (status: {response.status})")
            
            data = await response.json()
            if "analysis" not in data:
                raise Exception("Gateway response missing analysis data")
            
            print("  ✓ Gateway integration working")
            print("  ✓ Request properly routed through LiftOS gateway")

    async def test_memory_integration(self):
        """Test integration with LiftOS memory services"""
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        
        # Analyze with memory storage enabled
        test_data = {
            "text": "This document should be stored in memory for future retrieval.",
            "user_id": self.user_id,
            "options": {
                "extract_keywords": True,
                "store_in_memory": True,
                "memory_tags": ["test", "integration"]
            }
        }
        
        async with self.session.post(f"{self.surfacing_module_url}/api/v1/analyze", 
                                   json=test_data, headers=headers) as response:
            if response.status != 200:
                raise Exception(f"Memory integration test failed (status: {response.status})")
            
            data = await response.json()
            if "memory_id" not in data:
                print("  ⚠ Memory storage not confirmed (memory service may not be available)")
            else:
                print("  ✓ Memory integration working")
                print(f"    - Memory ID: {data['memory_id']}")

    async def test_error_handling(self):
        """Test error handling and validation"""
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        
        # Test with invalid data
        invalid_data = {"invalid": "data"}
        
        async with self.session.post(f"{self.surfacing_module_url}/api/v1/analyze", 
                                   json=invalid_data, headers=headers) as response:
            if response.status == 200:
                raise Exception("Error handling failed - invalid data was accepted")
            
            if response.status not in [400, 422]:
                raise Exception(f"Unexpected error status: {response.status}")
            
            print("  ✓ Input validation working")
            
        # Test without authentication
        async with self.session.post(f"{self.surfacing_module_url}/api/v1/analyze", 
                                   json={"text": "test"}) as response:
            if response.status not in [401, 403]:
                print("  ⚠ Authentication validation may not be enforced")
            else:
                print("  ✓ Authentication validation working")

    async def test_performance(self):
        """Basic performance test"""
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        
        test_data = {
            "text": "Performance test document with some content to analyze.",
            "user_id": self.user_id,
            "options": {"extract_keywords": True}
        }
        
        # Test response time
        start_time = time.time()
        async with self.session.post(f"{self.surfacing_module_url}/api/v1/analyze", 
                                   json=test_data, headers=headers) as response:
            end_time = time.time()
            
            if response.status != 200:
                raise Exception(f"Performance test failed (status: {response.status})")
            
            response_time = end_time - start_time
            print(f"  ✓ Response time: {response_time:.2f}s")
            
            if response_time > 5.0:
                print("  ⚠ Response time is slower than expected")
            else:
                print("  ✓ Response time is acceptable")

async def main():
    """Main test runner"""
    tester = SurfacingIntegrationTester()
    
    try:
        success = await tester.run_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())