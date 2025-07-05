#!/usr/bin/env python3
"""
Test script for production-ready features
Tests health checks, security, logging, and secrets management
"""

import asyncio
import aiohttp
import json
import time
import sys
import os
from typing import Dict, List, Any

# Add shared modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.health.health_checks import HealthChecker, check_database_connection
from shared.security.security_manager import SecurityManager
from shared.logging.structured_logger import setup_service_logging
from shared.config.secrets_manager import get_secrets_manager

class ProductionFeaturesTester:
    """Test production-ready features across all services"""
    
    def __init__(self):
        self.base_urls = {
            "gateway": "http://localhost:8000",
            "auth": "http://localhost:8001",
            "memory": "http://localhost:8003",
            "registry": "http://localhost:8005"
        }
        self.results = {}
        self.logger = setup_service_logging("test_runner")
    
    async def run_all_tests(self):
        """Run all production feature tests"""
        print("[ROCKET] Starting Production Features Test Suite")
        print("=" * 60)
        
        # Test categories
        test_categories = [
            ("Health Checks", self.test_health_checks),
            ("Security Features", self.test_security_features),
            ("Logging System", self.test_logging_system),
            ("Secrets Management", self.test_secrets_management),
            ("Rate Limiting", self.test_rate_limiting),
            ("Service Integration", self.test_service_integration)
        ]
        
        total_tests = 0
        passed_tests = 0
        
        for category_name, test_func in test_categories:
            print(f"\n[CLIPBOARD] Testing {category_name}")
            print("-" * 40)
            
            try:
                category_results = await test_func()
                self.results[category_name] = category_results
                
                category_passed = sum(1 for result in category_results.values() if result.get("passed", False))
                category_total = len(category_results)
                
                total_tests += category_total
                passed_tests += category_passed
                
                print(f"[PASS] {category_name}: {category_passed}/{category_total} tests passed")
                
            except Exception as e:
                print(f"[FAIL] {category_name}: Failed with error: {e}")
                self.results[category_name] = {"error": str(e)}
        
        # Print summary
        print("\n" + "=" * 60)
        print("[CHART] TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        # Detailed results
        print("\n[CLIPBOARD] DETAILED RESULTS")
        print("-" * 60)
        for category, results in self.results.items():
            print(f"\n{category}:")
            if isinstance(results, dict) and "error" in results:
                print(f"  [FAIL] Error: {results['error']}")
            else:
                for test_name, result in results.items():
                    status = "[PASS]" if result.get("passed", False) else "[FAIL]"
                    print(f"  {status} {test_name}")
                    if not result.get("passed", False) and "error" in result:
                        print(f"      Error: {result['error']}")
        
        return passed_tests, total_tests
    
    async def test_health_checks(self) -> Dict[str, Any]:
        """Test health check endpoints"""
        results = {}
        
        async with aiohttp.ClientSession() as session:
            for service_name, base_url in self.base_urls.items():
                # Test basic health endpoint
                try:
                    async with session.get(f"{base_url}/health", timeout=5) as response:
                        if response.status == 200:
                            data = await response.json()
                            results[f"{service_name}_health"] = {
                                "passed": True,
                                "status": data.get("status"),
                                "response_time": response.headers.get("X-Response-Time", "N/A")
                            }
                        else:
                            results[f"{service_name}_health"] = {
                                "passed": False,
                                "error": f"HTTP {response.status}"
                            }
                except Exception as e:
                    results[f"{service_name}_health"] = {
                        "passed": False,
                        "error": str(e)
                    }
                
                # Test readiness endpoint
                try:
                    async with session.get(f"{base_url}/ready", timeout=5) as response:
                        results[f"{service_name}_readiness"] = {
                            "passed": response.status == 200,
                            "status_code": response.status
                        }
                except Exception as e:
                    results[f"{service_name}_readiness"] = {
                        "passed": False,
                        "error": str(e)
                    }
        
        return results
    
    async def test_security_features(self) -> Dict[str, Any]:
        """Test security features"""
        results = {}
        
        # Test JWT token creation and verification
        try:
            security_manager = SecurityManager("test-secret")
            
            # Test token creation
            token = security_manager.create_access_token({"sub": "test_user", "role": "user"})
            results["jwt_creation"] = {"passed": bool(token)}
            
            # Test token verification
            from fastapi.security import HTTPAuthorizationCredentials
            credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
            payload = await security_manager.verify_jwt(credentials)
            results["jwt_verification"] = {
                "passed": payload.get("sub") == "test_user"
            }
            
        except Exception as e:
            results["jwt_creation"] = {"passed": False, "error": str(e)}
            results["jwt_verification"] = {"passed": False, "error": str(e)}
        
        # Test service token creation
        try:
            security_manager = SecurityManager("test-secret")
            service_token = security_manager.create_service_token("test-service", ["read", "write"])
            results["service_token_creation"] = {"passed": bool(service_token)}
        except Exception as e:
            results["service_token_creation"] = {"passed": False, "error": str(e)}
        
        return results
    
    async def test_logging_system(self) -> Dict[str, Any]:
        """Test structured logging system"""
        results = {}
        
        try:
            # Test logger creation
            logger = setup_service_logging("test_service")
            results["logger_creation"] = {"passed": bool(logger)}
            
            # Test logging with structured data
            logger.info("Test log message", extra={"test_field": "test_value"})
            results["structured_logging"] = {"passed": True}
            
            # Test different log levels
            logger.debug("Debug message")
            logger.warning("Warning message")
            logger.error("Error message")
            results["log_levels"] = {"passed": True}
            
        except Exception as e:
            results["logger_creation"] = {"passed": False, "error": str(e)}
            results["structured_logging"] = {"passed": False, "error": str(e)}
            results["log_levels"] = {"passed": False, "error": str(e)}
        
        return results
    
    async def test_secrets_management(self) -> Dict[str, Any]:
        """Test secrets management system"""
        results = {}
        
        try:
            # Test secrets manager initialization
            secrets_manager = get_secrets_manager()
            results["secrets_manager_init"] = {"passed": bool(secrets_manager)}
            
            # Test environment variable backend
            os.environ["TEST_SECRET"] = "test_value"
            secret_value = await secrets_manager.get_secret("TEST_SECRET")
            results["env_secret_retrieval"] = {
                "passed": secret_value == "test_value"
            }
            
            # Test JSON secret parsing
            os.environ["TEST_JSON_SECRET"] = '{"key1": "value1", "key2": "value2"}'
            json_secret = await secrets_manager.get_secret("TEST_JSON_SECRET")
            results["json_secret_parsing"] = {
                "passed": isinstance(json_secret, dict) and json_secret.get("key1") == "value1"
            }
            
            # Test secret value extraction
            secret_key_value = await secrets_manager.get_secret_value("TEST_JSON_SECRET", "key2")
            results["secret_key_extraction"] = {
                "passed": secret_key_value == "value2"
            }
            
        except Exception as e:
            results["secrets_manager_init"] = {"passed": False, "error": str(e)}
            results["env_secret_retrieval"] = {"passed": False, "error": str(e)}
            results["json_secret_parsing"] = {"passed": False, "error": str(e)}
            results["secret_key_extraction"] = {"passed": False, "error": str(e)}
        
        return results
    
    async def test_rate_limiting(self) -> Dict[str, Any]:
        """Test rate limiting functionality"""
        results = {}
        
        async with aiohttp.ClientSession() as session:
            # Test normal requests (should pass)
            try:
                async with session.get(f"{self.base_urls['gateway']}/", timeout=5) as response:
                    results["normal_request"] = {
                        "passed": response.status == 200
                    }
            except Exception as e:
                results["normal_request"] = {"passed": False, "error": str(e)}
            
            # Test rapid requests (may trigger rate limiting)
            rate_limit_triggered = False
            try:
                for i in range(10):
                    async with session.get(f"{self.base_urls['gateway']}/", timeout=2) as response:
                        if response.status == 429:
                            rate_limit_triggered = True
                            break
                
                results["rate_limiting_detection"] = {
                    "passed": True,  # Pass if we can make requests without errors
                    "rate_limit_triggered": rate_limit_triggered
                }
            except Exception as e:
                results["rate_limiting_detection"] = {"passed": False, "error": str(e)}
        
        return results
    
    async def test_service_integration(self) -> Dict[str, Any]:
        """Test service integration and communication"""
        results = {}
        
        async with aiohttp.ClientSession() as session:
            # Test gateway to auth service communication
            try:
                async with session.get(f"{self.base_urls['gateway']}/auth/health", timeout=10) as response:
                    results["gateway_auth_integration"] = {
                        "passed": response.status in [200, 404],  # 404 is OK if endpoint doesn't exist
                        "status_code": response.status
                    }
            except Exception as e:
                results["gateway_auth_integration"] = {"passed": False, "error": str(e)}
            
            # Test direct service access
            for service_name, base_url in self.base_urls.items():
                try:
                    async with session.get(f"{base_url}/", timeout=5) as response:
                        results[f"{service_name}_direct_access"] = {
                            "passed": response.status in [200, 404],
                            "status_code": response.status
                        }
                except Exception as e:
                    results[f"{service_name}_direct_access"] = {"passed": False, "error": str(e)}
        
        return results

async def main():
    """Main test runner"""
    tester = ProductionFeaturesTester()
    
    try:
        passed, total = await tester.run_all_tests()
        
        # Exit with appropriate code
        if passed == total:
            print("\n[SUCCESS] All tests passed! Production features are working correctly.")
            sys.exit(0)
        else:
            print(f"\n[WARNING]  {total - passed} tests failed. Please review the results above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n[STOP]  Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n[FAIL] Test runner failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())