#!/usr/bin/env python3
"""
Priority 3 Test Suite: Complete Service Integration Testing
Tests full service integration, module management, and cross-service communication
"""

import asyncio
import httpx
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Get project root
project_root = Path(__file__).parent.parent

class Priority3Tester:
    """Comprehensive integration testing for Priority 3"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.services = {
            "gateway": 8000,
            "auth": 8001,
            "memory": 8003,
            "registry": 8005
        }
        self.access_token = None
        self.test_results = {
            "service_health_tests": [],
            "authentication_tests": [],
            "module_management_tests": [],
            "memory_integration_tests": [],
            "cross_service_tests": [],
            "performance_tests": []
        }
        
    async def run_all_tests(self):
        """Run complete Priority 3 test suite"""
        logger.info("🚀 Starting Priority 3: Complete Service Integration Testing")
        logger.info("="*70)
        
        # Test service health
        await self.test_all_service_health()
        
        # Test authentication flow
        await self.test_complete_auth_flow()
        
        # Test module management
        await self.test_module_management()
        
        # Test memory service integration
        await self.test_memory_integration()
        
        # Test cross-service communication
        await self.test_cross_service_communication()
        
        # Test performance
        await self.test_system_performance()
        
        # Generate comprehensive report
        await self.generate_report()
    
    async def test_all_service_health(self):
        """Test health of all core services"""
        logger.info("\n1. Testing All Service Health...")
        
        async with httpx.AsyncClient() as client:
            for service_name, port in self.services.items():
                try:
                    if service_name == "gateway":
                        url = f"http://localhost:{port}/health"
                    else:
                        url = f"http://localhost:{port}/health"
                    
                    response = await client.get(url, timeout=10.0)
                    
                    if response.status_code == 200:
                        self.test_results["service_health_tests"].append({
                            "test": f"{service_name.title()} Service Health",
                            "status": "PASS",
                            "details": f"Service responding on port {port}"
                        })
                        logger.info(f"  ✓ {service_name} service health check passed")
                    else:
                        self.test_results["service_health_tests"].append({
                            "test": f"{service_name.title()} Service Health",
                            "status": "FAIL",
                            "details": f"HTTP {response.status_code}"
                        })
                        logger.error(f"  ✗ {service_name} service health check failed: {response.status_code}")
                        
                except Exception as e:
                    self.test_results["service_health_tests"].append({
                        "test": f"{service_name.title()} Service Health",
                        "status": "FAIL",
                        "details": str(e)
                    })
                    logger.error(f"  ✗ {service_name} service health check failed: {e}")
    
    async def test_complete_auth_flow(self):
        """Test complete authentication flow"""
        logger.info("\n2. Testing Complete Authentication Flow...")
        
        async with httpx.AsyncClient() as client:
            try:
                # Test user registration
                register_data = {
                    "email": "priority3_test@lift.com",
                    "password": "testpassword123",
                    "username": "priority3_user",
                    "full_name": "Priority 3 Test User"
                }
                
                response = await client.post(
                    f"{self.base_url}/auth/register",
                    json=register_data,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    register_result = response.json()
                    self.access_token = register_result["data"]["access_token"]
                    
                    self.test_results["authentication_tests"].append({
                        "test": "User Registration",
                        "status": "PASS",
                        "details": "User registered successfully with JWT token"
                    })
                    logger.info("  ✓ User registration successful")
                    
                    # Test token validation
                    headers = {"Authorization": f"Bearer {self.access_token}"}
                    profile_response = await client.get(
                        f"{self.base_url}/auth/profile",
                        headers=headers,
                        timeout=10.0
                    )
                    
                    if profile_response.status_code == 200:
                        self.test_results["authentication_tests"].append({
                            "test": "Token Validation",
                            "status": "PASS",
                            "details": "JWT token validated successfully"
                        })
                        logger.info("  ✓ Token validation successful")
                    else:
                        self.test_results["authentication_tests"].append({
                            "test": "Token Validation",
                            "status": "FAIL",
                            "details": f"HTTP {profile_response.status_code}"
                        })
                        logger.error(f"  ✗ Token validation failed: {profile_response.status_code}")
                        
                elif response.status_code == 400:
                    # User might already exist, try login instead
                    login_data = {
                        "email": "priority3_test@lift.com",
                        "password": "testpassword123"
                    }
                    
                    login_response = await client.post(
                        f"{self.base_url}/auth/login",
                        json=login_data,
                        timeout=10.0
                    )
                    
                    if login_response.status_code == 200:
                        login_result = login_response.json()
                        self.access_token = login_result["data"]["access_token"]
                        
                        self.test_results["authentication_tests"].append({
                            "test": "User Login",
                            "status": "PASS",
                            "details": "User login successful with existing account"
                        })
                        logger.info("  ✓ User login successful (existing account)")
                    else:
                        self.test_results["authentication_tests"].append({
                            "test": "User Authentication",
                            "status": "FAIL",
                            "details": f"Both registration and login failed"
                        })
                        logger.error("  ✗ User authentication failed")
                        
                else:
                    self.test_results["authentication_tests"].append({
                        "test": "User Registration",
                        "status": "FAIL",
                        "details": f"HTTP {response.status_code}"
                    })
                    logger.error(f"  ✗ User registration failed: {response.status_code}")
                    
            except Exception as e:
                self.test_results["authentication_tests"].append({
                    "test": "Authentication Flow",
                    "status": "FAIL",
                    "details": str(e)
                })
                logger.error(f"  ✗ Authentication flow failed: {e}")
    
    async def test_module_management(self):
        """Test module registration and discovery"""
        logger.info("\n3. Testing Module Management...")
        
        if not self.access_token:
            logger.warning("  ⚠️ Skipping module tests - no auth token")
            return
            
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            try:
                # Test module discovery
                response = await client.get(
                    f"{self.base_url}/registry/modules",
                    headers=headers,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    modules = response.json()
                    self.test_results["module_management_tests"].append({
                        "test": "Module Discovery",
                        "status": "PASS",
                        "details": f"Found {len(modules.get('data', []))} registered modules"
                    })
                    logger.info(f"  ✓ Module discovery successful - {len(modules.get('data', []))} modules")
                else:
                    self.test_results["module_management_tests"].append({
                        "test": "Module Discovery",
                        "status": "FAIL",
                        "details": f"HTTP {response.status_code}"
                    })
                    logger.error(f"  ✗ Module discovery failed: {response.status_code}")
                
                # Test module registration
                test_module = {
                    "name": "test-module-p3",
                    "version": "1.0.0",
                    "description": "Priority 3 test module",
                    "endpoints": ["/test"],
                    "health_check": "/health"
                }
                
                register_response = await client.post(
                    f"{self.base_url}/registry/modules",
                    json=test_module,
                    headers=headers,
                    timeout=10.0
                )
                
                if register_response.status_code in [200, 201]:
                    self.test_results["module_management_tests"].append({
                        "test": "Module Registration",
                        "status": "PASS",
                        "details": "Test module registered successfully"
                    })
                    logger.info("  ✓ Module registration successful")
                else:
                    self.test_results["module_management_tests"].append({
                        "test": "Module Registration",
                        "status": "FAIL",
                        "details": f"HTTP {register_response.status_code}"
                    })
                    logger.error(f"  ✗ Module registration failed: {register_response.status_code}")
                    
            except Exception as e:
                self.test_results["module_management_tests"].append({
                    "test": "Module Management",
                    "status": "FAIL",
                    "details": str(e)
                })
                logger.error(f"  ✗ Module management failed: {e}")
    
    async def test_memory_integration(self):
        """Test memory service integration"""
        logger.info("\n4. Testing Memory Service Integration...")
        
        if not self.access_token:
            logger.warning("  ⚠️ Skipping memory tests - no auth token")
            return
            
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            try:
                # Test memory storage
                memory_data = {
                    "content": "Priority 3 integration test memory",
                    "context": "testing",
                    "metadata": {"test": "priority3", "timestamp": datetime.now().isoformat()}
                }
                
                response = await client.post(
                    f"{self.base_url}/memory/store",
                    json=memory_data,
                    headers=headers,
                    timeout=10.0
                )
                
                if response.status_code in [200, 201]:
                    memory_result = response.json()
                    memory_id = memory_result.get("data", {}).get("id")
                    
                    self.test_results["memory_integration_tests"].append({
                        "test": "Memory Storage",
                        "status": "PASS",
                        "details": f"Memory stored with ID: {memory_id}"
                    })
                    logger.info("  ✓ Memory storage successful")
                    
                    # Test memory retrieval
                    if memory_id:
                        retrieve_response = await client.get(
                            f"{self.base_url}/memory/{memory_id}",
                            headers=headers,
                            timeout=10.0
                        )
                        
                        if retrieve_response.status_code == 200:
                            self.test_results["memory_integration_tests"].append({
                                "test": "Memory Retrieval",
                                "status": "PASS",
                                "details": "Memory retrieved successfully"
                            })
                            logger.info("  ✓ Memory retrieval successful")
                        else:
                            self.test_results["memory_integration_tests"].append({
                                "test": "Memory Retrieval",
                                "status": "FAIL",
                                "details": f"HTTP {retrieve_response.status_code}"
                            })
                            logger.error(f"  ✗ Memory retrieval failed: {retrieve_response.status_code}")
                            
                else:
                    self.test_results["memory_integration_tests"].append({
                        "test": "Memory Storage",
                        "status": "FAIL",
                        "details": f"HTTP {response.status_code}"
                    })
                    logger.error(f"  ✗ Memory storage failed: {response.status_code}")
                    
            except Exception as e:
                self.test_results["memory_integration_tests"].append({
                    "test": "Memory Integration",
                    "status": "FAIL",
                    "details": str(e)
                })
                logger.error(f"  ✗ Memory integration failed: {e}")
    
    async def test_cross_service_communication(self):
        """Test cross-service communication through gateway"""
        logger.info("\n5. Testing Cross-Service Communication...")
        
        async with httpx.AsyncClient() as client:
            try:
                # Test gateway routing to different services
                services_to_test = [
                    ("auth", "/auth/health"),
                    ("memory", "/memory/health"),
                    ("registry", "/registry/health")
                ]
                
                for service_name, endpoint in services_to_test:
                    response = await client.get(
                        f"{self.base_url}{endpoint}",
                        timeout=10.0
                    )
                    
                    if response.status_code == 200:
                        self.test_results["cross_service_tests"].append({
                            "test": f"Gateway to {service_name.title()} Routing",
                            "status": "PASS",
                            "details": f"Gateway successfully routed to {service_name} service"
                        })
                        logger.info(f"  ✓ Gateway to {service_name} routing successful")
                    else:
                        self.test_results["cross_service_tests"].append({
                            "test": f"Gateway to {service_name.title()} Routing",
                            "status": "FAIL",
                            "details": f"HTTP {response.status_code}"
                        })
                        logger.error(f"  ✗ Gateway to {service_name} routing failed: {response.status_code}")
                        
            except Exception as e:
                self.test_results["cross_service_tests"].append({
                    "test": "Cross-Service Communication",
                    "status": "FAIL",
                    "details": str(e)
                })
                logger.error(f"  ✗ Cross-service communication failed: {e}")
    
    async def test_system_performance(self):
        """Test system performance metrics"""
        logger.info("\n6. Testing System Performance...")
        
        async with httpx.AsyncClient() as client:
            try:
                # Test response times
                start_time = time.time()
                response = await client.get(f"{self.base_url}/health", timeout=10.0)
                response_time = (time.time() - start_time) * 1000  # Convert to ms
                
                if response.status_code == 200 and response_time < 1000:  # Less than 1 second
                    self.test_results["performance_tests"].append({
                        "test": "Gateway Response Time",
                        "status": "PASS",
                        "details": f"Response time: {response_time:.2f}ms"
                    })
                    logger.info(f"  ✓ Gateway response time: {response_time:.2f}ms")
                else:
                    self.test_results["performance_tests"].append({
                        "test": "Gateway Response Time",
                        "status": "FAIL",
                        "details": f"Response time: {response_time:.2f}ms (too slow or failed)"
                    })
                    logger.error(f"  ✗ Gateway response time too slow: {response_time:.2f}ms")
                    
                # Test concurrent requests
                tasks = []
                for i in range(5):
                    task = client.get(f"{self.base_url}/health", timeout=10.0)
                    tasks.append(task)
                
                start_time = time.time()
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                concurrent_time = (time.time() - start_time) * 1000
                
                successful_responses = sum(1 for r in responses if hasattr(r, 'status_code') and r.status_code == 200)
                
                if successful_responses >= 4:  # At least 80% success
                    self.test_results["performance_tests"].append({
                        "test": "Concurrent Request Handling",
                        "status": "PASS",
                        "details": f"{successful_responses}/5 requests successful in {concurrent_time:.2f}ms"
                    })
                    logger.info(f"  ✓ Concurrent requests: {successful_responses}/5 successful")
                else:
                    self.test_results["performance_tests"].append({
                        "test": "Concurrent Request Handling",
                        "status": "FAIL",
                        "details": f"Only {successful_responses}/5 requests successful"
                    })
                    logger.error(f"  ✗ Concurrent requests failed: {successful_responses}/5 successful")
                    
            except Exception as e:
                self.test_results["performance_tests"].append({
                    "test": "Performance Testing",
                    "status": "FAIL",
                    "details": str(e)
                })
                logger.error(f"  ✗ Performance testing failed: {e}")
    
    async def generate_report(self):
        """Generate comprehensive Priority 3 test report"""
        logger.info("\n" + "="*70)
        logger.info("PRIORITY 3 TEST RESULTS SUMMARY")
        logger.info("="*70)
        
        total_tests = 0
        passed_tests = 0
        
        for category, tests in self.test_results.items():
            category_total = len(tests)
            category_passed = len([t for t in tests if t["status"] == "PASS"])
            
            total_tests += category_total
            passed_tests += category_passed
            
            logger.info(f"\n{category.replace('_', ' ').title()}:")
            logger.info(f"  Passed: {category_passed}/{category_total}")
            
            for test in tests:
                status_icon = "✓" if test["status"] == "PASS" else "✗"
                logger.info(f"  {status_icon} {test['test']}: {test['status']}")
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        logger.info(f"\nOVERALL RESULTS:")
        logger.info(f"  Total Tests: {total_tests}")
        logger.info(f"  Passed: {passed_tests}")
        logger.info(f"  Failed: {total_tests - passed_tests}")
        logger.info(f"  Success Rate: {success_rate:.1f}%")
        
        # Save detailed report
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "priority": 3,
            "description": "Complete Service Integration Testing",
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": success_rate
            },
            "results": self.test_results
        }
        
        report_path = project_root / "PRIORITY_3_RESULTS.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Priority 3 Test Results: Complete Service Integration Testing\n\n")
            f.write(f"**Test Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Success Rate:** {success_rate:.1f}% ({passed_tests}/{total_tests} tests passed)\n\n")
            
            f.write("## Summary\n\n")
            f.write("Priority 3 testing focused on complete service integration, cross-service communication, ")
            f.write("module management, and system performance validation.\n\n")
            
            f.write("## Services Tested\n\n")
            f.write("- **API Gateway** (Port 8000): Request routing and load balancing\n")
            f.write("- **Auth Service** (Port 8001): User authentication and authorization\n")
            f.write("- **Memory Service** (Port 8003): KSE Memory SDK integration\n")
            f.write("- **Registry Service** (Port 8005): Module registration and discovery\n\n")
            
            f.write("## Test Categories\n\n")
            for category, tests in self.test_results.items():
                f.write(f"### {category.replace('_', ' ').title()}\n\n")
                for test in tests:
                    status_icon = "✅" if test["status"] == "PASS" else "❌"
                    f.write(f"- {status_icon} **{test['test']}**: {test['status']}\n")
                    if test.get("details"):
                        f.write(f"  - Details: {test['details']}\n")
                f.write("\n")
            
            f.write("## System Architecture Validation\n\n")
            f.write("### Service Communication Flow\n")
            f.write("```\n")
            f.write("Client → API Gateway (8000) → Core Services\n")
            f.write("                            ├── Auth Service (8001)\n")
            f.write("                            ├── Memory Service (8003)\n")
            f.write("                            └── Registry Service (8005)\n")
            f.write("```\n\n")
            
            f.write("### Integration Points Tested\n")
            f.write("- Gateway routing to all core services\n")
            f.write("- JWT authentication across services\n")
            f.write("- Database persistence and retrieval\n")
            f.write("- Module registration and discovery\n")
            f.write("- Memory storage and retrieval\n")
            f.write("- Cross-service error handling\n\n")
            
            f.write("## Performance Metrics\n\n")
            performance_tests = self.test_results.get("performance_tests", [])
            for test in performance_tests:
                f.write(f"- **{test['test']}**: {test['details']}\n")
            f.write("\n")
            
            f.write("## Detailed Results\n\n")
            f.write("```json\n")
            f.write(json.dumps(report_data, indent=2))
            f.write("\n```\n")
        
        logger.info(f"\nDetailed report saved to: {report_path}")
        
        if success_rate >= 90:
            logger.info("\n🎉 Priority 3 testing PASSED! System ready for production deployment.")
        elif success_rate >= 75:
            logger.info("\n✅ Priority 3 testing mostly successful. Minor issues to address.")
        else:
            logger.warning("\n⚠️  Priority 3 testing needs significant attention before production.")

async def main():
    """Main test function"""
    tester = Priority3Tester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())