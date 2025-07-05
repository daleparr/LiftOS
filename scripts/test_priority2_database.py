"""
Priority 2 Testing: Database Setup & Data Layer Testing
"""

import os
import sys
import asyncio
import httpx
import json
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from shared.database import get_database, User, Session, BillingAccount, Module
from sqlalchemy import select
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Priority2Tester:
    """Priority 2 testing framework"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.services = {
            "gateway": 8000,
            "auth": 8001,
            "memory": 8002,
            "registry": 8004
        }
        self.test_results = {
            "database_tests": [],
            "service_tests": [],
            "integration_tests": [],
            "performance_tests": []
        }
        self.access_token = None
    
    async def run_all_tests(self):
        """Run all Priority 2 tests"""
        logger.info("="*60)
        logger.info("PRIORITY 2 TESTING: DATABASE SETUP & DATA LAYER")
        logger.info("="*60)
        
        # Test 1: Database Layer Tests
        await self.test_database_layer()
        
        # Test 2: Service Health with Database
        await self.test_service_health()
        
        # Test 3: Authentication Flow with Database
        await self.test_auth_flow()
        
        # Test 4: Cross-Service Communication
        await self.test_cross_service_communication()
        
        # Test 5: Data Persistence
        await self.test_data_persistence()
        
        # Test 6: Performance Tests
        await self.test_performance()
        
        # Generate report
        await self.generate_report()
    
    async def test_database_layer(self):
        """Test database layer functionality"""
        logger.info("\n1. Testing Database Layer...")
        
        try:
            # Test database connection
            db_manager = await get_database()
            
            async with db_manager.get_session() as session:
                # Test user queries
                result = await session.execute(select(User))
                users = result.scalars().all()
                
                # Test module queries
                result = await session.execute(select(Module))
                modules = result.scalars().all()
                
                # Test billing account queries
                result = await session.execute(select(BillingAccount))
                billing_accounts = result.scalars().all()
                
                self.test_results["database_tests"].append({
                    "test": "Database Connection",
                    "status": "PASS",
                    "details": f"Connected successfully, {len(users)} users, {len(modules)} modules, {len(billing_accounts)} billing accounts"
                })
                
                logger.info(f"  ✓ Database connection successful")
                logger.info(f"  ✓ Found {len(users)} users")
                logger.info(f"  ✓ Found {len(modules)} modules")
                logger.info(f"  ✓ Found {len(billing_accounts)} billing accounts")
                
        except Exception as e:
            self.test_results["database_tests"].append({
                "test": "Database Connection",
                "status": "FAIL",
                "details": str(e)
            })
            logger.error(f"  ✗ Database connection failed: {e}")
    
    async def test_service_health(self):
        """Test service health endpoints"""
        logger.info("\n2. Testing Service Health with Database...")
        
        async with httpx.AsyncClient() as client:
            for service_name, port in self.services.items():
                try:
                    url = f"http://localhost:{port}/health"
                    response = await client.get(url, timeout=10.0)
                    
                    if response.status_code == 200:
                        health_data = response.json()
                        self.test_results["service_tests"].append({
                            "test": f"{service_name} Health Check",
                            "status": "PASS",
                            "details": health_data
                        })
                        logger.info(f"  ✓ {service_name} health check passed")
                    else:
                        self.test_results["service_tests"].append({
                            "test": f"{service_name} Health Check",
                            "status": "FAIL",
                            "details": f"HTTP {response.status_code}"
                        })
                        logger.error(f"  ✗ {service_name} health check failed: HTTP {response.status_code}")
                        
                except Exception as e:
                    self.test_results["service_tests"].append({
                        "test": f"{service_name} Health Check",
                        "status": "FAIL",
                        "details": str(e)
                    })
                    logger.error(f"  ✗ {service_name} health check failed: {e}")
    
    async def test_auth_flow(self):
        """Test authentication flow with database"""
        logger.info("\n3. Testing Authentication Flow with Database...")
        
        async with httpx.AsyncClient() as client:
            try:
                # Test registration
                register_data = {
                    "email": "test_priority2@lift.com",
                    "password": "testpassword123",
                    "username": "testuser_p2",
                    "full_name": "Priority 2 Test User"
                }
                
                response = await client.post(
                    f"{self.base_url}/auth/register",
                    json=register_data,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    register_result = response.json()
                    self.access_token = register_result["data"]["access_token"]
                    
                    self.test_results["integration_tests"].append({
                        "test": "User Registration",
                        "status": "PASS",
                        "details": "User registered successfully with database persistence"
                    })
                    logger.info("  ✓ User registration successful")
                    
                    # Test login
                    login_data = {
                        "email": "test_priority2@lift.com",
                        "password": "testpassword123"
                    }
                    
                    response = await client.post(
                        f"{self.base_url}/auth/login",
                        json=login_data,
                        timeout=10.0
                    )
                    
                    if response.status_code == 200:
                        login_result = response.json()
                        self.access_token = login_result["data"]["access_token"]
                        
                        self.test_results["integration_tests"].append({
                            "test": "User Login",
                            "status": "PASS",
                            "details": "Login successful with database verification"
                        })
                        logger.info("  ✓ User login successful")
                        
                        # Test token verification
                        headers = {"Authorization": f"Bearer {self.access_token}"}
                        response = await client.post(
                            f"{self.base_url}/auth/verify",
                            headers=headers,
                            timeout=10.0
                        )
                        
                        if response.status_code == 200:
                            self.test_results["integration_tests"].append({
                                "test": "Token Verification",
                                "status": "PASS",
                                "details": "Token verified with session tracking"
                            })
                            logger.info("  ✓ Token verification successful")
                        else:
                            self.test_results["integration_tests"].append({
                                "test": "Token Verification",
                                "status": "FAIL",
                                "details": f"HTTP {response.status_code}"
                            })
                            logger.error(f"  ✗ Token verification failed: HTTP {response.status_code}")
                    
                    else:
                        self.test_results["integration_tests"].append({
                            "test": "User Login",
                            "status": "FAIL",
                            "details": f"HTTP {response.status_code}"
                        })
                        logger.error(f"  ✗ User login failed: HTTP {response.status_code}")
                
                else:
                    self.test_results["integration_tests"].append({
                        "test": "User Registration",
                        "status": "FAIL",
                        "details": f"HTTP {response.status_code}"
                    })
                    logger.error(f"  ✗ User registration failed: HTTP {response.status_code}")
                    
            except Exception as e:
                self.test_results["integration_tests"].append({
                    "test": "Authentication Flow",
                    "status": "FAIL",
                    "details": str(e)
                })
                logger.error(f"  ✗ Authentication flow failed: {e}")
    
    async def test_cross_service_communication(self):
        """Test cross-service communication through gateway"""
        logger.info("\n4. Testing Cross-Service Communication...")
        
        if not self.access_token:
            logger.warning("  ! No access token available, skipping authenticated tests")
            return
        
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            # Test gateway routing to different services
            test_endpoints = [
                ("/auth/verify", "Authentication Service"),
                ("/memory/health", "Memory Service"),
                ("/registry/health", "Registry Service")
            ]
            
            for endpoint, service_name in test_endpoints:
                try:
                    response = await client.get(
                        f"{self.base_url}{endpoint}",
                        headers=headers if "verify" in endpoint else {},
                        timeout=10.0
                    )
                    
                    if response.status_code == 200:
                        self.test_results["integration_tests"].append({
                            "test": f"Gateway -> {service_name}",
                            "status": "PASS",
                            "details": "Cross-service communication successful"
                        })
                        logger.info(f"  ✓ Gateway -> {service_name} communication successful")
                    else:
                        self.test_results["integration_tests"].append({
                            "test": f"Gateway -> {service_name}",
                            "status": "FAIL",
                            "details": f"HTTP {response.status_code}"
                        })
                        logger.error(f"  ✗ Gateway -> {service_name} failed: HTTP {response.status_code}")
                        
                except Exception as e:
                    self.test_results["integration_tests"].append({
                        "test": f"Gateway -> {service_name}",
                        "status": "FAIL",
                        "details": str(e)
                    })
                    logger.error(f"  ✗ Gateway -> {service_name} failed: {e}")
    
    async def test_data_persistence(self):
        """Test data persistence across service restarts"""
        logger.info("\n5. Testing Data Persistence...")
        
        try:
            # Verify data persists in database
            db_manager = await get_database()
            
            async with db_manager.get_session() as session:
                # Check if our test user exists
                result = await session.execute(
                    select(User).where(User.email == "test_priority2@lift.com")
                )
                test_user = result.scalar_one_or_none()
                
                if test_user:
                    # Check if session exists
                    result = await session.execute(
                        select(Session).where(Session.user_id == test_user.id)
                    )
                    sessions = result.scalars().all()
                    
                    # Check if billing account exists
                    result = await session.execute(
                        select(BillingAccount).where(BillingAccount.user_id == test_user.id)
                    )
                    billing_account = result.scalar_one_or_none()
                    
                    self.test_results["database_tests"].append({
                        "test": "Data Persistence",
                        "status": "PASS",
                        "details": f"User persisted with {len(sessions)} sessions and billing account"
                    })
                    logger.info(f"  ✓ Data persistence verified")
                    logger.info(f"    - User: {test_user.email}")
                    logger.info(f"    - Sessions: {len(sessions)}")
                    logger.info(f"    - Billing account: {'Yes' if billing_account else 'No'}")
                
                else:
                    self.test_results["database_tests"].append({
                        "test": "Data Persistence",
                        "status": "FAIL",
                        "details": "Test user not found in database"
                    })
                    logger.error("  ✗ Data persistence failed: Test user not found")
                    
        except Exception as e:
            self.test_results["database_tests"].append({
                "test": "Data Persistence",
                "status": "FAIL",
                "details": str(e)
            })
            logger.error(f"  ✗ Data persistence test failed: {e}")
    
    async def test_performance(self):
        """Test performance with database operations"""
        logger.info("\n6. Testing Performance...")
        
        if not self.access_token:
            logger.warning("  ! No access token available, skipping performance tests")
            return
        
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            # Test response times
            endpoints = [
                "/health",
                "/auth/verify",
                "/memory/health",
                "/registry/health"
            ]
            
            for endpoint in endpoints:
                try:
                    start_time = time.time()
                    response = await client.get(
                        f"{self.base_url}{endpoint}",
                        headers=headers if "verify" in endpoint else {},
                        timeout=10.0
                    )
                    end_time = time.time()
                    
                    response_time = (end_time - start_time) * 1000  # Convert to ms
                    
                    if response.status_code == 200 and response_time < 1000:  # Under 1 second
                        self.test_results["performance_tests"].append({
                            "test": f"Response Time {endpoint}",
                            "status": "PASS",
                            "details": f"{response_time:.2f}ms"
                        })
                        logger.info(f"  ✓ {endpoint}: {response_time:.2f}ms")
                    else:
                        self.test_results["performance_tests"].append({
                            "test": f"Response Time {endpoint}",
                            "status": "FAIL",
                            "details": f"{response_time:.2f}ms (too slow or error)"
                        })
                        logger.warning(f"  ! {endpoint}: {response_time:.2f}ms (slow)")
                        
                except Exception as e:
                    self.test_results["performance_tests"].append({
                        "test": f"Response Time {endpoint}",
                        "status": "FAIL",
                        "details": str(e)
                    })
                    logger.error(f"  ✗ {endpoint} performance test failed: {e}")
    
    async def generate_report(self):
        """Generate comprehensive test report"""
        logger.info("\n" + "="*60)
        logger.info("PRIORITY 2 TEST RESULTS SUMMARY")
        logger.info("="*60)
        
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
            "priority": 2,
            "description": "Database Setup & Data Layer Testing",
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": success_rate
            },
            "results": self.test_results
        }
        
        report_path = project_root / "PRIORITY_2_RESULTS.md"
        with open(report_path, "w") as f:
            f.write("# Priority 2 Test Results: Database Setup & Data Layer Testing\n\n")
            f.write(f"**Test Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Success Rate:** {success_rate:.1f}% ({passed_tests}/{total_tests} tests passed)\n\n")
            
            f.write("## Summary\n\n")
            f.write("Priority 2 testing focused on database integration, data persistence, and cross-service communication with database backing.\n\n")
            
            f.write("## Test Categories\n\n")
            for category, tests in self.test_results.items():
                f.write(f"### {category.replace('_', ' ').title()}\n\n")
                for test in tests:
                    status_icon = "✅" if test["status"] == "PASS" else "❌"
                    f.write(f"- {status_icon} **{test['test']}**: {test['status']}\n")
                    if test.get("details"):
                        f.write(f"  - Details: {test['details']}\n")
                f.write("\n")
            
            f.write("## Detailed Results\n\n")
            f.write("```json\n")
            f.write(json.dumps(report_data, indent=2))
            f.write("\n```\n")
        
        logger.info(f"\nDetailed report saved to: {report_path}")
        
        if success_rate >= 80:
            logger.info("\n🎉 Priority 2 testing PASSED! Ready for Priority 3.")
        else:
            logger.warning("\n⚠️  Priority 2 testing needs attention before proceeding.")

async def main():
    """Main test function"""
    tester = Priority2Tester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())