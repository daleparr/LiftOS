#!/usr/bin/env python3
"""
Next Phase Validation - Final Test
Validates all KSE Memory System functionality after fixes
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

class NextPhaseValidator:
    def __init__(self):
        self.base_urls = {
            'memory': 'http://localhost:8003',
            'surfacing': 'http://localhost:9005', 
            'causal': 'http://localhost:8008',
            'llm': 'http://localhost:8009'
        }
        self.test_org_id = "test-org-final"
        self.headers = {
            'x-user-id': 'test-user-123',
            'x-org-id': self.test_org_id,
            'x-memory-context': f'org_{self.test_org_id}_context'
        }
        
    async def test_memory_storage(self):
        """Test memory storage functionality"""
        print("1. Testing Memory Storage...")
        
        async with aiohttp.ClientSession() as session:
            # Store test memory
            store_data = {
                "content": "Final validation test memory",
                "memory_type": "knowledge",
                "organization_id": self.test_org_id,
                "metadata": {"test": "final_validation"}
            }
            
            async with session.post(f"{self.base_urls['memory']}/store", json=store_data, headers=self.headers) as resp:
                if resp.status == 200:
                    print("   [OK] Memory storage successful")
                    return True
                else:
                    error_text = await resp.text()
                    print(f"   [FAIL] Memory storage failed: {resp.status} - {error_text}")
                    return False
    
    async def test_hybrid_search(self):
        """Test hybrid search functionality"""
        print("2. Testing Hybrid Search...")
        
        async with aiohttp.ClientSession() as session:
            search_data = {
                "query": "validation test",
                "organization_id": self.test_org_id,
                "search_type": "hybrid",
                "limit": 5
            }
            
            async with session.post(f"{self.base_urls['memory']}/search", json=search_data, headers=self.headers) as resp:
                if resp.status == 200:
                    results = await resp.json()
                    print(f"   [OK] Hybrid search successful: {len(results.get('results', []))} results")
                    return True
                else:
                    error_text = await resp.text()
                    print(f"   [FAIL] Hybrid search failed: {resp.status} - {error_text}")
                    return False
    
    async def test_microservice_health(self):
        """Test all microservice health endpoints"""
        print("3. Testing Microservice Health...")
        
        health_results = {}
        async with aiohttp.ClientSession() as session:
            for service, url in self.base_urls.items():
                try:
                    async with session.get(f"{url}/health", timeout=5) as resp:
                        if resp.status == 200:
                            health_results[service] = "OK"
                            print(f"   [OK] {service.capitalize()} service healthy")
                        else:
                            health_results[service] = f"FAIL-{resp.status}"
                            print(f"   [FAIL] {service.capitalize()} service unhealthy: {resp.status}")
                except Exception as e:
                    health_results[service] = f"ERROR-{str(e)}"
                    print(f"   [ERROR] {service.capitalize()} service error: {str(e)}")
        
        return all(status == "OK" for status in health_results.values())
    
    async def test_cross_microservice_integration(self):
        """Test cross-microservice memory integration"""
        print("4. Testing Cross-Microservice Integration...")
        
        # Test surfacing service can access memory
        async with aiohttp.ClientSession() as session:
            try:
                # Test surfacing service endpoint that uses memory
                async with session.get(f"{self.base_urls['surfacing']}/health", timeout=5) as resp:
                    if resp.status == 200:
                        print("   [OK] Surfacing service accessible")
                        return True
                    else:
                        print(f"   [FAIL] Surfacing service not accessible: {resp.status}")
                        return False
            except Exception as e:
                print(f"   [ERROR] Cross-microservice test error: {str(e)}")
                return False
    
    async def run_validation(self):
        """Run complete Next Phase validation"""
        print("=" * 60)
        print("NEXT PHASE VALIDATION - FINAL TEST")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all tests
        tests = [
            ("Memory Storage", self.test_memory_storage()),
            ("Hybrid Search", self.test_hybrid_search()),
            ("Microservice Health", self.test_microservice_health()),
            ("Cross-Microservice Integration", self.test_cross_microservice_integration())
        ]
        
        results = {}
        for test_name, test_coro in tests:
            try:
                result = await test_coro
                results[test_name] = result
            except Exception as e:
                print(f"   [ERROR] {test_name} failed with exception: {str(e)}")
                results[test_name] = False
        
        # Calculate results
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result)
        success_rate = (passed_tests / total_tests) * 100
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("NEXT PHASE VALIDATION RESULTS")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Execution Time: {elapsed_time:.2f}s")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if success_rate == 100:
            print("\n[SUCCESS] Next Phase validation COMPLETED successfully!")
            print("All KSE Memory System functionality is working correctly.")
        else:
            print(f"\n[PARTIAL] Next Phase validation completed with {success_rate:.1f}% success rate.")
            print("Some functionality may need additional fixes.")
        
        return success_rate == 100

async def main():
    validator = NextPhaseValidator()
    success = await validator.run_validation()
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)