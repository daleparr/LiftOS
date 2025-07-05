#!/usr/bin/env python3
"""
KSE Memory System Storage Test (No Search)
Tests basic memory storage functionality across microservices
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Any
import uuid

class KSEStorageTester:
    def __init__(self):
        self.base_urls = {
            'memory': 'http://localhost:8003',
            'surfacing': 'http://localhost:9005',
            'causal': 'http://localhost:8008', 
            'llm': 'http://localhost:8009'
        }
        self.test_org_id = "test-org-storage"
        self.test_user_id = "test-user-storage"
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def test_service_health(self) -> Dict[str, bool]:
        """Test health of available services"""
        print("Testing service health...")
        results = {}
        
        for service, url in self.base_urls.items():
            try:
                async with self.session.get(f"{url}/health", timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        status = data.get('status', 'unknown')
                        # Accept both 'healthy' and 'degraded' as functional
                        results[service] = status in ['healthy', 'degraded']
                        print(f"  [OK] {service}: {status}")
                    else:
                        results[service] = False
                        print(f"  [FAIL] {service}: HTTP {resp.status}")
            except Exception as e:
                results[service] = False
                print(f"  [ERROR] {service}: {str(e)}")

        healthy_count = sum(1 for h in results.values() if h)
        print(f"  Services functional: {healthy_count}/{len(results)}")
        return results

    async def test_memory_storage(self) -> bool:
        """Test memory storage from different microservices"""
        print("\nTesting cross-microservice memory storage...")
        
        # Test memories from different microservices
        test_memories = [
            {
                "content": "User engagement analysis shows 85% retention rate for new dashboard feature",
                "context": {
                    "source": "surfacing",
                    "type": "analysis",
                    "timestamp": time.time()
                },
                "metadata": {
                    "feature": "new_dashboard",
                    "metric": "retention_rate",
                    "value": 0.85
                }
            },
            {
                "content": "Causal analysis reveals marketing campaign increased conversions by 23%",
                "context": {
                    "source": "causal",
                    "type": "insight", 
                    "timestamp": time.time()
                },
                "metadata": {
                    "campaign": "summer_2024",
                    "metric": "conversion_rate",
                    "lift": 0.23
                }
            },
            {
                "content": "LLM analysis indicates customer sentiment improved by 15% after product update",
                "context": {
                    "source": "llm",
                    "type": "sentiment_analysis",
                    "timestamp": time.time()
                },
                "metadata": {
                    "product": "core_platform",
                    "sentiment_change": 0.15,
                    "analysis_type": "customer_feedback"
                }
            }
        ]
        
        stored_ids = []
        
        # Store memories
        for memory in test_memories:
            try:
                async with self.session.post(
                    f"{self.base_urls['memory']}/store",
                    json=memory,
                    headers={
                        "X-User-Id": self.test_user_id,
                        "X-Org-Id": self.test_org_id,
                        "Content-Type": "application/json"
                    }
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        stored_ids.append(data['data']['memory_id'])
                        print(f"  [OK] Stored memory from {memory['context']['source']}: {data['data']['memory_id']}")
                    else:
                        print(f"  [FAIL] Failed to store memory from {memory['context']['source']}: HTTP {resp.status}")
                        return False
            except Exception as e:
                print(f"  [ERROR] Error storing memory: {str(e)}")
                return False
        
        print(f"  [SUCCESS] Stored {len(stored_ids)} memories across microservices")
        return len(stored_ids) == len(test_memories)

    async def run_storage_test(self) -> Dict[str, Any]:
        """Run basic KSE storage tests"""
        print("Starting KSE Cross-Microservice Storage Tests")
        print("=" * 60)
        
        start_time = time.time()
        
        # Test results
        results = {
            'timestamp': time.time(),
            'test_org_id': self.test_org_id,
            'test_user_id': self.test_user_id
        }
        
        # 1. Service Health Check
        health_results = await self.test_service_health()
        results['service_health'] = health_results
        
        # Only proceed if memory service is functional
        if not health_results.get('memory', False):
            print("\n[ABORT] Memory service not functional - aborting tests")
            results['overall_success'] = False
            return results
        
        # 2. Cross-microservice memory storage
        results['memory_storage'] = await self.test_memory_storage()
        
        # Calculate results
        duration = time.time() - start_time
        results['duration'] = duration
        
        # Determine overall success
        test_results = [results['memory_storage']]
        successful_tests = sum(1 for r in test_results if r)
        total_tests = len(test_results)
        
        results['successful_tests'] = successful_tests
        results['total_tests'] = total_tests
        results['success_rate'] = successful_tests / total_tests if total_tests > 0 else 0
        results['overall_success'] = successful_tests == total_tests
        
        # Print summary
        print("\n" + "=" * 60)
        print(f"Tests passed: {successful_tests}/{total_tests}")
        print(f"Success rate: {results['success_rate']:.1%}")
        print(f"Total duration: {results['duration']:.2f}s")
        
        if results['overall_success']:
            print("[SUCCESS] KSE cross-microservice storage test passed!")
        else:
            print("[PARTIAL] Some tests failed - check individual results")
            
        return results

async def main():
    """Main test execution"""
    try:
        async with KSEStorageTester() as tester:
            results = await tester.run_storage_test()
            
            # Save results to file
            with open('kse_storage_test_results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\nResults saved to: kse_storage_test_results.json")
            return results['overall_success']
            
    except Exception as e:
        print(f"[FATAL] Test execution failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)