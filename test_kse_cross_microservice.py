#!/usr/bin/env python3
"""
KSE Memory System Cross-Microservice Integration Test
Tests the Next Phase capabilities:
1. Cross-microservice memory sharing
2. Intelligence coordination between services  
3. Organizational context isolation
4. Real-time memory analytics
5. Hybrid search operations
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Any
import uuid

class KSECrossServiceTester:
    def __init__(self):
        self.base_urls = {
            'memory': 'http://localhost:8003',
            'surfacing': 'http://localhost:9005',
            'causal': 'http://localhost:8008', 
            'llm': 'http://localhost:8009'
        }
        self.test_org_id = "test-org-kse"
        self.test_user_id = "test-user-kse"
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def test_service_health(self) -> Dict[str, bool]:
        """Test health of all services"""
        print("🔍 Testing service health...")
        results = {}
        
        for service, url in self.base_urls.items():
            try:
                async with self.session.get(f"{url}/health", timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results[service] = data.get('status') == 'healthy'
                        print(f"  ✅ {service}: {data.get('status', 'unknown')}")
                    else:
                        results[service] = False
                        print(f"  ❌ {service}: HTTP {resp.status}")
            except Exception as e:
                results[service] = False
                print(f"  ❌ {service}: {str(e)}")
                
        return results

    async def test_memory_storage_and_retrieval(self) -> bool:
        """Test 1: Cross-microservice memory sharing"""
        print("\n📝 Test 1: Cross-microservice memory sharing")
        
        # Store memories from different microservices
        test_memories = [
            {
                "content": "User prefers technical documentation with code examples",
                "memory_type": "preference",
                "context": {"source": "surfacing", "interaction_type": "documentation_request"}
            },
            {
                "content": "Causal analysis shows strong correlation between user engagement and interactive examples",
                "memory_type": "insight", 
                "context": {"source": "causal", "analysis_type": "engagement_correlation"}
            },
            {
                "content": "LLM generated high-quality Python code snippet for data processing",
                "memory_type": "generation",
                "context": {"source": "llm", "language": "python", "task": "data_processing"}
            }
        ]
        
        stored_ids = []
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
                        print(f"  ✅ Stored memory from {memory['context']['source']}: {data['data']['memory_id']}")
                    else:
                        print(f"  ❌ Failed to store memory from {memory['context']['source']}: HTTP {resp.status}")
                        return False
            except Exception as e:
                print(f"  ❌ Error storing memory: {str(e)}")
                return False
        
        # Test retrieval across services
        search_query = {
            "query": "user preferences and code examples",
            "search_type": "hybrid",
            "limit": 10
        }
        
        try:
            async with self.session.post(
                f"{self.base_urls['memory']}/search",
                json=search_query,
                headers={
                    "X-User-Id": self.test_user_id,
                    "X-Org-Id": self.test_org_id,
                    "Content-Type": "application/json"
                }
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results = data['data']['results']
                    print(f"  ✅ Retrieved {len(results)} memories via hybrid search")
                    
                    # Verify cross-service memories are found
                    sources = [r.get('context', {}).get('source') for r in results]
                    unique_sources = set(filter(None, sources))
                    print(f"  📊 Found memories from sources: {unique_sources}")
                    
                    return len(unique_sources) >= 2  # At least 2 different sources
                else:
                    print(f"  ❌ Search failed: HTTP {resp.status}")
                    return False
        except Exception as e:
            print(f"  ❌ Search error: {str(e)}")
            return False

    async def test_organizational_isolation(self) -> bool:
        """Test 3: Organizational context isolation"""
        print("\n🏢 Test 3: Organizational context isolation")
        
        # Store memory for org1
        org1_memory = {
            "content": "Confidential org1 business strategy",
            "memory_type": "confidential",
            "context": {"classification": "internal"}
        }
        
        # Store memory for org2  
        org2_memory = {
            "content": "Confidential org2 business strategy", 
            "memory_type": "confidential",
            "context": {"classification": "internal"}
        }
        
        try:
            # Store for org1
            async with self.session.post(
                f"{self.base_urls['memory']}/store",
                json=org1_memory,
                headers={
                    "X-User-Id": "user1",
                    "X-Org-Id": "org1",
                    "Content-Type": "application/json"
                }
            ) as resp:
                if resp.status != 200:
                    print(f"  ❌ Failed to store org1 memory: HTTP {resp.status}")
                    return False
                org1_data = await resp.json()
                print(f"  ✅ Stored org1 memory: {org1_data['data']['memory_id']}")
            
            # Store for org2
            async with self.session.post(
                f"{self.base_urls['memory']}/store",
                json=org2_memory,
                headers={
                    "X-User-Id": "user2", 
                    "X-Org-Id": "org2",
                    "Content-Type": "application/json"
                }
            ) as resp:
                if resp.status != 200:
                    print(f"  ❌ Failed to store org2 memory: HTTP {resp.status}")
                    return False
                org2_data = await resp.json()
                print(f"  ✅ Stored org2 memory: {org2_data['data']['memory_id']}")
            
            # Search from org1 - should only see org1 data
            search_query = {
                "query": "business strategy",
                "search_type": "neural",
                "limit": 10
            }
            
            async with self.session.post(
                f"{self.base_urls['memory']}/search",
                json=search_query,
                headers={
                    "X-User-Id": "user1",
                    "X-Org-Id": "org1", 
                    "Content-Type": "application/json"
                }
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results = data['data']['results']
                    
                    # Verify only org1 results
                    org1_results = [r for r in results if "org1" in r.get('content', '')]
                    org2_results = [r for r in results if "org2" in r.get('content', '')]
                    
                    print(f"  📊 Org1 search found {len(org1_results)} org1 results, {len(org2_results)} org2 results")
                    
                    return len(org1_results) > 0 and len(org2_results) == 0
                else:
                    print(f"  ❌ Org1 search failed: HTTP {resp.status}")
                    return False
                    
        except Exception as e:
            print(f"  ❌ Isolation test error: {str(e)}")
            return False

    async def test_memory_analytics(self) -> bool:
        """Test 4: Real-time memory analytics"""
        print("\n📊 Test 4: Real-time memory analytics")
        
        try:
            async with self.session.get(
                f"{self.base_urls['memory']}/analytics",
                headers={
                    "X-User-Id": self.test_user_id,
                    "X-Org-Id": self.test_org_id
                }
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    analytics = data['data']
                    
                    print(f"  ✅ Analytics retrieved successfully")
                    print(f"  📈 Total memories: {analytics.get('total_memories', 0)}")
                    print(f"  📈 Memory types: {analytics.get('memory_types', {})}")
                    print(f"  📈 Recent activity: {analytics.get('recent_activity', 0)}")
                    
                    # Verify analytics contain expected data
                    return (
                        'total_memories' in analytics and
                        'memory_types' in analytics and
                        analytics['total_memories'] > 0
                    )
                else:
                    print(f"  ❌ Analytics failed: HTTP {resp.status}")
                    return False
        except Exception as e:
            print(f"  ❌ Analytics error: {str(e)}")
            return False

    async def test_hybrid_search_operations(self) -> bool:
        """Test 5: Hybrid search operations"""
        print("\n🔍 Test 5: Hybrid search operations")
        
        search_types = ['neural', 'conceptual', 'knowledge', 'hybrid']
        results = {}
        
        for search_type in search_types:
            try:
                search_query = {
                    "query": "technical documentation with examples",
                    "search_type": search_type,
                    "limit": 5
                }
                
                async with self.session.post(
                    f"{self.base_urls['memory']}/search",
                    json=search_query,
                    headers={
                        "X-User-Id": self.test_user_id,
                        "X-Org-Id": self.test_org_id,
                        "Content-Type": "application/json"
                    }
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        result_count = data['data']['count']
                        duration = data['data']['duration']
                        results[search_type] = {
                            'count': result_count,
                            'duration': duration,
                            'success': True
                        }
                        print(f"  ✅ {search_type} search: {result_count} results in {duration:.3f}s")
                    else:
                        results[search_type] = {'success': False}
                        print(f"  ❌ {search_type} search failed: HTTP {resp.status}")
            except Exception as e:
                results[search_type] = {'success': False}
                print(f"  ❌ {search_type} search error: {str(e)}")
        
        # Verify all search types work
        successful_searches = sum(1 for r in results.values() if r.get('success'))
        print(f"  📊 {successful_searches}/{len(search_types)} search types successful")
        
        return successful_searches == len(search_types)

    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all KSE cross-microservice tests"""
        print("🚀 Starting KSE Cross-Microservice Integration Tests")
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
        
        # Only proceed if memory service is healthy
        if not health_results.get('memory', False):
            print("\n❌ Memory service not healthy - aborting tests")
            results['overall_success'] = False
            return results
        
        # 2. Cross-microservice memory sharing
        results['memory_sharing'] = await self.test_memory_storage_and_retrieval()
        
        # 3. Organizational isolation
        results['org_isolation'] = await self.test_organizational_isolation()
        
        # 4. Memory analytics
        results['memory_analytics'] = await self.test_memory_analytics()
        
        # 5. Hybrid search operations
        results['hybrid_search'] = await self.test_hybrid_search_operations()
        
        # Calculate overall success
        test_results = [
            results['memory_sharing'],
            results['org_isolation'], 
            results['memory_analytics'],
            results['hybrid_search']
        ]
        
        successful_tests = sum(1 for r in test_results if r)
        total_tests = len(test_results)
        
        results['successful_tests'] = successful_tests
        results['total_tests'] = total_tests
        results['success_rate'] = successful_tests / total_tests
        results['overall_success'] = successful_tests == total_tests
        results['duration'] = time.time() - start_time
        
        # Print summary
        print("\n" + "=" * 60)
        print("🎯 KSE Cross-Microservice Test Summary")
        print("=" * 60)
        print(f"✅ Tests passed: {successful_tests}/{total_tests}")
        print(f"📊 Success rate: {results['success_rate']:.1%}")
        print(f"⏱️  Total duration: {results['duration']:.2f}s")
        
        if results['overall_success']:
            print("🎉 ALL TESTS PASSED - KSE Memory System is fully operational!")
        else:
            print("⚠️  Some tests failed - review results above")
            
        return results

async def main():
    """Main test execution"""
    async with KSECrossServiceTester() as tester:
        results = await tester.run_comprehensive_test()
        
        # Save results
        with open('kse_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Results saved to kse_test_results.json")
        
        return results['overall_success']

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)