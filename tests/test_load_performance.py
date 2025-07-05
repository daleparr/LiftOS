#!/usr/bin/env python3
"""
Load testing for gateway performance and KSE scalability
Tests system performance under various load conditions
"""

import pytest
import asyncio
import aiohttp
import time
import statistics
import json
import os
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import threading

# Test configuration
LOAD_TEST_BASE_URL = os.getenv("LOAD_TEST_BASE_URL", "http://localhost:8000")
LOAD_TEST_AUTH_URL = os.getenv("LOAD_TEST_AUTH_URL", "http://localhost:8001")
LOAD_TEST_MEMORY_URL = os.getenv("LOAD_TEST_MEMORY_URL", "http://localhost:8003")

@pytest.mark.slow
@pytest.mark.load
class TestGatewayPerformance:
    """Load tests for Gateway service performance"""
    
    async def test_gateway_concurrent_requests(self):
        """Test gateway performance under concurrent load"""
        concurrent_requests = 50
        total_requests = 200
        
        async def make_request(session: aiohttp.ClientSession, request_id: int):
            """Make a single request and measure response time"""
            start_time = time.time()
            try:
                async with session.get(f"{LOAD_TEST_BASE_URL}/health") as response:
                    await response.json()
                    end_time = time.time()
                    return {
                        "request_id": request_id,
                        "status_code": response.status,
                        "response_time": end_time - start_time,
                        "success": response.status == 200
                    }
            except Exception as e:
                end_time = time.time()
                return {
                    "request_id": request_id,
                    "status_code": 0,
                    "response_time": end_time - start_time,
                    "success": False,
                    "error": str(e)
                }
        
        # Execute load test
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=50)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(concurrent_requests)
            
            async def bounded_request(request_id: int):
                async with semaphore:
                    return await make_request(session, request_id)
            
            # Execute all requests
            start_time = time.time()
            tasks = [bounded_request(i) for i in range(total_requests)]
            results = await asyncio.gather(*tasks)
            end_time = time.time()
        
        # Analyze results
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]
        
        response_times = [r["response_time"] for r in successful_requests]
        
        # Performance assertions
        success_rate = len(successful_requests) / total_requests
        assert success_rate >= 0.95, f"Success rate should be >= 95%, got {success_rate:.2%}"
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
            
            assert avg_response_time < 1.0, f"Average response time should be < 1s, got {avg_response_time:.3f}s"
            assert p95_response_time < 2.0, f"95th percentile response time should be < 2s, got {p95_response_time:.3f}s"
        
        total_time = end_time - start_time
        throughput = total_requests / total_time
        
        print(f"\n=== Gateway Load Test Results ===")
        print(f"Total Requests: {total_requests}")
        print(f"Concurrent Requests: {concurrent_requests}")
        print(f"Success Rate: {success_rate:.2%}")
        print(f"Failed Requests: {len(failed_requests)}")
        print(f"Average Response Time: {avg_response_time:.3f}s")
        print(f"95th Percentile Response Time: {p95_response_time:.3f}s")
        print(f"Throughput: {throughput:.2f} requests/second")
    
    async def test_gateway_sustained_load(self):
        """Test gateway performance under sustained load"""
        duration_seconds = 30
        requests_per_second = 10
        
        results = []
        start_time = time.time()
        
        async def sustained_request_loop():
            """Make requests at specified rate for duration"""
            connector = aiohttp.TCPConnector(limit=20)
            timeout = aiohttp.ClientTimeout(total=10)
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                request_count = 0
                while time.time() - start_time < duration_seconds:
                    request_start = time.time()
                    
                    try:
                        async with session.get(f"{LOAD_TEST_BASE_URL}/health") as response:
                            await response.json()
                            request_end = time.time()
                            
                            results.append({
                                "timestamp": request_start,
                                "response_time": request_end - request_start,
                                "status_code": response.status,
                                "success": response.status == 200
                            })
                    except Exception as e:
                        request_end = time.time()
                        results.append({
                            "timestamp": request_start,
                            "response_time": request_end - request_start,
                            "status_code": 0,
                            "success": False,
                            "error": str(e)
                        })
                    
                    request_count += 1
                    
                    # Rate limiting
                    elapsed = time.time() - request_start
                    sleep_time = (1.0 / requests_per_second) - elapsed
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
        
        # Run sustained load test
        await sustained_request_loop()
        
        # Analyze sustained load results
        successful_results = [r for r in results if r["success"]]
        success_rate = len(successful_results) / len(results) if results else 0
        
        assert success_rate >= 0.98, f"Sustained load success rate should be >= 98%, got {success_rate:.2%}"
        
        if successful_results:
            response_times = [r["response_time"] for r in successful_results]
            avg_response_time = statistics.mean(response_times)
            
            assert avg_response_time < 0.5, f"Sustained load avg response time should be < 0.5s, got {avg_response_time:.3f}s"
        
        print(f"\n=== Gateway Sustained Load Results ===")
        print(f"Duration: {duration_seconds}s")
        print(f"Target Rate: {requests_per_second} req/s")
        print(f"Total Requests: {len(results)}")
        print(f"Success Rate: {success_rate:.2%}")
        print(f"Average Response Time: {avg_response_time:.3f}s")

@pytest.mark.slow
@pytest.mark.load
@pytest.mark.memory
class TestMemoryServicePerformance:
    """Load tests for Memory service and KSE performance"""
    
    async def test_memory_storage_load(self):
        """Test memory storage performance under load"""
        concurrent_operations = 20
        total_operations = 100
        
        async def store_memory(session: aiohttp.ClientSession, operation_id: int):
            """Store a memory item and measure performance"""
            memory_data = {
                "key": f"load_test_memory_{operation_id}",
                "value": {
                    "content": f"Load test memory content {operation_id}",
                    "metadata": {"test_id": operation_id, "type": "load_test"}
                },
                "context": {
                    "user_id": f"user_{operation_id % 10}",
                    "org_id": "load_test_org"
                }
            }
            
            start_time = time.time()
            try:
                async with session.post(
                    f"{LOAD_TEST_MEMORY_URL}/memory/store",
                    json=memory_data
                ) as response:
                    await response.json()
                    end_time = time.time()
                    
                    return {
                        "operation_id": operation_id,
                        "operation_type": "store",
                        "status_code": response.status,
                        "response_time": end_time - start_time,
                        "success": response.status in [200, 201]
                    }
            except Exception as e:
                end_time = time.time()
                return {
                    "operation_id": operation_id,
                    "operation_type": "store",
                    "status_code": 0,
                    "response_time": end_time - start_time,
                    "success": False,
                    "error": str(e)
                }
        
        # Execute memory storage load test
        connector = aiohttp.TCPConnector(limit=50)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            semaphore = asyncio.Semaphore(concurrent_operations)
            
            async def bounded_store(operation_id: int):
                async with semaphore:
                    return await store_memory(session, operation_id)
            
            tasks = [bounded_store(i) for i in range(total_operations)]
            results = await asyncio.gather(*tasks)
        
        # Analyze memory storage results
        successful_operations = [r for r in results if r["success"]]
        success_rate = len(successful_operations) / total_operations
        
        assert success_rate >= 0.90, f"Memory storage success rate should be >= 90%, got {success_rate:.2%}"
        
        if successful_operations:
            response_times = [r["response_time"] for r in successful_operations]
            avg_response_time = statistics.mean(response_times)
            
            assert avg_response_time < 2.0, f"Memory storage avg response time should be < 2s, got {avg_response_time:.3f}s"
        
        print(f"\n=== Memory Storage Load Results ===")
        print(f"Total Operations: {total_operations}")
        print(f"Success Rate: {success_rate:.2%}")
        print(f"Average Response Time: {avg_response_time:.3f}s")
    
    async def test_kse_search_performance(self):
        """Test KSE search performance under load"""
        concurrent_searches = 15
        total_searches = 75
        
        search_queries = [
            "test content search",
            "memory retrieval query",
            "knowledge base lookup",
            "document search test",
            "semantic search query"
        ]
        
        async def perform_search(session: aiohttp.ClientSession, search_id: int):
            """Perform a KSE search and measure performance"""
            query = search_queries[search_id % len(search_queries)]
            search_data = {
                "query": f"{query} {search_id}",
                "context": {
                    "user_id": f"user_{search_id % 5}",
                    "org_id": "load_test_org"
                },
                "filters": {"type": "load_test"},
                "limit": 10
            }
            
            start_time = time.time()
            try:
                async with session.post(
                    f"{LOAD_TEST_MEMORY_URL}/memory/search",
                    json=search_data
                ) as response:
                    result = await response.json()
                    end_time = time.time()
                    
                    return {
                        "search_id": search_id,
                        "operation_type": "search",
                        "status_code": response.status,
                        "response_time": end_time - start_time,
                        "success": response.status == 200,
                        "result_count": len(result.get("results", [])) if isinstance(result, dict) else 0
                    }
            except Exception as e:
                end_time = time.time()
                return {
                    "search_id": search_id,
                    "operation_type": "search",
                    "status_code": 0,
                    "response_time": end_time - start_time,
                    "success": False,
                    "error": str(e)
                }
        
        # Execute KSE search load test
        connector = aiohttp.TCPConnector(limit=30)
        timeout = aiohttp.ClientTimeout(total=20)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            semaphore = asyncio.Semaphore(concurrent_searches)
            
            async def bounded_search(search_id: int):
                async with semaphore:
                    return await perform_search(session, search_id)
            
            tasks = [bounded_search(i) for i in range(total_searches)]
            results = await asyncio.gather(*tasks)
        
        # Analyze KSE search results
        successful_searches = [r for r in results if r["success"]]
        success_rate = len(successful_searches) / total_searches
        
        assert success_rate >= 0.85, f"KSE search success rate should be >= 85%, got {success_rate:.2%}"
        
        if successful_searches:
            response_times = [r["response_time"] for r in successful_searches]
            avg_response_time = statistics.mean(response_times)
            
            assert avg_response_time < 3.0, f"KSE search avg response time should be < 3s, got {avg_response_time:.3f}s"
        
        print(f"\n=== KSE Search Load Results ===")
        print(f"Total Searches: {total_searches}")
        print(f"Success Rate: {success_rate:.2%}")
        print(f"Average Response Time: {avg_response_time:.3f}s")

@pytest.mark.slow
@pytest.mark.load
class TestSystemScalability:
    """System-wide scalability tests"""
    
    async def test_mixed_workload_performance(self):
        """Test system performance under mixed workload"""
        duration_seconds = 60
        
        # Define workload mix
        workload_config = {
            "health_checks": {"rate": 5, "weight": 0.3},
            "memory_operations": {"rate": 2, "weight": 0.4},
            "search_operations": {"rate": 1, "weight": 0.3}
        }
        
        results = {"health_checks": [], "memory_operations": [], "search_operations": []}
        
        async def health_check_worker():
            """Worker for health check requests"""
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                while time.time() - start_time < duration_seconds:
                    try:
                        async with session.get(f"{LOAD_TEST_BASE_URL}/health") as response:
                            await response.json()
                            results["health_checks"].append({
                                "success": response.status == 200,
                                "response_time": time.time() - start_time
                            })
                    except:
                        results["health_checks"].append({"success": False})
                    
                    await asyncio.sleep(1.0 / workload_config["health_checks"]["rate"])
        
        async def memory_worker():
            """Worker for memory operations"""
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                operation_id = 0
                while time.time() - start_time < duration_seconds:
                    try:
                        memory_data = {
                            "key": f"mixed_load_memory_{operation_id}",
                            "value": {"content": f"Mixed load test {operation_id}"},
                            "context": {"user_id": "mixed_load_user"}
                        }
                        async with session.post(
                            f"{LOAD_TEST_MEMORY_URL}/memory/store",
                            json=memory_data
                        ) as response:
                            results["memory_operations"].append({
                                "success": response.status in [200, 201],
                                "response_time": time.time() - start_time
                            })
                    except:
                        results["memory_operations"].append({"success": False})
                    
                    operation_id += 1
                    await asyncio.sleep(1.0 / workload_config["memory_operations"]["rate"])
        
        async def search_worker():
            """Worker for search operations"""
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                search_id = 0
                while time.time() - start_time < duration_seconds:
                    try:
                        search_data = {
                            "query": f"mixed load search {search_id}",
                            "context": {"user_id": "mixed_load_user"}
                        }
                        async with session.post(
                            f"{LOAD_TEST_MEMORY_URL}/memory/search",
                            json=search_data
                        ) as response:
                            results["search_operations"].append({
                                "success": response.status == 200,
                                "response_time": time.time() - start_time
                            })
                    except:
                        results["search_operations"].append({"success": False})
                    
                    search_id += 1
                    await asyncio.sleep(1.0 / workload_config["search_operations"]["rate"])
        
        # Run mixed workload
        await asyncio.gather(
            health_check_worker(),
            memory_worker(),
            search_worker()
        )
        
        # Analyze mixed workload results
        for operation_type, operation_results in results.items():
            if operation_results:
                successful_ops = [r for r in operation_results if r["success"]]
                success_rate = len(successful_ops) / len(operation_results)
                
                assert success_rate >= 0.80, f"{operation_type} success rate should be >= 80%, got {success_rate:.2%}"
                
                print(f"\n=== {operation_type.title()} Results ===")
                print(f"Total Operations: {len(operation_results)}")
                print(f"Success Rate: {success_rate:.2%}")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "load"])