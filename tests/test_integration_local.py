"""
Local integration tests for Lift OS Core services
Tests services running locally without external dependencies
"""

import pytest
import pytest_asyncio
import asyncio
import httpx
import time
import json
from typing import Dict, Any

# Service endpoints for local testing
SERVICE_ENDPOINTS = {
    'gateway': 'http://localhost:8000',
    'auth': 'http://localhost:8001',
    'memory': 'http://localhost:8002',
    'registry': 'http://localhost:8003',
    'billing': 'http://localhost:8004',
    'observability': 'http://localhost:8005'
}

class TestLocalIntegration:
    """Integration tests for local service deployment"""
    
    @pytest_asyncio.fixture(autouse=True)
    async def setup(self):
        """Setup for each test"""
        self.client = httpx.AsyncClient(timeout=30.0)
        yield
        await self.client.aclose()
    
    async def wait_for_service(self, service_name: str, max_retries: int = 30) -> bool:
        """Wait for a service to become available"""
        endpoint = SERVICE_ENDPOINTS[service_name]
        
        for attempt in range(max_retries):
            try:
                response = await self.client.get(f"{endpoint}/health")
                if response.status_code == 200:
                    print(f"✓ {service_name} service is ready")
                    return True
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
            
            await asyncio.sleep(1)
        
        print(f"✗ {service_name} service failed to start after {max_retries} seconds")
        return False
    
    @pytest.mark.asyncio
    async def test_service_health_checks(self):
        """Test that all services respond to health checks"""
        print("\n=== Testing Service Health Checks ===")
        
        for service_name, endpoint in SERVICE_ENDPOINTS.items():
            print(f"Checking {service_name} health...")
            
            # Wait for service to be ready
            is_ready = await self.wait_for_service(service_name)
            assert is_ready, f"{service_name} service is not responding"
            
            # Test health endpoint
            response = await self.client.get(f"{endpoint}/health")
            assert response.status_code == 200
            
            health_data = response.json()
            assert health_data.get('status') == 'healthy'
            assert 'service' in health_data
            assert 'timestamp' in health_data
            
            print(f"✓ {service_name} health check passed")
    
    @pytest.mark.asyncio
    async def test_service_info_endpoints(self):
        """Test service info endpoints"""
        print("\n=== Testing Service Info Endpoints ===")
        
        for service_name, endpoint in SERVICE_ENDPOINTS.items():
            print(f"Testing {service_name} info endpoint...")
            
            # Test info endpoint
            response = await self.client.get(f"{endpoint}/info")
            assert response.status_code == 200
            
            info_data = response.json()
            assert 'service_name' in info_data
            assert 'version' in info_data
            assert info_data['service_name'] == service_name
            
            print(f"✓ {service_name} info endpoint passed")
    
    @pytest.mark.asyncio
    async def test_gateway_routing(self):
        """Test gateway service routing capabilities"""
        print("\n=== Testing Gateway Routing ===")
        
        gateway_endpoint = SERVICE_ENDPOINTS['gateway']
        
        # Test gateway health
        response = await self.client.get(f"{gateway_endpoint}/health")
        assert response.status_code == 200
        print("✓ Gateway health check passed")
        
        # Test gateway info
        response = await self.client.get(f"{gateway_endpoint}/info")
        assert response.status_code == 200
        print("✓ Gateway info endpoint passed")
        
        # Test routes endpoint
        response = await self.client.get(f"{gateway_endpoint}/routes")
        assert response.status_code == 200
        
        routes_data = response.json()
        assert 'routes' in routes_data
        print(f"✓ Gateway routes endpoint passed ({len(routes_data['routes'])} routes)")
    
    @pytest.mark.asyncio
    async def test_auth_service_basic(self):
        """Test basic auth service functionality"""
        print("\n=== Testing Auth Service ===")
        
        auth_endpoint = SERVICE_ENDPOINTS['auth']
        
        # Test auth service health
        response = await self.client.get(f"{auth_endpoint}/health")
        assert response.status_code == 200
        print("✓ Auth service health check passed")
        
        # Test auth endpoints
        response = await self.client.get(f"{auth_endpoint}/auth/status")
        assert response.status_code in [200, 401]  # Either OK or unauthorized
        print("✓ Auth status endpoint accessible")
    
    @pytest.mark.asyncio
    async def test_memory_service_basic(self):
        """Test basic memory service functionality"""
        print("\n=== Testing Memory Service ===")
        
        memory_endpoint = SERVICE_ENDPOINTS['memory']
        
        # Test memory service health
        response = await self.client.get(f"{memory_endpoint}/health")
        assert response.status_code == 200
        print("✓ Memory service health check passed")
        
        # Test memory status
        response = await self.client.get(f"{memory_endpoint}/memory/status")
        assert response.status_code == 200
        
        status_data = response.json()
        assert 'status' in status_data
        print("✓ Memory status endpoint passed")
    
    @pytest.mark.asyncio
    async def test_registry_service_basic(self):
        """Test basic registry service functionality"""
        print("\n=== Testing Registry Service ===")
        
        registry_endpoint = SERVICE_ENDPOINTS['registry']
        
        # Test registry service health
        response = await self.client.get(f"{registry_endpoint}/health")
        assert response.status_code == 200
        print("✓ Registry service health check passed")
        
        # Test modules endpoint
        response = await self.client.get(f"{registry_endpoint}/modules")
        assert response.status_code == 200
        
        modules_data = response.json()
        assert 'modules' in modules_data
        print(f"✓ Registry modules endpoint passed ({len(modules_data['modules'])} modules)")
    
    @pytest.mark.asyncio
    async def test_billing_service_basic(self):
        """Test basic billing service functionality"""
        print("\n=== Testing Billing Service ===")
        
        billing_endpoint = SERVICE_ENDPOINTS['billing']
        
        # Test billing service health
        response = await self.client.get(f"{billing_endpoint}/health")
        assert response.status_code == 200
        print("✓ Billing service health check passed")
        
        # Test billing status
        response = await self.client.get(f"{billing_endpoint}/billing/status")
        assert response.status_code == 200
        
        status_data = response.json()
        assert 'status' in status_data
        print("✓ Billing status endpoint passed")
    
    @pytest.mark.asyncio
    async def test_observability_service_basic(self):
        """Test basic observability service functionality"""
        print("\n=== Testing Observability Service ===")
        
        observability_endpoint = SERVICE_ENDPOINTS['observability']
        
        # Test observability service health
        response = await self.client.get(f"{observability_endpoint}/health")
        assert response.status_code == 200
        print("✓ Observability service health check passed")
        
        # Test metrics endpoint
        response = await self.client.get(f"{observability_endpoint}/metrics")
        assert response.status_code == 200
        print("✓ Observability metrics endpoint passed")
    
    @pytest.mark.asyncio
    async def test_cross_service_communication(self):
        """Test communication between services"""
        print("\n=== Testing Cross-Service Communication ===")
        
        # Test gateway to auth service communication
        gateway_endpoint = SERVICE_ENDPOINTS['gateway']
        
        # Test gateway can reach auth service
        response = await self.client.get(f"{gateway_endpoint}/api/auth/status")
        assert response.status_code in [200, 401, 404]  # Service reachable
        print("✓ Gateway to Auth communication test passed")
        
        # Test gateway can reach memory service
        response = await self.client.get(f"{gateway_endpoint}/api/memory/status")
        assert response.status_code in [200, 401, 404]  # Service reachable
        print("✓ Gateway to Memory communication test passed")

class TestServicePerformance:
    """Basic performance tests for services"""
    
    @pytest_asyncio.fixture(autouse=True)
    async def setup(self):
        """Setup for each test"""
        self.client = httpx.AsyncClient(timeout=30.0)
        yield
        await self.client.aclose()
    
    @pytest.mark.asyncio
    async def test_service_response_times(self):
        """Test service response times"""
        print("\n=== Testing Service Response Times ===")
        
        for service_name, endpoint in SERVICE_ENDPOINTS.items():
            start_time = time.time()
            
            try:
                response = await self.client.get(f"{endpoint}/health")
                response_time = time.time() - start_time
                
                assert response.status_code == 200
                assert response_time < 5.0  # Should respond within 5 seconds
                
                print(f"✓ {service_name} response time: {response_time:.3f}s")
                
            except Exception as e:
                print(f"✗ {service_name} failed: {e}")
                pytest.fail(f"{service_name} performance test failed")
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test concurrent requests to services"""
        print("\n=== Testing Concurrent Requests ===")
        
        async def make_request(service_name: str, endpoint: str):
            """Make a single request"""
            try:
                response = await self.client.get(f"{endpoint}/health")
                return service_name, response.status_code == 200
            except Exception:
                return service_name, False
        
        # Create concurrent requests
        tasks = []
        for service_name, endpoint in SERVICE_ENDPOINTS.items():
            for _ in range(3):  # 3 concurrent requests per service
                tasks.append(make_request(service_name, endpoint))
        
        # Execute all requests concurrently
        results = await asyncio.gather(*tasks)
        
        # Check results
        success_count = sum(1 for _, success in results if success)
        total_count = len(results)
        
        success_rate = success_count / total_count
        assert success_rate >= 0.8  # At least 80% success rate
        
        print(f"✓ Concurrent requests: {success_count}/{total_count} successful ({success_rate:.1%})")

if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-s"])