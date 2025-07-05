import pytest
import httpx
from conftest import (
    TEST_BASE_URL, TEST_AUTH_URL, TEST_MEMORY_URL,
    TEST_REGISTRY_URL, assert_response_structure,
    assert_valid_uuid, assert_valid_timestamp
)

@pytest.mark.integration
class TestSystemIntegration:
    """Integration tests for the complete Lift OS system."""

    async def test_health_checks(self, http_client: httpx.AsyncClient, wait_for_services):
        """Test that all services are healthy."""
        services = [
            (TEST_BASE_URL, "gateway"),
            (TEST_AUTH_URL, "auth"),
            (TEST_MEMORY_URL, "memory"),
            (TEST_REGISTRY_URL, "registry")
        ]
        
        for service_url, service_name in services:
            response = await http_client.get(f"{service_url}/health")
            assert response.status_code == 200
            
            health_data = response.json()
            assert health_data["status"] == "healthy"
            assert health_data["service"] == service_name
            assert_valid_timestamp(health_data["timestamp"])

    async def test_user_registration_and_login_flow(self, http_client: httpx.AsyncClient):
        """Test complete user registration and login flow."""
        # Register new user
        register_data = {
            "email": "integration_test@example.com",
            "password": "testpassword123"
        }
        
        response = await http_client.post(f"{TEST_AUTH_URL}/api/v1/auth/register", json=register_data)
        assert response.status_code in [200, 409]  # 409 if user already exists
        
        # Login
        login_data = {
            "email": "integration_test@example.com",
            "password": "testpassword123"
        }
        
        response = await http_client.post(f"{TEST_AUTH_URL}/api/v1/auth/login", json=login_data)
        assert response.status_code == 200
        
        auth_data = response.json()
        assert_response_structure(auth_data, ["access_token", "token_type", "user"])
        assert auth_data["token_type"] == "bearer"
        assert auth_data["user"]["email"] == "integration_test@example.com"

    async def test_memory_integration_flow(self, http_client: httpx.AsyncClient, auth_headers: dict):
        """Test memory service integration with authentication."""
        # Create memory
        memory_data = {
            "content": "Integration test memory content",
            "metadata": {
                "type": "integration_test",
                "test_case": "memory_flow"
            },
            "tags": ["integration", "test", "memory"]
        }
        
        response = await http_client.post(
            f"{TEST_MEMORY_URL}/api/v1/memories",
            json=memory_data,
            headers=auth_headers
        )
        assert response.status_code == 200
        
        created_memory = response.json()
        assert_response_structure(created_memory, ["id", "content", "metadata", "tags", "created_at"])
        assert_valid_uuid(created_memory["id"])
        
        memory_id = created_memory["id"]
        
        # Retrieve memory
        response = await http_client.get(
            f"{TEST_MEMORY_URL}/api/v1/memories/{memory_id}",
            headers=auth_headers
        )
        assert response.status_code == 200
        
        retrieved_memory = response.json()
        assert retrieved_memory["content"] == memory_data["content"]
        assert retrieved_memory["metadata"] == memory_data["metadata"]

    async def test_module_registration_flow(self, http_client: httpx.AsyncClient, auth_headers: dict):
        """Test module registration with registry service."""
        module_data = {
            "name": "test-integration-module",
            "version": "1.0.0",
            "description": "Integration test module",
            "status": "active",
            "endpoints": ["/health", "/api/v1/test"],
            "permissions": ["test:read", "test:write"],
            "ui_components": [
                {
                    "name": "TestDashboard",
                    "type": "dashboard",
                    "path": "/test/dashboard"
                }
            ]
        }
        
        response = await http_client.post(
            f"{TEST_REGISTRY_URL}/api/v1/modules/register",
            json=module_data,
            headers=auth_headers
        )
        assert response.status_code == 200
        
        registration_result = response.json()
        assert_response_structure(registration_result, ["module_id", "status", "registered_at"])
        assert registration_result["status"] == "registered"
        
        module_id = registration_result["module_id"]
        
        # Verify module is listed
        response = await http_client.get(
            f"{TEST_REGISTRY_URL}/api/v1/modules",
            headers=auth_headers
        )
        assert response.status_code == 200
        
        modules_list = response.json()
        assert "modules" in modules_list
        
        # Find our test module
        test_module = None
        for module in modules_list["modules"]:
            if module["name"] == "test-integration-module":
                test_module = module
                break
        
        assert test_module is not None
        assert test_module["version"] == "1.0.0"
        assert test_module["status"] == "active"

    async def test_cross_service_data_flow(self, http_client: httpx.AsyncClient, auth_headers: dict):
        """Test data flow between multiple services."""
        # 1. Create a memory context
        memory_data = {
            "content": "Cross-service integration test data",
            "metadata": {
                "type": "cross_service_test",
                "integration_id": "test_123"
            },
            "tags": ["cross-service", "integration"]
        }
        
        memory_response = await http_client.post(
            f"{TEST_MEMORY_URL}/api/v1/memories",
            json=memory_data,
            headers=auth_headers
        )
        assert memory_response.status_code == 200
        memory_result = memory_response.json()
        memory_id = memory_result["id"]
        
        # 2. Register a module that references this memory
        module_data = {
            "name": "cross-service-test-module",
            "version": "1.0.0",
            "description": "Module for cross-service testing",
            "status": "active",
            "memory_contexts": [memory_id],
            "endpoints": ["/health"],
            "permissions": ["memory:read"]
        }
        
        module_response = await http_client.post(
            f"{TEST_REGISTRY_URL}/api/v1/modules/register",
            json=module_data,
            headers=auth_headers
        )
        assert module_response.status_code == 200
        
        # 3. Verify the module can access the memory context
        response = await http_client.get(
            f"{TEST_MEMORY_URL}/api/v1/memories/{memory_id}",
            headers=auth_headers
        )
        assert response.status_code == 200
        
        retrieved_memory = response.json()
        assert retrieved_memory["content"] == memory_data["content"]

    async def test_error_handling_and_resilience(self, http_client: httpx.AsyncClient, auth_headers: dict):
        """Test system error handling and resilience."""
        # Test invalid memory ID
        response = await http_client.get(
            f"{TEST_MEMORY_URL}/api/v1/memories/invalid-uuid",
            headers=auth_headers
        )
        assert response.status_code == 422  # Validation error
        
        # Test non-existent memory
        response = await http_client.get(
            f"{TEST_MEMORY_URL}/api/v1/memories/00000000-0000-0000-0000-000000000000",
            headers=auth_headers
        )
        assert response.status_code == 404
        
        # Test unauthorized access
        response = await http_client.get(f"{TEST_MEMORY_URL}/api/v1/memories")
        assert response.status_code == 401
        
        # Test invalid module registration
        invalid_module_data = {
            "name": "",  # Invalid empty name
            "version": "invalid-version"
        }
        
        response = await http_client.post(
            f"{TEST_REGISTRY_URL}/api/v1/modules/register",
            json=invalid_module_data,
            headers=auth_headers
        )
        assert response.status_code == 422

@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceIntegration:
    """Performance-related integration tests."""

    async def test_concurrent_memory_operations(self, http_client: httpx.AsyncClient, auth_headers: dict):
        """Test concurrent memory operations."""
        import asyncio
        
        async def create_memory(index: int):
            memory_data = {
                "content": f"Concurrent test memory {index}",
                "metadata": {"test_index": index},
                "tags": ["concurrent", "performance"]
            }
            
            response = await http_client.post(
                f"{TEST_MEMORY_URL}/api/v1/memories",
                json=memory_data,
                headers=auth_headers
            )
            return response.status_code == 200
        
        # Create 10 memories concurrently
        tasks = [create_memory(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All operations should succeed
        assert all(results), "Some concurrent memory operations failed"

    async def test_system_load_handling(self, http_client: httpx.AsyncClient, auth_headers: dict):
        """Test system behavior under load."""
        import time
        
        start_time = time.time()
        
        # Perform multiple operations
        operations = []
        
        # Health checks
        for _ in range(5):
            operations.append(http_client.get(f"{TEST_BASE_URL}/health"))
        
        # Memory operations
        for i in range(5):
            memory_data = {
                "content": f"Load test memory {i}",
                "metadata": {"load_test": True},
                "tags": ["load", "test"]
            }
            operations.append(
                http_client.post(
                    f"{TEST_MEMORY_URL}/api/v1/memories",
                    json=memory_data,
                    headers=auth_headers
                )
            )
        
        # Execute all operations
        responses = await asyncio.gather(*operations, return_exceptions=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Check that operations completed in reasonable time (< 10 seconds)
        assert duration < 10, f"Load test took too long: {duration} seconds"
        
        # Check that most operations succeeded
        successful_responses = [r for r in responses if hasattr(r, 'status_code') and r.status_code < 400]
        success_rate = len(successful_responses) / len(responses)
        assert success_rate > 0.8, f"Success rate too low: {success_rate}"