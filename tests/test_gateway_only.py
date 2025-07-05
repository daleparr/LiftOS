"""
Simple test for gateway service only
"""
import pytest
import httpx
import asyncio

@pytest.mark.asyncio
async def test_gateway_health():
    """Test gateway health endpoint"""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get("http://localhost:8000/health")
            assert response.status_code == 200
            
            health_data = response.json()
            assert health_data.get('status') == 'healthy'
            assert 'timestamp' in health_data
            assert 'version' in health_data
            
            print(f"✓ Gateway health check passed: {health_data}")
            
        except Exception as e:
            pytest.fail(f"Gateway health check failed: {e}")

@pytest.mark.asyncio
async def test_gateway_docs():
    """Test gateway documentation endpoint"""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get("http://localhost:8000/docs")
            assert response.status_code == 200
            assert "swagger-ui" in response.text.lower()
            
            print("✓ Gateway documentation endpoint accessible")
            
        except Exception as e:
            pytest.fail(f"Gateway docs test failed: {e}")

@pytest.mark.asyncio
async def test_gateway_openapi():
    """Test gateway OpenAPI schema"""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get("http://localhost:8000/openapi.json")
            assert response.status_code == 200
            
            schema = response.json()
            assert "openapi" in schema
            assert "paths" in schema
            assert "/health" in schema["paths"]
            
            print(f"✓ Gateway OpenAPI schema valid with {len(schema['paths'])} endpoints")
            
        except Exception as e:
            pytest.fail(f"Gateway OpenAPI test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])