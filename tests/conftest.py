import pytest
import asyncio
import httpx
from typing import AsyncGenerator, Generator
import os
import time

# Test configuration
TEST_BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8000")
TEST_AUTH_URL = os.getenv("TEST_AUTH_URL", "http://localhost:8001")
TEST_MEMORY_URL = os.getenv("TEST_MEMORY_URL", "http://localhost:8002")
TEST_BILLING_URL = os.getenv("TEST_BILLING_URL", "http://localhost:8003")
TEST_REGISTRY_URL = os.getenv("TEST_REGISTRY_URL", "http://localhost:8004")
TEST_OBSERVABILITY_URL = os.getenv("TEST_OBSERVABILITY_URL", "http://localhost:8005")

# Test user credentials
TEST_USER_EMAIL = "test@example.com"
TEST_USER_PASSWORD = "testpassword123"

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def http_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create an HTTP client for testing."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        yield client

@pytest.fixture(scope="session")
async def wait_for_services():
    """Wait for all services to be ready before running tests."""
    services = [
        TEST_BASE_URL,
        TEST_AUTH_URL,
        TEST_MEMORY_URL,
        TEST_BILLING_URL,
        TEST_REGISTRY_URL,
        TEST_OBSERVABILITY_URL
    ]
    
    async with httpx.AsyncClient() as client:
        for service_url in services:
            max_retries = 30
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    response = await client.get(f"{service_url}/health", timeout=5.0)
                    if response.status_code == 200:
                        print(f"✓ Service {service_url} is ready")
                        break
                except (httpx.RequestError, httpx.TimeoutException):
                    pass
                
                retry_count += 1
                if retry_count < max_retries:
                    await asyncio.sleep(2)
                else:
                    pytest.fail(f"Service {service_url} failed to start after {max_retries} retries")

@pytest.fixture
async def auth_token(http_client: httpx.AsyncClient) -> str:
    """Get an authentication token for testing."""
    # First, try to register the test user
    register_data = {
        "email": TEST_USER_EMAIL,
        "password": TEST_USER_PASSWORD
    }
    
    try:
        await http_client.post(f"{TEST_AUTH_URL}/api/v1/auth/register", json=register_data)
    except:
        pass  # User might already exist
    
    # Login to get token
    login_data = {
        "email": TEST_USER_EMAIL,
        "password": TEST_USER_PASSWORD
    }
    
    response = await http_client.post(f"{TEST_AUTH_URL}/api/v1/auth/login", json=login_data)
    assert response.status_code == 200
    
    token_data = response.json()
    return token_data["access_token"]

@pytest.fixture
def auth_headers(auth_token: str) -> dict:
    """Get authorization headers for authenticated requests."""
    return {"Authorization": f"Bearer {auth_token}"}

@pytest.fixture
async def test_organization(http_client: httpx.AsyncClient, auth_headers: dict) -> dict:
    """Create a test organization."""
    org_data = {
        "name": "Test Organization",
        "description": "Organization for testing purposes"
    }
    
    response = await http_client.post(
        f"{TEST_BASE_URL}/api/v1/organizations",
        json=org_data,
        headers=auth_headers
    )
    
    if response.status_code == 201:
        return response.json()
    elif response.status_code == 409:
        # Organization already exists, get it
        response = await http_client.get(
            f"{TEST_BASE_URL}/api/v1/organizations",
            headers=auth_headers
        )
        orgs = response.json()
        return orgs["organizations"][0] if orgs["organizations"] else None
    else:
        pytest.fail(f"Failed to create test organization: {response.text}")

@pytest.fixture
async def test_memory_context(http_client: httpx.AsyncClient, auth_headers: dict) -> dict:
    """Create a test memory context."""
    memory_data = {
        "content": "This is a test memory for integration testing",
        "metadata": {
            "type": "test_memory",
            "test_id": "pytest_integration"
        },
        "tags": ["test", "integration", "pytest"]
    }
    
    response = await http_client.post(
        f"{TEST_MEMORY_URL}/api/v1/memories",
        json=memory_data,
        headers=auth_headers
    )
    
    assert response.status_code == 200
    return response.json()

@pytest.fixture
async def cleanup_test_data(http_client: httpx.AsyncClient, auth_headers: dict):
    """Cleanup test data after tests complete."""
    yield
    
    # Cleanup logic can be added here
    # For now, we'll just log that cleanup is happening
    print("🧹 Cleaning up test data...")

# Utility functions for tests
def assert_response_structure(response_data: dict, required_fields: list):
    """Assert that a response has the required structure."""
    for field in required_fields:
        assert field in response_data, f"Missing required field: {field}"

def assert_valid_uuid(uuid_string: str):
    """Assert that a string is a valid UUID."""
    import uuid
    try:
        uuid.UUID(uuid_string)
    except ValueError:
        pytest.fail(f"Invalid UUID: {uuid_string}")

def assert_valid_timestamp(timestamp_string: str):
    """Assert that a string is a valid ISO timestamp."""
    from datetime import datetime
    try:
        datetime.fromisoformat(timestamp_string.replace('Z', '+00:00'))
    except ValueError:
        pytest.fail(f"Invalid timestamp: {timestamp_string}")

# Markers for different test types
pytestmark = [
    pytest.mark.asyncio
]