"""
Pytest configuration and shared fixtures for Agentic microservice tests.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from fastapi.testclient import TestClient

from app import app
from utils.config import AgenticConfig


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config():
    """Create test configuration."""
    return AgenticConfig(
        environment="test",
        log_level="DEBUG",
        memory_service_url="http://test-memory:8000",
        auth_service_url="http://test-auth:8000",
        observability_service_url="http://test-observability:8000",
        llm_service_url="http://test-llm:8000",
        causal_service_url="http://test-causal:8000",
        database_url="sqlite:///:memory:",
        redis_url="redis://test-redis:6379"
    )


@pytest.fixture
def test_client():
    """Create test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_memory_service():
    """Create mock memory service."""
    mock = Mock()
    mock.store_agent = AsyncMock(return_value={"success": True})
    mock.get_agent = AsyncMock(return_value=None)
    mock.update_agent = AsyncMock(return_value={"success": True})
    mock.delete_agent = AsyncMock(return_value={"success": True})
    mock.list_agents = AsyncMock(return_value=[])
    mock.store_test_result = AsyncMock(return_value={"success": True})
    mock.get_test_result = AsyncMock(return_value=None)
    mock.list_test_results = AsyncMock(return_value=[])
    mock.store_evaluation_result = AsyncMock(return_value={"success": True})
    mock.get_evaluation_result = AsyncMock(return_value=None)
    mock.list_evaluation_results = AsyncMock(return_value=[])
    return mock


@pytest.fixture
def mock_auth_service():
    """Create mock auth service."""
    mock = Mock()
    mock.validate_token = AsyncMock(return_value={"valid": True, "user_id": "test_user"})
    mock.get_user_permissions = AsyncMock(return_value=["agent:read", "agent:write", "test:execute"])
    return mock


@pytest.fixture
def mock_llm_service():
    """Create mock LLM service."""
    mock = Mock()
    mock.generate_response = AsyncMock(return_value={
        "response": "Test response",
        "tokens_used": 100,
        "cost": 0.01
    })
    mock.get_available_models = AsyncMock(return_value=["gpt-4", "gpt-3.5-turbo", "claude-3"])
    return mock


@pytest.fixture
def mock_causal_service():
    """Create mock causal service."""
    mock = Mock()
    mock.analyze_campaign_impact = AsyncMock(return_value={
        "impact_score": 0.85,
        "confidence": 0.92,
        "recommendations": ["Increase budget", "Optimize targeting"]
    })
    return mock


@pytest.fixture
def mock_observability_service():
    """Create mock observability service."""
    mock = Mock()
    mock.log_event = AsyncMock(return_value={"success": True})
    mock.track_metric = AsyncMock(return_value={"success": True})
    mock.create_trace = AsyncMock(return_value={"trace_id": "test_trace_123"})
    return mock


@pytest.fixture
def sample_test_data():
    """Create sample test data for marketing scenarios."""
    return {
        "campaign_data": {
            "name": "Test Campaign",
            "budget": 10000,
            "target_audience": "B2B professionals",
            "channels": ["email", "social", "search"]
        },
        "creative_assets": {
            "variants": 3,
            "formats": ["image", "video"],
            "messages": ["Value proposition A", "Value proposition B"]
        },
        "performance_metrics": {
            "impressions": 100000,
            "clicks": 2500,
            "conversions": 125,
            "cost": 5000
        }
    }


@pytest.fixture
def sample_success_criteria():
    """Create sample success criteria for tests."""
    return [
        {
            "metric_name": "conversion_rate",
            "operator": ">=",
            "threshold": 0.05,
            "weight": 1.0,
            "description": "Conversion rate should be at least 5%"
        },
        {
            "metric_name": "cost_per_acquisition",
            "operator": "<=",
            "threshold": 40.0,
            "weight": 0.8,
            "description": "CPA should be under $40"
        }
    ]