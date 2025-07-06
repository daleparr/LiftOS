"""
Tests for the AgentManager class.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4
from datetime import datetime

from core.agent_manager import AgentManager
from models.agent_models import MarketingAgent, AgentCapability, ModelConfig, MarketingContext
from utils.config import AgenticConfig


@pytest.fixture
def config():
    """Create test configuration."""
    return AgenticConfig(
        environment="test",
        log_level="DEBUG",
        memory_service_url="http://test-memory:8000",
        auth_service_url="http://test-auth:8000"
    )


@pytest.fixture
def mock_memory_service():
    """Create mock memory service."""
    mock = Mock()
    mock.store_agent = AsyncMock(return_value={"success": True})
    mock.get_agent = AsyncMock(return_value=None)
    mock.update_agent = AsyncMock(return_value={"success": True})
    mock.delete_agent = AsyncMock(return_value={"success": True})
    mock.list_agents = AsyncMock(return_value=[])
    return mock


@pytest.fixture
def mock_auth_service():
    """Create mock auth service."""
    mock = Mock()
    mock.validate_token = AsyncMock(return_value={"valid": True, "user_id": "test_user"})
    return mock


@pytest.fixture
def agent_manager(config, mock_memory_service, mock_auth_service):
    """Create AgentManager instance with mocked dependencies."""
    manager = AgentManager(config)
    manager.memory_service = mock_memory_service
    manager.auth_service = mock_auth_service
    return manager


@pytest.fixture
def sample_agent():
    """Create a sample marketing agent."""
    return MarketingAgent(
        agent_id="test_agent_001",
        name="Test Marketing Agent",
        agent_type="content_optimizer",
        description="Test agent for content optimization",
        capabilities=[
            AgentCapability(
                name="content_generation",
                description="Generate marketing content",
                parameters={"max_length": 1000, "tone": "professional"}
            )
        ],
        model_config=ModelConfig(
            provider="openai",
            model_name="gpt-4",
            temperature=0.7,
            max_tokens=1000
        ),
        marketing_context=MarketingContext(
            target_audience="B2B professionals",
            brand_voice="professional",
            campaign_objectives=["awareness", "engagement"],
            budget_constraints={"max_cost_per_execution": 10.0}
        )
    )


class TestAgentManager:
    """Test cases for AgentManager."""

    @pytest.mark.asyncio
    async def test_create_agent_success(self, agent_manager, sample_agent, mock_memory_service):
        """Test successful agent creation."""
        # Setup
        mock_memory_service.get_agent.return_value = None  # Agent doesn't exist
        
        # Execute
        result = await agent_manager.create_agent(sample_agent, "test_user")
        
        # Verify
        assert result["success"] is True
        assert result["agent_id"] == sample_agent.agent_id
        mock_memory_service.store_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_agent_already_exists(self, agent_manager, sample_agent, mock_memory_service):
        """Test agent creation when agent already exists."""
        # Setup
        mock_memory_service.get_agent.return_value = sample_agent.dict()
        
        # Execute & Verify
        with pytest.raises(ValueError, match="Agent with ID .* already exists"):
            await agent_manager.create_agent(sample_agent, "test_user")

    @pytest.mark.asyncio
    async def test_get_agent_success(self, agent_manager, sample_agent, mock_memory_service):
        """Test successful agent retrieval."""
        # Setup
        mock_memory_service.get_agent.return_value = sample_agent.dict()
        
        # Execute
        result = await agent_manager.get_agent(sample_agent.agent_id)
        
        # Verify
        assert result is not None
        assert result.agent_id == sample_agent.agent_id
        assert result.name == sample_agent.name

    @pytest.mark.asyncio
    async def test_get_agent_not_found(self, agent_manager, mock_memory_service):
        """Test agent retrieval when agent doesn't exist."""
        # Setup
        mock_memory_service.get_agent.return_value = None
        
        # Execute
        result = await agent_manager.get_agent("nonexistent_agent")
        
        # Verify
        assert result is None

    @pytest.mark.asyncio
    async def test_update_agent_success(self, agent_manager, sample_agent, mock_memory_service):
        """Test successful agent update."""
        # Setup
        mock_memory_service.get_agent.return_value = sample_agent.dict()
        updated_agent = sample_agent.copy()
        updated_agent.name = "Updated Agent Name"
        
        # Execute
        result = await agent_manager.update_agent(updated_agent, "test_user")
        
        # Verify
        assert result["success"] is True
        mock_memory_service.update_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_agent_not_found(self, agent_manager, sample_agent, mock_memory_service):
        """Test agent update when agent doesn't exist."""
        # Setup
        mock_memory_service.get_agent.return_value = None
        
        # Execute & Verify
        with pytest.raises(ValueError, match="Agent with ID .* not found"):
            await agent_manager.update_agent(sample_agent, "test_user")

    @pytest.mark.asyncio
    async def test_delete_agent_success(self, agent_manager, sample_agent, mock_memory_service):
        """Test successful agent deletion."""
        # Setup
        mock_memory_service.get_agent.return_value = sample_agent.dict()
        
        # Execute
        result = await agent_manager.delete_agent(sample_agent.agent_id, "test_user")
        
        # Verify
        assert result["success"] is True
        mock_memory_service.delete_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_agent_not_found(self, agent_manager, mock_memory_service):
        """Test agent deletion when agent doesn't exist."""
        # Setup
        mock_memory_service.get_agent.return_value = None
        
        # Execute & Verify
        with pytest.raises(ValueError, match="Agent with ID .* not found"):
            await agent_manager.delete_agent("nonexistent_agent", "test_user")

    @pytest.mark.asyncio
    async def test_list_agents_success(self, agent_manager, sample_agent, mock_memory_service):
        """Test successful agent listing."""
        # Setup
        mock_memory_service.list_agents.return_value = [sample_agent.dict()]
        
        # Execute
        result = await agent_manager.list_agents()
        
        # Verify
        assert len(result) == 1
        assert result[0].agent_id == sample_agent.agent_id

    @pytest.mark.asyncio
    async def test_list_agents_with_filters(self, agent_manager, mock_memory_service):
        """Test agent listing with filters."""
        # Setup
        filters = {"agent_type": "content_optimizer", "status": "active"}
        mock_memory_service.list_agents.return_value = []
        
        # Execute
        result = await agent_manager.list_agents(filters=filters)
        
        # Verify
        assert len(result) == 0
        mock_memory_service.list_agents.assert_called_once_with(filters=filters)

    @pytest.mark.asyncio
    async def test_start_agent_session(self, agent_manager, sample_agent, mock_memory_service):
        """Test starting an agent session."""
        # Setup
        mock_memory_service.get_agent.return_value = sample_agent.dict()
        
        # Execute
        session_id = await agent_manager.start_agent_session(sample_agent.agent_id, "test_user")
        
        # Verify
        assert session_id is not None
        assert len(session_id) > 0

    @pytest.mark.asyncio
    async def test_stop_agent_session(self, agent_manager, sample_agent, mock_memory_service):
        """Test stopping an agent session."""
        # Setup
        mock_memory_service.get_agent.return_value = sample_agent.dict()
        session_id = await agent_manager.start_agent_session(sample_agent.agent_id, "test_user")
        
        # Execute
        result = await agent_manager.stop_agent_session(session_id, "test_user")
        
        # Verify
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_get_agent_status(self, agent_manager, sample_agent, mock_memory_service):
        """Test getting agent status."""
        # Setup
        mock_memory_service.get_agent.return_value = sample_agent.dict()
        
        # Execute
        status = await agent_manager.get_agent_status(sample_agent.agent_id)
        
        # Verify
        assert status["agent_id"] == sample_agent.agent_id
        assert "status" in status
        assert "last_activity" in status

    @pytest.mark.asyncio
    async def test_validate_agent_config(self, agent_manager, sample_agent):
        """Test agent configuration validation."""
        # Execute
        is_valid, errors = await agent_manager.validate_agent_config(sample_agent)
        
        # Verify
        assert is_valid is True
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_validate_agent_config_invalid(self, agent_manager):
        """Test agent configuration validation with invalid config."""
        # Setup - Create invalid agent
        invalid_agent = MarketingAgent(
            agent_id="",  # Invalid empty ID
            name="",      # Invalid empty name
            agent_type="invalid_type",
            description="Test agent",
            capabilities=[],
            model_config=ModelConfig(
                provider="invalid_provider",
                model_name="",
                temperature=2.0,  # Invalid temperature > 1.0
                max_tokens=-1     # Invalid negative tokens
            ),
            marketing_context=MarketingContext(
                target_audience="",
                brand_voice="",
                campaign_objectives=[],
                budget_constraints={}
            )
        )
        
        # Execute
        is_valid, errors = await agent_manager.validate_agent_config(invalid_agent)
        
        # Verify
        assert is_valid is False
        assert len(errors) > 0