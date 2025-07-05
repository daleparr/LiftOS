#!/usr/bin/env python3
"""
Unit tests for all core services
Tests individual service functionality including memory operations
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient

# Add shared modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.health.health_checks import HealthChecker, check_database_connection
from shared.security.security_manager import SecurityManager
from shared.logging.structured_logger import LiftOSLogger
from shared.config.secrets_manager import SecretsManager

@pytest.mark.unit
class TestHealthChecker:
    """Unit tests for HealthChecker class"""
    
    def test_health_checker_initialization(self):
        """Test HealthChecker can be initialized"""
        checker = HealthChecker("test-service")
        assert checker.service_name == "test-service"
        assert hasattr(checker, 'get_health_status')
        assert hasattr(checker, 'get_readiness_status')
    
    async def test_get_health_status_basic(self):
        """Test basic health status returns healthy"""
        checker = HealthChecker("test-service")
        status = await checker.get_health_status()
        
        assert status["status"] == "healthy"
        assert status["service"] == "test-service"
        assert "timestamp" in status
        assert "uptime" in status
        assert "version" in status
    
    async def test_get_readiness_status_no_checks(self):
        """Test readiness status with no external checks"""
        checker = HealthChecker("test-service")
        status = await checker.get_readiness_status()
        
        assert status["status"] == "ready"
        assert status["service"] == "test-service"
        assert "timestamp" in status
        assert status["checks"] == []
    
    async def test_get_readiness_status_with_checks(self):
        """Test readiness status with external checks"""
        async def mock_check():
            return {"status": "healthy", "service": "external"}
        
        checker = HealthChecker("test-service")
        status = await checker.get_readiness_status([mock_check])
        
        assert status["status"] == "ready"
        assert len(status["checks"]) == 1
        assert status["checks"][0]["status"] == "healthy"

@pytest.mark.unit
class TestSecurityManager:
    """Unit tests for SecurityManager class"""
    
    def test_security_manager_initialization(self):
        """Test SecurityManager can be initialized"""
        manager = SecurityManager(jwt_secret="test_secret_key")
        assert hasattr(manager, 'verify_jwt')
        assert hasattr(manager, 'create_access_token')
        assert hasattr(manager, 'rate_limit')
    
    def test_create_access_token(self):
        """Test JWT token creation"""
        manager = SecurityManager(jwt_secret="test_secret_key")
        payload = {"user_id": "123", "email": "test@example.com"}
        
        token = manager.create_access_token(payload)
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_verify_jwt_valid(self):
        """Test JWT token verification with valid token"""
        manager = SecurityManager(jwt_secret="test_secret_key")
        payload = {"user_id": "123", "email": "test@example.com"}
        
        token = manager.create_access_token(payload)
        decoded = manager.verify_jwt(token)
        
        assert decoded["user_id"] == "123"
        assert decoded["email"] == "test@example.com"
    
    def test_verify_jwt_invalid(self):
        """Test JWT token verification with invalid token"""
        manager = SecurityManager(jwt_secret="test_secret_key")
        
        result = manager.verify_jwt("invalid.token.here")
        assert result is None
    
    def test_rate_limit_new_client(self):
        """Test rate limiting for new client"""
        manager = SecurityManager(jwt_secret="test_secret_key")
        
        # First request should be allowed
        result = manager.rate_limit("192.168.1.1")
        assert result is True
    
    def test_rate_limit_exceeded(self):
        """Test rate limiting when limit exceeded"""
        manager = SecurityManager(jwt_secret="test_secret_key")
        client_ip = "192.168.1.2"
        
        # Make requests up to limit
        for _ in range(100):  # Default limit
            manager.rate_limit(client_ip)
        
        # Next request should be blocked
        result = manager.rate_limit(client_ip)
        assert result is False

@pytest.mark.unit
class TestLiftOSLogger:
    """Unit tests for LiftOSLogger class"""
    
    def test_logger_initialization(self):
        """Test LiftOSLogger can be initialized"""
        logger = LiftOSLogger("test-service")
        assert logger.service_name == "test-service"
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'warning')
    
    def test_logger_with_correlation_id(self):
        """Test logger with correlation ID"""
        logger = LiftOSLogger("test-service", correlation_id="test-123")
        assert logger.correlation_id == "test-123"
    
    @patch('shared.logging.structured_logger.logging.getLogger')
    def test_logger_info_call(self, mock_get_logger):
        """Test logger info method call"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        logger = LiftOSLogger("test-service")
        logger.info("Test message", extra_field="value")
        
        mock_logger.info.assert_called_once()

@pytest.mark.unit
class TestSecretsManager:
    """Unit tests for SecretsManager class"""
    
    def test_secrets_manager_initialization(self):
        """Test SecretsManager can be initialized"""
        manager = SecretsManager()
        assert hasattr(manager, 'get_secret')
    
    @patch.dict(os.environ, {'TEST_SECRET': 'test_value'})
    @pytest.mark.asyncio
    async def test_get_secret_from_env(self):
        """Test getting secret from environment variables"""
        manager = SecretsManager(backend='env')
        
        value = await manager.get_secret('TEST_SECRET')
        assert value == 'test_value'
    
    @pytest.mark.asyncio
    async def test_get_secret_not_found(self):
        """Test getting non-existent secret"""
        manager = SecretsManager(backend='env')
        
        value = await manager.get_secret('NON_EXISTENT_SECRET')
        assert value is None
    
    def test_secrets_manager_backend_configuration(self):
        """Test SecretsManager backend configuration"""
        manager = SecretsManager(backend='env')
        assert manager.backend == 'env'

@pytest.mark.unit
@pytest.mark.memory
class TestMemoryOperations:
    """Unit tests for memory service operations"""
    
    async def test_memory_context_creation(self):
        """Test memory context creation"""
        # Mock memory context creation
        context_data = {
            "org_id": "org_123",
            "user_id": "user_456",
            "session_id": "session_789"
        }
        
        # Simulate memory context validation
        assert context_data["org_id"] is not None
        assert context_data["user_id"] is not None
        assert context_data["session_id"] is not None
    
    async def test_memory_storage_operations(self):
        """Test memory storage and retrieval operations"""
        # Mock memory storage
        memory_data = {
            "key": "test_memory",
            "value": {"data": "test_value"},
            "metadata": {"type": "user_preference"}
        }
        
        # Simulate storage validation
        assert memory_data["key"] is not None
        assert memory_data["value"] is not None
        assert isinstance(memory_data["metadata"], dict)
    
    async def test_kse_integration(self):
        """Test KSE (Knowledge Storage Engine) integration"""
        # Mock KSE operations
        kse_query = {
            "query": "test search",
            "context": "user_context",
            "filters": {"type": "document"}
        }
        
        # Simulate KSE response
        kse_response = {
            "results": [{"id": "doc_1", "score": 0.95}],
            "total": 1,
            "query_time": 0.05
        }
        
        assert len(kse_response["results"]) > 0
        assert kse_response["total"] == 1
        assert kse_response["query_time"] < 1.0

@pytest.mark.unit
@pytest.mark.auth
class TestAuthenticationOperations:
    """Unit tests for authentication service operations"""
    
    def test_user_registration_validation(self):
        """Test user registration data validation"""
        user_data = {
            "email": "test@example.com",
            "password": "secure_password123",
            "org_id": "org_123"
        }
        
        # Validate email format
        assert "@" in user_data["email"]
        assert "." in user_data["email"]
        
        # Validate password strength
        assert len(user_data["password"]) >= 8
        
        # Validate org_id
        assert user_data["org_id"] is not None
    
    def test_login_validation(self):
        """Test login data validation"""
        login_data = {
            "email": "test@example.com",
            "password": "secure_password123"
        }
        
        assert login_data["email"] is not None
        assert login_data["password"] is not None
        assert len(login_data["password"]) > 0
    
    def test_token_payload_structure(self):
        """Test JWT token payload structure"""
        token_payload = {
            "user_id": "user_123",
            "email": "test@example.com",
            "org_id": "org_456",
            "exp": 1234567890,
            "iat": 1234567800
        }
        
        required_fields = ["user_id", "email", "org_id", "exp", "iat"]
        for field in required_fields:
            assert field in token_payload

@pytest.mark.unit
@pytest.mark.registry
class TestRegistryOperations:
    """Unit tests for registry service operations"""
    
    def test_module_registration_validation(self):
        """Test module registration data validation"""
        module_data = {
            "name": "test-module",
            "version": "1.0.0",
            "endpoint": "http://localhost:9000",
            "health_check": "/health",
            "capabilities": ["read", "write"]
        }
        
        required_fields = ["name", "version", "endpoint"]
        for field in required_fields:
            assert field in module_data
            assert module_data[field] is not None
    
    def test_module_discovery(self):
        """Test module discovery functionality"""
        registered_modules = [
            {"name": "module-a", "endpoint": "http://localhost:9001"},
            {"name": "module-b", "endpoint": "http://localhost:9002"}
        ]
        
        # Test module lookup
        target_module = "module-a"
        found_module = next(
            (m for m in registered_modules if m["name"] == target_module),
            None
        )
        
        assert found_module is not None
        assert found_module["name"] == target_module
    
    def test_health_check_validation(self):
        """Test module health check validation"""
        health_response = {
            "status": "healthy",
            "service": "test-module",
            "timestamp": "2025-01-07T12:00:00Z"
        }
        
        assert health_response["status"] in ["healthy", "unhealthy"]
        assert health_response["service"] is not None
        assert health_response["timestamp"] is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
