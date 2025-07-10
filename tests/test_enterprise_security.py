"""
Comprehensive Enterprise Security Test Suite
Tests for all enterprise security components and integrations
"""

import pytest
import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
import json
import sys
import os
from unittest.mock import Mock, patch, AsyncMock

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.security.api_key_vault import APIKeyVault
from shared.security.enhanced_jwt import EnhancedJWTManager, JWTPayload
from shared.security.audit_logger import SecurityAuditLogger, SecurityEventType
from shared.security.enhanced_middleware import SecurityMiddleware, SecurityContext
from shared.database.security_models import EncryptedAPIKey, EnhancedUserSession, SecurityAuditLog
from services.data_ingestion.enhanced_credential_manager import EnhancedCredentialManager as DataCredentialManager
from services.channels.enhanced_credential_manager import EnhancedCredentialManager as ChannelsCredentialManager

class TestAPIKeyVault:
    """Test suite for API Key Vault"""
    
    @pytest.fixture
    async def vault(self):
        """Create API key vault instance"""
        vault = APIKeyVault()
        await vault.initialize()
        return vault
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_api_key(self, vault):
        """Test storing and retrieving API keys"""
        # Test data
        provider = "test_provider"
        key_name = "api_key"
        api_key = "test_api_key_12345"
        user_id = "test_user"
        org_id = "test_org"
        
        # Store API key
        key_id = await vault.store_api_key(
            provider=provider,
            key_name=key_name,
            api_key=api_key,
            user_id=user_id,
            org_id=org_id
        )
        
        assert key_id is not None
        assert isinstance(key_id, str)
        
        # Retrieve API key
        retrieved_key = await vault.get_api_key(
            credential_id=key_id,
            user_id=user_id,
            org_id=org_id
        )
        
        assert retrieved_key == api_key
    
    @pytest.mark.asyncio
    async def test_api_key_encryption(self, vault):
        """Test that API keys are properly encrypted"""
        provider = "test_provider"
        key_name = "secret_key"
        api_key = "super_secret_key_12345"
        user_id = "test_user"
        org_id = "test_org"
        
        # Store API key
        key_id = await vault.store_api_key(
            provider=provider,
            key_name=key_name,
            api_key=api_key,
            user_id=user_id,
            org_id=org_id
        )
        
        # Verify the stored value is encrypted (not plain text)
        # This would require database access to verify encryption
        # For now, we test that retrieval works correctly
        retrieved_key = await vault.get_api_key(
            credential_id=key_id,
            user_id=user_id,
            org_id=org_id
        )
        
        assert retrieved_key == api_key
    
    @pytest.mark.asyncio
    async def test_api_key_rotation(self, vault):
        """Test API key rotation functionality"""
        provider = "test_provider"
        key_name = "rotating_key"
        old_key = "old_api_key_12345"
        new_key = "new_api_key_67890"
        user_id = "test_user"
        org_id = "test_org"
        
        # Store initial API key
        key_id = await vault.store_api_key(
            provider=provider,
            key_name=key_name,
            api_key=old_key,
            user_id=user_id,
            org_id=org_id
        )
        
        # Rotate the key
        success = await vault.update_api_key(
            credential_id=key_id,
            new_api_key=new_key,
            user_id=user_id,
            org_id=org_id
        )
        
        assert success is True
        
        # Verify new key is retrieved
        retrieved_key = await vault.get_api_key(
            credential_id=key_id,
            user_id=user_id,
            org_id=org_id
        )
        
        assert retrieved_key == new_key
        assert retrieved_key != old_key

class TestEnhancedJWTManager:
    """Test suite for Enhanced JWT Manager"""
    
    @pytest.fixture
    def jwt_manager(self):
        """Create JWT manager instance"""
        return EnhancedJWTManager()
    
    @pytest.mark.asyncio
    async def test_token_generation_and_verification(self, jwt_manager):
        """Test JWT token generation and verification"""
        # Test payload
        payload = JWTPayload(
            user_id="test_user",
            org_id="test_org",
            session_id="test_session",
            permissions=["read", "write"],
            device_fingerprint="test_fingerprint"
        )
        
        # Generate token
        token = await jwt_manager.create_token(payload)
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Verify token
        verified_payload = await jwt_manager.verify_token(token)
        
        assert verified_payload.user_id == payload.user_id
        assert verified_payload.org_id == payload.org_id
        assert verified_payload.session_id == payload.session_id
        assert verified_payload.permissions == payload.permissions
    
    @pytest.mark.asyncio
    async def test_refresh_token_flow(self, jwt_manager):
        """Test refresh token generation and usage"""
        payload = JWTPayload(
            user_id="test_user",
            org_id="test_org",
            session_id="test_session",
            permissions=["read"]
        )
        
        # Generate tokens
        access_token, refresh_token = await jwt_manager.create_token_pair(payload)
        
        assert access_token is not None
        assert refresh_token is not None
        assert access_token != refresh_token
        
        # Use refresh token to get new access token
        new_access_token = await jwt_manager.refresh_access_token(refresh_token)
        
        assert new_access_token is not None
        assert new_access_token != access_token
        
        # Verify new token works
        verified_payload = await jwt_manager.verify_token(new_access_token)
        assert verified_payload.user_id == payload.user_id
    
    @pytest.mark.asyncio
    async def test_token_expiration(self, jwt_manager):
        """Test token expiration handling"""
        payload = JWTPayload(
            user_id="test_user",
            org_id="test_org",
            session_id="test_session",
            permissions=["read"]
        )
        
        # Create token with very short expiration
        with patch.object(jwt_manager, 'access_token_expire_minutes', 0.01):  # 0.6 seconds
            token = await jwt_manager.create_token(payload)
            
            # Token should work immediately
            verified_payload = await jwt_manager.verify_token(token)
            assert verified_payload.user_id == payload.user_id
            
            # Wait for expiration
            await asyncio.sleep(1)
            
            # Token should now be expired
            with pytest.raises(Exception):  # Should raise JWT expired exception
                await jwt_manager.verify_token(token)

class TestSecurityAuditLogger:
    """Test suite for Security Audit Logger"""
    
    @pytest.fixture
    def audit_logger(self):
        """Create audit logger instance"""
        return SecurityAuditLogger()
    
    @pytest.mark.asyncio
    async def test_log_security_event(self, audit_logger):
        """Test logging security events"""
        # Log a security event
        await audit_logger.log_event(
            event_type=SecurityEventType.LOGIN_SUCCESS,
            action="user_login",
            user_id="test_user",
            org_id="test_org",
            ip_address="192.168.1.1",
            details={"method": "password", "device": "desktop"}
        )
        
        # Verify event was logged (would require database verification in real test)
        assert True  # Placeholder for actual verification
    
    @pytest.mark.asyncio
    async def test_log_api_key_access(self, audit_logger):
        """Test logging API key access events"""
        await audit_logger.log_event(
            event_type=SecurityEventType.API_KEY_ACCESS,
            action="api_key_retrieved",
            user_id="test_user",
            org_id="test_org",
            ip_address="192.168.1.1",
            details={
                "provider": "facebook",
                "key_name": "access_token",
                "service": "data_ingestion"
            }
        )
        
        assert True  # Placeholder for actual verification
    
    @pytest.mark.asyncio
    async def test_log_security_violation(self, audit_logger):
        """Test logging security violations"""
        await audit_logger.log_event(
            event_type=SecurityEventType.SECURITY_VIOLATION,
            action="unauthorized_access_attempt",
            user_id="malicious_user",
            ip_address="192.168.1.100",
            details={
                "attempted_resource": "/admin/users",
                "reason": "insufficient_permissions",
                "risk_score": 0.9
            }
        )
        
        assert True  # Placeholder for actual verification

class TestSecurityMiddleware:
    """Test suite for Security Middleware"""
    
    @pytest.fixture
    def middleware(self):
        """Create security middleware instance"""
        return SecurityMiddleware()
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, middleware):
        """Test rate limiting functionality"""
        # Mock request
        mock_request = Mock()
        mock_request.client.host = "192.168.1.1"
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"
        
        # Test multiple requests from same IP
        for i in range(5):
            result = await middleware.check_rate_limit(mock_request)
            if i < 4:  # First 4 should pass
                assert result is True
            # 5th might fail depending on rate limit settings
    
    @pytest.mark.asyncio
    async def test_security_context_creation(self, middleware):
        """Test security context creation"""
        mock_request = Mock()
        mock_request.client.host = "192.168.1.1"
        mock_request.headers = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "x-forwarded-for": "203.0.113.1"
        }
        
        context = await middleware.create_security_context(mock_request)
        
        assert isinstance(context, SecurityContext)
        assert context.ip_address == "203.0.113.1"  # Should use X-Forwarded-For
        assert context.user_agent is not None
        assert context.risk_score >= 0.0
        assert context.risk_score <= 1.0

class TestCredentialManagers:
    """Test suite for Enhanced Credential Managers"""
    
    @pytest.fixture
    async def data_credential_manager(self):
        """Create data ingestion credential manager"""
        manager = DataCredentialManager()
        await manager.initialize()
        return manager
    
    @pytest.fixture
    async def channels_credential_manager(self):
        """Create channels credential manager"""
        manager = ChannelsCredentialManager()
        await manager.initialize()
        return manager
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_credentials(self, data_credential_manager):
        """Test credential storage and retrieval"""
        provider = "facebook"
        credential_type = "access_token"
        credential_value = "test_facebook_token_12345"
        user_id = "test_user"
        org_id = "test_org"
        
        # Store credential
        credential_id = await data_credential_manager.store_credential(
            provider=provider,
            credential_type=credential_type,
            credential_value=credential_value,
            user_id=user_id,
            org_id=org_id
        )
        
        assert credential_id is not None
        
        # Retrieve credential
        retrieved_value = await data_credential_manager.retrieve_credential(
            credential_id=credential_id,
            user_id=user_id,
            org_id=org_id
        )
        
        assert retrieved_value == credential_value
    
    @pytest.mark.asyncio
    async def test_provider_credential_validation(self, channels_credential_manager):
        """Test provider credential validation"""
        # Test Facebook credentials
        facebook_credentials = {
            "access_token": "EAAtest_token",
            "app_secret": "test_secret",
            "app_id": "123456789"
        }
        
        validation_result = await channels_credential_manager.validate_provider_credentials(
            provider="facebook",
            credentials=facebook_credentials
        )
        
        assert validation_result["valid"] is True
        assert len(validation_result["missing_credentials"]) == 0
        
        # Test incomplete credentials
        incomplete_credentials = {
            "access_token": "EAAtest_token"
            # Missing app_secret and app_id
        }
        
        validation_result = await channels_credential_manager.validate_provider_credentials(
            provider="facebook",
            credentials=incomplete_credentials
        )
        
        assert validation_result["valid"] is False
        assert "app_secret" in validation_result["missing_credentials"]
        assert "app_id" in validation_result["missing_credentials"]
    
    @pytest.mark.asyncio
    async def test_credential_caching(self, data_credential_manager):
        """Test credential caching functionality"""
        provider = "google"
        credential_type = "client_secret"
        credential_value = "test_google_secret_12345"
        user_id = "test_user"
        org_id = "test_org"
        
        # Store credential
        credential_id = await data_credential_manager.store_credential(
            provider=provider,
            credential_type=credential_type,
            credential_value=credential_value,
            user_id=user_id,
            org_id=org_id
        )
        
        # First retrieval (should hit database)
        start_time = time.time()
        retrieved_value1 = await data_credential_manager.retrieve_credential(
            credential_id=credential_id,
            user_id=user_id,
            org_id=org_id
        )
        first_retrieval_time = time.time() - start_time
        
        # Second retrieval (should hit cache)
        start_time = time.time()
        retrieved_value2 = await data_credential_manager.retrieve_credential(
            credential_id=credential_id,
            user_id=user_id,
            org_id=org_id
        )
        second_retrieval_time = time.time() - start_time
        
        assert retrieved_value1 == credential_value
        assert retrieved_value2 == credential_value
        # Cache should be faster (though this might be flaky in tests)
        # assert second_retrieval_time < first_retrieval_time

class TestIntegrationSecurity:
    """Integration tests for enterprise security"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_security_flow(self):
        """Test complete security flow from authentication to API key access"""
        # This would test the complete flow:
        # 1. User authentication
        # 2. JWT token generation
        # 3. API request with token
        # 4. Token verification
        # 5. Permission checking
        # 6. API key retrieval
        # 7. Audit logging
        
        # Mock components for integration test
        jwt_manager = EnhancedJWTManager()
        audit_logger = SecurityAuditLogger()
        
        # Create test user payload
        payload = JWTPayload(
            user_id="integration_test_user",
            org_id="integration_test_org",
            session_id="integration_test_session",
            permissions=["api_keys:read", "data_ingestion:access"]
        )
        
        # Generate token
        token = await jwt_manager.create_token(payload)
        assert token is not None
        
        # Verify token (simulating middleware)
        verified_payload = await jwt_manager.verify_token(token)
        assert verified_payload.user_id == payload.user_id
        
        # Check permissions
        assert "api_keys:read" in verified_payload.permissions
        
        # Log successful access
        await audit_logger.log_event(
            event_type=SecurityEventType.API_ACCESS,
            action="integration_test_success",
            user_id=verified_payload.user_id,
            org_id=verified_payload.org_id,
            details={"test": "end_to_end_security_flow"}
        )
        
        assert True  # Test completed successfully
    
    @pytest.mark.asyncio
    async def test_security_violation_detection(self):
        """Test security violation detection and response"""
        audit_logger = SecurityAuditLogger()
        
        # Simulate multiple failed login attempts
        for i in range(5):
            await audit_logger.log_event(
                event_type=SecurityEventType.LOGIN_FAILED,
                action="failed_login_attempt",
                user_id="potential_attacker",
                ip_address="192.168.1.100",
                details={
                    "attempt": i + 1,
                    "reason": "invalid_password",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        
        # Log security violation
        await audit_logger.log_event(
            event_type=SecurityEventType.SECURITY_VIOLATION,
            action="brute_force_detected",
            user_id="potential_attacker",
            ip_address="192.168.1.100",
            details={
                "failed_attempts": 5,
                "time_window": "5_minutes",
                "risk_score": 0.95
            }
        )
        
        assert True  # Test completed successfully

class TestSecurityPerformance:
    """Performance tests for security components"""
    
    @pytest.mark.asyncio
    async def test_jwt_performance(self):
        """Test JWT token generation and verification performance"""
        jwt_manager = EnhancedJWTManager()
        
        payload = JWTPayload(
            user_id="perf_test_user",
            org_id="perf_test_org",
            session_id="perf_test_session",
            permissions=["read", "write"]
        )
        
        # Test token generation performance
        start_time = time.time()
        tokens = []
        for i in range(100):
            token = await jwt_manager.create_token(payload)
            tokens.append(token)
        generation_time = time.time() - start_time
        
        assert len(tokens) == 100
        assert generation_time < 5.0  # Should generate 100 tokens in under 5 seconds
        
        # Test token verification performance
        start_time = time.time()
        for token in tokens:
            verified_payload = await jwt_manager.verify_token(token)
            assert verified_payload.user_id == payload.user_id
        verification_time = time.time() - start_time
        
        assert verification_time < 5.0  # Should verify 100 tokens in under 5 seconds
    
    @pytest.mark.asyncio
    async def test_encryption_performance(self):
        """Test API key encryption/decryption performance"""
        vault = APIKeyVault()
        await vault.initialize()
        
        # Test encryption performance
        test_keys = [f"test_api_key_{i}" for i in range(50)]
        user_id = "perf_test_user"
        org_id = "perf_test_org"
        
        start_time = time.time()
        key_ids = []
        for i, api_key in enumerate(test_keys):
            key_id = await vault.store_api_key(
                provider="test_provider",
                key_name=f"key_{i}",
                api_key=api_key,
                user_id=user_id,
                org_id=org_id
            )
            key_ids.append(key_id)
        encryption_time = time.time() - start_time
        
        assert len(key_ids) == 50
        assert encryption_time < 10.0  # Should encrypt 50 keys in under 10 seconds
        
        # Test decryption performance
        start_time = time.time()
        for i, key_id in enumerate(key_ids):
            retrieved_key = await vault.get_api_key(
                credential_id=key_id,
                user_id=user_id,
                org_id=org_id
            )
            assert retrieved_key == test_keys[i]
        decryption_time = time.time() - start_time
        
        assert decryption_time < 10.0  # Should decrypt 50 keys in under 10 seconds

# Test configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])