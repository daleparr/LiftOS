"""
Comprehensive Security Integration Tests
Tests the complete enterprise security infrastructure including API key vault,
enhanced JWT, audit logging, and credential management.
"""

import pytest
import asyncio
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from shared.security.api_key_vault import APIKeyVault, get_api_key_vault
from shared.security.enhanced_jwt import EnhancedJWTManager, DeviceFingerprint, get_enhanced_jwt_manager
from shared.security.audit_logger import SecurityAuditLogger, SecurityEventType
from shared.database.database import get_async_session
from shared.database.migration_runner import MigrationRunner
from services.data_ingestion.enhanced_credential_manager import (
    EnhancedCredentialManager, 
    EnhancedCredentialProvider,
    get_enhanced_credential_manager
)

class TestSecurityInfrastructure:
    """Test suite for enterprise security infrastructure"""
    
    @pytest.fixture(scope="class")
    async def setup_database(self):
        """Set up test database with security tables"""
        try:
            runner = MigrationRunner()
            await runner.run_all_migrations()
            yield
        except Exception as e:
            pytest.skip(f"Database setup failed: {e}")
    
    @pytest.fixture
    def test_org_id(self):
        """Test organization ID"""
        return "test-org-123"
    
    @pytest.fixture
    def test_user_id(self):
        """Test user ID"""
        return "test-user-456"
    
    @pytest.fixture
    def test_credentials(self):
        """Sample test credentials"""
        return {
            "api_key": "test-api-key-12345",
            "secret": "test-secret-67890",
            "endpoint": "https://api.example.com"
        }
    
    @pytest.fixture
    def device_info(self):
        """Sample device information"""
        return {
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "ip_address": "192.168.1.100"
        }

class TestAPIKeyVault(TestSecurityInfrastructure):
    """Test API Key Vault functionality"""
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_api_key(self, setup_database, test_org_id, test_credentials):
        """Test storing and retrieving API keys"""
        vault = get_api_key_vault()
        
        async with get_async_session() as session:
            # Store API key
            success = await vault.store_api_key(
                session=session,
                org_id=test_org_id,
                provider="test_provider",
                key_name="test_key",
                api_key_data=test_credentials,
                created_by="test_user"
            )
            
            assert success, "Failed to store API key"
            
            # Retrieve API key
            retrieved_credentials = await vault.get_api_key(
                session=session,
                org_id=test_org_id,
                provider="test_provider",
                key_name="test_key"
            )
            
            assert retrieved_credentials is not None, "Failed to retrieve API key"
            assert retrieved_credentials["api_key"] == test_credentials["api_key"]
            assert retrieved_credentials["secret"] == test_credentials["secret"]
    
    @pytest.mark.asyncio
    async def test_api_key_rotation(self, setup_database, test_org_id, test_credentials):
        """Test API key rotation"""
        vault = get_api_key_vault()
        
        new_credentials = {
            "api_key": "new-api-key-54321",
            "secret": "new-secret-09876",
            "endpoint": "https://api.example.com"
        }
        
        async with get_async_session() as session:
            # Store initial API key
            await vault.store_api_key(
                session=session,
                org_id=test_org_id,
                provider="rotation_test",
                key_name="test_key",
                api_key_data=test_credentials
            )
            
            # Rotate API key
            success = await vault.rotate_api_key(
                session=session,
                org_id=test_org_id,
                provider="rotation_test",
                key_name="test_key",
                new_api_key_data=new_credentials,
                rotated_by="test_user"
            )
            
            assert success, "Failed to rotate API key"
            
            # Verify new credentials
            retrieved_credentials = await vault.get_api_key(
                session=session,
                org_id=test_org_id,
                provider="rotation_test",
                key_name="test_key"
            )
            
            assert retrieved_credentials["api_key"] == new_credentials["api_key"]
            assert retrieved_credentials["secret"] == new_credentials["secret"]
    
    @pytest.mark.asyncio
    async def test_api_key_revocation(self, setup_database, test_org_id, test_credentials):
        """Test API key revocation"""
        vault = get_api_key_vault()
        
        async with get_async_session() as session:
            # Store API key
            await vault.store_api_key(
                session=session,
                org_id=test_org_id,
                provider="revocation_test",
                key_name="test_key",
                api_key_data=test_credentials
            )
            
            # Revoke API key
            success = await vault.revoke_api_key(
                session=session,
                org_id=test_org_id,
                provider="revocation_test",
                key_name="test_key",
                revoked_by="test_user",
                reason="security_breach"
            )
            
            assert success, "Failed to revoke API key"
            
            # Verify key is no longer accessible
            retrieved_credentials = await vault.get_api_key(
                session=session,
                org_id=test_org_id,
                provider="revocation_test",
                key_name="test_key"
            )
            
            assert retrieved_credentials is None, "Revoked key should not be accessible"
    
    @pytest.mark.asyncio
    async def test_list_api_keys(self, setup_database, test_org_id, test_credentials):
        """Test listing API keys for an organization"""
        vault = get_api_key_vault()
        
        async with get_async_session() as session:
            # Store multiple API keys
            providers = ["provider1", "provider2", "provider3"]
            
            for provider in providers:
                await vault.store_api_key(
                    session=session,
                    org_id=test_org_id,
                    provider=provider,
                    key_name="default",
                    api_key_data=test_credentials
                )
            
            # List API keys
            api_keys = await vault.list_api_keys(session, test_org_id)
            
            assert len(api_keys) >= len(providers), "Should list all stored API keys"
            
            # Verify providers are in the list
            listed_providers = [key["provider"] for key in api_keys]
            for provider in providers:
                assert provider in listed_providers, f"Provider {provider} should be in the list"

class TestEnhancedJWT(TestSecurityInfrastructure):
    """Test Enhanced JWT functionality"""
    
    @pytest.mark.asyncio
    async def test_create_token_pair(self, setup_database, test_user_id, test_org_id, device_info):
        """Test creating JWT token pair"""
        jwt_manager = get_enhanced_jwt_manager()
        
        device_fingerprint = DeviceFingerprint.generate_fingerprint(
            user_agent=device_info["user_agent"],
            ip_address=device_info["ip_address"]
        )
        
        async with get_async_session() as session:
            access_token, refresh_token, session_info = await jwt_manager.create_token_pair(
                session=session,
                user_id=test_user_id,
                org_id=test_org_id,
                email="test@example.com",
                roles=["user", "admin"],
                permissions=["read", "write"],
                device_fingerprint=device_fingerprint,
                ip_address=device_info["ip_address"],
                user_agent=device_info["user_agent"]
            )
            
            assert access_token is not None, "Access token should be created"
            assert refresh_token is not None, "Refresh token should be created"
            assert session_info["session_id"] is not None, "Session ID should be provided"
    
    @pytest.mark.asyncio
    async def test_verify_access_token(self, setup_database, test_user_id, test_org_id, device_info):
        """Test verifying access token"""
        jwt_manager = get_enhanced_jwt_manager()
        
        device_fingerprint = DeviceFingerprint.generate_fingerprint(
            user_agent=device_info["user_agent"],
            ip_address=device_info["ip_address"]
        )
        
        async with get_async_session() as session:
            # Create token pair
            access_token, _, _ = await jwt_manager.create_token_pair(
                session=session,
                user_id=test_user_id,
                org_id=test_org_id,
                email="test@example.com",
                roles=["user"],
                permissions=["read"],
                device_fingerprint=device_fingerprint,
                ip_address=device_info["ip_address"],
                user_agent=device_info["user_agent"]
            )
            
            # Verify token
            payload = await jwt_manager.verify_access_token(session, access_token)
            
            assert payload["sub"] == test_user_id, "User ID should match"
            assert payload["org_id"] == test_org_id, "Org ID should match"
            assert "user" in payload["roles"], "User role should be present"
    
    @pytest.mark.asyncio
    async def test_refresh_token(self, setup_database, test_user_id, test_org_id, device_info):
        """Test refreshing access token"""
        jwt_manager = get_enhanced_jwt_manager()
        
        device_fingerprint = DeviceFingerprint.generate_fingerprint(
            user_agent=device_info["user_agent"],
            ip_address=device_info["ip_address"]
        )
        
        async with get_async_session() as session:
            # Create initial token pair
            _, refresh_token, _ = await jwt_manager.create_token_pair(
                session=session,
                user_id=test_user_id,
                org_id=test_org_id,
                email="test@example.com",
                roles=["user"],
                permissions=["read"],
                device_fingerprint=device_fingerprint,
                ip_address=device_info["ip_address"],
                user_agent=device_info["user_agent"]
            )
            
            # Refresh token
            new_access_token, new_refresh_token = await jwt_manager.refresh_access_token(
                session=session,
                refresh_token=refresh_token,
                device_fingerprint=device_fingerprint,
                ip_address=device_info["ip_address"],
                user_agent=device_info["user_agent"]
            )
            
            assert new_access_token is not None, "New access token should be created"
            assert new_refresh_token is not None, "New refresh token should be created"
            assert new_refresh_token != refresh_token, "New refresh token should be different"

class TestAuditLogger(TestSecurityInfrastructure):
    """Test Security Audit Logger functionality"""
    
    @pytest.mark.asyncio
    async def test_log_security_event(self, setup_database, test_user_id, test_org_id):
        """Test logging security events"""
        audit_logger = SecurityAuditLogger()
        
        async with get_async_session() as session:
            # Log a security event
            await audit_logger.log_security_event(
                session=session,
                event_type=SecurityEventType.LOGIN_SUCCESS,
                user_id=test_user_id,
                org_id=test_org_id,
                action="user_login",
                ip_address="192.168.1.100",
                user_agent="Test User Agent",
                success=True,
                details={"test": "data"}
            )
            
            # Verify event was logged (this would require querying the audit log table)
            # For now, we just ensure no exception was raised
            assert True, "Security event should be logged without error"
    
    @pytest.mark.asyncio
    async def test_log_api_key_access(self, setup_database, test_org_id):
        """Test logging API key access events"""
        audit_logger = SecurityAuditLogger()
        
        async with get_async_session() as session:
            # Log API key access
            await audit_logger.log_api_key_access(
                session=session,
                org_id=test_org_id,
                provider="test_provider",
                key_name="test_key",
                action="retrieve",
                success=True,
                user_id="test_user"
            )
            
            assert True, "API key access should be logged without error"

class TestEnhancedCredentialManager(TestSecurityInfrastructure):
    """Test Enhanced Credential Manager functionality"""
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_credentials(self, setup_database, test_org_id, test_credentials):
        """Test storing and retrieving credentials via enhanced manager"""
        manager = EnhancedCredentialManager(EnhancedCredentialProvider.VAULT)
        
        # Store credentials
        success = await manager.store_credentials_in_vault(
            org_id=test_org_id,
            provider="meta",
            credentials=test_credentials,
            created_by="test_user"
        )
        
        assert success, "Should successfully store credentials"
        
        # Retrieve credentials
        retrieved = await manager.get_meta_business_credentials(test_org_id)
        
        assert retrieved is not None, "Should retrieve stored credentials"
        assert retrieved["api_key"] == test_credentials["api_key"]
    
    @pytest.mark.asyncio
    async def test_hybrid_credential_retrieval(self, setup_database, test_org_id):
        """Test hybrid credential retrieval (vault + environment fallback)"""
        manager = EnhancedCredentialManager(EnhancedCredentialProvider.HYBRID)
        
        # Set environment variables for fallback
        os.environ["META_ACCESS_TOKEN"] = "env-token-123"
        os.environ["META_APP_ID"] = "env-app-456"
        os.environ["META_APP_SECRET"] = "env-secret-789"
        
        try:
            # Should fallback to environment since no vault credentials exist
            credentials = await manager.get_meta_business_credentials(test_org_id)
            
            assert credentials is not None, "Should retrieve credentials from environment"
            assert credentials["access_token"] == "env-token-123"
            
        finally:
            # Clean up environment variables
            for key in ["META_ACCESS_TOKEN", "META_APP_ID", "META_APP_SECRET"]:
                if key in os.environ:
                    del os.environ[key]
    
    @pytest.mark.asyncio
    async def test_credential_rotation(self, setup_database, test_org_id, test_credentials):
        """Test credential rotation via enhanced manager"""
        manager = EnhancedCredentialManager(EnhancedCredentialProvider.VAULT)
        
        # Store initial credentials
        await manager.store_credentials_in_vault(
            org_id=test_org_id,
            provider="google",
            credentials=test_credentials
        )
        
        # Rotate credentials
        new_credentials = {
            "developer_token": "new-dev-token",
            "client_id": "new-client-id",
            "client_secret": "new-client-secret",
            "refresh_token": "new-refresh-token"
        }
        
        success = await manager.rotate_credentials(
            org_id=test_org_id,
            provider="google",
            new_credentials=new_credentials,
            rotated_by="test_user"
        )
        
        assert success, "Should successfully rotate credentials"
        
        # Verify new credentials
        retrieved = await manager.get_google_ads_credentials(test_org_id)
        assert retrieved["developer_token"] == new_credentials["developer_token"]

class TestDeviceFingerprinting(TestSecurityInfrastructure):
    """Test Device Fingerprinting functionality"""
    
    def test_generate_fingerprint(self, device_info):
        """Test device fingerprint generation"""
        fingerprint = DeviceFingerprint.generate_fingerprint(
            user_agent=device_info["user_agent"],
            ip_address=device_info["ip_address"]
        )
        
        assert fingerprint is not None, "Should generate fingerprint"
        assert len(fingerprint) == 64, "Fingerprint should be 64 characters (SHA256 hex)"
    
    def test_fingerprint_consistency(self, device_info):
        """Test that same device info generates same fingerprint"""
        fingerprint1 = DeviceFingerprint.generate_fingerprint(
            user_agent=device_info["user_agent"],
            ip_address=device_info["ip_address"]
        )
        
        fingerprint2 = DeviceFingerprint.generate_fingerprint(
            user_agent=device_info["user_agent"],
            ip_address=device_info["ip_address"]
        )
        
        assert fingerprint1 == fingerprint2, "Same device info should generate same fingerprint"
    
    def test_fingerprint_difference(self, device_info):
        """Test that different device info generates different fingerprints"""
        fingerprint1 = DeviceFingerprint.generate_fingerprint(
            user_agent=device_info["user_agent"],
            ip_address=device_info["ip_address"]
        )
        
        fingerprint2 = DeviceFingerprint.generate_fingerprint(
            user_agent="Different User Agent",
            ip_address=device_info["ip_address"]
        )
        
        assert fingerprint1 != fingerprint2, "Different device info should generate different fingerprints"

class TestIntegrationScenarios(TestSecurityInfrastructure):
    """Test complete integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_complete_authentication_flow(self, setup_database, test_user_id, test_org_id, device_info):
        """Test complete authentication and credential access flow"""
        jwt_manager = get_enhanced_jwt_manager()
        credential_manager = EnhancedCredentialManager(EnhancedCredentialProvider.VAULT)
        
        # Generate device fingerprint
        device_fingerprint = DeviceFingerprint.generate_fingerprint(
            user_agent=device_info["user_agent"],
            ip_address=device_info["ip_address"]
        )
        
        async with get_async_session() as session:
            # 1. Create authentication session
            access_token, refresh_token, session_info = await jwt_manager.create_token_pair(
                session=session,
                user_id=test_user_id,
                org_id=test_org_id,
                email="test@example.com",
                roles=["user"],
                permissions=["api_access"],
                device_fingerprint=device_fingerprint,
                ip_address=device_info["ip_address"],
                user_agent=device_info["user_agent"]
            )
            
            # 2. Store API credentials
            test_credentials = {
                "access_token": "meta-token-123",
                "app_id": "meta-app-456",
                "app_secret": "meta-secret-789"
            }
            
            success = await credential_manager.store_credentials_in_vault(
                org_id=test_org_id,
                provider="meta",
                credentials=test_credentials,
                created_by=test_user_id
            )
            
            assert success, "Should store credentials successfully"
            
            # 3. Verify authentication token
            payload = await jwt_manager.verify_access_token(session, access_token)
            assert payload["sub"] == test_user_id
            
            # 4. Access stored credentials
            retrieved_credentials = await credential_manager.get_meta_business_credentials(test_org_id)
            assert retrieved_credentials is not None
            assert retrieved_credentials["access_token"] == test_credentials["access_token"]
            
            # 5. Refresh authentication token
            new_access_token, new_refresh_token = await jwt_manager.refresh_access_token(
                session=session,
                refresh_token=refresh_token,
                device_fingerprint=device_fingerprint,
                ip_address=device_info["ip_address"],
                user_agent=device_info["user_agent"]
            )
            
            assert new_access_token != access_token, "New access token should be different"

# Test configuration
if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])