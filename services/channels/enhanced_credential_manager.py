"""
Enhanced Credential Manager for Channels Service
Secure credential management with enterprise-grade encryption and audit logging
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import json
import sys
import os

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.security.api_key_vault import get_api_key_vault, APIKeyVault
from shared.security.audit_logger import SecurityAuditLogger, SecurityEventType
from shared.database.database import get_async_session
from shared.database.security_models import EncryptedAPIKey
from shared.utils.logging import setup_logging
from sqlalchemy import select, and_

logger = setup_logging("channels_credential_manager")

class EnhancedCredentialManager:
    """Enhanced credential manager with enterprise security for channels service"""
    
    def __init__(self):
        self.api_key_vault: Optional[APIKeyVault] = None
        self.audit_logger = SecurityAuditLogger()
        self.credential_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 300  # 5 minutes cache TTL
        
        # Channel-specific credential mappings
        self.channel_credential_types = {
            "facebook": ["access_token", "app_secret", "app_id"],
            "google": ["client_id", "client_secret", "refresh_token"],
            "klaviyo": ["api_key", "private_key"],
            "shopify": ["access_token", "shop_domain", "api_key"],
            "amazon": ["access_key_id", "secret_access_key", "marketplace_id"],
            "salesforce": ["client_id", "client_secret", "refresh_token", "instance_url"],
            "tiktok": ["access_token", "app_id", "app_secret"],
            "linkedin": ["access_token", "client_id", "client_secret"],
            "stripe": ["secret_key", "publishable_key", "webhook_secret"],
            "paypal": ["client_id", "client_secret", "webhook_id"],
            "meta": ["access_token", "app_secret", "app_id", "business_id"],
            "twitter": ["api_key", "api_secret", "access_token", "access_token_secret"],
            "snapchat": ["client_id", "client_secret", "refresh_token"],
            "pinterest": ["access_token", "app_id", "app_secret"],
            "youtube": ["api_key", "client_id", "client_secret"],
            "mailchimp": ["api_key", "server_prefix"],
            "hubspot": ["access_token", "refresh_token", "client_id"],
            "zendesk": ["api_token", "email", "subdomain"],
            "intercom": ["access_token", "app_id"],
            "segment": ["write_key", "source_id"]
        }
    
    async def initialize(self):
        """Initialize the credential manager"""
        try:
            self.api_key_vault = get_api_key_vault()
            logger.info("Enhanced credential manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize credential manager: {e}")
            raise
    
    async def store_credential(
        self,
        provider: str,
        credential_type: str,
        credential_value: str,
        user_id: str,
        org_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a credential securely with audit logging"""
        try:
            if not self.api_key_vault:
                raise ValueError("Credential manager not initialized")
            
            # Validate credential type for provider
            if provider in self.channel_credential_types:
                valid_types = self.channel_credential_types[provider]
                if credential_type not in valid_types:
                    logger.warning(f"Unusual credential type '{credential_type}' for provider '{provider}'")
            
            # Store credential in vault
            credential_id = await self.api_key_vault.store_api_key(
                provider=provider,
                key_name=credential_type,
                api_key=credential_value,
                user_id=user_id,
                org_id=org_id,
                metadata=metadata or {}
            )
            
            # Log credential storage
            await self.audit_logger.log_event(
                event_type=SecurityEventType.API_KEY_CREATED,
                action="credential_stored",
                user_id=user_id,
                org_id=org_id,
                details={
                    "provider": provider,
                    "credential_type": credential_type,
                    "credential_id": credential_id,
                    "service": "channels"
                }
            )
            
            # Clear cache for this provider/user combination
            cache_key = f"{provider}_{user_id}_{org_id}"
            if cache_key in self.credential_cache:
                del self.credential_cache[cache_key]
            
            logger.info(f"Credential stored: {provider}.{credential_type} for user {user_id}")
            return credential_id
            
        except Exception as e:
            await self.audit_logger.log_event(
                event_type=SecurityEventType.SECURITY_VIOLATION,
                action="credential_storage_failed",
                user_id=user_id,
                org_id=org_id,
                details={
                    "provider": provider,
                    "credential_type": credential_type,
                    "error": str(e),
                    "service": "channels"
                }
            )
            logger.error(f"Failed to store credential: {e}")
            raise
    
    async def retrieve_credential(
        self,
        credential_id: str,
        user_id: str,
        org_id: str
    ) -> str:
        """Retrieve a credential with audit logging"""
        try:
            if not self.api_key_vault:
                raise ValueError("Credential manager not initialized")
            
            # Check cache first
            cache_key = f"cred_{credential_id}_{user_id}_{org_id}"
            if cache_key in self.credential_cache:
                cache_entry = self.credential_cache[cache_key]
                if datetime.now(timezone.utc) < cache_entry["expires_at"]:
                    return cache_entry["value"]
                else:
                    del self.credential_cache[cache_key]
            
            # Retrieve from vault
            credential_value = await self.api_key_vault.get_api_key(
                credential_id=credential_id,
                user_id=user_id,
                org_id=org_id
            )
            
            # Cache the credential
            self.credential_cache[cache_key] = {
                "value": credential_value,
                "expires_at": datetime.now(timezone.utc) + timedelta(seconds=self.cache_ttl)
            }
            
            # Log credential access
            await self.audit_logger.log_event(
                event_type=SecurityEventType.API_KEY_ACCESS,
                action="credential_retrieved",
                user_id=user_id,
                org_id=org_id,
                details={
                    "credential_id": credential_id,
                    "service": "channels"
                }
            )
            
            return credential_value
            
        except Exception as e:
            await self.audit_logger.log_event(
                event_type=SecurityEventType.SECURITY_VIOLATION,
                action="credential_retrieval_failed",
                user_id=user_id,
                org_id=org_id,
                details={
                    "credential_id": credential_id,
                    "error": str(e),
                    "service": "channels"
                }
            )
            logger.error(f"Failed to retrieve credential: {e}")
            raise
    
    async def get_provider_credentials(
        self,
        provider: str,
        user_id: str,
        org_id: str
    ) -> Dict[str, str]:
        """Get all credentials for a specific provider"""
        try:
            if not self.api_key_vault:
                raise ValueError("Credential manager not initialized")
            
            # Check cache first
            cache_key = f"{provider}_{user_id}_{org_id}"
            if cache_key in self.credential_cache:
                cache_entry = self.credential_cache[cache_key]
                if datetime.now(timezone.utc) < cache_entry["expires_at"]:
                    return cache_entry["credentials"]
                else:
                    del self.credential_cache[cache_key]
            
            # Get credentials from database
            async with get_async_session() as session:
                result = await session.execute(
                    select(EncryptedAPIKey)
                    .where(
                        and_(
                            EncryptedAPIKey.provider == provider,
                            EncryptedAPIKey.user_id == user_id,
                            EncryptedAPIKey.org_id == org_id,
                            EncryptedAPIKey.is_active == True
                        )
                    )
                )
                api_keys = result.scalars().all()
            
            # Decrypt credentials
            credentials = {}
            for api_key in api_keys:
                try:
                    decrypted_value = await self.api_key_vault.get_api_key(
                        credential_id=api_key.id,
                        user_id=user_id,
                        org_id=org_id
                    )
                    credentials[api_key.key_name] = decrypted_value
                except Exception as e:
                    logger.error(f"Failed to decrypt credential {api_key.id}: {e}")
                    continue
            
            # Cache the credentials
            self.credential_cache[cache_key] = {
                "credentials": credentials,
                "expires_at": datetime.now(timezone.utc) + timedelta(seconds=self.cache_ttl)
            }
            
            # Log provider credential access
            await self.audit_logger.log_event(
                event_type=SecurityEventType.API_KEY_ACCESS,
                action="provider_credentials_retrieved",
                user_id=user_id,
                org_id=org_id,
                details={
                    "provider": provider,
                    "credential_count": len(credentials),
                    "credential_types": list(credentials.keys()),
                    "service": "channels"
                }
            )
            
            return credentials
            
        except Exception as e:
            await self.audit_logger.log_event(
                event_type=SecurityEventType.SECURITY_VIOLATION,
                action="provider_credentials_failed",
                user_id=user_id,
                org_id=org_id,
                details={
                    "provider": provider,
                    "error": str(e),
                    "service": "channels"
                }
            )
            logger.error(f"Failed to get provider credentials: {e}")
            raise
    
    async def update_credential(
        self,
        credential_id: str,
        new_value: str,
        user_id: str,
        org_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update an existing credential"""
        try:
            if not self.api_key_vault:
                raise ValueError("Credential manager not initialized")
            
            # Update credential in vault
            success = await self.api_key_vault.update_api_key(
                credential_id=credential_id,
                new_api_key=new_value,
                user_id=user_id,
                org_id=org_id,
                metadata=metadata
            )
            
            if success:
                # Clear relevant caches
                cache_keys_to_remove = [
                    key for key in self.credential_cache.keys()
                    if f"_{user_id}_{org_id}" in key
                ]
                for key in cache_keys_to_remove:
                    del self.credential_cache[key]
                
                # Log credential update
                await self.audit_logger.log_event(
                    event_type=SecurityEventType.API_KEY_UPDATED,
                    action="credential_updated",
                    user_id=user_id,
                    org_id=org_id,
                    details={
                        "credential_id": credential_id,
                        "service": "channels"
                    }
                )
                
                logger.info(f"Credential updated: {credential_id} for user {user_id}")
            
            return success
            
        except Exception as e:
            await self.audit_logger.log_event(
                event_type=SecurityEventType.SECURITY_VIOLATION,
                action="credential_update_failed",
                user_id=user_id,
                org_id=org_id,
                details={
                    "credential_id": credential_id,
                    "error": str(e),
                    "service": "channels"
                }
            )
            logger.error(f"Failed to update credential: {e}")
            raise
    
    async def delete_credential(
        self,
        credential_id: str,
        user_id: str,
        org_id: str
    ) -> bool:
        """Delete a credential"""
        try:
            if not self.api_key_vault:
                raise ValueError("Credential manager not initialized")
            
            # Delete credential from vault
            success = await self.api_key_vault.delete_api_key(
                credential_id=credential_id,
                user_id=user_id,
                org_id=org_id
            )
            
            if success:
                # Clear relevant caches
                cache_keys_to_remove = [
                    key for key in self.credential_cache.keys()
                    if f"_{user_id}_{org_id}" in key
                ]
                for key in cache_keys_to_remove:
                    del self.credential_cache[key]
                
                # Log credential deletion
                await self.audit_logger.log_event(
                    event_type=SecurityEventType.API_KEY_DELETED,
                    action="credential_deleted",
                    user_id=user_id,
                    org_id=org_id,
                    details={
                        "credential_id": credential_id,
                        "service": "channels"
                    }
                )
                
                logger.info(f"Credential deleted: {credential_id} for user {user_id}")
            
            return success
            
        except Exception as e:
            await self.audit_logger.log_event(
                event_type=SecurityEventType.SECURITY_VIOLATION,
                action="credential_deletion_failed",
                user_id=user_id,
                org_id=org_id,
                details={
                    "credential_id": credential_id,
                    "error": str(e),
                    "service": "channels"
                }
            )
            logger.error(f"Failed to delete credential: {e}")
            raise
    
    async def list_provider_credentials(
        self,
        provider: str,
        user_id: str,
        org_id: str
    ) -> List[Dict[str, Any]]:
        """List all credentials for a provider (metadata only, no values)"""
        try:
            async with get_async_session() as session:
                result = await session.execute(
                    select(EncryptedAPIKey)
                    .where(
                        and_(
                            EncryptedAPIKey.provider == provider,
                            EncryptedAPIKey.user_id == user_id,
                            EncryptedAPIKey.org_id == org_id,
                            EncryptedAPIKey.is_active == True
                        )
                    )
                    .order_by(EncryptedAPIKey.created_at.desc())
                )
                api_keys = result.scalars().all()
            
            credentials_info = []
            for api_key in api_keys:
                credentials_info.append({
                    "id": api_key.id,
                    "provider": api_key.provider,
                    "key_name": api_key.key_name,
                    "created_at": api_key.created_at,
                    "last_used": api_key.last_used,
                    "expires_at": api_key.expires_at,
                    "metadata": api_key.metadata
                })
            
            # Log credential listing
            await self.audit_logger.log_event(
                event_type=SecurityEventType.RESOURCE_ACCESSED,
                action="credentials_listed",
                user_id=user_id,
                org_id=org_id,
                details={
                    "provider": provider,
                    "credential_count": len(credentials_info),
                    "service": "channels"
                }
            )
            
            return credentials_info
            
        except Exception as e:
            await self.audit_logger.log_event(
                event_type=SecurityEventType.SECURITY_VIOLATION,
                action="credential_listing_failed",
                user_id=user_id,
                org_id=org_id,
                details={
                    "provider": provider,
                    "error": str(e),
                    "service": "channels"
                }
            )
            logger.error(f"Failed to list credentials: {e}")
            raise
    
    async def validate_provider_credentials(
        self,
        provider: str,
        credentials: Dict[str, str]
    ) -> Dict[str, Any]:
        """Validate credentials for a specific provider"""
        try:
            # Get expected credential types for provider
            expected_types = self.channel_credential_types.get(provider, [])
            
            validation_result = {
                "valid": True,
                "missing_credentials": [],
                "unexpected_credentials": [],
                "warnings": []
            }
            
            # Check for missing required credentials
            for expected_type in expected_types:
                if expected_type not in credentials:
                    validation_result["missing_credentials"].append(expected_type)
                    validation_result["valid"] = False
            
            # Check for unexpected credentials
            for credential_type in credentials.keys():
                if credential_type not in expected_types:
                    validation_result["unexpected_credentials"].append(credential_type)
                    validation_result["warnings"].append(
                        f"Unexpected credential type '{credential_type}' for provider '{provider}'"
                    )
            
            # Provider-specific validation
            if provider == "facebook" or provider == "meta":
                if "access_token" in credentials and not credentials["access_token"].startswith("EAA"):
                    validation_result["warnings"].append("Facebook access token format may be invalid")
            
            elif provider == "google":
                if "client_id" in credentials and not credentials["client_id"].endswith(".googleusercontent.com"):
                    validation_result["warnings"].append("Google client ID format may be invalid")
            
            elif provider == "stripe":
                if "secret_key" in credentials and not credentials["secret_key"].startswith("sk_"):
                    validation_result["warnings"].append("Stripe secret key format may be invalid")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Failed to validate credentials: {e}")
            return {
                "valid": False,
                "error": str(e),
                "missing_credentials": [],
                "unexpected_credentials": [],
                "warnings": []
            }
    
    def get_supported_providers(self) -> List[str]:
        """Get list of supported providers"""
        return list(self.channel_credential_types.keys())
    
    def get_provider_credential_types(self, provider: str) -> List[str]:
        """Get expected credential types for a provider"""
        return self.channel_credential_types.get(provider, [])
    
    async def clear_cache(self):
        """Clear the credential cache"""
        self.credential_cache.clear()
        logger.info("Credential cache cleared")