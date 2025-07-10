"""
Enhanced Credential Management for Data Ingestion Service
Integrates with enterprise API key vault for secure credential storage and retrieval
"""

import os
import sys
import asyncio
from typing import Dict, Optional, List, Any
from enum import Enum
import logging

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from shared.security.api_key_vault import APIKeyVault, get_api_key_vault
from shared.security.audit_logger import SecurityAuditLogger, SecurityEventType
from shared.database.database import get_async_session
from .credential_manager import CredentialManager, CredentialProvider

logger = logging.getLogger(__name__)

class EnhancedCredentialProvider(Enum):
    """Enhanced credential providers with vault support"""
    ENVIRONMENT = "environment"
    FILE = "file"
    VAULT = "vault"
    HYBRID = "hybrid"  # Try vault first, fallback to environment

class EnhancedCredentialManager(CredentialManager):
    """
    Enhanced credential manager with enterprise security features
    Integrates with API key vault for secure storage and comprehensive audit logging
    """
    
    def __init__(self, provider: EnhancedCredentialProvider = EnhancedCredentialProvider.HYBRID):
        # Initialize parent class with compatible provider
        if provider == EnhancedCredentialProvider.VAULT or provider == EnhancedCredentialProvider.HYBRID:
            super().__init__(CredentialProvider.ENVIRONMENT)  # Fallback
        else:
            super().__init__(CredentialProvider(provider.value))
        
        self.enhanced_provider = provider
        self.api_key_vault = get_api_key_vault()
        self.audit_logger = SecurityAuditLogger()
        self._vault_cache = {}
        
        # Provider mapping for vault integration
        self.provider_mapping = {
            "meta_business": "meta",
            "google_ads": "google",
            "klaviyo": "klaviyo",
            "shopify": "shopify",
            "woocommerce": "woocommerce",
            "amazon": "amazon",
            "hubspot": "hubspot",
            "salesforce": "salesforce",
            "stripe": "stripe",
            "paypal": "paypal",
            "tiktok": "tiktok",
            "snowflake": "snowflake",
            "databricks": "databricks",
            "zoho_crm": "zoho",
            "linkedin_ads": "linkedin",
            "x_ads": "twitter"
        }
    
    async def _get_credentials_from_vault(
        self,
        org_id: str,
        provider: str,
        key_name: str = "default"
    ) -> Optional[Dict[str, str]]:
        """Get credentials from the secure vault"""
        try:
            cache_key = f"vault_{org_id}_{provider}_{key_name}"
            
            # Check cache first
            if cache_key in self._vault_cache:
                return self._vault_cache[cache_key]
            
            async with get_async_session() as session:
                # Retrieve encrypted credentials from vault
                credentials = await self.api_key_vault.get_api_key(
                    session=session,
                    org_id=org_id,
                    provider=provider,
                    key_name=key_name
                )
                
                if credentials:
                    # Cache the decrypted credentials temporarily
                    self._vault_cache[cache_key] = credentials
                    
                    # Log successful retrieval
                    await self.audit_logger.log_api_key_access(
                        session=session,
                        org_id=org_id,
                        provider=provider,
                        key_name=key_name,
                        action="retrieve",
                        success=True
                    )
                    
                    logger.info(f"Retrieved {provider} credentials from vault for org {org_id}")
                    return credentials
                else:
                    logger.warning(f"No {provider} credentials found in vault for org {org_id}")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to retrieve {provider} credentials from vault: {e}")
            
            # Log failed retrieval
            try:
                async with get_async_session() as session:
                    await self.audit_logger.log_api_key_access(
                        session=session,
                        org_id=org_id,
                        provider=provider,
                        key_name=key_name,
                        action="retrieve",
                        success=False,
                        error_message=str(e)
                    )
            except:
                pass  # Don't fail on audit logging errors
            
            return None
    
    async def _get_credentials_hybrid(
        self,
        org_id: str,
        provider: str,
        env_method,
        file_method
    ) -> Optional[Dict[str, str]]:
        """Get credentials using hybrid approach (vault first, then fallback)"""
        
        # Try vault first
        if provider in self.provider_mapping:
            vault_provider = self.provider_mapping[provider]
            credentials = await self._get_credentials_from_vault(org_id, vault_provider)
            if credentials:
                return credentials
        
        # Fallback to environment variables
        if env_method:
            credentials = env_method()
            if credentials:
                logger.info(f"Retrieved {provider} credentials from environment (fallback)")
                return credentials
        
        # Fallback to file
        if file_method:
            credentials = file_method(org_id)
            if credentials:
                logger.info(f"Retrieved {provider} credentials from file (fallback)")
                return credentials
        
        return None
    
    # Override all credential methods to use enhanced vault integration
    
    async def get_meta_business_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get Meta Business API credentials with vault integration"""
        cache_key = f"meta_business_{org_id}"
        
        if cache_key in self._credentials_cache:
            return self._credentials_cache[cache_key]
        
        credentials = None
        
        if self.enhanced_provider == EnhancedCredentialProvider.VAULT:
            credentials = await self._get_credentials_from_vault(org_id, "meta")
        elif self.enhanced_provider == EnhancedCredentialProvider.HYBRID:
            credentials = await self._get_credentials_hybrid(
                org_id, "meta_business",
                self._get_meta_env_credentials,
                self._get_meta_file_credentials
            )
        else:
            # Use parent class method for environment/file
            credentials = await super().get_meta_business_credentials(org_id)
        
        if credentials:
            self._credentials_cache[cache_key] = credentials
        
        return credentials
    
    async def get_google_ads_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get Google Ads API credentials with vault integration"""
        cache_key = f"google_ads_{org_id}"
        
        if cache_key in self._credentials_cache:
            return self._credentials_cache[cache_key]
        
        credentials = None
        
        if self.enhanced_provider == EnhancedCredentialProvider.VAULT:
            credentials = await self._get_credentials_from_vault(org_id, "google")
        elif self.enhanced_provider == EnhancedCredentialProvider.HYBRID:
            credentials = await self._get_credentials_hybrid(
                org_id, "google_ads",
                self._get_google_env_credentials,
                self._get_google_file_credentials
            )
        else:
            credentials = await super().get_google_ads_credentials(org_id)
        
        if credentials:
            self._credentials_cache[cache_key] = credentials
        
        return credentials
    
    async def get_klaviyo_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get Klaviyo API credentials with vault integration"""
        cache_key = f"klaviyo_{org_id}"
        
        if cache_key in self._credentials_cache:
            return self._credentials_cache[cache_key]
        
        credentials = None
        
        if self.enhanced_provider == EnhancedCredentialProvider.VAULT:
            credentials = await self._get_credentials_from_vault(org_id, "klaviyo")
        elif self.enhanced_provider == EnhancedCredentialProvider.HYBRID:
            credentials = await self._get_credentials_hybrid(
                org_id, "klaviyo",
                self._get_klaviyo_env_credentials,
                self._get_klaviyo_file_credentials
            )
        else:
            credentials = await super().get_klaviyo_credentials(org_id)
        
        if credentials:
            self._credentials_cache[cache_key] = credentials
        
        return credentials
    
    async def get_shopify_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get Shopify API credentials with vault integration"""
        cache_key = f"shopify_{org_id}"
        
        if cache_key in self._credentials_cache:
            return self._credentials_cache[cache_key]
        
        credentials = None
        
        if self.enhanced_provider == EnhancedCredentialProvider.VAULT:
            credentials = await self._get_credentials_from_vault(org_id, "shopify")
        elif self.enhanced_provider == EnhancedCredentialProvider.HYBRID:
            credentials = await self._get_credentials_hybrid(
                org_id, "shopify",
                self._get_shopify_env_credentials,
                self._get_shopify_file_credentials
            )
        else:
            credentials = await super().get_shopify_credentials(org_id)
        
        if credentials:
            self._credentials_cache[cache_key] = credentials
        
        return credentials
    
    async def get_amazon_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get Amazon Seller Central API credentials with vault integration"""
        cache_key = f"amazon_{org_id}"
        
        if cache_key in self._credentials_cache:
            return self._credentials_cache[cache_key]
        
        credentials = None
        
        if self.enhanced_provider == EnhancedCredentialProvider.VAULT:
            credentials = await self._get_credentials_from_vault(org_id, "amazon")
        elif self.enhanced_provider == EnhancedCredentialProvider.HYBRID:
            credentials = await self._get_credentials_hybrid(
                org_id, "amazon",
                self._get_amazon_env_credentials,
                self._get_amazon_file_credentials
            )
        else:
            credentials = await super().get_amazon_credentials(org_id)
        
        if credentials:
            self._credentials_cache[cache_key] = credentials
        
        return credentials
    
    async def get_salesforce_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get Salesforce CRM API credentials with vault integration"""
        cache_key = f"salesforce_{org_id}"
        
        if cache_key in self._credentials_cache:
            return self._credentials_cache[cache_key]
        
        credentials = None
        
        if self.enhanced_provider == EnhancedCredentialProvider.VAULT:
            credentials = await self._get_credentials_from_vault(org_id, "salesforce")
        elif self.enhanced_provider == EnhancedCredentialProvider.HYBRID:
            credentials = await self._get_credentials_hybrid(
                org_id, "salesforce",
                self._get_salesforce_env_credentials,
                self._get_salesforce_file_credentials
            )
        else:
            credentials = await super().get_salesforce_credentials(org_id)
        
        if credentials:
            self._credentials_cache[cache_key] = credentials
        
        return credentials
    
    async def get_stripe_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get Stripe Payment API credentials with vault integration"""
        cache_key = f"stripe_{org_id}"
        
        if cache_key in self._credentials_cache:
            return self._credentials_cache[cache_key]
        
        credentials = None
        
        if self.enhanced_provider == EnhancedCredentialProvider.VAULT:
            credentials = await self._get_credentials_from_vault(org_id, "stripe")
        elif self.enhanced_provider == EnhancedCredentialProvider.HYBRID:
            credentials = await self._get_credentials_hybrid(
                org_id, "stripe",
                self._get_stripe_env_credentials,
                self._get_stripe_file_credentials
            )
        else:
            credentials = await super().get_stripe_credentials(org_id)
        
        if credentials:
            self._credentials_cache[cache_key] = credentials
        
        return credentials
    
    async def get_tiktok_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get TikTok for Business API credentials with vault integration"""
        cache_key = f"tiktok_{org_id}"
        
        if cache_key in self._credentials_cache:
            return self._credentials_cache[cache_key]
        
        credentials = None
        
        if self.enhanced_provider == EnhancedCredentialProvider.VAULT:
            credentials = await self._get_credentials_from_vault(org_id, "tiktok")
        elif self.enhanced_provider == EnhancedCredentialProvider.HYBRID:
            credentials = await self._get_credentials_hybrid(
                org_id, "tiktok",
                self._get_tiktok_env_credentials,
                self._get_tiktok_file_credentials
            )
        else:
            credentials = await super().get_tiktok_credentials(org_id)
        
        if credentials:
            self._credentials_cache[cache_key] = credentials
        
        return credentials
    
    async def get_linkedin_ads_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get LinkedIn Ads API credentials with vault integration"""
        cache_key = f"linkedin_ads_{org_id}"
        
        if cache_key in self._credentials_cache:
            return self._credentials_cache[cache_key]
        
        credentials = None
        
        if self.enhanced_provider == EnhancedCredentialProvider.VAULT:
            credentials = await self._get_credentials_from_vault(org_id, "linkedin")
        elif self.enhanced_provider == EnhancedCredentialProvider.HYBRID:
            credentials = await self._get_credentials_hybrid(
                org_id, "linkedin_ads",
                self._get_linkedin_ads_env_credentials,
                self._get_linkedin_ads_file_credentials
            )
        else:
            credentials = await super().get_linkedin_ads_credentials(org_id)
        
        if credentials:
            self._credentials_cache[cache_key] = credentials
        
        return credentials
    
    # Vault management methods
    
    async def store_credentials_in_vault(
        self,
        org_id: str,
        provider: str,
        credentials: Dict[str, str],
        key_name: str = "default",
        created_by: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store credentials securely in the vault"""
        try:
            async with get_async_session() as session:
                success = await self.api_key_vault.store_api_key(
                    session=session,
                    org_id=org_id,
                    provider=provider,
                    key_name=key_name,
                    api_key_data=credentials,
                    created_by=created_by,
                    metadata=metadata
                )
                
                if success:
                    # Clear cache to force refresh
                    cache_key = f"vault_{org_id}_{provider}_{key_name}"
                    if cache_key in self._vault_cache:
                        del self._vault_cache[cache_key]
                    
                    logger.info(f"Stored {provider} credentials in vault for org {org_id}")
                    return True
                else:
                    logger.error(f"Failed to store {provider} credentials in vault")
                    return False
                    
        except Exception as e:
            logger.error(f"Error storing credentials in vault: {e}")
            return False
    
    async def rotate_credentials(
        self,
        org_id: str,
        provider: str,
        new_credentials: Dict[str, str],
        key_name: str = "default",
        rotated_by: Optional[str] = None
    ) -> bool:
        """Rotate credentials in the vault"""
        try:
            async with get_async_session() as session:
                success = await self.api_key_vault.rotate_api_key(
                    session=session,
                    org_id=org_id,
                    provider=provider,
                    key_name=key_name,
                    new_api_key_data=new_credentials,
                    rotated_by=rotated_by
                )
                
                if success:
                    # Clear all related caches
                    cache_keys_to_clear = [
                        f"vault_{org_id}_{provider}_{key_name}",
                        f"{provider}_{org_id}"
                    ]
                    
                    for cache_key in cache_keys_to_clear:
                        if cache_key in self._vault_cache:
                            del self._vault_cache[cache_key]
                        if cache_key in self._credentials_cache:
                            del self._credentials_cache[cache_key]
                    
                    logger.info(f"Rotated {provider} credentials for org {org_id}")
                    return True
                else:
                    logger.error(f"Failed to rotate {provider} credentials")
                    return False
                    
        except Exception as e:
            logger.error(f"Error rotating credentials: {e}")
            return False
    
    async def revoke_credentials(
        self,
        org_id: str,
        provider: str,
        key_name: str = "default",
        revoked_by: Optional[str] = None,
        reason: str = "manual_revocation"
    ) -> bool:
        """Revoke credentials in the vault"""
        try:
            async with get_async_session() as session:
                success = await self.api_key_vault.revoke_api_key(
                    session=session,
                    org_id=org_id,
                    provider=provider,
                    key_name=key_name,
                    revoked_by=revoked_by,
                    reason=reason
                )
                
                if success:
                    # Clear all related caches
                    cache_keys_to_clear = [
                        f"vault_{org_id}_{provider}_{key_name}",
                        f"{provider}_{org_id}"
                    ]
                    
                    for cache_key in cache_keys_to_clear:
                        if cache_key in self._vault_cache:
                            del self._vault_cache[cache_key]
                        if cache_key in self._credentials_cache:
                            del self._credentials_cache[cache_key]
                    
                    logger.info(f"Revoked {provider} credentials for org {org_id}")
                    return True
                else:
                    logger.error(f"Failed to revoke {provider} credentials")
                    return False
                    
        except Exception as e:
            logger.error(f"Error revoking credentials: {e}")
            return False
    
    async def list_stored_credentials(self, org_id: str) -> List[Dict[str, Any]]:
        """List all stored credentials for an organization"""
        try:
            async with get_async_session() as session:
                credentials_list = await self.api_key_vault.list_api_keys(session, org_id)
                return credentials_list
                
        except Exception as e:
            logger.error(f"Error listing stored credentials: {e}")
            return []
    
    async def get_credential_usage_analytics(
        self,
        org_id: str,
        provider: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get usage analytics for stored credentials"""
        try:
            async with get_async_session() as session:
                analytics = await self.api_key_vault.get_usage_analytics(
                    session=session,
                    org_id=org_id,
                    provider=provider,
                    days=days
                )
                return analytics
                
        except Exception as e:
            logger.error(f"Error getting credential analytics: {e}")
            return {}
    
    def clear_vault_cache(self):
        """Clear the vault credential cache"""
        self._vault_cache.clear()
        logger.info("Cleared vault credential cache")
    
    def clear_all_caches(self):
        """Clear all credential caches"""
        self.clear_cache()  # Parent class cache
        self.clear_vault_cache()
        logger.info("Cleared all credential caches")

# Global enhanced credential manager instance
_enhanced_credential_manager: Optional[EnhancedCredentialManager] = None

def get_enhanced_credential_manager() -> EnhancedCredentialManager:
    """Get the global enhanced credential manager instance"""
    global _enhanced_credential_manager
    if _enhanced_credential_manager is None:
        _enhanced_credential_manager = EnhancedCredentialManager()
    return _enhanced_credential_manager

# Convenience functions for common operations

async def get_platform_credentials(org_id: str, platform: str) -> Optional[Dict[str, str]]:
    """Get credentials for any supported platform"""
    manager = get_enhanced_credential_manager()
    
    method_map = {
        "meta": manager.get_meta_business_credentials,
        "google": manager.get_google_ads_credentials,
        "klaviyo": manager.get_klaviyo_credentials,
        "shopify": manager.get_shopify_credentials,
        "amazon": manager.get_amazon_credentials,
        "salesforce": manager.get_salesforce_credentials,
        "stripe": manager.get_stripe_credentials,
        "tiktok": manager.get_tiktok_credentials,
        "linkedin": manager.get_linkedin_ads_credentials,
    }
    
    if platform in method_map:
        return await method_map[platform](org_id)
    else:
        logger.warning(f"Unsupported platform: {platform}")
        return None

async def store_platform_credentials(
    org_id: str,
    platform: str,
    credentials: Dict[str, str],
    created_by: Optional[str] = None
) -> bool:
    """Store credentials for any supported platform"""
    manager = get_enhanced_credential_manager()
    return await manager.store_credentials_in_vault(
        org_id=org_id,
        provider=platform,
        credentials=credentials,
        created_by=created_by
    )