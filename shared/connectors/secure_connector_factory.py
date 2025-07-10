"""
Secure Connector Factory
Factory for creating and managing secure API connectors with enterprise security
"""

import asyncio
from typing import Dict, List, Any, Optional, Type
from datetime import datetime, timezone
import sys
import os

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.connectors.base_secure_connector import BaseSecureConnector
from shared.connectors.secure_facebook_connector import SecureFacebookConnector
from shared.connectors.secure_google_connector import SecureGoogleConnector
from shared.security.audit_logger import SecurityAuditLogger, SecurityEventType
from shared.utils.logging import setup_logging

logger = setup_logging("secure_connector_factory")

class SecureConnectorFactory:
    """Factory for creating and managing secure API connectors"""
    
    def __init__(self):
        self.audit_logger = SecurityAuditLogger()
        self.connector_registry: Dict[str, Type[BaseSecureConnector]] = {}
        self.connector_instances: Dict[str, BaseSecureConnector] = {}
        
        # Register built-in connectors
        self._register_builtin_connectors()
    
    def _register_builtin_connectors(self):
        """Register built-in secure connectors"""
        self.register_connector("facebook", SecureFacebookConnector)
        self.register_connector("meta", SecureFacebookConnector)  # Alias for Facebook
        self.register_connector("google", SecureGoogleConnector)
        self.register_connector("google_ads", SecureGoogleConnector)  # Alias for Google
        
        logger.info("Registered built-in secure connectors")
    
    def register_connector(self, provider: str, connector_class: Type[BaseSecureConnector]):
        """Register a new secure connector"""
        try:
            if not issubclass(connector_class, BaseSecureConnector):
                raise ValueError(f"Connector class must inherit from BaseSecureConnector")
            
            self.connector_registry[provider.lower()] = connector_class
            logger.info(f"Registered secure connector for provider: {provider}")
            
        except Exception as e:
            logger.error(f"Failed to register connector for {provider}: {e}")
            raise
    
    async def get_connector(
        self,
        provider: str,
        user_id: str,
        org_id: str,
        ip_address: str
    ) -> BaseSecureConnector:
        """Get or create a secure connector instance"""
        try:
            provider_key = provider.lower()
            
            if provider_key not in self.connector_registry:
                available_providers = list(self.connector_registry.keys())
                raise ValueError(f"Unsupported provider: {provider}. Available: {available_providers}")
            
            # Create instance key for caching
            instance_key = f"{provider_key}_{user_id}_{org_id}"
            
            # Return existing instance if available
            if instance_key in self.connector_instances:
                connector = self.connector_instances[instance_key]
                
                # Log connector access
                await self.audit_logger.log_event(
                    event_type=SecurityEventType.RESOURCE_ACCESSED,
                    action="connector_accessed",
                    user_id=user_id,
                    org_id=org_id,
                    ip_address=ip_address,
                    details={
                        "provider": provider,
                        "connector_type": "cached",
                        "instance_key": instance_key
                    }
                )
                
                return connector
            
            # Create new connector instance
            connector_class = self.connector_registry[provider_key]
            connector = connector_class()
            await connector.initialize()
            
            # Cache the instance
            self.connector_instances[instance_key] = connector
            
            # Log connector creation
            await self.audit_logger.log_event(
                event_type=SecurityEventType.RESOURCE_CREATED,
                action="connector_created",
                user_id=user_id,
                org_id=org_id,
                ip_address=ip_address,
                details={
                    "provider": provider,
                    "connector_type": "new",
                    "instance_key": instance_key,
                    "connector_class": connector_class.__name__
                }
            )
            
            logger.info(f"Created secure connector for {provider} (user: {user_id}, org: {org_id})")
            return connector
            
        except Exception as e:
            await self.audit_logger.log_event(
                event_type=SecurityEventType.SECURITY_VIOLATION,
                action="connector_creation_failed",
                user_id=user_id,
                org_id=org_id,
                ip_address=ip_address,
                details={
                    "provider": provider,
                    "error": str(e)
                }
            )
            logger.error(f"Failed to get connector for {provider}: {e}")
            raise
    
    async def validate_provider_credentials(
        self,
        provider: str,
        credentials: Dict[str, str],
        user_id: str,
        org_id: str,
        ip_address: str
    ) -> Dict[str, Any]:
        """Validate credentials for a specific provider"""
        try:
            connector = await self.get_connector(provider, user_id, org_id, ip_address)
            
            # Log credential validation attempt
            await self.audit_logger.log_event(
                event_type=SecurityEventType.AUTHENTICATION_ATTEMPT,
                action="credential_validation_started",
                user_id=user_id,
                org_id=org_id,
                ip_address=ip_address,
                details={
                    "provider": provider,
                    "credential_types": list(credentials.keys())
                }
            )
            
            validation_result = await connector.validate_credentials(credentials)
            
            # Log validation result
            await self.audit_logger.log_event(
                event_type=SecurityEventType.AUTHENTICATION_SUCCESS if validation_result["valid"] else SecurityEventType.AUTHENTICATION_FAILED,
                action="credential_validation_completed",
                user_id=user_id,
                org_id=org_id,
                ip_address=ip_address,
                details={
                    "provider": provider,
                    "validation_result": validation_result,
                    "success": validation_result["valid"]
                }
            )
            
            return validation_result
            
        except Exception as e:
            await self.audit_logger.log_event(
                event_type=SecurityEventType.SECURITY_VIOLATION,
                action="credential_validation_exception",
                user_id=user_id,
                org_id=org_id,
                ip_address=ip_address,
                details={
                    "provider": provider,
                    "error": str(e)
                }
            )
            logger.error(f"Credential validation failed for {provider}: {e}")
            return {
                "valid": False,
                "error": str(e),
                "missing_credentials": [],
                "unexpected_credentials": [],
                "warnings": []
            }
    
    async def test_provider_connection(
        self,
        provider: str,
        user_id: str,
        org_id: str,
        ip_address: str
    ) -> Dict[str, Any]:
        """Test API connection for a provider"""
        try:
            connector = await self.get_connector(provider, user_id, org_id, ip_address)
            
            # Log connection test attempt
            await self.audit_logger.log_event(
                event_type=SecurityEventType.API_ACCESS,
                action="connection_test_started",
                user_id=user_id,
                org_id=org_id,
                ip_address=ip_address,
                details={
                    "provider": provider,
                    "test_type": "connection"
                }
            )
            
            test_result = await connector.test_connection(user_id, org_id, ip_address)
            
            # Log test result
            await self.audit_logger.log_event(
                event_type=SecurityEventType.API_ACCESS,
                action="connection_test_completed",
                user_id=user_id,
                org_id=org_id,
                ip_address=ip_address,
                details={
                    "provider": provider,
                    "test_result": test_result,
                    "success": test_result["success"]
                }
            )
            
            return test_result
            
        except Exception as e:
            await self.audit_logger.log_event(
                event_type=SecurityEventType.SECURITY_VIOLATION,
                action="connection_test_exception",
                user_id=user_id,
                org_id=org_id,
                ip_address=ip_address,
                details={
                    "provider": provider,
                    "error": str(e)
                }
            )
            logger.error(f"Connection test failed for {provider}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def sync_provider_data(
        self,
        provider: str,
        user_id: str,
        org_id: str,
        ip_address: str,
        sync_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Sync data from a provider"""
        try:
            connector = await self.get_connector(provider, user_id, org_id, ip_address)
            
            # Log sync attempt
            await self.audit_logger.log_event(
                event_type=SecurityEventType.RESOURCE_ACCESSED,
                action="data_sync_started",
                user_id=user_id,
                org_id=org_id,
                ip_address=ip_address,
                details={
                    "provider": provider,
                    "sync_config": sync_config,
                    "sync_start": datetime.now(timezone.utc).isoformat()
                }
            )
            
            sync_result = await connector.sync_data(user_id, org_id, ip_address, sync_config)
            
            # Log sync result
            await self.audit_logger.log_event(
                event_type=SecurityEventType.RESOURCE_ACCESSED,
                action="data_sync_completed",
                user_id=user_id,
                org_id=org_id,
                ip_address=ip_address,
                details={
                    "provider": provider,
                    "sync_result": {
                        "success": sync_result["success"],
                        "sync_duration": sync_result.get("sync_duration"),
                        "data_summary": {
                            k: len(v) if isinstance(v, list) else v
                            for k, v in sync_result.get("data", {}).items()
                            if k != "errors"
                        }
                    },
                    "success": sync_result["success"]
                }
            )
            
            return sync_result
            
        except Exception as e:
            await self.audit_logger.log_event(
                event_type=SecurityEventType.SECURITY_VIOLATION,
                action="data_sync_exception",
                user_id=user_id,
                org_id=org_id,
                ip_address=ip_address,
                details={
                    "provider": provider,
                    "error": str(e)
                }
            )
            logger.error(f"Data sync failed for {provider}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_supported_providers(self) -> List[str]:
        """Get list of supported providers"""
        return list(self.connector_registry.keys())
    
    def get_provider_info(self, provider: str) -> Dict[str, Any]:
        """Get information about a specific provider"""
        try:
            provider_key = provider.lower()
            
            if provider_key not in self.connector_registry:
                return {
                    "supported": False,
                    "error": f"Provider {provider} not supported"
                }
            
            connector_class = self.connector_registry[provider_key]
            
            # Create temporary instance to get info
            temp_connector = connector_class()
            
            return {
                "supported": True,
                "provider": provider,
                "connector_class": connector_class.__name__,
                "required_credentials": temp_connector.get_required_credentials(),
                "optional_credentials": temp_connector.get_optional_credentials(),
                "supported_endpoints": temp_connector.get_supported_endpoints(),
                "rate_limits": temp_connector.get_rate_limit_info()
            }
            
        except Exception as e:
            logger.error(f"Failed to get provider info for {provider}: {e}")
            return {
                "supported": False,
                "error": str(e)
            }
    
    async def get_connector_statistics(
        self,
        user_id: str,
        org_id: str
    ) -> Dict[str, Any]:
        """Get statistics for all connectors"""
        try:
            statistics = {
                "total_providers": len(self.connector_registry),
                "active_instances": len(self.connector_instances),
                "supported_providers": self.get_supported_providers(),
                "provider_details": {},
                "user_instances": []
            }
            
            # Get details for each provider
            for provider in self.connector_registry.keys():
                statistics["provider_details"][provider] = self.get_provider_info(provider)
            
            # Get user-specific instances
            user_prefix = f"_{user_id}_{org_id}"
            for instance_key, connector in self.connector_instances.items():
                if instance_key.endswith(user_prefix):
                    provider = instance_key.replace(user_prefix, "")
                    usage_stats = await connector.get_usage_statistics(user_id, org_id)
                    statistics["user_instances"].append({
                        "provider": provider,
                        "instance_key": instance_key,
                        "usage_stats": usage_stats
                    })
            
            return statistics
            
        except Exception as e:
            logger.error(f"Failed to get connector statistics: {e}")
            return {
                "error": str(e),
                "total_providers": 0,
                "active_instances": 0,
                "supported_providers": [],
                "provider_details": {},
                "user_instances": []
            }
    
    async def clear_connector_cache(self, provider: Optional[str] = None):
        """Clear connector cache"""
        try:
            if provider:
                # Clear cache for specific provider
                provider_key = provider.lower()
                keys_to_remove = [
                    key for key in self.connector_instances.keys()
                    if key.startswith(f"{provider_key}_")
                ]
                
                for key in keys_to_remove:
                    connector = self.connector_instances[key]
                    await connector.clear_cache()
                    del self.connector_instances[key]
                
                logger.info(f"Cleared cache for {provider} connector ({len(keys_to_remove)} instances)")
            else:
                # Clear all connector caches
                for connector in self.connector_instances.values():
                    await connector.clear_cache()
                
                instance_count = len(self.connector_instances)
                self.connector_instances.clear()
                
                logger.info(f"Cleared all connector caches ({instance_count} instances)")
                
        except Exception as e:
            logger.error(f"Failed to clear connector cache: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on connector factory"""
        try:
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_providers": len(self.connector_registry),
                "active_instances": len(self.connector_instances),
                "supported_providers": self.get_supported_providers(),
                "provider_health": {}
            }
            
            # Check each provider
            for provider in self.connector_registry.keys():
                try:
                    provider_info = self.get_provider_info(provider)
                    health_status["provider_health"][provider] = {
                        "status": "available",
                        "connector_class": provider_info.get("connector_class"),
                        "required_credentials": len(provider_info.get("required_credentials", [])),
                        "supported_endpoints": len(provider_info.get("supported_endpoints", []))
                    }
                except Exception as e:
                    health_status["provider_health"][provider] = {
                        "status": "error",
                        "error": str(e)
                    }
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

# Global factory instance
_secure_connector_factory: Optional[SecureConnectorFactory] = None

def get_secure_connector_factory() -> SecureConnectorFactory:
    """Get the global secure connector factory instance"""
    global _secure_connector_factory
    
    if _secure_connector_factory is None:
        _secure_connector_factory = SecureConnectorFactory()
    
    return _secure_connector_factory