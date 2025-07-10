"""
Base Secure Connector
Abstract base class for all secure API connectors with enterprise security
"""

import asyncio
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import sys
import os

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.security.api_key_vault import get_api_key_vault
from shared.security.audit_logger import SecurityAuditLogger, SecurityEventType
from shared.utils.logging import setup_logging

logger = setup_logging("base_secure_connector")

class BaseSecureConnector(ABC):
    """Abstract base class for secure API connectors"""
    
    def __init__(self, provider: str):
        self.provider = provider
        self.api_key_vault = get_api_key_vault()
        self.audit_logger = SecurityAuditLogger()
        
        # Rate limiting storage (in production, use Redis)
        self.rate_limit_storage = defaultdict(lambda: defaultdict(deque))
        
        # Connector configuration
        self.required_credentials: List[str] = []
        self.optional_credentials: List[str] = []
        self.rate_limits: Dict[str, Dict[str, int]] = {}
        
        # Security settings
        self.max_retries = 3
        self.retry_delay = 1.0
        self.timeout = 30.0
        
        # Credential cache
        self.credential_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 300  # 5 minutes
    
    async def initialize(self):
        """Initialize the connector"""
        try:
            if not self.api_key_vault:
                self.api_key_vault = get_api_key_vault()
                await self.api_key_vault.initialize()
            
            logger.info(f"Initialized secure {self.provider} connector")
        except Exception as e:
            logger.error(f"Failed to initialize {self.provider} connector: {e}")
            raise
    
    async def get_credentials(self, user_id: str, org_id: str) -> Dict[str, str]:
        """Get decrypted credentials for the provider"""
        try:
            # Check cache first
            cache_key = f"{self.provider}_{user_id}_{org_id}"
            if cache_key in self.credential_cache:
                cache_entry = self.credential_cache[cache_key]
                if datetime.now(timezone.utc) < cache_entry["expires_at"]:
                    return cache_entry["credentials"]
                else:
                    del self.credential_cache[cache_key]
            
            # Get credentials from vault
            from shared.database.database import get_async_session
            from shared.database.security_models import EncryptedAPIKey
            from sqlalchemy import select, and_
            
            async with get_async_session() as session:
                result = await session.execute(
                    select(EncryptedAPIKey)
                    .where(
                        and_(
                            EncryptedAPIKey.provider == self.provider,
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
            
            # Validate required credentials
            missing_credentials = [
                cred for cred in self.required_credentials
                if cred not in credentials
            ]
            
            if missing_credentials:
                raise ValueError(f"Missing required credentials: {missing_credentials}")
            
            # Cache credentials
            self.credential_cache[cache_key] = {
                "credentials": credentials,
                "expires_at": datetime.now(timezone.utc) + timedelta(seconds=self.cache_ttl)
            }
            
            return credentials
            
        except Exception as e:
            logger.error(f"Failed to get {self.provider} credentials: {e}")
            raise
    
    async def validate_credentials(self, credentials: Dict[str, str]) -> Dict[str, Any]:
        """Validate connector credentials"""
        try:
            validation_result = {
                "valid": True,
                "missing_credentials": [],
                "unexpected_credentials": [],
                "warnings": []
            }
            
            # Check for missing required credentials
            for required_cred in self.required_credentials:
                if required_cred not in credentials:
                    validation_result["missing_credentials"].append(required_cred)
                    validation_result["valid"] = False
            
            # Check for unexpected credentials
            expected_credentials = self.required_credentials + self.optional_credentials
            for cred_name in credentials.keys():
                if cred_name not in expected_credentials:
                    validation_result["unexpected_credentials"].append(cred_name)
                    validation_result["warnings"].append(
                        f"Unexpected credential '{cred_name}' for provider '{self.provider}'"
                    )
            
            # Check for empty credentials
            for cred_name, cred_value in credentials.items():
                if not cred_value or not cred_value.strip():
                    validation_result["warnings"].append(
                        f"Empty credential value for '{cred_name}'"
                    )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Credential validation failed for {self.provider}: {e}")
            return {
                "valid": False,
                "error": str(e),
                "missing_credentials": [],
                "unexpected_credentials": [],
                "warnings": []
            }
    
    async def _check_rate_limit(self, endpoint: str, ip_address: str) -> bool:
        """Check and enforce rate limiting"""
        try:
            if endpoint not in self.rate_limits:
                endpoint = "default"
            
            if endpoint not in self.rate_limits:
                return True  # No rate limit configured
            
            rate_config = self.rate_limits[endpoint]
            max_requests = rate_config["requests"]
            window_seconds = rate_config["window"]
            
            # Get current time
            now = time.time()
            window_start = now - window_seconds
            
            # Clean old requests
            request_times = self.rate_limit_storage[ip_address][endpoint]
            while request_times and request_times[0] < window_start:
                request_times.popleft()
            
            # Check if rate limit exceeded
            if len(request_times) >= max_requests:
                # Log rate limit violation
                await self.audit_logger.log_event(
                    event_type=SecurityEventType.SECURITY_VIOLATION,
                    action="rate_limit_exceeded",
                    ip_address=ip_address,
                    details={
                        "provider": self.provider,
                        "endpoint": endpoint,
                        "requests_in_window": len(request_times),
                        "max_requests": max_requests,
                        "window_seconds": window_seconds
                    }
                )
                
                raise Exception(f"Rate limit exceeded for {endpoint}: {len(request_times)}/{max_requests} requests in {window_seconds}s")
            
            # Add current request
            request_times.append(now)
            return True
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            raise
    
    async def _make_secure_request(
        self,
        method: str,
        url: str,
        user_id: str,
        org_id: str,
        ip_address: str,
        endpoint_name: str = "default",
        **kwargs
    ) -> Dict[str, Any]:
        """Make a secure API request with comprehensive logging"""
        try:
            # Check rate limit
            await self._check_rate_limit(endpoint_name, ip_address)
            
            # Log API request start
            request_start = datetime.now(timezone.utc)
            await self.audit_logger.log_event(
                event_type=SecurityEventType.API_ACCESS,
                action=f"{self.provider}_api_request_started",
                user_id=user_id,
                org_id=org_id,
                ip_address=ip_address,
                details={
                    "provider": self.provider,
                    "method": method,
                    "url": url,
                    "endpoint": endpoint_name,
                    "request_start": request_start.isoformat()
                }
            )
            
            # Make request with retries
            last_exception = None
            for attempt in range(self.max_retries):
                try:
                    import aiohttp
                    
                    timeout = aiohttp.ClientTimeout(total=self.timeout)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.request(method, url, **kwargs) as response:
                            response_data = await response.json()
                            request_end = datetime.now(timezone.utc)
                            request_duration = (request_end - request_start).total_seconds()
                            
                            # Log successful request
                            await self.audit_logger.log_event(
                                event_type=SecurityEventType.API_ACCESS,
                                action=f"{self.provider}_api_request_completed",
                                user_id=user_id,
                                org_id=org_id,
                                ip_address=ip_address,
                                details={
                                    "provider": self.provider,
                                    "method": method,
                                    "url": url,
                                    "endpoint": endpoint_name,
                                    "status_code": response.status,
                                    "request_duration": request_duration,
                                    "attempt": attempt + 1,
                                    "success": response.status < 400
                                }
                            )
                            
                            return {
                                "success": response.status < 400,
                                "status_code": response.status,
                                "data": response_data,
                                "request_duration": request_duration
                            }
                            
                except Exception as e:
                    last_exception = e
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                        continue
                    else:
                        break
            
            # Log failed request
            request_end = datetime.now(timezone.utc)
            request_duration = (request_end - request_start).total_seconds()
            
            await self.audit_logger.log_event(
                event_type=SecurityEventType.SECURITY_VIOLATION,
                action=f"{self.provider}_api_request_failed",
                user_id=user_id,
                org_id=org_id,
                ip_address=ip_address,
                details={
                    "provider": self.provider,
                    "method": method,
                    "url": url,
                    "endpoint": endpoint_name,
                    "error": str(last_exception),
                    "request_duration": request_duration,
                    "attempts": self.max_retries,
                    "success": False
                }
            )
            
            return {
                "success": False,
                "error": str(last_exception),
                "request_duration": request_duration
            }
            
        except Exception as e:
            logger.error(f"Secure request failed for {self.provider}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def test_connection(self, user_id: str, org_id: str, ip_address: str) -> Dict[str, Any]:
        """Test API connection with credentials"""
        try:
            credentials = await self.get_credentials(user_id, org_id)
            
            # Log connection test
            await self.audit_logger.log_event(
                event_type=SecurityEventType.API_ACCESS,
                action=f"{self.provider}_connection_test",
                user_id=user_id,
                org_id=org_id,
                ip_address=ip_address,
                details={
                    "provider": self.provider,
                    "test_type": "connection"
                }
            )
            
            # Subclasses should implement specific connection test
            return await self._test_api_connection(credentials)
            
        except Exception as e:
            await self.audit_logger.log_event(
                event_type=SecurityEventType.SECURITY_VIOLATION,
                action=f"{self.provider}_connection_test_failed",
                user_id=user_id,
                org_id=org_id,
                ip_address=ip_address,
                details={
                    "provider": self.provider,
                    "error": str(e)
                }
            )
            logger.error(f"Connection test failed for {self.provider}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @abstractmethod
    async def _test_api_connection(self, credentials: Dict[str, str]) -> Dict[str, Any]:
        """Test API connection - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    async def sync_data(
        self,
        user_id: str,
        org_id: str,
        ip_address: str,
        sync_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Sync data from the API - must be implemented by subclasses"""
        pass
    
    def get_supported_endpoints(self) -> List[str]:
        """Get list of supported API endpoints"""
        return list(self.rate_limits.keys())
    
    def get_rate_limit_info(self) -> Dict[str, Dict[str, int]]:
        """Get rate limit configuration"""
        return self.rate_limits.copy()
    
    def get_required_credentials(self) -> List[str]:
        """Get list of required credentials"""
        return self.required_credentials.copy()
    
    def get_optional_credentials(self) -> List[str]:
        """Get list of optional credentials"""
        return self.optional_credentials.copy()
    
    async def clear_cache(self):
        """Clear credential cache"""
        self.credential_cache.clear()
        logger.info(f"Cleared credential cache for {self.provider}")
    
    async def get_usage_statistics(self, user_id: str, org_id: str) -> Dict[str, Any]:
        """Get API usage statistics"""
        try:
            # This would typically query the audit logs for usage stats
            # For now, return basic info
            return {
                "provider": self.provider,
                "user_id": user_id,
                "org_id": org_id,
                "cache_size": len(self.credential_cache),
                "supported_endpoints": self.get_supported_endpoints(),
                "rate_limits": self.get_rate_limit_info(),
                "required_credentials": self.get_required_credentials(),
                "optional_credentials": self.get_optional_credentials()
            }
        except Exception as e:
            logger.error(f"Failed to get usage statistics for {self.provider}: {e}")
            return {
                "provider": self.provider,
                "error": str(e)
            }