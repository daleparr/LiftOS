"""
Auth Service Integration for Agentic Module

Provides integration with the LiftOS Auth Service for authentication
and authorization of agent operations.
"""

import logging
from typing import Dict, List, Optional, Any
import httpx
import json

logger = logging.getLogger(__name__)


class AuthService:
    """
    Integration with LiftOS Auth Service.
    
    This service handles authentication and authorization for
    agent operations within the LiftOS ecosystem.
    """
    
    def __init__(self, auth_service_url: str):
        """Initialize auth service client."""
        self.base_url = auth_service_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=30.0)
        
        logger.info(f"Auth Service client initialized for {self.base_url}")
    
    async def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate an authentication token."""
        try:
            headers = {"Authorization": f"Bearer {token}"}
            response = await self.client.get(f"{self.base_url}/validate", headers=headers)
            
            if response.status_code == 401:
                return None
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to validate token: {e}")
            return None
    
    async def check_permission(
        self,
        user_id: str,
        resource: str,
        action: str
    ) -> bool:
        """Check if a user has permission for a specific action on a resource."""
        try:
            data = {
                "user_id": user_id,
                "resource": resource,
                "action": action
            }
            
            response = await self.client.post(f"{self.base_url}/check-permission", json=data)
            
            if response.status_code == 403:
                return False
            
            response.raise_for_status()
            result = response.json()
            return result.get("allowed", False)
            
        except Exception as e:
            logger.error(f"Failed to check permission for {user_id}: {e}")
            return False
    
    async def get_user_permissions(self, user_id: str) -> List[str]:
        """Get all permissions for a user."""
        try:
            response = await self.client.get(f"{self.base_url}/users/{user_id}/permissions")
            response.raise_for_status()
            
            data = response.json()
            return data.get("permissions", [])
            
        except Exception as e:
            logger.error(f"Failed to get permissions for user {user_id}: {e}")
            return []
    
    async def create_api_key(
        self,
        user_id: str,
        name: str,
        permissions: List[str],
        expires_in_days: Optional[int] = None
    ) -> Optional[str]:
        """Create an API key for agent operations."""
        try:
            data = {
                "user_id": user_id,
                "name": name,
                "permissions": permissions
            }
            
            if expires_in_days:
                data["expires_in_days"] = expires_in_days
            
            response = await self.client.post(f"{self.base_url}/api-keys", json=data)
            response.raise_for_status()
            
            result = response.json()
            return result.get("api_key")
            
        except Exception as e:
            logger.error(f"Failed to create API key for user {user_id}: {e}")
            return None
    
    async def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        try:
            response = await self.client.delete(f"{self.base_url}/api-keys/{api_key}")
            response.raise_for_status()
            return True
            
        except Exception as e:
            logger.error(f"Failed to revoke API key: {e}")
            return False
    
    async def audit_log(
        self,
        user_id: str,
        action: str,
        resource: str,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Log an audit event."""
        try:
            data = {
                "user_id": user_id,
                "action": action,
                "resource": resource,
                "timestamp": "auto",  # Server will set timestamp
                "details": details or {}
            }
            
            response = await self.client.post(f"{self.base_url}/audit", json=data)
            response.raise_for_status()
            return True
            
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            return False
    
    async def close(self) -> None:
        """Close the HTTP client."""
        try:
            await self.client.aclose()
            logger.debug("Auth service client closed")
        except Exception as e:
            logger.error(f"Error closing auth service client: {e}")