"""
KSE Memory SDK Security Service
"""

from typing import Dict, Any, List, Optional, Set
import asyncio
import hashlib
import hmac
import time
import logging
import secrets
import jwt
from datetime import datetime, timedelta
from ..core.interfaces import SecurityInterface

logger = logging.getLogger(__name__)


class SecurityService(SecurityInterface):
    """
    Security service for KSE Memory operations including authentication,
    authorization, encryption, and audit logging.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize security service.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.access_tokens: Dict[str, Dict[str, Any]] = {}
        self.permissions: Dict[str, Set[str]] = {}
        self.audit_log: List[Dict[str, Any]] = []
        
        # Security configuration
        self.secret_key = config.get('secret_key', secrets.token_urlsafe(32))
        self.token_expiry = config.get('token_expiry_hours', 24)
        self.max_audit_entries = config.get('max_audit_entries', 10000)
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        
        # Default permissions
        self.default_permissions = {
            'admin': {'read', 'write', 'delete', 'admin'},
            'user': {'read', 'write'},
            'readonly': {'read'}
        }
        
    async def initialize(self) -> bool:
        """Initialize security service."""
        try:
            # Initialize default API keys if provided
            default_keys = self.config.get('default_api_keys', {})
            for key_id, key_data in default_keys.items():
                await self.create_api_key(
                    key_id=key_id,
                    permissions=key_data.get('permissions', ['read']),
                    metadata=key_data.get('metadata', {})
                )
            
            logger.info("Security service initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize security service: {e}")
            return False
    
    async def authenticate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate an API key.
        
        Args:
            api_key: API key to authenticate
            
        Returns:
            Authentication result with user info
        """
        try:
            # Hash the API key for lookup
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            if key_hash in self.api_keys:
                key_data = self.api_keys[key_hash]
                
                # Check if key is active
                if not key_data.get('active', True):
                    await self._log_audit('auth_failed', {'reason': 'inactive_key'})
                    return None
                
                # Check expiry
                if key_data.get('expires_at') and time.time() > key_data['expires_at']:
                    await self._log_audit('auth_failed', {'reason': 'expired_key'})
                    return None
                
                # Update last used
                key_data['last_used'] = time.time()
                key_data['usage_count'] = key_data.get('usage_count', 0) + 1
                
                await self._log_audit('auth_success', {
                    'key_id': key_data['key_id'],
                    'permissions': key_data['permissions']
                })
                
                return {
                    'authenticated': True,
                    'key_id': key_data['key_id'],
                    'permissions': key_data['permissions'],
                    'metadata': key_data.get('metadata', {})
                }
            
            await self._log_audit('auth_failed', {'reason': 'invalid_key'})
            return None
            
        except Exception as e:
            logger.error(f"Failed to authenticate API key: {e}")
            await self._log_audit('auth_error', {'error': str(e)})
            return None
    
    async def authenticate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate a JWT token.
        
        Args:
            token: JWT token to authenticate
            
        Returns:
            Authentication result with user info
        """
        try:
            # Decode JWT token
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
            # Check if token is in our active tokens
            token_id = payload.get('token_id')
            if token_id and token_id in self.access_tokens:
                token_data = self.access_tokens[token_id]
                
                # Check if token is active
                if not token_data.get('active', True):
                    await self._log_audit('token_auth_failed', {'reason': 'inactive_token'})
                    return None
                
                # Update last used
                token_data['last_used'] = time.time()
                
                await self._log_audit('token_auth_success', {
                    'user_id': payload.get('user_id'),
                    'permissions': payload.get('permissions', [])
                })
                
                return {
                    'authenticated': True,
                    'user_id': payload.get('user_id'),
                    'permissions': payload.get('permissions', []),
                    'metadata': payload.get('metadata', {})
                }
            
            await self._log_audit('token_auth_failed', {'reason': 'invalid_token'})
            return None
            
        except jwt.ExpiredSignatureError:
            await self._log_audit('token_auth_failed', {'reason': 'expired_token'})
            return None
        except jwt.InvalidTokenError as e:
            await self._log_audit('token_auth_failed', {'reason': 'invalid_token', 'error': str(e)})
            return None
        except Exception as e:
            logger.error(f"Failed to authenticate token: {e}")
            await self._log_audit('token_auth_error', {'error': str(e)})
            return None
    
    async def authorize_operation(self, user_info: Dict[str, Any], 
                                operation: str, resource: Optional[str] = None) -> bool:
        """
        Authorize an operation for a user.
        
        Args:
            user_info: User information from authentication
            operation: Operation to authorize (read, write, delete, admin)
            resource: Optional resource identifier
            
        Returns:
            True if authorized
        """
        try:
            permissions = user_info.get('permissions', [])
            
            # Check if user has required permission
            if operation in permissions or 'admin' in permissions:
                await self._log_audit('auth_success', {
                    'user': user_info.get('user_id') or user_info.get('key_id'),
                    'operation': operation,
                    'resource': resource
                })
                return True
            
            await self._log_audit('auth_denied', {
                'user': user_info.get('user_id') or user_info.get('key_id'),
                'operation': operation,
                'resource': resource,
                'permissions': permissions
            })
            return False
            
        except Exception as e:
            logger.error(f"Failed to authorize operation: {e}")
            await self._log_audit('auth_error', {'error': str(e)})
            return False
    
    async def create_api_key(self, key_id: str, permissions: List[str], 
                           metadata: Optional[Dict[str, Any]] = None,
                           expires_in_days: Optional[int] = None) -> Optional[str]:
        """
        Create a new API key.
        
        Args:
            key_id: Unique identifier for the key
            permissions: List of permissions
            metadata: Additional metadata
            expires_in_days: Expiry in days
            
        Returns:
            Generated API key
        """
        try:
            # Generate API key
            api_key = secrets.token_urlsafe(32)
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            # Calculate expiry
            expires_at = None
            if expires_in_days:
                expires_at = time.time() + (expires_in_days * 24 * 3600)
            
            # Store key data
            self.api_keys[key_hash] = {
                'key_id': key_id,
                'permissions': permissions,
                'metadata': metadata or {},
                'created_at': time.time(),
                'expires_at': expires_at,
                'active': True,
                'usage_count': 0,
                'last_used': None
            }
            
            await self._log_audit('api_key_created', {
                'key_id': key_id,
                'permissions': permissions
            })
            
            return api_key
            
        except Exception as e:
            logger.error(f"Failed to create API key: {e}")
            return None
    
    async def create_access_token(self, user_id: str, permissions: List[str],
                                metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Create a new access token.
        
        Args:
            user_id: User identifier
            permissions: List of permissions
            metadata: Additional metadata
            
        Returns:
            Generated JWT token
        """
        try:
            token_id = secrets.token_urlsafe(16)
            expires_at = time.time() + (self.token_expiry * 3600)
            
            # Create JWT payload
            payload = {
                'token_id': token_id,
                'user_id': user_id,
                'permissions': permissions,
                'metadata': metadata or {},
                'iat': time.time(),
                'exp': expires_at
            }
            
            # Generate JWT token
            token = jwt.encode(payload, self.secret_key, algorithm='HS256')
            
            # Store token data
            self.access_tokens[token_id] = {
                'user_id': user_id,
                'permissions': permissions,
                'metadata': metadata or {},
                'created_at': time.time(),
                'expires_at': expires_at,
                'active': True,
                'last_used': None
            }
            
            await self._log_audit('access_token_created', {
                'user_id': user_id,
                'permissions': permissions
            })
            
            return token
            
        except Exception as e:
            logger.error(f"Failed to create access token: {e}")
            return None
    
    async def revoke_api_key(self, key_id: str) -> bool:
        """
        Revoke an API key.
        
        Args:
            key_id: Key identifier to revoke
            
        Returns:
            True if successful
        """
        try:
            # Find and deactivate key
            for key_hash, key_data in self.api_keys.items():
                if key_data['key_id'] == key_id:
                    key_data['active'] = False
                    key_data['revoked_at'] = time.time()
                    
                    await self._log_audit('api_key_revoked', {'key_id': key_id})
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to revoke API key: {e}")
            return False
    
    async def revoke_access_token(self, token_id: str) -> bool:
        """
        Revoke an access token.
        
        Args:
            token_id: Token identifier to revoke
            
        Returns:
            True if successful
        """
        try:
            if token_id in self.access_tokens:
                self.access_tokens[token_id]['active'] = False
                self.access_tokens[token_id]['revoked_at'] = time.time()
                
                await self._log_audit('access_token_revoked', {'token_id': token_id})
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to revoke access token: {e}")
            return False
    
    async def check_rate_limit(self, identifier: str, operation: str,
                             limit: int, window_seconds: int) -> bool:
        """
        Check rate limit for an identifier.
        
        Args:
            identifier: User/key identifier
            operation: Operation being performed
            limit: Maximum operations allowed
            window_seconds: Time window in seconds
            
        Returns:
            True if within rate limit
        """
        try:
            current_time = time.time()
            key = f"{identifier}:{operation}"
            
            if key not in self.rate_limits:
                self.rate_limits[key] = []
            
            # Clean old entries
            cutoff_time = current_time - window_seconds
            self.rate_limits[key] = [
                timestamp for timestamp in self.rate_limits[key]
                if timestamp > cutoff_time
            ]
            
            # Check if within limit
            if len(self.rate_limits[key]) >= limit:
                await self._log_audit('rate_limit_exceeded', {
                    'identifier': identifier,
                    'operation': operation,
                    'current_count': len(self.rate_limits[key]),
                    'limit': limit
                })
                return False
            
            # Add current request
            self.rate_limits[key].append(current_time)
            return True
            
        except Exception as e:
            logger.error(f"Failed to check rate limit: {e}")
            return True  # Allow on error
    
    async def get_audit_log(self, limit: int = 100, 
                          filter_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get audit log entries.
        
        Args:
            limit: Maximum entries to return
            filter_type: Filter by event type
            
        Returns:
            List of audit log entries
        """
        try:
            if filter_type:
                filtered_log = [
                    entry for entry in self.audit_log
                    if entry.get('event_type') == filter_type
                ]
                return filtered_log[-limit:]
            else:
                return self.audit_log[-limit:]
                
        except Exception as e:
            logger.error(f"Failed to get audit log: {e}")
            return []
    
    async def _log_audit(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an audit event."""
        try:
            audit_entry = {
                'timestamp': time.time(),
                'event_type': event_type,
                'data': data
            }
            
            self.audit_log.append(audit_entry)
            
            # Trim audit log if too large
            if len(self.audit_log) > self.max_audit_entries:
                self.audit_log = self.audit_log[-self.max_audit_entries:]
                
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")