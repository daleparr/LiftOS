"""
JWT Authentication Module
Handles JWT token validation and user context extraction
"""

import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, status, Depends, Header
import os
import logging

from shared.utils.logging import setup_logging

logger = setup_logging("jwt_auth")

# JWT Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

class JWTAuth:
    """JWT Authentication handler"""
    
    def __init__(self):
        self.secret_key = JWT_SECRET_KEY
        self.algorithm = JWT_ALGORITHM
        self.expiration_hours = JWT_EXPIRATION_HOURS
    
    def create_access_token(self, user_id: str, org_id: str, 
                          roles: list = None, extra_claims: Dict[str, Any] = None) -> str:
        """Create a JWT access token"""
        try:
            # Token payload
            payload = {
                "user_id": user_id,
                "org_id": org_id,
                "roles": roles or [],
                "iat": datetime.utcnow(),
                "exp": datetime.utcnow() + timedelta(hours=self.expiration_hours),
                "type": "access_token"
            }
            
            # Add extra claims if provided
            if extra_claims:
                payload.update(extra_claims)
            
            # Create token
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            return token
            
        except Exception as e:
            logger.error(f"Failed to create access token: {str(e)}")
            raise
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode a JWT token"""
        try:
            # Decode token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check token type
            if payload.get("type") != "access_token":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type"
                )
            
            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired"
                )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        except Exception as e:
            logger.error(f"Token verification failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token verification failed"
            )
    
    def refresh_token(self, token: str) -> str:
        """Refresh an access token"""
        try:
            # Verify current token (allow expired for refresh)
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm],
                options={"verify_exp": False}  # Allow expired tokens for refresh
            )
            
            # Create new token with same claims
            new_token = self.create_access_token(
                user_id=payload["user_id"],
                org_id=payload["org_id"],
                roles=payload.get("roles", []),
                extra_claims={k: v for k, v in payload.items() 
                            if k not in ["user_id", "org_id", "roles", "iat", "exp", "type"]}
            )
            
            return new_token
            
        except Exception as e:
            logger.error(f"Token refresh failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token refresh failed"
            )

# Global JWT auth instance
jwt_auth = JWTAuth()

def get_jwt_auth() -> JWTAuth:
    """Get the global JWT auth instance"""
    return jwt_auth

# FastAPI Dependencies
async def get_current_user(authorization: Optional[str] = Header(None)) -> str:
    """FastAPI dependency to get current user ID from JWT token"""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required"
        )
    
    try:
        # Extract token from "Bearer <token>" format
        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header format"
            )
        
        token = authorization.split(" ")[1]
        payload = jwt_auth.verify_token(token)
        
        return payload["user_id"]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get current user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )

async def get_current_org(authorization: Optional[str] = Header(None)) -> str:
    """FastAPI dependency to get current organization ID from JWT token"""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required"
        )
    
    try:
        # Extract token from "Bearer <token>" format
        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header format"
            )
        
        token = authorization.split(" ")[1]
        payload = jwt_auth.verify_token(token)
        
        return payload["org_id"]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get current org: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )

async def get_current_user_context(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """FastAPI dependency to get full user context from JWT token"""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required"
        )
    
    try:
        # Extract token from "Bearer <token>" format
        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header format"
            )
        
        token = authorization.split(" ")[1]
        payload = jwt_auth.verify_token(token)
        
        return {
            "user_id": payload["user_id"],
            "org_id": payload["org_id"],
            "roles": payload.get("roles", []),
            "token_payload": payload
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user context: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )

async def require_role(required_role: str):
    """FastAPI dependency factory to require specific role"""
    async def role_checker(user_context: Dict[str, Any] = Depends(get_current_user_context)) -> Dict[str, Any]:
        user_roles = user_context.get("roles", [])
        
        if required_role not in user_roles and "admin" not in user_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' required"
            )
        
        return user_context
    
    return role_checker

async def require_any_role(required_roles: list):
    """FastAPI dependency factory to require any of the specified roles"""
    async def role_checker(user_context: Dict[str, Any] = Depends(get_current_user_context)) -> Dict[str, Any]:
        user_roles = user_context.get("roles", [])
        
        if not any(role in user_roles for role in required_roles) and "admin" not in user_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"One of roles {required_roles} required"
            )
        
        return user_context
    
    return role_checker

async def require_permissions(user_id: str, permissions: list) -> None:
    """Validate that a user has the required permissions"""
    # For now, implement a simple permission check
    # In a real system, this would check against a permission database
    # For development, we'll allow all permissions for authenticated users
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User authentication required"
        )
    
    # TODO: Implement actual permission checking against user roles/permissions
    # For now, just validate that user_id exists
    logger.debug(f"Permission check passed for user {user_id} with permissions {permissions}")
    return

# Alternative header-based authentication for backward compatibility
async def get_user_from_headers(
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """Get user context from headers (for backward compatibility)"""
    if not x_user_id or not x_org_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User context headers required"
        )
    
    return {
        "user_id": x_user_id,
        "org_id": x_org_id,
        "roles": x_user_roles.split(",") if x_user_roles else [],
        "source": "headers"
    }

# Utility functions
def create_user_token(user_id: str, org_id: str, roles: list = None) -> str:
    """Utility function to create a user token"""
    return jwt_auth.create_access_token(user_id, org_id, roles)

def verify_user_token(token: str) -> Dict[str, Any]:
    """Utility function to verify a user token"""
    return jwt_auth.verify_token(token)

def extract_user_from_token(token: str) -> tuple:
    """Extract user_id and org_id from token"""
    payload = jwt_auth.verify_token(token)
    return payload["user_id"], payload["org_id"]

# Mock authentication for development/testing
class MockAuth:
    """Mock authentication for development and testing"""
    
    def __init__(self):
        self.mock_users = {
            "test_user_1": {
                "user_id": "test_user_1",
                "org_id": "test_org_1",
                "roles": ["user", "platform_admin"]
            },
            "admin_user": {
                "user_id": "admin_user",
                "org_id": "test_org_1",
                "roles": ["admin", "platform_admin"]
            }
        }
    
    def get_mock_user_context(self, user_id: str = "test_user_1") -> Dict[str, Any]:
        """Get mock user context for testing"""
        return self.mock_users.get(user_id, self.mock_users["test_user_1"])
    
    def create_mock_token(self, user_id: str = "test_user_1") -> str:
        """Create a mock token for testing"""
        user_data = self.get_mock_user_context(user_id)
        return jwt_auth.create_access_token(
            user_data["user_id"],
            user_data["org_id"],
            user_data["roles"]
        )

# Global mock auth instance
mock_auth = MockAuth()

def get_mock_auth() -> MockAuth:
    """Get the global mock auth instance"""
    return mock_auth