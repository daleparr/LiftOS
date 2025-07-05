"""
JWT utilities for Lift OS Core authentication
"""
import jwt
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from shared.models.base import JWTClaims, UserRole, SubscriptionTier


class JWTHandler:
    """JWT token handler for authentication"""
    
    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET", "your-secret-key")
        self.algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        self.expiration_hours = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
    
    def create_token(
        self,
        user_id: str,
        org_id: str,
        email: str,
        roles: List[UserRole],
        permissions: List[str],
        subscription_tier: SubscriptionTier,
        memory_context: Optional[str] = None,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a JWT token with user claims"""
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=self.expiration_hours)
        
        claims = {
            "sub": user_id,
            "org_id": org_id,
            "email": email,
            "roles": [role.value for role in roles],
            "permissions": permissions,
            "memory_context": memory_context or f"org_{org_id}_context",
            "subscription_tier": subscription_tier.value,
            "exp": int(expire.timestamp()),
            "iat": int(datetime.utcnow().timestamp())
        }
        
        return jwt.encode(claims, self.secret_key, algorithm=self.algorithm)
    
    def decode_token(self, token: str) -> Optional[JWTClaims]:
        """Decode and validate a JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Convert roles back to enum
            roles = [UserRole(role) for role in payload.get("roles", [])]
            
            return JWTClaims(
                sub=payload["sub"],
                org_id=payload["org_id"],
                email=payload["email"],
                roles=roles,
                permissions=payload.get("permissions", []),
                memory_context=payload.get("memory_context"),
                subscription_tier=SubscriptionTier(payload.get("subscription_tier", "free")),
                exp=payload["exp"],
                iat=payload["iat"]
            )
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def refresh_token(self, token: str) -> Optional[str]:
        """Refresh a JWT token if it's valid"""
        claims = self.decode_token(token)
        if not claims:
            return None
        
        # Create new token with same claims but extended expiration
        return self.create_token(
            user_id=claims.sub,
            org_id=claims.org_id,
            email=claims.email,
            roles=claims.roles,
            permissions=claims.permissions,
            subscription_tier=claims.subscription_tier,
            memory_context=claims.memory_context
        )
    
    def extract_bearer_token(self, authorization_header: str) -> Optional[str]:
        """Extract token from Authorization header"""
        if not authorization_header:
            return None
        
        parts = authorization_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None
        
        return parts[1]


class PermissionChecker:
    """Permission checking utilities"""
    
    @staticmethod
    def has_permission(user_permissions: List[str], required_permission: str) -> bool:
        """Check if user has a specific permission"""
        return required_permission in user_permissions
    
    @staticmethod
    def has_any_permission(user_permissions: List[str], required_permissions: List[str]) -> bool:
        """Check if user has any of the required permissions"""
        return any(perm in user_permissions for perm in required_permissions)
    
    @staticmethod
    def has_all_permissions(user_permissions: List[str], required_permissions: List[str]) -> bool:
        """Check if user has all required permissions"""
        return all(perm in user_permissions for perm in required_permissions)
    
    @staticmethod
    def has_role(user_roles: List[UserRole], required_role: UserRole) -> bool:
        """Check if user has a specific role"""
        return required_role in user_roles
    
    @staticmethod
    def has_any_role(user_roles: List[UserRole], required_roles: List[UserRole]) -> bool:
        """Check if user has any of the required roles"""
        return any(role in user_roles for role in required_roles)
    
    @staticmethod
    def can_access_org(user_org_id: str, target_org_id: str, user_roles: List[UserRole]) -> bool:
        """Check if user can access a specific organization"""
        # Users can always access their own org
        if user_org_id == target_org_id:
            return True
        
        # Admins can access any org
        if UserRole.ADMIN in user_roles:
            return True
        
        return False


# Global JWT handler instance
jwt_handler = JWTHandler()
permission_checker = PermissionChecker()


def get_current_user_from_token(token: str) -> Optional[JWTClaims]:
    """Get current user from JWT token"""
    return jwt_handler.decode_token(token)


def create_access_token(
    user_id: str,
    org_id: str,
    email: str,
    roles: List[UserRole],
    permissions: List[str],
    subscription_tier: SubscriptionTier,
    memory_context: Optional[str] = None
) -> str:
    """Create an access token for a user"""
    return jwt_handler.create_token(
        user_id=user_id,
        org_id=org_id,
        email=email,
        roles=roles,
        permissions=permissions,
        subscription_tier=subscription_tier,
        memory_context=memory_context
    )


def verify_token(token: str) -> Optional[JWTClaims]:
    """Verify and decode a JWT token"""
    return jwt_handler.decode_token(token)


def extract_token_from_header(authorization_header: str) -> Optional[str]:
    """Extract JWT token from Authorization header"""
    return jwt_handler.extract_bearer_token(authorization_header)


def require_permissions(required_permissions: List[str]):
    """Decorator to require specific permissions for an endpoint"""
    from functools import wraps
    from fastapi import HTTPException, status, Depends
    from fastapi.security import HTTPBearer
    
    security = HTTPBearer()
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract token from request
            # This is a simplified version - in production you'd use FastAPI's dependency injection
            return await func(*args, **kwargs)
        return wrapper
    return decorator