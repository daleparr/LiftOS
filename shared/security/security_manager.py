"""
Security utilities for Lift OS Core services
Provides JWT verification, rate limiting, and security middleware
"""

import jwt
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import logging

logger = logging.getLogger(__name__)

class SecurityManager:
    """Centralized security management for all services"""
    
    def __init__(self, jwt_secret: str, jwt_algorithm: str = "HS256"):
        self.jwt_secret = jwt_secret
        self.jwt_algorithm = jwt_algorithm
        self.rate_limits: Dict[str, List[float]] = {}
        self.security = HTTPBearer()
        self.blocked_ips: set = set()
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create a new JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=24)
        
        to_encode.update({"exp": expire})
        
        encoded_jwt = jwt.encode(
            to_encode, 
            self.jwt_secret, 
            algorithm=self.jwt_algorithm
        )
        return encoded_jwt
    
    async def verify_jwt(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        """Verify JWT token and return payload"""
        try:
            payload = jwt.decode(
                credentials.credentials,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm]
            )
            
            # Check if token is expired
            exp = payload.get("exp")
            if exp and datetime.utcnow().timestamp() > exp:
                raise HTTPException(
                    status_code=401, 
                    detail="Token expired"
                )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            raise HTTPException(
                status_code=401, 
                detail="Token expired"
            )
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            raise HTTPException(
                status_code=401, 
                detail="Invalid token"
            )
        except Exception as e:
            logger.error(f"JWT verification error: {e}")
            raise HTTPException(
                status_code=401, 
                detail="Token verification failed"
            )
    
    async def rate_limit(
        self, 
        request: Request, 
        max_requests: int = 100, 
        window_seconds: int = 60,
        block_duration: int = 300  # 5 minutes
    ):
        """Rate limiting middleware with IP blocking"""
        client_ip = self._get_client_ip(request)
        current_time = time.time()
        
        # Check if IP is blocked
        if client_ip in self.blocked_ips:
            logger.warning(f"Blocked IP {client_ip} attempted request")
            raise HTTPException(
                status_code=429, 
                detail="IP temporarily blocked due to rate limit violations"
            )
        
        # Initialize rate limit tracking for new IPs
        if client_ip not in self.rate_limits:
            self.rate_limits[client_ip] = []
        
        # Clean old requests outside the window
        self.rate_limits[client_ip] = [
            req_time for req_time in self.rate_limits[client_ip]
            if current_time - req_time < window_seconds
        ]
        
        # Check if rate limit exceeded
        if len(self.rate_limits[client_ip]) >= max_requests:
            # Block IP for repeated violations
            self.blocked_ips.add(client_ip)
            
            # Schedule unblock (in production, use Redis with TTL)
            # For now, we'll implement a simple cleanup
            logger.warning(f"Rate limit exceeded for IP {client_ip}, blocking for {block_duration}s")
            
            raise HTTPException(
                status_code=429, 
                detail=f"Rate limit exceeded. IP blocked for {block_duration} seconds."
            )
        
        # Record this request
        self.rate_limits[client_ip].append(current_time)
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request, considering proxies"""
        # Check for forwarded headers (common in production behind load balancers)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        return request.client.host if request.client else "unknown"
    
    async def validate_api_key(self, api_key: str, valid_keys: List[str]) -> bool:
        """Validate API key for service-to-service communication"""
        # Hash the provided key for comparison
        hashed_key = hashlib.sha256(api_key.encode()).hexdigest()
        
        # In production, store hashed keys in database
        valid_hashes = [hashlib.sha256(key.encode()).hexdigest() for key in valid_keys]
        
        return hashed_key in valid_hashes
    
    def create_service_token(self, service_name: str, permissions: List[str]) -> str:
        """Create a service-to-service authentication token"""
        payload = {
            "service": service_name,
            "permissions": permissions,
            "type": "service",
            "iat": datetime.utcnow().timestamp(),
            "exp": (datetime.utcnow() + timedelta(hours=1)).timestamp()
        }
        
        return self.create_access_token(payload)
    
    async def verify_service_token(self, token: str, required_permission: str = None) -> Dict[str, Any]:
        """Verify service-to-service token"""
        try:
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm]
            )
            
            # Verify it's a service token
            if payload.get("type") != "service":
                raise HTTPException(
                    status_code=401,
                    detail="Invalid service token"
                )
            
            # Check permissions if required
            if required_permission:
                permissions = payload.get("permissions", [])
                if required_permission not in permissions:
                    raise HTTPException(
                        status_code=403,
                        detail=f"Missing required permission: {required_permission}"
                    )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Service token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid service token")

class SecurityMiddleware:
    """Security middleware configuration"""
    
    @staticmethod
    def add_security_middleware(app, allowed_hosts: List[str] = None, force_https: bool = False):
        """Add security middleware to FastAPI app"""
        
        if force_https:
            app.add_middleware(HTTPSRedirectMiddleware)
        
        if allowed_hosts:
            app.add_middleware(
                TrustedHostMiddleware,
                allowed_hosts=allowed_hosts
            )
    
    @staticmethod
    def add_cors_middleware(app, allowed_origins: List[str] = None):
        """Add CORS middleware for web applications"""
        from fastapi.middleware.cors import CORSMiddleware
        
        if not allowed_origins:
            allowed_origins = ["http://localhost:3000", "http://localhost:8080"]
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )

# Global security manager instance
security_manager = None

def get_security_manager() -> SecurityManager:
    """Get the global security manager instance"""
    global security_manager
    if security_manager is None:
        # In production, get JWT secret from environment or secrets manager
        import os
        jwt_secret = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
        security_manager = SecurityManager(jwt_secret)
    return security_manager

# Dependency for FastAPI routes
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
) -> Dict[str, Any]:
    """FastAPI dependency to get current authenticated user"""
    security = get_security_manager()
    return await security.verify_jwt(credentials)

async def require_permission(permission: str):
    """FastAPI dependency to require specific permission"""
    def permission_checker(current_user: Dict = Depends(get_current_user)):
        user_permissions = current_user.get("permissions", [])
        if permission not in user_permissions:
            raise HTTPException(
                status_code=403,
                detail=f"Missing required permission: {permission}"
            )
        return current_user
    return permission_checker