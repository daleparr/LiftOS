"""
Enhanced Security Middleware for LiftOS
Integrates with enhanced JWT, audit logging, and API key vault
"""

import asyncio
import time
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Callable, Tuple
from functools import wraps
from flask import Flask, request, jsonify, g
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import logging

from ..database.database import get_async_session
from ..database.security_models import SecurityConfiguration, EnhancedUserSession
from .enhanced_jwt import EnhancedJWTManager, DeviceFingerprint, get_enhanced_jwt_manager
from .audit_logger import SecurityAuditLogger, SecurityEventType
from .api_key_vault import APIKeyVault, get_api_key_vault

logger = logging.getLogger(__name__)

class SecurityContext:
    """Security context for the current request"""
    
    def __init__(self):
        self.user_id: Optional[str] = None
        self.org_id: Optional[str] = None
        self.email: Optional[str] = None
        self.roles: List[str] = []
        self.permissions: List[str] = []
        self.session_id: Optional[str] = None
        self.device_fingerprint: Optional[str] = None
        self.ip_address: Optional[str] = None
        self.user_agent: Optional[str] = None
        self.is_authenticated: bool = False
        self.risk_score: float = 0.0
        self.security_flags: List[str] = []

class RateLimiter:
    """Rate limiting implementation"""
    
    def __init__(self):
        self.requests = {}  # In production, use Redis
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
    
    def is_allowed(
        self,
        identifier: str,
        limit: int,
        window: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is within rate limits"""
        now = time.time()
        
        # Cleanup old entries periodically
        if now - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_entries(now)
            self.last_cleanup = now
        
        # Get or create request history for identifier
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        request_times = self.requests[identifier]
        
        # Remove requests outside the window
        cutoff_time = now - window
        request_times[:] = [t for t in request_times if t > cutoff_time]
        
        # Check if under limit
        if len(request_times) < limit:
            request_times.append(now)
            return True, {
                "requests_made": len(request_times),
                "limit": limit,
                "window": window,
                "reset_time": cutoff_time + window
            }
        
        return False, {
            "requests_made": len(request_times),
            "limit": limit,
            "window": window,
            "reset_time": cutoff_time + window
        }
    
    def _cleanup_old_entries(self, now: float):
        """Remove old rate limit entries"""
        cutoff = now - 3600  # Keep 1 hour of history
        for identifier in list(self.requests.keys()):
            self.requests[identifier] = [
                t for t in self.requests[identifier] if t > cutoff
            ]
            if not self.requests[identifier]:
                del self.requests[identifier]

class EnhancedSecurityMiddleware:
    """
    Enhanced security middleware with comprehensive protection
    """
    
    def __init__(self, app: Optional[Flask] = None):
        self.app = app
        self.jwt_manager = get_enhanced_jwt_manager()
        self.audit_logger = SecurityAuditLogger()
        self.api_key_vault = get_api_key_vault()
        self.rate_limiter = RateLimiter()
        
        # Security configuration
        self.default_rate_limits = {
            "auth": {"limit": 5, "window": 300},      # 5 auth attempts per 5 minutes
            "api": {"limit": 1000, "window": 3600},   # 1000 API calls per hour
            "sensitive": {"limit": 10, "window": 60}   # 10 sensitive ops per minute
        }
        
        if app:
            self.init_app(app)
    
    def init_app(self, app: Flask):
        """Initialize middleware with Flask app"""
        self.app = app
        
        # Register before_request handlers
        app.before_request(self._before_request)
        app.after_request(self._after_request)
        
        # Register error handlers
        app.errorhandler(401)(self._handle_unauthorized)
        app.errorhandler(403)(self._handle_forbidden)
        app.errorhandler(429)(self._handle_rate_limited)
    
    async def _before_request(self):
        """Process request before handling"""
        try:
            # Initialize security context
            g.security_context = SecurityContext()
            g.request_start_time = time.time()
            
            # Extract request information
            ip_address = self._get_client_ip()
            user_agent = request.headers.get('User-Agent', '')
            
            g.security_context.ip_address = ip_address
            g.security_context.user_agent = user_agent
            
            # Generate device fingerprint
            device_fingerprint = DeviceFingerprint.generate_fingerprint(
                user_agent=user_agent,
                ip_address=ip_address,
                additional_data={
                    "accept_language": request.headers.get('Accept-Language', ''),
                    "accept_encoding": request.headers.get('Accept-Encoding', '')
                }
            )
            g.security_context.device_fingerprint = device_fingerprint
            
            # Check rate limits
            await self._check_rate_limits(ip_address)
            
            # Authenticate request if token present
            auth_header = request.headers.get('Authorization')
            if auth_header and auth_header.startswith('Bearer '):
                token = auth_header[7:]
                await self._authenticate_request(token)
            
            # Calculate risk score
            g.security_context.risk_score = await self._calculate_risk_score()
            
            # Log request
            await self._log_request()
            
        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            # Don't block request for middleware errors, but log them
            await self._log_security_error(str(e))
    
    async def _after_request(self, response):
        """Process response after handling"""
        try:
            # Log response
            await self._log_response(response)
            
            # Add security headers
            self._add_security_headers(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Security middleware after_request error: {e}")
            return response
    
    async def _authenticate_request(self, token: str):
        """Authenticate request using JWT token"""
        try:
            async with get_async_session() as session:
                payload = await self.jwt_manager.verify_access_token(session, token)
                
                # Update security context
                g.security_context.user_id = payload["sub"]
                g.security_context.org_id = payload["org_id"]
                g.security_context.email = payload["email"]
                g.security_context.roles = payload["roles"]
                g.security_context.permissions = payload["permissions"]
                g.security_context.is_authenticated = True
                
                # Verify device fingerprint if available
                token_fingerprint = payload.get("device_fp")
                if token_fingerprint:
                    current_fingerprint = g.security_context.device_fingerprint[:16]
                    if token_fingerprint != current_fingerprint:
                        g.security_context.security_flags.append("device_fingerprint_mismatch")
                        g.security_context.risk_score += 0.3
                
        except Exception as e:
            logger.warning(f"Authentication failed: {e}")
            g.security_context.security_flags.append("authentication_failed")
            # Don't raise exception - let endpoint handle unauthenticated requests
    
    async def _check_rate_limits(self, ip_address: str):
        """Check rate limits for the request"""
        try:
            # Determine rate limit category
            endpoint = request.endpoint or "unknown"
            
            if any(auth_path in request.path for auth_path in ['/auth/', '/login', '/register']):
                limit_config = self.default_rate_limits["auth"]
            elif any(sensitive_path in request.path for sensitive_path in ['/admin/', '/api-keys/', '/security/']):
                limit_config = self.default_rate_limits["sensitive"]
            else:
                limit_config = self.default_rate_limits["api"]
            
            # Check rate limit
            identifier = f"{ip_address}:{endpoint}"
            allowed, limit_info = self.rate_limiter.is_allowed(
                identifier,
                limit_config["limit"],
                limit_config["window"]
            )
            
            if not allowed:
                # Log rate limit violation
                async with get_async_session() as session:
                    await self.audit_logger.log_security_event(
                        session=session,
                        event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
                        user_id=g.security_context.user_id,
                        org_id=g.security_context.org_id,
                        action="rate_limit_exceeded",
                        ip_address=ip_address,
                        user_agent=g.security_context.user_agent,
                        success=False,
                        details=limit_info
                    )
                
                # Return rate limit error
                response = jsonify({
                    "error": "Rate limit exceeded",
                    "details": limit_info
                })
                response.status_code = 429
                response.headers['Retry-After'] = str(int(limit_info["reset_time"] - time.time()))
                return response
                
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # Don't block request for rate limit errors
    
    async def _calculate_risk_score(self) -> float:
        """Calculate risk score for the request"""
        risk_score = 0.0
        
        try:
            # IP-based risk factors
            ip_address = g.security_context.ip_address
            
            # Check for suspicious IPs (implement IP reputation checking)
            if self._is_suspicious_ip(ip_address):
                risk_score += 0.4
            
            # Time-based risk factors
            current_hour = datetime.now().hour
            if current_hour < 6 or current_hour > 22:  # Outside business hours
                risk_score += 0.1
            
            # Device fingerprint risk
            if "device_fingerprint_mismatch" in g.security_context.security_flags:
                risk_score += 0.3
            
            # Authentication risk
            if "authentication_failed" in g.security_context.security_flags:
                risk_score += 0.2
            
            # Geographic risk (implement geolocation checking)
            # risk_score += self._calculate_geographic_risk()
            
            return min(risk_score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Risk calculation failed: {e}")
            return 0.5  # Default moderate risk
    
    def _is_suspicious_ip(self, ip_address: str) -> bool:
        """Check if IP address is suspicious"""
        # Implement IP reputation checking
        # For now, just check for common suspicious patterns
        suspicious_patterns = [
            "127.0.0.1",  # Localhost (might be suspicious in production)
            # Add more patterns as needed
        ]
        
        return any(pattern in ip_address for pattern in suspicious_patterns)
    
    async def _log_request(self):
        """Log the incoming request"""
        try:
            async with get_async_session() as session:
                await self.audit_logger.log_security_event(
                    session=session,
                    event_type=SecurityEventType.API_ACCESS,
                    user_id=g.security_context.user_id,
                    org_id=g.security_context.org_id,
                    action=f"{request.method} {request.path}",
                    ip_address=g.security_context.ip_address,
                    user_agent=g.security_context.user_agent,
                    success=True,
                    details={
                        "endpoint": request.endpoint,
                        "method": request.method,
                        "risk_score": g.security_context.risk_score,
                        "security_flags": g.security_context.security_flags,
                        "device_fingerprint": g.security_context.device_fingerprint[:16]
                    }
                )
        except Exception as e:
            logger.error(f"Failed to log request: {e}")
    
    async def _log_response(self, response):
        """Log the response"""
        try:
            processing_time = time.time() - g.request_start_time
            
            async with get_async_session() as session:
                await self.audit_logger.log_security_event(
                    session=session,
                    event_type=SecurityEventType.API_ACCESS,
                    user_id=g.security_context.user_id,
                    org_id=g.security_context.org_id,
                    action=f"response_{response.status_code}",
                    ip_address=g.security_context.ip_address,
                    user_agent=g.security_context.user_agent,
                    success=response.status_code < 400,
                    details={
                        "status_code": response.status_code,
                        "processing_time_ms": round(processing_time * 1000, 2),
                        "response_size": len(response.get_data())
                    }
                )
        except Exception as e:
            logger.error(f"Failed to log response: {e}")
    
    async def _log_security_error(self, error_message: str):
        """Log security-related errors"""
        try:
            async with get_async_session() as session:
                await self.audit_logger.log_security_event(
                    session=session,
                    event_type=SecurityEventType.SECURITY_VIOLATION,
                    user_id=getattr(g.security_context, 'user_id', None),
                    org_id=getattr(g.security_context, 'org_id', None),
                    action="middleware_error",
                    ip_address=getattr(g.security_context, 'ip_address', None),
                    user_agent=getattr(g.security_context, 'user_agent', None),
                    success=False,
                    details={"error": error_message}
                )
        except Exception as e:
            logger.error(f"Failed to log security error: {e}")
    
    def _get_client_ip(self) -> str:
        """Get the real client IP address"""
        # Check for forwarded headers (be careful with these in production)
        forwarded_ips = [
            request.headers.get('X-Forwarded-For'),
            request.headers.get('X-Real-IP'),
            request.headers.get('CF-Connecting-IP'),  # Cloudflare
        ]
        
        for ip in forwarded_ips:
            if ip:
                # Take the first IP if there are multiple
                return ip.split(',')[0].strip()
        
        return request.remote_addr or "unknown"
    
    def _add_security_headers(self, response):
        """Add security headers to response"""
        security_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'",
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value
    
    def _handle_unauthorized(self, error):
        """Handle 401 Unauthorized errors"""
        return jsonify({
            "error": "Unauthorized",
            "message": "Authentication required"
        }), 401
    
    def _handle_forbidden(self, error):
        """Handle 403 Forbidden errors"""
        return jsonify({
            "error": "Forbidden",
            "message": "Insufficient permissions"
        }), 403
    
    def _handle_rate_limited(self, error):
        """Handle 429 Rate Limited errors"""
        return jsonify({
            "error": "Rate limit exceeded",
            "message": "Too many requests"
        }), 429

# Decorators for endpoint protection

def require_auth(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not g.security_context.is_authenticated:
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)
    return decorated_function

def require_roles(*required_roles):
    """Decorator to require specific roles"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not g.security_context.is_authenticated:
                return jsonify({"error": "Authentication required"}), 401
            
            user_roles = set(g.security_context.roles)
            if not any(role in user_roles for role in required_roles):
                return jsonify({"error": "Insufficient permissions"}), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def require_permissions(*required_permissions):
    """Decorator to require specific permissions"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not g.security_context.is_authenticated:
                return jsonify({"error": "Authentication required"}), 401
            
            user_permissions = set(g.security_context.permissions)
            if not all(perm in user_permissions for perm in required_permissions):
                return jsonify({"error": "Insufficient permissions"}), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def require_low_risk(max_risk: float = 0.5):
    """Decorator to require low risk score"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if g.security_context.risk_score > max_risk:
                return jsonify({
                    "error": "High risk request blocked",
                    "risk_score": g.security_context.risk_score
                }), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Global middleware instance
_enhanced_security_middleware: Optional[EnhancedSecurityMiddleware] = None

def get_enhanced_security_middleware() -> EnhancedSecurityMiddleware:
    """Get the global enhanced security middleware instance"""
    global _enhanced_security_middleware
    if _enhanced_security_middleware is None:
        _enhanced_security_middleware = EnhancedSecurityMiddleware()
    return _enhanced_security_middleware