"""
Lift OS Core - API Gateway Service
Production-ready with enhanced security, health checks, and logging
"""
import asyncio
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import FastAPI, Request, Response, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import httpx
import time

# Import shared modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# KSE-SDK Integration
from shared.kse_sdk.client import LiftKSEClient
from shared.kse_sdk.core import Entity, SearchQuery, SearchResult
from shared.kse_sdk.models import EntityType, Domain

# Production-ready imports
from shared.health.health_checks import HealthChecker, check_database_connection, check_external_service
from shared.security.security_manager import get_security_manager, SecurityMiddleware
from shared.logging.structured_logger import setup_service_logging, get_request_logger, get_security_logger
from shared.config.secrets_manager import get_secrets_manager, get_jwt_config

# Legacy imports (to be migrated)
from shared.models.base import APIResponse, HealthCheck, JWTClaims
from shared.auth.jwt_utils import verify_token, extract_token_from_header
from shared.utils.config import get_service_config

# Service configuration
config = get_service_config("gateway", 8000)

# Production-ready logging
logger = setup_service_logging("gateway", log_level="INFO")
request_logger = get_request_logger("gateway")
security_logger = get_security_logger("gateway")

# Health checker
health_checker = HealthChecker("gateway")

# Security manager
security_manager = get_security_manager()

# FastAPI app with enhanced configuration
app = FastAPI(
    title="Lift OS Core - API Gateway",
    description="Unified API Gateway for Lift OS Core services with production-ready security and monitoring",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "health", "description": "Health check operations"},
        {"name": "auth", "description": "Authentication service proxy"},
        {"name": "billing", "description": "Billing service proxy"},
        {"name": "memory", "description": "Memory service proxy"},
        {"name": "registry", "description": "Registry service proxy"},
        {"name": "observability", "description": "Observability service proxy"},
        {"name": "intelligence", "description": "Intelligence service proxy"},
        {"name": "feedback", "description": "Feedback service proxy"},
        {"name": "business-metrics", "description": "Business metrics and KPI tracking"},
        {"name": "user-analytics", "description": "User behavior and analytics"},
        {"name": "impact-monitoring", "description": "Decision impact monitoring"},
        {"name": "strategic-intelligence", "description": "Strategic intelligence and market analysis"},
        {"name": "business-intelligence", "description": "Business intelligence dashboards and reporting"},
        {"name": "modules", "description": "Dynamic module proxy"},
    ]
)

# Production-ready middleware
SecurityMiddleware.add_security_middleware(
    app,
    allowed_hosts=["*"] if config.DEBUG else ["localhost", "127.0.0.1", "*.lift.co"],
    force_https=False  # Disable HTTPS redirect for development
)

SecurityMiddleware.add_cors_middleware(
    app,
    allowed_origins=["*"] if config.DEBUG else [
        "http://localhost:3000",
        "http://localhost:8080",
        "https://*.lift.co"
    ]
)

# Service URLs
SERVICE_URLS = {
    "auth": config.AUTH_SERVICE_URL,
    "billing": config.BILLING_SERVICE_URL,
    "memory": config.MEMORY_SERVICE_URL,
    "observability": config.OBSERVABILITY_SERVICE_URL,
    "registry": config.REGISTRY_SERVICE_URL,
    "intelligence": os.getenv("INTELLIGENCE_SERVICE_URL", "http://localhost:8009"),
    "feedback": os.getenv("FEEDBACK_SERVICE_URL", "http://localhost:8010"),
    # Phase 4: Business Observability Services
    "business-metrics": os.getenv("BUSINESS_METRICS_SERVICE_URL", "http://localhost:8012"),
    "user-analytics": os.getenv("USER_ANALYTICS_SERVICE_URL", "http://localhost:8013"),
    "impact-monitoring": os.getenv("IMPACT_MONITORING_SERVICE_URL", "http://localhost:8014"),
    "strategic-intelligence": os.getenv("STRATEGIC_INTELLIGENCE_SERVICE_URL", "http://localhost:8015"),
    "business-intelligence": os.getenv("BUSINESS_INTELLIGENCE_SERVICE_URL", "http://localhost:8016"),
}

# HTTP client for service communication
http_client = httpx.AsyncClient(timeout=30.0)


class GatewayMiddleware:
    """Enhanced middleware for request processing with security and monitoring"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Add correlation ID
            correlation_id = str(uuid.uuid4())
            scope["correlation_id"] = correlation_id
            
            # Add request start time
            scope["start_time"] = time.time()
        
        await self.app(scope, receive, send)


# Add custom middleware
app.add_middleware(GatewayMiddleware)


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Security middleware with rate limiting and monitoring"""
    start_time = time.time()
    
    try:
        # Apply rate limiting (skip for health checks)
        if not request.url.path.startswith("/health"):
            await security_manager.rate_limit(
                request,
                max_requests=100,  # 100 requests per minute
                window_seconds=60
            )
        
        # Process request
        response = await call_next(request)
        
        # Log security events
        if response.status_code >= 400:
            security_logger.log_security_violation(
                "http_error",
                {
                    "status_code": response.status_code,
                    "path": request.url.path,
                    "method": request.method,
                    "ip": request.client.host if request.client else "unknown"
                }
            )
        
        return response
        
    except HTTPException as e:
        # Log rate limiting and other security exceptions
        if e.status_code == 429:
            security_logger.log_rate_limit(
                ip_address=request.client.host if request.client else "unknown",
                endpoint=request.url.path,
                limit_exceeded=True
            )
        
        # Re-raise the exception
        raise
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Security middleware error: {e}", exc_info=True)
        raise


async def get_current_user(request: Request) -> Optional[JWTClaims]:
    """Extract and validate user from JWT token"""
    authorization = request.headers.get("Authorization")
    if not authorization:
        return None
    
    token = extract_token_from_header(authorization)
    if not token:
        return None
    
    return verify_token(token)


async def require_auth(request: Request) -> JWTClaims:
    """Require authentication for protected endpoints"""
    user = await get_current_user(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    return user


async def forward_request(
    service_name: str,
    path: str,
    method: str,
    headers: Dict[str, str],
    body: bytes = None,
    params: Dict[str, str] = None
) -> httpx.Response:
    """Forward request to a service"""
    service_url = SERVICE_URLS.get(service_name)
    if not service_url:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Service {service_name} not found"
        )
    
    url = f"{service_url}{path}"
    
    # Remove hop-by-hop headers
    filtered_headers = {
        k: v for k, v in headers.items()
        if k.lower() not in ["host", "connection", "upgrade"]
    }
    
    try:
        response = await http_client.request(
            method=method,
            url=url,
            headers=filtered_headers,
            content=body,
            params=params
        )
        return response
    except httpx.RequestError as e:
        logger.error(f"Request to {service_name} failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service {service_name} unavailable"
        )


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log all requests and responses using shared RequestLogger"""
    correlation_id = str(uuid.uuid4())
    request.scope["correlation_id"] = correlation_id
    
    # Use the shared RequestLogger which handles all logging internally
    response = await request_logger.log_request(request, call_next)
    
    # Add correlation ID to response headers
    response.headers["X-Correlation-ID"] = correlation_id
    
    return response


@app.get("/health", tags=["health"])
async def health_check():
    """Gateway liveness probe - service is running"""
    return await health_checker.get_health_status()


@app.get("/ready", tags=["health"])
async def readiness_check():
    """Gateway readiness probe - service is ready to handle requests"""
    return await health_checker.get_readiness_status()


@app.get("/health/detailed", response_model=HealthCheck)
async def detailed_health_check():
    """Detailed health check with service dependencies"""
    dependencies = {}
    
    # Check service health
    for service_name, service_url in SERVICE_URLS.items():
        try:
            response = await http_client.get(f"{service_url}/health", timeout=5.0)
            dependencies[service_name] = "healthy" if response.status_code == 200 else "unhealthy"
        except Exception as e:
            logger.warning(f"Health check failed for {service_name}: {e}")
            dependencies[service_name] = "unhealthy"
    
    return HealthCheck(
        status="healthy",
        dependencies=dependencies,
        uptime=time.time() - getattr(app.state, "start_time", time.time())
    )


@app.get("/", response_model=APIResponse)
async def root():
    """Gateway root endpoint"""
    return APIResponse(
        message="Lift OS Core API Gateway",
        data={
            "version": "1.0.0",
            "services": list(SERVICE_URLS.keys()),
            "docs": "/docs"
        }
    )


# Auth service routes
@app.api_route("/auth/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def auth_proxy(request: Request, path: str):
    """Proxy requests to auth service"""
    body = await request.body()
    
    response = await forward_request(
        service_name="auth",
        path=f"/{path}",
        method=request.method,
        headers=dict(request.headers),
        body=body,
        params=dict(request.query_params)
    )
    
    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=dict(response.headers)
    )


# Billing service routes
@app.api_route("/billing/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def billing_proxy(request: Request, path: str, user: JWTClaims = Depends(require_auth)):
    """Proxy requests to billing service (requires auth)"""
    body = await request.body()
    
    # Add user context to headers
    headers = dict(request.headers)
    headers["X-User-ID"] = user.sub
    headers["X-Org-ID"] = user.org_id
    headers["X-User-Roles"] = ",".join([role.value for role in user.roles])
    
    response = await forward_request(
        service_name="billing",
        path=f"/{path}",
        method=request.method,
        headers=headers,
        body=body,
        params=dict(request.query_params)
    )
    
    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=dict(response.headers)
    )


# Memory service routes
@app.api_route("/memory/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def memory_proxy(request: Request, path: str, user: JWTClaims = Depends(require_auth)):
    """Proxy requests to memory service (requires auth)"""
    body = await request.body()
    
    # Add user context to headers
    headers = dict(request.headers)
    headers["X-User-ID"] = user.sub
    headers["X-Org-ID"] = user.org_id
    headers["X-Memory-Context"] = user.memory_context or f"org_{user.org_id}_context"
    headers["X-User-Roles"] = ",".join([role.value for role in user.roles])
    
    response = await forward_request(
        service_name="memory",
        path=f"/{path}",
        method=request.method,
        headers=headers,
        body=body,
        params=dict(request.query_params)
    )
    
    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=dict(response.headers)
    )


# Registry service routes
@app.api_route("/registry/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def registry_proxy(request: Request, path: str, user: JWTClaims = Depends(require_auth)):
    """Proxy requests to registry service (requires auth)"""
    body = await request.body()
    
    # Add user context to headers
    headers = dict(request.headers)
    headers["X-User-ID"] = user.sub
    headers["X-Org-ID"] = user.org_id
    headers["X-User-Roles"] = ",".join([role.value for role in user.roles])
    
    response = await forward_request(
        service_name="registry",
        path=f"/{path}",
        method=request.method,
        headers=headers,
        body=body,
        params=dict(request.query_params)
    )
    
    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=dict(response.headers)
    )


# Observability service routes
@app.api_route("/observability/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def observability_proxy(request: Request, path: str):
    """Proxy requests to observability service"""
    body = await request.body()
    
    response = await forward_request(
        service_name="observability",
        path=f"/{path}",
        method=request.method,
        headers=dict(request.headers),
        body=body,
        params=dict(request.query_params)
    )
    
    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=dict(response.headers)
    )


# Module proxy routes (dynamic routing based on registry)
@app.api_route("/modules/{module_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def module_proxy(request: Request, module_id: str, path: str, user: JWTClaims = Depends(require_auth)):
    """Proxy requests to registered modules"""
    # Get module info from registry
    try:
        registry_response = await http_client.get(
            f"{SERVICE_URLS['registry']}/modules/{module_id}",
            headers={"Authorization": request.headers.get("Authorization")}
        )
        
        if registry_response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Module {module_id} not found"
            )
        
        module_info = registry_response.json()
        module_url = module_info["data"]["base_url"]
        
    except httpx.RequestError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Registry service unavailable"
        )
    
    # Forward request to module
    body = await request.body()
    
    # Add user context to headers
    headers = dict(request.headers)
    headers["X-User-ID"] = user.sub
    headers["X-Org-ID"] = user.org_id
    headers["X-Memory-Context"] = user.memory_context or f"org_{user.org_id}_context"
    headers["X-User-Roles"] = ",".join([role.value for role in user.roles])
    headers["X-Module-ID"] = module_id
    
    try:
        response = await http_client.request(
            method=request.method,
            url=f"{module_url}/{path}",
            headers=headers,
            content=body,
            params=dict(request.query_params)
        )
        
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers)
        )
        
    except httpx.RequestError as e:
        logger.error(f"Request to module {module_id} failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Module {module_id} unavailable"
        )


@app.on_event("startup")
async def startup_event():
    """Initialize gateway on startup with production-ready features"""
    app.state.start_time = time.time()
    
    # Configure health checks
    async def check_auth_service():
        return await check_external_service("http://localhost:8001", "auth_service")
    
    async def check_registry_service():
        return await check_external_service("http://localhost:8005", "registry_service")
    
    health_checker.add_readiness_check(check_auth_service, "auth_service")
    health_checker.add_readiness_check(check_registry_service, "registry_service")
    
    # Initialize secrets (in production, this would load from secrets manager)
    try:
        secrets_manager = get_secrets_manager()
        # Test secrets access
        logger.info("Secrets manager initialized successfully")
    except Exception as e:
        logger.warning(f"Secrets manager initialization failed: {e}")
    
    # Log startup with service info
    logger.info(
        "Lift OS Core API Gateway started",
        extra={
            "service": "gateway",
            "version": "1.0.0",
            "port": config.PORT,
            "debug": config.DEBUG,
            "services": list(SERVICE_URLS.keys())
        }
    )


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await http_client.aclose()
    
    # Clear security manager caches
    security_manager.rate_limits.clear()
    security_manager.blocked_ips.clear()
    
    logger.info("Lift OS Core API Gateway stopped gracefully")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level=config.LOG_LEVEL.lower()
    )