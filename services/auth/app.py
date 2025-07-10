"""
Lift OS Core - Authentication Service
Production-ready with enhanced security, health checks, and logging
"""
import time
import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, status, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
import bcrypt
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from contextlib import asynccontextmanager

# Import shared modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# KSE-SDK Integration
from shared.kse_sdk.client import LiftKSEClient
from shared.kse_sdk.core import Entity, SearchQuery, SearchResult
# EntityType and Domain are not available in the current KSE SDK models
# from shared.kse_sdk.models import EntityType, Domain

# Production-ready imports
from shared.health.health_checks import HealthChecker, check_database_connection
from shared.security.security_manager import get_security_manager, SecurityMiddleware
from shared.logging.structured_logger import setup_service_logging, get_security_logger, get_database_logger
from shared.config.secrets_manager import get_secrets_manager, get_jwt_config

# Legacy imports (to be migrated)
from shared.models.base import (
    APIResponse, HealthCheck, UserRole, SubscriptionTier
)
from shared.models.causal_marketing import (
    CausalAccessRequest, CausalAccessResponse, CausalPermissionLevel,
    CausalDataScope, CausalRoleRequest, CausalRoleResponse
)
from shared.auth.jwt_utils import create_access_token, verify_token
from shared.utils.config import get_service_config
from shared.database import get_database, User, Session, BillingAccount

# Service configuration
config = get_service_config("auth", 8001)

# Production-ready logging
logger = setup_service_logging("auth", log_level="INFO")
security_logger = get_security_logger("auth")
db_logger = get_database_logger("auth")

# Health checker
health_checker = HealthChecker("auth")

# Security manager
security_manager = get_security_manager()

# Database manager
db_manager = None

# KSE Client for intelligence integration
kse_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan with production-ready initialization"""
    global db_manager, kse_client
    
    # Startup
    from shared.database import DatabaseManager
    db_manager = DatabaseManager()
    await db_manager.initialize()
    
    # Initialize KSE Client for intelligence integration
    try:
        kse_client = LiftKSEClient()
        logger.info("KSE Client initialized successfully for auth service")
    except Exception as e:
        logger.warning(f"KSE Client initialization failed: {e}")
        kse_client = None
    
    # Configure health checks
    health_checker.add_readiness_check(check_database_connection, "database")
    
    # Initialize secrets manager
    try:
        secrets_manager = get_secrets_manager()
        logger.info("Secrets manager initialized successfully")
    except Exception as e:
        logger.warning(f"Secrets manager initialization failed: {e}")
    
    # Create default admin user if none exists
    await create_default_admin()
    
    logger.info(
        "Authentication service started",
        extra={
            "service": "auth",
            "version": "1.0.0",
            "port": config.PORT,
            "debug": config.DEBUG
        }
    )
    
    yield
    
    # Shutdown
    if db_manager:
        await db_manager.close()
    
    # Clear security caches
    security_manager.rate_limits.clear()
    security_manager.blocked_ips.clear()
    
    logger.info("Authentication service stopped gracefully")

# FastAPI app with enhanced configuration
app = FastAPI(
    title="Lift OS Core - Authentication Service",
    description="Production-ready Authentication and Authorization Service with enhanced security",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "health", "description": "Health check operations"},
        {"name": "auth", "description": "Authentication operations"},
        {"name": "users", "description": "User management operations"},
        {"name": "sessions", "description": "Session management operations"},
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


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Security middleware with rate limiting and monitoring"""
    try:
        # Apply rate limiting (skip for health checks)
        if not request.url.path.startswith("/health"):
            await security_manager.rate_limit(
                request,
                max_requests=50,  # 50 requests per minute for auth service
                window_seconds=60
            )
        
        # Process request
        response = await call_next(request)
        
        # Log authentication events
        if request.url.path in ["/login", "/register", "/logout"]:
            user_id = "unknown"
            success = response.status_code < 400
            
            security_logger.log_authentication(
                user_id=user_id,
                success=success,
                ip_address=request.client.host if request.client else "unknown"
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
        
        raise
    except Exception as e:
        logger.error(f"Security middleware error: {e}", exc_info=True)
        raise

# Security
security = HTTPBearer()

# Request/Response Models
class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    username: str
    full_name: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: Dict[str, Any]

class PasswordResetRequest(BaseModel):
    email: EmailStr

class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str

# Database helper functions
async def get_db_session() -> AsyncSession:
    """Get database session"""
    if not db_manager:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not initialized"
        )
    
    async with db_manager.get_session() as session:
        yield session

def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

async def get_user_by_email(session: AsyncSession, email: str) -> Optional[User]:
    """Get user by email"""
    result = await session.execute(
        select(User).where(User.email == email)
    )
    return result.scalar_one_or_none()

async def get_user_by_id(session: AsyncSession, user_id: str) -> Optional[User]:
    """Get user by ID"""
    result = await session.execute(
        select(User).where(User.id == uuid.UUID(user_id))
    )
    return result.scalar_one_or_none()

async def create_user_session(session: AsyncSession, user: User, token_jti: str, device_info: Dict = None) -> Session:
    """Create a new user session"""
    expires_at = datetime.now(timezone.utc) + timedelta(hours=config.JWT_EXPIRATION_HOURS)
    
    user_session = Session(
        user_id=user.id,
        token_jti=token_jti,
        device_info=device_info,
        expires_at=expires_at,
        is_active=True
    )
    
    session.add(user_session)
    await session.commit()
    return user_session

async def create_default_admin():
    """Create default admin user if none exists"""
    try:
        async with db_manager.get_session() as session:
            # Check if any users exist
            result = await session.execute(select(User).limit(1))
            existing_user = result.scalar_one_or_none()
            
            if not existing_user:
                # Create default admin user
                admin_user = User(
                    email="admin@lift.dev",
                    username="admin",
                    hashed_password=hash_password("admin123"),  # Change in production!
                    full_name="Admin User",
                    is_active=True,
                    is_verified=True,
                    is_superuser=True
                )
                
                session.add(admin_user)
                await session.flush()  # Get the user ID
                
                # Create billing account
                billing_account = BillingAccount(
                    user_id=admin_user.id,
                    plan_type="enterprise",
                    billing_email="admin@lift.dev",
                    status="active"
                )
                
                session.add(billing_account)
                await session.commit()
                
                logger.info("Created default admin user: admin@lift.dev")
                
    except Exception as e:
        logger.error(f"Failed to create default admin user: {e}")

@app.get("/health", tags=["health"])
async def health_check():
    """Auth service liveness probe - service is running"""
    return await health_checker.get_health_status()


@app.get("/ready", tags=["health"])
async def readiness_check():
    """Auth service readiness probe - service is ready to handle requests"""
    return await health_checker.get_readiness_status()


@app.get("/health/detailed", response_model=HealthCheck)
async def detailed_health_check():
    """Detailed health check with database statistics"""
    try:
        async with db_manager.get_session() as session:
            # Count users
            result = await session.execute(select(User))
            users_count = len(result.scalars().all())
            
            # Count active sessions
            active_sessions = await session.execute(
                select(Session).where(Session.expires_at > datetime.utcnow())
            )
            sessions_count = len(active_sessions.scalars().all())
            
            return HealthCheck(
                status="healthy",
                dependencies={
                    "database": "connected",
                    "users_count": str(users_count),
                    "active_sessions": str(sessions_count)
                },
                uptime=time.time() - getattr(app.state, "start_time", time.time())
            )
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return HealthCheck(
            status="unhealthy",
            dependencies={
                "database": "disconnected",
                "error": str(e)
            },
            uptime=time.time() - getattr(app.state, "start_time", time.time())
        )

@app.get("/", response_model=APIResponse)
async def root():
    """Auth service root endpoint"""
    return APIResponse(
        message="Lift OS Core Authentication Service",
        data={
            "version": "1.0.0",
            "endpoints": ["/login", "/register", "/verify", "/refresh"],
            "docs": "/docs"
        }
    )

@app.post("/register", response_model=APIResponse)
async def register(request: RegisterRequest):
    """Register a new user"""
    try:
        async with db_manager.get_session() as session:
            # Check if user already exists
            existing_user = await get_user_by_email(session, request.email)
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="User with this email already exists"
                )
            
            # Check if username is taken
            result = await session.execute(
                select(User).where(User.username == request.username)
            )
            existing_username = result.scalar_one_or_none()
            if existing_username:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already taken"
                )
            
            # Create user
            user = User(
                email=request.email,
                username=request.username,
                hashed_password=hash_password(request.password),
                full_name=request.full_name,
                is_active=True,
                is_verified=False,  # Require email verification
                is_superuser=False
            )
            
            session.add(user)
            await session.flush()  # Get the user ID
            
            # Create billing account
            billing_account = BillingAccount(
                user_id=user.id,
                plan_type="free",
                billing_email=request.email,
                status="active",
                current_usage={"api_calls": 0, "storage_mb": 0},
                usage_limits={"api_calls": 1000, "storage_mb": 100}
            )
            
            session.add(billing_account)
            await session.commit()
            
            # Create access token
            token_jti = str(uuid.uuid4())
            access_token = create_access_token(
                user_id=str(user.id),
                org_id=str(user.id),  # Use user ID as org ID for now
                email=user.email,
                roles=[UserRole.USER],
                permissions=get_user_permissions([UserRole.USER.value]),
                subscription_tier=SubscriptionTier.FREE
            )
            
            # Create session
            await create_user_session(session, user, token_jti)
            
            # Log successful registration
            security_logger.log_auth_success(
                user_id=str(user.id),
                org_id=str(user.id),
                method="registration"
            )
            
            # Prepare user data for response
            user_response = {
                "id": str(user.id),
                "email": user.email,
                "username": user.username,
                "full_name": user.full_name,
                "is_active": user.is_active,
                "is_verified": user.is_verified,
                "created_at": user.created_at.isoformat() if user.created_at else None
            }
            
            return APIResponse(
                message="User registered successfully",
                data={
                    "access_token": access_token,
                    "token_type": "bearer",
                    "expires_in": config.JWT_EXPIRATION_HOURS * 3600,
                    "user": user_response
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@app.post("/login", response_model=APIResponse)
async def login(request: LoginRequest):
    """Authenticate user and return access token"""
    try:
        async with db_manager.get_session() as session:
            # Get user by email
            user = await get_user_by_email(session, request.email)
            if not user:
                security_logger.log_auth_failure(
                    email=request.email,
                    method="login",
                    reason="user_not_found"
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials"
                )
            
            # Verify password
            if not verify_password(request.password, user.hashed_password):
                security_logger.log_auth_failure(
                    email=request.email,
                    method="login",
                    reason="invalid_password"
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials"
                )
            
            # Check if user is active
            if not user.is_active:
                security_logger.log_auth_failure(
                    email=request.email,
                    method="login",
                    reason="user_inactive"
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Account is inactive"
                )
            
            # Update last login
            user.last_login = datetime.now(timezone.utc)
            user.updated_at = datetime.now(timezone.utc)
            
            # Get billing account for subscription tier
            result = await session.execute(
                select(BillingAccount).where(BillingAccount.user_id == user.id)
            )
            billing_account = result.scalar_one_or_none()
            subscription_tier = SubscriptionTier.FREE
            if billing_account:
                if billing_account.plan_type == "pro":
                    subscription_tier = SubscriptionTier.PRO
                elif billing_account.plan_type == "enterprise":
                    subscription_tier = SubscriptionTier.ENTERPRISE
            
            # Determine user roles
            roles = [UserRole.USER]
            if user.is_superuser:
                roles.append(UserRole.ADMIN)
            
            # Create access token
            token_jti = str(uuid.uuid4())
            access_token = create_access_token(
                user_id=str(user.id),
                org_id=str(user.id),  # Use user ID as org ID for now
                email=user.email,
                roles=roles,
                permissions=get_user_permissions([role.value for role in roles]),
                subscription_tier=subscription_tier,
                
            )
            
            # Create session
            await create_user_session(session, user, token_jti)
            
            await session.commit()
            
            # Log successful login
            security_logger.log_auth_success(
                user_id=str(user.id),
                org_id=str(user.id),
                method="login"
            )
            
            # Prepare user data for response
            user_response = {
                "id": str(user.id),
                "email": user.email,
                "username": user.username,
                "full_name": user.full_name,
                "is_active": user.is_active,
                "is_verified": user.is_verified,
                "last_login": user.last_login.isoformat() if user.last_login else None
            }
            
            return APIResponse(
                message="Login successful",
                data={
                    "access_token": access_token,
                    "token_type": "bearer",
                    "expires_in": config.JWT_EXPIRATION_HOURS * 3600,
                    "user": user_response
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@app.post("/verify", response_model=APIResponse)
async def verify_token_endpoint(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and return user info"""
    try:
        # Verify token
        claims = verify_token(credentials.credentials)
        if not claims:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
        
        async with db_manager.get_session() as session:
            # Get user data
            user = await get_user_by_id(session, claims.sub)
            if not user or not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found or inactive"
                )
            
            # Check if session exists and is active
            result = await session.execute(
                select(Session).where(
                    Session.token_jti == claims.jti,
                    Session.is_active == True
                )
            )
            user_session = result.scalar_one_or_none()
            if not user_session:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Session not found or expired"
                )
            
            # Update session last used
            user_session.last_used = datetime.now(timezone.utc)
            await session.commit()
            
            # Prepare user data for response
            user_response = {
                "id": str(user.id),
                "email": user.email,
                "username": user.username,
                "full_name": user.full_name,
                "is_active": user.is_active,
                "is_verified": user.is_verified
            }
            
            return APIResponse(
                message="Token verified",
                data={
                    "valid": True,
                    "user": user_response,
                    "claims": claims.dict()
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token verification failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token verification failed"
        )

@app.post("/refresh", response_model=APIResponse)
async def refresh_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Refresh JWT token"""
    try:
        # Verify current token
        claims = verify_token(credentials.credentials)
        if not claims:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
        
        async with db_manager.get_session() as session:
            # Get user data
            user = await get_user_by_id(session, claims.sub)
            if not user or not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found or inactive"
                )
            
            # Invalidate old session
            result = await session.execute(
                select(Session).where(Session.token_jti == claims.jti)
            )
            old_session = result.scalar_one_or_none()
            if old_session:
                old_session.is_active = False
            
            # Create new access token
            token_jti = str(uuid.uuid4())
            roles = [UserRole.USER]
            if user.is_superuser:
                roles.append(UserRole.ADMIN)
                
            access_token = create_access_token(
                user_id=str(user.id),
                org_id=str(user.id),
                email=user.email,
                roles=roles,
                permissions=get_user_permissions([role.value for role in roles]),
                subscription_tier=claims.subscription_tier,
                
            )
            
            # Create new session
            await create_user_session(session, user, token_jti)
            await session.commit()
            
            return APIResponse(
                message="Token refreshed",
                data={
                    "access_token": access_token,
                    "token_type": "bearer",
                    "expires_in": config.JWT_EXPIRATION_HOURS * 3600
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )

def get_user_permissions(roles: List[str]) -> List[str]:
    """Get permissions based on user roles"""
    permissions = set()
    
    for role in roles:
        if role == UserRole.ADMIN.value:
            permissions.update([
                "user:read", "user:write", "user:delete",
                "org:read", "org:write", "org:delete",
                "billing:read", "billing:write",
                "memory:read", "memory:write", "memory:admin",
                "modules:read", "modules:write", "modules:admin",
                # Causal permissions for admin
                "causal:read", "causal:write", "causal:admin",
                "causal:experiments:design", "causal:experiments:run",
                "causal:data:access", "causal:data:export",
                "causal:models:create", "causal:models:deploy",
                "causal:insights:generate", "causal:optimization:run"
            ])
        elif role == UserRole.ANALYST.value:
            permissions.update([
                "user:read", "user:write",
                "memory:read", "memory:write",
                "modules:read", "modules:write",
                # Causal permissions for analyst
                "causal:read", "causal:write",
                "causal:experiments:design", "causal:experiments:run",
                "causal:data:access", "causal:insights:generate",
                "causal:optimization:run"
            ])
        elif role == UserRole.DEVELOPER.value:
            permissions.update([
                "user:read", "user:write",
                "memory:read", "memory:write",
                "modules:read", "modules:write", "modules:admin",
                # Causal permissions for developer
                "causal:read", "causal:write",
                "causal:models:create", "causal:models:deploy",
                "causal:data:access"
            ])
        elif role == UserRole.VIEWER.value:
            permissions.update([
                "user:read",
                "memory:read",
                "modules:read",
                # Causal permissions for viewer
                "causal:read"
            ])
        else:  # USER
            permissions.update([
                "user:read", "user:write",
                "memory:read", "memory:write",
                "modules:read",
                # Basic causal permissions for user
                "causal:read"
            ])
    
    return list(permissions)

def get_causal_permissions(user_roles: List[str], subscription_tier: SubscriptionTier) -> Dict[str, Any]:
    """Get causal-specific permissions based on user roles and subscription tier"""
    causal_permissions = {
        "data_access": False,
        "experiment_design": False,
        "model_creation": False,
        "optimization": False,
        "export": False,
        "admin": False,
        "max_experiments": 0,
        "max_data_points": 0,
        "allowed_platforms": [],
        "advanced_methods": False
    }
    
    # Base permissions by role
    for role in user_roles:
        if role == UserRole.ADMIN.value:
            causal_permissions.update({
                "data_access": True,
                "experiment_design": True,
                "model_creation": True,
                "optimization": True,
                "export": True,
                "admin": True,
                "advanced_methods": True
            })
        elif role == UserRole.ANALYST.value:
            causal_permissions.update({
                "data_access": True,
                "experiment_design": True,
                "optimization": True,
                "advanced_methods": True
            })
        elif role == UserRole.DEVELOPER.value:
            causal_permissions.update({
                "data_access": True,
                "model_creation": True
            })
        elif role == UserRole.VIEWER.value:
            causal_permissions.update({
                "data_access": False  # Read-only access through specific endpoints
            })
    
    # Subscription tier limits
    if subscription_tier == SubscriptionTier.FREE:
        causal_permissions.update({
            "max_experiments": 5,
            "max_data_points": 10000,
            "allowed_platforms": ["meta", "google"],
            "export": False
        })
    elif subscription_tier == SubscriptionTier.PRO:
        causal_permissions.update({
            "max_experiments": 50,
            "max_data_points": 100000,
            "allowed_platforms": ["meta", "google", "klaviyo"],
            "export": True
        })
    elif subscription_tier == SubscriptionTier.ENTERPRISE:
        causal_permissions.update({
            "max_experiments": -1,  # Unlimited
            "max_data_points": -1,  # Unlimited
            "allowed_platforms": ["meta", "google", "klaviyo", "custom"],
            "export": True
        })
    
    return causal_permissions

async def validate_causal_access(
    user_id: str,
    requested_scope: CausalDataScope,
    permission_level: CausalPermissionLevel,
    session: AsyncSession
) -> bool:
    """Validate if user has access to specific causal data and operations"""
    try:
        # Get user and their roles
        user = await get_user_by_id(session, user_id)
        if not user or not user.is_active:
            return False
        
        # Get billing account for subscription tier
        result = await session.execute(
            select(BillingAccount).where(BillingAccount.user_id == user.id)
        )
        billing_account = result.scalar_one_or_none()
        subscription_tier = SubscriptionTier.FREE
        if billing_account:
            if billing_account.plan_type == "pro":
                subscription_tier = SubscriptionTier.PRO
            elif billing_account.plan_type == "enterprise":
                subscription_tier = SubscriptionTier.ENTERPRISE
        
        # Determine user roles
        roles = [UserRole.USER.value]
        if user.is_superuser:
            roles.append(UserRole.ADMIN.value)
        
        # Get causal permissions
        causal_perms = get_causal_permissions(roles, subscription_tier)
        
        # Validate based on permission level
        if permission_level == CausalPermissionLevel.READ:
            return True  # All authenticated users can read
        elif permission_level == CausalPermissionLevel.WRITE:
            return causal_perms["data_access"]
        elif permission_level == CausalPermissionLevel.EXPERIMENT:
            return causal_perms["experiment_design"]
        elif permission_level == CausalPermissionLevel.MODEL:
            return causal_perms["model_creation"]
        elif permission_level == CausalPermissionLevel.OPTIMIZE:
            return causal_perms["optimization"]
        elif permission_level == CausalPermissionLevel.EXPORT:
            return causal_perms["export"]
        elif permission_level == CausalPermissionLevel.ADMIN:
            return causal_perms["admin"]
        
        # Validate scope-specific access
        if requested_scope.platform and requested_scope.platform not in causal_perms["allowed_platforms"]:
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Causal access validation failed: {e}")
        return False

@app.post("/causal/access/validate", response_model=APIResponse, tags=["auth"])
async def validate_causal_access_endpoint(
    request: CausalAccessRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Validate causal data access permissions for a user"""
    try:
        # Verify token
        claims = verify_token(credentials.credentials)
        if not claims:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
        
        async with db_manager.get_session() as session:
            # Validate access
            has_access = await validate_causal_access(
                user_id=claims.sub,
                requested_scope=request.scope,
                permission_level=request.permission_level,
                session=session
            )
            
            # Get user for additional context
            user = await get_user_by_id(session, claims.sub)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            # Get causal permissions
            roles = [UserRole.USER.value]
            if user.is_superuser:
                roles.append(UserRole.ADMIN.value)
            
            # Get subscription tier
            result = await session.execute(
                select(BillingAccount).where(BillingAccount.user_id == user.id)
            )
            billing_account = result.scalar_one_or_none()
            subscription_tier = SubscriptionTier.FREE
            if billing_account:
                if billing_account.plan_type == "pro":
                    subscription_tier = SubscriptionTier.PRO
                elif billing_account.plan_type == "enterprise":
                    subscription_tier = SubscriptionTier.ENTERPRISE
            
            causal_permissions = get_causal_permissions(roles, subscription_tier)
            
            return APIResponse(
                message="Causal access validation completed",
                data=CausalAccessResponse(
                    user_id=claims.sub,
                    has_access=has_access,
                    permission_level=request.permission_level,
                    scope=request.scope,
                    causal_permissions=causal_permissions,
                    subscription_tier=subscription_tier.value,
                    expires_at=datetime.fromtimestamp(claims.exp, tz=timezone.utc)
                ).dict()
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Causal access validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Access validation failed"
        )

@app.get("/causal/permissions/{user_id}", response_model=APIResponse, tags=["auth"])
async def get_causal_permissions_endpoint(
    user_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get causal permissions for a specific user"""
    try:
        # Verify token
        claims = verify_token(credentials.credentials)
        if not claims:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
        
        # Check if requesting user has admin permissions or is requesting their own permissions
        if claims.sub != user_id and not any(role == UserRole.ADMIN.value for role in claims.roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to view user permissions"
            )
        
        async with db_manager.get_session() as session:
            # Get target user
            user = await get_user_by_id(session, user_id)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            # Get user roles
            roles = [UserRole.USER.value]
            if user.is_superuser:
                roles.append(UserRole.ADMIN.value)
            
            # Get subscription tier
            result = await session.execute(
                select(BillingAccount).where(BillingAccount.user_id == user.id)
            )
            billing_account = result.scalar_one_or_none()
            subscription_tier = SubscriptionTier.FREE
            if billing_account:
                if billing_account.plan_type == "pro":
                    subscription_tier = SubscriptionTier.PRO
                elif billing_account.plan_type == "enterprise":
                    subscription_tier = SubscriptionTier.ENTERPRISE
            
            # Get causal permissions
            causal_permissions = get_causal_permissions(roles, subscription_tier)
            standard_permissions = get_user_permissions([role for role in roles])
            
            return APIResponse(
                message="User permissions retrieved",
                data={
                    "user_id": user_id,
                    "roles": roles,
                    "subscription_tier": subscription_tier.value,
                    "standard_permissions": standard_permissions,
                    "causal_permissions": causal_permissions,
                    "is_active": user.is_active,
                    "is_verified": user.is_verified
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get causal permissions failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve permissions"
        )

@app.post("/causal/roles/assign", response_model=APIResponse, tags=["auth"])
async def assign_causal_role(
    request: CausalRoleRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Assign causal analyst role to a user (admin only)"""
    try:
        # Verify token
        claims = verify_token(credentials.credentials)
        if not claims:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
        
        # Check admin permissions
        if not any(role == UserRole.ADMIN.value for role in claims.roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin permissions required"
            )
        
        async with db_manager.get_session() as session:
            # Get target user
            user = await get_user_by_id(session, request.user_id)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            # For now, we'll use the is_superuser flag to indicate analyst role
            # In a more complex system, you'd have a separate roles table
            if request.role == "causal_analyst":
                # This would typically involve updating a user_roles table
                # For now, we'll log the role assignment
                logger.info(f"Causal analyst role assigned to user {request.user_id} by admin {claims.sub}")
                
                # Update user's last updated timestamp
                user.updated_at = datetime.now(timezone.utc)
                await session.commit()
                
                return APIResponse(
                    message=f"Causal analyst role assigned to user {request.user_id}",
                    data=CausalRoleResponse(
                        user_id=request.user_id,
                        role=request.role,
                        assigned_by=claims.sub,
                        assigned_at=datetime.now(timezone.utc),
                        permissions=get_causal_permissions([UserRole.ANALYST.value], SubscriptionTier.PRO)
                    ).dict()
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid role specified"
                )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Role assignment failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Role assignment failed"
        )

@app.get("/causal/capabilities", response_model=APIResponse, tags=["auth"])
async def get_causal_capabilities():
    """Get available causal capabilities and permission levels"""
    return APIResponse(
        message="Causal capabilities retrieved",
        data={
            "permission_levels": [level.value for level in CausalPermissionLevel],
            "data_scopes": {
                "platforms": ["meta", "google", "klaviyo", "custom"],
                "time_ranges": ["1d", "7d", "30d", "90d", "custom"],
                "data_types": ["campaigns", "ad_sets", "ads", "audiences", "creatives"]
            },
            "subscription_tiers": {
                "free": {
                    "max_experiments": 5,
                    "max_data_points": 10000,
                    "allowed_platforms": ["meta", "google"],
                    "advanced_methods": False
                },
                "pro": {
                    "max_experiments": 50,
                    "max_data_points": 100000,
                    "allowed_platforms": ["meta", "google", "klaviyo"],
                    "advanced_methods": True
                },
                "enterprise": {
                    "max_experiments": -1,
                    "max_data_points": -1,
                    "allowed_platforms": ["meta", "google", "klaviyo", "custom"],
                    "advanced_methods": True
                }
            },
            "available_methods": [
                "difference_in_differences",
                "instrumental_variables", 
                "synthetic_control",
                "causal_discovery",
                "propensity_score_matching",
                "regression_discontinuity"
            ]
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level=config.LOG_LEVEL.lower()
    )