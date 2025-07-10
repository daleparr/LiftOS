"""
LiftOS Enhanced Channels Service
Marketing channel management with enterprise security integration
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import json
import sys
import os
from contextlib import asynccontextmanager

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.security.enhanced_middleware import SecurityMiddleware, SecurityContext
from shared.security.enhanced_jwt import get_enhanced_jwt_manager, JWTPayload
from shared.security.audit_logger import SecurityAuditLogger, SecurityEventType
from shared.security.api_key_vault import get_api_key_vault
from services.channels.enhanced_credential_manager import EnhancedCredentialManager
from shared.database.database import get_async_session
from shared.database.security_models import EnhancedUserSession
from shared.utils.config import get_service_config
from shared.utils.logging import setup_logging
from shared.models.base import APIResponse
from sqlalchemy import select

# Service configuration
config = get_service_config("channels", 8002)
logger = setup_logging("channels")

# Security components
jwt_manager = get_enhanced_jwt_manager()
audit_logger = SecurityAuditLogger()
api_key_vault = get_api_key_vault()
security_middleware = SecurityMiddleware()
credential_manager = EnhancedCredentialManager()

# FastAPI lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting Enhanced Channels Service with enterprise security")
    
    # Initialize security components
    await security_middleware.initialize()
    await credential_manager.initialize()
    
    # Log service startup
    await audit_logger.log_event(
        event_type=SecurityEventType.SYSTEM_EVENT,
        action="service_startup",
        details={"service": "channels", "version": "1.0.0", "security_enabled": True}
    )
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down Enhanced Channels Service")
    await audit_logger.log_event(
        event_type=SecurityEventType.SYSTEM_EVENT,
        action="service_shutdown",
        details={"service": "channels"}
    )

# FastAPI app with security
app = FastAPI(
    title="LiftOS Enhanced Channels Service",
    description="Marketing channel management with enterprise security",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add security middleware
app.add_middleware(SecurityMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if config.DEBUG else ["http://localhost:3000", "http://localhost:8501", "http://localhost:8502"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security dependencies
security = HTTPBearer()

async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> JWTPayload:
    """Get current authenticated user with enhanced security"""
    try:
        # Verify JWT token
        payload = await jwt_manager.verify_token(credentials.credentials)
        
        # Get security context from middleware
        security_context: SecurityContext = getattr(request.state, 'security_context', None)
        
        if not security_context:
            raise HTTPException(status_code=401, detail="Security context not found")
        
        # Verify session is still valid
        async with get_async_session() as session:
            result = await session.execute(
                select(EnhancedUserSession)
                .where(
                    EnhancedUserSession.user_id == payload.user_id,
                    EnhancedUserSession.session_id == payload.session_id,
                    EnhancedUserSession.is_active == True
                )
            )
            user_session = result.scalar_one_or_none()
            
            if not user_session:
                await audit_logger.log_event(
                    event_type=SecurityEventType.AUTHENTICATION_FAILED,
                    action="invalid_session",
                    user_id=payload.user_id,
                    ip_address=security_context.ip_address,
                    details={"session_id": payload.session_id, "reason": "session_not_found"}
                )
                raise HTTPException(status_code=401, detail="Invalid session")
        
        # Log successful authentication
        await audit_logger.log_event(
            event_type=SecurityEventType.AUTHENTICATION_SUCCESS,
            action="token_verified",
            user_id=payload.user_id,
            org_id=payload.org_id,
            ip_address=security_context.ip_address,
            details={"service": "channels", "session_id": payload.session_id}
        )
        
        return payload
        
    except Exception as e:
        await audit_logger.log_event(
            event_type=SecurityEventType.AUTHENTICATION_FAILED,
            action="token_verification_failed",
            ip_address=getattr(request.state, 'security_context', SecurityContext()).ip_address,
            details={"error": str(e), "service": "channels"}
        )
        raise HTTPException(status_code=401, detail="Invalid authentication")

async def require_permission(permission: str):
    """Dependency to require specific permission"""
    async def check_permission(
        request: Request,
        current_user: JWTPayload = Depends(get_current_user)
    ):
        if permission not in current_user.permissions:
            security_context: SecurityContext = getattr(request.state, 'security_context', SecurityContext())
            
            await audit_logger.log_event(
                event_type=SecurityEventType.AUTHORIZATION_FAILED,
                action="insufficient_permissions",
                user_id=current_user.user_id,
                org_id=current_user.org_id,
                ip_address=security_context.ip_address,
                details={
                    "required_permission": permission,
                    "user_permissions": current_user.permissions,
                    "service": "channels"
                }
            )
            raise HTTPException(status_code=403, detail=f"Permission required: {permission}")
        
        return current_user
    
    return check_permission

# Models
class ChannelConfig(BaseModel):
    """Channel configuration model"""
    channel_id: str
    channel_type: str  # facebook, google, klaviyo, shopify, etc.
    name: str
    description: Optional[str] = None
    is_active: bool = True
    config: Dict[str, Any] = {}
    credentials: Dict[str, str] = {}

class ChannelStatus(BaseModel):
    """Channel status model"""
    channel_id: str
    status: str  # active, inactive, error, syncing
    last_sync: Optional[datetime] = None
    last_error: Optional[str] = None
    metrics: Dict[str, Any] = {}

class ChannelData(BaseModel):
    """Channel data model"""
    channel_id: str
    data_type: str  # campaigns, audiences, metrics, etc.
    data: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any] = {}

# Channel management
class ChannelManager:
    """Enhanced channel manager with security integration"""
    
    def __init__(self):
        self.channels: Dict[str, ChannelConfig] = {}
        self.channel_status: Dict[str, ChannelStatus] = {}
        self.sync_tasks: Dict[str, asyncio.Task] = {}
    
    async def create_channel(
        self,
        channel_config: ChannelConfig,
        user_id: str,
        org_id: str,
        ip_address: str
    ) -> ChannelConfig:
        """Create a new marketing channel with secure credential storage"""
        try:
            # Store credentials securely
            if channel_config.credentials:
                encrypted_credentials = {}
                for key, value in channel_config.credentials.items():
                    credential_id = await credential_manager.store_credential(
                        provider=channel_config.channel_type,
                        credential_type=key,
                        credential_value=value,
                        user_id=user_id,
                        org_id=org_id
                    )
                    encrypted_credentials[key] = credential_id
                
                # Replace plain credentials with encrypted references
                channel_config.credentials = encrypted_credentials
            
            # Store channel configuration
            self.channels[channel_config.channel_id] = channel_config
            
            # Initialize channel status
            self.channel_status[channel_config.channel_id] = ChannelStatus(
                channel_id=channel_config.channel_id,
                status="inactive"
            )
            
            # Log channel creation
            await audit_logger.log_event(
                event_type=SecurityEventType.RESOURCE_CREATED,
                action="channel_created",
                user_id=user_id,
                org_id=org_id,
                ip_address=ip_address,
                details={
                    "channel_id": channel_config.channel_id,
                    "channel_type": channel_config.channel_type,
                    "channel_name": channel_config.name
                }
            )
            
            logger.info(f"Channel created: {channel_config.channel_id} ({channel_config.channel_type})")
            return channel_config
            
        except Exception as e:
            await audit_logger.log_event(
                event_type=SecurityEventType.SECURITY_VIOLATION,
                action="channel_creation_failed",
                user_id=user_id,
                org_id=org_id,
                ip_address=ip_address,
                details={
                    "channel_id": channel_config.channel_id,
                    "error": str(e)
                }
            )
            raise HTTPException(status_code=500, detail=f"Failed to create channel: {str(e)}")
    
    async def get_channel_credentials(
        self,
        channel_id: str,
        user_id: str,
        org_id: str,
        ip_address: str
    ) -> Dict[str, str]:
        """Retrieve decrypted channel credentials"""
        try:
            if channel_id not in self.channels:
                raise HTTPException(status_code=404, detail="Channel not found")
            
            channel = self.channels[channel_id]
            decrypted_credentials = {}
            
            # Decrypt stored credentials
            for key, credential_id in channel.credentials.items():
                credential_value = await credential_manager.retrieve_credential(
                    credential_id=credential_id,
                    user_id=user_id,
                    org_id=org_id
                )
                decrypted_credentials[key] = credential_value
            
            # Log credential access
            await audit_logger.log_event(
                event_type=SecurityEventType.API_KEY_ACCESS,
                action="channel_credentials_accessed",
                user_id=user_id,
                org_id=org_id,
                ip_address=ip_address,
                details={
                    "channel_id": channel_id,
                    "channel_type": channel.channel_type,
                    "credentials_accessed": list(decrypted_credentials.keys())
                }
            )
            
            return decrypted_credentials
            
        except Exception as e:
            await audit_logger.log_event(
                event_type=SecurityEventType.SECURITY_VIOLATION,
                action="credential_access_failed",
                user_id=user_id,
                org_id=org_id,
                ip_address=ip_address,
                details={
                    "channel_id": channel_id,
                    "error": str(e)
                }
            )
            raise HTTPException(status_code=500, detail=f"Failed to retrieve credentials: {str(e)}")
    
    async def sync_channel(
        self,
        channel_id: str,
        user_id: str,
        org_id: str,
        ip_address: str
    ):
        """Start channel data synchronization"""
        try:
            if channel_id not in self.channels:
                raise HTTPException(status_code=404, detail="Channel not found")
            
            # Cancel existing sync task if running
            if channel_id in self.sync_tasks:
                self.sync_tasks[channel_id].cancel()
            
            # Update status to syncing
            self.channel_status[channel_id].status = "syncing"
            
            # Start sync task
            self.sync_tasks[channel_id] = asyncio.create_task(
                self._perform_channel_sync(channel_id, user_id, org_id, ip_address)
            )
            
            # Log sync start
            await audit_logger.log_event(
                event_type=SecurityEventType.RESOURCE_ACCESSED,
                action="channel_sync_started",
                user_id=user_id,
                org_id=org_id,
                ip_address=ip_address,
                details={"channel_id": channel_id}
            )
            
        except Exception as e:
            await audit_logger.log_event(
                event_type=SecurityEventType.SECURITY_VIOLATION,
                action="channel_sync_failed",
                user_id=user_id,
                org_id=org_id,
                ip_address=ip_address,
                details={
                    "channel_id": channel_id,
                    "error": str(e)
                }
            )
            raise HTTPException(status_code=500, detail=f"Failed to start sync: {str(e)}")
    
    async def _perform_channel_sync(
        self,
        channel_id: str,
        user_id: str,
        org_id: str,
        ip_address: str
    ):
        """Perform actual channel synchronization"""
        try:
            channel = self.channels[channel_id]
            
            # Get credentials
            credentials = await self.get_channel_credentials(
                channel_id, user_id, org_id, ip_address
            )
            
            # Simulate channel-specific sync logic
            await asyncio.sleep(2)  # Simulate API calls
            
            # Update status
            self.channel_status[channel_id].status = "active"
            self.channel_status[channel_id].last_sync = datetime.now(timezone.utc)
            
            # Log successful sync
            await audit_logger.log_event(
                event_type=SecurityEventType.RESOURCE_ACCESSED,
                action="channel_sync_completed",
                user_id=user_id,
                org_id=org_id,
                ip_address=ip_address,
                details={
                    "channel_id": channel_id,
                    "channel_type": channel.channel_type,
                    "sync_duration": "2s"
                }
            )
            
        except Exception as e:
            # Update status to error
            self.channel_status[channel_id].status = "error"
            self.channel_status[channel_id].last_error = str(e)
            
            # Log sync error
            await audit_logger.log_event(
                event_type=SecurityEventType.SECURITY_VIOLATION,
                action="channel_sync_error",
                user_id=user_id,
                org_id=org_id,
                ip_address=ip_address,
                details={
                    "channel_id": channel_id,
                    "error": str(e)
                }
            )
        finally:
            # Clean up sync task
            if channel_id in self.sync_tasks:
                del self.sync_tasks[channel_id]

# Global channel manager
channel_manager = ChannelManager()

# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "channels",
        "timestamp": datetime.now(timezone.utc),
        "security_enabled": True
    }

@app.post("/channels", response_model=ChannelConfig)
async def create_channel(
    request: Request,
    channel_config: ChannelConfig,
    current_user: JWTPayload = Depends(require_permission("channels:create"))
):
    """Create a new marketing channel"""
    security_context: SecurityContext = getattr(request.state, 'security_context', SecurityContext())
    
    return await channel_manager.create_channel(
        channel_config=channel_config,
        user_id=current_user.user_id,
        org_id=current_user.org_id,
        ip_address=security_context.ip_address
    )

@app.get("/channels", response_model=List[ChannelConfig])
async def list_channels(
    current_user: JWTPayload = Depends(require_permission("channels:read"))
):
    """List all channels for the organization"""
    # Filter channels by organization
    org_channels = [
        channel for channel in channel_manager.channels.values()
        # In a real implementation, channels would be filtered by org_id
    ]
    
    return org_channels

@app.get("/channels/{channel_id}", response_model=ChannelConfig)
async def get_channel(
    channel_id: str,
    current_user: JWTPayload = Depends(require_permission("channels:read"))
):
    """Get a specific channel"""
    if channel_id not in channel_manager.channels:
        raise HTTPException(status_code=404, detail="Channel not found")
    
    return channel_manager.channels[channel_id]

@app.get("/channels/{channel_id}/status", response_model=ChannelStatus)
async def get_channel_status(
    channel_id: str,
    current_user: JWTPayload = Depends(require_permission("channels:read"))
):
    """Get channel status"""
    if channel_id not in channel_manager.channel_status:
        raise HTTPException(status_code=404, detail="Channel not found")
    
    return channel_manager.channel_status[channel_id]

@app.post("/channels/{channel_id}/sync")
async def sync_channel(
    request: Request,
    channel_id: str,
    background_tasks: BackgroundTasks,
    current_user: JWTPayload = Depends(require_permission("channels:sync"))
):
    """Start channel synchronization"""
    security_context: SecurityContext = getattr(request.state, 'security_context', SecurityContext())
    
    await channel_manager.sync_channel(
        channel_id=channel_id,
        user_id=current_user.user_id,
        org_id=current_user.org_id,
        ip_address=security_context.ip_address
    )
    
    return APIResponse(
        success=True,
        message=f"Channel {channel_id} sync started",
        data={"channel_id": channel_id, "status": "syncing"}
    )

@app.delete("/channels/{channel_id}")
async def delete_channel(
    request: Request,
    channel_id: str,
    current_user: JWTPayload = Depends(require_permission("channels:delete"))
):
    """Delete a channel"""
    security_context: SecurityContext = getattr(request.state, 'security_context', SecurityContext())
    
    if channel_id not in channel_manager.channels:
        raise HTTPException(status_code=404, detail="Channel not found")
    
    # Cancel sync task if running
    if channel_id in channel_manager.sync_tasks:
        channel_manager.sync_tasks[channel_id].cancel()
        del channel_manager.sync_tasks[channel_id]
    
    # Remove channel
    del channel_manager.channels[channel_id]
    del channel_manager.channel_status[channel_id]
    
    # Log channel deletion
    await audit_logger.log_event(
        event_type=SecurityEventType.RESOURCE_DELETED,
        action="channel_deleted",
        user_id=current_user.user_id,
        org_id=current_user.org_id,
        ip_address=security_context.ip_address,
        details={"channel_id": channel_id}
    )
    
    return APIResponse(
        success=True,
        message=f"Channel {channel_id} deleted",
        data={"channel_id": channel_id}
    )

@app.get("/security/audit")
async def get_security_audit(
    request: Request,
    hours: int = 24,
    current_user: JWTPayload = Depends(require_permission("security:audit"))
):
    """Get security audit logs for channels service"""
    security_context: SecurityContext = getattr(request.state, 'security_context', SecurityContext())
    
    # Log audit access
    await audit_logger.log_event(
        event_type=SecurityEventType.RESOURCE_ACCESSED,
        action="audit_logs_accessed",
        user_id=current_user.user_id,
        org_id=current_user.org_id,
        ip_address=security_context.ip_address,
        details={"service": "channels", "hours_requested": hours}
    )
    
    # Get recent audit logs (simplified for demo)
    return APIResponse(
        success=True,
        message="Audit logs retrieved",
        data={
            "service": "channels",
            "hours": hours,
            "logs_available": True,
            "note": "Full audit logs available through security dashboard"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "enhanced_app:app",
        host="0.0.0.0",
        port=8002,
        reload=config.DEBUG,
        log_level="info"
    )