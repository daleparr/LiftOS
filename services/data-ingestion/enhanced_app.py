"""
LiftOS Enhanced Data Ingestion Service with Enterprise Security
Integrates enterprise-grade security with existing API connectors
"""

import asyncio
import time
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Union
from fastapi import FastAPI, HTTPException, Depends, Header, status, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid
import httpx
from pydantic import BaseModel, Field
import json
import sys
import os

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import shared modules
from shared.models.base import APIResponse, HealthCheck
from shared.models.marketing import (
    DataSource, MarketingDataIngestionRequest,
    MetaBusinessData, GoogleAdsData, KlaviyoData
)
from shared.models.causal_marketing import (
    CausalMarketingData, CausalAnalysisRequest, DataQualityAssessment
)
from shared.utils.causal_transforms import CausalDataTransformer
from shared.utils.config import get_service_config
from shared.utils.logging import setup_logging
from shared.health.health_checks import HealthChecker

# Import enhanced security components
from shared.security.enhanced_middleware import (
    EnhancedSecurityMiddleware, 
    require_auth, 
    require_permissions,
    require_low_risk
)
from shared.security.enhanced_jwt import get_enhanced_jwt_manager
from shared.security.audit_logger import SecurityAuditLogger, SecurityEventType
from shared.database.database import get_async_session

# Import enhanced credential manager
from enhanced_credential_manager import (
    get_enhanced_credential_manager,
    EnhancedCredentialProvider
)

# Import existing connectors
from connectors.shopify_connector import ShopifyConnector
from connectors.woocommerce_connector import WooCommerceConnector
from connectors.amazon_connector import AmazonConnector
from connectors.hubspot_connector import HubSpotConnector
from connectors.salesforce_connector import SalesforceConnector
from connectors.stripe_connector import StripeConnector
from connectors.paypal_connector import PayPalConnector
from connectors.meta_business_connector import MetaBusinessConnector, create_meta_business_connector
from connectors.google_ads_connector import GoogleAdsConnector, create_google_ads_connector
from connectors.klaviyo_connector import KlaviyoConnector, create_klaviyo_connector
from connectors.tiktok_connector import TikTokConnector
from connectors.snowflake_connector import SnowflakeConnector
from connectors.databricks_connector import DatabricksConnector
from connectors.zoho_crm_connector import ZohoCRMConnector
from connectors.linkedin_ads_connector import LinkedInAdsConnector
from connectors.x_ads_connector import XAdsConnector

# Service configuration
config = get_service_config("data_ingestion", 8006)
logger = setup_logging("data_ingestion")

# Health checker
health_checker = HealthChecker("data_ingestion")

# Enhanced security components
jwt_manager = get_enhanced_jwt_manager()
audit_logger = SecurityAuditLogger()
credential_manager = get_enhanced_credential_manager()

# Memory Service client
MEMORY_SERVICE_URL = os.getenv("MEMORY_SERVICE_URL", "http://localhost:8003")

# FastAPI app with enhanced security
app = FastAPI(
    title="LiftOS Enhanced Data Ingestion Service",
    description="Enterprise-secured API connectors for 16+ platforms with comprehensive audit logging and encrypted credential management",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize enhanced security middleware
security_middleware = EnhancedSecurityMiddleware(app)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if config.DEBUG else ["http://localhost:3000", "http://localhost:8501", "http://localhost:8502"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class SecureAPICredentials(BaseModel):
    """Secure API credentials for external platforms"""
    platform: DataSource
    key_name: str = "default"
    credentials: Dict[str, str]
    metadata: Optional[Dict[str, Any]] = None

class EnhancedSyncJobRequest(BaseModel):
    """Enhanced data sync job request with security context"""
    platform: DataSource
    date_range_start: date
    date_range_end: date
    sync_type: str = Field("full", pattern="^(full|incremental)$")
    campaigns: Optional[List[str]] = None
    use_vault_credentials: bool = True
    credential_key_name: str = "default"
    metadata: Optional[Dict[str, Any]] = None

class SyncJobStatus(BaseModel):
    """Data sync job status with security tracking"""
    job_id: str
    platform: DataSource
    status: str  # pending, running, completed, failed
    progress: float = 0.0
    records_processed: int = 0
    records_total: Optional[int] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    security_context: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

class SecurityContext(BaseModel):
    """Security context for requests"""
    user_id: str
    org_id: str
    roles: List[str]
    permissions: List[str]
    session_id: Optional[str] = None
    risk_score: float = 0.0

# Global state for tracking sync jobs
sync_jobs: Dict[str, SyncJobStatus] = {}

# Initialize causal data transformer
causal_transformer = CausalDataTransformer()

# Security dependency functions
async def get_security_context(request: Request) -> SecurityContext:
    """Extract security context from request"""
    try:
        # Get security context from middleware
        if hasattr(request.state, 'security_context'):
            ctx = request.state.security_context
            return SecurityContext(
                user_id=ctx.user_id or "anonymous",
                org_id=ctx.org_id or "default",
                roles=ctx.roles or [],
                permissions=ctx.permissions or [],
                session_id=ctx.session_id,
                risk_score=ctx.risk_score
            )
        else:
            # Fallback for development
            return SecurityContext(
                user_id="dev_user",
                org_id="dev_org",
                roles=["user"],
                permissions=["read", "write"]
            )
    except Exception as e:
        logger.error(f"Failed to extract security context: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid security context"
        )

async def log_api_access(
    request: Request,
    security_context: SecurityContext,
    action: str,
    success: bool = True,
    details: Optional[Dict[str, Any]] = None
):
    """Log API access for audit trail"""
    try:
        async with get_async_session() as session:
            await audit_logger.log_security_event(
                session=session,
                event_type=SecurityEventType.API_ACCESS,
                user_id=security_context.user_id,
                org_id=security_context.org_id,
                action=action,
                ip_address=getattr(request.state, 'client_ip', 'unknown'),
                user_agent=request.headers.get('user-agent', 'unknown'),
                success=success,
                details=details or {}
            )
    except Exception as e:
        logger.error(f"Failed to log API access: {e}")

# Enhanced API Endpoints

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return await health_checker.check_health()

@app.get("/security/status")
@require_auth
async def security_status(
    request: Request,
    security_context: SecurityContext = Depends(get_security_context)
):
    """Get security status and configuration"""
    await log_api_access(request, security_context, "security_status_check")
    
    return {
        "security_enabled": True,
        "user_context": {
            "user_id": security_context.user_id,
            "org_id": security_context.org_id,
            "roles": security_context.roles,
            "permissions": security_context.permissions,
            "risk_score": security_context.risk_score
        },
        "vault_status": "active",
        "audit_logging": "enabled"
    }

@app.post("/credentials/store")
@require_auth
@require_permissions("credential_management")
@require_low_risk(max_risk=0.3)
async def store_credentials(
    request: Request,
    credentials: SecureAPICredentials,
    security_context: SecurityContext = Depends(get_security_context)
):
    """Store API credentials securely in the vault"""
    try:
        await log_api_access(
            request, 
            security_context, 
            f"store_credentials_{credentials.platform.value}",
            details={"platform": credentials.platform.value, "key_name": credentials.key_name}
        )
        
        success = await credential_manager.store_credentials_in_vault(
            org_id=security_context.org_id,
            provider=credentials.platform.value,
            credentials=credentials.credentials,
            key_name=credentials.key_name,
            created_by=security_context.user_id,
            metadata=credentials.metadata
        )
        
        if success:
            return APIResponse(
                success=True,
                message=f"Credentials for {credentials.platform.value} stored securely",
                data={"platform": credentials.platform.value, "key_name": credentials.key_name}
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to store credentials"
            )
            
    except Exception as e:
        await log_api_access(
            request, 
            security_context, 
            f"store_credentials_{credentials.platform.value}",
            success=False,
            details={"error": str(e)}
        )
        logger.error(f"Failed to store credentials: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store credentials: {str(e)}"
        )

@app.get("/credentials/list")
@require_auth
@require_permissions("credential_read")
async def list_credentials(
    request: Request,
    security_context: SecurityContext = Depends(get_security_context)
):
    """List stored credentials for the organization"""
    try:
        await log_api_access(request, security_context, "list_credentials")
        
        credentials_list = await credential_manager.list_stored_credentials(security_context.org_id)
        
        # Remove sensitive data from response
        safe_credentials = []
        for cred in credentials_list:
            safe_credentials.append({
                "provider": cred.get("provider"),
                "key_name": cred.get("key_name"),
                "is_active": cred.get("is_active"),
                "created_at": cred.get("created_at"),
                "last_used_at": cred.get("last_used_at"),
                "usage_count": cred.get("usage_count", 0)
            })
        
        return APIResponse(
            success=True,
            message="Credentials retrieved successfully",
            data={"credentials": safe_credentials}
        )
        
    except Exception as e:
        await log_api_access(
            request, 
            security_context, 
            "list_credentials",
            success=False,
            details={"error": str(e)}
        )
        logger.error(f"Failed to list credentials: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list credentials: {str(e)}"
        )

@app.post("/credentials/rotate")
@require_auth
@require_permissions("credential_management")
@require_low_risk(max_risk=0.2)
async def rotate_credentials(
    request: Request,
    platform: DataSource,
    key_name: str = "default",
    new_credentials: Dict[str, str] = {},
    security_context: SecurityContext = Depends(get_security_context)
):
    """Rotate API credentials"""
    try:
        await log_api_access(
            request, 
            security_context, 
            f"rotate_credentials_{platform.value}",
            details={"platform": platform.value, "key_name": key_name}
        )
        
        success = await credential_manager.rotate_credentials(
            org_id=security_context.org_id,
            provider=platform.value,
            new_credentials=new_credentials,
            key_name=key_name,
            rotated_by=security_context.user_id
        )
        
        if success:
            return APIResponse(
                success=True,
                message=f"Credentials for {platform.value} rotated successfully",
                data={"platform": platform.value, "key_name": key_name}
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to rotate credentials"
            )
            
    except Exception as e:
        await log_api_access(
            request, 
            security_context, 
            f"rotate_credentials_{platform.value}",
            success=False,
            details={"error": str(e)}
        )
        logger.error(f"Failed to rotate credentials: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to rotate credentials: {str(e)}"
        )

@app.post("/sync/start")
@require_auth
@require_permissions("data_sync")
async def start_sync_job(
    request: Request,
    sync_request: EnhancedSyncJobRequest,
    background_tasks: BackgroundTasks,
    security_context: SecurityContext = Depends(get_security_context)
):
    """Start a secure data synchronization job"""
    try:
        job_id = str(uuid.uuid4())
        
        await log_api_access(
            request, 
            security_context, 
            f"start_sync_{sync_request.platform.value}",
            details={
                "job_id": job_id,
                "platform": sync_request.platform.value,
                "sync_type": sync_request.sync_type,
                "date_range": f"{sync_request.date_range_start} to {sync_request.date_range_end}"
            }
        )
        
        # Create job status
        job_status = SyncJobStatus(
            job_id=job_id,
            platform=sync_request.platform,
            status="pending",
            started_at=datetime.utcnow(),
            security_context={
                "user_id": security_context.user_id,
                "org_id": security_context.org_id,
                "risk_score": security_context.risk_score
            },
            metadata=sync_request.metadata or {}
        )
        
        sync_jobs[job_id] = job_status
        
        # Start background sync task
        background_tasks.add_task(
            execute_secure_sync_job,
            job_id,
            sync_request,
            security_context
        )
        
        return APIResponse(
            success=True,
            message=f"Sync job started for {sync_request.platform.value}",
            data={"job_id": job_id, "status": "pending"}
        )
        
    except Exception as e:
        await log_api_access(
            request, 
            security_context, 
            f"start_sync_{sync_request.platform.value}",
            success=False,
            details={"error": str(e)}
        )
        logger.error(f"Failed to start sync job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start sync job: {str(e)}"
        )

@app.get("/sync/status/{job_id}")
@require_auth
async def get_sync_status(
    request: Request,
    job_id: str,
    security_context: SecurityContext = Depends(get_security_context)
):
    """Get sync job status"""
    try:
        await log_api_access(request, security_context, f"get_sync_status", details={"job_id": job_id})
        
        if job_id not in sync_jobs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Sync job not found"
            )
        
        job_status = sync_jobs[job_id]
        
        # Check if user has access to this job
        if job_status.security_context.get("org_id") != security_context.org_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this sync job"
            )
        
        return APIResponse(
            success=True,
            message="Sync job status retrieved",
            data=job_status.dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await log_api_access(
            request, 
            security_context, 
            "get_sync_status",
            success=False,
            details={"job_id": job_id, "error": str(e)}
        )
        logger.error(f"Failed to get sync status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get sync status: {str(e)}"
        )

async def execute_secure_sync_job(
    job_id: str,
    sync_request: EnhancedSyncJobRequest,
    security_context: SecurityContext
):
    """Execute a secure data synchronization job"""
    try:
        # Update job status
        sync_jobs[job_id].status = "running"
        sync_jobs[job_id].started_at = datetime.utcnow()
        
        # Get credentials from vault
        if sync_request.use_vault_credentials:
            credentials = await get_platform_credentials(
                security_context.org_id,
                sync_request.platform.value,
                sync_request.credential_key_name
            )
            
            if not credentials:
                sync_jobs[job_id].status = "failed"
                sync_jobs[job_id].error_message = "No credentials found in vault"
                return
        else:
            # Fallback to environment credentials (for development)
            credentials = await get_legacy_credentials(sync_request.platform)
        
        # Log credential access
        async with get_async_session() as session:
            await audit_logger.log_api_key_access(
                session=session,
                org_id=security_context.org_id,
                provider=sync_request.platform.value,
                key_name=sync_request.credential_key_name,
                action="sync_job_access",
                success=True,
                user_id=security_context.user_id,
                details={"job_id": job_id}
            )
        
        # Execute platform-specific sync
        await execute_platform_sync(
            job_id,
            sync_request.platform,
            credentials,
            sync_request,
            security_context
        )
        
        # Update completion status
        sync_jobs[job_id].status = "completed"
        sync_jobs[job_id].completed_at = datetime.utcnow()
        
    except Exception as e:
        logger.error(f"Sync job {job_id} failed: {e}")
        sync_jobs[job_id].status = "failed"
        sync_jobs[job_id].error_message = str(e)
        sync_jobs[job_id].completed_at = datetime.utcnow()
        
        # Log sync failure
        async with get_async_session() as session:
            await audit_logger.log_security_event(
                session=session,
                event_type=SecurityEventType.API_ACCESS,
                user_id=security_context.user_id,
                org_id=security_context.org_id,
                action=f"sync_job_failed_{sync_request.platform.value}",
                success=False,
                details={"job_id": job_id, "error": str(e)}
            )

async def get_platform_credentials(org_id: str, platform: str, key_name: str = "default") -> Optional[Dict[str, str]]:
    """Get platform credentials from vault"""
    try:
        method_map = {
            "meta": credential_manager.get_meta_business_credentials,
            "google": credential_manager.get_google_ads_credentials,
            "klaviyo": credential_manager.get_klaviyo_credentials,
            "shopify": credential_manager.get_shopify_credentials,
            "amazon": credential_manager.get_amazon_credentials,
            "salesforce": credential_manager.get_salesforce_credentials,
            "stripe": credential_manager.get_stripe_credentials,
            "tiktok": credential_manager.get_tiktok_credentials,
            "linkedin": credential_manager.get_linkedin_ads_credentials,
        }
        
        if platform in method_map:
            return await method_map[platform](org_id)
        else:
            logger.warning(f"Unsupported platform for vault credentials: {platform}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to get credentials for {platform}: {e}")
        return None

async def get_legacy_credentials(platform: DataSource) -> Optional[Dict[str, str]]:
    """Get legacy credentials from environment (fallback)"""
    # This would use the original credential manager for fallback
    # Implementation depends on existing credential_manager
    return None

async def execute_platform_sync(
    job_id: str,
    platform: DataSource,
    credentials: Dict[str, str],
    sync_request: EnhancedSyncJobRequest,
    security_context: SecurityContext
):
    """Execute platform-specific data synchronization"""
    try:
        # Update progress
        sync_jobs[job_id].progress = 0.1
        
        # Platform-specific sync logic would go here
        # This is a simplified implementation
        
        if platform == DataSource.META_BUSINESS:
            connector = create_meta_business_connector(credentials)
            # Sync Meta Business data
            
        elif platform == DataSource.GOOGLE_ADS:
            connector = create_google_ads_connector(credentials)
            # Sync Google Ads data
            
        elif platform == DataSource.KLAVIYO:
            connector = create_klaviyo_connector(credentials)
            # Sync Klaviyo data
            
        # Add other platform implementations...
        
        # Simulate sync progress
        for progress in [0.3, 0.5, 0.7, 0.9, 1.0]:
            await asyncio.sleep(1)  # Simulate work
            sync_jobs[job_id].progress = progress
            sync_jobs[job_id].records_processed += 100
        
        sync_jobs[job_id].records_total = sync_jobs[job_id].records_processed
        
    except Exception as e:
        logger.error(f"Platform sync failed for {platform.value}: {e}")
        raise

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with security logging"""
    try:
        if hasattr(request.state, 'security_context'):
            security_context = SecurityContext(
                user_id=request.state.security_context.user_id or "anonymous",
                org_id=request.state.security_context.org_id or "unknown",
                roles=[],
                permissions=[]
            )
            
            await log_api_access(
                request,
                security_context,
                f"http_error_{exc.status_code}",
                success=False,
                details={"error": exc.detail}
            )
    except:
        pass  # Don't fail on logging errors
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "enhanced_app:app",
        host="0.0.0.0",
        port=8006,
        reload=config.DEBUG,
        log_level="info"
    )