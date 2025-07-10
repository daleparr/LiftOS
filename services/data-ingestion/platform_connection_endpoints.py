"""
Platform Connection API Endpoints
FastAPI endpoints for managing user platform connections and OAuth flows
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from fastapi.responses import RedirectResponse
import logging

from shared.models.platform_connections import (
    CreateConnectionRequest, UpdateConnectionRequest, ConnectionResponse,
    OAuthInitiateRequest, OAuthInitiateResponse, OAuthCallbackRequest,
    SyncRequest, SyncResponse, ConnectionTestResponse,
    DataPreferencesRequest, DataPreferencesResponse,
    ConnectionDashboard, PlatformConfig, SystemHealthResponse,
    BulkSyncRequest, BulkSyncResponse, PlatformDataRequest, PlatformDataResponse
)
from shared.services.platform_connection_service import get_platform_connection_service
from shared.auth.jwt_auth import get_current_user, get_current_org
from shared.utils.logging import setup_logging

logger = setup_logging("platform_connection_endpoints")

# Create router
router = APIRouter(prefix="/api/v1/platform-connections", tags=["Platform Connections"])

# Dependency injection
def get_connection_service():
    return get_platform_connection_service()

@router.get("/platforms", response_model=List[PlatformConfig])
async def get_supported_platforms(
    service = Depends(get_connection_service)
):
    """Get list of supported marketing platforms"""
    try:
        platforms = await service.get_supported_platforms()
        return platforms
    except Exception as e:
        logger.error(f"Failed to get supported platforms: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve supported platforms"
        )

@router.post("/connections", response_model=ConnectionResponse)
async def create_platform_connection(
    request: CreateConnectionRequest,
    user_id: str = Depends(get_current_user),
    org_id: str = Depends(get_current_org),
    service = Depends(get_connection_service)
):
    """Create a new platform connection"""
    try:
        connection = await service.create_connection(user_id, org_id, request)
        logger.info(f"Created connection to {request.platform} for user {user_id}")
        return connection
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to create connection: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create platform connection"
        )

@router.get("/connections", response_model=List[ConnectionResponse])
async def get_user_connections(
    platform: Optional[str] = Query(None, description="Filter by platform"),
    user_id: str = Depends(get_current_user),
    org_id: str = Depends(get_current_org),
    service = Depends(get_connection_service)
):
    """Get user's platform connections"""
    try:
        connections = await service.get_user_connections(user_id, org_id, platform)
        return connections
    except Exception as e:
        logger.error(f"Failed to get connections for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve connections"
        )

@router.get("/connections/{connection_id}", response_model=ConnectionResponse)
async def get_connection(
    connection_id: str,
    user_id: str = Depends(get_current_user),
    org_id: str = Depends(get_current_org),
    service = Depends(get_connection_service)
):
    """Get a specific platform connection"""
    try:
        connections = await service.get_user_connections(user_id, org_id)
        connection = next((c for c in connections if c.id == connection_id), None)
        if not connection:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Connection not found"
            )
        return connection
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get connection {connection_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve connection"
        )

@router.put("/connections/{connection_id}", response_model=ConnectionResponse)
async def update_platform_connection(
    connection_id: str,
    request: UpdateConnectionRequest,
    user_id: str = Depends(get_current_user),
    org_id: str = Depends(get_current_org),
    service = Depends(get_connection_service)
):
    """Update a platform connection"""
    try:
        connection = await service.update_connection(user_id, org_id, connection_id, request)
        logger.info(f"Updated connection {connection_id} for user {user_id}")
        return connection
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to update connection {connection_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update connection"
        )

@router.delete("/connections/{connection_id}")
async def delete_platform_connection(
    connection_id: str,
    user_id: str = Depends(get_current_user),
    org_id: str = Depends(get_current_org),
    service = Depends(get_connection_service)
):
    """Delete a platform connection"""
    try:
        success = await service.delete_connection(user_id, org_id, connection_id)
        if success:
            logger.info(f"Deleted connection {connection_id} for user {user_id}")
            return {"message": "Connection deleted successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Connection not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete connection {connection_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete connection"
        )

@router.post("/connections/{connection_id}/test", response_model=ConnectionTestResponse)
async def test_platform_connection(
    connection_id: str,
    user_id: str = Depends(get_current_user),
    org_id: str = Depends(get_current_org),
    service = Depends(get_connection_service)
):
    """Test a platform connection"""
    try:
        test_result = await service.test_connection(user_id, org_id, connection_id)
        return test_result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to test connection {connection_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to test connection"
        )

# OAuth Flow Endpoints
@router.post("/oauth/initiate", response_model=OAuthInitiateResponse)
async def initiate_oauth_flow(
    request: OAuthInitiateRequest,
    user_id: str = Depends(get_current_user),
    org_id: str = Depends(get_current_org),
    service = Depends(get_connection_service)
):
    """Initiate OAuth flow for a platform"""
    try:
        oauth_response = await service.initiate_oauth_flow(user_id, org_id, request)
        logger.info(f"Initiated OAuth flow for {request.platform} for user {user_id}")
        return oauth_response
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to initiate OAuth flow: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initiate OAuth flow"
        )

@router.get("/oauth/callback")
async def oauth_callback(
    state: str = Query(..., description="OAuth state token"),
    code: Optional[str] = Query(None, description="OAuth authorization code"),
    error: Optional[str] = Query(None, description="OAuth error"),
    error_description: Optional[str] = Query(None, description="OAuth error description"),
    service = Depends(get_connection_service)
):
    """Handle OAuth callback from platforms"""
    try:
        callback_request = OAuthCallbackRequest(
            state=state,
            code=code or "",
            error=error,
            error_description=error_description
        )
        
        result = await service.handle_oauth_callback(callback_request)
        
        if result['success']:
            # Redirect to success page with connection info
            redirect_url = f"/platform-connections/success?connection_id={result['connection_id']}&platform={result['platform']}"
            return RedirectResponse(url=redirect_url, status_code=302)
        else:
            # Redirect to error page
            redirect_url = f"/platform-connections/error?message={result['error_message']}"
            return RedirectResponse(url=redirect_url, status_code=302)
            
    except Exception as e:
        logger.error(f"Failed to handle OAuth callback: {str(e)}")
        redirect_url = f"/platform-connections/error?message=OAuth callback failed"
        return RedirectResponse(url=redirect_url, status_code=302)

# Data Sync Endpoints
@router.post("/connections/{connection_id}/sync", response_model=SyncResponse)
async def sync_platform_data(
    connection_id: str,
    request: SyncRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user),
    org_id: str = Depends(get_current_org),
    service = Depends(get_connection_service)
):
    """Initiate data sync for a platform connection"""
    try:
        # Set connection_id from path parameter
        request.connection_id = connection_id
        
        sync_response = await service.sync_platform_data(user_id, org_id, request)
        logger.info(f"Initiated sync for connection {connection_id}")
        return sync_response
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to initiate sync for connection {connection_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initiate data sync"
        )

@router.post("/sync/bulk", response_model=BulkSyncResponse)
async def bulk_sync_connections(
    request: BulkSyncRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user),
    org_id: str = Depends(get_current_org),
    service = Depends(get_connection_service)
):
    """Initiate bulk data sync for multiple connections"""
    try:
        sync_responses = []
        errors = []
        syncs_initiated = 0
        
        for connection_id in request.connection_ids:
            try:
                sync_request = SyncRequest(
                    connection_id=connection_id,
                    sync_type=request.sync_type,
                    sync_config=request.sync_config
                )
                
                sync_response = await service.sync_platform_data(user_id, org_id, sync_request)
                sync_responses.append(sync_response)
                syncs_initiated += 1
                
            except Exception as e:
                errors.append({
                    'error_code': 'SYNC_FAILED',
                    'error_message': str(e),
                    'error_details': {'connection_id': connection_id}
                })
        
        return BulkSyncResponse(
            total_requested=len(request.connection_ids),
            syncs_initiated=syncs_initiated,
            syncs_failed=len(errors),
            sync_responses=sync_responses,
            errors=errors
        )
        
    except Exception as e:
        logger.error(f"Failed to initiate bulk sync: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initiate bulk sync"
        )

# Dashboard and Preferences Endpoints
@router.get("/dashboard", response_model=ConnectionDashboard)
async def get_connection_dashboard(
    user_id: str = Depends(get_current_user),
    org_id: str = Depends(get_current_org),
    service = Depends(get_connection_service)
):
    """Get dashboard view of user's platform connections"""
    try:
        dashboard = await service.get_connection_dashboard(user_id, org_id)
        return dashboard
    except Exception as e:
        logger.error(f"Failed to get dashboard for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve dashboard"
        )

@router.get("/preferences", response_model=DataPreferencesResponse)
async def get_data_preferences(
    user_id: str = Depends(get_current_user),
    org_id: str = Depends(get_current_org),
    service = Depends(get_connection_service)
):
    """Get user's data preferences"""
    try:
        preferences = await service.get_user_data_preferences(user_id, org_id)
        return preferences
    except Exception as e:
        logger.error(f"Failed to get preferences for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve preferences"
        )

@router.put("/preferences", response_model=DataPreferencesResponse)
async def update_data_preferences(
    request: DataPreferencesRequest,
    user_id: str = Depends(get_current_user),
    org_id: str = Depends(get_current_org),
    service = Depends(get_connection_service)
):
    """Update user's data preferences"""
    try:
        preferences = await service.update_user_data_preferences(user_id, org_id, request)
        logger.info(f"Updated data preferences for user {user_id}")
        return preferences
    except Exception as e:
        logger.error(f"Failed to update preferences for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update preferences"
        )

# Platform Data Endpoints
@router.post("/data", response_model=PlatformDataResponse)
async def get_platform_data(
    request: PlatformDataRequest,
    user_id: str = Depends(get_current_user),
    org_id: str = Depends(get_current_org),
    service = Depends(get_connection_service)
):
    """Get data from a platform (live or mock based on preferences)"""
    try:
        # Get user preferences
        preferences = await service.get_user_data_preferences(user_id, org_id)
        
        # Determine data source based on preferences and connection status
        if preferences.prefer_live_data and request.connection_id:
            try:
                # Attempt to get live data
                data_response = await service.get_live_platform_data(
                    user_id, org_id, request
                )
                return data_response
            except Exception as e:
                if preferences.fallback_to_mock:
                    logger.warning(f"Live data failed, falling back to mock: {str(e)}")
                    return await service.get_mock_platform_data(request)
                else:
                    raise
        else:
            # Use mock data
            return await service.get_mock_platform_data(request)
            
    except Exception as e:
        logger.error(f"Failed to get platform data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve platform data"
        )

# Health and Monitoring Endpoints
@router.get("/health", response_model=SystemHealthResponse)
async def get_system_health(
    user_id: str = Depends(get_current_user),
    org_id: str = Depends(get_current_org),
    service = Depends(get_connection_service)
):
    """Get system health for platform connections"""
    try:
        health_response = await service.get_system_health(user_id, org_id)
        return health_response
    except Exception as e:
        logger.error(f"Failed to get system health: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system health"
        )

@router.get("/connections/{connection_id}/sync-logs")
async def get_connection_sync_logs(
    connection_id: str,
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user_id: str = Depends(get_current_user),
    org_id: str = Depends(get_current_org),
    service = Depends(get_connection_service)
):
    """Get sync logs for a specific connection"""
    try:
        sync_logs = await service.get_connection_sync_logs(
            user_id, org_id, connection_id, limit, offset
        )
        return {
            "sync_logs": sync_logs,
            "total_count": len(sync_logs),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Failed to get sync logs for connection {connection_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve sync logs"
        )

# Error handling - Note: Exception handlers should be added to the main app, not router
# These can be moved to app.py if needed

# Exception handlers removed - should be added to main app if needed