"""
Platform Connection Service
Manages user platform connections, OAuth flows, and data synchronization
"""

import asyncio
import secrets
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_, func
from sqlalchemy.orm import selectinload
import logging
import httpx

from ..database.connection import get_session
from ..database.user_platform_models import (
    UserPlatformConnection, DataSyncLog, PlatformOAuthState, 
    UserDataPreferences, SUPPORTED_PLATFORMS, get_platform_config
)
from ..database.security_models import EncryptedAPIKey
from ..models.platform_connections import (
    ConnectionStatus, SyncType, SyncStatus, AuthType,
    CreateConnectionRequest, UpdateConnectionRequest, ConnectionResponse,
    OAuthInitiateRequest, OAuthInitiateResponse, OAuthCallbackRequest,
    SyncRequest, SyncResponse, ConnectionTestResponse,
    DataPreferencesRequest, DataPreferencesResponse,
    PlatformCredentials, ConnectionDashboard
)
from ..security.api_key_vault import get_api_key_vault
from ..security.audit_logger import SecurityAuditLogger, SecurityEventType
from ..utils.logging import setup_logging

logger = setup_logging("platform_connection_service")

class PlatformConnectionService:
    """Service for managing user platform connections"""
    
    def __init__(self):
        self.api_key_vault = get_api_key_vault()
        self.audit_logger = SecurityAuditLogger()
        self.oauth_configs = self._load_oauth_configs()
    
    def _load_oauth_configs(self) -> Dict[str, Dict[str, str]]:
        """Load OAuth configurations for platforms"""
        import os
        return {
            'meta_business': {
                'client_id': os.getenv('META_CLIENT_ID'),
                'client_secret': os.getenv('META_CLIENT_SECRET'),
                'auth_url': 'https://www.facebook.com/v18.0/dialog/oauth',
                'token_url': 'https://graph.facebook.com/v18.0/oauth/access_token'
            },
            'google_ads': {
                'client_id': os.getenv('GOOGLE_CLIENT_ID'),
                'client_secret': os.getenv('GOOGLE_CLIENT_SECRET'),
                'auth_url': 'https://accounts.google.com/o/oauth2/v2/auth',
                'token_url': 'https://oauth2.googleapis.com/token'
            },
            'shopify': {
                'client_id': os.getenv('SHOPIFY_CLIENT_ID'),
                'client_secret': os.getenv('SHOPIFY_CLIENT_SECRET'),
                'auth_url': 'https://{shop}.myshopify.com/admin/oauth/authorize',
                'token_url': 'https://{shop}.myshopify.com/admin/oauth/access_token'
            },
            'hubspot': {
                'client_id': os.getenv('HUBSPOT_CLIENT_ID'),
                'client_secret': os.getenv('HUBSPOT_CLIENT_SECRET'),
                'auth_url': 'https://app.hubspot.com/oauth/authorize',
                'token_url': 'https://api.hubapi.com/oauth/v1/token'
            }
        }
    
    async def get_supported_platforms(self) -> List[Dict[str, Any]]:
        """Get list of supported platforms"""
        from ..database.user_platform_models import get_supported_platform_list
        return get_supported_platform_list()
    
    async def create_connection(
        self, 
        user_id: str, 
        org_id: str, 
        request: CreateConnectionRequest
    ) -> ConnectionResponse:
        """Create a new platform connection"""
        try:
            async with get_session() as session:
                # Check if connection already exists
                existing = await session.execute(
                    select(UserPlatformConnection).where(
                        and_(
                            UserPlatformConnection.user_id == user_id,
                            UserPlatformConnection.org_id == org_id,
                            UserPlatformConnection.platform == request.platform
                        )
                    )
                )
                if existing.scalar_one_or_none():
                    raise ValueError(f"Connection to {request.platform} already exists")
                
                # Store credentials in vault
                credential_id = await self.api_key_vault.store_api_key(
                    provider=request.platform,
                    credentials=request.credentials.credentials,
                    user_id=user_id,
                    org_id=org_id,
                    expires_at=request.credentials.expires_at
                )
                
                # Create connection record
                platform_config = get_platform_config(request.platform)
                connection = UserPlatformConnection(
                    user_id=user_id,
                    org_id=org_id,
                    platform=request.platform,
                    platform_display_name=platform_config.get('display_name', request.platform),
                    credential_id=credential_id,
                    connection_status=ConnectionStatus.PENDING,
                    connection_config=request.connection_config,
                    sync_frequency=timedelta(hours=request.sync_frequency_hours),
                    auto_sync_enabled=request.auto_sync_enabled
                )
                
                session.add(connection)
                await session.commit()
                await session.refresh(connection)
                
                # Test connection
                test_result = await self._test_connection_internal(connection, session)
                if test_result.success:
                    connection.connection_status = ConnectionStatus.ACTIVE
                    connection.last_test_at = datetime.utcnow()
                else:
                    connection.connection_status = ConnectionStatus.ERROR
                    connection.last_error_message = test_result.message
                
                await session.commit()
                
                # Log security event
                await self.audit_logger.log_api_key_access(
                    user_id=user_id,
                    org_id=org_id,
                    provider=request.platform,
                    action="connection_created",
                    success=test_result.success
                )
                
                return self._connection_to_response(connection)
                
        except Exception as e:
            logger.error(f"Failed to create connection for {user_id}: {str(e)}")
            raise
    
    async def get_user_connections(
        self, 
        user_id: str, 
        org_id: str,
        platform: Optional[str] = None
    ) -> List[ConnectionResponse]:
        """Get user's platform connections"""
        try:
            async with get_session() as session:
                query = select(UserPlatformConnection).where(
                    and_(
                        UserPlatformConnection.user_id == user_id,
                        UserPlatformConnection.org_id == org_id
                    )
                )
                
                if platform:
                    query = query.where(UserPlatformConnection.platform == platform)
                
                result = await session.execute(query)
                connections = result.scalars().all()
                
                return [self._connection_to_response(conn) for conn in connections]
                
        except Exception as e:
            logger.error(f"Failed to get connections for {user_id}: {str(e)}")
            raise
    
    async def update_connection(
        self,
        user_id: str,
        org_id: str,
        connection_id: str,
        request: UpdateConnectionRequest
    ) -> ConnectionResponse:
        """Update an existing platform connection"""
        try:
            async with get_session() as session:
                # Get connection
                connection = await self._get_user_connection(
                    session, user_id, org_id, connection_id
                )
                
                # Update fields
                if request.connection_config is not None:
                    connection.connection_config = request.connection_config
                
                if request.auto_sync_enabled is not None:
                    connection.auto_sync_enabled = request.auto_sync_enabled
                
                if request.sync_frequency_hours is not None:
                    connection.sync_frequency = timedelta(hours=request.sync_frequency_hours)
                
                # Update credentials if provided
                if request.credentials:
                    await self.api_key_vault.update_api_key(
                        credential_id=connection.credential_id,
                        credentials=request.credentials.credentials,
                        user_id=user_id,
                        org_id=org_id
                    )
                    
                    # Test updated connection
                    test_result = await self._test_connection_internal(connection, session)
                    if test_result.success:
                        connection.connection_status = ConnectionStatus.ACTIVE
                        connection.last_test_at = datetime.utcnow()
                        connection.reset_error_count()
                    else:
                        connection.connection_status = ConnectionStatus.ERROR
                        connection.increment_error_count(test_result.message)
                
                await session.commit()
                
                return self._connection_to_response(connection)
                
        except Exception as e:
            logger.error(f"Failed to update connection {connection_id}: {str(e)}")
            raise
    
    async def delete_connection(
        self,
        user_id: str,
        org_id: str,
        connection_id: str
    ) -> bool:
        """Delete a platform connection"""
        try:
            async with get_session() as session:
                connection = await self._get_user_connection(
                    session, user_id, org_id, connection_id
                )
                
                # Revoke API key
                await self.api_key_vault.revoke_api_key(
                    credential_id=connection.credential_id,
                    user_id=user_id,
                    org_id=org_id
                )
                
                # Delete connection (cascade will handle sync logs)
                await session.delete(connection)
                await session.commit()
                
                # Log security event
                await self.audit_logger.log_api_key_access(
                    user_id=user_id,
                    org_id=org_id,
                    provider=connection.platform,
                    action="connection_deleted",
                    success=True
                )
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete connection {connection_id}: {str(e)}")
            raise
    
    async def test_connection(
        self,
        user_id: str,
        org_id: str,
        connection_id: str
    ) -> ConnectionTestResponse:
        """Test a platform connection"""
        try:
            async with get_session() as session:
                connection = await self._get_user_connection(
                    session, user_id, org_id, connection_id
                )
                
                test_result = await self._test_connection_internal(connection, session)
                
                # Update connection status based on test
                if test_result.success:
                    connection.connection_status = ConnectionStatus.ACTIVE
                    connection.last_test_at = datetime.utcnow()
                    connection.reset_error_count()
                else:
                    connection.increment_error_count(test_result.message)
                
                await session.commit()
                
                return test_result
                
        except Exception as e:
            logger.error(f"Failed to test connection {connection_id}: {str(e)}")
            raise
    
    async def initiate_oauth_flow(
        self,
        user_id: str,
        org_id: str,
        request: OAuthInitiateRequest
    ) -> OAuthInitiateResponse:
        """Initiate OAuth flow for a platform"""
        try:
            platform_config = get_platform_config(request.platform)
            if not platform_config.get('oauth_enabled'):
                raise ValueError(f"OAuth not supported for {request.platform}")
            
            oauth_config = self.oauth_configs.get(request.platform)
            if not oauth_config:
                raise ValueError(f"OAuth configuration missing for {request.platform}")
            
            # Create OAuth state
            async with get_session() as session:
                oauth_state = PlatformOAuthState.create_state(
                    user_id=user_id,
                    org_id=org_id,
                    platform=request.platform,
                    redirect_uri=request.redirect_uri,
                    scopes=request.scopes or platform_config.get('required_scopes', [])
                )
                
                session.add(oauth_state)
                await session.commit()
                
                # Build authorization URL
                scopes = ' '.join(oauth_state.scopes)
                auth_params = {
                    'client_id': oauth_config['client_id'],
                    'redirect_uri': oauth_state.redirect_uri,
                    'scope': scopes,
                    'state': oauth_state.state_token,
                    'response_type': 'code'
                }
                
                # Platform-specific parameters
                if request.platform == 'meta_business':
                    auth_params['display'] = 'popup'
                elif request.platform == 'google_ads':
                    auth_params['access_type'] = 'offline'
                    auth_params['prompt'] = 'consent'
                
                auth_url = oauth_config['auth_url']
                query_string = '&'.join([f"{k}={v}" for k, v in auth_params.items()])
                authorization_url = f"{auth_url}?{query_string}"
                
                return OAuthInitiateResponse(
                    authorization_url=authorization_url,
                    state_token=oauth_state.state_token,
                    expires_in=600
                )
                
        except Exception as e:
            logger.error(f"Failed to initiate OAuth for {request.platform}: {str(e)}")
            raise
    
    async def handle_oauth_callback(
        self,
        request: OAuthCallbackRequest
    ) -> Dict[str, Any]:
        """Handle OAuth callback and create connection"""
        try:
            if request.error:
                raise ValueError(f"OAuth error: {request.error_description or request.error}")
            
            async with get_session() as session:
                # Get OAuth state
                oauth_state_result = await session.execute(
                    select(PlatformOAuthState).where(
                        PlatformOAuthState.state_token == request.state
                    )
                )
                oauth_state = oauth_state_result.scalar_one_or_none()
                
                if not oauth_state or oauth_state.is_expired:
                    raise ValueError("Invalid or expired OAuth state")
                
                # Exchange code for tokens
                tokens = await self._exchange_oauth_code(
                    oauth_state.platform, request.code, oauth_state
                )
                
                # Create platform credentials
                credentials = PlatformCredentials(
                    platform=oauth_state.platform,
                    auth_type=AuthType.OAUTH2,
                    credentials=tokens,
                    scopes=oauth_state.scopes,
                    expires_at=tokens.get('expires_at')
                )
                
                # Create connection
                connection_request = CreateConnectionRequest(
                    platform=oauth_state.platform,
                    credentials=credentials
                )
                
                connection = await self.create_connection(
                    user_id=oauth_state.user_id,
                    org_id=oauth_state.org_id,
                    request=connection_request
                )
                
                # Clean up OAuth state
                await session.delete(oauth_state)
                await session.commit()
                
                return {
                    'success': True,
                    'connection_id': connection.id,
                    'platform': connection.platform,
                    'status': connection.connection_status
                }
                
        except Exception as e:
            logger.error(f"Failed to handle OAuth callback: {str(e)}")
            return {
                'success': False,
                'error_message': str(e)
            }
    
    async def sync_platform_data(
        self,
        user_id: str,
        org_id: str,
        request: SyncRequest
    ) -> SyncResponse:
        """Initiate data sync for a platform connection"""
        try:
            async with get_session() as session:
                connection = await self._get_user_connection(
                    session, user_id, org_id, request.connection_id
                )
                
                if connection.connection_status != ConnectionStatus.ACTIVE:
                    raise ValueError("Connection is not active")
                
                # Create sync log
                sync_log = DataSyncLog(
                    connection_id=connection.id,
                    sync_type=request.sync_type,
                    sync_status=SyncStatus.RUNNING,
                    sync_metadata=request.sync_config or {}
                )
                
                session.add(sync_log)
                await session.commit()
                await session.refresh(sync_log)
                
                # Start background sync task
                asyncio.create_task(
                    self._execute_data_sync(sync_log.id, connection, request)
                )
                
                return SyncResponse(
                    sync_id=str(sync_log.id),
                    connection_id=str(connection.id),
                    sync_status=SyncStatus.RUNNING,
                    message="Data sync initiated",
                    estimated_duration_seconds=300
                )
                
        except Exception as e:
            logger.error(f"Failed to initiate sync: {str(e)}")
            raise
    
    async def get_connection_dashboard(
        self,
        user_id: str,
        org_id: str
    ) -> ConnectionDashboard:
        """Get dashboard view of user's connections"""
        try:
            async with get_session() as session:
                # Get connections
                connections_result = await session.execute(
                    select(UserPlatformConnection).where(
                        and_(
                            UserPlatformConnection.user_id == user_id,
                            UserPlatformConnection.org_id == org_id
                        )
                    ).options(selectinload(UserPlatformConnection.sync_logs))
                )
                connections = connections_result.scalars().all()
                
                # Get recent sync logs
                recent_syncs_result = await session.execute(
                    select(DataSyncLog)
                    .join(UserPlatformConnection)
                    .where(
                        and_(
                            UserPlatformConnection.user_id == user_id,
                            UserPlatformConnection.org_id == org_id
                        )
                    )
                    .order_by(DataSyncLog.started_at.desc())
                    .limit(10)
                )
                recent_syncs = recent_syncs_result.scalars().all()
                
                # Get user preferences
                preferences = await self.get_user_data_preferences(user_id, org_id)
                
                # Calculate metrics
                total_connections = len(connections)
                active_connections = sum(1 for c in connections if c.connection_status == ConnectionStatus.ACTIVE)
                error_connections = sum(1 for c in connections if c.connection_status == ConnectionStatus.ERROR)
                pending_connections = sum(1 for c in connections if c.connection_status == ConnectionStatus.PENDING)
                
                return ConnectionDashboard(
                    total_connections=total_connections,
                    active_connections=active_connections,
                    error_connections=error_connections,
                    pending_connections=pending_connections,
                    connections=[self._connection_to_summary(c) for c in connections],
                    recent_syncs=[self._sync_log_to_response(s) for s in recent_syncs],
                    data_preferences=preferences
                )
                
        except Exception as e:
            logger.error(f"Failed to get dashboard for {user_id}: {str(e)}")
            raise
    
    # Helper methods
    async def _get_user_connection(
        self,
        session: AsyncSession,
        user_id: str,
        org_id: str,
        connection_id: str
    ) -> UserPlatformConnection:
        """Get user connection with validation"""
        result = await session.execute(
            select(UserPlatformConnection).where(
                and_(
                    UserPlatformConnection.id == connection_id,
                    UserPlatformConnection.user_id == user_id,
                    UserPlatformConnection.org_id == org_id
                )
            )
        )
        connection = result.scalar_one_or_none()
        if not connection:
            raise ValueError("Connection not found")
        return connection
    
    async def _test_connection_internal(
        self,
        connection: UserPlatformConnection,
        session: AsyncSession
    ) -> ConnectionTestResponse:
        """Test platform connection internally"""
        try:
            # Get credentials
            credentials = await self.api_key_vault.get_api_key(
                credential_id=connection.credential_id,
                user_id=connection.user_id,
                org_id=connection.org_id
            )
            
            # Import and test with appropriate connector
            from ..connectors.connector_factory import get_platform_connector
            connector = get_platform_connector(connection.platform)
            
            start_time = datetime.utcnow()
            test_result = await connector.test_connection(credentials)
            end_time = datetime.utcnow()
            
            response_time = int((end_time - start_time).total_seconds() * 1000)
            
            return ConnectionTestResponse(
                success=test_result.get('success', False),
                message=test_result.get('message', 'Connection test completed'),
                test_details=test_result.get('details', {}),
                response_time_ms=response_time,
                api_limits=test_result.get('api_limits')
            )
            
        except Exception as e:
            return ConnectionTestResponse(
                success=False,
                message=f"Connection test failed: {str(e)}",
                test_details={'error': str(e)}
            )
    
    def _connection_to_response(self, connection: UserPlatformConnection) -> ConnectionResponse:
        """Convert database model to response model"""
        return ConnectionResponse(
            id=str(connection.id),
            user_id=str(connection.user_id),
            org_id=str(connection.org_id),
            platform=connection.platform,
            platform_display_name=connection.platform_display_name,
            connection_status=connection.connection_status,
            connection_config=connection.connection_config,
            last_sync_at=connection.last_sync_at,
            last_test_at=connection.last_test_at,
            sync_frequency_hours=int(connection.sync_frequency.total_seconds() / 3600),
            auto_sync_enabled=connection.auto_sync_enabled,
            error_count=connection.error_count,
            last_error_message=connection.last_error_message,
            is_healthy=connection.is_healthy,
            needs_sync=connection.needs_sync,
            created_at=connection.created_at,
            updated_at=connection.updated_at
        )
    
    async def get_user_data_preferences(
        self,
        user_id: str,
        org_id: str
    ) -> DataPreferencesResponse:
        """Get user data preferences"""
        try:
            async with get_session() as session:
                result = await session.execute(
                    select(UserDataPreferences).where(
                        and_(
                            UserDataPreferences.user_id == user_id,
                            UserDataPreferences.org_id == org_id
                        )
                    )
                )
                preferences = result.scalar_one_or_none()
                
                if not preferences:
                    # Create default preferences
                    preferences = UserDataPreferences(
                        user_id=user_id,
                        org_id=org_id
                    )
                    session.add(preferences)
                    await session.commit()
                    await session.refresh(preferences)
                
                return DataPreferencesResponse(
                    id=str(preferences.id),
                    user_id=str(preferences.user_id),
                    org_id=str(preferences.org_id),
                    prefer_live_data=preferences.prefer_live_data,
                    fallback_to_mock=preferences.fallback_to_mock,
                    data_retention_days=preferences.data_retention_days,
                    sync_preferences=preferences.sync_preferences,
                    effective_data_mode=preferences.effective_data_mode,
                    created_at=preferences.created_at,
                    updated_at=preferences.updated_at
                )
                
        except Exception as e:
            logger.error(f"Failed to get preferences for {user_id}: {str(e)}")
            raise

# Global service instance
_platform_connection_service = None

def get_platform_connection_service() -> PlatformConnectionService:
    """Get global platform connection service instance"""
    global _platform_connection_service
    if _platform_connection_service is None:
        _platform_connection_service = PlatformConnectionService()
    return _platform_connection_service