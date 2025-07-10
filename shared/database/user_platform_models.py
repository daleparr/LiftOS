"""
Database Models for User Platform Connections
Handles user-specific marketing platform API connections and data sync tracking
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from sqlalchemy import Column, String, Boolean, Integer, DateTime, Text, JSON, ForeignKey, BigInteger, Interval, CheckConstraint, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from .models import Base

class UserPlatformConnection(Base):
    """User's connection to a marketing platform"""
    __tablename__ = "user_platform_connections"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    org_id = Column(UUID(as_uuid=True), nullable=False)
    platform = Column(String(50), nullable=False)
    platform_display_name = Column(String(100), nullable=False)
    credential_id = Column(UUID(as_uuid=True), ForeignKey('encrypted_api_keys.id', ondelete='CASCADE'))
    connection_status = Column(String(20), default='pending', nullable=False)
    connection_config = Column(JSON, default=dict)
    last_sync_at = Column(DateTime(timezone=True))
    last_test_at = Column(DateTime(timezone=True))
    sync_frequency = Column(Interval, default=timedelta(hours=1))
    auto_sync_enabled = Column(Boolean, default=True)
    error_count = Column(Integer, default=0)
    last_error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    sync_logs = relationship("DataSyncLog", back_populates="connection", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('user_id', 'org_id', 'platform', name='uq_user_org_platform'),
        CheckConstraint('connection_status IN (\'pending\', \'active\', \'error\', \'disabled\')', name='ck_connection_status'),
        CheckConstraint('error_count >= 0', name='ck_error_count_positive'),
    )
    
    def __repr__(self):
        return f"<UserPlatformConnection(user_id={self.user_id}, platform={self.platform}, status={self.connection_status})>"
    
    @property
    def is_healthy(self) -> bool:
        """Check if connection is healthy"""
        return (
            self.connection_status == 'active' and
            self.error_count < 5 and
            (self.last_test_at is None or 
             (datetime.utcnow() - self.last_test_at.replace(tzinfo=None)) < timedelta(hours=24))
        )
    
    @property
    def needs_sync(self) -> bool:
        """Check if connection needs data sync"""
        if not self.auto_sync_enabled or self.connection_status != 'active':
            return False
        
        if self.last_sync_at is None:
            return True
        
        time_since_sync = datetime.utcnow() - self.last_sync_at.replace(tzinfo=None)
        return time_since_sync >= self.sync_frequency
    
    def increment_error_count(self, error_message: str = None):
        """Increment error count and update status if needed"""
        self.error_count += 1
        if error_message:
            self.last_error_message = error_message
        
        # Disable connection if too many errors
        if self.error_count >= 10:
            self.connection_status = 'error'
            self.auto_sync_enabled = False
    
    def reset_error_count(self):
        """Reset error count on successful operation"""
        self.error_count = 0
        self.last_error_message = None
        if self.connection_status == 'error':
            self.connection_status = 'active'


class DataSyncLog(Base):
    """Log of data synchronization attempts"""
    __tablename__ = "data_sync_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    connection_id = Column(UUID(as_uuid=True), ForeignKey('user_platform_connections.id', ondelete='CASCADE'), nullable=False)
    sync_type = Column(String(20), nullable=False)
    sync_status = Column(String(20), nullable=False)
    records_requested = Column(Integer)
    records_processed = Column(Integer, default=0)
    records_failed = Column(Integer, default=0)
    data_size_bytes = Column(BigInteger)
    error_message = Column(Text)
    error_details = Column(JSON)
    sync_metadata = Column(JSON, default=dict)
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    duration_seconds = Column(Integer)
    
    # Relationships
    connection = relationship("UserPlatformConnection", back_populates="sync_logs")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('sync_type IN (\'manual\', \'scheduled\', \'webhook\', \'initial\')', name='ck_sync_type'),
        CheckConstraint('sync_status IN (\'running\', \'success\', \'failed\', \'partial\', \'cancelled\')', name='ck_sync_status'),
        CheckConstraint('records_processed >= 0', name='ck_records_processed_positive'),
        CheckConstraint('records_failed >= 0', name='ck_records_failed_positive'),
        CheckConstraint('data_size_bytes >= 0', name='ck_data_size_positive'),
    )
    
    def __repr__(self):
        return f"<DataSyncLog(connection_id={self.connection_id}, status={self.sync_status}, records={self.records_processed})>"
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate for this sync"""
        if not self.records_requested or self.records_requested == 0:
            return 1.0 if self.sync_status == 'success' else 0.0
        
        return (self.records_processed - self.records_failed) / self.records_requested
    
    @property
    def is_completed(self) -> bool:
        """Check if sync is completed"""
        return self.sync_status in ['success', 'failed', 'partial', 'cancelled']
    
    def mark_completed(self, status: str, error_message: str = None):
        """Mark sync as completed with given status"""
        self.sync_status = status
        self.completed_at = datetime.utcnow()
        if self.started_at:
            delta = self.completed_at - self.started_at.replace(tzinfo=None)
            self.duration_seconds = int(delta.total_seconds())
        
        if error_message:
            self.error_message = error_message


class PlatformOAuthState(Base):
    """Temporary storage for OAuth flow state tokens"""
    __tablename__ = "platform_oauth_states"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    org_id = Column(UUID(as_uuid=True), nullable=False)
    platform = Column(String(50), nullable=False)
    state_token = Column(String(255), nullable=False, unique=True)
    redirect_uri = Column(Text)
    scopes = Column(ARRAY(String))
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Constraints
    __table_args__ = (
        CheckConstraint('expires_at > created_at', name='ck_oauth_expires_after_created'),
    )
    
    def __repr__(self):
        return f"<PlatformOAuthState(user_id={self.user_id}, platform={self.platform}, expires_at={self.expires_at})>"
    
    @property
    def is_expired(self) -> bool:
        """Check if OAuth state has expired"""
        return datetime.utcnow() > self.expires_at.replace(tzinfo=None)
    
    @classmethod
    def create_state(cls, user_id: str, org_id: str, platform: str, 
                    redirect_uri: str = None, scopes: List[str] = None,
                    expires_in_minutes: int = 10):
        """Create a new OAuth state token"""
        import secrets
        
        state_token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(minutes=expires_in_minutes)
        
        return cls(
            user_id=user_id,
            org_id=org_id,
            platform=platform,
            state_token=state_token,
            redirect_uri=redirect_uri,
            scopes=scopes or [],
            expires_at=expires_at
        )


class UserDataPreferences(Base):
    """User preferences for data handling and synchronization"""
    __tablename__ = "user_data_preferences"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    org_id = Column(UUID(as_uuid=True), nullable=False)
    prefer_live_data = Column(Boolean, default=True)
    fallback_to_mock = Column(Boolean, default=True)
    data_retention_days = Column(Integer, default=90)
    sync_preferences = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('user_id', 'org_id', name='uq_user_data_preferences'),
        CheckConstraint('data_retention_days > 0', name='ck_retention_positive'),
    )
    
    def __repr__(self):
        return f"<UserDataPreferences(user_id={self.user_id}, prefer_live={self.prefer_live_data})>"
    
    @property
    def effective_data_mode(self) -> str:
        """Get effective data mode based on preferences"""
        if self.prefer_live_data:
            return 'live' if not self.fallback_to_mock else 'hybrid'
        return 'mock'
    
    def get_sync_preference(self, key: str, default=None):
        """Get a specific sync preference"""
        return self.sync_preferences.get(key, default)
    
    def set_sync_preference(self, key: str, value):
        """Set a specific sync preference"""
        if self.sync_preferences is None:
            self.sync_preferences = {}
        self.sync_preferences[key] = value


# Platform configuration constants
SUPPORTED_PLATFORMS = {
    'meta_business': {
        'display_name': 'Meta Business (Facebook)',
        'oauth_enabled': True,
        'required_scopes': ['ads_read', 'ads_management'],
        'auth_type': 'oauth2',
        'icon': 'facebook',
        'color': '#1877F2'
    },
    'google_ads': {
        'display_name': 'Google Ads',
        'oauth_enabled': True,
        'required_scopes': ['https://www.googleapis.com/auth/adwords'],
        'auth_type': 'oauth2',
        'icon': 'google',
        'color': '#4285F4'
    },
    'klaviyo': {
        'display_name': 'Klaviyo',
        'oauth_enabled': False,
        'auth_type': 'api_key',
        'icon': 'klaviyo',
        'color': '#FF6900'
    },
    'shopify': {
        'display_name': 'Shopify',
        'oauth_enabled': True,
        'required_scopes': ['read_orders', 'read_products', 'read_customers'],
        'auth_type': 'oauth2',
        'icon': 'shopify',
        'color': '#96BF48'
    },
    'hubspot': {
        'display_name': 'HubSpot',
        'oauth_enabled': True,
        'required_scopes': ['contacts', 'content'],
        'auth_type': 'oauth2',
        'icon': 'hubspot',
        'color': '#FF7A59'
    },
    'stripe': {
        'display_name': 'Stripe',
        'oauth_enabled': False,
        'auth_type': 'api_key',
        'icon': 'stripe',
        'color': '#635BFF'
    }
}

def get_platform_config(platform: str) -> Dict[str, Any]:
    """Get configuration for a specific platform"""
    return SUPPORTED_PLATFORMS.get(platform, {})

def get_supported_platform_list() -> List[Dict[str, Any]]:
    """Get list of all supported platforms with their configurations"""
    return [
        {'id': platform_id, **config}
        for platform_id, config in SUPPORTED_PLATFORMS.items()
    ]


class ConnectionAuditLog(Base):
    """Audit log for platform connection activities"""
    __tablename__ = "connection_audit_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    connection_id = Column(String, nullable=True, index=True)
    action = Column(String, nullable=False, index=True)
    details = Column(Text, nullable=True)  # JSON string for additional details
    timestamp = Column(DateTime, nullable=False, default=func.now(), index=True)
    
    # Add indexes for common query patterns
    __table_args__ = (
        CheckConstraint("action != ''", name="check_action_not_empty"),
    )
    
    def __repr__(self):
        return f"<ConnectionAuditLog(id={self.id}, user_id={self.user_id}, action={self.action})>"