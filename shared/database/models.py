"""
Database models for Lift OS Core
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, JSON, ForeignKey, Numeric
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from .connection import Base

# Import observability models to ensure they're registered
from .observability_models import (
    MetricEntry, LogEntry, HealthCheckEntry, AlertEntry,
    ServiceRegistry, MetricAggregation, SystemSnapshot
)

# Import security models to ensure they're registered
from .security_models import (
    EncryptedAPIKey, SecurityAuditLog, EnhancedUserSession,
    APIKeyUsageAnalytics, SecurityConfiguration, RevokedToken
)

class User(Base):
    """User model for authentication and authorization"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    is_superuser = Column(Boolean, default=False)
    
    # OAuth fields
    google_id = Column(String(255), unique=True, nullable=True)
    github_id = Column(String(255), unique=True, nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    billing_accounts = relationship("BillingAccount", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, username={self.username})>"

class Session(Base):
    """User session model for JWT token management"""
    __tablename__ = "sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    token_jti = Column(String(255), unique=True, nullable=False, index=True)  # JWT ID
    device_info = Column(JSON, nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    expires_at = Column(DateTime(timezone=True), nullable=False)
    last_used = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    
    def __repr__(self):
        return f"<Session(id={self.id}, user_id={self.user_id}, active={self.is_active})>"

class Module(Base):
    """Module registry model"""
    __tablename__ = "modules"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), unique=True, nullable=False, index=True)
    display_name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    version = Column(String(50), nullable=False)
    
    # Module configuration
    config = Column(JSON, nullable=True)
    endpoints = Column(JSON, nullable=True)  # Available endpoints
    dependencies = Column(JSON, nullable=True)  # Module dependencies
    
    # Status and health
    status = Column(String(20), default="inactive")  # inactive, active, error, maintenance
    health_check_url = Column(String(255), nullable=True)
    last_health_check = Column(DateTime(timezone=True), nullable=True)
    
    # Deployment info
    service_url = Column(String(255), nullable=False)
    port = Column(Integer, nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    registered_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    
    def __repr__(self):
        return f"<Module(id={self.id}, name={self.name}, version={self.version}, status={self.status})>"

class BillingAccount(Base):
    """Billing account model"""
    __tablename__ = "billing_accounts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Stripe integration
    stripe_customer_id = Column(String(255), unique=True, nullable=True)
    stripe_subscription_id = Column(String(255), unique=True, nullable=True)
    
    # Account details
    plan_type = Column(String(50), default="free")  # free, basic, pro, enterprise
    billing_email = Column(String(255), nullable=True)
    
    # Usage tracking
    current_usage = Column(JSON, nullable=True)  # Current period usage
    usage_limits = Column(JSON, nullable=True)   # Plan limits
    
    # Status
    status = Column(String(20), default="active")  # active, suspended, cancelled
    trial_ends_at = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    user = relationship("User", back_populates="billing_accounts")
    
    def __repr__(self):
        return f"<BillingAccount(id={self.id}, user_id={self.user_id}, plan={self.plan_type})>"

class ObservabilityEvent(Base):
    """Observability and monitoring events"""
    __tablename__ = "observability_events"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Event classification
    event_type = Column(String(50), nullable=False, index=True)  # request, error, metric, log
    service_name = Column(String(100), nullable=False, index=True)
    endpoint = Column(String(255), nullable=True)
    
    # Event data
    message = Column(Text, nullable=True)
    level = Column(String(20), default="info")  # debug, info, warning, error, critical
    data = Column(JSON, nullable=True)
    
    # Request tracking
    request_id = Column(String(255), nullable=True, index=True)
    user_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    session_id = Column(UUID(as_uuid=True), nullable=True)
    
    # Performance metrics
    duration_ms = Column(Numeric(10, 3), nullable=True)
    status_code = Column(Integer, nullable=True)
    
    # Metadata
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    environment = Column(String(20), default="development")
    
    def __repr__(self):
        return f"<ObservabilityEvent(id={self.id}, type={self.event_type}, service={self.service_name})>"

# Additional utility models for caching and temporary data
class CacheEntry(Base):
    """Generic cache entry model for Redis fallback"""
    __tablename__ = "cache_entries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key = Column(String(255), unique=True, nullable=False, index=True)
    value = Column(JSON, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    def __repr__(self):
        return f"<CacheEntry(key={self.key}, expires_at={self.expires_at})>"