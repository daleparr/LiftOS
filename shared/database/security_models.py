"""
Security-related database models for LiftOS
Includes encrypted API keys, security audit logs, and session management
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, JSON, ForeignKey, Numeric
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from .connection import Base

class EncryptedAPIKey(Base):
    """Encrypted API key storage with enterprise security features"""
    __tablename__ = "encrypted_api_keys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)  # Can be UUID or string
    org_id = Column(String(255), nullable=False, index=True)   # Organization ID
    provider = Column(String(100), nullable=False, index=True)  # API provider name
    
    # Encryption fields
    encrypted_key = Column(Text, nullable=False)  # Base64 encoded encrypted credentials
    salt = Column(String(255), nullable=False)    # Base64 encoded salt for key derivation
    iv = Column(String(255), nullable=False)      # Base64 encoded initialization vector
    
    # Key management
    status = Column(String(20), default='active', index=True)  # active, rotated, revoked, expired
    usage_count = Column(Integer, default=0)
    last_used = Column(DateTime(timezone=True), nullable=True)
    next_rotation = Column(DateTime(timezone=True), nullable=True)
    
    # Rotation and revocation tracking
    rotated_at = Column(DateTime(timezone=True), nullable=True)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    revocation_reason = Column(String(255), nullable=True)
    
    # Metadata and audit
    provider_metadata = Column(JSON, nullable=True)  # Additional provider-specific metadata
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    def __repr__(self):
        return f"<EncryptedAPIKey(id={self.id}, provider={self.provider}, status={self.status})>"

class SecurityAuditLog(Base):
    """Comprehensive security audit log for SOC 2 compliance"""
    __tablename__ = "security_audit_log"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Event identification
    event_type = Column(String(100), nullable=False, index=True)  # Type of security event
    user_id = Column(String(255), nullable=True, index=True)      # User involved (if applicable)
    org_id = Column(String(255), nullable=True, index=True)       # Organization involved
    
    # Event details
    resource = Column(String(255), nullable=True)     # Resource accessed/modified
    action = Column(String(100), nullable=True)       # Action performed
    success = Column(Boolean, nullable=False)         # Whether the action succeeded
    
    # Request context
    ip_address = Column(String(45), nullable=True)    # IPv6 compatible
    user_agent = Column(Text, nullable=True)          # Browser/client information
    
    # Risk assessment
    risk_level = Column(String(20), default='low')    # low, medium, high, critical
    
    # Additional details
    details = Column(JSON, nullable=True)             # Event-specific details
    
    # Timestamp
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    
    def __repr__(self):
        return f"<SecurityAuditLog(id={self.id}, event_type={self.event_type}, success={self.success})>"

class EnhancedUserSession(Base):
    """Enhanced user session with security features"""
    __tablename__ = "enhanced_user_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    
    # Token management
    access_token_jti = Column(String(255), unique=True, nullable=False, index=True)
    refresh_token_jti = Column(String(255), unique=True, nullable=False, index=True)
    
    # Security context
    device_fingerprint = Column(String(255), nullable=True, index=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    location = Column(JSON, nullable=True)  # Geolocation data if available
    
    # Session management
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    expires_at = Column(DateTime(timezone=True), nullable=False)
    last_activity = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Security flags
    is_suspicious = Column(Boolean, default=False)
    security_alerts = Column(JSON, nullable=True)  # Any security alerts for this session
    
    def __repr__(self):
        return f"<EnhancedUserSession(id={self.id}, user_id={self.user_id}, active={self.is_active})>"

class APIKeyUsageAnalytics(Base):
    """API key usage analytics for monitoring and billing"""
    __tablename__ = "api_key_usage_analytics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key_id = Column(UUID(as_uuid=True), ForeignKey("encrypted_api_keys.id"), nullable=False, index=True)
    
    # Usage metrics
    endpoint = Column(String(255), nullable=True)     # API endpoint called
    requests_count = Column(Integer, default=0)       # Number of requests
    response_time_ms = Column(Numeric(10, 2), nullable=True)  # Average response time
    error_count = Column(Integer, default=0)          # Number of errors
    data_transferred_mb = Column(Numeric(10, 2), default=0)   # Data transferred
    
    # Cost tracking
    estimated_cost = Column(Numeric(10, 4), default=0)  # Estimated API cost
    
    # Time period
    date = Column(DateTime(timezone=True), nullable=False, index=True)
    hour = Column(Integer, nullable=True)  # Hour of day (0-23) for hourly analytics
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    api_key = relationship("EncryptedAPIKey", backref="usage_analytics")
    
    def __repr__(self):
        return f"<APIKeyUsageAnalytics(id={self.id}, key_id={self.key_id}, date={self.date})>"

class SecurityConfiguration(Base):
    """Security configuration settings per organization"""
    __tablename__ = "security_configuration"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(String(255), unique=True, nullable=False, index=True)
    
    # Authentication settings
    require_mfa = Column(Boolean, default=False)
    session_timeout_minutes = Column(Integer, default=60)
    max_concurrent_sessions = Column(Integer, default=3)
    password_policy = Column(JSON, nullable=True)
    
    # API key settings
    api_key_rotation_days = Column(Integer, default=90)
    require_key_approval = Column(Boolean, default=False)
    allowed_providers = Column(JSON, nullable=True)  # List of allowed API providers
    
    # Access control
    ip_whitelist = Column(JSON, nullable=True)        # Allowed IP addresses/ranges
    allowed_countries = Column(JSON, nullable=True)   # Allowed countries for access
    
    # Rate limiting
    rate_limit_requests_per_minute = Column(Integer, default=100)
    rate_limit_burst = Column(Integer, default=200)
    
    # Compliance settings
    audit_retention_days = Column(Integer, default=2555)  # 7 years for SOC 2
    enable_data_encryption = Column(Boolean, default=True)
    compliance_mode = Column(String(50), default='standard')  # standard, soc2, hipaa, gdpr
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    updated_by = Column(String(255), nullable=True)
    
    def __repr__(self):
        return f"<SecurityConfiguration(id={self.id}, org_id={self.org_id}, compliance_mode={self.compliance_mode})>"

class RevokedToken(Base):
    """Revoked JWT tokens for security"""
    __tablename__ = "revoked_tokens"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    token_jti = Column(String(255), unique=True, nullable=False, index=True)  # JWT ID
    user_id = Column(String(255), nullable=False, index=True)
    
    # Revocation details
    revoked_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    reason = Column(String(255), nullable=True)       # Reason for revocation
    revoked_by = Column(String(255), nullable=True)   # Who revoked it
    
    # Original token info
    original_expires_at = Column(DateTime(timezone=True), nullable=True)
    token_type = Column(String(20), default='access')  # access, refresh
    
    def __repr__(self):
        return f"<RevokedToken(id={self.id}, token_jti={self.token_jti}, revoked_at={self.revoked_at})>"