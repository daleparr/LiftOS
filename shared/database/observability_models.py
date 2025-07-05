"""
Database models for Observability Service
Time-series data storage for metrics, logs, and monitoring
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, JSON, ForeignKey, Numeric, Index
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.hybrid import hybrid_property
import uuid

from .connection import Base


class MetricEntry(Base):
    """Time-series metric storage"""
    __tablename__ = "metric_entries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    value = Column(Numeric(precision=20, scale=6), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    labels = Column(JSONB, nullable=True)
    organization_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    service_name = Column(String(100), nullable=True, index=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Indexes for time-series queries
    __table_args__ = (
        Index('idx_metric_name_timestamp', 'name', 'timestamp'),
        Index('idx_metric_service_timestamp', 'service_name', 'timestamp'),
        Index('idx_metric_org_timestamp', 'organization_id', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<MetricEntry(name={self.name}, value={self.value}, timestamp={self.timestamp})>"


class LogEntry(Base):
    """Structured log storage"""
    __tablename__ = "log_entries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    level = Column(String(20), nullable=False, index=True)
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    service_name = Column(String(100), nullable=False, index=True)
    organization_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    correlation_id = Column(String(255), nullable=True, index=True)
    meta_data = Column(JSONB, nullable=True)
    
    # Full-text search support
    search_vector = Column(Text, nullable=True)  # For PostgreSQL full-text search
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Indexes for log queries
    __table_args__ = (
        Index('idx_log_service_timestamp', 'service_name', 'timestamp'),
        Index('idx_log_level_timestamp', 'level', 'timestamp'),
        Index('idx_log_correlation_id', 'correlation_id'),
        Index('idx_log_org_timestamp', 'organization_id', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<LogEntry(level={self.level}, service={self.service_name}, timestamp={self.timestamp})>"


class HealthCheckEntry(Base):
    """Service health check results"""
    __tablename__ = "health_check_entries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    service_name = Column(String(100), nullable=False, index=True)
    status = Column(String(20), nullable=False, index=True)  # healthy, unhealthy, degraded
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    response_time_ms = Column(Numeric(precision=10, scale=3), nullable=False)
    details = Column(JSONB, nullable=True)
    endpoint_url = Column(String(500), nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Indexes for health monitoring
    __table_args__ = (
        Index('idx_health_service_timestamp', 'service_name', 'timestamp'),
        Index('idx_health_status_timestamp', 'status', 'timestamp'),
    )
    
    @hybrid_property
    def is_healthy(self):
        return self.status == 'healthy'
    
    def __repr__(self):
        return f"<HealthCheckEntry(service={self.service_name}, status={self.status}, timestamp={self.timestamp})>"


class AlertEntry(Base):
    """Alert management and history"""
    __tablename__ = "alert_entries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    alert_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    severity = Column(String(20), nullable=False, index=True)  # critical, warning, info
    status = Column(String(20), nullable=False, index=True)  # firing, resolved, acknowledged
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    acknowledged_at = Column(DateTime(timezone=True), nullable=True)
    organization_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    service_name = Column(String(100), nullable=True, index=True)
    meta_data = Column(JSONB, nullable=True)
    
    # Alert rule information
    rule_name = Column(String(255), nullable=True)
    rule_expression = Column(Text, nullable=True)
    
    # Notification tracking
    notifications_sent = Column(Integer, default=0)
    last_notification_at = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Indexes for alert queries
    __table_args__ = (
        Index('idx_alert_status_timestamp', 'status', 'timestamp'),
        Index('idx_alert_severity_timestamp', 'severity', 'timestamp'),
        Index('idx_alert_service_timestamp', 'service_name', 'timestamp'),
        Index('idx_alert_org_timestamp', 'organization_id', 'timestamp'),
    )
    
    @hybrid_property
    def is_active(self):
        return self.status == 'firing'
    
    @hybrid_property
    def duration_seconds(self):
        if self.resolved_at:
            return (self.resolved_at - self.timestamp).total_seconds()
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds()
    
    def __repr__(self):
        return f"<AlertEntry(name={self.name}, severity={self.severity}, status={self.status})>"


class ServiceRegistry(Base):
    """Service registry for health monitoring"""
    __tablename__ = "service_registry"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    service_name = Column(String(100), unique=True, nullable=False, index=True)
    health_endpoint = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, index=True)
    check_interval_seconds = Column(Integer, default=30)
    timeout_seconds = Column(Integer, default=10)
    
    # Service metadata
    version = Column(String(50), nullable=True)
    environment = Column(String(50), nullable=True)
    tags = Column(JSONB, nullable=True)
    
    # Monitoring configuration
    alert_on_failure = Column(Boolean, default=True)
    failure_threshold = Column(Integer, default=3)  # Number of consecutive failures before alert
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    last_check_at = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<ServiceRegistry(service={self.service_name}, active={self.is_active})>"


class MetricAggregation(Base):
    """Pre-computed metric aggregations for performance"""
    __tablename__ = "metric_aggregations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_name = Column(String(255), nullable=False, index=True)
    aggregation_type = Column(String(20), nullable=False)  # avg, sum, min, max, count
    time_window = Column(String(20), nullable=False)  # 1m, 5m, 1h, 1d
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    value = Column(Numeric(precision=20, scale=6), nullable=False)
    sample_count = Column(Integer, nullable=False)
    
    # Grouping dimensions
    service_name = Column(String(100), nullable=True, index=True)
    organization_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    labels_hash = Column(String(64), nullable=True, index=True)  # Hash of labels for grouping
    labels = Column(JSONB, nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Indexes for aggregation queries
    __table_args__ = (
        Index('idx_agg_metric_window_timestamp', 'metric_name', 'time_window', 'timestamp'),
        Index('idx_agg_service_window_timestamp', 'service_name', 'time_window', 'timestamp'),
        Index('idx_agg_labels_hash_timestamp', 'labels_hash', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<MetricAggregation(metric={self.metric_name}, type={self.aggregation_type}, window={self.time_window})>"


class SystemSnapshot(Base):
    """Periodic system health snapshots"""
    __tablename__ = "system_snapshots"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # System health metrics
    total_services = Column(Integer, nullable=False)
    healthy_services = Column(Integer, nullable=False)
    unhealthy_services = Column(Integer, nullable=False)
    degraded_services = Column(Integer, nullable=False)
    
    # Alert metrics
    total_alerts = Column(Integer, nullable=False)
    critical_alerts = Column(Integer, nullable=False)
    warning_alerts = Column(Integer, nullable=False)
    
    # Performance metrics
    avg_response_time_ms = Column(Numeric(precision=10, scale=3), nullable=True)
    p95_response_time_ms = Column(Numeric(precision=10, scale=3), nullable=True)
    p99_response_time_ms = Column(Numeric(precision=10, scale=3), nullable=True)
    
    # Resource utilization (if available)
    cpu_usage_percent = Column(Numeric(precision=5, scale=2), nullable=True)
    memory_usage_percent = Column(Numeric(precision=5, scale=2), nullable=True)
    disk_usage_percent = Column(Numeric(precision=5, scale=2), nullable=True)
    
    # Additional metrics
    total_requests = Column(Integer, nullable=True)
    error_rate_percent = Column(Numeric(precision=5, scale=2), nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    def __repr__(self):
        return f"<SystemSnapshot(timestamp={self.timestamp}, healthy={self.healthy_services}/{self.total_services})>"