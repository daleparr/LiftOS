"""
Pydantic Models for Platform Connection Management
Handles API request/response models for user platform connections
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field, validator
from enum import Enum

from .base import TimestampMixin

class ConnectionStatus(str, Enum):
    """Platform connection status"""
    PENDING = "pending"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"

class SyncType(str, Enum):
    """Data sync type"""
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    WEBHOOK = "webhook"
    INITIAL = "initial"

class SyncStatus(str, Enum):
    """Data sync status"""
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"

class AuthType(str, Enum):
    """Platform authentication type"""
    OAUTH2 = "oauth2"
    API_KEY = "api_key"
    BASIC_AUTH = "basic_auth"

# Platform Configuration Models
class PlatformConfig(BaseModel):
    """Platform configuration information"""
    id: str
    display_name: str
    oauth_enabled: bool
    auth_type: AuthType
    required_scopes: Optional[List[str]] = []
    icon: Optional[str] = None
    color: Optional[str] = None
    description: Optional[str] = None
    documentation_url: Optional[str] = None

class PlatformCredentials(BaseModel):
    """Platform credentials for API access"""
    platform: str
    auth_type: AuthType
    credentials: Dict[str, str] = Field(..., description="Encrypted credential data")
    scopes: Optional[List[str]] = []
    expires_at: Optional[datetime] = None
    refresh_token: Optional[str] = None

    @validator('credentials')
    def validate_credentials(cls, v, values):
        """Validate required credential fields based on auth type"""
        auth_type = values.get('auth_type')
        
        if auth_type == AuthType.API_KEY:
            if 'api_key' not in v:
                raise ValueError("API key required for api_key auth type")
        elif auth_type == AuthType.OAUTH2:
            required_fields = ['access_token']
            missing = [field for field in required_fields if field not in v]
            if missing:
                raise ValueError(f"Missing required OAuth2 fields: {missing}")
        elif auth_type == AuthType.BASIC_AUTH:
            required_fields = ['username', 'password']
            missing = [field for field in required_fields if field not in v]
            if missing:
                raise ValueError(f"Missing required basic auth fields: {missing}")
        
        return v

# Connection Management Models
class CreateConnectionRequest(BaseModel):
    """Request to create a new platform connection"""
    platform: str = Field(..., description="Platform identifier")
    credentials: PlatformCredentials
    connection_config: Optional[Dict[str, Any]] = {}
    auto_sync_enabled: bool = True
    sync_frequency_hours: int = Field(1, ge=1, le=168, description="Sync frequency in hours")

    @validator('platform')
    def validate_platform(cls, v):
        """Validate platform is supported"""
        from shared.database.user_platform_models import SUPPORTED_PLATFORMS
        if v not in SUPPORTED_PLATFORMS:
            raise ValueError(f"Unsupported platform: {v}")
        return v

class UpdateConnectionRequest(BaseModel):
    """Request to update an existing platform connection"""
    connection_config: Optional[Dict[str, Any]] = None
    auto_sync_enabled: Optional[bool] = None
    sync_frequency_hours: Optional[int] = Field(None, ge=1, le=168)
    credentials: Optional[PlatformCredentials] = None

class ConnectionResponse(TimestampMixin):
    """Platform connection response"""
    id: str
    user_id: str
    org_id: str
    platform: str
    platform_display_name: str
    connection_status: ConnectionStatus
    connection_config: Dict[str, Any] = {}
    last_sync_at: Optional[datetime] = None
    last_test_at: Optional[datetime] = None
    sync_frequency_hours: int
    auto_sync_enabled: bool
    error_count: int = 0
    last_error_message: Optional[str] = None
    is_healthy: bool
    needs_sync: bool

    class Config:
        from_attributes = True

class ConnectionSummary(BaseModel):
    """Summary of platform connection"""
    id: str
    platform: str
    platform_display_name: str
    connection_status: ConnectionStatus
    last_sync_at: Optional[datetime] = None
    is_healthy: bool
    error_count: int = 0

# OAuth Flow Models
class OAuthInitiateRequest(BaseModel):
    """Request to initiate OAuth flow"""
    platform: str
    redirect_uri: Optional[str] = None
    scopes: Optional[List[str]] = None

class OAuthInitiateResponse(BaseModel):
    """Response for OAuth initiation"""
    authorization_url: str
    state_token: str
    expires_in: int = 600  # 10 minutes

class OAuthCallbackRequest(BaseModel):
    """OAuth callback request"""
    state: str
    code: str
    error: Optional[str] = None
    error_description: Optional[str] = None

class OAuthCallbackResponse(BaseModel):
    """OAuth callback response"""
    success: bool
    connection_id: Optional[str] = None
    error_message: Optional[str] = None

# Data Sync Models
class SyncRequest(BaseModel):
    """Request to sync data from platform"""
    connection_id: str
    sync_type: SyncType = SyncType.MANUAL
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None
    force_full_sync: bool = False
    sync_config: Optional[Dict[str, Any]] = {}

class SyncResponse(BaseModel):
    """Response for sync request"""
    sync_id: str
    connection_id: str
    sync_status: SyncStatus
    message: str
    estimated_duration_seconds: Optional[int] = None

class SyncLogResponse(TimestampMixin):
    """Data sync log response"""
    id: str
    connection_id: str
    sync_type: SyncType
    sync_status: SyncStatus
    records_requested: Optional[int] = None
    records_processed: int = 0
    records_failed: int = 0
    data_size_bytes: Optional[int] = None
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    sync_metadata: Dict[str, Any] = {}
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    success_rate: float

    class Config:
        from_attributes = True

# Connection Testing Models
class ConnectionTestRequest(BaseModel):
    """Request to test platform connection"""
    connection_id: Optional[str] = None
    credentials: Optional[PlatformCredentials] = None

class ConnectionTestResponse(BaseModel):
    """Response for connection test"""
    success: bool
    message: str
    test_details: Dict[str, Any] = {}
    response_time_ms: Optional[int] = None
    api_limits: Optional[Dict[str, Any]] = None

# User Preferences Models
class DataPreferencesRequest(BaseModel):
    """Request to update user data preferences"""
    prefer_live_data: Optional[bool] = None
    fallback_to_mock: Optional[bool] = None
    data_retention_days: Optional[int] = Field(None, ge=1, le=365)
    sync_preferences: Optional[Dict[str, Any]] = None

class DataPreferencesResponse(TimestampMixin):
    """User data preferences response"""
    id: str
    user_id: str
    org_id: str
    prefer_live_data: bool
    fallback_to_mock: bool
    data_retention_days: int
    sync_preferences: Dict[str, Any] = {}
    effective_data_mode: str

    class Config:
        from_attributes = True

# Dashboard Models
class ConnectionDashboard(BaseModel):
    """Dashboard view of all user connections"""
    total_connections: int
    active_connections: int
    error_connections: int
    pending_connections: int
    connections: List[ConnectionSummary]
    recent_syncs: List[SyncLogResponse]
    data_preferences: DataPreferencesResponse

class PlatformMetrics(BaseModel):
    """Metrics for a specific platform"""
    platform: str
    total_syncs: int
    successful_syncs: int
    failed_syncs: int
    average_sync_duration: Optional[float] = None
    last_successful_sync: Optional[datetime] = None
    total_records_processed: int
    average_success_rate: float

class SystemHealthResponse(BaseModel):
    """System health for platform connections"""
    total_platforms: int
    healthy_connections: int
    unhealthy_connections: int
    pending_syncs: int
    failed_syncs_last_24h: int
    average_response_time_ms: float
    platform_metrics: List[PlatformMetrics]

# Error Models
class ConnectionError(BaseModel):
    """Connection error details"""
    error_code: str
    error_message: str
    error_details: Optional[Dict[str, Any]] = None
    suggested_action: Optional[str] = None
    documentation_url: Optional[str] = None

class ValidationError(BaseModel):
    """Validation error for requests"""
    field: str
    message: str
    invalid_value: Any

# Bulk Operations Models
class BulkSyncRequest(BaseModel):
    """Request to sync multiple connections"""
    connection_ids: List[str] = Field(..., min_items=1, max_items=10)
    sync_type: SyncType = SyncType.MANUAL
    sync_config: Optional[Dict[str, Any]] = {}

class BulkSyncResponse(BaseModel):
    """Response for bulk sync request"""
    total_requested: int
    syncs_initiated: int
    syncs_failed: int
    sync_responses: List[SyncResponse]
    errors: List[ConnectionError] = []

# Platform Data Models
class PlatformDataRequest(BaseModel):
    """Request for platform data"""
    platform: str
    connection_id: Optional[str] = None
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None
    data_types: Optional[List[str]] = []
    use_cache: bool = True
    fallback_to_mock: bool = True

class PlatformDataResponse(BaseModel):
    """Response with platform data"""
    platform: str
    data_source: str  # 'live', 'mock', 'cached'
    data: Dict[str, Any]
    metadata: Dict[str, Any] = {}
    cache_info: Optional[Dict[str, Any]] = None
    sync_info: Optional[Dict[str, Any]] = None

# Data Validation Models
class ValidationResultResponse(BaseModel):
    """Response model for validation result"""
    rule_id: str
    status: str  # passed, warning, failed, skipped
    score: float = Field(..., ge=0, le=100)
    message: str
    details: Dict[str, Any] = {}
    recommendations: List[str] = []

class DataQualityReportResponse(BaseModel):
    """Response model for data quality report"""
    connection_id: str
    platform: str
    overall_score: float = Field(..., ge=0, le=100)
    quality_level: str  # excellent, good, fair, poor, critical
    validation_results: List[ValidationResultResponse]
    data_freshness: Dict[str, Any] = {}
    completeness_metrics: Dict[str, Any] = {}
    consistency_metrics: Dict[str, Any] = {}
    reliability_metrics: Dict[str, Any] = {}
    recommendations: List[str] = []
    generated_at: datetime

class ValidationRuleResponse(BaseModel):
    """Response model for validation rule"""
    rule_id: str
    name: str
    description: str
    severity: str  # critical, high, medium, low
    platform_specific: bool = False
    applicable_platforms: Optional[List[str]] = None

class QualitySummaryResponse(BaseModel):
    """Response model for quality summary"""
    total_connections: int
    average_score: float
    quality_distribution: Dict[str, int] = {}
    top_issues: List[Dict[str, Any]] = []
    recommendations: List[str] = []

# Export all models
__all__ = [
    'ConnectionStatus', 'SyncType', 'SyncStatus', 'AuthType',
    'PlatformConfig', 'PlatformCredentials',
    'CreateConnectionRequest', 'UpdateConnectionRequest', 'ConnectionResponse', 'ConnectionSummary',
    'OAuthInitiateRequest', 'OAuthInitiateResponse', 'OAuthCallbackRequest', 'OAuthCallbackResponse',
    'SyncRequest', 'SyncResponse', 'SyncLogResponse',
    'ConnectionTestRequest', 'ConnectionTestResponse',
    'DataPreferencesRequest', 'DataPreferencesResponse',
    'ConnectionDashboard', 'PlatformMetrics', 'SystemHealthResponse',
    'ConnectionError', 'ValidationError',
    'BulkSyncRequest', 'BulkSyncResponse',
    'PlatformDataRequest', 'PlatformDataResponse',
    'ValidationResultResponse', 'DataQualityReportResponse', 'ValidationRuleResponse', 'QualitySummaryResponse'
]
# Rollout and Monitoring Models

class RolloutStrategy(str, Enum):
    """Rollout strategy types"""
    PERCENTAGE = "percentage"
    USER_BASED = "user_based"
    PLATFORM_BASED = "platform_based"
    FEATURE_FLAG = "feature_flag"
    A_B_TEST = "a_b_test"

class RolloutPhase(str, Enum):
    """Rollout phase enumeration"""
    PLANNING = "planning"
    TESTING = "testing"
    ROLLOUT = "rollout"
    MONITORING = "monitoring"
    COMPLETED = "completed"

class RolloutConfigRequest(BaseModel):
    """Request model for creating rollout configuration"""
    rollout_id: str = Field(..., description="Unique rollout identifier")
    name: str = Field(..., description="Rollout name")
    description: str = Field(..., description="Rollout description")
    rollout_type: str = Field(..., description="Type of rollout")
    target_percentage: Optional[float] = Field(None, ge=0, le=100, description="Target percentage for percentage rollout")
    target_users: Optional[List[str]] = Field(None, description="Target users for user-based rollout")
    target_platforms: Optional[List[str]] = Field(None, description="Target platforms for platform-based rollout")
    feature_flags: Optional[Dict[str, bool]] = Field(None, description="Feature flags for feature flag rollout")
    start_date: Optional[datetime] = Field(None, description="Rollout start date")
    end_date: Optional[datetime] = Field(None, description="Rollout end date")
    success_criteria: Optional[Dict[str, Any]] = Field(None, description="Success criteria for rollout")
    rollback_criteria: Optional[Dict[str, Any]] = Field(None, description="Rollback criteria for rollout")
    monitoring_config: Optional[Dict[str, Any]] = Field(None, description="Monitoring configuration")

class RolloutStatusResponse(BaseModel):
    """Response model for rollout status"""
    success: bool
    rollout_id: str
    config: Dict[str, Any]
    progress: Dict[str, Any]
    latest_metrics: Optional[Dict[str, Any]] = None
    metrics_history: List[Dict[str, Any]] = []
    recommendations: List[Dict[str, Any]] = []

class MonitoringDashboardResponse(BaseModel):
    """Response model for monitoring dashboard"""
    health_status: Dict[str, Any]
    recent_alerts: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    platform_statistics: Dict[str, Any]
    error_analysis: Dict[str, Any]
    timestamp: str

class AlertRequest(BaseModel):
    """Request model for creating alerts"""
    alert_type: str = Field(..., description="Type of alert")
    severity: str = Field(..., description="Alert severity")
    title: str = Field(..., description="Alert title")
    description: str = Field(..., description="Alert description")
    source: str = Field(..., description="Alert source")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional alert metadata")

class MetricRequest(BaseModel):
    """Request model for recording metrics"""
    name: str = Field(..., description="Metric name")
    value: float = Field(..., description="Metric value")
    metric_type: str = Field(default="gauge", description="Type of metric")
    tags: Optional[Dict[str, str]] = Field(None, description="Metric tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metric metadata")


# Analytics and Optimization Models
class AnalyticsTimeframe(str, Enum):
    """Analytics timeframe options"""
    LAST_7_DAYS = "last_7_days"
    LAST_30_DAYS = "last_30_days"
    LAST_90_DAYS = "last_90_days"
    LAST_6_MONTHS = "last_6_months"
    LAST_YEAR = "last_year"
    CUSTOM = "custom"

class OptimizationType(str, Enum):
    """Optimization type options"""
    COST = "cost"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    EFFICIENCY = "efficiency"
    RELIABILITY = "reliability"

class PerformanceAnalytics(BaseModel):
    """Performance analytics response"""
    platform: str
    timeframe: AnalyticsTimeframe
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    success_rate: float
    error_rate: float
    throughput: float
    peak_usage: Dict[str, Any] = {}
    trends: Dict[str, Any] = {}
    generated_at: datetime

class CostAnalytics(BaseModel):
    """Cost analytics response"""
    platform: str
    timeframe: AnalyticsTimeframe
    total_cost: float
    cost_per_request: float
    cost_breakdown: Dict[str, float] = {}
    cost_trends: Dict[str, Any] = {}
    cost_optimization_potential: float
    recommendations: List[str] = []
    generated_at: datetime

class QualityTrends(BaseModel):
    """Data quality trends"""
    platform: str
    timeframe: AnalyticsTimeframe
    quality_score_trend: List[Dict[str, Any]] = []
    completeness_trend: List[Dict[str, Any]] = []
    accuracy_trend: List[Dict[str, Any]] = []
    consistency_trend: List[Dict[str, Any]] = []
    timeliness_trend: List[Dict[str, Any]] = []
    overall_trend: str  # improving, declining, stable
    generated_at: datetime

class PredictiveAnalytics(BaseModel):
    """Predictive analytics response"""
    platform: str
    prediction_type: str
    confidence_level: float
    predictions: Dict[str, Any] = {}
    forecast_period: str
    model_accuracy: float
    risk_factors: List[str] = []
    opportunities: List[str] = []
    generated_at: datetime

class OptimizationRecommendation(BaseModel):
    """Optimization recommendation"""
    id: str
    platform: str
    optimization_type: OptimizationType
    title: str
    description: str
    impact_level: str  # high, medium, low
    effort_level: str  # high, medium, low
    estimated_savings: Optional[float] = None
    estimated_improvement: Optional[float] = None
    implementation_steps: List[str] = []
    prerequisites: List[str] = []
    risks: List[str] = []
    priority_score: float
    created_at: datetime

class PlatformOptimization(BaseModel):
    """Platform optimization summary"""
    platform: str
    current_performance: Dict[str, Any] = {}
    optimization_potential: Dict[str, Any] = {}
    recommendations: List[OptimizationRecommendation] = []
    quick_wins: List[Dict[str, Any]] = []
    long_term_improvements: List[Dict[str, Any]] = []
    roi_analysis: Dict[str, Any] = {}
    implementation_roadmap: List[Dict[str, Any]] = []
    generated_at: datetime
