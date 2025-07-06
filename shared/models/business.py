"""
Business observability data models for LiftOS.
Comprehensive business metrics, KPIs, and intelligence tracking.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from enum import Enum
import uuid

class MetricType(str, Enum):
    """Types of business metrics"""
    REVENUE = "revenue"
    COST = "cost"
    CUSTOMER = "customer"
    OPERATIONAL = "operational"
    STRATEGIC = "strategic"
    QUALITY = "quality"

class MetricFrequency(str, Enum):
    """Frequency of metric collection"""
    REAL_TIME = "real_time"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"

class BusinessMetric(BaseModel):
    """Core business metric model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    metric_type: MetricType
    value: float
    unit: str  # e.g., "USD", "percentage", "count"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    frequency: MetricFrequency
    source: str  # Data source identifier
    tags: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Contextual information
    business_unit: Optional[str] = None
    department: Optional[str] = None
    product: Optional[str] = None
    campaign: Optional[str] = None
    
    # Quality indicators
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    data_quality_score: float = Field(ge=0.0, le=1.0, default=1.0)
    
class RevenueMetrics(BaseModel):
    """Revenue-specific metrics"""
    total_revenue: float
    revenue_growth_rate: float
    revenue_per_user: float
    revenue_per_decision: float
    recurring_revenue: float
    new_revenue: float
    lost_revenue: float
    
    # Attribution
    liftos_attributed_revenue: float
    attribution_confidence: float = Field(ge=0.0, le=1.0)
    
    # Breakdown
    revenue_by_product: Dict[str, float] = Field(default_factory=dict)
    revenue_by_channel: Dict[str, float] = Field(default_factory=dict)
    revenue_by_segment: Dict[str, float] = Field(default_factory=dict)

class CustomerMetrics(BaseModel):
    """Customer-specific metrics"""
    total_customers: int
    new_customers: int
    churned_customers: int
    retention_rate: float = Field(ge=0.0, le=1.0)
    
    # Value metrics
    customer_lifetime_value: float
    customer_acquisition_cost: float
    average_order_value: float
    
    # Satisfaction
    net_promoter_score: float = Field(ge=-100.0, le=100.0)
    customer_satisfaction_score: float = Field(ge=0.0, le=10.0)
    support_ticket_volume: int
    
    # Segmentation
    customers_by_segment: Dict[str, int] = Field(default_factory=dict)
    value_by_segment: Dict[str, float] = Field(default_factory=dict)

class OperationalMetrics(BaseModel):
    """Operational efficiency metrics"""
    decision_volume: int
    decision_accuracy: float = Field(ge=0.0, le=1.0)
    decision_speed: float  # Average time in seconds
    automation_rate: float = Field(ge=0.0, le=1.0)
    
    # Efficiency
    cost_per_decision: float
    time_savings: float  # Hours saved
    error_reduction: float = Field(ge=0.0, le=1.0)
    process_efficiency: float = Field(ge=0.0, le=1.0)
    
    # Quality
    data_quality_score: float = Field(ge=0.0, le=1.0)
    system_uptime: float = Field(ge=0.0, le=1.0)
    user_adoption_rate: float = Field(ge=0.0, le=1.0)

class ROIMetrics(BaseModel):
    """Return on Investment metrics"""
    total_investment: float
    total_return: float
    roi_percentage: float
    payback_period: float  # Months
    
    # Breakdown
    roi_by_feature: Dict[str, float] = Field(default_factory=dict)
    roi_by_department: Dict[str, float] = Field(default_factory=dict)
    roi_by_use_case: Dict[str, float] = Field(default_factory=dict)
    
    # Time-based
    short_term_roi: float  # 0-6 months
    medium_term_roi: float  # 6-18 months
    long_term_roi: float  # 18+ months

class BusinessKPI(BaseModel):
    """Key Performance Indicator"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    current_value: float
    target_value: float
    unit: str
    
    # Performance
    performance_percentage: float = Field(ge=0.0)  # Current/Target * 100
    trend: str  # "improving", "declining", "stable"
    
    # Thresholds
    critical_threshold: float
    warning_threshold: float
    excellent_threshold: float
    
    # Context
    owner: str
    review_frequency: MetricFrequency
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
class BusinessGoal(BaseModel):
    """Business goal tracking"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    target_value: float
    current_value: float
    unit: str
    
    # Timeline
    start_date: datetime
    target_date: datetime
    
    # Progress
    progress_percentage: float = Field(ge=0.0, le=100.0)
    on_track: bool
    
    # Associated KPIs
    kpi_ids: List[str] = Field(default_factory=list)
    
    # Ownership
    owner: str
    stakeholders: List[str] = Field(default_factory=list)

class BusinessImpactAssessment(BaseModel):
    """Assessment of business impact from decisions/actions"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    decision_id: str
    action_id: Optional[str] = None
    
    # Impact measurement
    predicted_impact: float
    actual_impact: float
    impact_accuracy: float = Field(ge=0.0, le=1.0)
    
    # Impact breakdown
    revenue_impact: float
    cost_impact: float
    efficiency_impact: float
    customer_impact: float
    
    # Attribution
    direct_impact: float
    indirect_impact: float
    attribution_confidence: float = Field(ge=0.0, le=1.0)
    
    # Timeline
    impact_date: datetime
    measurement_date: datetime = Field(default_factory=datetime.utcnow)
    
    # Context
    business_context: Dict[str, Any] = Field(default_factory=dict)
    external_factors: List[str] = Field(default_factory=list)

class CompetitiveIntelligence(BaseModel):
    """Competitive intelligence data"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    competitor_name: str
    
    # Performance comparison
    our_performance: float
    competitor_performance: float
    performance_gap: float
    
    # Market position
    market_share_us: float = Field(ge=0.0, le=1.0)
    market_share_them: float = Field(ge=0.0, le=1.0)
    
    # Intelligence
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    opportunities: List[str] = Field(default_factory=list)
    threats: List[str] = Field(default_factory=list)
    
    # Data source
    source: str
    confidence: float = Field(ge=0.0, le=1.0)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

class MarketIntelligence(BaseModel):
    """Market intelligence and trends"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    market_segment: str
    
    # Market metrics
    market_size: float
    market_growth_rate: float
    our_market_share: float = Field(ge=0.0, le=1.0)
    
    # Trends
    emerging_trends: List[str] = Field(default_factory=list)
    declining_trends: List[str] = Field(default_factory=list)
    
    # Opportunities
    market_opportunities: List[str] = Field(default_factory=list)
    threat_indicators: List[str] = Field(default_factory=list)
    
    # Predictions
    predicted_growth: float
    confidence_interval: tuple[float, float]
    
    # Source
    data_sources: List[str] = Field(default_factory=list)
    analysis_date: datetime = Field(default_factory=datetime.utcnow)

class BusinessDashboard(BaseModel):
    """Business dashboard configuration"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    dashboard_type: str  # "executive", "operational", "analytical"
    
    # Content
    kpi_ids: List[str] = Field(default_factory=list)
    metric_ids: List[str] = Field(default_factory=list)
    chart_configs: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Access
    owner: str
    viewers: List[str] = Field(default_factory=list)
    
    # Configuration
    refresh_frequency: MetricFrequency
    auto_refresh: bool = True
    
    # Layout
    layout_config: Dict[str, Any] = Field(default_factory=dict)
    
class BusinessAlert(BaseModel):
    """Business metric alerts"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    
    # Trigger conditions
    metric_id: str
    condition: str  # "above", "below", "equals", "change"
    threshold_value: float
    
    # Alert details
    severity: str  # "low", "medium", "high", "critical"
    message: str
    
    # Recipients
    recipients: List[str] = Field(default_factory=list)
    notification_channels: List[str] = Field(default_factory=list)
    
    # Status
    is_active: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    
class BusinessReport(BaseModel):
    """Business intelligence report"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    report_type: str  # "performance", "roi", "competitive", "strategic"
    
    # Content
    metrics: List[BusinessMetric] = Field(default_factory=list)
    kpis: List[BusinessKPI] = Field(default_factory=list)
    insights: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    # Generation
    generated_date: datetime = Field(default_factory=datetime.utcnow)
    period_start: datetime
    period_end: datetime
    
    # Distribution
    recipients: List[str] = Field(default_factory=list)
    distribution_schedule: Optional[MetricFrequency] = None
    
    # Format
    format_type: str = "json"  # "json", "pdf", "excel", "html"
    template_id: Optional[str] = None