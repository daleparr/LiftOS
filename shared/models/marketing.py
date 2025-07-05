"""
Marketing Data Models for Centralized Data Ingestion
Extends base models for Meta, Google, and Klaviyo data structures
"""
from datetime import datetime, date
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field
from enum import Enum

from .base import TimestampMixin


class DataSource(str, Enum):
    """Marketing data sources"""
    META_BUSINESS = "meta_business"
    GOOGLE_ADS = "google_ads"
    KLAVIYO = "klaviyo"
    MANUAL = "manual"


class CampaignObjective(str, Enum):
    """Campaign objectives across platforms"""
    AWARENESS = "awareness"
    TRAFFIC = "traffic"
    ENGAGEMENT = "engagement"
    LEADS = "leads"
    CONVERSIONS = "conversions"
    SALES = "sales"
    APP_INSTALLS = "app_installs"
    VIDEO_VIEWS = "video_views"


class AdStatus(str, Enum):
    """Ad status across platforms"""
    ACTIVE = "active"
    PAUSED = "paused"
    DELETED = "deleted"
    PENDING = "pending"
    DISAPPROVED = "disapproved"
    LEARNING = "learning"


# Base Marketing Data Models
class MarketingDataEntry(TimestampMixin):
    """Base model for all marketing data entries"""
    id: str
    org_id: str
    data_source: DataSource
    source_id: str  # Original ID from the platform
    raw_data: Dict[str, Any]  # Original API response
    processed_data: Dict[str, Any]  # Normalized data
    date_range_start: date
    date_range_end: date
    memory_type: str = "marketing_data"
    metadata: Dict[str, Any] = {}


# Campaign Data Models
class CampaignData(MarketingDataEntry):
    """Campaign-level marketing data"""
    campaign_name: str
    campaign_objective: Optional[CampaignObjective] = None
    status: AdStatus
    budget_daily: Optional[float] = None
    budget_lifetime: Optional[float] = None
    spend: float = 0.0
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    revenue: float = 0.0
    memory_type: str = "campaign_data"


class AdSetData(MarketingDataEntry):
    """Ad Set/Ad Group level marketing data"""
    campaign_id: str
    adset_name: str
    targeting: Dict[str, Any] = {}
    bid_strategy: Optional[str] = None
    status: AdStatus
    spend: float = 0.0
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    revenue: float = 0.0
    memory_type: str = "adset_data"


class AdData(MarketingDataEntry):
    """Individual ad level marketing data"""
    campaign_id: str
    adset_id: str
    ad_name: str
    ad_creative: Dict[str, Any] = {}
    status: AdStatus
    spend: float = 0.0
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    revenue: float = 0.0
    memory_type: str = "ad_data"


# Platform-Specific Models
class MetaBusinessData(MarketingDataEntry):
    """Meta Business API specific data"""
    account_id: str
    campaign_id: Optional[str] = None
    adset_id: Optional[str] = None
    ad_id: Optional[str] = None
    data_source: DataSource = DataSource.META_BUSINESS
    
    # Meta-specific metrics
    cpm: Optional[float] = None
    cpc: Optional[float] = None
    ctr: Optional[float] = None
    frequency: Optional[float] = None
    reach: Optional[int] = None
    video_views: Optional[int] = None
    memory_type: str = "meta_business_data"


class GoogleAdsData(MarketingDataEntry):
    """Google Ads API specific data"""
    customer_id: str
    campaign_id: Optional[str] = None
    ad_group_id: Optional[str] = None
    ad_id: Optional[str] = None
    data_source: DataSource = DataSource.GOOGLE_ADS
    
    # Google Ads specific metrics
    quality_score: Optional[float] = None
    search_impression_share: Optional[float] = None
    search_rank_lost_impression_share: Optional[float] = None
    cost_per_conversion: Optional[float] = None
    conversion_rate: Optional[float] = None
    memory_type: str = "google_ads_data"


class KlaviyoData(MarketingDataEntry):
    """Klaviyo API specific data"""
    list_id: Optional[str] = None
    segment_id: Optional[str] = None
    campaign_id: Optional[str] = None
    flow_id: Optional[str] = None
    data_source: DataSource = DataSource.KLAVIYO
    
    # Klaviyo specific metrics
    delivered: Optional[int] = None
    opened: Optional[int] = None
    clicked: Optional[int] = None
    unsubscribed: Optional[int] = None
    bounced: Optional[int] = None
    open_rate: Optional[float] = None
    click_rate: Optional[float] = None
    memory_type: str = "klaviyo_data"


# Calendar Dimension Models
class CalendarDimension(BaseModel):
    """Calendar dimensions for causal modeling"""
    date: date
    year: int
    quarter: int
    month: int
    week: int
    day_of_year: int
    day_of_month: int
    day_of_week: int
    is_weekend: bool
    is_holiday: bool = False
    holiday_name: Optional[str] = None
    season: str  # spring, summer, fall, winter
    fiscal_year: Optional[int] = None
    fiscal_quarter: Optional[int] = None


# Request/Response Models for API
class MarketingDataIngestionRequest(BaseModel):
    """Request model for marketing data ingestion"""
    data_source: DataSource
    data_entries: List[Dict[str, Any]]
    date_range_start: date
    date_range_end: date
    metadata: Optional[Dict[str, Any]] = None


class MarketingDataSearchRequest(BaseModel):
    """Request model for marketing data search"""
    query: str
    data_sources: Optional[List[DataSource]] = None
    date_range_start: Optional[date] = None
    date_range_end: Optional[date] = None
    campaign_ids: Optional[List[str]] = None
    metrics: Optional[List[str]] = None
    limit: int = Field(10, ge=1, le=100)
    search_type: str = Field("hybrid", pattern="^(neural|conceptual|knowledge|hybrid)$")


class MarketingDataAggregationRequest(BaseModel):
    """Request model for marketing data aggregation"""
    data_sources: List[DataSource]
    date_range_start: date
    date_range_end: date
    group_by: List[str]  # campaign, adset, ad, date, etc.
    metrics: List[str]  # spend, impressions, clicks, conversions, revenue
    filters: Optional[Dict[str, Any]] = None


class MarketingInsights(BaseModel):
    """Marketing data insights and analytics"""
    total_spend: float
    total_impressions: int
    total_clicks: int
    total_conversions: int
    total_revenue: float
    average_cpc: float
    average_cpm: float
    average_ctr: float
    conversion_rate: float
    roas: float  # Return on Ad Spend
    data_sources_summary: Dict[DataSource, Dict[str, Any]]
    top_campaigns: List[Dict[str, Any]]
    performance_trends: Dict[str, List[Dict[str, Any]]]
    calendar_insights: Dict[str, Any]


# Data Transformation Models
class DataTransformationRule(BaseModel):
    """Rules for transforming raw marketing data"""
    source_field: str
    target_field: str
    transformation_type: str  # map, calculate, aggregate, normalize
    transformation_logic: Dict[str, Any]
    conditions: Optional[Dict[str, Any]] = None


class PandasTransformationConfig(BaseModel):
    """Configuration for pandas data transformations"""
    data_source: DataSource
    transformation_rules: List[DataTransformationRule]
    aggregation_rules: Optional[Dict[str, Any]] = None
    calendar_join: bool = True
    output_format: str = "normalized"  # normalized, raw, aggregated