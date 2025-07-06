"""
Enhanced Causal Marketing Data Models
Optimized for causal inference and attribution analysis
"""
from datetime import datetime, date
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field
from enum import Enum
import numpy as np

from .base import TimestampMixin
from .marketing import DataSource, CampaignObjective, AdStatus


class TreatmentType(str, Enum):
    """Types of marketing treatments"""
    BUDGET_INCREASE = "budget_increase"
    BUDGET_DECREASE = "budget_decrease"
    TARGETING_CHANGE = "targeting_change"
    CREATIVE_CHANGE = "creative_change"
    BID_STRATEGY_CHANGE = "bid_strategy_change"
    CAMPAIGN_LAUNCH = "campaign_launch"
    CAMPAIGN_PAUSE = "campaign_pause"
    CONTROL = "control"


class RandomizationUnit(str, Enum):
    """Units of randomization for experiments"""
    CAMPAIGN = "campaign"
    AD_SET = "ad_set"
    GEOGRAPHIC_REGION = "geographic_region"
    TIME_PERIOD = "time_period"
    USER_SEGMENT = "user_segment"
    PRODUCT_CATEGORY = "product_category"


class CausalMethod(str, Enum):
    """Causal inference methods"""
    DIFFERENCE_IN_DIFFERENCES = "difference_in_differences"
    INSTRUMENTAL_VARIABLES = "instrumental_variables"
    REGRESSION_DISCONTINUITY = "regression_discontinuity"
    MATCHING = "matching"
    SYNTHETIC_CONTROL = "synthetic_control"
    RANDOMIZED_EXPERIMENT = "randomized_experiment"


class ExternalFactor(BaseModel):
    """External factors that may confound causal relationships"""
    factor_name: str
    factor_type: str  # economic, seasonal, competitive, regulatory
    value: Union[float, str, bool]
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in factor measurement")
    source: str = Field(description="Source of factor data")
    timestamp: datetime


class ConfounderVariable(BaseModel):
    """Confounding variables that need to be controlled"""
    variable_name: str
    variable_type: str  # continuous, categorical, binary
    value: Union[float, str, bool]
    importance_score: float = Field(ge=0.0, le=1.0, description="Importance for causal inference")
    detection_method: str = Field(description="How the confounder was detected")
    control_strategy: str = Field(description="How to control for this confounder")


class CausalGraph(BaseModel):
    """Causal graph representation"""
    nodes: List[str] = Field(description="Variables in the causal graph")
    edges: List[Dict[str, str]] = Field(description="Causal relationships")
    treatment_nodes: List[str] = Field(description="Treatment variables")
    outcome_nodes: List[str] = Field(description="Outcome variables")
    confounder_nodes: List[str] = Field(description="Confounding variables")
    mediator_nodes: List[str] = Field(description="Mediating variables")
    graph_version: str = Field(description="Version of the causal graph")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CalendarDimension(BaseModel):
    """Enhanced calendar dimensions for causal modeling"""
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
    
    # Economic indicators
    economic_indicators: Dict[str, float] = Field(default_factory=dict)
    market_conditions: Dict[str, Any] = Field(default_factory=dict)
    
    # Competitive landscape
    competitor_activity: Dict[str, Any] = Field(default_factory=dict)
    industry_events: List[str] = Field(default_factory=list)


class CausalMarketingData(TimestampMixin):
    """Enhanced marketing data schema optimized for causal inference"""
    
    # Core Identifiers
    id: str = Field(..., description="Unique record identifier")
    org_id: str = Field(..., description="Organization identifier")
    timestamp: datetime = Field(..., description="Precise event timestamp")
    
    # Data Source & Campaign Info
    data_source: DataSource = Field(..., description="Marketing platform")
    campaign_id: str = Field(..., description="Campaign identifier")
    campaign_name: str = Field(..., description="Human-readable campaign name")
    campaign_objective: CampaignObjective = Field(..., description="Campaign goal")
    campaign_status: AdStatus = Field(..., description="Campaign status")
    
    # Hierarchical Structure
    ad_set_id: Optional[str] = Field(None, description="Ad set identifier")
    ad_set_name: Optional[str] = Field(None, description="Ad set name")
    ad_id: Optional[str] = Field(None, description="Individual ad identifier")
    ad_name: Optional[str] = Field(None, description="Individual ad name")
    
    # Treatment Assignment (Critical for Causal Inference)
    treatment_group: str = Field(..., description="Treatment/control assignment")
    treatment_type: TreatmentType = Field(..., description="Type of treatment applied")
    treatment_intensity: float = Field(0.0, ge=0.0, le=1.0, description="Treatment strength")
    randomization_unit: RandomizationUnit = Field(..., description="Unit of randomization")
    experiment_id: Optional[str] = Field(None, description="Associated experiment ID")
    
    # Core Marketing Metrics
    spend: float = Field(0.0, ge=0.0, description="Marketing spend")
    impressions: int = Field(0, ge=0, description="Ad impressions")
    clicks: int = Field(0, ge=0, description="Ad clicks")
    conversions: int = Field(0, ge=0, description="Conversions")
    revenue: float = Field(0.0, ge=0.0, description="Revenue attributed")
    
    # Platform-specific metrics
    platform_metrics: Dict[str, Any] = Field(default_factory=dict, description="Platform-specific metrics")
    
    # Causal Confounders (Critical for Unbiased Estimation)
    confounders: List[ConfounderVariable] = Field(default_factory=list, description="Confounding variables")
    external_factors: List[ExternalFactor] = Field(default_factory=list, description="External influences")
    
    # Temporal Context
    calendar_features: CalendarDimension = Field(..., description="Calendar context")
    lag_features: Dict[str, float] = Field(default_factory=dict, description="Lagged variables")
    
    # Geographic Context
    geographic_data: Dict[str, Any] = Field(default_factory=dict, description="Geographic information")
    
    # Audience Context
    audience_data: Dict[str, Any] = Field(default_factory=dict, description="Audience characteristics")
    
    # Causal Metadata
    causal_metadata: Dict[str, Any] = Field(default_factory=dict, description="Causal inference metadata")
    data_quality_score: float = Field(1.0, ge=0.0, le=1.0, description="Data quality for causal inference")
    causal_graph_id: Optional[str] = Field(None, description="Associated causal graph")
    
    # KSE Integration
    embedding_vector: Optional[List[float]] = Field(None, description="Neural embedding")
    conceptual_space: Optional[str] = Field(None, description="Conceptual space assignment")
    knowledge_graph_nodes: List[str] = Field(default_factory=list, description="Knowledge graph connections")
    
    # Validation
    validation_flags: Dict[str, bool] = Field(default_factory=dict, description="Data validation results")
    anomaly_score: float = Field(0.0, ge=0.0, le=1.0, description="Anomaly detection score")


class CausalExperiment(TimestampMixin):
    """Causal experiment definition and tracking"""
    experiment_id: str = Field(..., description="Unique experiment identifier")
    org_id: str = Field(..., description="Organization identifier")
    experiment_name: str = Field(..., description="Human-readable experiment name")
    experiment_type: str = Field(..., description="Type of experiment")
    
    # Experiment Design
    treatment_definition: Dict[str, Any] = Field(..., description="Treatment definition")
    control_definition: Dict[str, Any] = Field(..., description="Control definition")
    randomization_strategy: Dict[str, Any] = Field(..., description="Randomization approach")
    
    # Timeline
    start_date: datetime = Field(..., description="Experiment start date")
    end_date: datetime = Field(..., description="Experiment end date")
    
    # Statistical Design
    statistical_power: float = Field(0.8, ge=0.0, le=1.0, description="Desired statistical power")
    significance_level: float = Field(0.05, ge=0.0, le=1.0, description="Significance level")
    minimum_detectable_effect: float = Field(..., description="Minimum effect size to detect")
    
    # Causal Structure
    causal_graph: CausalGraph = Field(..., description="Causal graph for experiment")
    primary_outcome: str = Field(..., description="Primary outcome variable")
    secondary_outcomes: List[str] = Field(default_factory=list, description="Secondary outcomes")
    
    # Status
    status: str = Field("planned", description="Experiment status")
    results: Optional[Dict[str, Any]] = Field(None, description="Experiment results")


class AttributionModel(TimestampMixin):
    """Causal attribution model definition"""
    model_id: str = Field(..., description="Unique model identifier")
    org_id: str = Field(..., description="Organization identifier")
    model_name: str = Field(..., description="Human-readable model name")
    model_type: str = Field(..., description="Type of attribution model")
    
    # Model Configuration
    causal_method: CausalMethod = Field(..., description="Causal inference method")
    model_configuration: Dict[str, Any] = Field(..., description="Model configuration")
    feature_config: Dict[str, Any] = Field(..., description="Feature engineering config")
    
    # Training Data
    training_data_query: Dict[str, Any] = Field(..., description="Query for training data")
    training_period_start: date = Field(..., description="Training period start")
    training_period_end: date = Field(..., description="Training period end")
    
    # Model Performance
    validation_metrics: Dict[str, float] = Field(default_factory=dict, description="Model validation metrics")
    feature_importance: Dict[str, float] = Field(default_factory=dict, description="Feature importance scores")
    causal_effects: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Estimated causal effects")
    
    # Model Status
    status: str = Field("training", description="Model status")
    version: str = Field("1.0", description="Model version")
    last_retrained: Optional[datetime] = Field(None, description="Last retraining timestamp")


class CausalInsight(TimestampMixin):
    """Causal insights generated from analysis"""
    insight_id: str = Field(..., description="Unique insight identifier")
    org_id: str = Field(..., description="Organization identifier")
    insight_type: str = Field(..., description="Type of insight")
    
    # Source Information
    source_model_id: str = Field(..., description="Source attribution model")
    source_experiment_id: Optional[str] = Field(None, description="Source experiment")
    
    # Insight Content
    title: str = Field(..., description="Insight title")
    description: str = Field(..., description="Detailed description")
    causal_claim: str = Field(..., description="Causal claim being made")
    evidence_strength: float = Field(ge=0.0, le=1.0, description="Strength of causal evidence")
    
    # Quantitative Results
    effect_size: float = Field(..., description="Estimated effect size")
    confidence_interval: List[float] = Field(..., description="Confidence interval")
    p_value: Optional[float] = Field(None, description="Statistical significance")
    
    # Recommendations
    recommendations: List[str] = Field(default_factory=list, description="Action recommendations")
    expected_impact: Dict[str, float] = Field(default_factory=dict, description="Expected impact of recommendations")
    
    # Validation
    validation_status: str = Field("pending", description="Validation status")
    validation_notes: Optional[str] = Field(None, description="Validation notes")


# Request/Response Models
class CausalAnalysisRequest(BaseModel):
    """Request for causal analysis"""
    org_id: str
    analysis_type: str = Field(..., description="Type of causal analysis")
    data_query: Dict[str, Any] = Field(..., description="Query for data selection")
    causal_method: CausalMethod = Field(..., description="Causal inference method")
    treatment_variable: str = Field(..., description="Treatment variable")
    outcome_variable: str = Field(..., description="Outcome variable")
    confounders: List[str] = Field(default_factory=list, description="Known confounders")
    time_window: Dict[str, date] = Field(..., description="Analysis time window")
    options: Dict[str, Any] = Field(default_factory=dict, description="Analysis options")


class CausalAnalysisResponse(BaseModel):
    """Response from causal analysis"""
    analysis_id: str
    causal_effects: Dict[str, Dict[str, float]]
    confidence_intervals: Dict[str, List[float]]
    model_diagnostics: Dict[str, Any]
    causal_graph: CausalGraph
    insights: List[CausalInsight]
    recommendations: List[str]
    data_quality_assessment: Dict[str, float]
    validation_results: Dict[str, Any]


class DataQualityAssessment(BaseModel):
    """Data quality assessment for causal inference"""
    overall_score: float = Field(ge=0.0, le=1.0)
    temporal_consistency: float = Field(ge=0.0, le=1.0)
    confounder_coverage: float = Field(ge=0.0, le=1.0)
    treatment_assignment_quality: float = Field(ge=0.0, le=1.0)
    outcome_measurement_quality: float = Field(ge=0.0, le=1.0)
    external_validity: float = Field(ge=0.0, le=1.0)
    missing_data_score: float = Field(ge=0.0, le=1.0)
    anomaly_detection_score: float = Field(ge=0.0, le=1.0)
    recommendations: List[str] = Field(default_factory=list)