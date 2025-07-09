"""
Channels Service Data Models
Advanced multi-objective optimization with Bayesian inference
"""

from datetime import datetime, date
from typing import Optional, Dict, Any, List, Union, Tuple
from pydantic import BaseModel, Field, validator
from enum import Enum
import numpy as np

from shared.models.base import TimestampMixin


class ChannelType(str, Enum):
    """Supported marketing channels"""
    META = "meta"
    GOOGLE_ADS = "google_ads"
    TIKTOK = "tiktok"
    SNAPCHAT = "snapchat"
    PINTEREST = "pinterest"
    LINKEDIN = "linkedin"
    TWITTER = "twitter"
    EMAIL = "email"
    SMS = "sms"
    DISPLAY = "display"
    VIDEO = "video"
    INFLUENCER = "influencer"
    AFFILIATE = "affiliate"
    DIRECT_MAIL = "direct_mail"
    RADIO = "radio"
    TV = "tv"
    PODCAST = "podcast"
    OTHER = "other"


class OptimizationObjective(str, Enum):
    """Optimization objectives"""
    MAXIMIZE_REVENUE = "maximize_revenue"
    MAXIMIZE_CONVERSIONS = "maximize_conversions"
    MAXIMIZE_ROAS = "maximize_roas"
    MINIMIZE_CAC = "minimize_cac"
    MAXIMIZE_REACH = "maximize_reach"
    MAXIMIZE_BRAND_AWARENESS = "maximize_brand_awareness"
    MINIMIZE_RISK = "minimize_risk"
    MAXIMIZE_EFFICIENCY = "maximize_efficiency"


class ConstraintType(str, Enum):
    """Types of optimization constraints"""
    MIN_SPEND = "min_spend"
    MAX_SPEND = "max_spend"
    MIN_ROAS = "min_roas"
    MAX_CAC = "max_cac"
    GEOGRAPHIC = "geographic"
    TEMPORAL = "temporal"
    BUDGET_SHARE = "budget_share"
    DEPENDENCY = "dependency"
    EXCLUSION = "exclusion"


class SaturationFunction(str, Enum):
    """Saturation function types"""
    HILL = "hill"
    ADSTOCK = "adstock"
    MICHAELIS_MENTEN = "michaelis_menten"
    GOMPERTZ = "gompertz"
    LOGISTIC = "logistic"
    EXPONENTIAL = "exponential"


class OptimizationStatus(str, Enum):
    """Optimization run status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class SimulationStatus(str, Enum):
    """Simulation run status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ScenarioType(str, Enum):
    """Types of simulation scenarios"""
    BUDGET_INCREASE = "budget_increase"
    BUDGET_DECREASE = "budget_decrease"
    CHANNEL_REALLOCATION = "channel_reallocation"
    NEW_CHANNEL = "new_channel"
    SEASONAL_ADJUSTMENT = "seasonal_adjustment"
    COMPETITIVE_RESPONSE = "competitive_response"
    MARKET_EXPANSION = "market_expansion"
    CRISIS_RESPONSE = "crisis_response"


class ChannelPerformance(BaseModel):
    """Current channel performance metrics"""
    channel_id: str = Field(..., description="Unique channel identifier")
    channel_name: str = Field(..., description="Human-readable channel name")
    channel_type: ChannelType = Field(..., description="Type of marketing channel")
    
    # Current metrics
    current_spend: float = Field(..., ge=0.0, description="Current daily/weekly spend")
    current_roas: float = Field(..., ge=0.0, description="Current return on ad spend")
    current_conversions: int = Field(..., ge=0, description="Current conversions")
    current_revenue: float = Field(..., ge=0.0, description="Current revenue")
    current_cac: float = Field(..., ge=0.0, description="Current customer acquisition cost")
    
    # Performance indicators
    saturation_level: float = Field(..., ge=0.0, le=1.0, description="Current saturation level")
    efficiency_score: float = Field(..., ge=0.0, le=1.0, description="Channel efficiency score")
    trend_direction: str = Field(..., description="Performance trend (up/down/stable)")
    
    # Metadata
    last_updated: datetime = Field(..., description="Last update timestamp")
    data_quality_score: float = Field(1.0, ge=0.0, le=1.0, description="Data quality score")
    confidence_level: float = Field(0.95, ge=0.0, le=1.0, description="Confidence in metrics")


class OptimizationConstraint(BaseModel):
    """Optimization constraint definition"""
    constraint_id: str = Field(..., description="Unique constraint identifier")
    constraint_type: ConstraintType = Field(..., description="Type of constraint")
    channel_id: Optional[str] = Field(None, description="Channel this constraint applies to")
    
    # Constraint parameters
    min_value: Optional[float] = Field(None, description="Minimum value")
    max_value: Optional[float] = Field(None, description="Maximum value")
    target_value: Optional[float] = Field(None, description="Target value")
    
    # Geographic constraints
    geographic_regions: List[str] = Field(default_factory=list, description="Geographic regions")
    
    # Temporal constraints
    start_date: Optional[date] = Field(None, description="Constraint start date")
    end_date: Optional[date] = Field(None, description="Constraint end date")
    
    # Dependencies
    dependent_channels: List[str] = Field(default_factory=list, description="Dependent channels")
    exclusion_channels: List[str] = Field(default_factory=list, description="Mutually exclusive channels")
    
    # Constraint metadata
    priority: int = Field(1, ge=1, le=5, description="Constraint priority (1=highest)")
    is_hard_constraint: bool = Field(True, description="Hard vs soft constraint")
    penalty_weight: float = Field(1.0, ge=0.0, description="Penalty weight for soft constraints")
    description: str = Field("", description="Human-readable constraint description")


class SaturationModel(BaseModel):
    """Channel saturation model parameters"""
    channel_id: str = Field(..., description="Channel identifier")
    function_type: SaturationFunction = Field(..., description="Saturation function type")
    
    # Hill function parameters
    alpha: Optional[float] = Field(None, description="Hill function alpha parameter")
    gamma: Optional[float] = Field(None, description="Hill function gamma parameter")
    
    # Adstock parameters
    adstock_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Adstock decay rate")
    adstock_max_lag: Optional[int] = Field(None, ge=0, description="Maximum adstock lag")
    
    # General parameters
    saturation_point: float = Field(..., gt=0.0, description="Saturation point")
    diminishing_returns_start: float = Field(..., gt=0.0, description="Point where diminishing returns start")
    max_response: float = Field(..., gt=0.0, description="Maximum response at saturation")
    
    # Model quality
    r_squared: float = Field(..., ge=0.0, le=1.0, description="Model R-squared")
    confidence_interval: Tuple[float, float] = Field(..., description="95% confidence interval")
    last_calibrated: datetime = Field(..., description="Last calibration timestamp")
    
    def evaluate(self, spend: float) -> float:
        """Evaluate saturation function at given spend level"""
        if self.function_type == SaturationFunction.HILL:
            return self._hill_function(spend)
        elif self.function_type == SaturationFunction.MICHAELIS_MENTEN:
            return self._michaelis_menten(spend)
        else:
            # Default to Hill function
            return self._hill_function(spend)
    
    def _hill_function(self, spend: float) -> float:
        """Hill saturation function"""
        if not self.alpha or not self.gamma:
            return spend  # Linear if parameters not set
        
        return self.max_response * (spend ** self.alpha) / (self.gamma ** self.alpha + spend ** self.alpha)
    
    def _michaelis_menten(self, spend: float) -> float:
        """Michaelis-Menten saturation function"""
        return self.max_response * spend / (self.saturation_point + spend)


class BudgetOptimizationRequest(BaseModel):
    """Request for budget optimization"""
    org_id: str = Field(..., description="Organization identifier")
    
    # Budget parameters
    total_budget: float = Field(..., gt=0.0, description="Total budget to allocate")
    time_horizon: int = Field(..., gt=0, description="Optimization time horizon (days)")
    
    # Channels and objectives
    channels: List[str] = Field(..., min_items=2, description="Channels to optimize")
    objectives: List[OptimizationObjective] = Field(..., min_items=1, description="Optimization objectives")
    objective_weights: Dict[str, float] = Field(default_factory=dict, description="Objective weights")
    
    # Constraints
    constraints: List[OptimizationConstraint] = Field(default_factory=list, description="Optimization constraints")
    
    # Risk and preferences
    risk_tolerance: float = Field(0.5, ge=0.0, le=1.0, description="Risk tolerance (0=conservative, 1=aggressive)")
    confidence_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Minimum confidence for recommendations")
    
    # Advanced options
    use_bayesian_optimization: bool = Field(True, description="Use Bayesian optimization")
    include_interaction_effects: bool = Field(True, description="Include channel interaction effects")
    monte_carlo_samples: int = Field(1000, ge=100, le=10000, description="Monte Carlo simulation samples")
    
    # Metadata
    optimization_name: Optional[str] = Field(None, description="Human-readable optimization name")
    description: Optional[str] = Field(None, description="Optimization description")


class OptimizationAction(BaseModel):
    """Individual optimization action"""
    action_id: str = Field(..., description="Unique action identifier")
    channel_id: str = Field(..., description="Target channel")
    action_type: str = Field(..., description="Type of action (increase/decrease/maintain)")
    
    # Budget changes
    current_budget: float = Field(..., ge=0.0, description="Current budget allocation")
    recommended_budget: float = Field(..., ge=0.0, description="Recommended budget allocation")
    budget_change: float = Field(..., description="Absolute budget change")
    budget_change_percent: float = Field(..., description="Percentage budget change")
    
    # Expected impact
    expected_revenue_impact: float = Field(..., description="Expected revenue impact")
    expected_conversion_impact: int = Field(..., description="Expected conversion impact")
    expected_roas_change: float = Field(..., description="Expected ROAS change")
    
    # Implementation
    priority: int = Field(..., ge=1, le=5, description="Implementation priority")
    implementation_timeline: str = Field(..., description="Recommended implementation timeline")
    risk_level: str = Field(..., description="Risk level (low/medium/high)")
    
    # Confidence
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in recommendation")
    confidence_interval: Tuple[float, float] = Field(..., description="Confidence interval for impact")


class OptimizationResult(BaseModel):
    """Result of budget optimization"""
    optimization_id: str = Field(..., description="Unique optimization identifier")
    org_id: str = Field(..., description="Organization identifier")
    status: OptimizationStatus = Field(..., description="Optimization status")
    
    # Input summary
    total_budget: float = Field(..., description="Total budget optimized")
    channels_optimized: List[str] = Field(..., description="Channels included in optimization")
    objectives: List[OptimizationObjective] = Field(..., description="Optimization objectives")
    
    # Recommended allocation
    recommended_allocation: Dict[str, float] = Field(..., description="Recommended budget allocation by channel")
    current_allocation: Dict[str, float] = Field(..., description="Current budget allocation by channel")
    
    # Expected performance
    expected_performance: Dict[str, float] = Field(..., description="Expected performance metrics")
    performance_improvement: Dict[str, float] = Field(..., description="Expected improvement over current")
    
    # Risk and confidence
    confidence_intervals: Dict[str, Tuple[float, float]] = Field(..., description="Confidence intervals")
    risk_metrics: Dict[str, float] = Field(..., description="Risk assessment metrics")
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    
    # Implementation plan
    implementation_plan: List[OptimizationAction] = Field(..., description="Detailed implementation actions")
    estimated_implementation_time: str = Field(..., description="Estimated implementation time")
    
    # Optimization metadata
    algorithm_used: str = Field(..., description="Optimization algorithm used")
    convergence_status: str = Field(..., description="Algorithm convergence status")
    iterations: int = Field(..., description="Number of optimization iterations")
    computation_time: float = Field(..., description="Computation time in seconds")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")


class ScenarioDefinition(BaseModel):
    """Definition of a what-if scenario"""
    scenario_id: str = Field(..., description="Unique scenario identifier")
    scenario_name: str = Field(..., description="Human-readable scenario name")
    
    # Budget allocation for scenario
    budget_allocation: Dict[str, float] = Field(..., description="Budget allocation by channel")
    
    # External factors
    external_factors: Dict[str, float] = Field(default_factory=dict, description="External factor adjustments")
    market_conditions: Dict[str, Any] = Field(default_factory=dict, description="Market condition assumptions")
    
    # Scenario metadata
    description: Optional[str] = Field(None, description="Scenario description")
    probability: Optional[float] = Field(None, ge=0.0, le=1.0, description="Scenario probability")


class SimulationRequest(BaseModel):
    """Request for scenario simulation"""
    org_id: str = Field(..., description="Organization identifier")
    
    # Scenarios to simulate
    scenarios: List[ScenarioDefinition] = Field(..., min_items=1, description="Scenarios to simulate")
    
    # Simulation parameters
    time_horizon: int = Field(..., gt=0, description="Simulation time horizon (days)")
    monte_carlo_samples: int = Field(1000, ge=100, le=10000, description="Monte Carlo samples per scenario")
    confidence_levels: List[float] = Field([0.8, 0.9, 0.95], description="Confidence levels for intervals")
    
    # Advanced options
    include_uncertainty: bool = Field(True, description="Include parameter uncertainty")
    include_interactions: bool = Field(True, description="Include channel interactions")
    sensitivity_analysis: bool = Field(True, description="Perform sensitivity analysis")
    
    # Metadata
    simulation_name: Optional[str] = Field(None, description="Simulation name")
    description: Optional[str] = Field(None, description="Simulation description")


class ScenarioResult(BaseModel):
    """Result of a single scenario simulation"""
    scenario_id: str = Field(..., description="Scenario identifier")
    scenario_name: str = Field(..., description="Scenario name")
    
    # Performance predictions
    predicted_revenue: float = Field(..., description="Predicted total revenue")
    predicted_conversions: int = Field(..., description="Predicted total conversions")
    predicted_roas: float = Field(..., description="Predicted overall ROAS")
    
    # Channel-level results
    channel_performance: Dict[str, Dict[str, float]] = Field(..., description="Performance by channel")
    
    # Uncertainty quantification
    confidence_intervals: Dict[str, Dict[str, Tuple[float, float]]] = Field(..., description="Confidence intervals")
    prediction_variance: Dict[str, float] = Field(..., description="Prediction variance")
    
    # Risk assessment
    downside_risk: float = Field(..., description="Downside risk assessment")
    upside_potential: float = Field(..., description="Upside potential")
    risk_adjusted_return: float = Field(..., description="Risk-adjusted return")


class SimulationResult(BaseModel):
    """Result of scenario simulation"""
    simulation_id: str = Field(..., description="Unique simulation identifier")
    org_id: str = Field(..., description="Organization identifier")
    
    # Scenario results
    scenario_results: List[ScenarioResult] = Field(..., description="Results for each scenario")
    
    # Comparative analysis
    best_scenario: str = Field(..., description="Best performing scenario ID")
    worst_scenario: str = Field(..., description="Worst performing scenario ID")
    scenario_rankings: List[Dict[str, Any]] = Field(..., description="Scenario rankings by objective")
    
    # Sensitivity analysis
    sensitivity_analysis: Optional[Dict[str, Any]] = Field(None, description="Sensitivity analysis results")
    
    # Simulation metadata
    total_scenarios: int = Field(..., description="Total scenarios simulated")
    samples_per_scenario: int = Field(..., description="Monte Carlo samples per scenario")
    computation_time: float = Field(..., description="Total computation time")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")


class ChannelConfiguration(BaseModel):
    """Channel configuration and settings"""
    channel_id: str = Field(..., description="Unique channel identifier")
    channel_name: str = Field(..., description="Human-readable channel name")
    channel_type: ChannelType = Field(..., description="Type of marketing channel")
    org_id: str = Field(..., description="Organization identifier")
    
    # Configuration
    is_active: bool = Field(True, description="Whether channel is active")
    auto_optimization: bool = Field(False, description="Enable automatic optimization")
    
    # Budget settings
    min_daily_budget: float = Field(0.0, ge=0.0, description="Minimum daily budget")
    max_daily_budget: Optional[float] = Field(None, description="Maximum daily budget")
    current_daily_budget: float = Field(..., ge=0.0, description="Current daily budget")
    
    # Performance targets
    target_roas: Optional[float] = Field(None, gt=0.0, description="Target ROAS")
    target_cac: Optional[float] = Field(None, gt=0.0, description="Target CAC")
    
    # Saturation model
    saturation_model: Optional[SaturationModel] = Field(None, description="Channel saturation model")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    created_by: str = Field(..., description="User who created the configuration")


class RecommendationResponse(BaseModel):
    """Response containing budget recommendations"""
    org_id: str = Field(..., description="Organization identifier")
    
    # Recommendations
    recommendations: List[OptimizationAction] = Field(..., description="Budget reallocation recommendations")
    
    # Summary metrics
    total_budget_impact: float = Field(..., description="Total budget change")
    expected_revenue_lift: float = Field(..., description="Expected revenue increase")
    expected_efficiency_gain: float = Field(..., description="Expected efficiency improvement")
    
    # Implementation guidance
    quick_wins: List[OptimizationAction] = Field(..., description="Quick win recommendations")
    strategic_moves: List[OptimizationAction] = Field(..., description="Strategic recommendations")
    
    # Confidence and risk
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    implementation_risk: str = Field(..., description="Implementation risk level")
    
    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    valid_until: datetime = Field(..., description="Recommendation validity period")
    recommendation_version: str = Field("1.0", description="Recommendation version")