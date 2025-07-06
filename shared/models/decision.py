"""
Decision Models for LiftOS Intelligence Enhancement
Core data structures for decision engines, recommendations, and automated actions
"""
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class DecisionType(str, Enum):
    """Types of decisions the system can make"""
    RECOMMENDATION = "recommendation"
    AUTOMATED_ACTION = "automated_action"
    ALERT = "alert"
    OPTIMIZATION = "optimization"
    PREDICTION = "prediction"
    STRATEGY = "strategy"


class ConfidenceLevel(str, Enum):
    """Confidence levels for decisions"""
    VERY_LOW = "very_low"      # 0-20%
    LOW = "low"                # 20-40%
    MEDIUM = "medium"          # 40-60%
    HIGH = "high"              # 60-80%
    VERY_HIGH = "very_high"    # 80-100%


class DecisionStatus(str, Enum):
    """Status of decisions"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    FAILED = "failed"
    EXPIRED = "expired"


class ActionType(str, Enum):
    """Types of automated actions"""
    BUDGET_REALLOCATION = "budget_reallocation"
    CAMPAIGN_PAUSE = "campaign_pause"
    CAMPAIGN_RESUME = "campaign_resume"
    BID_ADJUSTMENT = "bid_adjustment"
    AUDIENCE_EXPANSION = "audience_expansion"
    CREATIVE_ROTATION = "creative_rotation"
    ALERT_GENERATION = "alert_generation"
    REPORT_GENERATION = "report_generation"


class RiskLevel(str, Enum):
    """Risk levels for decisions"""
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Decision(BaseModel):
    """Core decision model"""
    id: str
    decision_type: DecisionType
    title: str
    description: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel
    risk_level: RiskLevel
    status: DecisionStatus = DecisionStatus.PENDING
    reasoning: List[str] = []
    evidence: List[str] = []
    alternatives: List[Dict[str, Any]] = []
    expected_impact: Dict[str, float] = {}
    cost_benefit_analysis: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    executed_at: Optional[datetime] = None
    created_by: str  # user_id or system
    organization_id: str
    domain: str  # marketing, causal, performance, etc.
    tags: List[str] = []
    metadata: Dict[str, Any] = {}


class Recommendation(BaseModel):
    """Recommendation with detailed context"""
    id: str
    decision_id: str
    title: str
    description: str
    category: str  # budget, targeting, creative, etc.
    priority: int = Field(..., ge=1, le=5)  # 1=highest, 5=lowest
    confidence: float = Field(..., ge=0.0, le=1.0)
    expected_impact: Dict[str, float] = {}
    implementation_steps: List[str] = []
    required_resources: List[str] = []
    timeline: Optional[str] = None
    dependencies: List[str] = []
    risks: List[str] = []
    success_metrics: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AutomatedAction(BaseModel):
    """Automated action configuration and execution"""
    id: str
    decision_id: str
    action_type: ActionType
    name: str
    description: str
    parameters: Dict[str, Any]
    conditions: List[Dict[str, Any]] = []  # Conditions that trigger the action
    constraints: List[Dict[str, Any]] = []  # Constraints on execution
    approval_required: bool = False
    confidence_threshold: float = Field(..., ge=0.0, le=1.0)
    risk_threshold: RiskLevel = RiskLevel.MEDIUM
    execution_schedule: Optional[str] = None
    retry_policy: Dict[str, Any] = {}
    rollback_plan: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_executed: Optional[datetime] = None
    execution_count: int = 0
    success_rate: Optional[float] = None


class DecisionContext(BaseModel):
    """Context information for decision making"""
    id: str
    organization_id: str
    user_id: Optional[str] = None
    domain: str
    current_state: Dict[str, Any] = {}
    historical_data: Dict[str, Any] = {}
    constraints: List[Dict[str, Any]] = []
    objectives: List[Dict[str, Any]] = []
    preferences: Dict[str, Any] = {}
    external_factors: Dict[str, Any] = {}
    time_horizon: Optional[str] = None
    budget_constraints: Optional[Dict[str, float]] = None
    performance_targets: Dict[str, float] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DecisionOutcome(BaseModel):
    """Outcome tracking for decisions"""
    id: str
    decision_id: str
    actual_impact: Dict[str, float] = {}
    success_metrics: Dict[str, float] = {}
    unexpected_consequences: List[str] = []
    lessons_learned: List[str] = []
    accuracy_score: Optional[float] = None  # How accurate was the prediction
    satisfaction_score: Optional[float] = None  # User satisfaction
    business_value: Optional[float] = None
    measured_at: datetime = Field(default_factory=datetime.utcnow)
    measurement_period: Optional[str] = None
    follow_up_actions: List[str] = []


class ConfidenceScore(BaseModel):
    """Detailed confidence scoring"""
    decision_id: str
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    data_quality_score: float = Field(..., ge=0.0, le=1.0)
    model_reliability_score: float = Field(..., ge=0.0, le=1.0)
    historical_accuracy_score: float = Field(..., ge=0.0, le=1.0)
    domain_expertise_score: float = Field(..., ge=0.0, le=1.0)
    uncertainty_factors: List[str] = []
    confidence_intervals: Dict[str, Tuple[float, float]] = {}
    sensitivity_analysis: Dict[str, float] = {}
    robustness_score: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DecisionRule(BaseModel):
    """Rule-based decision logic"""
    id: str
    name: str
    description: str
    domain: str
    conditions: List[Dict[str, Any]]  # If conditions
    actions: List[Dict[str, Any]]     # Then actions
    priority: int = Field(..., ge=1, le=10)
    is_active: bool = True
    confidence_modifier: float = Field(default=1.0, ge=0.0, le=2.0)
    success_rate: Optional[float] = None
    usage_count: int = 0
    last_used: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str
    tags: List[str] = []


class DecisionStrategy(BaseModel):
    """High-level decision strategy"""
    id: str
    name: str
    description: str
    domain: str
    objectives: List[str]
    principles: List[str]
    decision_rules: List[str]  # DecisionRule IDs
    risk_tolerance: RiskLevel
    time_horizon: str
    success_metrics: List[str]
    constraints: List[Dict[str, Any]] = []
    is_active: bool = True
    performance_history: List[Dict[str, Any]] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: Optional[datetime] = None


class DecisionRequest(BaseModel):
    """Request for decision making"""
    decision_type: DecisionType
    title: str
    description: Optional[str] = None
    context: Dict[str, Any] = {}
    constraints: List[Dict[str, Any]] = []
    objectives: List[str] = []
    time_horizon: Optional[str] = None
    risk_tolerance: RiskLevel = RiskLevel.MEDIUM
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    user_id: str
    organization_id: str
    domain: str
    metadata: Dict[str, Any] = {}


class DecisionResponse(BaseModel):
    """Response from decision engine"""
    decision_id: str
    status: str = "processing"
    estimated_completion: Optional[datetime] = None
    preliminary_recommendations: List[str] = []
    confidence_estimate: Optional[float] = None
    message: str = "Decision processing initiated"


class MultiCriteriaDecision(BaseModel):
    """Multi-criteria decision analysis"""
    id: str
    decision_id: str
    criteria: List[Dict[str, Any]]  # Criteria with weights
    alternatives: List[Dict[str, Any]]  # Alternative options
    scoring_method: str  # weighted_sum, topsis, ahp, etc.
    scores: Dict[str, Dict[str, float]] = {}  # alternative -> criteria -> score
    rankings: List[Dict[str, Any]] = []
    sensitivity_analysis: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DecisionMetrics(BaseModel):
    """Metrics for decision engine performance"""
    decision_engine_id: str
    time_period: Tuple[datetime, datetime]
    total_decisions: int = 0
    decisions_by_type: Dict[DecisionType, int] = {}
    decisions_by_confidence: Dict[ConfidenceLevel, int] = {}
    average_confidence: float = 0.0
    accuracy_rate: float = 0.0
    user_acceptance_rate: float = 0.0
    business_impact: Dict[str, float] = {}
    processing_time_stats: Dict[str, float] = {}
    error_rate: float = 0.0
    improvement_over_baseline: Dict[str, float] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DecisionLearning(BaseModel):
    """Learning from decision outcomes"""
    id: str
    decision_id: str
    outcome_id: str
    learning_type: str  # accuracy_improvement, bias_correction, etc.
    insights: List[str] = []
    model_updates: Dict[str, Any] = {}
    rule_updates: List[str] = []
    confidence_calibration: Dict[str, float] = {}
    performance_improvement: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)