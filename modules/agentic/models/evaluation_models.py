"""
Evaluation Models for Agentic Module

Defines data structures for agent evaluation results, metrics,
and assessment categories based on AgentSIM's evaluation framework.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator, ConfigDict
from datetime import datetime


class EvaluationCategory(str, Enum):
    """Categories for agent evaluation based on AgentSIM framework."""
    FUNCTIONALITY = "functionality"  # 30% weight
    RELIABILITY = "reliability"      # 25% weight
    PERFORMANCE = "performance"      # 20% weight
    SECURITY = "security"           # 15% weight
    USABILITY = "usability"         # 10% weight


class MetricType(str, Enum):
    """Types of metrics that can be measured."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    AVAILABILITY = "availability"
    CONSISTENCY = "consistency"
    ROBUSTNESS = "robustness"
    SECURITY_SCORE = "security_score"
    USABILITY_SCORE = "usability_score"
    COST_EFFICIENCY = "cost_efficiency"
    RESOURCE_UTILIZATION = "resource_utilization"


class DeploymentReadiness(str, Enum):
    """Deployment readiness levels."""
    NOT_READY = "not_ready"
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class MetricScore(BaseModel):
    """Individual metric score with context."""
    metric_type: MetricType = Field(..., description="Type of metric")
    value: float = Field(..., description="Metric value")
    max_value: float = Field(default=1.0, description="Maximum possible value")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    description: Optional[str] = Field(None, description="Metric description")
    
    @validator('value')
    def validate_value(cls, v, values):
        """Ensure value is within reasonable bounds."""
        if v < 0:
            raise ValueError("Metric value cannot be negative")
        return v
    
    @property
    def normalized_score(self) -> float:
        """Get normalized score (0-1)."""
        if self.max_value == 0:
            return 0.0
        return min(self.value / self.max_value, 1.0)


class CategoryAssessment(BaseModel):
    """Assessment for a specific evaluation category."""
    category: EvaluationCategory = Field(..., description="Evaluation category")
    metrics: List[MetricScore] = Field(..., description="Metrics in this category")
    weight: float = Field(..., ge=0.0, le=1.0, description="Category weight")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall category score")
    feedback: Optional[str] = Field(None, description="Qualitative feedback")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")
    
    @validator('overall_score')
    def validate_overall_score(cls, v):
        """Ensure overall score is between 0 and 1."""
        return max(0.0, min(v, 1.0))


class MarketingMetrics(BaseModel):
    """Marketing-specific metrics for agent evaluation."""
    attribution_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    budget_optimization_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    campaign_performance_prediction: Optional[float] = Field(None, ge=0.0, le=1.0)
    audience_segmentation_quality: Optional[float] = Field(None, ge=0.0, le=1.0)
    creative_effectiveness_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    channel_mix_optimization: Optional[float] = Field(None, ge=0.0, le=1.0)
    roi_prediction_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    incrementality_measurement: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    def get_non_null_metrics(self) -> Dict[str, float]:
        """Get all non-null metrics as a dictionary."""
        return {
            field: value for field, value in self.dict().items() 
            if value is not None
        }


class EvaluationMatrix(BaseModel):
    """Complete evaluation matrix with category weights."""
    functionality_weight: float = Field(default=0.30, ge=0.0, le=1.0)
    reliability_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    performance_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    security_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    usability_weight: float = Field(default=0.10, ge=0.0, le=1.0)
    
    @validator('usability_weight')
    def validate_weights_sum_to_one(cls, v, values):
        """Ensure all weights sum to 1.0."""
        total = (
            values.get('functionality_weight', 0) +
            values.get('reliability_weight', 0) +
            values.get('performance_weight', 0) +
            values.get('security_weight', 0) +
            v
        )
        if abs(total - 1.0) > 0.001:  # Allow small floating point errors
            raise ValueError(f"Category weights must sum to 1.0, got {total}")
        return v
    
    def get_category_weight(self, category: EvaluationCategory) -> float:
        """Get weight for a specific category."""
        weight_map = {
            EvaluationCategory.FUNCTIONALITY: self.functionality_weight,
            EvaluationCategory.RELIABILITY: self.reliability_weight,
            EvaluationCategory.PERFORMANCE: self.performance_weight,
            EvaluationCategory.SECURITY: self.security_weight,
            EvaluationCategory.USABILITY: self.usability_weight,
        }
        return weight_map[category]


class AgentEvaluationResult(BaseModel):
    """
    Complete evaluation result for a marketing agent.
    
    This model captures all aspects of agent performance evaluation
    based on the AgentSIM framework adapted for marketing use cases.
    """
    
    # Identity
    evaluation_id: str = Field(..., description="Unique evaluation identifier")
    agent_id: str = Field(..., description="ID of evaluated agent")
    agent_name: str = Field(..., description="Name of evaluated agent")
    
    # Evaluation context
    test_case_id: Optional[str] = Field(None, description="Associated test case")
    evaluation_type: str = Field(..., description="Type of evaluation performed")
    evaluator_version: str = Field(default="1.0.0", description="Evaluator version")
    
    # Results
    category_assessments: List[CategoryAssessment] = Field(..., description="Category-wise assessments")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Weighted overall score")
    deployment_readiness: DeploymentReadiness = Field(..., description="Deployment readiness level")
    
    # Marketing-specific metrics
    marketing_metrics: Optional[MarketingMetrics] = Field(None, description="Marketing-specific metrics")
    
    # Evaluation matrix used
    evaluation_matrix: EvaluationMatrix = Field(default_factory=EvaluationMatrix)
    
    # Metadata
    evaluation_date: datetime = Field(default_factory=datetime.utcnow)
    duration_seconds: Optional[float] = Field(None, gt=0.0)
    evaluator_notes: Optional[str] = Field(None, description="Additional evaluator notes")
    
    # Recommendations
    strengths: List[str] = Field(default_factory=list, description="Agent strengths")
    weaknesses: List[str] = Field(default_factory=list, description="Areas for improvement")
    recommendations: List[str] = Field(default_factory=list, description="Specific recommendations")
    
    # Cost tracking
    evaluation_cost: Optional[float] = Field(None, ge=0.0, description="Cost of evaluation in USD")
    token_usage: Optional[Dict[str, int]] = Field(None, description="Token usage breakdown")
    
    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True
    )
    
    def get_category_score(self, category: EvaluationCategory) -> Optional[float]:
        """Get score for a specific category."""
        for assessment in self.category_assessments:
            if assessment.category == category:
                return assessment.overall_score
        return None
    
    def calculate_weighted_score(self) -> float:
        """Calculate weighted overall score from category assessments."""
        total_score = 0.0
        total_weight = 0.0
        
        for assessment in self.category_assessments:
            weight = self.evaluation_matrix.get_category_weight(assessment.category)
            total_score += assessment.overall_score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def get_deployment_recommendation(self) -> str:
        """Get deployment recommendation based on overall score."""
        if self.overall_score >= 0.9:
            return "Ready for production deployment"
        elif self.overall_score >= 0.8:
            return "Ready for staging deployment with monitoring"
        elif self.overall_score >= 0.7:
            return "Suitable for testing environment"
        elif self.overall_score >= 0.6:
            return "Requires improvement before deployment"
        else:
            return "Not ready for deployment - significant issues detected"
    
    def get_priority_improvements(self) -> List[str]:
        """Get priority improvements based on lowest scoring categories."""
        sorted_assessments = sorted(
            self.category_assessments,
            key=lambda x: x.overall_score
        )
        
        improvements = []
        for assessment in sorted_assessments[:2]:  # Top 2 lowest scores
            if assessment.overall_score < 0.8:
                improvements.extend(assessment.recommendations)
        
        return improvements[:5]  # Return top 5 recommendations