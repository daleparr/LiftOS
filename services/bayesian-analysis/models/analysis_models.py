"""
Analysis Models for Paid Tier Bayesian Analysis

Pydantic models for advanced Bayesian analysis features including comprehensive SBC,
advanced prior updating, and evidence assessment.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

class ModelComplexity(str, Enum):
    """Model complexity levels"""
    SIMPLE = "simple"      # <5 parameters
    MODERATE = "moderate"  # 5-15 parameters
    COMPLEX = "complex"    # 15-50 parameters
    VERY_COMPLEX = "very_complex"  # >50 parameters

class UpdateStrategy(str, Enum):
    """Prior updating strategies"""
    CONSERVATIVE = "conservative"  # Minimal updates, high evidence threshold
    BALANCED = "balanced"         # Standard Bayesian updating
    AGGRESSIVE = "aggressive"     # Quick adaptation to new evidence
    EVIDENCE_WEIGHTED = "evidence_weighted"  # Weight by evidence quality

class ValidationLevel(str, Enum):
    """SBC validation levels"""
    BASIC = "basic"           # Standard rank statistics
    STANDARD = "standard"     # Rank stats + coverage analysis
    COMPREHENSIVE = "comprehensive"  # Full diagnostic suite
    RESEARCH_GRADE = "research_grade"  # Publication-quality validation

class EvidenceType(str, Enum):
    """Types of evidence for assessment"""
    OBSERVATIONAL = "observational"
    EXPERIMENTAL = "experimental"
    HISTORICAL = "historical"
    EXPERT_OPINION = "expert_opinion"
    LITERATURE = "literature"

# Model definitions for requests

class BayesianModel(BaseModel):
    """Bayesian model specification for analysis"""
    id: str
    name: str
    description: Optional[str] = None
    parameters: List[Dict[str, Any]]
    priors: List[Dict[str, Any]]
    likelihood: Dict[str, Any]
    complexity: ModelComplexity
    created_at: datetime
    last_updated: datetime

class SBCConfiguration(BaseModel):
    """Configuration for SBC validation"""
    num_simulations: int = Field(1000, ge=100, le=10000)
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    include_diagnostics: bool = True
    include_plots: bool = True
    include_recommendations: bool = True
    parallel_execution: bool = True
    random_seed: Optional[int] = None

class ComprehensiveSBCRequest(BaseModel):
    """Request for comprehensive SBC validation"""
    model: BayesianModel
    configuration: SBCConfiguration
    business_context: Optional[Dict[str, Any]] = None
    performance_requirements: Optional[Dict[str, Any]] = None
    custom_diagnostics: Optional[List[str]] = None

class Evidence(BaseModel):
    """Evidence item for prior updating"""
    id: str
    type: EvidenceType
    source: str
    data: Dict[str, Any]
    quality_score: float = Field(ge=0, le=1)
    reliability: float = Field(ge=0, le=1)
    relevance: float = Field(ge=0, le=1)
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

class PriorSpecification(BaseModel):
    """Prior specification for updating"""
    parameter_name: str
    distribution_type: str  # "normal", "beta", "gamma", etc.
    parameters: Dict[str, float]
    confidence: float = Field(ge=0, le=1)
    source: str
    elicitation_method: Optional[str] = None

class UpdateConfiguration(BaseModel):
    """Configuration for prior updating"""
    strategy: UpdateStrategy = UpdateStrategy.BALANCED
    confidence_threshold: float = Field(0.95, ge=0.5, le=0.99)
    evidence_weight_method: str = "quality_weighted"
    include_uncertainty_quantification: bool = True
    include_sensitivity_analysis: bool = True
    max_update_magnitude: Optional[float] = None

class AdvancedPriorUpdateRequest(BaseModel):
    """Request for advanced prior updating"""
    priors: List[PriorSpecification]
    evidence: List[Evidence]
    configuration: UpdateConfiguration
    current_workflow: Optional[Dict[str, Any]] = None
    business_constraints: Optional[Dict[str, Any]] = None

class EvidenceAssessmentRequest(BaseModel):
    """Request for evidence strength assessment"""
    evidence: List[Evidence]
    priors: List[PriorSpecification]
    assessment_type: str = "comprehensive"
    include_robustness_analysis: bool = True
    include_recommendations: bool = True
    custom_metrics: Optional[List[str]] = None

# Response models

class RankStatistics(BaseModel):
    """SBC rank statistics results"""
    uniformity_test: Dict[str, float]
    coverage_probability: Dict[str, float]
    calibration_score: float
    rank_histogram: Dict[str, Any]
    p_values: Dict[str, float]

class DiagnosticAnalysis(BaseModel):
    """SBC diagnostic analysis results"""
    rank_histogram: Dict[str, Any]
    coverage_plots: Dict[str, Any]
    shrinkage_analysis: Dict[str, Any]
    parameter_recovery: Dict[str, Any]
    convergence_diagnostics: Dict[str, Any]

class SBCRecommendations(BaseModel):
    """SBC-based recommendations"""
    model_improvements: List[str]
    prior_adjustments: List[Dict[str, Any]]
    validation_schedule: str
    risk_mitigation: List[str]
    reliability_gain: float
    confidence_improvement: float
    risk_reduction: float

class ComprehensiveSBCResponse(BaseModel):
    """Response from comprehensive SBC analysis"""
    validation_id: str
    model_id: str
    validation_status: str
    reliability_score: float
    rank_statistics: RankStatistics
    diagnostic_analysis: DiagnosticAnalysis
    recommendations: SBCRecommendations
    business_impact: Dict[str, Any]
    execution_time: float
    generated_at: datetime

class EvidenceMetrics(BaseModel):
    """Evidence strength metrics"""
    bayes_factors: Dict[str, float]
    kl_divergences: Dict[str, float]
    wasserstein_distances: Dict[str, float]
    evidence_weights: Dict[str, float]
    reliability_scores: Dict[str, float]

class UncertaintyQuantification(BaseModel):
    """Uncertainty quantification results"""
    posterior: Dict[str, Any]
    prediction_intervals: Dict[str, Any]
    sensitivity: Dict[str, Any]
    robustness: Dict[str, Any]

class UpdateReasoning(BaseModel):
    """Reasoning for prior updates"""
    statistical: List[str]
    business: List[str]
    risk: List[str]
    evidence_summary: str

class IntegrationAdvice(BaseModel):
    """Integration recommendations"""
    implementation_steps: List[Dict[str, str]]
    validation_points: List[str]
    monitoring_plan: Dict[str, Any]
    rollback_strategy: List[str]

class AdvancedPriorUpdateResponse(BaseModel):
    """Response from advanced prior updating"""
    update_id: str
    updated_priors: List[PriorSpecification]
    evidence_analysis: Dict[str, Any]
    uncertainty_quantification: UncertaintyQuantification
    update_justification: UpdateReasoning
    integration_recommendations: IntegrationAdvice
    business_impact: Dict[str, Any]
    generated_at: datetime

class RobustnessAnalysis(BaseModel):
    """Robustness analysis results"""
    outlier_sensitivity: Dict[str, float]
    assumption_dependence: Dict[str, float]
    data_quality_impact: Dict[str, float]
    stability_metrics: Dict[str, float]

class EvidenceRecommendations(BaseModel):
    """Recommendations for evidence improvement"""
    evidence_improvements: List[str]
    data_collection: List[str]
    analysis_refinements: List[str]
    quality_enhancements: List[str]

class EvidenceAssessmentResponse(BaseModel):
    """Response from evidence assessment"""
    assessment_id: str
    overall_strength: float
    evidence_breakdown: Dict[str, Any]
    statistical_metrics: EvidenceMetrics
    robustness_analysis: RobustnessAnalysis
    recommendations: EvidenceRecommendations
    quality_score: float
    generated_at: datetime

# Workflow and integration models

class WorkflowStep(BaseModel):
    """Step in analysis workflow"""
    step_id: str
    name: str
    description: str
    inputs: List[str]
    outputs: List[str]
    dependencies: List[str]
    estimated_duration: str
    status: str = "pending"

class AnalysisWorkflow(BaseModel):
    """Complete analysis workflow"""
    workflow_id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    current_step: str
    progress: float = Field(ge=0, le=1)
    estimated_completion: datetime
    created_at: datetime

class IntegrationEndpoint(BaseModel):
    """Integration endpoint specification"""
    service: str
    endpoint: str
    method: str
    parameters: Dict[str, Any]
    authentication: Optional[Dict[str, str]] = None

class CrossServiceRequest(BaseModel):
    """Request for cross-service integration"""
    source_service: str
    target_service: str
    operation: str
    data: Dict[str, Any]
    callback_url: Optional[str] = None
    timeout: int = 30

# Error and status models

class AnalysisError(BaseModel):
    """Error in analysis operations"""
    error_type: str
    message: str
    details: Dict[str, Any]
    recovery_suggestions: List[str]
    support_contact: str

class AnalysisStatus(BaseModel):
    """Status of analysis operations"""
    operation_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: float = Field(ge=0, le=1)
    estimated_completion: Optional[datetime] = None
    current_step: Optional[str] = None
    messages: List[str] = []

# Configuration models

class ServiceConfiguration(BaseModel):
    """Service configuration for analysis"""
    max_concurrent_analyses: int = 5
    default_timeout: int = 300
    cache_results: bool = True
    enable_monitoring: bool = True
    log_level: str = "INFO"

class PerformanceMetrics(BaseModel):
    """Performance metrics for analysis"""
    execution_time: float
    memory_usage: float
    cpu_usage: float
    cache_hit_rate: float
    error_rate: float
    throughput: float