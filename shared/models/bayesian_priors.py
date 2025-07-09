"""
Bayesian Prior Specification and Management for LiftOS MMM
Comprehensive framework for prior elicitation, conflict detection, and updating
"""
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
import numpy as np
from scipy import stats
import json


class DistributionType(str, Enum):
    """Supported prior distribution types"""
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    BETA = "beta"
    GAMMA = "gamma"
    UNIFORM = "uniform"
    EXPONENTIAL = "exponential"
    STUDENT_T = "student_t"
    HALF_NORMAL = "half_normal"
    HALF_CAUCHY = "half_cauchy"


class ParameterCategory(str, Enum):
    """MMM parameter categories"""
    MEDIA_EFFECT = "media_effect"
    SATURATION = "saturation"
    ADSTOCK = "adstock"
    BASELINE = "baseline"
    SEASONALITY = "seasonality"
    TREND = "trend"
    EXTERNAL_FACTOR = "external_factor"
    INTERACTION = "interaction"


class ElicitationMethod(str, Enum):
    """Methods for prior elicitation"""
    EXPERT_JUDGMENT = "expert_judgment"
    HISTORICAL_ANALOGY = "historical_analogy"
    INDUSTRY_BENCHMARK = "industry_benchmark"
    PREVIOUS_STUDY = "previous_study"
    THEORETICAL_CONSTRAINT = "theoretical_constraint"
    REGULATORY_REQUIREMENT = "regulatory_requirement"


class ConflictSeverity(str, Enum):
    """Severity levels for prior-data conflicts"""
    NONE = "none"           # BF < 1.5
    WEAK = "weak"           # 1.5 <= BF < 3
    MODERATE = "moderate"   # 3 <= BF < 10
    STRONG = "strong"       # 10 <= BF < 30
    DECISIVE = "decisive"   # BF >= 30


class UpdateStrategy(str, Enum):
    """Strategies for prior updating"""
    CONSERVATIVE = "conservative"  # Minimal update, respect client confidence
    MODERATE = "moderate"         # Balanced update based on evidence
    AGGRESSIVE = "aggressive"     # Strong update when evidence is decisive
    HIERARCHICAL = "hierarchical" # Use hierarchical structure for updating


class PriorSpecification(BaseModel):
    """Structured prior specification for MMM parameters"""
    parameter_name: str = Field(..., description="Name of the MMM parameter")
    parameter_category: ParameterCategory = Field(..., description="Category of parameter")
    distribution_type: DistributionType = Field(..., description="Type of prior distribution")
    hyperparameters: Dict[str, float] = Field(..., description="Distribution hyperparameters")
    confidence_level: float = Field(..., ge=0.0, le=1.0, description="Client's confidence in this prior")
    domain_rationale: str = Field(..., description="Business rationale for this prior")
    data_sources: List[str] = Field(default=[], description="Sources that informed this belief")
    elicitation_method: ElicitationMethod = Field(..., description="How this prior was elicited")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = Field(..., description="Who specified this prior")
    
    @validator('hyperparameters')
    def validate_hyperparameters(cls, v, values):
        """Validate hyperparameters match distribution type"""
        dist_type = values.get('distribution_type')
        if not dist_type:
            return v
            
        required_params = {
            DistributionType.NORMAL: ['loc', 'scale'],
            DistributionType.LOGNORMAL: ['s', 'scale'],
            DistributionType.BETA: ['a', 'b'],
            DistributionType.GAMMA: ['a', 'scale'],
            DistributionType.UNIFORM: ['loc', 'scale'],
            DistributionType.EXPONENTIAL: ['scale'],
            DistributionType.STUDENT_T: ['df', 'loc', 'scale'],
            DistributionType.HALF_NORMAL: ['scale'],
            DistributionType.HALF_CAUCHY: ['scale']
        }
        
        if dist_type in required_params:
            missing = set(required_params[dist_type]) - set(v.keys())
            if missing:
                raise ValueError(f"Missing hyperparameters for {dist_type}: {missing}")
        
        return v
    
    def to_scipy_distribution(self):
        """Convert to scipy distribution object"""
        dist_map = {
            DistributionType.NORMAL: stats.norm,
            DistributionType.LOGNORMAL: stats.lognorm,
            DistributionType.BETA: stats.beta,
            DistributionType.GAMMA: stats.gamma,
            DistributionType.UNIFORM: stats.uniform,
            DistributionType.EXPONENTIAL: stats.expon,
            DistributionType.STUDENT_T: stats.t,
            DistributionType.HALF_NORMAL: stats.halfnorm,
            DistributionType.HALF_CAUCHY: stats.halfcauchy
        }
        
        dist_class = dist_map[self.distribution_type]
        return dist_class(**self.hyperparameters)
    
    def sample(self, n_samples: int = 1000) -> np.ndarray:
        """Generate samples from the prior distribution"""
        return self.to_scipy_distribution().rvs(size=n_samples)
    
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate probability density function"""
        return self.to_scipy_distribution().pdf(x)
    
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate cumulative distribution function"""
        return self.to_scipy_distribution().cdf(x)


class PriorElicitationSession(BaseModel):
    """Complete prior elicitation session with client"""
    session_id: str = Field(..., description="Unique session identifier")
    client_id: str = Field(..., description="Client organization identifier")
    session_date: datetime = Field(default_factory=datetime.utcnow)
    facilitator: str = Field(..., description="Who facilitated the session")
    participants: List[str] = Field(..., description="Session participants")
    priors: List[PriorSpecification] = Field(..., description="Elicited priors")
    elicitation_method: ElicitationMethod = Field(..., description="Primary elicitation method")
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence in priors")
    session_notes: str = Field(default="", description="Additional session notes")
    validation_status: str = Field(default="pending", description="Validation status")
    
    def get_priors_by_category(self, category: ParameterCategory) -> List[PriorSpecification]:
        """Get priors filtered by parameter category"""
        return [p for p in self.priors if p.parameter_category == category]
    
    def get_high_confidence_priors(self, threshold: float = 0.8) -> List[PriorSpecification]:
        """Get priors with high client confidence"""
        return [p for p in self.priors if p.confidence_level >= threshold]
    
    def get_low_confidence_priors(self, threshold: float = 0.5) -> List[PriorSpecification]:
        """Get priors with low client confidence"""
        return [p for p in self.priors if p.confidence_level <= threshold]


class EvidenceMetrics(BaseModel):
    """Metrics quantifying evidence strength against priors"""
    parameter_name: str = Field(..., description="Parameter being evaluated")
    bayes_factor: float = Field(..., description="Bayes factor (data vs prior)")
    kl_divergence: float = Field(..., description="KL divergence between prior and posterior")
    wasserstein_distance: float = Field(..., description="Wasserstein distance")
    overlap_coefficient: float = Field(..., description="Distribution overlap coefficient")
    effective_sample_size: float = Field(..., description="Effective sample size from data")
    prior_predictive_pvalue: float = Field(..., description="Prior predictive p-value")
    conflict_severity: ConflictSeverity = Field(..., description="Conflict severity classification")
    evidence_interpretation: str = Field(..., description="Human-readable interpretation")
    
    @validator('conflict_severity', pre=True, always=True)
    def classify_conflict_severity(cls, v, values):
        """Automatically classify conflict severity based on Bayes factor"""
        bf = values.get('bayes_factor', 1.0)
        
        if bf < 1.5:
            return ConflictSeverity.NONE
        elif bf < 3:
            return ConflictSeverity.WEAK
        elif bf < 10:
            return ConflictSeverity.MODERATE
        elif bf < 30:
            return ConflictSeverity.STRONG
        else:
            return ConflictSeverity.DECISIVE


class ConflictReport(BaseModel):
    """Comprehensive report on prior-data conflicts"""
    session_id: str = Field(..., description="Associated elicitation session")
    analysis_date: datetime = Field(default_factory=datetime.utcnow)
    data_period: Tuple[datetime, datetime] = Field(..., description="Data period analyzed")
    evidence_metrics: List[EvidenceMetrics] = Field(..., description="Evidence for each parameter")
    overall_conflict_score: float = Field(..., description="Overall conflict severity score")
    high_conflict_parameters: List[str] = Field(..., description="Parameters with strong conflicts")
    recommendations: List[str] = Field(..., description="Recommended actions")
    
    def get_conflicts_by_severity(self, severity: ConflictSeverity) -> List[EvidenceMetrics]:
        """Get conflicts filtered by severity level"""
        return [e for e in self.evidence_metrics if e.conflict_severity == severity]
    
    def get_most_conflicted_parameters(self, n: int = 5) -> List[EvidenceMetrics]:
        """Get the most conflicted parameters"""
        return sorted(self.evidence_metrics, key=lambda x: x.bayes_factor, reverse=True)[:n]


class PriorUpdateRecommendation(BaseModel):
    """Recommendation for updating client beliefs"""
    parameter_name: str = Field(..., description="Parameter to update")
    original_prior: PriorSpecification = Field(..., description="Original client prior")
    data_evidence: EvidenceMetrics = Field(..., description="Evidence from data")
    recommended_update: PriorSpecification = Field(..., description="Recommended updated prior")
    update_strategy: UpdateStrategy = Field(..., description="Recommended update strategy")
    confidence_in_update: float = Field(..., ge=0.0, le=1.0, description="Confidence in recommendation")
    business_impact: str = Field(..., description="Expected business impact")
    explanation: str = Field(..., description="Detailed explanation for client")
    supporting_analysis: Dict[str, Any] = Field(default={}, description="Supporting statistical analysis")
    implementation_priority: int = Field(..., ge=1, le=5, description="Implementation priority (1=highest)")
    
    def get_update_summary(self) -> Dict[str, Any]:
        """Get summary of the recommended update"""
        return {
            "parameter": self.parameter_name,
            "conflict_severity": self.data_evidence.conflict_severity,
            "bayes_factor": self.data_evidence.bayes_factor,
            "strategy": self.update_strategy,
            "priority": self.implementation_priority,
            "business_impact": self.business_impact
        }


class HierarchicalPriorStructure(BaseModel):
    """Hierarchical prior structure for MMM parameters"""
    structure_id: str = Field(..., description="Unique structure identifier")
    industry_category: str = Field(..., description="Industry category")
    hyperpriors: Dict[str, PriorSpecification] = Field(..., description="Industry-level hyperpriors")
    client_priors: Dict[str, PriorSpecification] = Field(..., description="Client-specific priors")
    parameter_hierarchy: Dict[str, List[str]] = Field(..., description="Parameter hierarchy mapping")
    shrinkage_parameters: Dict[str, float] = Field(default={}, description="Shrinkage parameters")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def get_effective_prior(self, parameter_name: str, shrinkage: float = 0.5) -> PriorSpecification:
        """Get effective prior combining hyperprior and client prior"""
        if parameter_name not in self.client_priors:
            raise ValueError(f"Parameter {parameter_name} not found in client priors")
        
        client_prior = self.client_priors[parameter_name]
        
        # Find corresponding hyperprior
        hyperprior = None
        for category, params in self.parameter_hierarchy.items():
            if parameter_name in params and category in self.hyperpriors:
                hyperprior = self.hyperpriors[category]
                break
        
        if not hyperprior:
            return client_prior
        
        # Combine priors using shrinkage
        # This is a simplified combination - in practice would use proper Bayesian updating
        return self._combine_priors(client_prior, hyperprior, shrinkage)
    
    def _combine_priors(
        self, 
        client_prior: PriorSpecification, 
        hyperprior: PriorSpecification, 
        shrinkage: float
    ) -> PriorSpecification:
        """Combine client and hyperprior using shrinkage"""
        # Simplified combination for normal distributions
        if (client_prior.distribution_type == DistributionType.NORMAL and 
            hyperprior.distribution_type == DistributionType.NORMAL):
            
            client_mean = client_prior.hyperparameters['loc']
            client_var = client_prior.hyperparameters['scale'] ** 2
            hyper_mean = hyperprior.hyperparameters['loc']
            hyper_var = hyperprior.hyperparameters['scale'] ** 2
            
            # Precision-weighted combination
            client_precision = 1 / client_var
            hyper_precision = 1 / hyper_var
            
            combined_precision = (1 - shrinkage) * client_precision + shrinkage * hyper_precision
            combined_mean = ((1 - shrinkage) * client_precision * client_mean + 
                           shrinkage * hyper_precision * hyper_mean) / combined_precision
            combined_var = 1 / combined_precision
            
            combined_hyperparams = {
                'loc': combined_mean,
                'scale': np.sqrt(combined_var)
            }
            
            return PriorSpecification(
                parameter_name=client_prior.parameter_name,
                parameter_category=client_prior.parameter_category,
                distribution_type=DistributionType.NORMAL,
                hyperparameters=combined_hyperparams,
                confidence_level=min(client_prior.confidence_level + 0.1, 1.0),
                domain_rationale=f"Hierarchical combination: {client_prior.domain_rationale}",
                elicitation_method=ElicitationMethod.THEORETICAL_CONSTRAINT,
                created_by="hierarchical_system"
            )
        
        # For other distributions, return client prior (could be extended)
        return client_prior


class BayesianUpdateSession(BaseModel):
    """Session for updating priors based on data evidence"""
    update_session_id: str = Field(..., description="Unique update session identifier")
    original_session_id: str = Field(..., description="Original elicitation session")
    client_id: str = Field(..., description="Client identifier")
    update_date: datetime = Field(default_factory=datetime.utcnow)
    conflict_report: ConflictReport = Field(..., description="Prior-data conflict analysis")
    update_recommendations: List[PriorUpdateRecommendation] = Field(..., description="Update recommendations")
    client_responses: Dict[str, str] = Field(default={}, description="Client responses to recommendations")
    final_updated_priors: List[PriorSpecification] = Field(default=[], description="Final updated priors")
    update_summary: Dict[str, Any] = Field(default={}, description="Summary of updates made")
    
    def get_accepted_updates(self) -> List[PriorUpdateRecommendation]:
        """Get updates that were accepted by client"""
        return [rec for rec in self.update_recommendations 
                if self.client_responses.get(rec.parameter_name) == "accepted"]
    
    def get_rejected_updates(self) -> List[PriorUpdateRecommendation]:
        """Get updates that were rejected by client"""
        return [rec for rec in self.update_recommendations 
                if self.client_responses.get(rec.parameter_name) == "rejected"]
    
    def calculate_update_impact(self) -> Dict[str, float]:
        """Calculate the impact of accepted updates"""
        accepted = self.get_accepted_updates()
        if not accepted:
            return {"total_parameters_updated": 0, "average_bayes_factor": 0}
        
        total_updated = len(accepted)
        avg_bayes_factor = np.mean([rec.data_evidence.bayes_factor for rec in accepted])
        
        return {
            "total_parameters_updated": total_updated,
            "average_bayes_factor": avg_bayes_factor,
            "high_impact_updates": len([rec for rec in accepted if rec.data_evidence.bayes_factor > 10])
        }