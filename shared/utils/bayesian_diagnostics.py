"""
Bayesian Diagnostic Utilities for LiftOS MMM
Comprehensive framework for prior-data conflict detection and evidence quantification
"""
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wasserstein_distance
from scipy.integrate import quad
import warnings
from dataclasses import dataclass
import logging

from shared.models.bayesian_priors import (
    PriorSpecification, EvidenceMetrics, ConflictReport, ConflictSeverity,
    PriorUpdateRecommendation, UpdateStrategy, DistributionType
)

logger = logging.getLogger(__name__)


@dataclass
class ConflictAnalysisConfig:
    """Configuration for conflict analysis"""
    bayes_factor_threshold: float = 3.0
    kl_divergence_threshold: float = 0.5
    confidence_level: float = 0.95
    n_bootstrap_samples: int = 1000
    n_prior_predictive_samples: int = 10000
    include_hierarchical_shrinkage: bool = True
    min_effective_sample_size: float = 100.0


class BayesianDiagnostics:
    """Core Bayesian diagnostic utilities"""
    
    def __init__(self, config: Optional[ConflictAnalysisConfig] = None):
        self.config = config or ConflictAnalysisConfig()
        
    def calculate_bayes_factor(
        self, 
        prior_samples: np.ndarray, 
        data_samples: np.ndarray,
        method: str = "savage_dickey"
    ) -> float:
        """
        Calculate Bayes factor comparing data evidence vs prior belief
        
        Args:
            prior_samples: Samples from prior distribution
            data_samples: Samples from data/posterior
            method: Method for BF calculation ('savage_dickey', 'bridge_sampling')
            
        Returns:
            Bayes factor (BF > 1 favors data, BF < 1 favors prior)
        """
        try:
            if method == "savage_dickey":
                return self._savage_dickey_bayes_factor(prior_samples, data_samples)
            elif method == "bridge_sampling":
                return self._bridge_sampling_bayes_factor(prior_samples, data_samples)
            else:
                raise ValueError(f"Unknown method: {method}")
                
        except Exception as e:
            logger.error(f"Error calculating Bayes factor: {e}")
            return 1.0  # Neutral evidence
    
    def _savage_dickey_bayes_factor(
        self, 
        prior_samples: np.ndarray, 
        data_samples: np.ndarray
    ) -> float:
        """Savage-Dickey ratio for Bayes factor calculation"""
        # Estimate densities at the null hypothesis point (typically 0 or prior mean)
        null_point = np.mean(prior_samples)
        
        # Kernel density estimation
        prior_kde = stats.gaussian_kde(prior_samples)
        data_kde = stats.gaussian_kde(data_samples)
        
        # Evaluate densities at null point
        prior_density = prior_kde.evaluate([null_point])[0]
        data_density = data_kde.evaluate([null_point])[0]
        
        # Bayes factor = posterior density / prior density at null
        if prior_density > 0:
            bf = data_density / prior_density
            return max(bf, 1e-6)  # Avoid division by zero
        else:
            return 1.0
    
    def _bridge_sampling_bayes_factor(
        self, 
        prior_samples: np.ndarray, 
        data_samples: np.ndarray
    ) -> float:
        """Bridge sampling approximation for Bayes factor"""
        # Simplified bridge sampling - in practice would use more sophisticated methods
        # This is a placeholder for the full implementation
        
        # Calculate overlap between distributions
        overlap = self.calculate_overlap_coefficient(prior_samples, data_samples)
        
        # Convert overlap to approximate Bayes factor
        # Higher overlap = lower evidence against prior
        if overlap > 0.8:
            return 1.0 + (1 - overlap) * 2  # Weak evidence
        elif overlap > 0.5:
            return 1.0 + (1 - overlap) * 10  # Moderate evidence
        else:
            return 1.0 + (1 - overlap) * 50  # Strong evidence
    
    def calculate_kl_divergence(
        self, 
        prior_samples: np.ndarray, 
        data_samples: np.ndarray,
        n_bins: int = 50
    ) -> float:
        """
        Calculate Kullback-Leibler divergence between prior and data distributions
        
        Args:
            prior_samples: Samples from prior distribution
            data_samples: Samples from data/posterior distribution
            n_bins: Number of bins for histogram estimation
            
        Returns:
            KL divergence D(data || prior)
        """
        try:
            # Determine common range
            all_samples = np.concatenate([prior_samples, data_samples])
            min_val, max_val = np.min(all_samples), np.max(all_samples)
            bins = np.linspace(min_val, max_val, n_bins + 1)
            
            # Create histograms
            prior_hist, _ = np.histogram(prior_samples, bins=bins, density=True)
            data_hist, _ = np.histogram(data_samples, bins=bins, density=True)
            
            # Normalize to probabilities
            bin_width = bins[1] - bins[0]
            prior_prob = prior_hist * bin_width
            data_prob = data_hist * bin_width
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            prior_prob = np.maximum(prior_prob, epsilon)
            data_prob = np.maximum(data_prob, epsilon)
            
            # Calculate KL divergence
            kl_div = np.sum(data_prob * np.log(data_prob / prior_prob))
            
            return max(kl_div, 0.0)  # KL divergence is non-negative
            
        except Exception as e:
            logger.error(f"Error calculating KL divergence: {e}")
            return 0.0
    
    def calculate_wasserstein_distance(
        self, 
        prior_samples: np.ndarray, 
        data_samples: np.ndarray
    ) -> float:
        """
        Calculate Wasserstein (Earth Mover's) distance between distributions
        
        Args:
            prior_samples: Samples from prior distribution
            data_samples: Samples from data/posterior distribution
            
        Returns:
            Wasserstein distance
        """
        try:
            return wasserstein_distance(prior_samples, data_samples)
        except Exception as e:
            logger.error(f"Error calculating Wasserstein distance: {e}")
            return 0.0
    
    def calculate_overlap_coefficient(
        self, 
        prior_samples: np.ndarray, 
        data_samples: np.ndarray,
        n_bins: int = 50
    ) -> float:
        """
        Calculate overlap coefficient between two distributions
        
        Args:
            prior_samples: Samples from prior distribution
            data_samples: Samples from data/posterior distribution
            n_bins: Number of bins for histogram estimation
            
        Returns:
            Overlap coefficient (0 = no overlap, 1 = complete overlap)
        """
        try:
            # Determine common range
            all_samples = np.concatenate([prior_samples, data_samples])
            min_val, max_val = np.min(all_samples), np.max(all_samples)
            bins = np.linspace(min_val, max_val, n_bins + 1)
            
            # Create normalized histograms
            prior_hist, _ = np.histogram(prior_samples, bins=bins, density=True)
            data_hist, _ = np.histogram(data_samples, bins=bins, density=True)
            
            # Normalize to probabilities
            bin_width = bins[1] - bins[0]
            prior_prob = prior_hist * bin_width
            data_prob = data_hist * bin_width
            
            # Calculate overlap coefficient
            overlap = np.sum(np.minimum(prior_prob, data_prob))
            
            return min(overlap, 1.0)  # Ensure it's bounded by 1
            
        except Exception as e:
            logger.error(f"Error calculating overlap coefficient: {e}")
            return 0.0
    
    def calculate_effective_sample_size(
        self, 
        data_samples: np.ndarray,
        autocorr_method: str = "integrated"
    ) -> float:
        """
        Calculate effective sample size accounting for autocorrelation
        
        Args:
            data_samples: MCMC samples or time series data
            autocorr_method: Method for autocorrelation calculation
            
        Returns:
            Effective sample size
        """
        try:
            n = len(data_samples)
            
            if autocorr_method == "integrated":
                # Calculate autocorrelation function
                autocorr = self._calculate_autocorrelation(data_samples)
                
                # Find integrated autocorrelation time
                tau_int = 1 + 2 * np.sum(autocorr[1:])
                
                # Effective sample size
                ess = n / (2 * tau_int + 1)
                
                return max(ess, 1.0)
            else:
                # Simple variance-based ESS
                return float(n)
                
        except Exception as e:
            logger.error(f"Error calculating effective sample size: {e}")
            return float(len(data_samples))
    
    def _calculate_autocorrelation(self, samples: np.ndarray, max_lag: int = None) -> np.ndarray:
        """Calculate autocorrelation function"""
        n = len(samples)
        if max_lag is None:
            max_lag = min(n // 4, 200)  # Reasonable default
        
        # Center the data
        centered = samples - np.mean(samples)
        
        # Calculate autocorrelation using FFT
        autocorr = np.correlate(centered, centered, mode='full')
        autocorr = autocorr[n-1:]  # Take positive lags only
        autocorr = autocorr / autocorr[0]  # Normalize
        
        return autocorr[:max_lag]
    
    def calculate_prior_predictive_pvalue(
        self, 
        prior_spec: PriorSpecification,
        observed_data: np.ndarray,
        n_simulations: int = 1000
    ) -> float:
        """
        Calculate prior predictive p-value
        
        Args:
            prior_spec: Prior specification
            observed_data: Observed data
            n_simulations: Number of prior predictive simulations
            
        Returns:
            Prior predictive p-value
        """
        try:
            # Generate prior predictive samples
            prior_samples = prior_spec.sample(n_simulations)
            
            # Calculate test statistic for observed data
            observed_stat = np.mean(observed_data)
            
            # Calculate test statistic for each prior predictive sample
            # In practice, this would simulate full datasets, not just single values
            prior_stats = prior_samples  # Simplified - would be more complex in practice
            
            # Calculate p-value
            p_value = np.mean(np.abs(prior_stats - observed_stat) >= 
                            np.abs(observed_stat - np.mean(prior_stats)))
            
            return min(max(p_value, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating prior predictive p-value: {e}")
            return 0.5  # Neutral p-value
    
    def interpret_evidence_strength(self, bayes_factor: float) -> str:
        """
        Provide human-readable interpretation of evidence strength
        
        Args:
            bayes_factor: Calculated Bayes factor
            
        Returns:
            Human-readable interpretation
        """
        if bayes_factor < 1.5:
            return "No meaningful evidence against prior belief"
        elif bayes_factor < 3:
            return "Weak evidence against prior belief"
        elif bayes_factor < 10:
            return "Moderate evidence against prior belief"
        elif bayes_factor < 30:
            return "Strong evidence against prior belief"
        else:
            return "Decisive evidence against prior belief"


class ConflictAnalyzer:
    """Analyzer for detecting and quantifying prior-data conflicts"""
    
    def __init__(self, config: Optional[ConflictAnalysisConfig] = None):
        self.config = config or ConflictAnalysisConfig()
        self.diagnostics = BayesianDiagnostics(config)
    
    def analyze_conflicts(
        self,
        priors: List[PriorSpecification],
        data: Dict[str, np.ndarray],
        session_id: str
    ) -> ConflictReport:
        """
        Analyze conflicts between priors and data
        
        Args:
            priors: List of prior specifications
            data: Dictionary mapping parameter names to data samples
            session_id: Associated elicitation session ID
            
        Returns:
            Comprehensive conflict report
        """
        evidence_metrics = []
        high_conflict_params = []
        
        for prior in priors:
            param_name = prior.parameter_name
            
            if param_name not in data:
                logger.warning(f"No data available for parameter {param_name}")
                continue
            
            # Calculate evidence metrics
            evidence = self._calculate_evidence_metrics(prior, data[param_name])
            evidence_metrics.append(evidence)
            
            # Track high conflict parameters
            if evidence.conflict_severity in [ConflictSeverity.STRONG, ConflictSeverity.DECISIVE]:
                high_conflict_params.append(param_name)
        
        # Calculate overall conflict score
        overall_score = self._calculate_overall_conflict_score(evidence_metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(evidence_metrics)
        
        return ConflictReport(
            session_id=session_id,
            data_period=(min(data.keys()), max(data.keys())),  # Simplified
            evidence_metrics=evidence_metrics,
            overall_conflict_score=overall_score,
            high_conflict_parameters=high_conflict_params,
            recommendations=recommendations
        )
    
    def _calculate_evidence_metrics(
        self, 
        prior: PriorSpecification, 
        data_samples: np.ndarray
    ) -> EvidenceMetrics:
        """Calculate comprehensive evidence metrics for a parameter"""
        
        # Generate prior samples
        prior_samples = prior.sample(self.config.n_prior_predictive_samples)
        
        # Calculate all metrics
        bayes_factor = self.diagnostics.calculate_bayes_factor(prior_samples, data_samples)
        kl_divergence = self.diagnostics.calculate_kl_divergence(prior_samples, data_samples)
        wasserstein_dist = self.diagnostics.calculate_wasserstein_distance(prior_samples, data_samples)
        overlap_coeff = self.diagnostics.calculate_overlap_coefficient(prior_samples, data_samples)
        eff_sample_size = self.diagnostics.calculate_effective_sample_size(data_samples)
        prior_pred_pvalue = self.diagnostics.calculate_prior_predictive_pvalue(prior, data_samples)
        
        # Generate interpretation
        interpretation = self.diagnostics.interpret_evidence_strength(bayes_factor)
        
        return EvidenceMetrics(
            parameter_name=prior.parameter_name,
            bayes_factor=bayes_factor,
            kl_divergence=kl_divergence,
            wasserstein_distance=wasserstein_dist,
            overlap_coefficient=overlap_coeff,
            effective_sample_size=eff_sample_size,
            prior_predictive_pvalue=prior_pred_pvalue,
            conflict_severity=ConflictSeverity.NONE,  # Will be auto-classified by validator
            evidence_interpretation=interpretation
        )
    
    def _calculate_overall_conflict_score(self, evidence_metrics: List[EvidenceMetrics]) -> float:
        """Calculate overall conflict score across all parameters"""
        if not evidence_metrics:
            return 0.0
        
        # Weight by conflict severity
        severity_weights = {
            ConflictSeverity.NONE: 0.0,
            ConflictSeverity.WEAK: 0.2,
            ConflictSeverity.MODERATE: 0.5,
            ConflictSeverity.STRONG: 0.8,
            ConflictSeverity.DECISIVE: 1.0
        }
        
        total_weight = 0.0
        weighted_score = 0.0
        
        for evidence in evidence_metrics:
            weight = severity_weights[evidence.conflict_severity]
            total_weight += 1.0
            weighted_score += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _generate_recommendations(self, evidence_metrics: List[EvidenceMetrics]) -> List[str]:
        """Generate actionable recommendations based on evidence"""
        recommendations = []
        
        # Count conflicts by severity
        severity_counts = {}
        for evidence in evidence_metrics:
            severity = evidence.conflict_severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Generate recommendations based on patterns
        if severity_counts.get(ConflictSeverity.DECISIVE, 0) > 0:
            recommendations.append(
                "Immediate review required: Decisive evidence against some priors detected"
            )
            recommendations.append(
                "Consider SBC validation before implementing model updates"
            )
        
        if severity_counts.get(ConflictSeverity.STRONG, 0) > 2:
            recommendations.append(
                "Multiple strong conflicts detected - comprehensive prior review recommended"
            )
        
        if severity_counts.get(ConflictSeverity.MODERATE, 0) > 5:
            recommendations.append(
                "Consider hierarchical modeling to better capture parameter uncertainty"
            )
        
        if not any(severity_counts.get(s, 0) > 0 for s in [
            ConflictSeverity.MODERATE, ConflictSeverity.STRONG, ConflictSeverity.DECISIVE
        ]):
            recommendations.append(
                "Priors appear well-calibrated with data - proceed with confidence"
            )
        
        return recommendations


class PriorUpdater:
    """Generates recommendations for updating priors based on evidence"""
    
    def __init__(self, config: Optional[ConflictAnalysisConfig] = None):
        self.config = config or ConflictAnalysisConfig()
    
    def generate_update_recommendations(
        self,
        conflict_report: ConflictReport,
        original_priors: List[PriorSpecification]
    ) -> List[PriorUpdateRecommendation]:
        """
        Generate prior update recommendations based on conflict analysis
        
        Args:
            conflict_report: Results of conflict analysis
            original_priors: Original prior specifications
            
        Returns:
            List of update recommendations
        """
        recommendations = []
        prior_dict = {p.parameter_name: p for p in original_priors}
        
        for evidence in conflict_report.evidence_metrics:
            param_name = evidence.parameter_name
            
            if param_name not in prior_dict:
                continue
            
            original_prior = prior_dict[param_name]
            
            # Only recommend updates for meaningful conflicts
            if evidence.conflict_severity in [
                ConflictSeverity.MODERATE, 
                ConflictSeverity.STRONG, 
                ConflictSeverity.DECISIVE
            ]:
                recommendation = self._create_update_recommendation(
                    original_prior, evidence
                )
                recommendations.append(recommendation)
        
        # Sort by priority (highest Bayes factor first)
        recommendations.sort(key=lambda x: x.data_evidence.bayes_factor, reverse=True)
        
        # Assign priorities
        for i, rec in enumerate(recommendations):
            rec.implementation_priority = min(i + 1, 5)
        
        return recommendations
    
    def _create_update_recommendation(
        self,
        original_prior: PriorSpecification,
        evidence: EvidenceMetrics
    ) -> PriorUpdateRecommendation:
        """Create a specific update recommendation"""
        
        # Determine update strategy based on evidence strength and client confidence
        strategy = self._determine_update_strategy(original_prior, evidence)
        
        # Generate updated prior
        updated_prior = self._generate_updated_prior(original_prior, evidence, strategy)
        
        # Calculate confidence in update
        confidence = self._calculate_update_confidence(evidence, original_prior.confidence_level)
        
        # Generate business impact assessment
        business_impact = self._assess_business_impact(original_prior, evidence)
        
        # Generate explanation
        explanation = self._generate_explanation(original_prior, evidence, strategy)
        
        return PriorUpdateRecommendation(
            parameter_name=original_prior.parameter_name,
            original_prior=original_prior,
            data_evidence=evidence,
            recommended_update=updated_prior,
            update_strategy=strategy,
            confidence_in_update=confidence,
            business_impact=business_impact,
            explanation=explanation,
            implementation_priority=1  # Will be set later
        )
    
    def _determine_update_strategy(
        self, 
        prior: PriorSpecification, 
        evidence: EvidenceMetrics
    ) -> UpdateStrategy:
        """Determine appropriate update strategy"""
        
        # High client confidence + weak evidence = conservative
        if prior.confidence_level > 0.8 and evidence.conflict_severity == ConflictSeverity.MODERATE:
            return UpdateStrategy.CONSERVATIVE
        
        # Low client confidence + strong evidence = aggressive
        if prior.confidence_level < 0.5 and evidence.conflict_severity in [
            ConflictSeverity.STRONG, ConflictSeverity.DECISIVE
        ]:
            return UpdateStrategy.AGGRESSIVE
        
        # Decisive evidence regardless of confidence = aggressive
        if evidence.conflict_severity == ConflictSeverity.DECISIVE:
            return UpdateStrategy.AGGRESSIVE
        
        # Default to moderate
        return UpdateStrategy.MODERATE
    
    def _generate_updated_prior(
        self,
        original_prior: PriorSpecification,
        evidence: EvidenceMetrics,
        strategy: UpdateStrategy
    ) -> PriorSpecification:
        """Generate updated prior specification"""
        
        # Strategy-dependent update factors
        update_factors = {
            UpdateStrategy.CONSERVATIVE: 0.2,
            UpdateStrategy.MODERATE: 0.5,
            UpdateStrategy.AGGRESSIVE: 0.8,
            UpdateStrategy.HIERARCHICAL: 0.3
        }
        
        update_factor = update_factors[strategy]
        
        # For normal distributions, update mean towards data
        if original_prior.distribution_type == DistributionType.NORMAL:
            original_mean = original_prior.hyperparameters['loc']
            original_scale = original_prior.hyperparameters['scale']
            
            # Simplified update - in practice would use proper Bayesian updating
            # This is a placeholder for demonstration
            data_mean = original_mean + (evidence.bayes_factor - 1) * original_scale * 0.1
            updated_mean = original_mean + update_factor * (data_mean - original_mean)
            
            # Slightly reduce uncertainty after seeing data
            updated_scale = original_scale * (1 - update_factor * 0.1)
            
            updated_hyperparams = {
                'loc': updated_mean,
                'scale': max(updated_scale, original_scale * 0.5)  # Don't reduce too much
            }
        else:
            # For other distributions, keep original (could be extended)
            updated_hyperparams = original_prior.hyperparameters.copy()
        
        return PriorSpecification(
            parameter_name=original_prior.parameter_name,
            parameter_category=original_prior.parameter_category,
            distribution_type=original_prior.distribution_type,
            hyperparameters=updated_hyperparams,
            confidence_level=min(original_prior.confidence_level + 0.1, 1.0),
            domain_rationale=f"Updated based on data evidence: {original_prior.domain_rationale}",
            data_sources=original_prior.data_sources + ["empirical_data_update"],
            elicitation_method=original_prior.elicitation_method,
            created_by=f"bayesian_update_system"
        )
    
    def _calculate_update_confidence(
        self, 
        evidence: EvidenceMetrics, 
        client_confidence: float
    ) -> float:
        """Calculate confidence in the update recommendation"""
        
        # Base confidence on evidence strength
        evidence_confidence = {
            ConflictSeverity.NONE: 0.1,
            ConflictSeverity.WEAK: 0.3,
            ConflictSeverity.MODERATE: 0.6,
            ConflictSeverity.STRONG: 0.8,
            ConflictSeverity.DECISIVE: 0.95
        }
        
        base_confidence = evidence_confidence[evidence.conflict_severity]
        
        # Adjust based on effective sample size
        ess_factor = min(evidence.effective_sample_size / 1000.0, 1.0)
        
        # Adjust based on overlap (higher overlap = lower confidence in update)
        overlap_factor = 1.0 - evidence.overlap_coefficient
        
        final_confidence = base_confidence * ess_factor * overlap_factor
        
        return min(max(final_confidence, 0.1), 0.95)
    
    def _assess_business_impact(
        self, 
        prior: PriorSpecification, 
        evidence: EvidenceMetrics
    ) -> str:
        """Assess potential business impact of the update"""
        
        if evidence.conflict_severity == ConflictSeverity.DECISIVE:
            if prior.parameter_category in [
                "media_effect", "saturation"
            ]:
                return "High impact: May significantly affect media budget allocation and ROI estimates"
            else:
                return "Medium impact: May affect model accuracy and business insights"
        
        elif evidence.conflict_severity == ConflictSeverity.STRONG:
            return "Medium impact: Likely to improve model performance and decision quality"
        
        elif evidence.conflict_severity == ConflictSeverity.MODERATE:
            return "Low to medium impact: May provide incremental improvements"
        
        else:
            return "Low impact: Minor adjustment to improve calibration"
    
    def _generate_explanation(
        self,
        prior: PriorSpecification,
        evidence: EvidenceMetrics,
        strategy: UpdateStrategy
    ) -> str:
        """Generate human-readable explanation for the update"""
        
        explanation = f"Analysis of 24-month data shows {evidence.evidence_interpretation.lower()}. "
        
        explanation += f"The Bayes factor of {evidence.bayes_factor:.2f} indicates "
        
        if evidence.bayes_factor > 10:
            explanation += "strong statistical evidence that the data contradicts the original prior belief. "
        elif evidence.bayes_factor > 3:
            explanation += "moderate statistical evidence against the original prior belief. "
        else:
            explanation += "weak evidence against the original prior belief. "
        
        explanation += f"Given your confidence level of {prior.confidence_level:.0%} in the original prior, "
        explanation += f"we recommend a {strategy.value} update strategy. "
        
        if evidence.effective_sample_size > 500:
            explanation += "The large effective sample size provides reliable evidence for this recommendation."
        else:
            explanation += "The limited sample size suggests caution in implementing this update."
        
        return explanation