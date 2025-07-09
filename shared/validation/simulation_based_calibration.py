"""
Simulation Based Calibration (SBC) Framework for LiftOS MMM
Comprehensive framework for validating Bayesian model calibration
"""
from typing import Dict, List, Any, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime

from shared.models.bayesian_priors import PriorSpecification, DistributionType
from shared.utils.bayesian_diagnostics import BayesianDiagnostics

logger = logging.getLogger(__name__)


@dataclass
class SBCConfig:
    """Configuration for SBC validation"""
    n_simulations: int = 1000
    n_posterior_samples: int = 1000
    confidence_level: float = 0.95
    rank_statistic_bins: int = 20
    coverage_tolerance: float = 0.05
    uniformity_test_alpha: float = 0.05
    parallel_execution: bool = True
    max_workers: int = 4
    random_seed: Optional[int] = None
    save_diagnostics: bool = True


@dataclass
class SBCResult:
    """Results from SBC validation"""
    validation_id: str
    passed: bool
    calibration_quality: float
    coverage_probability: float
    rank_statistics: Dict[str, np.ndarray]
    uniformity_test_results: Dict[str, float]
    coverage_test_results: Dict[str, float]
    diagnostics: Dict[str, Any]
    execution_time_seconds: float
    failure_reasons: List[str]
    recommendations: List[str]
    simulation_summary: Dict[str, Any]


class SBCValidator:
    """Core SBC validation engine"""
    
    def __init__(self, config: Optional[SBCConfig] = None):
        self.config = config or SBCConfig()
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
    
    def validate_model_calibration(
        self,
        model_function: Callable,
        priors: List[PriorSpecification],
        data_generating_process: Callable,
        model_id: str,
        **kwargs
    ) -> SBCResult:
        """
        Perform comprehensive SBC validation
        
        Args:
            model_function: Function that fits model and returns posterior samples
            priors: List of prior specifications
            data_generating_process: Function that generates synthetic data
            model_id: Unique identifier for the model
            **kwargs: Additional arguments for model and data generation
            
        Returns:
            Comprehensive SBC validation results
        """
        start_time = time.time()
        validation_id = f"sbc_{model_id}_{int(start_time)}"
        
        logger.info(f"Starting SBC validation {validation_id} with {self.config.n_simulations} simulations")
        
        try:
            # Run SBC simulations
            simulation_results = self._run_sbc_simulations(
                model_function, priors, data_generating_process, **kwargs
            )
            
            # Calculate rank statistics
            rank_statistics = self._calculate_rank_statistics(simulation_results)
            
            # Test uniformity of ranks
            uniformity_results = self._test_rank_uniformity(rank_statistics)
            
            # Test coverage probability
            coverage_results = self._test_coverage_probability(simulation_results)
            
            # Calculate overall calibration quality
            calibration_quality = self._calculate_calibration_quality(
                uniformity_results, coverage_results
            )
            
            # Determine if validation passed
            passed, failure_reasons = self._evaluate_validation_results(
                uniformity_results, coverage_results, calibration_quality
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                passed, failure_reasons, uniformity_results, coverage_results
            )
            
            # Compile diagnostics
            diagnostics = self._compile_diagnostics(
                simulation_results, rank_statistics, uniformity_results, coverage_results
            )
            
            execution_time = time.time() - start_time
            
            result = SBCResult(
                validation_id=validation_id,
                passed=passed,
                calibration_quality=calibration_quality,
                coverage_probability=coverage_results.get('empirical_coverage', 0.0),
                rank_statistics=rank_statistics,
                uniformity_test_results=uniformity_results,
                coverage_test_results=coverage_results,
                diagnostics=diagnostics,
                execution_time_seconds=execution_time,
                failure_reasons=failure_reasons,
                recommendations=recommendations,
                simulation_summary=self._create_simulation_summary(simulation_results)
            )
            
            logger.info(f"SBC validation {validation_id} completed in {execution_time:.2f}s. Passed: {passed}")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"SBC validation {validation_id} failed: {e}")
            
            return SBCResult(
                validation_id=validation_id,
                passed=False,
                calibration_quality=0.0,
                coverage_probability=0.0,
                rank_statistics={},
                uniformity_test_results={},
                coverage_test_results={},
                diagnostics={'error': str(e)},
                execution_time_seconds=execution_time,
                failure_reasons=[f"Validation failed with error: {e}"],
                recommendations=["Review model implementation and data generation process"],
                simulation_summary={}
            )
    
    def _run_sbc_simulations(
        self,
        model_function: Callable,
        priors: List[PriorSpecification],
        data_generating_process: Callable,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Run SBC simulations in parallel or sequential"""
        
        if self.config.parallel_execution:
            return self._run_parallel_simulations(
                model_function, priors, data_generating_process, **kwargs
            )
        else:
            return self._run_sequential_simulations(
                model_function, priors, data_generating_process, **kwargs
            )
    
    def _run_parallel_simulations(
        self,
        model_function: Callable,
        priors: List[PriorSpecification],
        data_generating_process: Callable,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Run simulations in parallel"""
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all simulation tasks
            future_to_sim = {
                executor.submit(
                    self._run_single_simulation,
                    i, model_function, priors, data_generating_process, **kwargs
                ): i for i in range(self.config.n_simulations)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_sim):
                sim_id = future_to_sim[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if len(results) % 100 == 0:
                        logger.info(f"Completed {len(results)}/{self.config.n_simulations} simulations")
                        
                except Exception as e:
                    logger.error(f"Simulation {sim_id} failed: {e}")
        
        return results
    
    def _run_sequential_simulations(
        self,
        model_function: Callable,
        priors: List[PriorSpecification],
        data_generating_process: Callable,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Run simulations sequentially"""
        
        results = []
        
        for i in range(self.config.n_simulations):
            try:
                result = self._run_single_simulation(
                    i, model_function, priors, data_generating_process, **kwargs
                )
                results.append(result)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Completed {i + 1}/{self.config.n_simulations} simulations")
                    
            except Exception as e:
                logger.error(f"Simulation {i} failed: {e}")
        
        return results
    
    def _run_single_simulation(
        self,
        sim_id: int,
        model_function: Callable,
        priors: List[PriorSpecification],
        data_generating_process: Callable,
        **kwargs
    ) -> Dict[str, Any]:
        """Run a single SBC simulation"""
        
        # Step 1: Sample true parameters from priors
        true_parameters = {}
        for prior in priors:
            true_parameters[prior.parameter_name] = prior.sample(1)[0]
        
        # Step 2: Generate synthetic data using true parameters
        synthetic_data = data_generating_process(true_parameters, **kwargs)
        
        # Step 3: Fit model to synthetic data and get posterior samples
        posterior_samples = model_function(synthetic_data, priors, **kwargs)
        
        # Step 4: Calculate ranks for each parameter
        ranks = {}
        for param_name, true_value in true_parameters.items():
            if param_name in posterior_samples:
                post_samples = posterior_samples[param_name]
                # Rank of true value among posterior samples
                rank = np.sum(post_samples < true_value)
                ranks[param_name] = rank
        
        return {
            'simulation_id': sim_id,
            'true_parameters': true_parameters,
            'posterior_samples': posterior_samples,
            'ranks': ranks,
            'synthetic_data_size': len(synthetic_data) if hasattr(synthetic_data, '__len__') else 1
        }
    
    def _calculate_rank_statistics(
        self, 
        simulation_results: List[Dict[str, Any]]
    ) -> Dict[str, np.ndarray]:
        """Calculate rank statistics for each parameter"""
        
        rank_statistics = {}
        
        # Get all parameter names
        param_names = set()
        for result in simulation_results:
            param_names.update(result['ranks'].keys())
        
        # Calculate rank statistics for each parameter
        for param_name in param_names:
            ranks = []
            for result in simulation_results:
                if param_name in result['ranks']:
                    ranks.append(result['ranks'][param_name])
            
            if ranks:
                rank_statistics[param_name] = np.array(ranks)
        
        return rank_statistics
    
    def _test_rank_uniformity(
        self, 
        rank_statistics: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Test uniformity of rank statistics"""
        
        uniformity_results = {}
        
        for param_name, ranks in rank_statistics.items():
            # Kolmogorov-Smirnov test against uniform distribution
            # Ranks should be uniform on [0, n_posterior_samples]
            uniform_cdf = lambda x: x / self.config.n_posterior_samples
            ks_statistic, p_value = stats.kstest(
                ranks / self.config.n_posterior_samples, 
                uniform_cdf
            )
            
            uniformity_results[f'{param_name}_ks_statistic'] = ks_statistic
            uniformity_results[f'{param_name}_ks_pvalue'] = p_value
            uniformity_results[f'{param_name}_uniform'] = p_value > self.config.uniformity_test_alpha
        
        # Overall uniformity test
        all_p_values = [v for k, v in uniformity_results.items() if k.endswith('_ks_pvalue')]
        if all_p_values:
            # Bonferroni correction for multiple testing
            min_p_value = min(all_p_values)
            corrected_p_value = min_p_value * len(all_p_values)
            uniformity_results['overall_uniform'] = corrected_p_value > self.config.uniformity_test_alpha
            uniformity_results['overall_ks_pvalue'] = corrected_p_value
        
        return uniformity_results
    
    def _test_coverage_probability(
        self, 
        simulation_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Test coverage probability of credible intervals"""
        
        coverage_results = {}
        
        # Get parameter names
        param_names = set()
        for result in simulation_results:
            param_names.update(result['true_parameters'].keys())
        
        # Calculate coverage for each parameter
        for param_name in param_names:
            coverages = []
            
            for result in simulation_results:
                if (param_name in result['true_parameters'] and 
                    param_name in result['posterior_samples']):
                    
                    true_value = result['true_parameters'][param_name]
                    post_samples = result['posterior_samples'][param_name]
                    
                    # Calculate credible interval
                    alpha = 1 - self.config.confidence_level
                    lower = np.percentile(post_samples, 100 * alpha / 2)
                    upper = np.percentile(post_samples, 100 * (1 - alpha / 2))
                    
                    # Check if true value is covered
                    covered = lower <= true_value <= upper
                    coverages.append(covered)
            
            if coverages:
                empirical_coverage = np.mean(coverages)
                coverage_results[f'{param_name}_coverage'] = empirical_coverage
                
                # Test if coverage is significantly different from nominal
                n_sims = len(coverages)
                expected_coverage = self.config.confidence_level
                
                # Binomial test
                n_covered = sum(coverages)
                p_value = 2 * min(
                    stats.binom.cdf(n_covered, n_sims, expected_coverage),
                    1 - stats.binom.cdf(n_covered - 1, n_sims, expected_coverage)
                )
                
                coverage_results[f'{param_name}_coverage_test_pvalue'] = p_value
                coverage_results[f'{param_name}_coverage_ok'] = (
                    abs(empirical_coverage - expected_coverage) <= self.config.coverage_tolerance
                )
        
        # Overall coverage assessment
        coverage_values = [v for k, v in coverage_results.items() if k.endswith('_coverage')]
        if coverage_values:
            coverage_results['empirical_coverage'] = np.mean(coverage_values)
            coverage_results['coverage_std'] = np.std(coverage_values)
        
        return coverage_results
    
    def _calculate_calibration_quality(
        self,
        uniformity_results: Dict[str, float],
        coverage_results: Dict[str, float]
    ) -> float:
        """Calculate overall calibration quality score"""
        
        quality_score = 0.0
        total_weight = 0.0
        
        # Uniformity component (50% weight)
        uniform_indicators = [v for k, v in uniformity_results.items() if k.endswith('_uniform')]
        if uniform_indicators:
            uniformity_score = np.mean(uniform_indicators)
            quality_score += 0.5 * uniformity_score
            total_weight += 0.5
        
        # Coverage component (50% weight)
        coverage_indicators = [v for k, v in coverage_results.items() if k.endswith('_coverage_ok')]
        if coverage_indicators:
            coverage_score = np.mean(coverage_indicators)
            quality_score += 0.5 * coverage_score
            total_weight += 0.5
        
        return quality_score / total_weight if total_weight > 0 else 0.0
    
    def _evaluate_validation_results(
        self,
        uniformity_results: Dict[str, float],
        coverage_results: Dict[str, float],
        calibration_quality: float
    ) -> Tuple[bool, List[str]]:
        """Evaluate if validation passed and identify failure reasons"""
        
        passed = True
        failure_reasons = []
        
        # Check overall uniformity
        if not uniformity_results.get('overall_uniform', False):
            passed = False
            failure_reasons.append("Rank statistics are not uniform - model may be miscalibrated")
        
        # Check individual parameter uniformity
        param_uniform_failures = [
            k.replace('_uniform', '') for k, v in uniformity_results.items() 
            if k.endswith('_uniform') and not v and k != 'overall_uniform'
        ]
        if param_uniform_failures:
            failure_reasons.append(f"Non-uniform ranks for parameters: {', '.join(param_uniform_failures)}")
        
        # Check coverage
        poor_coverage_params = [
            k.replace('_coverage_ok', '') for k, v in coverage_results.items()
            if k.endswith('_coverage_ok') and not v
        ]
        if poor_coverage_params:
            passed = False
            failure_reasons.append(f"Poor coverage for parameters: {', '.join(poor_coverage_params)}")
        
        # Check overall calibration quality
        if calibration_quality < 0.8:
            passed = False
            failure_reasons.append(f"Low overall calibration quality: {calibration_quality:.3f}")
        
        return passed, failure_reasons
    
    def _generate_recommendations(
        self,
        passed: bool,
        failure_reasons: List[str],
        uniformity_results: Dict[str, float],
        coverage_results: Dict[str, float]
    ) -> List[str]:
        """Generate actionable recommendations based on SBC results"""
        
        recommendations = []
        
        if passed:
            recommendations.append("Model appears well-calibrated and suitable for production use")
            recommendations.append("Consider periodic SBC validation to monitor ongoing calibration")
        else:
            recommendations.append("Model calibration issues detected - review before production deployment")
            
            # Specific recommendations based on failure types
            if any("uniform" in reason for reason in failure_reasons):
                recommendations.append("Review prior specifications - they may be too informative or inappropriate")
                recommendations.append("Consider hierarchical modeling to better capture parameter uncertainty")
            
            if any("coverage" in reason for reason in failure_reasons):
                recommendations.append("Check model implementation for bugs or numerical issues")
                recommendations.append("Increase MCMC chain length or improve sampler efficiency")
            
            if any("quality" in reason for reason in failure_reasons):
                recommendations.append("Comprehensive model review recommended before deployment")
                recommendations.append("Consider alternative model specifications or priors")
        
        # Data-driven recommendations
        empirical_coverage = coverage_results.get('empirical_coverage', 0.0)
        if empirical_coverage < 0.9:
            recommendations.append("Low coverage suggests overconfident posteriors - review prior specifications")
        elif empirical_coverage > 0.98:
            recommendations.append("High coverage suggests underconfident posteriors - consider more informative priors")
        
        return recommendations
    
    def _compile_diagnostics(
        self,
        simulation_results: List[Dict[str, Any]],
        rank_statistics: Dict[str, np.ndarray],
        uniformity_results: Dict[str, float],
        coverage_results: Dict[str, float]
    ) -> Dict[str, Any]:
        """Compile comprehensive diagnostic information"""
        
        diagnostics = {
            'n_simulations_completed': len(simulation_results),
            'n_simulations_requested': self.config.n_simulations,
            'completion_rate': len(simulation_results) / self.config.n_simulations,
            'rank_statistics_summary': {},
            'uniformity_test_summary': uniformity_results,
            'coverage_test_summary': coverage_results,
            'parameter_diagnostics': {}
        }
        
        # Rank statistics summary
        for param_name, ranks in rank_statistics.items():
            diagnostics['rank_statistics_summary'][param_name] = {
                'mean_rank': float(np.mean(ranks)),
                'std_rank': float(np.std(ranks)),
                'min_rank': int(np.min(ranks)),
                'max_rank': int(np.max(ranks)),
                'expected_mean': self.config.n_posterior_samples / 2,
                'expected_std': self.config.n_posterior_samples / np.sqrt(12)
            }
        
        # Parameter-specific diagnostics
        param_names = set()
        for result in simulation_results:
            param_names.update(result['true_parameters'].keys())
        
        for param_name in param_names:
            true_values = []
            posterior_means = []
            posterior_stds = []
            
            for result in simulation_results:
                if (param_name in result['true_parameters'] and 
                    param_name in result['posterior_samples']):
                    
                    true_values.append(result['true_parameters'][param_name])
                    post_samples = result['posterior_samples'][param_name]
                    posterior_means.append(np.mean(post_samples))
                    posterior_stds.append(np.std(post_samples))
            
            if true_values:
                diagnostics['parameter_diagnostics'][param_name] = {
                    'bias': float(np.mean(np.array(posterior_means) - np.array(true_values))),
                    'rmse': float(np.sqrt(np.mean((np.array(posterior_means) - np.array(true_values))**2))),
                    'mean_posterior_std': float(np.mean(posterior_stds)),
                    'coverage_probability': coverage_results.get(f'{param_name}_coverage', 0.0)
                }
        
        return diagnostics
    
    def _create_simulation_summary(
        self, 
        simulation_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create summary of simulation results"""
        
        return {
            'total_simulations': len(simulation_results),
            'successful_simulations': len([r for r in simulation_results if 'ranks' in r]),
            'average_data_size': np.mean([
                r.get('synthetic_data_size', 0) for r in simulation_results
            ]),
            'parameters_analyzed': len(set().union(*[
                r.get('true_parameters', {}).keys() for r in simulation_results
            ])),
            'timestamp': datetime.utcnow().isoformat()
        }


class SBCDecisionFramework:
    """Framework for deciding when SBC validation becomes essential"""
    
    def __init__(self):
        self.decision_criteria = {
            'model_complexity': {
                'low_threshold': 5,      # < 5 parameters
                'high_threshold': 15     # > 15 parameters
            },
            'conflict_severity': {
                'moderate_threshold': 3.0,   # Bayes factor
                'strong_threshold': 10.0     # Bayes factor
            },
            'business_impact': {
                'low_threshold': 100000,     # $100K
                'high_threshold': 1000000    # $1M
            },
            'client_confidence': {
                'low_threshold': 0.5,
                'high_threshold': 0.8
            }
        }
    
    def should_run_sbc(
        self,
        model_complexity: int,
        max_conflict_severity: float,
        business_impact_value: float,
        min_client_confidence: float,
        **kwargs
    ) -> Tuple[bool, str, int]:
        """
        Determine if SBC validation is essential
        
        Args:
            model_complexity: Number of parameters in model
            max_conflict_severity: Maximum Bayes factor from conflict analysis
            business_impact_value: Estimated business impact in dollars
            min_client_confidence: Minimum client confidence in priors
            
        Returns:
            Tuple of (should_run, reason, priority_level)
        """
        
        priority_score = 0
        reasons = []
        
        # Model complexity criterion
        if model_complexity >= self.decision_criteria['model_complexity']['high_threshold']:
            priority_score += 3
            reasons.append(f"High model complexity ({model_complexity} parameters)")
        elif model_complexity >= self.decision_criteria['model_complexity']['low_threshold']:
            priority_score += 1
            reasons.append(f"Moderate model complexity ({model_complexity} parameters)")
        
        # Conflict severity criterion
        if max_conflict_severity >= self.decision_criteria['conflict_severity']['strong_threshold']:
            priority_score += 3
            reasons.append(f"Strong prior-data conflicts detected (BF={max_conflict_severity:.1f})")
        elif max_conflict_severity >= self.decision_criteria['conflict_severity']['moderate_threshold']:
            priority_score += 2
            reasons.append(f"Moderate prior-data conflicts detected (BF={max_conflict_severity:.1f})")
        
        # Business impact criterion
        if business_impact_value >= self.decision_criteria['business_impact']['high_threshold']:
            priority_score += 3
            reasons.append(f"High business impact (${business_impact_value:,.0f})")
        elif business_impact_value >= self.decision_criteria['business_impact']['low_threshold']:
            priority_score += 1
            reasons.append(f"Moderate business impact (${business_impact_value:,.0f})")
        
        # Client confidence criterion
        if min_client_confidence <= self.decision_criteria['client_confidence']['low_threshold']:
            priority_score += 2
            reasons.append(f"Low client confidence in priors ({min_client_confidence:.1%})")
        elif min_client_confidence <= self.decision_criteria['client_confidence']['high_threshold']:
            priority_score += 1
            reasons.append(f"Moderate client confidence in priors ({min_client_confidence:.1%})")
        
        # Decision logic
        should_run = priority_score >= 3
        
        if priority_score >= 6:
            priority_level = 1  # Critical
        elif priority_score >= 4:
            priority_level = 2  # High
        elif priority_score >= 3:
            priority_level = 3  # Medium
        else:
            priority_level = 4  # Low
        
        reason = "SBC validation " + ("essential" if should_run else "recommended") + ": " + "; ".join(reasons)
        
        return should_run, reason, priority_level
    
    def get_sbc_recommendations(
        self,
        should_run: bool,
        priority_level: int,
        model_complexity: int
    ) -> Dict[str, Any]:
        """Get specific SBC configuration recommendations"""
        
        if not should_run:
            return {
                'recommended': False,
                'reason': 'SBC validation not essential for this model',
                'alternative': 'Consider basic posterior predictive checks'
            }
        
        # Base configuration
        config = SBCConfig()
        
        # Adjust based on priority and complexity
        if priority_level == 1:  # Critical
            config.n_simulations = 2000
            config.n_posterior_samples = 2000
        elif priority_level == 2:  # High
            config.n_simulations = 1500
            config.n_posterior_samples = 1500
        elif priority_level == 3:  # Medium
            config.n_simulations = 1000
            config.n_posterior_samples = 1000
        
        # Adjust for model complexity
        if model_complexity > 20:
            config.n_simulations = min(config.n_simulations * 1.5, 3000)
        
        return {
            'recommended': True,
            'priority_level': priority_level,
            'config': config,
            'estimated_runtime_hours': self._estimate_runtime(config, model_complexity),
            'resource_requirements': self._estimate_resources(config, model_complexity)
        }
    
    def _estimate_runtime(self, config: SBCConfig, model_complexity: int) -> float:
        """Estimate SBC runtime in hours"""
        # Rough estimation based on complexity and simulations
        base_time_per_sim = 0.1 + (model_complexity * 0.01)  # seconds
        total_time_seconds = config.n_simulations * base_time_per_sim
        
        if config.parallel_execution:
            total_time_seconds /= config.max_workers
        
        return total_time_seconds / 3600  # Convert to hours
    
    def _estimate_resources(self, config: SBCConfig, model_complexity: int) -> Dict[str, str]:
        """Estimate computational resource requirements"""
        
        # Memory estimation
        memory_per_sim_mb = 10 + (model_complexity * 2)
        total_memory_mb = memory_per_sim_mb * (config.max_workers if config.parallel_execution else 1)
        
        # CPU estimation
        cpu_cores = config.max_workers if config.parallel_execution else 1
        
        return {
            'memory_mb': f"{total_memory_mb:.0f}",
            'cpu_cores': f"{cpu_cores}",
            'storage_mb': f"{config.n_simulations * 0.1:.0f}",  # For storing results
            'recommended_instance': self._recommend_instance_type(total_memory_mb, cpu_cores)
        }
    
    def _recommend_instance_type(self, memory_mb: float, cpu_cores: int) -> str:
        """Recommend cloud instance type based on requirements"""
        
        if memory_mb > 8000 or cpu_cores > 4:
            return "Large instance (8+ cores, 16+ GB RAM)"
        elif memory_mb > 4000 or cpu_cores > 2:
            return "Medium instance (4 cores, 8 GB RAM)"
        else:
            return "Small instance (2 cores, 4 GB RAM)"