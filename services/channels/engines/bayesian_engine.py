"""
Bayesian Engine for Channels Service
Bayesian inference and uncertainty quantification
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from scipy.stats import norm, beta, gamma, invgamma
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from models.channels import ChannelPerformance

logger = logging.getLogger(__name__)


class BayesianEngine:
    """Bayesian inference engine for uncertainty quantification and prior specification"""
    
    def __init__(self, memory_client, bayesian_analysis_client):
        self.memory_client = memory_client
        self.bayesian_analysis_client = bayesian_analysis_client
        
        # Bayesian parameters
        self.prior_update_threshold = 0.1
        self.credible_interval_level = 0.95
        self.mcmc_samples = 5000
        self.burn_in_samples = 1000
        
        # Default priors
        self.default_priors = {
            "roas": {
                "distribution": "gamma",
                "alpha": 2.0,
                "beta": 0.5,
                "mean": 4.0,
                "std": 2.0
            },
            "conversion_rate": {
                "distribution": "beta",
                "alpha": 2.0,
                "beta": 98.0,
                "mean": 0.02,
                "std": 0.01
            },
            "cac": {
                "distribution": "gamma",
                "alpha": 4.0,
                "beta": 0.4,
                "mean": 10.0,
                "std": 5.0
            },
            "saturation_alpha": {
                "distribution": "gamma",
                "alpha": 3.0,
                "beta": 0.003,
                "mean": 1000.0,
                "std": 577.0
            },
            "saturation_beta": {
                "distribution": "gamma",
                "alpha": 2.0,
                "beta": 0.004,
                "mean": 500.0,
                "std": 354.0
            }
        }
    
    async def get_channel_priors(self, org_id: str, channel_id: str) -> Dict[str, Any]:
        """Get Bayesian priors for a channel"""
        
        try:
            # Try to get existing priors from memory
            cached_priors = await self._get_cached_priors(org_id, channel_id)
            
            if cached_priors and self._are_priors_fresh(cached_priors):
                logger.info(f"Using cached priors for {channel_id}")
                return cached_priors
            
            # Get historical data for prior updating
            historical_data = await self._get_historical_performance_data(org_id, channel_id)
            
            # Update priors with historical data
            updated_priors = await self._update_priors_with_data(
                org_id, channel_id, historical_data
            )
            
            # Cache the updated priors
            await self._cache_priors(org_id, channel_id, updated_priors)
            
            return updated_priors
            
        except Exception as e:
            logger.error(f"Failed to get priors for {channel_id}: {str(e)}")
            return self._get_default_channel_priors(channel_id)
    
    async def update_posterior(
        self, 
        org_id: str, 
        channel_id: str, 
        observed_data: Dict[str, float],
        prior_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update posterior distributions with new observed data"""
        
        try:
            posterior_params = {}
            
            # Update ROAS posterior
            if "roas" in observed_data and "roas" in prior_params:
                posterior_params["roas"] = self._update_gamma_posterior(
                    prior_params["roas"], observed_data["roas"], "roas"
                )
            
            # Update conversion rate posterior
            if "conversion_rate" in observed_data and "conversion_rate" in prior_params:
                posterior_params["conversion_rate"] = self._update_beta_posterior(
                    prior_params["conversion_rate"], observed_data["conversion_rate"]
                )
            
            # Update CAC posterior
            if "cac" in observed_data and "cac" in prior_params:
                posterior_params["cac"] = self._update_gamma_posterior(
                    prior_params["cac"], observed_data["cac"], "cac"
                )
            
            # Calculate posterior predictive distributions
            posterior_params["predictive"] = self._calculate_posterior_predictive(
                posterior_params, observed_data
            )
            
            # Calculate uncertainty metrics
            posterior_params["uncertainty"] = self._calculate_uncertainty_metrics(
                posterior_params
            )
            
            return posterior_params
            
        except Exception as e:
            logger.error(f"Failed to update posterior for {channel_id}: {str(e)}")
            return prior_params
    
    async def calculate_credible_intervals(
        self, 
        org_id: str, 
        channel_id: str, 
        parameters: Dict[str, Any],
        confidence_level: float = 0.95
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate Bayesian credible intervals"""
        
        credible_intervals = {}
        alpha = 1 - confidence_level
        
        try:
            for param_name, param_data in parameters.items():
                if isinstance(param_data, dict) and "distribution" in param_data:
                    interval = self._calculate_credible_interval(param_data, alpha)
                    credible_intervals[param_name] = interval
            
            return credible_intervals
            
        except Exception as e:
            logger.error(f"Failed to calculate credible intervals for {channel_id}: {str(e)}")
            return {}
    
    async def perform_bayesian_model_averaging(
        self, 
        org_id: str, 
        channel_id: str, 
        models: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform Bayesian Model Averaging across different models"""
        
        try:
            if not models:
                return {}
            
            # Calculate model weights using marginal likelihoods
            model_weights = await self._calculate_model_weights(models)
            
            # Average predictions across models
            averaged_predictions = self._average_model_predictions(models, model_weights)
            
            # Calculate model uncertainty
            model_uncertainty = self._calculate_model_uncertainty(models, model_weights)
            
            return {
                "averaged_predictions": averaged_predictions,
                "model_weights": model_weights,
                "model_uncertainty": model_uncertainty,
                "effective_models": len([w for w in model_weights if w > 0.01])
            }
            
        except Exception as e:
            logger.error(f"Bayesian model averaging failed for {channel_id}: {str(e)}")
            return {}
    
    async def estimate_causal_uncertainty(
        self, 
        org_id: str, 
        channel_id: str, 
        causal_estimate: float,
        historical_variance: float
    ) -> Dict[str, float]:
        """Estimate uncertainty in causal effect estimates"""
        
        try:
            # Use hierarchical Bayesian model for causal uncertainty
            prior_precision = 1.0 / (historical_variance + 1e-6)
            
            # Posterior parameters for causal effect
            posterior_mean = causal_estimate
            posterior_variance = 1.0 / (prior_precision + 1.0)
            posterior_std = np.sqrt(posterior_variance)
            
            # Calculate credible interval
            alpha = 1 - self.credible_interval_level
            lower = norm.ppf(alpha/2, posterior_mean, posterior_std)
            upper = norm.ppf(1 - alpha/2, posterior_mean, posterior_std)
            
            # Calculate probability of positive effect
            prob_positive = 1 - norm.cdf(0, posterior_mean, posterior_std)
            
            return {
                "posterior_mean": float(posterior_mean),
                "posterior_std": float(posterior_std),
                "credible_interval_lower": float(lower),
                "credible_interval_upper": float(upper),
                "probability_positive_effect": float(prob_positive),
                "uncertainty_level": float(posterior_std / abs(posterior_mean)) if posterior_mean != 0 else float('inf')
            }
            
        except Exception as e:
            logger.error(f"Failed to estimate causal uncertainty for {channel_id}: {str(e)}")
            return {
                "posterior_mean": causal_estimate,
                "posterior_std": 0.1,
                "credible_interval_lower": causal_estimate * 0.8,
                "credible_interval_upper": causal_estimate * 1.2,
                "probability_positive_effect": 0.8,
                "uncertainty_level": 0.1
            }
    
    async def _get_historical_performance_data(self, org_id: str, channel_id: str) -> List[Dict[str, float]]:
        """Get historical performance data for prior updating"""
        
        try:
            # Get data from Bayesian analysis service
            historical_data = await self.bayesian_analysis_client.get_channel_performance_history(
                org_id=org_id,
                channel_id=channel_id,
                days_back=90
            )
            
            return historical_data
            
        except Exception as e:
            logger.warning(f"Failed to get historical data for {channel_id}: {str(e)}")
            # Return mock data for development
            return self._generate_mock_performance_data(channel_id)
    
    async def _update_priors_with_data(
        self, 
        org_id: str, 
        channel_id: str, 
        historical_data: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Update priors using historical data"""
        
        if not historical_data:
            return self._get_default_channel_priors(channel_id)
        
        # Extract metrics from historical data
        roas_values = [d.get("roas", 0) for d in historical_data if d.get("roas", 0) > 0]
        conversion_rates = [d.get("conversion_rate", 0) for d in historical_data if d.get("conversion_rate", 0) > 0]
        cac_values = [d.get("cac", 0) for d in historical_data if d.get("cac", 0) > 0]
        
        updated_priors = {}
        
        # Update ROAS prior
        if roas_values:
            updated_priors["roas"] = self._fit_gamma_prior(roas_values, "roas")
        else:
            updated_priors["roas"] = self.default_priors["roas"].copy()
        
        # Update conversion rate prior
        if conversion_rates:
            updated_priors["conversion_rate"] = self._fit_beta_prior(conversion_rates)
        else:
            updated_priors["conversion_rate"] = self.default_priors["conversion_rate"].copy()
        
        # Update CAC prior
        if cac_values:
            updated_priors["cac"] = self._fit_gamma_prior(cac_values, "cac")
        else:
            updated_priors["cac"] = self.default_priors["cac"].copy()
        
        # Add saturation priors (use defaults for now)
        updated_priors["saturation_alpha"] = self.default_priors["saturation_alpha"].copy()
        updated_priors["saturation_beta"] = self.default_priors["saturation_beta"].copy()
        
        # Add metadata
        updated_priors["last_updated"] = datetime.utcnow().isoformat()
        updated_priors["data_points"] = len(historical_data)
        updated_priors["channel_id"] = channel_id
        
        return updated_priors
    
    def _fit_gamma_prior(self, values: List[float], metric_name: str) -> Dict[str, Any]:
        """Fit gamma distribution to observed values"""
        
        values_array = np.array([v for v in values if v > 0])
        
        if len(values_array) < 2:
            return self.default_priors.get(metric_name, self.default_priors["roas"]).copy()
        
        # Method of moments estimation
        sample_mean = np.mean(values_array)
        sample_var = np.var(values_array)
        
        if sample_var <= 0:
            return self.default_priors.get(metric_name, self.default_priors["roas"]).copy()
        
        # Gamma parameters: mean = alpha/beta, var = alpha/beta^2
        beta_est = sample_mean / sample_var
        alpha_est = sample_mean * beta_est
        
        # Ensure reasonable bounds
        alpha_est = max(0.1, min(alpha_est, 100.0))
        beta_est = max(0.001, min(beta_est, 10.0))
        
        return {
            "distribution": "gamma",
            "alpha": float(alpha_est),
            "beta": float(beta_est),
            "mean": float(sample_mean),
            "std": float(np.sqrt(sample_var))
        }
    
    def _fit_beta_prior(self, values: List[float]) -> Dict[str, Any]:
        """Fit beta distribution to observed conversion rates"""
        
        values_array = np.array([v for v in values if 0 < v < 1])
        
        if len(values_array) < 2:
            return self.default_priors["conversion_rate"].copy()
        
        # Method of moments estimation for beta distribution
        sample_mean = np.mean(values_array)
        sample_var = np.var(values_array)
        
        if sample_var <= 0 or sample_mean <= 0 or sample_mean >= 1:
            return self.default_priors["conversion_rate"].copy()
        
        # Beta parameters: mean = alpha/(alpha+beta), var = alpha*beta/((alpha+beta)^2*(alpha+beta+1))
        common_factor = sample_mean * (1 - sample_mean) / sample_var - 1
        alpha_est = sample_mean * common_factor
        beta_est = (1 - sample_mean) * common_factor
        
        # Ensure reasonable bounds
        alpha_est = max(0.1, min(alpha_est, 1000.0))
        beta_est = max(0.1, min(beta_est, 1000.0))
        
        return {
            "distribution": "beta",
            "alpha": float(alpha_est),
            "beta": float(beta_est),
            "mean": float(sample_mean),
            "std": float(np.sqrt(sample_var))
        }
    
    def _update_gamma_posterior(
        self, 
        prior_params: Dict[str, Any], 
        observed_value: float,
        metric_name: str
    ) -> Dict[str, Any]:
        """Update gamma posterior with new observation"""
        
        if observed_value <= 0:
            return prior_params
        
        # Conjugate update for gamma distribution
        prior_alpha = prior_params.get("alpha", 2.0)
        prior_beta = prior_params.get("beta", 0.5)
        
        # Assuming single observation with known precision
        posterior_alpha = prior_alpha + 1
        posterior_beta = prior_beta + 1.0 / observed_value
        
        posterior_mean = posterior_alpha / posterior_beta
        posterior_var = posterior_alpha / (posterior_beta ** 2)
        
        return {
            "distribution": "gamma",
            "alpha": float(posterior_alpha),
            "beta": float(posterior_beta),
            "mean": float(posterior_mean),
            "std": float(np.sqrt(posterior_var)),
            "updated": True
        }
    
    def _update_beta_posterior(
        self, 
        prior_params: Dict[str, Any], 
        observed_rate: float
    ) -> Dict[str, Any]:
        """Update beta posterior with new conversion rate observation"""
        
        if not (0 < observed_rate < 1):
            return prior_params
        
        # Simplified update assuming single trial
        prior_alpha = prior_params.get("alpha", 2.0)
        prior_beta = prior_params.get("beta", 98.0)
        
        # Add pseudo-observations based on observed rate
        pseudo_successes = observed_rate * 100  # Scale to reasonable counts
        pseudo_failures = (1 - observed_rate) * 100
        
        posterior_alpha = prior_alpha + pseudo_successes
        posterior_beta = prior_beta + pseudo_failures
        
        posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
        posterior_var = (posterior_alpha * posterior_beta) / (
            (posterior_alpha + posterior_beta) ** 2 * (posterior_alpha + posterior_beta + 1)
        )
        
        return {
            "distribution": "beta",
            "alpha": float(posterior_alpha),
            "beta": float(posterior_beta),
            "mean": float(posterior_mean),
            "std": float(np.sqrt(posterior_var)),
            "updated": True
        }
    
    def _calculate_posterior_predictive(
        self, 
        posterior_params: Dict[str, Any], 
        observed_data: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate posterior predictive distributions"""
        
        predictive = {}
        
        # ROAS prediction
        if "roas" in posterior_params:
            roas_params = posterior_params["roas"]
            if roas_params.get("distribution") == "gamma":
                alpha = roas_params.get("alpha", 2.0)
                beta = roas_params.get("beta", 0.5)
                
                # Predictive mean and variance for gamma
                pred_mean = alpha / beta
                pred_var = alpha / (beta ** 2)
                
                predictive["roas"] = {
                    "mean": float(pred_mean),
                    "std": float(np.sqrt(pred_var)),
                    "distribution": "gamma",
                    "parameters": {"alpha": alpha, "beta": beta}
                }
        
        # Conversion rate prediction
        if "conversion_rate" in posterior_params:
            cr_params = posterior_params["conversion_rate"]
            if cr_params.get("distribution") == "beta":
                alpha = cr_params.get("alpha", 2.0)
                beta = cr_params.get("beta", 98.0)
                
                pred_mean = alpha / (alpha + beta)
                pred_var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
                
                predictive["conversion_rate"] = {
                    "mean": float(pred_mean),
                    "std": float(np.sqrt(pred_var)),
                    "distribution": "beta",
                    "parameters": {"alpha": alpha, "beta": beta}
                }
        
        return predictive
    
    def _calculate_uncertainty_metrics(self, posterior_params: Dict[str, Any]) -> Dict[str, float]:
        """Calculate various uncertainty metrics"""
        
        uncertainty_metrics = {}
        
        for param_name, param_data in posterior_params.items():
            if isinstance(param_data, dict) and "mean" in param_data and "std" in param_data:
                mean = param_data["mean"]
                std = param_data["std"]
                
                # Coefficient of variation
                cv = std / abs(mean) if mean != 0 else float('inf')
                
                # Relative uncertainty
                rel_uncertainty = std / abs(mean) if mean != 0 else 1.0
                
                uncertainty_metrics[f"{param_name}_coefficient_of_variation"] = float(cv)
                uncertainty_metrics[f"{param_name}_relative_uncertainty"] = float(rel_uncertainty)
        
        # Overall uncertainty score
        if uncertainty_metrics:
            avg_uncertainty = np.mean([
                v for k, v in uncertainty_metrics.items() 
                if "relative_uncertainty" in k and not np.isinf(v)
            ])
            uncertainty_metrics["overall_uncertainty"] = float(avg_uncertainty)
        
        return uncertainty_metrics
    
    def _calculate_credible_interval(
        self, 
        param_data: Dict[str, Any], 
        alpha: float
    ) -> Tuple[float, float]:
        """Calculate credible interval for a parameter"""
        
        distribution = param_data.get("distribution", "normal")
        
        if distribution == "gamma":
            alpha_param = param_data.get("alpha", 2.0)
            beta_param = param_data.get("beta", 0.5)
            
            lower = gamma.ppf(alpha/2, alpha_param, scale=1/beta_param)
            upper = gamma.ppf(1 - alpha/2, alpha_param, scale=1/beta_param)
            
        elif distribution == "beta":
            alpha_param = param_data.get("alpha", 2.0)
            beta_param = param_data.get("beta", 98.0)
            
            lower = beta.ppf(alpha/2, alpha_param, beta_param)
            upper = beta.ppf(1 - alpha/2, alpha_param, beta_param)
            
        else:  # Normal distribution
            mean = param_data.get("mean", 0.0)
            std = param_data.get("std", 1.0)
            
            lower = norm.ppf(alpha/2, mean, std)
            upper = norm.ppf(1 - alpha/2, mean, std)
        
        return (float(lower), float(upper))
    
    async def _calculate_model_weights(self, models: List[Dict[str, Any]]) -> List[float]:
        """Calculate Bayesian model weights using marginal likelihoods"""
        
        # Simplified implementation - in practice would use proper marginal likelihood calculation
        log_marginal_likelihoods = []
        
        for model in models:
            # Placeholder: use model fit metrics as proxy for marginal likelihood
            r_squared = model.get("r_squared", 0.5)
            aic = model.get("aic", 100.0)
            
            # Convert to log marginal likelihood proxy
            log_ml = r_squared * 10 - aic * 0.1
            log_marginal_likelihoods.append(log_ml)
        
        # Convert to weights using softmax
        log_ml_array = np.array(log_marginal_likelihoods)
        log_ml_array = log_ml_array - np.max(log_ml_array)  # Numerical stability
        
        weights = np.exp(log_ml_array)
        weights = weights / np.sum(weights)
        
        return weights.tolist()
    
    def _average_model_predictions(
        self, 
        models: List[Dict[str, Any]], 
        weights: List[float]
    ) -> Dict[str, float]:
        """Average predictions across models using Bayesian weights"""
        
        averaged = {}
        
        # Get common prediction keys
        prediction_keys = set()
        for model in models:
            if "predictions" in model:
                prediction_keys.update(model["predictions"].keys())
        
        # Average each prediction
        for key in prediction_keys:
            weighted_sum = 0.0
            total_weight = 0.0
            
            for model, weight in zip(models, weights):
                if "predictions" in model and key in model["predictions"]:
                    weighted_sum += model["predictions"][key] * weight
                    total_weight += weight
            
            if total_weight > 0:
                averaged[key] = weighted_sum / total_weight
        
        return averaged
    
    def _calculate_model_uncertainty(
        self, 
        models: List[Dict[str, Any]], 
        weights: List[float]
    ) -> Dict[str, float]:
        """Calculate uncertainty due to model selection"""
        
        model_uncertainty = {}
        
        # Calculate variance across models for each prediction
        prediction_keys = set()
        for model in models:
            if "predictions" in model:
                prediction_keys.update(model["predictions"].keys())
        
        for key in prediction_keys:
            predictions = []
            model_weights = []
            
            for model, weight in zip(models, weights):
                if "predictions" in model and key in model["predictions"]:
                    predictions.append(model["predictions"][key])
                    model_weights.append(weight)
            
            if len(predictions) > 1:
                # Weighted variance
                weighted_mean = np.average(predictions, weights=model_weights)
                weighted_var = np.average(
                    (np.array(predictions) - weighted_mean) ** 2, 
                    weights=model_weights
                )
                model_uncertainty[f"{key}_model_variance"] = float(weighted_var)
                model_uncertainty[f"{key}_model_std"] = float(np.sqrt(weighted_var))
        
        return model_uncertainty
    
    # Caching and utility methods
    
    async def _get_cached_priors(self, org_id: str, channel_id: str) -> Optional[Dict[str, Any]]:
        """Get cached priors from memory service"""
        try:
            cache_key = f"bayesian_priors:{org_id}:{channel_id}"
            cached_data = await self.memory_client.get(cache_key)
            return cached_data
        except Exception as e:
            logger.warning(f"Failed to get cached priors for {channel_id}: {str(e)}")
            return None
    
    async def _cache_priors(self, org_id: str, channel_id: str, priors: Dict[str, Any]):
        """Cache priors in memory service"""
        try:
            cache_key = f"bayesian_priors:{org_id}:{channel_id}"
            # Cache for 7 days
            await self.memory_client.set(cache_key, priors, ttl=604800)
        except Exception as e:
            logger.warning(f"Failed to cache priors for {channel_id}: {str(e)}")
    
    def _are_priors_fresh(self, priors: Dict[str, Any]) -> bool:
        """Check if cached priors are fresh enough"""
        if "last_updated" not in priors:
            return False
        
        try:
            last_updated = datetime.fromisoformat(priors["last_updated"])
            age = datetime.utcnow() - last_updated
            return age.days < 7  # Fresh for 7 days
        except:
            return False
    
    def _get_default_channel_priors(self, channel_id: str) -> Dict[str, Any]:
        """Get default priors for a channel"""
        priors = {}
        for key, value in self.default_priors.items():
            priors[key] = value.copy()
        
        priors["last_updated"] = datetime.utcnow().isoformat()
        priors["data_points"] = 0
        priors["channel_id"] = channel_id
        
        return priors
    
    def _generate_mock_performance_data(self, channel_id: str) -> List[Dict[str, float]]:
        """Generate mock performance data for development"""
        np.random.seed(42)
        
        data = []
        for i in range(30):
            # Generate realistic performance metrics with some variation
            roas = np.random.gamma(2.0, 2.0)  # Mean around 4.0
            conversion_rate = np.random.beta(2.0, 98.0)  # Mean around 0.02
            cac = np.random.gamma(4.0, 2.5)  # Mean around 10.0
            
            data.append({
                "roas": float(roas),
                "conversion_rate": float(conversion_rate),
                "cac": float(cac),
                "date": (datetime.utcnow() - timedelta(days=30-i)).isoformat()
            })
        
        return data