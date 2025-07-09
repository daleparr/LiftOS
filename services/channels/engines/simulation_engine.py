"""
Simulation Engine for Channels Service
Monte Carlo simulation for what-if scenario modeling
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from scipy.stats import norm, beta, gamma
import json

from models.channels import (
    SimulationRequest, SimulationResult, ScenarioResult, 
    ChannelPerformance, SimulationStatus, ScenarioType
)

logger = logging.getLogger(__name__)


class SimulationEngine:
    """Monte Carlo simulation engine for what-if scenario analysis"""
    
    def __init__(self, saturation_engine, bayesian_engine, optimization_engine):
        self.saturation_engine = saturation_engine
        self.bayesian_engine = bayesian_engine
        self.optimization_engine = optimization_engine
        
        # Simulation parameters
        self.default_samples = 10000
        self.confidence_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
        
        # Random seed for reproducibility
        self.random_seed = 42
    
    async def run_simulation(
        self, 
        org_id: str, 
        request: SimulationRequest,
        simulation_id: str
    ) -> SimulationResult:
        """Main simulation function"""
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Starting simulation {simulation_id} for org {org_id}")
            
            # Set random seed for reproducibility
            np.random.seed(self.random_seed)
            
            # 1. Gather baseline data
            baseline_data = await self._get_baseline_data(org_id, request.channels)
            
            # 2. Get uncertainty parameters
            uncertainty_params = await self._get_uncertainty_parameters(org_id, request.channels)
            
            # 3. Run scenarios
            scenario_results = []
            for scenario in request.scenarios:
                scenario_result = await self._run_scenario_simulation(
                    scenario, baseline_data, uncertainty_params, request.monte_carlo_samples
                )
                scenario_results.append(scenario_result)
            
            # 4. Generate comparative analysis
            comparative_analysis = self._generate_comparative_analysis(scenario_results, baseline_data)
            
            # 5. Calculate risk metrics
            risk_analysis = self._calculate_risk_analysis(scenario_results)
            
            # 6. Generate insights and recommendations
            insights = self._generate_simulation_insights(scenario_results, comparative_analysis)
            
            # Create simulation result
            result = SimulationResult(
                simulation_id=simulation_id,
                org_id=org_id,
                status=SimulationStatus.COMPLETED,
                scenarios_tested=len(request.scenarios),
                monte_carlo_samples=request.monte_carlo_samples,
                confidence_level=request.confidence_level,
                baseline_performance=baseline_data,
                scenario_results=scenario_results,
                comparative_analysis=comparative_analysis,
                risk_analysis=risk_analysis,
                sensitivity_analysis=self._calculate_sensitivity_analysis(scenario_results),
                insights=insights,
                recommendations=self._generate_scenario_recommendations(scenario_results),
                computation_time=(datetime.utcnow() - start_time).total_seconds(),
                completed_at=datetime.utcnow()
            )
            
            logger.info(f"Completed simulation {simulation_id}")
            return result
            
        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}")
            # Return failed result
            return SimulationResult(
                simulation_id=simulation_id,
                org_id=org_id,
                status=SimulationStatus.FAILED,
                scenarios_tested=0,
                monte_carlo_samples=request.monte_carlo_samples,
                confidence_level=request.confidence_level,
                baseline_performance={},
                scenario_results=[],
                comparative_analysis={},
                risk_analysis={"error": str(e)},
                sensitivity_analysis={},
                insights=[f"Simulation failed: {str(e)}"],
                recommendations=[],
                computation_time=(datetime.utcnow() - start_time).total_seconds(),
                completed_at=datetime.utcnow()
            )
    
    async def _get_baseline_data(self, org_id: str, channels: List[str]) -> Dict[str, Any]:
        """Get baseline performance data"""
        baseline_data = {}
        
        # Get current channel performance
        for channel_id in channels:
            try:
                performance = await self.optimization_engine._fetch_channel_performance(org_id, channel_id)
                baseline_data[channel_id] = {
                    "current_spend": performance.current_spend,
                    "current_revenue": performance.current_revenue,
                    "current_roas": performance.current_roas,
                    "current_conversions": performance.current_conversions,
                    "current_cac": performance.current_cac
                }
            except Exception as e:
                logger.warning(f"Failed to get baseline data for {channel_id}: {str(e)}")
                baseline_data[channel_id] = self._get_default_baseline(channel_id)
        
        # Calculate totals
        baseline_data["totals"] = {
            "total_spend": sum(ch["current_spend"] for ch in baseline_data.values() if isinstance(ch, dict)),
            "total_revenue": sum(ch["current_revenue"] for ch in baseline_data.values() if isinstance(ch, dict)),
            "total_conversions": sum(ch["current_conversions"] for ch in baseline_data.values() if isinstance(ch, dict)),
            "overall_roas": 0.0,
            "overall_cac": 0.0
        }
        
        # Calculate overall metrics
        if baseline_data["totals"]["total_spend"] > 0:
            baseline_data["totals"]["overall_roas"] = (
                baseline_data["totals"]["total_revenue"] / baseline_data["totals"]["total_spend"]
            )
        
        if baseline_data["totals"]["total_conversions"] > 0:
            baseline_data["totals"]["overall_cac"] = (
                baseline_data["totals"]["total_spend"] / baseline_data["totals"]["total_conversions"]
            )
        
        return baseline_data
    
    async def _get_uncertainty_parameters(self, org_id: str, channels: List[str]) -> Dict[str, Any]:
        """Get uncertainty parameters for Monte Carlo simulation"""
        uncertainty_params = {}
        
        for channel_id in channels:
            try:
                # Get Bayesian parameters
                bayesian_params = await self.bayesian_engine.get_channel_priors(org_id, channel_id)
                
                # Get saturation model uncertainty
                saturation_model = await self.saturation_engine.get_channel_saturation(org_id, channel_id)
                
                uncertainty_params[channel_id] = {
                    "roas_mean": bayesian_params.get("roas_mean", 3.5),
                    "roas_std": bayesian_params.get("roas_std", 0.5),
                    "conversion_rate_mean": bayesian_params.get("conversion_rate_mean", 0.02),
                    "conversion_rate_std": bayesian_params.get("conversion_rate_std", 0.005),
                    "saturation_uncertainty": saturation_model.get("confidence_interval", (0.8, 1.2)),
                    "seasonal_variance": bayesian_params.get("seasonal_variance", 0.1),
                    "market_volatility": bayesian_params.get("market_volatility", 0.15)
                }
                
            except Exception as e:
                logger.warning(f"Failed to get uncertainty params for {channel_id}: {str(e)}")
                uncertainty_params[channel_id] = self._get_default_uncertainty_params(channel_id)
        
        return uncertainty_params
    
    async def _run_scenario_simulation(
        self, 
        scenario: Dict[str, Any], 
        baseline_data: Dict[str, Any],
        uncertainty_params: Dict[str, Any],
        monte_carlo_samples: int
    ) -> ScenarioResult:
        """Run Monte Carlo simulation for a single scenario"""
        
        scenario_type = ScenarioType(scenario.get("scenario_type", "budget_change"))
        scenario_name = scenario.get("name", f"Scenario {scenario_type}")
        scenario_description = scenario.get("description", "")
        
        # Initialize results storage
        simulation_results = {
            "revenue": [],
            "roas": [],
            "conversions": [],
            "cac": [],
            "total_spend": []
        }
        
        # Run Monte Carlo simulation
        for sample in range(monte_carlo_samples):
            sample_result = self._simulate_single_sample(
                scenario, baseline_data, uncertainty_params
            )
            
            for metric, value in sample_result.items():
                simulation_results[metric].append(value)
        
        # Calculate statistics
        statistics = self._calculate_simulation_statistics(simulation_results)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(simulation_results)
        
        # Calculate scenario-specific metrics
        scenario_metrics = self._calculate_scenario_metrics(scenario, baseline_data, statistics)
        
        return ScenarioResult(
            scenario_id=scenario.get("scenario_id", f"scenario_{hash(scenario_name)}"),
            scenario_name=scenario_name,
            scenario_type=scenario_type,
            scenario_description=scenario_description,
            scenario_parameters=scenario.get("parameters", {}),
            expected_performance=statistics,
            confidence_intervals=confidence_intervals,
            probability_distributions=self._create_probability_distributions(simulation_results),
            risk_metrics=self._calculate_scenario_risk_metrics(simulation_results, baseline_data),
            scenario_metrics=scenario_metrics,
            monte_carlo_samples=monte_carlo_samples
        )
    
    def _simulate_single_sample(
        self, 
        scenario: Dict[str, Any], 
        baseline_data: Dict[str, Any],
        uncertainty_params: Dict[str, Any]
    ) -> Dict[str, float]:
        """Simulate a single Monte Carlo sample"""
        
        scenario_type = ScenarioType(scenario.get("scenario_type", "budget_change"))
        parameters = scenario.get("parameters", {})
        
        total_revenue = 0.0
        total_spend = 0.0
        total_conversions = 0.0
        
        # Simulate each channel
        for channel_id, baseline in baseline_data.items():
            if channel_id == "totals" or not isinstance(baseline, dict):
                continue
            
            # Get uncertainty parameters for this channel
            uncertainty = uncertainty_params[channel_id]
            
            # Apply scenario changes
            if scenario_type == ScenarioType.BUDGET_CHANGE:
                spend_multiplier = parameters.get("budget_changes", {}).get(channel_id, 1.0)
                channel_spend = baseline["current_spend"] * spend_multiplier
            elif scenario_type == ScenarioType.MARKET_SHOCK:
                market_impact = parameters.get("market_impact", 0.0)
                channel_spend = baseline["current_spend"] * (1 + market_impact)
            elif scenario_type == ScenarioType.SEASONALITY:
                seasonal_factor = parameters.get("seasonal_factors", {}).get(channel_id, 1.0)
                channel_spend = baseline["current_spend"] * seasonal_factor
            elif scenario_type == ScenarioType.COMPETITIVE_RESPONSE:
                competitive_pressure = parameters.get("competitive_pressure", {}).get(channel_id, 0.0)
                channel_spend = baseline["current_spend"] * (1 + competitive_pressure)
            else:
                channel_spend = baseline["current_spend"]
            
            # Add uncertainty to ROAS
            roas_sample = np.random.normal(
                uncertainty["roas_mean"], 
                uncertainty["roas_std"]
            )
            roas_sample = max(0.1, roas_sample)  # Ensure positive ROAS
            
            # Add uncertainty to conversion rate
            conversion_rate_sample = np.random.normal(
                uncertainty["conversion_rate_mean"],
                uncertainty["conversion_rate_std"]
            )
            conversion_rate_sample = max(0.001, conversion_rate_sample)  # Ensure positive rate
            
            # Apply saturation effects
            saturation_factor = self._apply_saturation_uncertainty(
                channel_spend, baseline["current_spend"], uncertainty["saturation_uncertainty"]
            )
            
            # Apply market volatility
            market_factor = np.random.normal(1.0, uncertainty["market_volatility"])
            market_factor = max(0.1, market_factor)  # Ensure positive factor
            
            # Calculate channel performance
            effective_roas = roas_sample * saturation_factor * market_factor
            channel_revenue = channel_spend * effective_roas
            channel_conversions = channel_spend * conversion_rate_sample
            
            total_spend += channel_spend
            total_revenue += channel_revenue
            total_conversions += channel_conversions
        
        # Calculate aggregate metrics
        overall_roas = total_revenue / max(total_spend, 1.0)
        overall_cac = total_spend / max(total_conversions, 1.0)
        
        return {
            "revenue": total_revenue,
            "roas": overall_roas,
            "conversions": total_conversions,
            "cac": overall_cac,
            "total_spend": total_spend
        }
    
    def _apply_saturation_uncertainty(
        self, 
        new_spend: float, 
        baseline_spend: float, 
        saturation_interval: Tuple[float, float]
    ) -> float:
        """Apply saturation effects with uncertainty"""
        
        spend_ratio = new_spend / max(baseline_spend, 1.0)
        
        # Simple saturation model with uncertainty
        if spend_ratio > 1.0:
            # Diminishing returns for increased spend
            saturation_factor = np.random.uniform(saturation_interval[0], saturation_interval[1])
            return 1.0 + (spend_ratio - 1.0) * saturation_factor
        else:
            # Linear scaling for decreased spend
            return spend_ratio
    
    def _calculate_simulation_statistics(self, simulation_results: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate statistics from simulation results"""
        statistics = {}
        
        for metric, values in simulation_results.items():
            values_array = np.array(values)
            statistics[metric] = {
                "mean": float(np.mean(values_array)),
                "median": float(np.median(values_array)),
                "std": float(np.std(values_array)),
                "min": float(np.min(values_array)),
                "max": float(np.max(values_array)),
                "p25": float(np.percentile(values_array, 25)),
                "p75": float(np.percentile(values_array, 75))
            }
        
        return statistics
    
    def _calculate_confidence_intervals(self, simulation_results: Dict[str, List[float]]) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Calculate confidence intervals for simulation results"""
        confidence_intervals = {}
        
        for metric, values in simulation_results.items():
            values_array = np.array(values)
            confidence_intervals[metric] = {}
            
            for confidence_level in self.confidence_levels:
                lower_percentile = (1 - confidence_level) / 2 * 100
                upper_percentile = (1 + confidence_level) / 2 * 100
                
                lower = float(np.percentile(values_array, lower_percentile))
                upper = float(np.percentile(values_array, upper_percentile))
                
                confidence_intervals[metric][f"{int(confidence_level*100)}%"] = (lower, upper)
        
        return confidence_intervals
    
    def _create_probability_distributions(self, simulation_results: Dict[str, List[float]]) -> Dict[str, Dict[str, Any]]:
        """Create probability distribution summaries"""
        distributions = {}
        
        for metric, values in simulation_results.items():
            values_array = np.array(values)
            
            # Create histogram data
            hist, bin_edges = np.histogram(values_array, bins=50, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            distributions[metric] = {
                "histogram": {
                    "bins": bin_centers.tolist(),
                    "frequencies": hist.tolist()
                },
                "distribution_type": "empirical",
                "skewness": float(self._calculate_skewness(values_array)),
                "kurtosis": float(self._calculate_kurtosis(values_array))
            }
        
        return distributions
    
    def _calculate_skewness(self, values: np.ndarray) -> float:
        """Calculate skewness of distribution"""
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0.0
        return np.mean(((values - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, values: np.ndarray) -> float:
        """Calculate kurtosis of distribution"""
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0.0
        return np.mean(((values - mean) / std) ** 4) - 3
    
    def _calculate_scenario_risk_metrics(
        self, 
        simulation_results: Dict[str, List[float]], 
        baseline_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate risk metrics for scenario"""
        
        revenue_values = np.array(simulation_results["revenue"])
        baseline_revenue = baseline_data["totals"]["total_revenue"]
        
        # Value at Risk (VaR)
        var_95 = float(np.percentile(revenue_values, 5))
        var_99 = float(np.percentile(revenue_values, 1))
        
        # Expected Shortfall (Conditional VaR)
        es_95 = float(np.mean(revenue_values[revenue_values <= var_95]))
        es_99 = float(np.mean(revenue_values[revenue_values <= var_99]))
        
        # Probability of loss
        prob_loss = float(np.mean(revenue_values < baseline_revenue))
        
        # Maximum drawdown
        max_loss = float(baseline_revenue - np.min(revenue_values))
        max_loss_percent = max_loss / max(baseline_revenue, 1.0) * 100
        
        return {
            "value_at_risk_95": var_95,
            "value_at_risk_99": var_99,
            "expected_shortfall_95": es_95,
            "expected_shortfall_99": es_99,
            "probability_of_loss": prob_loss,
            "maximum_loss": max_loss,
            "maximum_loss_percent": max_loss_percent,
            "volatility": float(np.std(revenue_values)),
            "downside_deviation": float(np.std(revenue_values[revenue_values < baseline_revenue]))
        }
    
    def _calculate_scenario_metrics(
        self, 
        scenario: Dict[str, Any], 
        baseline_data: Dict[str, Any], 
        statistics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate scenario-specific metrics"""
        
        baseline_revenue = baseline_data["totals"]["total_revenue"]
        baseline_roas = baseline_data["totals"]["overall_roas"]
        
        expected_revenue = statistics["revenue"]["mean"]
        expected_roas = statistics["roas"]["mean"]
        
        return {
            "revenue_change": expected_revenue - baseline_revenue,
            "revenue_change_percent": ((expected_revenue - baseline_revenue) / max(baseline_revenue, 1.0)) * 100,
            "roas_change": expected_roas - baseline_roas,
            "roas_change_percent": ((expected_roas - baseline_roas) / max(baseline_roas, 1.0)) * 100,
            "probability_of_improvement": self._calculate_improvement_probability(statistics, baseline_data),
            "expected_roi": ((expected_revenue - baseline_revenue) / max(baseline_revenue, 1.0)) * 100,
            "risk_adjusted_return": expected_revenue / max(statistics["revenue"]["std"], 1.0)
        }
    
    def _calculate_improvement_probability(
        self, 
        statistics: Dict[str, Dict[str, float]], 
        baseline_data: Dict[str, Any]
    ) -> float:
        """Calculate probability of improvement over baseline"""
        
        # Simplified calculation - in practice, would use the full distribution
        expected_revenue = statistics["revenue"]["mean"]
        revenue_std = statistics["revenue"]["std"]
        baseline_revenue = baseline_data["totals"]["total_revenue"]
        
        if revenue_std == 0:
            return 1.0 if expected_revenue > baseline_revenue else 0.0
        
        # Assume normal distribution
        z_score = (baseline_revenue - expected_revenue) / revenue_std
        probability = 1 - norm.cdf(z_score)
        
        return float(probability)
    
    def _generate_comparative_analysis(
        self, 
        scenario_results: List[ScenarioResult], 
        baseline_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comparative analysis across scenarios"""
        
        if not scenario_results:
            return {}
        
        # Rank scenarios by expected revenue
        ranked_scenarios = sorted(
            scenario_results, 
            key=lambda x: x.expected_performance["revenue"]["mean"], 
            reverse=True
        )
        
        # Find best and worst scenarios
        best_scenario = ranked_scenarios[0]
        worst_scenario = ranked_scenarios[-1]
        
        # Calculate scenario correlations (simplified)
        scenario_correlations = self._calculate_scenario_correlations(scenario_results)
        
        return {
            "total_scenarios": len(scenario_results),
            "best_scenario": {
                "name": best_scenario.scenario_name,
                "expected_revenue": best_scenario.expected_performance["revenue"]["mean"],
                "improvement_over_baseline": best_scenario.scenario_metrics["revenue_change_percent"]
            },
            "worst_scenario": {
                "name": worst_scenario.scenario_name,
                "expected_revenue": worst_scenario.expected_performance["revenue"]["mean"],
                "change_from_baseline": worst_scenario.scenario_metrics["revenue_change_percent"]
            },
            "scenario_rankings": [
                {
                    "rank": i + 1,
                    "scenario_name": scenario.scenario_name,
                    "expected_revenue": scenario.expected_performance["revenue"]["mean"],
                    "risk_score": scenario.risk_metrics["volatility"]
                }
                for i, scenario in enumerate(ranked_scenarios)
            ],
            "scenario_correlations": scenario_correlations,
            "diversification_benefit": self._calculate_diversification_benefit(scenario_results)
        }
    
    def _calculate_scenario_correlations(self, scenario_results: List[ScenarioResult]) -> Dict[str, float]:
        """Calculate correlations between scenarios (simplified)"""
        # Placeholder implementation
        return {
            "average_correlation": 0.3,
            "max_correlation": 0.8,
            "min_correlation": -0.2
        }
    
    def _calculate_diversification_benefit(self, scenario_results: List[ScenarioResult]) -> float:
        """Calculate diversification benefit across scenarios"""
        # Placeholder implementation
        return 0.15
    
    def _calculate_risk_analysis(self, scenario_results: List[ScenarioResult]) -> Dict[str, Any]:
        """Calculate overall risk analysis"""
        
        if not scenario_results:
            return {}
        
        # Aggregate risk metrics
        all_volatilities = [scenario.risk_metrics["volatility"] for scenario in scenario_results]
        all_vars = [scenario.risk_metrics["value_at_risk_95"] for scenario in scenario_results]
        all_loss_probs = [scenario.risk_metrics["probability_of_loss"] for scenario in scenario_results]
        
        return {
            "portfolio_volatility": {
                "mean": float(np.mean(all_volatilities)),
                "min": float(np.min(all_volatilities)),
                "max": float(np.max(all_volatilities))
            },
            "value_at_risk": {
                "mean": float(np.mean(all_vars)),
                "min": float(np.min(all_vars)),
                "max": float(np.max(all_vars))
            },
            "loss_probability": {
                "mean": float(np.mean(all_loss_probs)),
                "min": float(np.min(all_loss_probs)),
                "max": float(np.max(all_loss_probs))
            },
            "risk_concentration": self._calculate_risk_concentration(scenario_results),
            "tail_risk": self._calculate_tail_risk(scenario_results)
        }
    
    def _calculate_risk_concentration(self, scenario_results: List[ScenarioResult]) -> float:
        """Calculate risk concentration measure"""
        # Placeholder implementation
        return 0.25
    
    def _calculate_tail_risk(self, scenario_results: List[ScenarioResult]) -> Dict[str, float]:
        """Calculate tail risk measures"""
        # Placeholder implementation
        return {
            "tail_expectation": -50000.0,
            "tail_probability": 0.05
        }
    
    def _calculate_sensitivity_analysis(self, scenario_results: List[ScenarioResult]) -> Dict[str, Any]:
        """Calculate sensitivity analysis"""
        
        # Placeholder implementation
        return {
            "most_sensitive_channels": ["google_ads", "meta_ads"],
            "sensitivity_scores": {
                "google_ads": 0.8,
                "meta_ads": 0.7,
                "tiktok_ads": 0.4
            },
            "parameter_sensitivity": {
                "budget_changes": 0.9,
                "market_conditions": 0.6,
                "seasonality": 0.3
            }
        }
    
    def _generate_simulation_insights(
        self, 
        scenario_results: List[ScenarioResult], 
        comparative_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate insights from simulation results"""
        
        insights = []
        
        if not scenario_results:
            return ["No scenarios were successfully simulated."]
        
        # Best scenario insight
        if comparative_analysis.get("best_scenario"):
            best = comparative_analysis["best_scenario"]
            insights.append(
                f"The '{best['name']}' scenario shows the highest potential with "
                f"{best['improvement_over_baseline']:.1f}% improvement over baseline."
            )
        
        # Risk insight
        high_risk_scenarios = [
            s for s in scenario_results 
            if s.risk_metrics["probability_of_loss"] > 0.3
        ]
        
        if high_risk_scenarios:
            insights.append(
                f"{len(high_risk_scenarios)} out of {len(scenario_results)} scenarios "
                f"show significant downside risk (>30% probability of loss)."
            )
        
        # Volatility insight
        volatilities = [s.risk_metrics["volatility"] for s in scenario_results]
        avg_volatility = np.mean(volatilities)
        
        if avg_volatility > 100000:
            insights.append(
                "High volatility detected across scenarios. Consider risk mitigation strategies."
            )
        elif avg_volatility < 50000:
            insights.append(
                "Low volatility across scenarios suggests stable expected outcomes."
            )
        
        # Diversification insight
        if len(scenario_results) > 3:
            insights.append(
                "Multiple scenarios tested provide good coverage of potential outcomes. "
                "Consider implementing a diversified strategy."
            )
        
        return insights
    
    def _generate_scenario_recommendations(self, scenario_results: List[ScenarioResult]) -> List[str]:
        """Generate recommendations based on scenario results"""
        
        recommendations = []
        
        if not scenario_results:
            return ["Unable to generate recommendations due to simulation failure."]
        
        # Sort by risk-adjusted return
        sorted_scenarios = sorted(
            scenario_results,
            key=lambda x: x.scenario_metrics.get("risk_adjusted_return", 0),
            reverse=True
        )
        
        best_scenario = sorted_scenarios[0]
        
        recommendations.append(
            f"Recommend implementing '{best_scenario.scenario_name}' scenario "
            f"for optimal risk-adjusted returns."
        )
        
        # Risk management recommendation
        high_risk_scenarios = [
            s for s in scenario_results 
            if s.risk_metrics["probability_of_loss"] > 0.25
        ]
        
        if high_risk_scenarios:
            recommendations.append(
                "Implement risk monitoring and contingency plans for high-risk scenarios."
            )
        
        # Gradual implementation recommendation
        if best_scenario.scenario_metrics.get("revenue_change_percent", 0) > 20:
            recommendations.append(
                "Consider gradual implementation of budget changes to minimize risk."
            )
        
        return recommendations
    
    # Helper methods
    
    def _get_default_baseline(self, channel_id: str) -> Dict[str, float]:
        """Get default baseline data for a channel"""
        return {
            "current_spend": 1000.0,
            "current_revenue": 3500.0,
            "current_roas": 3.5,
            "current_conversions": 100,
            "current_cac": 10.0
        }
    
    def _get_default_uncertainty_params(self, channel_id: str) -> Dict[str, Any]:
        """Get default uncertainty parameters for a channel"""
        return {
            "roas_mean": 3.5,
            "roas_std": 0.5,
            "conversion_rate_mean": 0.02,
            "conversion_rate_std": 0.005,
            "saturation_uncertainty": (0.8, 1.2),
            "seasonal_variance": 0.1,
            "market_volatility": 0.15
        }