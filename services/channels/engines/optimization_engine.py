"""
Optimization Engine for Channels Service
Advanced multi-objective optimization with Bayesian inference
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

from models.channels import (
    BudgetOptimizationRequest, OptimizationResult, OptimizationAction,
    OptimizationConstraint, ChannelPerformance, OptimizationObjective,
    OptimizationStatus, ConstraintType
)

logger = logging.getLogger(__name__)


class OptimizationEngine:
    """Advanced multi-objective optimization engine for budget allocation"""
    
    def __init__(self, saturation_engine, bayesian_engine, causal_client):
        self.saturation_engine = saturation_engine
        self.bayesian_engine = bayesian_engine
        self.causal_client = causal_client
        
        # Optimization parameters
        self.max_iterations = 1000
        self.convergence_tolerance = 1e-6
        self.population_size = 50
        
        # Multi-objective weights (default)
        self.default_objective_weights = {
            OptimizationObjective.MAXIMIZE_REVENUE: 0.4,
            OptimizationObjective.MAXIMIZE_ROAS: 0.3,
            OptimizationObjective.MINIMIZE_CAC: 0.2,
            OptimizationObjective.MINIMIZE_RISK: 0.1
        }
    
    async def optimize_budget(
        self, 
        org_id: str, 
        request: BudgetOptimizationRequest,
        optimization_id: str
    ) -> OptimizationResult:
        """Main budget optimization function"""
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Starting budget optimization {optimization_id} for org {org_id}")
            
            # 1. Gather current channel performance data
            channel_data = await self._get_channel_data(org_id, request.channels)
            
            # 2. Get causal lift estimates
            causal_effects = await self._get_causal_effects(org_id, request.channels)
            
            # 3. Get saturation models
            saturation_models = await self._get_saturation_models(org_id, request.channels)
            
            # 4. Get Bayesian priors and uncertainty
            bayesian_params = await self._get_bayesian_parameters(org_id, request.channels)
            
            # 5. Set up optimization problem
            optimization_problem = self._setup_optimization_problem(
                request, channel_data, causal_effects, saturation_models, bayesian_params
            )
            
            # 6. Run optimization
            if request.use_bayesian_optimization:
                optimization_result = await self._run_bayesian_optimization(optimization_problem)
            else:
                optimization_result = await self._run_classical_optimization(optimization_problem)
            
            # 7. Generate implementation plan
            implementation_plan = await self._generate_implementation_plan(
                optimization_result, channel_data, request
            )
            
            # 8. Calculate confidence intervals and risk metrics
            confidence_intervals, risk_metrics = await self._calculate_uncertainty_metrics(
                optimization_result, bayesian_params, request.monte_carlo_samples
            )
            
            # 9. Create optimization result
            result = OptimizationResult(
                optimization_id=optimization_id,
                org_id=org_id,
                status=OptimizationStatus.COMPLETED,
                total_budget=request.total_budget,
                channels_optimized=request.channels,
                objectives=request.objectives,
                recommended_allocation=optimization_result["allocation"],
                current_allocation=optimization_result["current_allocation"],
                expected_performance=optimization_result["expected_performance"],
                performance_improvement=optimization_result["improvement"],
                confidence_intervals=confidence_intervals,
                risk_metrics=risk_metrics,
                overall_confidence=optimization_result["confidence"],
                implementation_plan=implementation_plan,
                estimated_implementation_time=self._estimate_implementation_time(implementation_plan),
                algorithm_used=optimization_result["algorithm"],
                convergence_status=optimization_result["convergence"],
                iterations=optimization_result["iterations"],
                computation_time=(datetime.utcnow() - start_time).total_seconds(),
                completed_at=datetime.utcnow()
            )
            
            logger.info(f"Completed budget optimization {optimization_id}")
            return result
            
        except Exception as e:
            logger.error(f"Budget optimization failed: {str(e)}")
            # Return failed result
            return OptimizationResult(
                optimization_id=optimization_id,
                org_id=org_id,
                status=OptimizationStatus.FAILED,
                total_budget=request.total_budget,
                channels_optimized=request.channels,
                objectives=request.objectives,
                recommended_allocation={},
                current_allocation={},
                expected_performance={},
                performance_improvement={},
                confidence_intervals={},
                risk_metrics={},
                overall_confidence=0.0,
                implementation_plan=[],
                estimated_implementation_time="N/A",
                algorithm_used="failed",
                convergence_status="failed",
                iterations=0,
                computation_time=(datetime.utcnow() - start_time).total_seconds(),
                completed_at=datetime.utcnow()
            )
    
    async def _get_channel_data(self, org_id: str, channels: List[str]) -> Dict[str, ChannelPerformance]:
        """Get current channel performance data"""
        channel_data = {}
        
        for channel_id in channels:
            try:
                # Get performance data from data ingestion service
                performance = await self._fetch_channel_performance(org_id, channel_id)
                channel_data[channel_id] = performance
            except Exception as e:
                logger.warning(f"Failed to get data for channel {channel_id}: {str(e)}")
                # Use default/estimated values
                channel_data[channel_id] = self._get_default_channel_performance(channel_id)
        
        return channel_data
    
    async def _get_causal_effects(self, org_id: str, channels: List[str]) -> Dict[str, float]:
        """Get causal lift estimates from causal service"""
        causal_effects = {}
        
        try:
            # Call causal service for lift estimates
            for channel_id in channels:
                effect = await self.causal_client.get_channel_lift(org_id, channel_id)
                causal_effects[channel_id] = effect.get("lift_coefficient", 1.0)
        except Exception as e:
            logger.warning(f"Failed to get causal effects: {str(e)}")
            # Use default values
            for channel_id in channels:
                causal_effects[channel_id] = 1.0
        
        return causal_effects
    
    async def _get_saturation_models(self, org_id: str, channels: List[str]) -> Dict[str, Any]:
        """Get saturation models for channels"""
        saturation_models = {}
        
        for channel_id in channels:
            try:
                model = await self.saturation_engine.get_channel_saturation(org_id, channel_id)
                saturation_models[channel_id] = model
            except Exception as e:
                logger.warning(f"Failed to get saturation model for {channel_id}: {str(e)}")
                # Use default saturation model
                saturation_models[channel_id] = self._get_default_saturation_model(channel_id)
        
        return saturation_models
    
    async def _get_bayesian_parameters(self, org_id: str, channels: List[str]) -> Dict[str, Any]:
        """Get Bayesian parameters and uncertainty estimates"""
        bayesian_params = {}
        
        try:
            for channel_id in channels:
                params = await self.bayesian_engine.get_channel_priors(org_id, channel_id)
                bayesian_params[channel_id] = params
        except Exception as e:
            logger.warning(f"Failed to get Bayesian parameters: {str(e)}")
            # Use default parameters
            for channel_id in channels:
                bayesian_params[channel_id] = self._get_default_bayesian_params(channel_id)
        
        return bayesian_params
    
    def _setup_optimization_problem(
        self, 
        request: BudgetOptimizationRequest,
        channel_data: Dict[str, ChannelPerformance],
        causal_effects: Dict[str, float],
        saturation_models: Dict[str, Any],
        bayesian_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Set up the optimization problem"""
        
        # Prepare objective weights
        objective_weights = request.objective_weights or self.default_objective_weights
        
        # Normalize weights
        total_weight = sum(objective_weights.values())
        if total_weight > 0:
            objective_weights = {k: v/total_weight for k, v in objective_weights.items()}
        
        # Set up bounds (min/max spend per channel)
        bounds = []
        channel_indices = {channel: i for i, channel in enumerate(request.channels)}
        
        for channel_id in request.channels:
            channel_perf = channel_data[channel_id]
            
            # Default bounds: 10% to 200% of current spend
            min_spend = max(0.1 * channel_perf.current_spend, 1.0)
            max_spend = min(2.0 * channel_perf.current_spend, request.total_budget * 0.8)
            
            # Apply constraint-based bounds
            for constraint in request.constraints:
                if constraint.channel_id == channel_id:
                    if constraint.constraint_type == ConstraintType.MIN_SPEND and constraint.min_value:
                        min_spend = max(min_spend, constraint.min_value)
                    elif constraint.constraint_type == ConstraintType.MAX_SPEND and constraint.max_value:
                        max_spend = min(max_spend, constraint.max_value)
            
            bounds.append((min_spend, max_spend))
        
        # Set up constraints
        constraints = self._setup_constraints(request, channel_indices)
        
        return {
            "channels": request.channels,
            "channel_indices": channel_indices,
            "channel_data": channel_data,
            "causal_effects": causal_effects,
            "saturation_models": saturation_models,
            "bayesian_params": bayesian_params,
            "objectives": request.objectives,
            "objective_weights": objective_weights,
            "total_budget": request.total_budget,
            "bounds": bounds,
            "constraints": constraints,
            "risk_tolerance": request.risk_tolerance,
            "confidence_threshold": request.confidence_threshold
        }
    
    def _setup_constraints(self, request: BudgetOptimizationRequest, channel_indices: Dict[str, int]) -> List[Dict]:
        """Set up optimization constraints"""
        constraints = []
        
        # Budget constraint (equality)
        def budget_constraint(x):
            return request.total_budget - np.sum(x)
        
        constraints.append({
            'type': 'eq',
            'fun': budget_constraint
        })
        
        # Additional constraints from request
        # Note: MIN_SPEND and MAX_SPEND are handled in bounds, not constraints
        for constraint in request.constraints:
            if constraint.constraint_type == ConstraintType.MIN_ROAS:
                # Minimum ROAS constraint
                def min_roas_constraint(x, channel_id=constraint.channel_id, min_roas=constraint.min_value):
                    if channel_id and channel_id in channel_indices:
                        idx = channel_indices[channel_id]
                        # Simplified ROAS calculation
                        return self._calculate_channel_roas(x[idx], channel_id) - min_roas
                    return 0
                
                constraints.append({
                    'type': 'ineq',
                    'fun': min_roas_constraint
                })
            elif constraint.constraint_type == ConstraintType.MAX_CAC:
                # Maximum CAC constraint
                def max_cac_constraint(x, channel_id=constraint.channel_id, max_cac=constraint.max_value):
                    if channel_id and channel_id in channel_indices:
                        idx = channel_indices[channel_id]
                        # Simplified CAC calculation
                        return max_cac - self._calculate_channel_cac(x[idx], channel_id)
                    return 0
                
                constraints.append({
                    'type': 'ineq',
                    'fun': max_cac_constraint
                })
            # MIN_SPEND and MAX_SPEND are handled in bounds, not constraints
            # Other constraint types (GEOGRAPHIC, TEMPORAL, etc.) not yet implemented
        
        return constraints
    
    async def _run_bayesian_optimization(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Run Bayesian optimization"""
        logger.info("Running Bayesian optimization")
        
        # For now, use differential evolution as a proxy for Bayesian optimization
        # In a full implementation, you would use libraries like scikit-optimize
        return await self._run_classical_optimization(problem, method="differential_evolution")
    
    async def _run_classical_optimization(self, problem: Dict[str, Any], method: str = "SLSQP") -> Dict[str, Any]:
        """Run classical optimization"""
        logger.info(f"Running classical optimization with method: {method}")
        
        # Initial guess: current allocation
        x0 = np.array([
            problem["channel_data"][channel].current_spend 
            for channel in problem["channels"]
        ])
        
        # Scale to match total budget
        x0 = x0 * (problem["total_budget"] / np.sum(x0))
        
        # Objective function
        def objective(x):
            return -self._evaluate_multi_objective(x, problem)
        
        try:
            if method == "differential_evolution":
                # Global optimization - differential_evolution doesn't support constraints directly
                # We'll use bounds and handle constraints in the objective function
                def constrained_objective(x):
                    # Check constraints
                    for constraint in problem["constraints"]:
                        if constraint['type'] == 'eq':
                            if abs(constraint['fun'](x)) > 1e-6:
                                return 1e10  # Large penalty for constraint violation
                        elif constraint['type'] == 'ineq':
                            if constraint['fun'](x) < 0:
                                return 1e10  # Large penalty for constraint violation
                    return objective(x)
                
                result = differential_evolution(
                    constrained_objective,
                    bounds=problem["bounds"],
                    maxiter=self.max_iterations,
                    popsize=15,
                    seed=42
                )
            else:
                # Local optimization
                result = minimize(
                    objective,
                    x0,
                    method=method,
                    bounds=problem["bounds"],
                    constraints=problem["constraints"],
                    options={'maxiter': self.max_iterations}
                )
            
            if result.success:
                optimal_allocation = dict(zip(problem["channels"], result.x))
                current_allocation = dict(zip(
                    problem["channels"], 
                    [problem["channel_data"][ch].current_spend for ch in problem["channels"]]
                ))
                
                # Calculate expected performance
                expected_performance = self._calculate_expected_performance(result.x, problem)
                current_performance = self._calculate_expected_performance(x0, problem)
                
                # Calculate improvement
                improvement = {
                    metric: expected_performance[metric] - current_performance[metric]
                    for metric in expected_performance.keys()
                }
                
                return {
                    "allocation": optimal_allocation,
                    "current_allocation": current_allocation,
                    "expected_performance": expected_performance,
                    "improvement": improvement,
                    "confidence": 0.85,  # Placeholder
                    "algorithm": method,
                    "convergence": "converged" if result.success else "failed",
                    "iterations": result.nit if hasattr(result, 'nit') else 0
                }
            else:
                raise Exception(f"Optimization failed: {result.message}")
                
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            raise
    
    def _evaluate_multi_objective(self, x: np.ndarray, problem: Dict[str, Any]) -> float:
        """Evaluate multi-objective function"""
        total_score = 0.0
        
        for objective in problem["objectives"]:
            weight = problem["objective_weights"].get(objective, 0.0)
            
            if objective == OptimizationObjective.MAXIMIZE_REVENUE:
                score = self._calculate_total_revenue(x, problem)
            elif objective == OptimizationObjective.MAXIMIZE_ROAS:
                score = self._calculate_weighted_roas(x, problem)
            elif objective == OptimizationObjective.MINIMIZE_CAC:
                score = -self._calculate_weighted_cac(x, problem)  # Negative for minimization
            elif objective == OptimizationObjective.MINIMIZE_RISK:
                score = -self._calculate_portfolio_risk(x, problem)  # Negative for minimization
            else:
                score = 0.0
            
            total_score += weight * score
        
        return total_score
    
    def _calculate_total_revenue(self, x: np.ndarray, problem: Dict[str, Any]) -> float:
        """Calculate total expected revenue"""
        total_revenue = 0.0
        
        for i, channel_id in enumerate(problem["channels"]):
            spend = x[i]
            
            # Get saturation-adjusted response
            saturation_model = problem["saturation_models"][channel_id]
            response = saturation_model.evaluate(spend)
            
            # Apply causal effect
            causal_effect = problem["causal_effects"][channel_id]
            adjusted_response = response * causal_effect
            
            # Convert to revenue (simplified)
            channel_data = problem["channel_data"][channel_id]
            revenue_per_response = channel_data.current_revenue / max(channel_data.current_spend, 1.0)
            
            total_revenue += adjusted_response * revenue_per_response
        
        return total_revenue
    
    def _calculate_weighted_roas(self, x: np.ndarray, problem: Dict[str, Any]) -> float:
        """Calculate weighted ROAS across channels"""
        total_revenue = self._calculate_total_revenue(x, problem)
        total_spend = np.sum(x)
        
        return total_revenue / max(total_spend, 1.0)
    
    def _calculate_weighted_cac(self, x: np.ndarray, problem: Dict[str, Any]) -> float:
        """Calculate weighted CAC across channels"""
        total_conversions = 0.0
        total_spend = np.sum(x)
        
        for i, channel_id in enumerate(problem["channels"]):
            spend = x[i]
            
            # Get saturation-adjusted conversions
            saturation_model = problem["saturation_models"][channel_id]
            response = saturation_model.evaluate(spend)
            
            # Apply causal effect
            causal_effect = problem["causal_effects"][channel_id]
            adjusted_conversions = response * causal_effect
            
            total_conversions += adjusted_conversions
        
        return total_spend / max(total_conversions, 1.0)
    
    def _calculate_portfolio_risk(self, x: np.ndarray, problem: Dict[str, Any]) -> float:
        """Calculate portfolio risk (variance of returns)"""
        # Simplified risk calculation based on allocation concentration
        allocation_shares = x / np.sum(x)
        
        # Herfindahl index (concentration measure)
        concentration = np.sum(allocation_shares ** 2)
        
        # Risk increases with concentration
        return concentration
    
    def _calculate_channel_roas(self, spend: float, channel_id: str) -> float:
        """Calculate ROAS for a specific channel (simplified)"""
        # Placeholder implementation
        return max(1.0, 5.0 - spend / 1000.0)  # Decreasing ROAS with spend
    
    def _calculate_channel_cac(self, spend: float, channel_id: str) -> float:
        """Calculate CAC (Customer Acquisition Cost) for a channel given spend"""
        # Simplified CAC calculation - CAC typically increases with spend due to saturation
        # Placeholder implementation
        return max(1.0, 50.0 + (spend / 1000))  # Increasing CAC with spend
    
    def _calculate_expected_performance(self, x: np.ndarray, problem: Dict[str, Any]) -> Dict[str, float]:
        """Calculate expected performance metrics"""
        return {
            "total_revenue": self._calculate_total_revenue(x, problem),
            "total_roas": self._calculate_weighted_roas(x, problem),
            "total_cac": self._calculate_weighted_cac(x, problem),
            "portfolio_risk": self._calculate_portfolio_risk(x, problem)
        }
    
    async def _generate_implementation_plan(
        self, 
        optimization_result: Dict[str, Any],
        channel_data: Dict[str, ChannelPerformance],
        request: BudgetOptimizationRequest
    ) -> List[OptimizationAction]:
        """Generate detailed implementation plan"""
        actions = []
        
        for channel_id, recommended_budget in optimization_result["allocation"].items():
            current_budget = channel_data[channel_id].current_spend
            budget_change = recommended_budget - current_budget
            budget_change_percent = (budget_change / max(current_budget, 1.0)) * 100
            
            # Determine action type
            if abs(budget_change_percent) < 5:
                action_type = "maintain"
                priority = 3
                risk_level = "low"
            elif budget_change > 0:
                action_type = "increase"
                priority = 1 if budget_change_percent > 20 else 2
                risk_level = "medium" if budget_change_percent > 50 else "low"
            else:
                action_type = "decrease"
                priority = 2
                risk_level = "low"
            
            # Calculate expected impact (simplified)
            expected_revenue_impact = budget_change * 2.5  # Placeholder multiplier
            expected_conversion_impact = int(budget_change / 10)  # Placeholder
            expected_roas_change = -0.1 if budget_change > 0 else 0.1  # Diminishing returns
            
            action = OptimizationAction(
                action_id=f"action_{channel_id}_{len(actions)}",
                channel_id=channel_id,
                action_type=action_type,
                current_budget=current_budget,
                recommended_budget=recommended_budget,
                budget_change=budget_change,
                budget_change_percent=budget_change_percent,
                expected_revenue_impact=expected_revenue_impact,
                expected_conversion_impact=expected_conversion_impact,
                expected_roas_change=expected_roas_change,
                priority=priority,
                implementation_timeline=self._get_implementation_timeline(priority),
                risk_level=risk_level,
                confidence_score=0.8,  # Placeholder
                confidence_interval=(expected_revenue_impact * 0.8, expected_revenue_impact * 1.2)
            )
            
            actions.append(action)
        
        # Sort by priority
        actions.sort(key=lambda x: x.priority)
        
        return actions
    
    def _get_implementation_timeline(self, priority: int) -> str:
        """Get implementation timeline based on priority"""
        timelines = {
            1: "Immediate (within 24 hours)",
            2: "Short-term (within 1 week)",
            3: "Medium-term (within 2 weeks)",
            4: "Long-term (within 1 month)",
            5: "Strategic (within 3 months)"
        }
        return timelines.get(priority, "Medium-term")
    
    def _estimate_implementation_time(self, implementation_plan: List[OptimizationAction]) -> str:
        """Estimate total implementation time"""
        if not implementation_plan:
            return "N/A"
        
        high_priority_actions = len([a for a in implementation_plan if a.priority <= 2])
        
        if high_priority_actions > 5:
            return "2-3 weeks"
        elif high_priority_actions > 2:
            return "1-2 weeks"
        else:
            return "3-5 days"
    
    async def _calculate_uncertainty_metrics(
        self, 
        optimization_result: Dict[str, Any],
        bayesian_params: Dict[str, Any],
        monte_carlo_samples: int
    ) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, float]]:
        """Calculate confidence intervals and risk metrics using Monte Carlo"""
        
        # Placeholder implementation
        confidence_intervals = {}
        risk_metrics = {}
        
        for channel_id, allocation in optimization_result["allocation"].items():
            # Simple confidence interval (±20%)
            lower = allocation * 0.8
            upper = allocation * 1.2
            confidence_intervals[channel_id] = (lower, upper)
        
        # Risk metrics
        risk_metrics = {
            "portfolio_volatility": 0.15,  # Placeholder
            "value_at_risk_95": 0.05,      # Placeholder
            "expected_shortfall": 0.03,    # Placeholder
            "sharpe_ratio": 1.2            # Placeholder
        }
        
        return confidence_intervals, risk_metrics
    
    async def validate_constraints(
        self, 
        org_id: str, 
        constraints: List[OptimizationConstraint]
    ) -> Dict[str, Any]:
        """Validate optimization constraints"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "constraint_summary": {}
        }
        
        for constraint in constraints:
            constraint_validation = self._validate_single_constraint(constraint)
            
            if not constraint_validation["valid"]:
                validation_result["valid"] = False
                validation_result["errors"].extend(constraint_validation["errors"])
            
            validation_result["warnings"].extend(constraint_validation["warnings"])
            validation_result["constraint_summary"][constraint.constraint_id] = constraint_validation
        
        return validation_result
    
    def _validate_single_constraint(self, constraint: OptimizationConstraint) -> Dict[str, Any]:
        """Validate a single constraint"""
        result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Validate constraint parameters
        if constraint.constraint_type in [ConstraintType.MIN_SPEND, ConstraintType.MAX_SPEND]:
            if constraint.min_value is not None and constraint.max_value is not None:
                if constraint.min_value > constraint.max_value:
                    result["valid"] = False
                    result["errors"].append("Minimum value cannot be greater than maximum value")
        
        # Add more validation logic as needed
        
        return result
    
    # Helper methods for default values
    
    async def _fetch_channel_performance(self, org_id: str, channel_id: str) -> ChannelPerformance:
        """Fetch channel performance from data ingestion service"""
        # This would make an actual API call to the data ingestion service
        # For now, return mock data
        return self._get_default_channel_performance(channel_id)
    
    def _get_default_channel_performance(self, channel_id: str) -> ChannelPerformance:
        """Get default channel performance data"""
        from models.channels import ChannelType
        
        return ChannelPerformance(
            channel_id=channel_id,
            channel_name=channel_id.replace("_", " ").title(),
            channel_type=ChannelType.OTHER,
            current_spend=1000.0,
            current_roas=3.5,
            current_conversions=100,
            current_revenue=3500.0,
            current_cac=10.0,
            saturation_level=0.6,
            efficiency_score=0.75,
            trend_direction="stable",
            last_updated=datetime.utcnow()
        )
    
    def _get_default_saturation_model(self, channel_id: str) -> Any:
        """Get default saturation model"""
        from models.channels import SaturationModel, SaturationFunction
        
        return SaturationModel(
            channel_id=channel_id,
            function_type=SaturationFunction.HILL,
            alpha=0.5,
            gamma=1000.0,
            saturation_point=2000.0,
            diminishing_returns_start=500.0,
            max_response=5000.0,
            r_squared=0.85,
            confidence_interval=(0.8, 0.9),
            last_calibrated=datetime.utcnow()
        )
    
    def _get_default_bayesian_params(self, channel_id: str) -> Dict[str, Any]:
        """Get default Bayesian parameters"""
        return {
            "prior_mean": 1.0,
            "prior_std": 0.2,
            "uncertainty": 0.15,
            "confidence": 0.8
        }