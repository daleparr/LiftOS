"""
Causal-Aware Agent Testing Integration

This module integrates the Agentic microservice with LiftOS's causal inference
capabilities to provide causal-aware agent testing and evaluation.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Import LiftOS shared components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from shared.kse_sdk.causal_models import (
    CausalRelationship, CausalRelationType, TemporalDirection, CausalEmbedding
)
from shared.models.causal_marketing import (
    CausalMarketingData, CausalExperiment, ConfounderVariable,
    ExternalFactor, CausalMethod, TreatmentType
)
from shared.utils.logging import get_logger

logger = get_logger(__name__)


class CausalAgentTestingEngine:
    """
    Engine for causal-aware agent testing that considers causal relationships
    when evaluating agent performance and recommendations.
    """
    
    def __init__(self):
        self.causal_relationships: List[CausalRelationship] = []
        self.confounders: List[ConfounderVariable] = []
        self.external_factors: List[ExternalFactor] = []
        
    async def register_causal_relationship(
        self, 
        relationship: CausalRelationship
    ) -> None:
        """Register a causal relationship for consideration in agent testing."""
        self.causal_relationships.append(relationship)
        logger.info(f"Registered causal relationship: {relationship.cause_variable} -> {relationship.effect_variable}")
    
    async def identify_confounders(
        self,
        treatment_variable: str,
        outcome_variable: str,
        data: pd.DataFrame
    ) -> List[str]:
        """
        Identify potential confounders that could bias agent testing results.
        
        Args:
            treatment_variable: The variable representing the agent's action/recommendation
            outcome_variable: The variable representing the outcome being measured
            data: Historical data for analysis
            
        Returns:
            List of potential confounder variable names
        """
        confounders = []
        
        # Find variables that are correlated with both treatment and outcome
        if treatment_variable in data.columns and outcome_variable in data.columns:
            for col in data.columns:
                if col not in [treatment_variable, outcome_variable]:
                    # Check correlation with treatment
                    treatment_corr = data[col].corr(data[treatment_variable])
                    outcome_corr = data[col].corr(data[outcome_variable])
                    
                    # If correlated with both (threshold of 0.3), consider as confounder
                    if abs(treatment_corr) > 0.3 and abs(outcome_corr) > 0.3:
                        confounders.append(col)
                        logger.info(f"Identified potential confounder: {col}")
        
        return confounders
    
    async def design_causal_experiment(
        self,
        agent_id: str,
        treatment_type: TreatmentType,
        outcome_metrics: List[str],
        randomization_unit: str = "campaign",
        duration_days: int = 14
    ) -> CausalExperiment:
        """
        Design a causal experiment to test agent effectiveness.
        
        Args:
            agent_id: ID of the agent being tested
            treatment_type: Type of treatment/intervention
            outcome_metrics: Metrics to measure
            randomization_unit: Unit of randomization
            duration_days: Duration of experiment
            
        Returns:
            Designed causal experiment
        """
        experiment = CausalExperiment(
            experiment_id=f"agent_test_{agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            agent_id=agent_id,
            treatment_type=treatment_type,
            outcome_metrics=outcome_metrics,
            randomization_unit=randomization_unit,
            start_date=datetime.now().date(),
            end_date=(datetime.now() + timedelta(days=duration_days)).date(),
            causal_method=CausalMethod.RANDOMIZED_EXPERIMENT,
            sample_size_calculation={
                "minimum_detectable_effect": 0.05,
                "power": 0.8,
                "significance_level": 0.05
            }
        )
        
        logger.info(f"Designed causal experiment: {experiment.experiment_id}")
        return experiment
    
    async def evaluate_causal_impact(
        self,
        experiment_data: pd.DataFrame,
        treatment_column: str,
        outcome_column: str,
        confounders: List[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the causal impact of an agent's actions using causal inference.
        
        Args:
            experiment_data: Data from the experiment
            treatment_column: Column indicating treatment assignment
            outcome_column: Column with outcome measurements
            confounders: List of confounder variables to control for
            
        Returns:
            Causal impact analysis results
        """
        if confounders is None:
            confounders = []
        
        # Simple difference-in-means for demonstration
        # In production, would use more sophisticated causal inference methods
        treatment_group = experiment_data[experiment_data[treatment_column] == 1]
        control_group = experiment_data[experiment_data[treatment_column] == 0]
        
        treatment_mean = treatment_group[outcome_column].mean()
        control_mean = control_group[outcome_column].mean()
        
        causal_effect = treatment_mean - control_mean
        
        # Calculate confidence interval (simplified)
        treatment_std = treatment_group[outcome_column].std()
        control_std = control_group[outcome_column].std()
        pooled_std = np.sqrt((treatment_std**2 + control_std**2) / 2)
        
        n_treatment = len(treatment_group)
        n_control = len(control_group)
        se = pooled_std * np.sqrt(1/n_treatment + 1/n_control)
        
        ci_lower = causal_effect - 1.96 * se
        ci_upper = causal_effect + 1.96 * se
        
        # Calculate p-value (simplified t-test)
        t_stat = causal_effect / se if se > 0 else 0
        p_value = 2 * (1 - abs(t_stat) / (abs(t_stat) + 1))  # Approximation
        
        results = {
            "causal_effect": causal_effect,
            "confidence_interval": [ci_lower, ci_upper],
            "p_value": p_value,
            "treatment_mean": treatment_mean,
            "control_mean": control_mean,
            "sample_sizes": {
                "treatment": n_treatment,
                "control": n_control
            },
            "confounders_controlled": confounders,
            "statistical_significance": p_value < 0.05
        }
        
        logger.info(f"Causal impact analysis completed. Effect: {causal_effect:.4f}")
        return results
    
    async def validate_agent_causality(
        self,
        agent_recommendations: List[Dict[str, Any]],
        historical_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Validate that agent recommendations respect known causal relationships.
        
        Args:
            agent_recommendations: List of agent recommendations
            historical_data: Historical data for validation
            
        Returns:
            Validation results
        """
        validation_results = {
            "valid_recommendations": [],
            "invalid_recommendations": [],
            "causal_violations": [],
            "overall_validity": True
        }
        
        for rec in agent_recommendations:
            is_valid = True
            violations = []
            
            # Check against known causal relationships
            for relationship in self.causal_relationships:
                if relationship.cause_variable in rec and relationship.effect_variable in rec:
                    # Validate that recommendation respects causal direction
                    cause_value = rec[relationship.cause_variable]
                    effect_value = rec[relationship.effect_variable]
                    
                    # Simple validation: if increasing cause, effect should increase for positive relationships
                    if relationship.strength > 0:
                        if cause_value > 0 and effect_value < 0:
                            violations.append(f"Violates positive causal relationship: {relationship.cause_variable} -> {relationship.effect_variable}")
                            is_valid = False
                    elif relationship.strength < 0:
                        if cause_value > 0 and effect_value > 0:
                            violations.append(f"Violates negative causal relationship: {relationship.cause_variable} -> {relationship.effect_variable}")
                            is_valid = False
            
            if is_valid:
                validation_results["valid_recommendations"].append(rec)
            else:
                validation_results["invalid_recommendations"].append(rec)
                validation_results["causal_violations"].extend(violations)
                validation_results["overall_validity"] = False
        
        logger.info(f"Agent causality validation completed. Valid: {len(validation_results['valid_recommendations'])}, Invalid: {len(validation_results['invalid_recommendations'])}")
        return validation_results
    
    async def generate_causal_insights(
        self,
        agent_performance_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Generate causal insights about agent performance patterns.
        
        Args:
            agent_performance_data: Data about agent performance over time
            
        Returns:
            List of causal insights
        """
        insights = []
        
        # Analyze temporal patterns
        if 'timestamp' in agent_performance_data.columns:
            # Look for time-based causal patterns
            agent_performance_data['hour'] = pd.to_datetime(agent_performance_data['timestamp']).dt.hour
            agent_performance_data['day_of_week'] = pd.to_datetime(agent_performance_data['timestamp']).dt.dayofweek
            
            # Find performance patterns by time
            hourly_performance = agent_performance_data.groupby('hour')['performance_score'].mean()
            best_hour = hourly_performance.idxmax()
            worst_hour = hourly_performance.idxmin()
            
            insights.append({
                "type": "temporal_pattern",
                "insight": f"Agent performs best at hour {best_hour} and worst at hour {worst_hour}",
                "confidence": 0.8,
                "actionable": True,
                "recommendation": f"Schedule critical agent tasks around hour {best_hour}"
            })
        
        # Analyze feature importance for causal relationships
        numeric_columns = agent_performance_data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 1:
            correlation_matrix = agent_performance_data[numeric_columns].corr()
            
            # Find strong correlations that might indicate causal relationships
            for i, col1 in enumerate(numeric_columns):
                for j, col2 in enumerate(numeric_columns):
                    if i < j and abs(correlation_matrix.loc[col1, col2]) > 0.7:
                        insights.append({
                            "type": "correlation_insight",
                            "insight": f"Strong correlation between {col1} and {col2} ({correlation_matrix.loc[col1, col2]:.3f})",
                            "confidence": 0.6,
                            "actionable": True,
                            "recommendation": f"Investigate potential causal relationship between {col1} and {col2}"
                        })
        
        logger.info(f"Generated {len(insights)} causal insights")
        return insights


class CausalAgentValidator:
    """
    Validator that ensures agent recommendations are causally sound.
    """
    
    def __init__(self, causal_engine: CausalAgentTestingEngine):
        self.causal_engine = causal_engine
    
    async def validate_recommendation(
        self,
        recommendation: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate a single agent recommendation for causal soundness.
        
        Args:
            recommendation: The agent's recommendation
            context: Context data for validation
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for causal consistency
        validation_result = await self.causal_engine.validate_agent_causality(
            [recommendation], 
            pd.DataFrame([context])
        )
        
        if not validation_result["overall_validity"]:
            issues.extend(validation_result["causal_violations"])
        
        # Check for temporal consistency
        if 'timestamp' in recommendation and 'timestamp' in context:
            rec_time = pd.to_datetime(recommendation['timestamp'])
            context_time = pd.to_datetime(context['timestamp'])
            
            if rec_time < context_time:
                issues.append("Recommendation timestamp is before context timestamp")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    async def batch_validate_recommendations(
        self,
        recommendations: List[Dict[str, Any]],
        context_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Validate multiple recommendations in batch.
        
        Args:
            recommendations: List of agent recommendations
            context_data: Context data for validation
            
        Returns:
            Batch validation results
        """
        results = {
            "total_recommendations": len(recommendations),
            "valid_count": 0,
            "invalid_count": 0,
            "validation_details": []
        }
        
        for i, rec in enumerate(recommendations):
            context = context_data.iloc[i % len(context_data)].to_dict()
            is_valid, issues = await self.validate_recommendation(rec, context)
            
            results["validation_details"].append({
                "recommendation_index": i,
                "is_valid": is_valid,
                "issues": issues
            })
            
            if is_valid:
                results["valid_count"] += 1
            else:
                results["invalid_count"] += 1
        
        results["validity_rate"] = results["valid_count"] / results["total_recommendations"]
        
        logger.info(f"Batch validation completed. Validity rate: {results['validity_rate']:.3f}")
        return results


# Integration helper functions
async def create_causal_testing_pipeline(
    agent_id: str,
    historical_data: pd.DataFrame,
    treatment_variable: str,
    outcome_variable: str
) -> CausalAgentTestingEngine:
    """
    Create a complete causal testing pipeline for an agent.
    
    Args:
        agent_id: ID of the agent to test
        historical_data: Historical data for analysis
        treatment_variable: Variable representing agent actions
        outcome_variable: Variable representing outcomes
        
    Returns:
        Configured causal testing engine
    """
    engine = CausalAgentTestingEngine()
    
    # Identify confounders
    confounders = await engine.identify_confounders(
        treatment_variable, outcome_variable, historical_data
    )
    
    # Register confounders
    for confounder in confounders:
        engine.confounders.append(
            ConfounderVariable(
                variable_name=confounder,
                variable_type="continuous",  # Simplified
                impact_strength=0.5  # Default
            )
        )
    
    logger.info(f"Created causal testing pipeline for agent {agent_id}")
    return engine


async def integrate_with_memory_service(
    causal_relationships: List[CausalRelationship]
) -> None:
    """
    Store causal relationships in the LiftOS memory service for persistence.
    
    Args:
        causal_relationships: List of causal relationships to store
    """
    # This would integrate with the actual memory service
    # For now, just log the integration
    logger.info(f"Storing {len(causal_relationships)} causal relationships in memory service")
    
    # In production, this would make API calls to the memory service
    # to store the causal relationships as embeddings