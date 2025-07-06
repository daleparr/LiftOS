"""
Media Mix Modeling (MMM) Integration for Agentic Microservice

This module integrates the Agentic microservice with LiftOS's MMM capabilities
to provide marketing-specific agent testing, media optimization recommendations,
and attribution analysis for agent-driven marketing decisions.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass

# Import LiftOS shared components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from shared.models.base import MMMRequest, MMMResponse, MediaChannel
from shared.kse_sdk.client import kse_client
from shared.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MarketingScenario:
    """Represents a marketing scenario for agent testing."""
    scenario_id: str
    name: str
    description: str
    budget_allocation: Dict[str, float]  # channel -> budget
    target_metrics: Dict[str, float]  # metric -> target value
    constraints: Dict[str, Any]
    duration_days: int


@dataclass
class AgentRecommendation:
    """Represents an agent's marketing recommendation."""
    agent_id: str
    scenario_id: str
    recommended_allocation: Dict[str, float]
    predicted_outcomes: Dict[str, float]
    confidence_score: float
    reasoning: str
    timestamp: datetime


class MMMAgentTestingEngine:
    """
    Engine for testing marketing agents using MMM capabilities.
    Provides scenario-based testing, attribution analysis, and performance validation.
    """
    
    def __init__(self):
        self.test_scenarios: Dict[str, MarketingScenario] = {}
        self.agent_recommendations: Dict[str, List[AgentRecommendation]] = {}
        self.mmm_models: Dict[str, Any] = {}
        
    async def create_marketing_scenario(
        self,
        name: str,
        description: str,
        total_budget: float,
        available_channels: List[str],
        target_metrics: Dict[str, float],
        constraints: Optional[Dict[str, Any]] = None,
        duration_days: int = 30
    ) -> MarketingScenario:
        """
        Create a marketing scenario for agent testing.
        
        Args:
            name: Scenario name
            description: Scenario description
            total_budget: Total marketing budget
            available_channels: List of available marketing channels
            target_metrics: Target metrics (e.g., {'roas': 4.0, 'conversions': 1000})
            constraints: Budget or channel constraints
            duration_days: Scenario duration in days
            
        Returns:
            Created marketing scenario
        """
        scenario_id = f"scenario_{len(self.test_scenarios) + 1}_{int(datetime.now().timestamp())}"
        
        # Initialize equal budget allocation as baseline
        budget_per_channel = total_budget / len(available_channels)
        budget_allocation = {channel: budget_per_channel for channel in available_channels}
        
        scenario = MarketingScenario(
            scenario_id=scenario_id,
            name=name,
            description=description,
            budget_allocation=budget_allocation,
            target_metrics=target_metrics,
            constraints=constraints or {},
            duration_days=duration_days
        )
        
        self.test_scenarios[scenario_id] = scenario
        
        logger.info(f"Created marketing scenario: {name} (ID: {scenario_id})")
        return scenario
    
    async def test_agent_recommendation(
        self,
        agent_id: str,
        scenario_id: str,
        agent_recommendation: Dict[str, float],
        reasoning: str = ""
    ) -> Dict[str, Any]:
        """
        Test an agent's marketing recommendation using MMM analysis.
        
        Args:
            agent_id: ID of the agent making the recommendation
            scenario_id: ID of the test scenario
            agent_recommendation: Recommended budget allocation
            reasoning: Agent's reasoning for the recommendation
            
        Returns:
            Test results including predicted performance and validation
        """
        scenario = self.test_scenarios.get(scenario_id)
        if not scenario:
            raise ValueError(f"Scenario {scenario_id} not found")
        
        # Validate recommendation format
        total_recommended = sum(agent_recommendation.values())
        total_budget = sum(scenario.budget_allocation.values())
        
        if abs(total_recommended - total_budget) > 0.01:  # Allow small rounding errors
            return {
                "valid": False,
                "error": f"Recommended budget ({total_recommended}) doesn't match scenario budget ({total_budget})",
                "agent_id": agent_id,
                "scenario_id": scenario_id
            }
        
        # Use MMM to predict outcomes
        try:
            predicted_outcomes = await self._predict_mmm_outcomes(
                budget_allocation=agent_recommendation,
                scenario=scenario
            )
            
            # Calculate performance scores
            performance_scores = await self._calculate_performance_scores(
                predicted_outcomes=predicted_outcomes,
                target_metrics=scenario.target_metrics
            )
            
            # Store agent recommendation
            recommendation = AgentRecommendation(
                agent_id=agent_id,
                scenario_id=scenario_id,
                recommended_allocation=agent_recommendation,
                predicted_outcomes=predicted_outcomes,
                confidence_score=performance_scores["overall_score"],
                reasoning=reasoning,
                timestamp=datetime.now()
            )
            
            if agent_id not in self.agent_recommendations:
                self.agent_recommendations[agent_id] = []
            self.agent_recommendations[agent_id].append(recommendation)
            
            # Generate detailed test results
            test_results = {
                "valid": True,
                "agent_id": agent_id,
                "scenario_id": scenario_id,
                "recommendation": agent_recommendation,
                "predicted_outcomes": predicted_outcomes,
                "performance_scores": performance_scores,
                "baseline_comparison": await self._compare_to_baseline(
                    agent_recommendation, scenario
                ),
                "optimization_suggestions": await self._generate_optimization_suggestions(
                    agent_recommendation, predicted_outcomes, scenario
                ),
                "confidence_score": performance_scores["overall_score"],
                "reasoning": reasoning,
                "timestamp": recommendation.timestamp.isoformat()
            }
            
            logger.info(f"Tested recommendation from agent {agent_id} for scenario {scenario_id}: {performance_scores['overall_score']:.2f} confidence")
            return test_results
            
        except Exception as e:
            logger.error(f"Failed to test agent recommendation: {e}")
            return {
                "valid": False,
                "error": f"MMM prediction failed: {str(e)}",
                "agent_id": agent_id,
                "scenario_id": scenario_id
            }
    
    async def _predict_mmm_outcomes(
        self,
        budget_allocation: Dict[str, float],
        scenario: MarketingScenario
    ) -> Dict[str, float]:
        """Predict marketing outcomes using MMM models."""
        
        # Prepare MMM request
        media_channels = []
        for channel, budget in budget_allocation.items():
            media_channels.append(MediaChannel(
                name=channel,
                spend=budget,
                impressions=budget * 1000,  # Simplified conversion
                clicks=budget * 50,  # Simplified conversion
                conversions=budget * 2  # Simplified conversion
            ))
        
        mmm_request = MMMRequest(
            media_channels=media_channels,
            time_period_days=scenario.duration_days,
            target_metrics=list(scenario.target_metrics.keys())
        )
        
        try:
            # Call MMM service
            mmm_response = await kse_client.predict_mmm(mmm_request)
            
            # Extract predicted outcomes
            predicted_outcomes = {
                "total_conversions": mmm_response.predicted_conversions,
                "total_revenue": mmm_response.predicted_revenue,
                "roas": mmm_response.predicted_revenue / sum(budget_allocation.values()) if sum(budget_allocation.values()) > 0 else 0,
                "cpa": sum(budget_allocation.values()) / mmm_response.predicted_conversions if mmm_response.predicted_conversions > 0 else float('inf'),
                "incremental_conversions": mmm_response.incremental_conversions,
                "attribution_scores": mmm_response.attribution_scores
            }
            
            return predicted_outcomes
            
        except Exception as e:
            logger.warning(f"MMM service unavailable, using simplified prediction: {e}")
            
            # Fallback to simplified prediction
            total_budget = sum(budget_allocation.values())
            return {
                "total_conversions": total_budget * 2.5,  # Simplified model
                "total_revenue": total_budget * 4.0,
                "roas": 4.0,
                "cpa": total_budget / (total_budget * 2.5) if total_budget > 0 else 0,
                "incremental_conversions": total_budget * 1.8,
                "attribution_scores": {channel: budget / total_budget for channel, budget in budget_allocation.items()}
            }
    
    async def _calculate_performance_scores(
        self,
        predicted_outcomes: Dict[str, float],
        target_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate performance scores against target metrics."""
        
        scores = {}
        total_score = 0
        metric_count = 0
        
        for metric, target in target_metrics.items():
            if metric in predicted_outcomes:
                predicted = predicted_outcomes[metric]
                
                # Calculate score (0-1 scale)
                if target > 0:
                    score = min(predicted / target, 2.0)  # Cap at 200% of target
                else:
                    score = 1.0 if predicted >= target else 0.0
                
                scores[f"{metric}_score"] = score
                total_score += score
                metric_count += 1
        
        # Overall score
        scores["overall_score"] = total_score / metric_count if metric_count > 0 else 0.0
        
        return scores
    
    async def _compare_to_baseline(
        self,
        agent_recommendation: Dict[str, float],
        scenario: MarketingScenario
    ) -> Dict[str, Any]:
        """Compare agent recommendation to baseline allocation."""
        
        # Baseline is equal allocation
        baseline_outcomes = await self._predict_mmm_outcomes(
            budget_allocation=scenario.budget_allocation,
            scenario=scenario
        )
        
        agent_outcomes = await self._predict_mmm_outcomes(
            budget_allocation=agent_recommendation,
            scenario=scenario
        )
        
        comparison = {}
        for metric in baseline_outcomes:
            baseline_value = baseline_outcomes[metric]
            agent_value = agent_outcomes[metric]
            
            if isinstance(baseline_value, (int, float)) and isinstance(agent_value, (int, float)):
                if baseline_value != 0:
                    improvement = (agent_value - baseline_value) / baseline_value * 100
                else:
                    improvement = 0.0
                
                comparison[metric] = {
                    "baseline": baseline_value,
                    "agent": agent_value,
                    "improvement_percent": improvement
                }
        
        return comparison
    
    async def _generate_optimization_suggestions(
        self,
        agent_recommendation: Dict[str, float],
        predicted_outcomes: Dict[str, float],
        scenario: MarketingScenario
    ) -> List[str]:
        """Generate optimization suggestions based on the recommendation."""
        
        suggestions = []
        
        # Analyze budget allocation efficiency
        total_budget = sum(agent_recommendation.values())
        attribution_scores = predicted_outcomes.get("attribution_scores", {})
        
        for channel, budget in agent_recommendation.items():
            budget_share = budget / total_budget if total_budget > 0 else 0
            attribution_share = attribution_scores.get(channel, 0)
            
            if attribution_share > budget_share * 1.2:
                suggestions.append(f"Consider increasing budget for {channel} (high attribution vs. spend)")
            elif attribution_share < budget_share * 0.8:
                suggestions.append(f"Consider reducing budget for {channel} (low attribution vs. spend)")
        
        # Check against targets
        roas = predicted_outcomes.get("roas", 0)
        target_roas = scenario.target_metrics.get("roas", 0)
        
        if target_roas > 0 and roas < target_roas:
            suggestions.append(f"ROAS ({roas:.2f}) below target ({target_roas:.2f}) - consider reallocating to higher-performing channels")
        
        return suggestions
    
    async def get_agent_performance_summary(self, agent_id: str) -> Dict[str, Any]:
        """
        Get performance summary for an agent across all tested scenarios.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Performance summary
        """
        agent_recs = self.agent_recommendations.get(agent_id, [])
        
        if not agent_recs:
            return {
                "agent_id": agent_id,
                "total_tests": 0,
                "average_confidence": 0.0,
                "best_performance": None,
                "improvement_trend": None
            }
        
        # Calculate metrics
        confidence_scores = [rec.confidence_score for rec in agent_recs]
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        best_rec = max(agent_recs, key=lambda x: x.confidence_score)
        
        # Calculate improvement trend
        if len(agent_recs) >= 2:
            recent_scores = confidence_scores[-5:]  # Last 5 tests
            early_scores = confidence_scores[:5]  # First 5 tests
            
            recent_avg = sum(recent_scores) / len(recent_scores)
            early_avg = sum(early_scores) / len(early_scores)
            improvement_trend = (recent_avg - early_avg) / early_avg * 100 if early_avg > 0 else 0
        else:
            improvement_trend = 0.0
        
        return {
            "agent_id": agent_id,
            "total_tests": len(agent_recs),
            "average_confidence": avg_confidence,
            "best_performance": {
                "scenario_id": best_rec.scenario_id,
                "confidence_score": best_rec.confidence_score,
                "predicted_roas": best_rec.predicted_outcomes.get("roas", 0),
                "timestamp": best_rec.timestamp.isoformat()
            },
            "improvement_trend": improvement_trend,
            "recent_performance": confidence_scores[-5:] if len(confidence_scores) >= 5 else confidence_scores
        }
    
    async def run_agent_tournament(
        self,
        agent_ids: List[str],
        scenario_id: str,
        get_agent_recommendation_func: callable
    ) -> Dict[str, Any]:
        """
        Run a tournament between multiple agents on the same scenario.
        
        Args:
            agent_ids: List of agent IDs to test
            scenario_id: Scenario to test on
            get_agent_recommendation_func: Function to get recommendation from agent
            
        Returns:
            Tournament results
        """
        tournament_results = {
            "scenario_id": scenario_id,
            "participants": agent_ids,
            "results": {},
            "rankings": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Get recommendations from all agents
        for agent_id in agent_ids:
            try:
                # Get agent's recommendation
                recommendation = await get_agent_recommendation_func(agent_id, scenario_id)
                
                # Test the recommendation
                test_result = await self.test_agent_recommendation(
                    agent_id=agent_id,
                    scenario_id=scenario_id,
                    agent_recommendation=recommendation["allocation"],
                    reasoning=recommendation.get("reasoning", "")
                )
                
                tournament_results["results"][agent_id] = test_result
                
            except Exception as e:
                logger.error(f"Failed to test agent {agent_id} in tournament: {e}")
                tournament_results["results"][agent_id] = {
                    "valid": False,
                    "error": str(e)
                }
        
        # Rank agents by performance
        valid_results = {
            agent_id: result for agent_id, result in tournament_results["results"].items()
            if result.get("valid", False)
        }
        
        rankings = sorted(
            valid_results.items(),
            key=lambda x: x[1]["confidence_score"],
            reverse=True
        )
        
        tournament_results["rankings"] = [
            {
                "rank": i + 1,
                "agent_id": agent_id,
                "confidence_score": result["confidence_score"],
                "predicted_roas": result["predicted_outcomes"].get("roas", 0)
            }
            for i, (agent_id, result) in enumerate(rankings)
        ]
        
        logger.info(f"Completed agent tournament for scenario {scenario_id}: {len(valid_results)} valid participants")
        return tournament_results


# Global MMM testing engine instance
_global_mmm_engine: Optional[MMMAgentTestingEngine] = None


async def get_mmm_testing_engine() -> MMMAgentTestingEngine:
    """Get the global MMM testing engine instance."""
    global _global_mmm_engine
    
    if _global_mmm_engine is None:
        _global_mmm_engine = MMMAgentTestingEngine()
    
    return _global_mmm_engine


async def initialize_mmm_integration() -> None:
    """Initialize the MMM integration."""
    await get_mmm_testing_engine()
    logger.info("Agentic MMM integration initialized")