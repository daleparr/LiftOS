"""
Recommendation Engine for Channels Service
Recommendation ranking and prioritization for budget optimization
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

from models.channels import (
    OptimizationAction, BudgetOptimizationRequest, OptimizationResult,
    ChannelPerformance, OptimizationObjective, ConstraintType
)

logger = logging.getLogger(__name__)


class RecommendationPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    INFORMATIONAL = 5


class RecommendationType(Enum):
    BUDGET_INCREASE = "budget_increase"
    BUDGET_DECREASE = "budget_decrease"
    BUDGET_REALLOCATION = "budget_reallocation"
    CHANNEL_OPTIMIZATION = "channel_optimization"
    RISK_MITIGATION = "risk_mitigation"
    PERFORMANCE_MONITORING = "performance_monitoring"
    STRATEGIC_SHIFT = "strategic_shift"


@dataclass
class Recommendation:
    recommendation_id: str
    recommendation_type: RecommendationType
    title: str
    description: str
    priority: RecommendationPriority
    confidence_score: float
    expected_impact: Dict[str, float]
    implementation_effort: str
    timeline: str
    risk_level: str
    channels_affected: List[str]
    actions: List[OptimizationAction]
    supporting_evidence: List[str]
    potential_risks: List[str]
    success_metrics: List[str]
    created_at: datetime


class RecommendationEngine:
    """Recommendation engine for generating actionable insights and prioritized recommendations"""
    
    def __init__(self, optimization_engine, simulation_engine, saturation_engine, bayesian_engine):
        self.optimization_engine = optimization_engine
        self.simulation_engine = simulation_engine
        self.saturation_engine = saturation_engine
        self.bayesian_engine = bayesian_engine
        
        # Recommendation parameters
        self.min_confidence_threshold = 0.6
        self.high_impact_threshold = 0.15  # 15% improvement
        self.risk_tolerance_mapping = {
            "conservative": 0.1,
            "moderate": 0.25,
            "aggressive": 0.5
        }
        
        # Scoring weights for recommendation prioritization
        self.priority_weights = {
            "expected_impact": 0.4,
            "confidence": 0.3,
            "implementation_ease": 0.2,
            "risk_level": 0.1
        }
    
    async def generate_recommendations(
        self, 
        org_id: str, 
        optimization_result: OptimizationResult,
        current_performance: Dict[str, ChannelPerformance],
        request: BudgetOptimizationRequest
    ) -> List[Recommendation]:
        """Generate comprehensive recommendations based on optimization results"""
        
        try:
            logger.info(f"Generating recommendations for org {org_id}")
            
            recommendations = []
            
            # 1. Budget allocation recommendations
            budget_recommendations = await self._generate_budget_recommendations(
                optimization_result, current_performance, request
            )
            recommendations.extend(budget_recommendations)
            
            # 2. Channel optimization recommendations
            channel_recommendations = await self._generate_channel_optimization_recommendations(
                org_id, optimization_result, current_performance
            )
            recommendations.extend(channel_recommendations)
            
            # 3. Risk mitigation recommendations
            risk_recommendations = await self._generate_risk_mitigation_recommendations(
                optimization_result, current_performance, request
            )
            recommendations.extend(risk_recommendations)
            
            # 4. Performance monitoring recommendations
            monitoring_recommendations = await self._generate_monitoring_recommendations(
                optimization_result, current_performance
            )
            recommendations.extend(monitoring_recommendations)
            
            # 5. Strategic recommendations
            strategic_recommendations = await self._generate_strategic_recommendations(
                org_id, optimization_result, current_performance, request
            )
            recommendations.extend(strategic_recommendations)
            
            # Prioritize and rank recommendations
            prioritized_recommendations = self._prioritize_recommendations(recommendations)
            
            # Filter by confidence threshold
            filtered_recommendations = [
                rec for rec in prioritized_recommendations 
                if rec.confidence_score >= self.min_confidence_threshold
            ]
            
            logger.info(f"Generated {len(filtered_recommendations)} recommendations")
            return filtered_recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {str(e)}")
            return []
    
    async def _generate_budget_recommendations(
        self, 
        optimization_result: OptimizationResult,
        current_performance: Dict[str, ChannelPerformance],
        request: BudgetOptimizationRequest
    ) -> List[Recommendation]:
        """Generate budget allocation recommendations"""
        
        recommendations = []
        
        # Analyze budget changes from optimization
        for action in optimization_result.implementation_plan:
            if abs(action.budget_change_percent) > 10:  # Significant budget change
                
                if action.budget_change > 0:
                    # Budget increase recommendation
                    rec = await self._create_budget_increase_recommendation(action, optimization_result)
                    recommendations.append(rec)
                else:
                    # Budget decrease recommendation
                    rec = await self._create_budget_decrease_recommendation(action, optimization_result)
                    recommendations.append(rec)
        
        # Cross-channel reallocation recommendations
        reallocation_rec = await self._create_reallocation_recommendation(
            optimization_result, current_performance, request
        )
        if reallocation_rec:
            recommendations.append(reallocation_rec)
        
        return recommendations
    
    async def _generate_channel_optimization_recommendations(
        self, 
        org_id: str,
        optimization_result: OptimizationResult,
        current_performance: Dict[str, ChannelPerformance]
    ) -> List[Recommendation]:
        """Generate channel-specific optimization recommendations"""
        
        recommendations = []
        
        for channel_id, performance in current_performance.items():
            
            # Saturation analysis
            if performance.saturation_level > 0.8:
                rec = await self._create_saturation_warning_recommendation(channel_id, performance)
                recommendations.append(rec)
            
            # Efficiency analysis
            if performance.efficiency_score < 0.6:
                rec = await self._create_efficiency_improvement_recommendation(channel_id, performance)
                recommendations.append(rec)
            
            # ROAS optimization
            if performance.current_roas < 2.0:
                rec = await self._create_roas_improvement_recommendation(channel_id, performance)
                recommendations.append(rec)
            
            # Trend analysis
            if performance.trend_direction == "declining":
                rec = await self._create_trend_reversal_recommendation(channel_id, performance)
                recommendations.append(rec)
        
        return recommendations
    
    async def _generate_risk_mitigation_recommendations(
        self, 
        optimization_result: OptimizationResult,
        current_performance: Dict[str, ChannelPerformance],
        request: BudgetOptimizationRequest
    ) -> List[Recommendation]:
        """Generate risk mitigation recommendations"""
        
        recommendations = []
        
        # Portfolio concentration risk
        allocation_values = list(optimization_result.recommended_allocation.values())
        max_allocation = max(allocation_values)
        total_allocation = sum(allocation_values)
        concentration_ratio = max_allocation / total_allocation if total_allocation > 0 else 0
        
        if concentration_ratio > 0.6:  # More than 60% in single channel
            rec = await self._create_diversification_recommendation(
                optimization_result, concentration_ratio
            )
            recommendations.append(rec)
        
        # High-risk actions
        high_risk_actions = [
            action for action in optimization_result.implementation_plan
            if action.risk_level == "high"
        ]
        
        if high_risk_actions:
            rec = await self._create_risk_monitoring_recommendation(high_risk_actions)
            recommendations.append(rec)
        
        # Confidence interval analysis
        wide_intervals = [
            channel for channel, interval in optimization_result.confidence_intervals.items()
            if (interval[1] - interval[0]) / interval[0] > 0.5  # Wide confidence interval
        ]
        
        if wide_intervals:
            rec = await self._create_uncertainty_reduction_recommendation(wide_intervals)
            recommendations.append(rec)
        
        return recommendations
    
    async def _generate_monitoring_recommendations(
        self, 
        optimization_result: OptimizationResult,
        current_performance: Dict[str, ChannelPerformance]
    ) -> List[Recommendation]:
        """Generate performance monitoring recommendations"""
        
        recommendations = []
        
        # Key metrics to monitor
        critical_channels = [
            action.channel_id for action in optimization_result.implementation_plan
            if action.priority <= 2
        ]
        
        if critical_channels:
            rec = await self._create_monitoring_recommendation(critical_channels, optimization_result)
            recommendations.append(rec)
        
        # A/B testing recommendations
        uncertain_channels = [
            channel for channel, interval in optimization_result.confidence_intervals.items()
            if (interval[1] - interval[0]) / interval[0] > 0.3
        ]
        
        if uncertain_channels:
            rec = await self._create_testing_recommendation(uncertain_channels)
            recommendations.append(rec)
        
        return recommendations
    
    async def _generate_strategic_recommendations(
        self, 
        org_id: str,
        optimization_result: OptimizationResult,
        current_performance: Dict[str, ChannelPerformance],
        request: BudgetOptimizationRequest
    ) -> List[Recommendation]:
        """Generate strategic recommendations"""
        
        recommendations = []
        
        # Channel mix optimization
        if len(optimization_result.channels_optimized) < 5:
            rec = await self._create_channel_expansion_recommendation(
                optimization_result, current_performance
            )
            recommendations.append(rec)
        
        # Objective rebalancing
        if request.objectives and len(request.objectives) == 1:
            rec = await self._create_multi_objective_recommendation(request.objectives[0])
            recommendations.append(rec)
        
        # Long-term strategy
        total_improvement = sum(
            action.expected_revenue_impact for action in optimization_result.implementation_plan
        )
        
        if total_improvement > 100000:  # Significant improvement potential
            rec = await self._create_strategic_scaling_recommendation(total_improvement)
            recommendations.append(rec)
        
        return recommendations
    
    # Recommendation creation methods
    
    async def _create_budget_increase_recommendation(
        self, 
        action: OptimizationAction, 
        optimization_result: OptimizationResult
    ) -> Recommendation:
        """Create budget increase recommendation"""
        
        return Recommendation(
            recommendation_id=f"budget_increase_{action.channel_id}_{int(datetime.utcnow().timestamp())}",
            recommendation_type=RecommendationType.BUDGET_INCREASE,
            title=f"Increase {action.channel_id} Budget by {action.budget_change_percent:.1f}%",
            description=(
                f"Increase budget for {action.channel_id} from ${action.current_budget:,.0f} "
                f"to ${action.recommended_budget:,.0f} to capture additional revenue opportunities."
            ),
            priority=RecommendationPriority.HIGH if action.priority <= 2 else RecommendationPriority.MEDIUM,
            confidence_score=action.confidence_score,
            expected_impact={
                "revenue_increase": action.expected_revenue_impact,
                "roas_change": action.expected_roas_change,
                "conversion_increase": action.expected_conversion_impact
            },
            implementation_effort="Low" if action.budget_change_percent < 25 else "Medium",
            timeline=action.implementation_timeline,
            risk_level=action.risk_level,
            channels_affected=[action.channel_id],
            actions=[action],
            supporting_evidence=[
                f"Expected revenue increase: ${action.expected_revenue_impact:,.0f}",
                f"Confidence score: {action.confidence_score:.1%}",
                f"Current channel efficiency: Above average"
            ],
            potential_risks=[
                "Diminishing returns at higher spend levels",
                "Market saturation effects",
                "Increased competition for ad inventory"
            ],
            success_metrics=[
                "Revenue per dollar spent",
                "Conversion rate maintenance",
                "Cost per acquisition trends"
            ],
            created_at=datetime.utcnow()
        )
    
    async def _create_budget_decrease_recommendation(
        self, 
        action: OptimizationAction, 
        optimization_result: OptimizationResult
    ) -> Recommendation:
        """Create budget decrease recommendation"""
        
        return Recommendation(
            recommendation_id=f"budget_decrease_{action.channel_id}_{int(datetime.utcnow().timestamp())}",
            recommendation_type=RecommendationType.BUDGET_DECREASE,
            title=f"Reduce {action.channel_id} Budget by {abs(action.budget_change_percent):.1f}%",
            description=(
                f"Reduce budget for {action.channel_id} from ${action.current_budget:,.0f} "
                f"to ${action.recommended_budget:,.0f} due to diminishing returns or poor performance."
            ),
            priority=RecommendationPriority.MEDIUM,
            confidence_score=action.confidence_score,
            expected_impact={
                "cost_savings": abs(action.budget_change),
                "efficiency_improvement": abs(action.expected_roas_change),
                "reallocation_opportunity": abs(action.budget_change)
            },
            implementation_effort="Low",
            timeline=action.implementation_timeline,
            risk_level="low",
            channels_affected=[action.channel_id],
            actions=[action],
            supporting_evidence=[
                f"Current saturation level indicates diminishing returns",
                f"Funds can be reallocated to higher-performing channels",
                f"Confidence score: {action.confidence_score:.1%}"
            ],
            potential_risks=[
                "Loss of market share",
                "Reduced brand visibility",
                "Competitor advantage in reduced channels"
            ],
            success_metrics=[
                "Overall portfolio ROAS improvement",
                "Reallocation effectiveness",
                "Maintained conversion volume"
            ],
            created_at=datetime.utcnow()
        )
    
    async def _create_reallocation_recommendation(
        self, 
        optimization_result: OptimizationResult,
        current_performance: Dict[str, ChannelPerformance],
        request: BudgetOptimizationRequest
    ) -> Optional[Recommendation]:
        """Create cross-channel reallocation recommendation"""
        
        # Find significant reallocations
        large_increases = [
            action for action in optimization_result.implementation_plan
            if action.budget_change_percent > 20
        ]
        
        large_decreases = [
            action for action in optimization_result.implementation_plan
            if action.budget_change_percent < -20
        ]
        
        if not (large_increases and large_decreases):
            return None
        
        total_reallocation = sum(abs(action.budget_change) for action in optimization_result.implementation_plan)
        
        return Recommendation(
            recommendation_id=f"reallocation_{int(datetime.utcnow().timestamp())}",
            recommendation_type=RecommendationType.BUDGET_REALLOCATION,
            title=f"Reallocate ${total_reallocation:,.0f} Across Channels",
            description=(
                f"Implement strategic budget reallocation moving funds from "
                f"saturated channels to high-opportunity channels for optimal performance."
            ),
            priority=RecommendationPriority.HIGH,
            confidence_score=optimization_result.overall_confidence,
            expected_impact={
                "total_revenue_improvement": sum(
                    action.expected_revenue_impact for action in optimization_result.implementation_plan
                ),
                "portfolio_efficiency": 0.15,  # Estimated 15% efficiency improvement
                "risk_reduction": 0.1
            },
            implementation_effort="Medium",
            timeline="1-2 weeks",
            risk_level="medium",
            channels_affected=optimization_result.channels_optimized,
            actions=optimization_result.implementation_plan,
            supporting_evidence=[
                f"Optimization algorithm confidence: {optimization_result.overall_confidence:.1%}",
                f"Expected total improvement: ${sum(action.expected_revenue_impact for action in optimization_result.implementation_plan):,.0f}",
                "Cross-channel synergies identified"
            ],
            potential_risks=[
                "Temporary performance dips during transition",
                "Channel-specific learning curve effects",
                "Market timing considerations"
            ],
            success_metrics=[
                "Overall portfolio ROAS",
                "Total revenue growth",
                "Channel efficiency scores"
            ],
            created_at=datetime.utcnow()
        )
    
    async def _create_saturation_warning_recommendation(
        self, 
        channel_id: str, 
        performance: ChannelPerformance
    ) -> Recommendation:
        """Create saturation warning recommendation"""
        
        return Recommendation(
            recommendation_id=f"saturation_warning_{channel_id}_{int(datetime.utcnow().timestamp())}",
            recommendation_type=RecommendationType.CHANNEL_OPTIMIZATION,
            title=f"{channel_id} Approaching Saturation",
            description=(
                f"{channel_id} is showing high saturation levels ({performance.saturation_level:.1%}). "
                f"Consider optimizing creative, targeting, or exploring new audiences."
            ),
            priority=RecommendationPriority.MEDIUM,
            confidence_score=0.8,
            expected_impact={
                "efficiency_improvement": 0.1,
                "saturation_reduction": 0.2,
                "audience_expansion": 0.15
            },
            implementation_effort="Medium",
            timeline="2-3 weeks",
            risk_level="low",
            channels_affected=[channel_id],
            actions=[],
            supporting_evidence=[
                f"Current saturation level: {performance.saturation_level:.1%}",
                f"Efficiency score: {performance.efficiency_score:.1%}",
                "Diminishing returns pattern detected"
            ],
            potential_risks=[
                "Creative fatigue",
                "Audience overlap",
                "Increased competition"
            ],
            success_metrics=[
                "Saturation level reduction",
                "Efficiency score improvement",
                "Cost per acquisition trends"
            ],
            created_at=datetime.utcnow()
        )
    
    async def _create_diversification_recommendation(
        self, 
        optimization_result: OptimizationResult,
        concentration_ratio: float
    ) -> Recommendation:
        """Create portfolio diversification recommendation"""
        
        return Recommendation(
            recommendation_id=f"diversification_{int(datetime.utcnow().timestamp())}",
            recommendation_type=RecommendationType.RISK_MITIGATION,
            title="Reduce Portfolio Concentration Risk",
            description=(
                f"Current allocation shows high concentration ({concentration_ratio:.1%} in single channel). "
                f"Consider diversifying to reduce risk and capture cross-channel opportunities."
            ),
            priority=RecommendationPriority.HIGH,
            confidence_score=0.9,
            expected_impact={
                "risk_reduction": 0.3,
                "portfolio_stability": 0.2,
                "opportunity_capture": 0.15
            },
            implementation_effort="Medium",
            timeline="2-4 weeks",
            risk_level="low",
            channels_affected=optimization_result.channels_optimized,
            actions=[],
            supporting_evidence=[
                f"Concentration ratio: {concentration_ratio:.1%}",
                "Portfolio theory suggests diversification benefits",
                "Cross-channel synergies available"
            ],
            potential_risks=[
                "Short-term performance impact",
                "Learning curve for new channels",
                "Resource allocation challenges"
            ],
            success_metrics=[
                "Portfolio concentration ratio",
                "Risk-adjusted returns",
                "Channel correlation metrics"
            ],
            created_at=datetime.utcnow()
        )
    
    def _prioritize_recommendations(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """Prioritize recommendations using multi-criteria scoring"""
        
        scored_recommendations = []
        
        for rec in recommendations:
            score = self._calculate_recommendation_score(rec)
            scored_recommendations.append((score, rec))
        
        # Sort by score (descending) and then by priority
        scored_recommendations.sort(key=lambda x: (-x[0], x[1].priority.value))
        
        return [rec for score, rec in scored_recommendations]
    
    def _calculate_recommendation_score(self, recommendation: Recommendation) -> float:
        """Calculate priority score for a recommendation"""
        
        # Impact score (normalized)
        impact_values = list(recommendation.expected_impact.values())
        impact_score = np.mean([abs(v) for v in impact_values]) if impact_values else 0
        impact_score = min(impact_score / 100000, 1.0)  # Normalize to 0-1
        
        # Confidence score (already 0-1)
        confidence_score = recommendation.confidence_score
        
        # Implementation ease score
        ease_mapping = {"Low": 1.0, "Medium": 0.7, "High": 0.4}
        ease_score = ease_mapping.get(recommendation.implementation_effort, 0.5)
        
        # Risk score (inverted - lower risk = higher score)
        risk_mapping = {"low": 1.0, "medium": 0.7, "high": 0.4}
        risk_score = risk_mapping.get(recommendation.risk_level, 0.5)
        
        # Calculate weighted score
        total_score = (
            self.priority_weights["expected_impact"] * impact_score +
            self.priority_weights["confidence"] * confidence_score +
            self.priority_weights["implementation_ease"] * ease_score +
            self.priority_weights["risk_level"] * risk_score
        )
        
        return total_score
    
    # Additional helper methods for creating specific recommendation types
    
    async def _create_efficiency_improvement_recommendation(
        self, channel_id: str, performance: ChannelPerformance
    ) -> Recommendation:
        """Create efficiency improvement recommendation"""
        
        return Recommendation(
            recommendation_id=f"efficiency_{channel_id}_{int(datetime.utcnow().timestamp())}",
            recommendation_type=RecommendationType.CHANNEL_OPTIMIZATION,
            title=f"Improve {channel_id} Efficiency",
            description=f"Optimize {channel_id} performance to improve efficiency score from {performance.efficiency_score:.1%}",
            priority=RecommendationPriority.MEDIUM,
            confidence_score=0.75,
            expected_impact={"efficiency_improvement": 0.2},
            implementation_effort="Medium",
            timeline="2-3 weeks",
            risk_level="low",
            channels_affected=[channel_id],
            actions=[],
            supporting_evidence=[f"Current efficiency: {performance.efficiency_score:.1%}"],
            potential_risks=["Temporary performance impact"],
            success_metrics=["Efficiency score improvement"],
            created_at=datetime.utcnow()
        )
    
    async def _create_monitoring_recommendation(
        self, channels: List[str], optimization_result: OptimizationResult
    ) -> Recommendation:
        """Create monitoring recommendation"""
        
        return Recommendation(
            recommendation_id=f"monitoring_{int(datetime.utcnow().timestamp())}",
            recommendation_type=RecommendationType.PERFORMANCE_MONITORING,
            title="Implement Enhanced Performance Monitoring",
            description=f"Set up enhanced monitoring for {len(channels)} critical channels post-optimization",
            priority=RecommendationPriority.HIGH,
            confidence_score=0.95,
            expected_impact={"risk_reduction": 0.2, "early_detection": 0.3},
            implementation_effort="Low",
            timeline="1 week",
            risk_level="low",
            channels_affected=channels,
            actions=[],
            supporting_evidence=["Critical channels identified", "Optimization changes require monitoring"],
            potential_risks=["Alert fatigue"],
            success_metrics=["Monitoring coverage", "Alert accuracy"],
            created_at=datetime.utcnow()
        )
    
    # Placeholder methods for additional recommendation types
    
    async def _create_roas_improvement_recommendation(self, channel_id: str, performance: ChannelPerformance) -> Recommendation:
        """Create ROAS improvement recommendation"""
        # Implementation similar to efficiency improvement
        pass
    
    async def _create_trend_reversal_recommendation(self, channel_id: str, performance: ChannelPerformance) -> Recommendation:
        """Create trend reversal recommendation"""
        # Implementation for addressing declining trends
        pass
    
    async def _create_risk_monitoring_recommendation(self, actions: List[OptimizationAction]) -> Recommendation:
        """Create risk monitoring recommendation"""
        # Implementation for monitoring high-risk actions
        pass
    
    async def _create_uncertainty_reduction_recommendation(self, channels: List[str]) -> Recommendation:
        """Create uncertainty reduction recommendation"""
        # Implementation for reducing uncertainty in channel performance
        pass
    
    async def _create_testing_recommendation(self, channels: List[str]) -> Recommendation:
        """Create A/B testing recommendation"""
        # Implementation for recommending tests to reduce uncertainty
        pass
    
    async def _create_channel_expansion_recommendation(
        self, optimization_result: OptimizationResult, current_performance: Dict[str, ChannelPerformance]
    ) -> Recommendation:
        """Create channel expansion recommendation"""
        # Implementation for recommending new channels
        pass
    
    async def _create_multi_objective_recommendation(self, current_objective: OptimizationObjective) -> Recommendation:
        """Create multi-objective optimization recommendation"""
        # Implementation for recommending multiple objectives
        pass
    
    async def _create_strategic_scaling_recommendation(self, improvement_potential: float) -> Recommendation:
        """Create strategic scaling recommendation"""
        # Implementation for strategic scaling opportunities
        pass