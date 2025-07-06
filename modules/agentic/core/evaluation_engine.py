"""
Evaluation Engine for Agentic Module

Implements comprehensive agent evaluation based on the AgentSIM framework,
adapted for marketing analytics use cases.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import uuid
import statistics

from ..models.agent_models import MarketingAgent
from ..models.evaluation_models import (
    AgentEvaluationResult, CategoryAssessment, MetricScore,
    EvaluationCategory, MetricType, DeploymentReadiness,
    MarketingMetrics, EvaluationMatrix
)
from ..models.test_models import MarketingTestCase, TestResult
from ..services.memory_service import MemoryService
from ..utils.config import AgenticConfig

logger = logging.getLogger(__name__)


class EvaluationEngine:
    """
    Comprehensive evaluation engine for marketing agents.
    
    This engine implements the AgentSIM evaluation framework adapted
    for marketing analytics, providing detailed assessments across
    multiple categories and metrics.
    """
    
    def __init__(self, memory_service: MemoryService, config: AgenticConfig):
        """Initialize the evaluation engine."""
        self.memory_service = memory_service
        self.config = config
        
        # Default evaluation matrix (based on AgentSIM)
        self.default_matrix = EvaluationMatrix()
        
        # Evaluation cache
        self._evaluation_cache: Dict[str, AgentEvaluationResult] = {}
        
        # Performance tracking
        self._evaluation_history: List[AgentEvaluationResult] = []
        
        logger.info("Evaluation Engine initialized")
    
    async def evaluate_agent(
        self,
        agent: MarketingAgent,
        test_case_id: Optional[str] = None,
        evaluation_type: str = "comprehensive",
        custom_matrix: Optional[EvaluationMatrix] = None
    ) -> AgentEvaluationResult:
        """
        Perform comprehensive evaluation of a marketing agent.
        
        Args:
            agent: The agent to evaluate
            test_case_id: Optional specific test case to use
            evaluation_type: Type of evaluation (comprehensive, quick, custom)
            custom_matrix: Custom evaluation matrix to use
            
        Returns:
            Complete evaluation result
        """
        try:
            logger.info(f"Starting evaluation for agent {agent.name} ({agent.agent_id})")
            
            # Generate evaluation ID
            evaluation_id = f"eval_{uuid.uuid4().hex[:8]}"
            
            # Use custom matrix or default
            matrix = custom_matrix or self.default_matrix
            
            # Record start time
            start_time = datetime.utcnow()
            
            # Perform category evaluations
            category_assessments = []
            
            if evaluation_type in ["comprehensive", "functionality"]:
                functionality = await self._evaluate_functionality(agent, test_case_id)
                category_assessments.append(functionality)
            
            if evaluation_type in ["comprehensive", "reliability"]:
                reliability = await self._evaluate_reliability(agent, test_case_id)
                category_assessments.append(reliability)
            
            if evaluation_type in ["comprehensive", "performance"]:
                performance = await self._evaluate_performance(agent, test_case_id)
                category_assessments.append(performance)
            
            if evaluation_type in ["comprehensive", "security"]:
                security = await self._evaluate_security(agent)
                category_assessments.append(security)
            
            if evaluation_type in ["comprehensive", "usability"]:
                usability = await self._evaluate_usability(agent)
                category_assessments.append(usability)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(category_assessments, matrix)
            
            # Determine deployment readiness
            deployment_readiness = self._determine_deployment_readiness(overall_score)
            
            # Generate marketing-specific metrics
            marketing_metrics = await self._evaluate_marketing_metrics(agent, test_case_id)
            
            # Calculate evaluation duration
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            # Generate recommendations
            strengths, weaknesses, recommendations = self._generate_recommendations(
                category_assessments, marketing_metrics
            )
            
            # Create evaluation result
            evaluation_result = AgentEvaluationResult(
                evaluation_id=evaluation_id,
                agent_id=agent.agent_id,
                agent_name=agent.name,
                test_case_id=test_case_id,
                evaluation_type=evaluation_type,
                category_assessments=category_assessments,
                overall_score=overall_score,
                deployment_readiness=deployment_readiness,
                marketing_metrics=marketing_metrics,
                evaluation_matrix=matrix,
                duration_seconds=duration,
                strengths=strengths,
                weaknesses=weaknesses,
                recommendations=recommendations
            )
            
            # Store evaluation result
            await self.memory_service.store_evaluation_result(evaluation_result)
            
            # Cache result
            self._evaluation_cache[evaluation_id] = evaluation_result
            self._evaluation_history.append(evaluation_result)
            
            logger.info(f"Completed evaluation {evaluation_id} with score {overall_score:.3f}")
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Failed to evaluate agent {agent.agent_id}: {e}")
            raise
    
    async def get_evaluation_result(self, evaluation_id: str) -> Optional[AgentEvaluationResult]:
        """Get evaluation result by ID."""
        try:
            # Check cache first
            if evaluation_id in self._evaluation_cache:
                return self._evaluation_cache[evaluation_id]
            
            # Load from memory service
            result = await self.memory_service.get_evaluation_result(evaluation_id)
            if result:
                self._evaluation_cache[evaluation_id] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get evaluation result {evaluation_id}: {e}")
            return None
    
    async def get_performance_analytics(
        self,
        agent_id: Optional[str] = None,
        time_range: str = "7d"
    ) -> Dict[str, Any]:
        """Get performance analytics for agents."""
        try:
            # Parse time range
            days = int(time_range.rstrip('d'))
            since = datetime.utcnow() - timedelta(days=days)
            
            # Get evaluations
            evaluations = await self.memory_service.get_evaluations_since(since, agent_id)
            
            if not evaluations:
                return {"message": "No evaluations found for the specified criteria"}
            
            # Calculate analytics
            analytics = {
                "total_evaluations": len(evaluations),
                "time_range": time_range,
                "agent_id": agent_id,
                "average_score": statistics.mean([e.overall_score for e in evaluations]),
                "score_trend": self._calculate_score_trend(evaluations),
                "category_breakdown": self._calculate_category_breakdown(evaluations),
                "deployment_readiness_distribution": self._calculate_readiness_distribution(evaluations),
                "top_strengths": self._get_top_strengths(evaluations),
                "common_weaknesses": self._get_common_weaknesses(evaluations)
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get performance analytics: {e}")
            return {"error": str(e)}
    
    async def get_evaluation_trends(
        self,
        category: Optional[EvaluationCategory] = None,
        time_range: str = "30d"
    ) -> Dict[str, Any]:
        """Get evaluation trends over time."""
        try:
            # Parse time range
            days = int(time_range.rstrip('d'))
            since = datetime.utcnow() - timedelta(days=days)
            
            # Get evaluations
            evaluations = await self.memory_service.get_evaluations_since(since)
            
            if not evaluations:
                return {"message": "No evaluations found for the specified time range"}
            
            # Calculate trends
            trends = {
                "time_range": time_range,
                "category": category.value if category else "all",
                "total_evaluations": len(evaluations),
                "score_over_time": self._calculate_score_over_time(evaluations, category),
                "evaluation_frequency": self._calculate_evaluation_frequency(evaluations),
                "agent_performance_ranking": self._calculate_agent_ranking(evaluations)
            }
            
            return trends
            
        except Exception as e:
            logger.error(f"Failed to get evaluation trends: {e}")
            return {"error": str(e)}
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            self._evaluation_cache.clear()
            self._evaluation_history.clear()
            logger.info("Evaluation Engine cleanup completed")
        except Exception as e:
            logger.error(f"Error during Evaluation Engine cleanup: {e}")
    
    async def _evaluate_functionality(
        self,
        agent: MarketingAgent,
        test_case_id: Optional[str] = None
    ) -> CategoryAssessment:
        """Evaluate agent functionality (30% weight)."""
        metrics = []
        
        # Task completion accuracy
        accuracy_score = await self._measure_task_accuracy(agent, test_case_id)
        metrics.append(MetricScore(
            metric_type=MetricType.ACCURACY,
            value=accuracy_score,
            description="Task completion accuracy"
        ))
        
        # Feature completeness
        completeness_score = self._measure_feature_completeness(agent)
        metrics.append(MetricScore(
            metric_type=MetricType.PRECISION,
            value=completeness_score,
            description="Feature completeness score"
        ))
        
        # Output quality
        quality_score = await self._measure_output_quality(agent, test_case_id)
        metrics.append(MetricScore(
            metric_type=MetricType.F1_SCORE,
            value=quality_score,
            description="Output quality assessment"
        ))
        
        # Calculate overall functionality score
        overall_score = statistics.mean([m.normalized_score for m in metrics])
        
        # Generate recommendations
        recommendations = []
        if accuracy_score < 0.8:
            recommendations.append("Improve task completion accuracy through better training data")
        if completeness_score < 0.7:
            recommendations.append("Enhance feature coverage for marketing use cases")
        if quality_score < 0.8:
            recommendations.append("Refine output formatting and structure")
        
        return CategoryAssessment(
            category=EvaluationCategory.FUNCTIONALITY,
            metrics=metrics,
            weight=0.30,
            overall_score=overall_score,
            recommendations=recommendations,
            feedback="Functionality assessment based on task accuracy and feature completeness"
        )
    
    async def _evaluate_reliability(
        self,
        agent: MarketingAgent,
        test_case_id: Optional[str] = None
    ) -> CategoryAssessment:
        """Evaluate agent reliability (25% weight)."""
        metrics = []
        
        # Error rate
        error_rate = await self._measure_error_rate(agent, test_case_id)
        metrics.append(MetricScore(
            metric_type=MetricType.ERROR_RATE,
            value=1.0 - error_rate,  # Invert for scoring
            description="Error rate (inverted for scoring)"
        ))
        
        # Consistency
        consistency_score = await self._measure_consistency(agent, test_case_id)
        metrics.append(MetricScore(
            metric_type=MetricType.CONSISTENCY,
            value=consistency_score,
            description="Output consistency across runs"
        ))
        
        # Robustness
        robustness_score = await self._measure_robustness(agent)
        metrics.append(MetricScore(
            metric_type=MetricType.ROBUSTNESS,
            value=robustness_score,
            description="Robustness to input variations"
        ))
        
        # Calculate overall reliability score
        overall_score = statistics.mean([m.normalized_score for m in metrics])
        
        # Generate recommendations
        recommendations = []
        if error_rate > 0.1:
            recommendations.append("Implement better error handling and validation")
        if consistency_score < 0.8:
            recommendations.append("Improve output consistency through prompt engineering")
        if robustness_score < 0.7:
            recommendations.append("Enhance robustness to edge cases and input variations")
        
        return CategoryAssessment(
            category=EvaluationCategory.RELIABILITY,
            metrics=metrics,
            weight=0.25,
            overall_score=overall_score,
            recommendations=recommendations,
            feedback="Reliability assessment based on error rates and consistency"
        )
    
    async def _evaluate_performance(
        self,
        agent: MarketingAgent,
        test_case_id: Optional[str] = None
    ) -> CategoryAssessment:
        """Evaluate agent performance (20% weight)."""
        metrics = []
        
        # Response time
        response_time = await self._measure_response_time(agent, test_case_id)
        # Normalize response time (lower is better)
        normalized_time = max(0, 1.0 - (response_time / 60.0))  # 60s baseline
        metrics.append(MetricScore(
            metric_type=MetricType.RESPONSE_TIME,
            value=normalized_time,
            unit="normalized",
            description="Response time performance"
        ))
        
        # Throughput
        throughput_score = await self._measure_throughput(agent)
        metrics.append(MetricScore(
            metric_type=MetricType.THROUGHPUT,
            value=throughput_score,
            description="Task processing throughput"
        ))
        
        # Resource utilization
        resource_score = self._measure_resource_efficiency(agent)
        metrics.append(MetricScore(
            metric_type=MetricType.RESOURCE_UTILIZATION,
            value=resource_score,
            description="Resource utilization efficiency"
        ))
        
        # Calculate overall performance score
        overall_score = statistics.mean([m.normalized_score for m in metrics])
        
        # Generate recommendations
        recommendations = []
        if response_time > 30:
            recommendations.append("Optimize response time through model selection or caching")
        if throughput_score < 0.7:
            recommendations.append("Improve throughput with parallel processing")
        if resource_score < 0.8:
            recommendations.append("Optimize resource usage and token efficiency")
        
        return CategoryAssessment(
            category=EvaluationCategory.PERFORMANCE,
            metrics=metrics,
            weight=0.20,
            overall_score=overall_score,
            recommendations=recommendations,
            feedback="Performance assessment based on speed and efficiency"
        )
    
    async def _evaluate_security(self, agent: MarketingAgent) -> CategoryAssessment:
        """Evaluate agent security (15% weight)."""
        metrics = []
        
        # Security configuration
        security_config_score = self._assess_security_configuration(agent)
        metrics.append(MetricScore(
            metric_type=MetricType.SECURITY_SCORE,
            value=security_config_score,
            description="Security configuration assessment"
        ))
        
        # Data handling
        data_handling_score = self._assess_data_handling_security(agent)
        metrics.append(MetricScore(
            metric_type=MetricType.SECURITY_SCORE,
            value=data_handling_score,
            description="Data handling security"
        ))
        
        # Calculate overall security score
        overall_score = statistics.mean([m.normalized_score for m in metrics])
        
        # Generate recommendations
        recommendations = []
        if security_config_score < 0.8:
            recommendations.append("Review and enhance security configuration")
        if data_handling_score < 0.8:
            recommendations.append("Implement better data privacy and handling practices")
        
        return CategoryAssessment(
            category=EvaluationCategory.SECURITY,
            metrics=metrics,
            weight=0.15,
            overall_score=overall_score,
            recommendations=recommendations,
            feedback="Security assessment based on configuration and data handling"
        )
    
    async def _evaluate_usability(self, agent: MarketingAgent) -> CategoryAssessment:
        """Evaluate agent usability (10% weight)."""
        metrics = []
        
        # Configuration simplicity
        config_simplicity = self._assess_configuration_simplicity(agent)
        metrics.append(MetricScore(
            metric_type=MetricType.USABILITY_SCORE,
            value=config_simplicity,
            description="Configuration simplicity"
        ))
        
        # Output clarity
        output_clarity = self._assess_output_clarity(agent)
        metrics.append(MetricScore(
            metric_type=MetricType.USABILITY_SCORE,
            value=output_clarity,
            description="Output clarity and interpretability"
        ))
        
        # Calculate overall usability score
        overall_score = statistics.mean([m.normalized_score for m in metrics])
        
        # Generate recommendations
        recommendations = []
        if config_simplicity < 0.7:
            recommendations.append("Simplify agent configuration and setup")
        if output_clarity < 0.8:
            recommendations.append("Improve output formatting and explanations")
        
        return CategoryAssessment(
            category=EvaluationCategory.USABILITY,
            metrics=metrics,
            weight=0.10,
            overall_score=overall_score,
            recommendations=recommendations,
            feedback="Usability assessment based on ease of use and clarity"
        )
    
    async def _evaluate_marketing_metrics(
        self,
        agent: MarketingAgent,
        test_case_id: Optional[str] = None
    ) -> MarketingMetrics:
        """Evaluate marketing-specific metrics."""
        # This would be implemented based on specific marketing capabilities
        # For now, return placeholder values based on agent type
        
        metrics = MarketingMetrics()
        
        if agent.agent_type == "attribution_analyst":
            metrics.attribution_accuracy = 0.85
            metrics.incrementality_measurement = 0.80
        elif agent.agent_type == "budget_allocator":
            metrics.budget_optimization_score = 0.88
            metrics.roi_prediction_accuracy = 0.82
        elif agent.agent_type == "campaign_optimizer":
            metrics.campaign_performance_prediction = 0.86
            metrics.channel_mix_optimization = 0.84
        elif agent.agent_type == "audience_segmenter":
            metrics.audience_segmentation_quality = 0.89
        elif agent.agent_type == "creative_tester":
            metrics.creative_effectiveness_score = 0.83
        
        return metrics
    
    def _calculate_overall_score(
        self,
        assessments: List[CategoryAssessment],
        matrix: EvaluationMatrix
    ) -> float:
        """Calculate weighted overall score."""
        total_score = 0.0
        total_weight = 0.0
        
        for assessment in assessments:
            weight = matrix.get_category_weight(assessment.category)
            total_score += assessment.overall_score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_deployment_readiness(self, overall_score: float) -> DeploymentReadiness:
        """Determine deployment readiness based on overall score."""
        if overall_score >= 0.9:
            return DeploymentReadiness.PRODUCTION
        elif overall_score >= 0.8:
            return DeploymentReadiness.STAGING
        elif overall_score >= 0.7:
            return DeploymentReadiness.TESTING
        elif overall_score >= 0.6:
            return DeploymentReadiness.DEVELOPMENT
        else:
            return DeploymentReadiness.NOT_READY
    
    def _generate_recommendations(
        self,
        assessments: List[CategoryAssessment],
        marketing_metrics: MarketingMetrics
    ) -> Tuple[List[str], List[str], List[str]]:
        """Generate strengths, weaknesses, and recommendations."""
        strengths = []
        weaknesses = []
        recommendations = []
        
        # Analyze category scores
        for assessment in assessments:
            if assessment.overall_score >= 0.8:
                strengths.append(f"Strong {assessment.category.value} performance")
            elif assessment.overall_score < 0.6:
                weaknesses.append(f"Weak {assessment.category.value} performance")
            
            recommendations.extend(assessment.recommendations)
        
        # Analyze marketing metrics
        marketing_scores = marketing_metrics.get_non_null_metrics()
        if marketing_scores:
            avg_marketing_score = statistics.mean(marketing_scores.values())
            if avg_marketing_score >= 0.8:
                strengths.append("Strong marketing-specific capabilities")
            elif avg_marketing_score < 0.6:
                weaknesses.append("Marketing capabilities need improvement")
        
        return strengths[:5], weaknesses[:5], recommendations[:10]
    
    # Placeholder measurement methods (would be implemented with actual testing)
    async def _measure_task_accuracy(self, agent: MarketingAgent, test_case_id: Optional[str]) -> float:
        """Measure task completion accuracy."""
        # Placeholder implementation
        base_score = 0.8
        if agent.success_rate:
            return min(agent.success_rate + 0.1, 1.0)
        return base_score
    
    def _measure_feature_completeness(self, agent: MarketingAgent) -> float:
        """Measure feature completeness."""
        # Based on number of capabilities
        max_capabilities = 12  # Total possible capabilities
        return min(len(agent.capabilities) / max_capabilities, 1.0)
    
    async def _measure_output_quality(self, agent: MarketingAgent, test_case_id: Optional[str]) -> float:
        """Measure output quality."""
        # Placeholder implementation
        return 0.85
    
    async def _measure_error_rate(self, agent: MarketingAgent, test_case_id: Optional[str]) -> float:
        """Measure error rate."""
        if agent.success_rate:
            return 1.0 - agent.success_rate
        return 0.1  # Default 10% error rate
    
    async def _measure_consistency(self, agent: MarketingAgent, test_case_id: Optional[str]) -> float:
        """Measure output consistency."""
        return 0.82
    
    async def _measure_robustness(self, agent: MarketingAgent) -> float:
        """Measure robustness to input variations."""
        return 0.78
    
    async def _measure_response_time(self, agent: MarketingAgent, test_case_id: Optional[str]) -> float:
        """Measure average response time in seconds."""
        if agent.average_task_duration:
            return agent.average_task_duration
        return 15.0  # Default 15 seconds
    
    async def _measure_throughput(self, agent: MarketingAgent) -> float:
        """Measure task processing throughput."""
        return 0.75
    
    def _measure_resource_efficiency(self, agent: MarketingAgent) -> float:
        """Measure resource utilization efficiency."""
        return 0.80
    
    def _assess_security_configuration(self, agent: MarketingAgent) -> float:
        """Assess security configuration."""
        return 0.85
    
    def _assess_data_handling_security(self, agent: MarketingAgent) -> float:
        """Assess data handling security."""
        return 0.88
    
    def _assess_configuration_simplicity(self, agent: MarketingAgent) -> float:
        """Assess configuration simplicity."""
        return 0.75
    
    def _assess_output_clarity(self, agent: MarketingAgent) -> float:
        """Assess output clarity."""
        return 0.82
    
    # Analytics helper methods
    def _calculate_score_trend(self, evaluations: List[AgentEvaluationResult]) -> List[float]:
        """Calculate score trend over time."""
        return [e.overall_score for e in sorted(evaluations, key=lambda x: x.evaluation_date)]
    
    def _calculate_category_breakdown(self, evaluations: List[AgentEvaluationResult]) -> Dict[str, float]:
        """Calculate average scores by category."""
        category_scores = {}
        for category in EvaluationCategory:
            scores = []
            for eval_result in evaluations:
                score = eval_result.get_category_score(category)
                if score is not None:
                    scores.append(score)
            if scores:
                category_scores[category.value] = statistics.mean(scores)
        return category_scores
    
    def _calculate_readiness_distribution(self, evaluations: List[AgentEvaluationResult]) -> Dict[str, int]:
        """Calculate deployment readiness distribution."""
        distribution = {}
        for eval_result in evaluations:
            readiness = eval_result.deployment_readiness.value
            distribution[readiness] = distribution.get(readiness, 0) + 1
        return distribution
    
    def _get_top_strengths(self, evaluations: List[AgentEvaluationResult]) -> List[str]:
        """Get most common strengths."""
        all_strengths = []
        for eval_result in evaluations:
            all_strengths.extend(eval_result.strengths)
        
        # Count occurrences and return top 5
        strength_counts = {}
        for strength in all_strengths:
            strength_counts[strength] = strength_counts.get(strength, 0) + 1
        
        return sorted(strength_counts.keys(), key=strength_counts.get, reverse=True)[:5]
    
    def _get_common_weaknesses(self, evaluations: List[AgentEvaluationResult]) -> List[str]:
        """Get most common weaknesses."""
        all_weaknesses = []
        for eval_result in evaluations:
            all_weaknesses.extend(eval_result.weaknesses)
        
        # Count occurrences and return top 5
        weakness_counts = {}
        for weakness in all_weaknesses:
            weakness_counts[weakness] = weakness_counts.get(weakness, 0) + 1
        
        return sorted(weakness_counts.keys(), key=weakness_counts.get, reverse=True)[:5]
    
    def _calculate_score_over_time(
        self,
        evaluations: List[AgentEvaluationResult],
        category: Optional[EvaluationCategory]
    ) -> List[Dict[str, Any]]:
        """Calculate scores over time."""
        sorted_evals = sorted(evaluations, key=lambda x: x.evaluation_date)
        
        time_series = []
        for eval_result in sorted_evals:
            if category:
                score = eval_result.get_category_score(category)
            else:
                score = eval_result.overall_score
            
            if score is not None:
                time_series.append({
                    "date": eval_result.evaluation_date.isoformat(),
                    "score": score,
                    "agent_id": eval_result.agent_id
                })
        
        return time_series
    
    def _calculate_evaluation_frequency(self, evaluations: List[AgentEvaluationResult]) -> Dict[str, int]:
        """Calculate evaluation frequency by day."""
        frequency = {}
        for eval_result in evaluations:
            date_key = eval_result.evaluation_date.strftime("%Y-%m-%d")
            frequency[date_key] = frequency.get(date_key, 0) + 1
        return frequency
    
    def _calculate_agent_ranking(self, evaluations: List[AgentEvaluationResult]) -> List[Dict[str, Any]]:
        """Calculate agent performance ranking."""
        agent_scores = {}
        agent_counts = {}
        
        for eval_result in evaluations:
            agent_id = eval_result.agent_id
            if agent_id not in agent_scores:
                agent_scores[agent_id] = []
                agent_counts[agent_id] = 0
            
            agent_scores[agent_id].append(eval_result.overall_score)
            agent_counts[agent_id] += 1
        
        # Calculate average scores
        agent_averages = []
        for agent_id, scores in agent_scores.items():
            avg_score = statistics.mean(scores)
            agent_averages.append({
                "agent_id": agent_id,
                "average_score": avg_score,
                "evaluation_count": agent_counts[agent_id]
            })
        
        # Sort by average score
        return sorted(agent_averages, key=lambda x: x["average_score"], reverse=True)