"""
Unified Integration Layer for Agentic Microservice

This module provides a unified interface for all LiftOS Core integrations,
orchestrating interactions between causal inference, observability, memory,
and MMM capabilities for comprehensive agent testing and validation.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

# Import integration modules
from .causal_integration import get_causal_testing_engine, CausalAgentTestingEngine
from .observability_integration import get_observability_manager, AgenticObservabilityManager
from .memory_integration import get_memory_manager, AgentMemoryManager
from .mmm_integration import get_mmm_testing_engine, MMMAgentTestingEngine

# Import LiftOS shared components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from shared.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IntegratedTestResult:
    """Comprehensive test result combining all integration capabilities."""
    test_id: str
    agent_id: str
    test_type: str
    timestamp: datetime
    
    # Data quality results
    data_quality_score: float
    data_quality_issues: List[str]
    
    # Causal validation results
    causal_validity_score: float
    causal_relationships: Dict[str, Any]
    confounders_identified: List[str]
    
    # Performance metrics
    performance_metrics: Dict[str, float]
    observability_traces: List[str]
    
    # Memory insights
    relevant_experiences: List[Dict[str, Any]]
    learning_patterns: List[Dict[str, Any]]
    
    # MMM results (if applicable)
    mmm_predictions: Optional[Dict[str, float]]
    marketing_performance: Optional[Dict[str, float]]
    
    # Overall assessment
    overall_confidence: float
    recommendations: List[str]
    success: bool


class UnifiedAgentIntegrationManager:
    """
    Unified manager that orchestrates all LiftOS Core integrations
    for comprehensive agent testing and validation.
    """
    
    def __init__(self):
        self.causal_engine: Optional[CausalAgentTestingEngine] = None
        self.observability_manager: Optional[AgenticObservabilityManager] = None
        self.memory_manager: Optional[AgentMemoryManager] = None
        self.mmm_engine: Optional[MMMAgentTestingEngine] = None
        
        self.test_results: Dict[str, IntegratedTestResult] = {}
        self.agent_profiles: Dict[str, Dict[str, Any]] = {}
        
    async def initialize(self) -> None:
        """Initialize all integration components."""
        try:
            self.causal_engine = await get_causal_testing_engine()
            self.observability_manager = await get_observability_manager()
            self.memory_manager = await get_memory_manager()
            self.mmm_engine = await get_mmm_testing_engine()
            
            logger.info("Unified agent integration manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize unified integration manager: {e}")
            raise
    
    async def run_comprehensive_agent_test(
        self,
        agent_id: str,
        test_scenario: Dict[str, Any],
        test_data: Dict[str, Any],
        include_mmm: bool = False
    ) -> IntegratedTestResult:
        """
        Run a comprehensive test of an agent using all available integrations.
        
        Args:
            agent_id: ID of the agent to test
            test_scenario: Test scenario configuration
            test_data: Test data for the agent
            include_mmm: Whether to include MMM testing
            
        Returns:
            Comprehensive test results
        """
        test_id = f"integrated_test_{agent_id}_{int(datetime.now().timestamp())}"
        
        # Start observability tracing
        trace_id = await self.observability_manager.start_agent_trace(
            agent_id=agent_id,
            operation="comprehensive_test",
            metadata={"test_id": test_id, "scenario": test_scenario["name"]}
        )
        
        try:
            # 1. Data Quality Assessment
            data_quality_result = await self._assess_data_quality(test_data)
            
            # 2. Retrieve Relevant Agent Experiences
            memory_insights = await self._get_memory_insights(agent_id, test_scenario)
            
            # 3. Causal Validation
            causal_result = await self._validate_causal_relationships(
                agent_id, test_scenario, test_data
            )
            
            # 4. Performance Monitoring
            performance_metrics = await self._monitor_agent_performance(
                agent_id, test_scenario, test_data
            )
            
            # 5. MMM Testing (if requested)
            mmm_result = None
            if include_mmm and test_scenario.get("type") == "marketing":
                mmm_result = await self._test_marketing_performance(
                    agent_id, test_scenario, test_data
                )
            
            # 6. Calculate Overall Confidence
            overall_confidence = await self._calculate_overall_confidence(
                data_quality_result, causal_result, performance_metrics, mmm_result
            )
            
            # 7. Generate Recommendations
            recommendations = await self._generate_comprehensive_recommendations(
                data_quality_result, causal_result, performance_metrics, 
                memory_insights, mmm_result
            )
            
            # Create integrated test result
            test_result = IntegratedTestResult(
                test_id=test_id,
                agent_id=agent_id,
                test_type=test_scenario.get("type", "general"),
                timestamp=datetime.now(),
                data_quality_score=data_quality_result["overall_score"],
                data_quality_issues=data_quality_result["issues"],
                causal_validity_score=causal_result["validity_score"],
                causal_relationships=causal_result["relationships"],
                confounders_identified=causal_result["confounders"],
                performance_metrics=performance_metrics,
                observability_traces=[trace_id],
                relevant_experiences=memory_insights["experiences"],
                learning_patterns=memory_insights["patterns"],
                mmm_predictions=mmm_result["predictions"] if mmm_result else None,
                marketing_performance=mmm_result["performance"] if mmm_result else None,
                overall_confidence=overall_confidence,
                recommendations=recommendations,
                success=overall_confidence >= 0.7  # 70% confidence threshold
            )
            
            # Store test result
            self.test_results[test_id] = test_result
            
            # Store experience in agent memory
            await self.memory_manager.store_agent_experience(
                agent_id=agent_id,
                experience_type="comprehensive_test",
                experience_data=asdict(test_result),
                importance_score=overall_confidence
            )
            
            # End observability trace
            await self.observability_manager.end_agent_trace(
                trace_id=trace_id,
                success=test_result.success,
                metadata={"confidence": overall_confidence}
            )
            
            logger.info(f"Completed comprehensive test for agent {agent_id}: {overall_confidence:.2f} confidence")
            return test_result
            
        except Exception as e:
            await self.observability_manager.end_agent_trace(
                trace_id=trace_id,
                success=False,
                metadata={"error": str(e)}
            )
            logger.error(f"Comprehensive test failed for agent {agent_id}: {e}")
            raise
    
    async def _assess_data_quality(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess data quality for the test."""
        # Import data quality engine
        from ..core.data_quality_engine import DataQualityEngine
        
        quality_engine = DataQualityEngine()
        quality_report = await quality_engine.evaluate_data_quality(test_data)
        
        return {
            "overall_score": quality_report.overall_score,
            "issues": [issue.description for issue in quality_report.issues],
            "recommendations": quality_report.recommendations
        }
    
    async def _get_memory_insights(
        self, 
        agent_id: str, 
        test_scenario: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get relevant memory insights for the agent."""
        
        # Retrieve relevant experiences
        experiences = await self.memory_manager.retrieve_relevant_experiences(
            agent_id=agent_id,
            query=test_scenario.get("description", ""),
            limit=10
        )
        
        # Get learning patterns
        patterns = await self.memory_manager.retrieve_relevant_experiences(
            agent_id=agent_id,
            query="learned_pattern",
            experience_types=["learned_pattern"],
            limit=5
        )
        
        return {
            "experiences": experiences,
            "patterns": patterns,
            "knowledge_summary": await self.memory_manager.get_agent_knowledge_summary(agent_id)
        }
    
    async def _validate_causal_relationships(
        self,
        agent_id: str,
        test_scenario: Dict[str, Any],
        test_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate causal relationships in agent recommendations."""
        
        # Extract variables from test scenario
        treatment_vars = test_scenario.get("treatment_variables", [])
        outcome_vars = test_scenario.get("outcome_variables", [])
        
        if not treatment_vars or not outcome_vars:
            return {
                "validity_score": 1.0,  # No causal validation needed
                "relationships": {},
                "confounders": []
            }
        
        # Run causal validation
        causal_result = await self.causal_engine.validate_agent_recommendation(
            agent_id=agent_id,
            treatment_variables=treatment_vars,
            outcome_variables=outcome_vars,
            data=test_data,
            agent_recommendation=test_scenario.get("agent_recommendation", {})
        )
        
        return {
            "validity_score": causal_result["causal_validity_score"],
            "relationships": causal_result["causal_relationships"],
            "confounders": causal_result["identified_confounders"]
        }
    
    async def _monitor_agent_performance(
        self,
        agent_id: str,
        test_scenario: Dict[str, Any],
        test_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Monitor agent performance during testing."""
        
        # Get performance metrics from observability
        metrics = await self.observability_manager.get_agent_performance_metrics(agent_id)
        
        # Add scenario-specific metrics
        scenario_metrics = {
            "test_duration": 0.0,  # Will be updated by observability
            "data_processing_time": len(str(test_data)) / 1000,  # Simplified
            "scenario_complexity": len(test_scenario.get("variables", [])),
            "memory_usage": metrics.get("memory_usage_mb", 0)
        }
        
        return {**metrics, **scenario_metrics}
    
    async def _test_marketing_performance(
        self,
        agent_id: str,
        test_scenario: Dict[str, Any],
        test_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Test marketing-specific performance using MMM."""
        
        # Extract marketing scenario details
        budget_allocation = test_scenario.get("budget_allocation", {})
        
        if not budget_allocation:
            return None
        
        # Test with MMM engine
        mmm_result = await self.mmm_engine.test_agent_recommendation(
            agent_id=agent_id,
            scenario_id=test_scenario.get("scenario_id", "default"),
            agent_recommendation=budget_allocation,
            reasoning=test_scenario.get("reasoning", "")
        )
        
        return {
            "predictions": mmm_result.get("predicted_outcomes", {}),
            "performance": mmm_result.get("performance_scores", {}),
            "baseline_comparison": mmm_result.get("baseline_comparison", {})
        }
    
    async def _calculate_overall_confidence(
        self,
        data_quality_result: Dict[str, Any],
        causal_result: Dict[str, Any],
        performance_metrics: Dict[str, float],
        mmm_result: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate overall confidence score."""
        
        # Weight different components
        weights = {
            "data_quality": 0.3,
            "causal_validity": 0.3,
            "performance": 0.2,
            "mmm": 0.2 if mmm_result else 0.0
        }
        
        # Normalize weights if MMM not included
        if not mmm_result:
            total_weight = sum(w for k, w in weights.items() if k != "mmm")
            for k in weights:
                if k != "mmm":
                    weights[k] = weights[k] / total_weight
        
        # Calculate weighted score
        confidence = 0.0
        confidence += weights["data_quality"] * data_quality_result["overall_score"]
        confidence += weights["causal_validity"] * causal_result["validity_score"]
        
        # Performance score (normalized)
        perf_score = min(performance_metrics.get("success_rate", 0.5), 1.0)
        confidence += weights["performance"] * perf_score
        
        # MMM score (if available)
        if mmm_result:
            mmm_score = mmm_result["performance"].get("overall_score", 0.5)
            confidence += weights["mmm"] * mmm_score
        
        return min(confidence, 1.0)
    
    async def _generate_comprehensive_recommendations(
        self,
        data_quality_result: Dict[str, Any],
        causal_result: Dict[str, Any],
        performance_metrics: Dict[str, float],
        memory_insights: Dict[str, Any],
        mmm_result: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate comprehensive recommendations."""
        
        recommendations = []
        
        # Data quality recommendations
        if data_quality_result["overall_score"] < 0.8:
            recommendations.extend(data_quality_result["recommendations"])
        
        # Causal validity recommendations
        if causal_result["validity_score"] < 0.7:
            recommendations.append("Review causal assumptions and consider additional confounders")
            if causal_result["confounders"]:
                recommendations.append(f"Address identified confounders: {', '.join(causal_result['confounders'])}")
        
        # Performance recommendations
        if performance_metrics.get("success_rate", 1.0) < 0.8:
            recommendations.append("Improve agent performance through additional training or optimization")
        
        # Memory-based recommendations
        if len(memory_insights["experiences"]) < 5:
            recommendations.append("Agent needs more experience data for better decision-making")
        
        # MMM recommendations
        if mmm_result and mmm_result["performance"].get("overall_score", 1.0) < 0.7:
            recommendations.append("Optimize marketing budget allocation based on MMM insights")
        
        return recommendations
    
    async def get_agent_profile(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive profile for an agent."""
        
        # Get test history
        agent_tests = [
            result for result in self.test_results.values()
            if result.agent_id == agent_id
        ]
        
        # Get memory summary
        memory_summary = await self.memory_manager.get_agent_knowledge_summary(agent_id)
        
        # Get performance summary
        performance_summary = await self.observability_manager.get_agent_performance_summary(agent_id)
        
        # Calculate trends
        if agent_tests:
            confidence_scores = [test.overall_confidence for test in agent_tests]
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            
            # Recent vs. historical performance
            if len(confidence_scores) >= 5:
                recent_avg = sum(confidence_scores[-3:]) / 3
                historical_avg = sum(confidence_scores[:-3]) / (len(confidence_scores) - 3)
                improvement_trend = (recent_avg - historical_avg) / historical_avg * 100
            else:
                improvement_trend = 0.0
        else:
            avg_confidence = 0.0
            improvement_trend = 0.0
        
        # Get strengths and improvement areas
        strengths = await self._identify_agent_strengths(agent_tests)
        improvement_areas = await self._identify_improvement_areas(agent_tests)
        
        profile = {
            "agent_id": agent_id,
            "total_tests": len(agent_tests),
            "average_confidence": avg_confidence,
            "improvement_trend": improvement_trend,
            "memory_summary": memory_summary,
            "performance_summary": performance_summary,
            "recent_tests": [
                {
                    "test_id": test.test_id,
                    "confidence": test.overall_confidence,
                    "timestamp": test.timestamp.isoformat(),
                    "success": test.success
                }
                for test in sorted(agent_tests, key=lambda x: x.timestamp, reverse=True)[:5]
            ],
            "strengths": strengths,
            "improvement_areas": improvement_areas,
            "last_updated": datetime.now().isoformat()
        }
        
        self.agent_profiles[agent_id] = profile
        return profile
    
    async def _identify_agent_strengths(self, agent_tests: List[IntegratedTestResult]) -> List[str]:
        """Identify agent strengths based on test history."""
        strengths = []
        
        if not agent_tests:
            return strengths
        
        # Analyze performance patterns
        avg_data_quality = sum(test.data_quality_score for test in agent_tests) / len(agent_tests)
        avg_causal_validity = sum(test.causal_validity_score for test in agent_tests) / len(agent_tests)
        
        if avg_data_quality > 0.8:
            strengths.append("Excellent data quality assessment")
        
        if avg_causal_validity > 0.8:
            strengths.append("Strong causal reasoning capabilities")
        
        # Check consistency
        confidence_scores = [test.overall_confidence for test in agent_tests]
        if len(confidence_scores) > 1:
            std_dev = (sum((x - sum(confidence_scores)/len(confidence_scores))**2 for x in confidence_scores) / len(confidence_scores))**0.5
            if std_dev < 0.1:
                strengths.append("Consistent performance across scenarios")
        
        return strengths
    
    async def _identify_improvement_areas(self, agent_tests: List[IntegratedTestResult]) -> List[str]:
        """Identify areas for agent improvement."""
        improvement_areas = []
        
        if not agent_tests:
            return improvement_areas
        
        # Analyze weak points
        avg_data_quality = sum(test.data_quality_score for test in agent_tests) / len(agent_tests)
        avg_causal_validity = sum(test.causal_validity_score for test in agent_tests) / len(agent_tests)
        
        if avg_data_quality < 0.6:
            improvement_areas.append("Data quality assessment needs improvement")
        
        if avg_causal_validity < 0.6:
            improvement_areas.append("Causal reasoning requires strengthening")
        
        # Check for declining performance
        if len(agent_tests) >= 3:
            recent_confidence = sum(test.overall_confidence for test in agent_tests[-3:]) / 3
            earlier_confidence = sum(test.overall_confidence for test in agent_tests[:-3]) / (len(agent_tests) - 3)
            
            if recent_confidence < earlier_confidence * 0.9:
                improvement_areas.append("Recent performance decline detected")
        
        return improvement_areas


# Global unified integration manager instance
_global_integration_manager: Optional[UnifiedAgentIntegrationManager] = None


async def get_integration_manager() -> UnifiedAgentIntegrationManager:
    """Get the global unified integration manager instance."""
    global _global_integration_manager
    
    if _global_integration_manager is None:
        _global_integration_manager = UnifiedAgentIntegrationManager()
        await _global_integration_manager.initialize()
    
    return _global_integration_manager


async def initialize_unified_integration() -> None:
    """Initialize the unified integration layer."""
    await get_integration_manager()
    logger.info("Unified agent integration layer initialized")