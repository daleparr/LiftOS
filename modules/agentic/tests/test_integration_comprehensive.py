"""
Comprehensive Integration Tests for Agentic Microservice

This module tests the complete integration of all LiftOS Core components
including causal inference, observability, memory, and MMM capabilities.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch

# Import integration components
from ..integrations.unified_integration import UnifiedAgentIntegrationManager, IntegratedTestResult
from ..integrations.causal_integration import CausalAgentTestingEngine
from ..integrations.observability_integration import AgenticObservabilityManager
from ..integrations.memory_integration import AgentMemoryManager
from ..integrations.mmm_integration import MMMAgentTestingEngine, MarketingScenario

# Import core components
from ..core.data_quality_engine import DataQualityEngine


class TestUnifiedIntegration:
    """Test suite for unified integration functionality."""
    
    @pytest.fixture
    async def integration_manager(self):
        """Create integration manager for testing."""
        manager = UnifiedAgentIntegrationManager()
        
        # Mock the integration components
        manager.causal_engine = Mock(spec=CausalAgentTestingEngine)
        manager.observability_manager = Mock(spec=AgenticObservabilityManager)
        manager.memory_manager = Mock(spec=AgentMemoryManager)
        manager.mmm_engine = Mock(spec=MMMAgentTestingEngine)
        
        return manager
    
    @pytest.fixture
    def sample_test_scenario(self):
        """Sample test scenario for testing."""
        return {
            "name": "Marketing Campaign Test",
            "description": "Test agent performance on marketing campaign optimization",
            "type": "marketing",
            "treatment_variables": ["budget_allocation", "channel_mix"],
            "outcome_variables": ["conversions", "revenue"],
            "budget_allocation": {
                "google_ads": 5000,
                "facebook_ads": 3000,
                "email_marketing": 2000
            },
            "agent_recommendation": {
                "increase_google_budget": True,
                "optimize_facebook_targeting": True
            }
        }
    
    @pytest.fixture
    def sample_test_data(self):
        """Sample test data for testing."""
        return {
            "historical_campaigns": [
                {"budget": 10000, "conversions": 250, "revenue": 50000},
                {"budget": 8000, "conversions": 200, "revenue": 40000},
                {"budget": 12000, "conversions": 300, "revenue": 60000}
            ],
            "market_conditions": {
                "seasonality": "high",
                "competition": "medium",
                "economic_indicators": "positive"
            },
            "customer_segments": {
                "segment_a": {"size": 1000, "conversion_rate": 0.25},
                "segment_b": {"size": 1500, "conversion_rate": 0.15}
            }
        }
    
    @pytest.mark.asyncio
    async def test_comprehensive_agent_test_success(
        self, 
        integration_manager, 
        sample_test_scenario, 
        sample_test_data
    ):
        """Test successful comprehensive agent testing."""
        agent_id = "test_agent_001"
        
        # Mock component responses
        integration_manager.observability_manager.start_agent_trace = AsyncMock(return_value="trace_123")
        integration_manager.observability_manager.end_agent_trace = AsyncMock()
        integration_manager.memory_manager.store_agent_experience = AsyncMock(return_value="memory_456")
        integration_manager.memory_manager.retrieve_relevant_experiences = AsyncMock(return_value=[])
        integration_manager.memory_manager.get_agent_knowledge_summary = AsyncMock(return_value={
            "total_experiences": 10,
            "recent_experiences_7d": 3,
            "average_importance": 0.7
        })
        
        # Mock data quality assessment
        with patch.object(integration_manager, '_assess_data_quality') as mock_dq:
            mock_dq.return_value = {
                "overall_score": 0.85,
                "issues": ["Minor completeness issues"],
                "recommendations": ["Fill missing values"]
            }
            
            # Mock causal validation
            with patch.object(integration_manager, '_validate_causal_relationships') as mock_causal:
                mock_causal.return_value = {
                    "validity_score": 0.9,
                    "relationships": {"budget_allocation": "conversions"},
                    "confounders": ["seasonality"]
                }
                
                # Mock performance monitoring
                with patch.object(integration_manager, '_monitor_agent_performance') as mock_perf:
                    mock_perf.return_value = {
                        "success_rate": 0.88,
                        "response_time": 150,
                        "memory_usage_mb": 64
                    }
                    
                    # Run comprehensive test
                    result = await integration_manager.run_comprehensive_agent_test(
                        agent_id=agent_id,
                        test_scenario=sample_test_scenario,
                        test_data=sample_test_data,
                        include_mmm=False
                    )
                    
                    # Verify result
                    assert isinstance(result, IntegratedTestResult)
                    assert result.agent_id == agent_id
                    assert result.success is True
                    assert result.overall_confidence > 0.7
                    assert result.data_quality_score == 0.85
                    assert result.causal_validity_score == 0.9
                    assert len(result.recommendations) > 0
                    
                    # Verify component interactions
                    integration_manager.observability_manager.start_agent_trace.assert_called_once()
                    integration_manager.observability_manager.end_agent_trace.assert_called_once()
                    integration_manager.memory_manager.store_agent_experience.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_comprehensive_agent_test_with_mmm(
        self, 
        integration_manager, 
        sample_test_scenario, 
        sample_test_data
    ):
        """Test comprehensive agent testing with MMM integration."""
        agent_id = "test_agent_002"
        
        # Mock MMM testing
        with patch.object(integration_manager, '_test_marketing_performance') as mock_mmm:
            mock_mmm.return_value = {
                "predictions": {"roas": 4.2, "conversions": 280},
                "performance": {"overall_score": 0.82}
            }
            
            # Mock other components
            integration_manager.observability_manager.start_agent_trace = AsyncMock(return_value="trace_124")
            integration_manager.observability_manager.end_agent_trace = AsyncMock()
            integration_manager.memory_manager.store_agent_experience = AsyncMock(return_value="memory_457")
            integration_manager.memory_manager.retrieve_relevant_experiences = AsyncMock(return_value=[])
            integration_manager.memory_manager.get_agent_knowledge_summary = AsyncMock(return_value={})
            
            with patch.object(integration_manager, '_assess_data_quality') as mock_dq:
                mock_dq.return_value = {"overall_score": 0.9, "issues": [], "recommendations": []}
                
                with patch.object(integration_manager, '_validate_causal_relationships') as mock_causal:
                    mock_causal.return_value = {"validity_score": 0.85, "relationships": {}, "confounders": []}
                    
                    with patch.object(integration_manager, '_monitor_agent_performance') as mock_perf:
                        mock_perf.return_value = {"success_rate": 0.9, "response_time": 120}
                        
                        # Run test with MMM
                        result = await integration_manager.run_comprehensive_agent_test(
                            agent_id=agent_id,
                            test_scenario=sample_test_scenario,
                            test_data=sample_test_data,
                            include_mmm=True
                        )
                        
                        # Verify MMM results included
                        assert result.mmm_predictions is not None
                        assert result.marketing_performance is not None
                        assert result.mmm_predictions["roas"] == 4.2
                        assert result.marketing_performance["overall_score"] == 0.82
                        
                        # Verify MMM testing was called
                        mock_mmm.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_agent_profile_generation(self, integration_manager):
        """Test agent profile generation with comprehensive insights."""
        agent_id = "test_agent_003"
        
        # Mock test results
        test_result_1 = IntegratedTestResult(
            test_id="test_1",
            agent_id=agent_id,
            test_type="marketing",
            timestamp=datetime.now() - timedelta(days=1),
            data_quality_score=0.85,
            data_quality_issues=[],
            causal_validity_score=0.9,
            causal_relationships={},
            confounders_identified=[],
            performance_metrics={"success_rate": 0.88},
            observability_traces=["trace_1"],
            relevant_experiences=[],
            learning_patterns=[],
            mmm_predictions=None,
            marketing_performance=None,
            overall_confidence=0.87,
            recommendations=[],
            success=True
        )
        
        test_result_2 = IntegratedTestResult(
            test_id="test_2",
            agent_id=agent_id,
            test_type="general",
            timestamp=datetime.now(),
            data_quality_score=0.9,
            data_quality_issues=[],
            causal_validity_score=0.85,
            causal_relationships={},
            confounders_identified=[],
            performance_metrics={"success_rate": 0.92},
            observability_traces=["trace_2"],
            relevant_experiences=[],
            learning_patterns=[],
            mmm_predictions=None,
            marketing_performance=None,
            overall_confidence=0.89,
            recommendations=[],
            success=True
        )
        
        integration_manager.test_results = {
            "test_1": test_result_1,
            "test_2": test_result_2
        }
        
        # Mock memory and performance summaries
        integration_manager.memory_manager.get_agent_knowledge_summary = AsyncMock(return_value={
            "total_experiences": 15,
            "recent_experiences_7d": 5,
            "average_importance": 0.75
        })
        
        integration_manager.observability_manager.get_agent_performance_summary = AsyncMock(return_value={
            "total_operations": 25,
            "average_response_time": 145,
            "success_rate": 0.9
        })
        
        with patch.object(integration_manager, '_identify_agent_strengths') as mock_strengths:
            mock_strengths.return_value = ["Consistent performance", "Strong causal reasoning"]
            
            with patch.object(integration_manager, '_identify_improvement_areas') as mock_improvements:
                mock_improvements.return_value = ["Data quality assessment"]
                
                # Generate profile
                profile = await integration_manager.get_agent_profile(agent_id)
                
                # Verify profile structure
                assert profile["agent_id"] == agent_id
                assert profile["total_tests"] == 2
                assert profile["average_confidence"] == 0.88  # (0.87 + 0.89) / 2
                assert len(profile["recent_tests"]) == 2
                assert len(profile["strengths"]) == 2
                assert len(profile["improvement_areas"]) == 1
                assert "memory_summary" in profile
                assert "performance_summary" in profile
    
    @pytest.mark.asyncio
    async def test_confidence_calculation(self, integration_manager):
        """Test overall confidence score calculation."""
        # Test data with different quality scores
        data_quality_result = {"overall_score": 0.8}
        causal_result = {"validity_score": 0.9}
        performance_metrics = {"success_rate": 0.85}
        mmm_result = {"performance": {"overall_score": 0.75}}
        
        # Test without MMM
        confidence = await integration_manager._calculate_overall_confidence(
            data_quality_result, causal_result, performance_metrics, None
        )
        
        # Expected: (0.3*0.8 + 0.3*0.9 + 0.4*0.85) = 0.24 + 0.27 + 0.34 = 0.85
        assert abs(confidence - 0.85) < 0.01
        
        # Test with MMM
        confidence_with_mmm = await integration_manager._calculate_overall_confidence(
            data_quality_result, causal_result, performance_metrics, mmm_result
        )
        
        # Expected: (0.3*0.8 + 0.3*0.9 + 0.2*0.85 + 0.2*0.75) = 0.24 + 0.27 + 0.17 + 0.15 = 0.83
        assert abs(confidence_with_mmm - 0.83) < 0.01
    
    @pytest.mark.asyncio
    async def test_error_handling(self, integration_manager, sample_test_scenario, sample_test_data):
        """Test error handling in comprehensive testing."""
        agent_id = "test_agent_error"
        
        # Mock observability to raise exception
        integration_manager.observability_manager.start_agent_trace = AsyncMock(side_effect=Exception("Trace failed"))
        integration_manager.observability_manager.end_agent_trace = AsyncMock()
        
        # Test should handle errors gracefully
        with pytest.raises(Exception) as exc_info:
            await integration_manager.run_comprehensive_agent_test(
                agent_id=agent_id,
                test_scenario=sample_test_scenario,
                test_data=sample_test_data
            )
        
        assert "Trace failed" in str(exc_info.value)
        
        # Verify cleanup was attempted
        integration_manager.observability_manager.end_agent_trace.assert_called_once()


class TestMemoryIntegration:
    """Test suite for memory integration functionality."""
    
    @pytest.fixture
    async def memory_manager(self):
        """Create memory manager for testing."""
        return AgentMemoryManager()
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_experience(self, memory_manager):
        """Test storing and retrieving agent experiences."""
        agent_id = "test_agent_memory"
        experience_data = {
            "decision": "increase_budget",
            "outcome": "improved_performance",
            "metrics": {"roas": 4.2, "conversions": 150}
        }
        
        # Store experience
        memory_id = await memory_manager.store_agent_experience(
            agent_id=agent_id,
            experience_type="decision",
            experience_data=experience_data,
            importance_score=0.8
        )
        
        assert memory_id is not None
        assert agent_id in memory_manager.agent_memories
        assert len(memory_manager.agent_memories[agent_id]) == 1
        
        stored_experience = memory_manager.agent_memories[agent_id][0]
        assert stored_experience["experience_type"] == "decision"
        assert stored_experience["importance_score"] == 0.8
        assert stored_experience["experience_data"] == experience_data
        
        # Retrieve experiences
        with patch('modules.agentic.integrations.memory_integration.kse_client') as mock_kse:
            mock_kse.search_memories = AsyncMock(return_value=Mock(results=[]))
            
            experiences = await memory_manager.retrieve_relevant_experiences(
                agent_id=agent_id,
                query="budget decision",
                limit=5
            )
            
            # Should fall back to local search
            assert len(experiences) == 1
            assert experiences[0]["experience_type"] == "decision"
    
    @pytest.mark.asyncio
    async def test_knowledge_summary(self, memory_manager):
        """Test agent knowledge summary generation."""
        agent_id = "test_agent_summary"
        
        # Add multiple experiences
        for i in range(5):
            await memory_manager.store_agent_experience(
                agent_id=agent_id,
                experience_type="test",
                experience_data={"test_id": i},
                importance_score=0.5 + (i * 0.1)
            )
        
        summary = await memory_manager.get_agent_knowledge_summary(agent_id)
        
        assert summary["agent_id"] == agent_id
        assert summary["total_experiences"] == 5
        assert summary["average_importance"] == 0.7  # (0.5+0.6+0.7+0.8+0.9)/5
        assert summary["knowledge_depth_score"] == 0.05  # 5/100
        assert "experience_type_distribution" in summary


class TestMMMIntegration:
    """Test suite for MMM integration functionality."""
    
    @pytest.fixture
    async def mmm_engine(self):
        """Create MMM testing engine for testing."""
        return MMMAgentTestingEngine()
    
    @pytest.mark.asyncio
    async def test_create_marketing_scenario(self, mmm_engine):
        """Test marketing scenario creation."""
        scenario = await mmm_engine.create_marketing_scenario(
            name="Test Campaign",
            description="Test marketing campaign scenario",
            total_budget=10000,
            available_channels=["google_ads", "facebook_ads", "email"],
            target_metrics={"roas": 4.0, "conversions": 250},
            duration_days=30
        )
        
        assert isinstance(scenario, MarketingScenario)
        assert scenario.name == "Test Campaign"
        assert sum(scenario.budget_allocation.values()) == 10000
        assert len(scenario.budget_allocation) == 3
        assert scenario.target_metrics["roas"] == 4.0
        assert scenario.duration_days == 30
        
        # Verify scenario is stored
        assert scenario.scenario_id in mmm_engine.test_scenarios
    
    @pytest.mark.asyncio
    async def test_agent_recommendation_testing(self, mmm_engine):
        """Test agent recommendation testing with MMM."""
        # Create scenario
        scenario = await mmm_engine.create_marketing_scenario(
            name="Test Scenario",
            description="Test scenario for agent testing",
            total_budget=10000,
            available_channels=["google_ads", "facebook_ads"],
            target_metrics={"roas": 4.0}
        )
        
        # Test agent recommendation
        agent_recommendation = {
            "google_ads": 6000,
            "facebook_ads": 4000
        }
        
        with patch.object(mmm_engine, '_predict_mmm_outcomes') as mock_predict:
            mock_predict.return_value = {
                "total_conversions": 250,
                "total_revenue": 40000,
                "roas": 4.0,
                "cpa": 40,
                "attribution_scores": {"google_ads": 0.6, "facebook_ads": 0.4}
            }
            
            result = await mmm_engine.test_agent_recommendation(
                agent_id="test_agent",
                scenario_id=scenario.scenario_id,
                agent_recommendation=agent_recommendation,
                reasoning="Optimize for higher ROAS"
            )
            
            assert result["valid"] is True
            assert result["agent_id"] == "test_agent"
            assert result["predicted_outcomes"]["roas"] == 4.0
            assert result["confidence_score"] == 1.0  # Perfect match with target
            assert "baseline_comparison" in result
            assert "optimization_suggestions" in result
    
    @pytest.mark.asyncio
    async def test_agent_performance_summary(self, mmm_engine):
        """Test agent performance summary generation."""
        agent_id = "test_agent_perf"
        
        # Add some test recommendations
        from datetime import datetime
        from ..integrations.mmm_integration import AgentRecommendation
        
        recommendations = []
        for i in range(3):
            rec = AgentRecommendation(
                agent_id=agent_id,
                scenario_id=f"scenario_{i}",
                recommended_allocation={"google_ads": 5000, "facebook_ads": 5000},
                predicted_outcomes={"roas": 3.5 + (i * 0.2)},
                confidence_score=0.7 + (i * 0.1),
                reasoning=f"Test reasoning {i}",
                timestamp=datetime.now()
            )
            recommendations.append(rec)
        
        mmm_engine.agent_recommendations[agent_id] = recommendations
        
        summary = await mmm_engine.get_agent_performance_summary(agent_id)
        
        assert summary["agent_id"] == agent_id
        assert summary["total_tests"] == 3
        assert summary["average_confidence"] == 0.8  # (0.7+0.8+0.9)/3
        assert summary["best_performance"]["confidence_score"] == 0.9
        assert summary["improvement_trend"] == 0.0  # Not enough data for trend


if __name__ == "__main__":
    pytest.main([__file__, "-v"])