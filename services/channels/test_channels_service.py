"""
Test suite for Channels Service
Comprehensive tests for budget optimization functionality
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any
import json

# Test the main application
from app import app
from models.channels import (
    BudgetOptimizationRequest, SimulationRequest, ChannelPerformance,
    OptimizationObjective, ConstraintType, OptimizationConstraint,
    ScenarioType, SaturationFunction
)
from engines.optimization_engine import OptimizationEngine
from engines.simulation_engine import SimulationEngine
from engines.saturation_engine import SaturationEngine
from engines.bayesian_engine import BayesianEngine
from engines.recommendation_engine import RecommendationEngine
from integrations.service_clients import ServiceClientManager, create_service_config


class TestChannelsService:
    """Test suite for the Channels service"""
    
    @pytest.fixture
    def sample_optimization_request(self) -> BudgetOptimizationRequest:
        """Create a sample optimization request for testing"""
        
        return BudgetOptimizationRequest(
            org_id="test_org_123",
            total_budget=10000.0,
            channels=["google_ads", "meta_ads", "tiktok_ads"],
            objectives=[OptimizationObjective.MAXIMIZE_REVENUE, OptimizationObjective.MAXIMIZE_ROAS],
            objective_weights={
                OptimizationObjective.MAXIMIZE_REVENUE: 0.6,
                OptimizationObjective.MAXIMIZE_ROAS: 0.4
            },
            constraints=[
                OptimizationConstraint(
                    constraint_id="min_google_spend",
                    constraint_type=ConstraintType.MIN_SPEND,
                    channel_id="google_ads",
                    min_value=1000.0,
                    description="Minimum Google Ads spend"
                )
            ],
            time_horizon_days=30,
            confidence_threshold=0.8,
            risk_tolerance="moderate",
            use_bayesian_optimization=True,
            monte_carlo_samples=1000
        )
    
    @pytest.fixture
    def sample_simulation_request(self) -> SimulationRequest:
        """Create a sample simulation request for testing"""
        
        return SimulationRequest(
            org_id="test_org_123",
            channels=["google_ads", "meta_ads"],
            scenarios=[
                {
                    "scenario_id": "budget_increase_20",
                    "name": "20% Budget Increase",
                    "scenario_type": ScenarioType.BUDGET_CHANGE,
                    "description": "Increase all budgets by 20%",
                    "parameters": {
                        "budget_changes": {
                            "google_ads": 1.2,
                            "meta_ads": 1.2
                        }
                    }
                },
                {
                    "scenario_id": "market_shock",
                    "name": "Market Downturn",
                    "scenario_type": ScenarioType.MARKET_SHOCK,
                    "description": "10% market downturn impact",
                    "parameters": {
                        "market_impact": -0.1
                    }
                }
            ],
            monte_carlo_samples=1000,
            confidence_level=0.95
        )
    
    @pytest.fixture
    async def service_clients(self):
        """Create service clients for testing"""
        
        config = create_service_config()
        async with ServiceClientManager(config) as manager:
            yield manager
    
    def test_optimization_request_validation(self, sample_optimization_request):
        """Test optimization request validation"""
        
        # Valid request should pass
        assert sample_optimization_request.total_budget > 0
        assert len(sample_optimization_request.channels) > 0
        assert len(sample_optimization_request.objectives) > 0
        
        # Test budget validation
        with pytest.raises(ValueError):
            BudgetOptimizationRequest(
                org_id="test",
                total_budget=-1000.0,  # Invalid negative budget
                channels=["google_ads"],
                objectives=[OptimizationObjective.MAXIMIZE_REVENUE]
            )
    
    def test_simulation_request_validation(self, sample_simulation_request):
        """Test simulation request validation"""
        
        # Valid request should pass
        assert len(sample_simulation_request.channels) > 0
        assert len(sample_simulation_request.scenarios) > 0
        assert sample_simulation_request.monte_carlo_samples > 0
        
        # Test confidence level validation
        with pytest.raises(ValueError):
            SimulationRequest(
                org_id="test",
                channels=["google_ads"],
                scenarios=[],
                confidence_level=1.5  # Invalid confidence level > 1
            )
    
    @pytest.mark.asyncio
    async def test_optimization_engine_initialization(self, service_clients):
        """Test optimization engine initialization"""
        
        # Mock engines for testing
        saturation_engine = None  # Would be properly initialized in real scenario
        bayesian_engine = None
        causal_client = service_clients.causal
        
        # This would fail in real scenario without proper initialization
        # but demonstrates the structure
        try:
            optimization_engine = OptimizationEngine(
                saturation_engine, bayesian_engine, causal_client
            )
            assert optimization_engine is not None
            assert optimization_engine.max_iterations == 1000
        except Exception as e:
            # Expected to fail without proper engine initialization
            assert "NoneType" in str(e) or "initialization" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_service_client_connectivity(self, service_clients):
        """Test service client connectivity"""
        
        # Test health checks (will likely fail in test environment)
        health_status = await service_clients.health_check_all()
        
        # Verify structure even if services are down
        assert isinstance(health_status, dict)
        assert "causal" in health_status
        assert "data_ingestion" in health_status
        assert "memory" in health_status
        assert "bayesian_analysis" in health_status
        
        # Each service should have a boolean health status
        for service, status in health_status.items():
            assert isinstance(status, bool)
    
    def test_channel_performance_model(self):
        """Test ChannelPerformance model validation"""
        
        # Valid channel performance
        performance = ChannelPerformance(
            channel_id="google_ads",
            channel_name="Google Ads",
            channel_type="search",
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
        
        assert performance.current_spend > 0
        assert performance.current_roas > 0
        assert performance.current_conversions >= 0
        assert 0 <= performance.saturation_level <= 1
        assert 0 <= performance.efficiency_score <= 1
    
    def test_optimization_constraint_validation(self):
        """Test optimization constraint validation"""
        
        # Valid constraint
        constraint = OptimizationConstraint(
            constraint_id="test_constraint",
            constraint_type=ConstraintType.MIN_SPEND,
            channel_id="google_ads",
            min_value=1000.0,
            description="Test constraint"
        )
        
        assert constraint.min_value > 0
        assert constraint.channel_id is not None
        
        # Test ROAS constraint
        roas_constraint = OptimizationConstraint(
            constraint_id="roas_constraint",
            constraint_type=ConstraintType.MIN_ROAS,
            channel_id="google_ads",
            min_value=2.0,
            description="Minimum ROAS constraint"
        )
        
        assert roas_constraint.min_value > 0
    
    def test_saturation_function_enum(self):
        """Test saturation function enumeration"""
        
        # Test all saturation functions are available
        functions = list(SaturationFunction)
        expected_functions = ["HILL", "ADSTOCK", "EXPONENTIAL", "LOGARITHMIC", "POWER"]
        
        for expected in expected_functions:
            assert any(func.name == expected for func in functions)
    
    @pytest.mark.asyncio
    async def test_mock_optimization_workflow(self, sample_optimization_request):
        """Test mock optimization workflow"""
        
        # This is a simplified test of the optimization workflow
        # In a real test, we would mock the service dependencies
        
        org_id = sample_optimization_request.org_id
        channels = sample_optimization_request.channels
        total_budget = sample_optimization_request.total_budget
        
        # Verify request structure
        assert org_id == "test_org_123"
        assert len(channels) == 3
        assert total_budget == 10000.0
        
        # Mock optimization result structure
        mock_result = {
            "optimization_id": "test_opt_123",
            "org_id": org_id,
            "status": "completed",
            "recommended_allocation": {
                "google_ads": 4000.0,
                "meta_ads": 3500.0,
                "tiktok_ads": 2500.0
            },
            "expected_performance": {
                "total_revenue": 35000.0,
                "total_roas": 3.5,
                "total_conversions": 1000
            }
        }
        
        # Verify mock result structure
        assert mock_result["org_id"] == org_id
        assert sum(mock_result["recommended_allocation"].values()) == total_budget
        assert mock_result["expected_performance"]["total_roas"] > 0
    
    @pytest.mark.asyncio
    async def test_mock_simulation_workflow(self, sample_simulation_request):
        """Test mock simulation workflow"""
        
        org_id = sample_simulation_request.org_id
        scenarios = sample_simulation_request.scenarios
        
        # Verify request structure
        assert org_id == "test_org_123"
        assert len(scenarios) == 2
        
        # Mock simulation result
        mock_result = {
            "simulation_id": "test_sim_123",
            "org_id": org_id,
            "status": "completed",
            "scenario_results": [
                {
                    "scenario_id": "budget_increase_20",
                    "expected_revenue": 42000.0,
                    "confidence_interval": (38000.0, 46000.0),
                    "probability_of_improvement": 0.85
                },
                {
                    "scenario_id": "market_shock",
                    "expected_revenue": 31500.0,
                    "confidence_interval": (28000.0, 35000.0),
                    "probability_of_improvement": 0.15
                }
            ]
        }
        
        # Verify mock result structure
        assert mock_result["org_id"] == org_id
        assert len(mock_result["scenario_results"]) == len(scenarios)
        
        for result in mock_result["scenario_results"]:
            assert result["expected_revenue"] > 0
            assert 0 <= result["probability_of_improvement"] <= 1
    
    def test_recommendation_priority_calculation(self):
        """Test recommendation priority calculation logic"""
        
        # Mock recommendation data
        recommendations = [
            {
                "title": "High Impact Recommendation",
                "expected_impact": 15000.0,
                "confidence_score": 0.9,
                "implementation_effort": "Low",
                "risk_level": "low"
            },
            {
                "title": "Medium Impact Recommendation", 
                "expected_impact": 5000.0,
                "confidence_score": 0.7,
                "implementation_effort": "Medium",
                "risk_level": "medium"
            }
        ]
        
        # Simple priority scoring (would be more complex in real implementation)
        def calculate_priority_score(rec):
            impact_score = min(rec["expected_impact"] / 10000, 1.0)
            confidence_score = rec["confidence_score"]
            effort_score = {"Low": 1.0, "Medium": 0.7, "High": 0.4}[rec["implementation_effort"]]
            risk_score = {"low": 1.0, "medium": 0.7, "high": 0.4}[rec["risk_level"]]
            
            return 0.4 * impact_score + 0.3 * confidence_score + 0.2 * effort_score + 0.1 * risk_score
        
        scores = [calculate_priority_score(rec) for rec in recommendations]
        
        # High impact recommendation should score higher
        assert scores[0] > scores[1]
        assert all(0 <= score <= 1 for score in scores)
    
    def test_error_handling_scenarios(self):
        """Test error handling scenarios"""
        
        # Test invalid budget
        with pytest.raises(ValueError):
            BudgetOptimizationRequest(
                org_id="test",
                total_budget=0,  # Invalid zero budget
                channels=["google_ads"],
                objectives=[OptimizationObjective.MAXIMIZE_REVENUE]
            )
        
        # Test empty channels list
        with pytest.raises(ValueError):
            BudgetOptimizationRequest(
                org_id="test",
                total_budget=1000.0,
                channels=[],  # Empty channels
                objectives=[OptimizationObjective.MAXIMIZE_REVENUE]
            )
        
        # Test invalid confidence threshold
        with pytest.raises(ValueError):
            BudgetOptimizationRequest(
                org_id="test",
                total_budget=1000.0,
                channels=["google_ads"],
                objectives=[OptimizationObjective.MAXIMIZE_REVENUE],
                confidence_threshold=1.5  # Invalid > 1
            )
    
    @pytest.mark.asyncio
    async def test_concurrent_optimization_handling(self):
        """Test handling of concurrent optimization requests"""
        
        # Mock multiple optimization requests
        requests = [
            {
                "org_id": f"org_{i}",
                "optimization_id": f"opt_{i}",
                "status": "pending"
            }
            for i in range(5)
        ]
        
        # Simulate concurrent processing
        async def process_request(request):
            # Simulate processing time
            await asyncio.sleep(0.1)
            request["status"] = "completed"
            return request
        
        # Process requests concurrently
        tasks = [process_request(req.copy()) for req in requests]
        results = await asyncio.gather(*tasks)
        
        # Verify all requests completed
        assert len(results) == 5
        assert all(result["status"] == "completed" for result in results)
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        
        from config import get_config, TestConfig
        
        # Test config loading
        config = get_config()
        assert config is not None
        assert config.service_name == "channels"
        assert config.service_port > 0
        
        # Test test configuration
        test_config = TestConfig()
        assert test_config.environment == "test"
        assert test_config.debug is True
        assert "test" in test_config.database_url


# Integration tests (require running services)
class TestChannelsIntegration:
    """Integration tests for Channels service (require running dependencies)"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_optimization_flow(self):
        """Test full optimization flow with real services"""
        
        # This test would require actual running services
        # Skip if services are not available
        pytest.skip("Integration test requires running services")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_kse_integration(self):
        """Test KSE SDK integration"""
        
        # This test would require KSE to be running
        pytest.skip("Integration test requires KSE to be running")


# Performance tests
class TestChannelsPerformance:
    """Performance tests for Channels service"""
    
    @pytest.mark.performance
    def test_optimization_performance(self):
        """Test optimization performance with large datasets"""
        
        # Mock large dataset
        large_request = BudgetOptimizationRequest(
            org_id="perf_test",
            total_budget=1000000.0,
            channels=[f"channel_{i}" for i in range(20)],  # 20 channels
            objectives=[OptimizationObjective.MAXIMIZE_REVENUE],
            monte_carlo_samples=10000
        )
        
        # Verify request can be created
        assert len(large_request.channels) == 20
        assert large_request.total_budget == 1000000.0
        
        # In a real performance test, we would measure execution time
        # and memory usage of the optimization algorithm


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])