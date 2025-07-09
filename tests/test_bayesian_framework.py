"""
Comprehensive test suite for LiftOS Bayesian Framework
Tests prior-data conflict detection, SBC validation, and Bayesian integration
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
from typing import Dict, Any, List

# Import framework components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.models.bayesian_priors import (
    PriorElicitationRequest, PriorUpdateRequest, ConflictAnalysisRequest,
    SBCValidationRequest, BayesianSessionRequest, PriorBelief,
    ConflictAnalysisResult, UpdateRecommendation, SBCValidationResult
)
from shared.utils.bayesian_diagnostics import (
    ConflictAnalyzer, PriorUpdater, BayesianDiagnostics
)
from shared.validation.simulation_based_calibration import (
    SBCValidator, SBCDecisionFramework, SBCDiagnostics
)
from shared.database.bayesian_models import BayesianDatabaseManager


class TestBayesianPriorModels:
    """Test Bayesian prior data models"""
    
    def test_prior_belief_creation(self):
        """Test PriorBelief model creation and validation"""
        prior = PriorBelief(
            parameter_name="google_ads_attribution",
            parameter_category="channel_attribution",
            distribution_type="beta",
            distribution_params={"alpha": 2.0, "beta": 5.0},
            confidence_level=0.8,
            source="client_expertise"
        )
        
        assert prior.parameter_name == "google_ads_attribution"
        assert prior.distribution_type == "beta"
        assert prior.confidence_level == 0.8
        assert prior.distribution_params["alpha"] == 2.0
    
    def test_prior_elicitation_request(self):
        """Test PriorElicitationRequest validation"""
        request = PriorElicitationRequest(
            user_id="test_user",
            session_name="Test Session",
            analysis_type="marketing_attribution",
            prior_beliefs=[
                PriorBelief(
                    parameter_name="facebook_ads_attribution",
                    parameter_category="channel_attribution",
                    distribution_type="normal",
                    distribution_params={"mean": 0.25, "std": 0.05},
                    confidence_level=0.9
                )
            ]
        )
        
        assert len(request.prior_beliefs) == 1
        assert request.analysis_type == "marketing_attribution"
    
    def test_conflict_analysis_result(self):
        """Test ConflictAnalysisResult model"""
        result = ConflictAnalysisResult(
            conflict_detected=True,
            conflict_severity="moderate",
            evidence_strength="strong",
            bayes_factor=8.5,
            kl_divergence=0.42,
            wasserstein_distance=0.35,
            conflicting_parameters=["google_ads_attribution", "facebook_ads_attribution"],
            recommendations=[
                UpdateRecommendation(
                    parameter_name="google_ads_attribution",
                    current_prior={"mean": 0.4, "std": 0.1},
                    recommended_prior={"mean": 0.35, "std": 0.08},
                    update_strength=0.7,
                    evidence_weight=0.8,
                    rationale="Strong evidence suggests lower attribution"
                )
            ]
        )
        
        assert result.conflict_detected is True
        assert result.bayes_factor == 8.5
        assert len(result.recommendations) == 1


class TestConflictAnalyzer:
    """Test prior-data conflict detection"""
    
    @pytest.fixture
    def conflict_analyzer(self):
        return ConflictAnalyzer()
    
    @pytest.fixture
    def sample_priors(self):
        return {
            "google_ads_attribution": {
                "distribution": "beta",
                "params": {"alpha": 3, "beta": 7},  # Prior belief: ~30% attribution
                "confidence": 0.8
            },
            "facebook_ads_attribution": {
                "distribution": "beta", 
                "params": {"alpha": 2, "beta": 8},  # Prior belief: ~20% attribution
                "confidence": 0.7
            }
        }
    
    @pytest.fixture
    def sample_data(self):
        return {
            "attribution_scores": {
                "google_ads_attribution": 0.45,  # Observed: 45% attribution
                "facebook_ads_attribution": 0.35  # Observed: 35% attribution
            },
            "sample_size": 1000,
            "confidence_intervals": {
                "google_ads_attribution": [0.40, 0.50],
                "facebook_ads_attribution": [0.30, 0.40]
            }
        }
    
    def test_calculate_bayes_factor(self, conflict_analyzer, sample_priors, sample_data):
        """Test Bayes factor calculation"""
        bf = conflict_analyzer.calculate_bayes_factor(
            prior_params=sample_priors["google_ads_attribution"]["params"],
            observed_value=sample_data["attribution_scores"]["google_ads_attribution"],
            sample_size=sample_data["sample_size"]
        )
        
        assert isinstance(bf, float)
        assert bf > 0
        # Strong evidence against prior (BF > 10 indicates strong evidence)
        assert bf > 10
    
    def test_calculate_kl_divergence(self, conflict_analyzer, sample_priors, sample_data):
        """Test KL divergence calculation"""
        kl_div = conflict_analyzer.calculate_kl_divergence(
            prior_params=sample_priors["google_ads_attribution"]["params"],
            posterior_params={"alpha": 450, "beta": 550},  # Updated with data
            distribution_type="beta"
        )
        
        assert isinstance(kl_div, float)
        assert kl_div >= 0
        assert kl_div > 0.1  # Significant divergence
    
    def test_calculate_wasserstein_distance(self, conflict_analyzer):
        """Test Wasserstein distance calculation"""
        prior_samples = np.random.beta(3, 7, 1000)
        posterior_samples = np.random.beta(450, 550, 1000)
        
        distance = conflict_analyzer.calculate_wasserstein_distance(
            prior_samples, posterior_samples
        )
        
        assert isinstance(distance, float)
        assert distance >= 0
        assert distance > 0.05  # Significant distance
    
    @pytest.mark.asyncio
    async def test_analyze_conflict(self, conflict_analyzer, sample_priors, sample_data):
        """Test complete conflict analysis"""
        result = await conflict_analyzer.analyze_conflict(
            prior_beliefs=sample_priors,
            observed_data=sample_data,
            user_id="test_user"
        )
        
        assert isinstance(result, dict)
        assert "conflict_detected" in result
        assert "conflict_severity" in result
        assert "evidence_strength" in result
        assert "bayes_factor" in result
        assert "recommendations" in result
        
        # Should detect conflict given the mismatch
        assert result["conflict_detected"] is True
        assert result["conflict_severity"] in ["low", "moderate", "high", "severe"]
    
    def test_classify_conflict_severity(self, conflict_analyzer):
        """Test conflict severity classification"""
        # Test different Bayes factor levels
        assert conflict_analyzer.classify_conflict_severity(2.5) == "low"
        assert conflict_analyzer.classify_conflict_severity(8.0) == "moderate" 
        assert conflict_analyzer.classify_conflict_severity(25.0) == "high"
        assert conflict_analyzer.classify_conflict_severity(150.0) == "severe"
    
    def test_assess_evidence_strength(self, conflict_analyzer):
        """Test evidence strength assessment"""
        # Test different evidence combinations
        assert conflict_analyzer.assess_evidence_strength(
            bayes_factor=5.0, sample_size=100, effect_size=0.2
        ) == "moderate"
        
        assert conflict_analyzer.assess_evidence_strength(
            bayes_factor=50.0, sample_size=1000, effect_size=0.8
        ) == "very_strong"


class TestPriorUpdater:
    """Test Bayesian prior updating"""
    
    @pytest.fixture
    def prior_updater(self):
        return PriorUpdater()
    
    def test_update_beta_prior(self, prior_updater):
        """Test Beta prior updating with binomial data"""
        updated = prior_updater.update_beta_prior(
            prior_alpha=2,
            prior_beta=8,
            observed_successes=450,
            observed_trials=1000
        )
        
        assert "alpha" in updated
        assert "beta" in updated
        assert updated["alpha"] == 452  # 2 + 450
        assert updated["beta"] == 558   # 8 + (1000 - 450)
    
    def test_update_normal_prior(self, prior_updater):
        """Test Normal prior updating"""
        updated = prior_updater.update_normal_prior(
            prior_mean=0.3,
            prior_variance=0.01,
            observed_mean=0.45,
            observed_variance=0.005,
            sample_size=1000
        )
        
        assert "mean" in updated
        assert "variance" in updated
        # Updated mean should be between prior and observed
        assert 0.3 < updated["mean"] < 0.45
        # Updated variance should be smaller
        assert updated["variance"] < 0.01
    
    @pytest.mark.asyncio
    async def test_generate_recommendations(self, prior_updater):
        """Test update recommendation generation"""
        conflict_result = {
            "conflicting_parameters": ["google_ads_attribution"],
            "bayes_factor": 15.0,
            "evidence_strength": "strong"
        }
        
        prior_beliefs = {
            "google_ads_attribution": {
                "distribution": "beta",
                "params": {"alpha": 3, "beta": 7},
                "confidence": 0.8
            }
        }
        
        observed_data = {
            "attribution_scores": {"google_ads_attribution": 0.45},
            "sample_size": 1000
        }
        
        recommendations = await prior_updater.generate_recommendations(
            conflict_result=conflict_result,
            prior_beliefs=prior_beliefs,
            observed_data=observed_data
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        rec = recommendations[0]
        assert "parameter_name" in rec
        assert "current_prior" in rec
        assert "recommended_prior" in rec
        assert "update_strength" in rec
        assert "rationale" in rec


class TestSBCValidator:
    """Test Simulation Based Calibration"""
    
    @pytest.fixture
    def sbc_validator(self):
        return SBCValidator()
    
    @pytest.fixture
    def sample_model_params(self):
        return {
            "google_ads_attribution": {"alpha": 3, "beta": 7},
            "facebook_ads_attribution": {"alpha": 2, "beta": 8},
            "email_attribution": {"alpha": 1, "beta": 9}
        }
    
    @pytest.mark.asyncio
    async def test_run_sbc_validation(self, sbc_validator, sample_model_params):
        """Test SBC validation execution"""
        result = await sbc_validator.run_sbc_validation(
            model_parameters=sample_model_params,
            num_simulations=100,  # Reduced for testing
            validation_type="marketing_attribution"
        )
        
        assert isinstance(result, dict)
        assert "validation_passed" in result
        assert "rank_statistics" in result
        assert "coverage_probability" in result
        assert "diagnostic_plots" in result
        
        # Check rank statistics structure
        rank_stats = result["rank_statistics"]
        assert "mean_rank" in rank_stats
        assert "rank_variance" in rank_stats
        assert 0 <= rank_stats["mean_rank"] <= 1
    
    def test_calculate_rank_statistics(self, sbc_validator):
        """Test rank statistics calculation"""
        # Simulate well-calibrated ranks (should be uniform)
        ranks = np.random.uniform(0, 1, 1000)
        
        stats = sbc_validator.calculate_rank_statistics(ranks)
        
        assert "mean_rank" in stats
        assert "rank_variance" in stats
        assert "uniformity_test_p_value" in stats
        
        # Mean should be close to 0.5 for uniform distribution
        assert 0.4 < stats["mean_rank"] < 0.6
    
    def test_assess_coverage_probability(self, sbc_validator):
        """Test coverage probability assessment"""
        # Simulate credible intervals and true values
        intervals = [(0.2, 0.8) for _ in range(100)]
        true_values = [0.5 for _ in range(100)]  # All within intervals
        
        coverage = sbc_validator.assess_coverage_probability(intervals, true_values)
        
        assert coverage == 1.0  # 100% coverage
        
        # Test partial coverage
        true_values_partial = [0.1 if i < 20 else 0.5 for i in range(100)]
        coverage_partial = sbc_validator.assess_coverage_probability(intervals, true_values_partial)
        
        assert coverage_partial == 0.8  # 80% coverage


class TestSBCDecisionFramework:
    """Test SBC decision logic"""
    
    @pytest.fixture
    def decision_framework(self):
        return SBCDecisionFramework()
    
    def test_should_run_sbc_high_complexity(self, decision_framework):
        """Test SBC requirement for high complexity models"""
        decision = decision_framework.should_run_sbc(
            model_complexity=15,  # > 10 parameters
            conflict_severity="low",
            business_impact=500000,
            client_confidence=0.8
        )
        
        assert decision["sbc_required"] is True
        assert "high model complexity" in decision["reason"].lower()
    
    def test_should_run_sbc_high_conflict(self, decision_framework):
        """Test SBC requirement for high conflict severity"""
        decision = decision_framework.should_run_sbc(
            model_complexity=5,
            conflict_severity="high",  # High conflict
            business_impact=500000,
            client_confidence=0.8
        )
        
        assert decision["sbc_required"] is True
        assert "high conflict" in decision["reason"].lower()
    
    def test_should_run_sbc_high_business_impact(self, decision_framework):
        """Test SBC requirement for high business impact"""
        decision = decision_framework.should_run_sbc(
            model_complexity=5,
            conflict_severity="low",
            business_impact=2000000,  # > $1M
            client_confidence=0.8
        )
        
        assert decision["sbc_required"] is True
        assert "business impact" in decision["reason"].lower()
    
    def test_should_run_sbc_low_confidence(self, decision_framework):
        """Test SBC requirement for low client confidence"""
        decision = decision_framework.should_run_sbc(
            model_complexity=5,
            conflict_severity="low",
            business_impact=500000,
            client_confidence=0.5  # < 0.6
        )
        
        assert decision["sbc_required"] is True
        assert "low client confidence" in decision["reason"].lower()
    
    def test_should_not_run_sbc(self, decision_framework):
        """Test when SBC is not required"""
        decision = decision_framework.should_run_sbc(
            model_complexity=3,
            conflict_severity="low",
            business_impact=100000,
            client_confidence=0.9
        )
        
        assert decision["sbc_required"] is False


class TestBayesianDatabaseManager:
    """Test database operations"""
    
    @pytest.fixture
    def db_manager(self):
        # Mock database for testing
        with patch('shared.database.bayesian_models.create_engine'):
            return BayesianDatabaseManager("sqlite:///:memory:")
    
    @pytest.mark.asyncio
    async def test_create_session(self, db_manager):
        """Test Bayesian session creation"""
        with patch.object(db_manager, 'execute_query', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {"session_id": "test-session-id"}
            
            session_id = await db_manager.create_session(
                user_id="test_user",
                session_name="Test Session",
                analysis_type="marketing_attribution"
            )
            
            assert session_id == "test-session-id"
            mock_execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_store_prior_beliefs(self, db_manager):
        """Test storing prior beliefs"""
        with patch.object(db_manager, 'execute_query', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {"success": True}
            
            prior_beliefs = [
                PriorBelief(
                    parameter_name="test_param",
                    parameter_category="channel_attribution",
                    distribution_type="beta",
                    distribution_params={"alpha": 2, "beta": 5},
                    confidence_level=0.8
                )
            ]
            
            result = await db_manager.store_prior_beliefs(
                session_id="test-session",
                prior_beliefs=prior_beliefs
            )
            
            assert result["success"] is True
            mock_execute.assert_called_once()
        with patch.object(db_manager, 'execute_query', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {"success": True}
            
            prior_beliefs = [
                PriorBelief(
                    parameter_name="test_param",
                    parameter_category="channel_attribution",
                    distribution_type="beta",
                    distribution_params={"alpha": 2, "beta": 5},
                    confidence_level=0.8
                )
            ]
            
            result = await db_manager.store_prior_beliefs(
                session_id="test-session",
                prior_beliefs=prior_beliefs
            )
            
            assert result["success"] is True
            mock_execute.assert_called_once()


class TestBayesianIntegration:
    """Integration tests for complete Bayesian workflow"""
    
    @pytest.mark.asyncio
    async def test_complete_bayesian_workflow(self):
        """Test end-to-end Bayesian analysis workflow"""
        # 1. Setup components
        conflict_analyzer = ConflictAnalyzer()
        prior_updater = PriorUpdater()
        sbc_validator = SBCValidator()
        decision_framework = SBCDecisionFramework()
        
        # 2. Define prior beliefs and observed data
        prior_beliefs = {
            "google_ads_attribution": {
                "distribution": "beta",
                "params": {"alpha": 3, "beta": 7},
                "confidence": 0.8
            }
        }
        
        observed_data = {
            "attribution_scores": {"google_ads_attribution": 0.45},
            "sample_size": 1000,
            "confidence_intervals": {"google_ads_attribution": [0.40, 0.50]}
        }
        
        # 3. Analyze conflict
        conflict_result = await conflict_analyzer.analyze_conflict(
            prior_beliefs=prior_beliefs,
            observed_data=observed_data,
            user_id="test_user"
        )
        
        assert conflict_result["conflict_detected"] is True
        
        # 4. Check SBC necessity
        sbc_decision = decision_framework.should_run_sbc(
            model_complexity=1,
            conflict_severity=conflict_result["conflict_severity"],
            business_impact=1500000,  # High impact
            client_confidence=0.7
        )
        
        # Should require SBC due to high business impact
        assert sbc_decision["sbc_required"] is True
        
        # 5. Generate update recommendations
        recommendations = await prior_updater.generate_recommendations(
            conflict_result=conflict_result,
            prior_beliefs=prior_beliefs,
            observed_data=observed_data
        )
        
        assert len(recommendations) > 0
        assert recommendations[0]["parameter_name"] == "google_ads_attribution"
        
        # 6. Run SBC validation if required
        if sbc_decision["sbc_required"]:
            sbc_result = await sbc_validator.run_sbc_validation(
                model_parameters=prior_beliefs,
                num_simulations=50,  # Reduced for testing
                validation_type="marketing_attribution"
            )
            
            assert "validation_passed" in sbc_result
            assert "rank_statistics" in sbc_result


class TestBayesianAPIIntegration:
    """Test API integration with Bayesian framework"""
    
    @pytest.mark.asyncio
    async def test_causal_service_bayesian_integration(self):
        """Test Causal Service integration with Bayesian analysis"""
        # Mock the Bayesian service calls
        with patch('modules.causal.app.call_bayesian_service', new_callable=AsyncMock) as mock_service:
            mock_service.return_value = {
                "conflict_detected": True,
                "conflict_severity": "moderate",
                "evidence_strength": "strong",
                "bayes_factor": 12.5,
                "recommendations": ["Update Google Ads attribution prior"]
            }
            
            # Import and test the function
            from modules.causal.app import analyze_prior_data_conflict
            
            result = await analyze_prior_data_conflict(
                user_id="test_user",
                prior_beliefs={"google_ads": {"mean": 0.3, "std": 0.1}},
                observed_data={"google_ads": 0.45}
            )
            
            assert result["conflict_detected"] is True
            assert result["bayes_factor"] == 12.5
            mock_service.assert_called_once()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])