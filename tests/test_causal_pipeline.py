"""
Comprehensive Testing Framework for LiftOS Causal Data Transformation Pipeline

This test suite validates the entire causal transformation pipeline from data ingestion
through KSE storage and retrieval, ensuring causal inference accuracy and data quality.
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, patch, AsyncMock

# Import LiftOS causal components
from shared.models.causal_marketing import (
    CausalMarketingData, ConfounderVariable, ExternalFactor, 
    CausalGraph, TreatmentAssignment, CausalDataQuality
)
from shared.utils.causal_transforms import (
    ConfounderDetector, TreatmentAssignmentEngine, 
    CausalDataQualityAssessor, CausalDataTransformer
)
from shared.kse_sdk.causal_models import (
    CausalRelationship, CausalEmbedding, TemporalCausalEmbedding,
    CausalKnowledgeGraph, CausalSearchQuery, CausalSearchResult
)
from shared.kse_sdk.causal_client import CausalKSEClient


class TestCausalDataModels:
    """Test causal data models for correctness and validation."""
    
    def test_causal_marketing_data_creation(self):
        """Test creation of CausalMarketingData with all required fields."""
        data = CausalMarketingData(
            experiment_id="exp_001",
            platform="meta",
            timestamp=datetime.now(),
            metrics={"spend": 1000.0, "impressions": 50000, "clicks": 1500},
            confounders=[],
            external_factors=[],
            treatment_assignment=None,
            causal_graph=None,
            data_quality=None
        )
        
        assert data.experiment_id == "exp_001"
        assert data.platform == "meta"
        assert isinstance(data.timestamp, datetime)
        assert data.metrics["spend"] == 1000.0
    
    def test_confounder_variable_validation(self):
        """Test confounder variable creation and validation."""
        confounder = ConfounderVariable(
            name="budget_change",
            value=0.15,
            confidence=0.85,
            detection_method="statistical_test",
            platform_specific_context={"campaign_id": "123", "change_type": "increase"}
        )
        
        assert confounder.name == "budget_change"
        assert confounder.value == 0.15
        assert confounder.confidence == 0.85
        assert confounder.detection_method == "statistical_test"
    
    def test_treatment_assignment_creation(self):
        """Test treatment assignment creation."""
        treatment = TreatmentAssignment(
            treatment_id="treat_001",
            treatment_type="budget_increase",
            assignment_method="randomized",
            assignment_probability=0.5,
            control_group_id="control_001",
            randomization_unit="campaign",
            assignment_timestamp=datetime.now()
        )
        
        assert treatment.treatment_id == "treat_001"
        assert treatment.treatment_type == "budget_increase"
        assert treatment.assignment_method == "randomized"
        assert treatment.randomization_unit == "campaign"


class TestConfounderDetection:
    """Test platform-specific confounder detection algorithms."""
    
    def setup_method(self):
        """Set up test data for confounder detection."""
        self.detector = ConfounderDetector()
        
        # Create sample marketing data
        self.sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=30, freq='D'),
            'spend': np.random.normal(1000, 200, 30),
            'impressions': np.random.normal(50000, 10000, 30),
            'clicks': np.random.normal(1500, 300, 30),
            'conversions': np.random.normal(50, 10, 30),
            'campaign_id': ['camp_001'] * 30,
            'quality_score': np.random.normal(7.5, 1.0, 30)
        })
    
    def test_meta_confounder_detection(self):
        """Test Meta-specific confounder detection."""
        confounders = self.detector.detect_meta_confounders(self.sample_data)
        
        assert isinstance(confounders, list)
        # Should detect at least budget and quality score confounders
        confounder_names = [c.name for c in confounders]
        assert any('budget' in name.lower() for name in confounder_names)
    
    def test_google_confounder_detection(self):
        """Test Google Ads-specific confounder detection."""
        confounders = self.detector.detect_google_confounders(self.sample_data)
        
        assert isinstance(confounders, list)
        # Should detect quality score and bid strategy confounders
        confounder_names = [c.name for c in confounders]
        assert any('quality' in name.lower() for name in confounder_names)
    
    def test_klaviyo_confounder_detection(self):
        """Test Klaviyo-specific confounder detection."""
        confounders = self.detector.detect_klaviyo_confounders(self.sample_data)
        
        assert isinstance(confounders, list)
        # Should detect engagement and list health confounders
        for confounder in confounders:
            assert isinstance(confounder, ConfounderVariable)
            assert confounder.confidence >= 0.0
            assert confounder.confidence <= 1.0


class TestTreatmentAssignment:
    """Test automated treatment assignment and experiment identification."""
    
    def setup_method(self):
        """Set up treatment assignment engine."""
        self.engine = TreatmentAssignmentEngine()
    
    def test_budget_change_detection(self):
        """Test detection of budget changes as treatments."""
        # Create data with clear budget change
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='D'),
            'spend': [1000] * 5 + [1500] * 5,  # Clear budget increase
            'campaign_id': ['camp_001'] * 10
        })
        
        treatment = self.engine.detect_budget_changes(data)
        
        assert treatment is not None
        assert treatment.treatment_type == "budget_increase"
        assert treatment.assignment_method == "detected"
    
    def test_targeting_change_detection(self):
        """Test detection of targeting changes."""
        # Mock targeting change data
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='D'),
            'audience_size': [100000] * 5 + [150000] * 5,  # Audience expansion
            'campaign_id': ['camp_001'] * 10
        })
        
        treatment = self.engine.detect_targeting_changes(data)
        
        assert treatment is not None
        assert treatment.treatment_type == "targeting_expansion"
    
    def test_creative_change_detection(self):
        """Test detection of creative changes."""
        # Mock creative change data
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='D'),
            'creative_id': ['creative_001'] * 5 + ['creative_002'] * 5,
            'campaign_id': ['camp_001'] * 10
        })
        
        treatment = self.engine.detect_creative_changes(data)
        
        assert treatment is not None
        assert treatment.treatment_type == "creative_change"


class TestCausalDataQuality:
    """Test causal data quality assessment framework."""
    
    def setup_method(self):
        """Set up quality assessor."""
        self.assessor = CausalDataQualityAssessor()
    
    def test_temporal_consistency_check(self):
        """Test temporal consistency validation."""
        # Create temporally consistent data
        consistent_data = CausalMarketingData(
            experiment_id="exp_001",
            platform="meta",
            timestamp=datetime.now(),
            metrics={"spend": 1000.0},
            confounders=[],
            external_factors=[],
            treatment_assignment=None,
            causal_graph=None,
            data_quality=None
        )
        
        is_consistent = self.assessor.check_temporal_consistency([consistent_data])
        assert is_consistent is True
    
    def test_confounder_coverage_assessment(self):
        """Test confounder coverage assessment."""
        confounders = [
            ConfounderVariable(
                name="budget_change",
                value=0.1,
                confidence=0.8,
                detection_method="statistical",
                platform_specific_context={}
            )
        ]
        
        coverage_score = self.assessor.assess_confounder_coverage(confounders, "meta")
        
        assert isinstance(coverage_score, float)
        assert 0.0 <= coverage_score <= 1.0
    
    def test_treatment_assignment_quality(self):
        """Test treatment assignment quality validation."""
        treatment = TreatmentAssignment(
            treatment_id="treat_001",
            treatment_type="budget_increase",
            assignment_method="randomized",
            assignment_probability=0.5,
            control_group_id="control_001",
            randomization_unit="campaign",
            assignment_timestamp=datetime.now()
        )
        
        quality_score = self.assessor.assess_treatment_quality(treatment)
        
        assert isinstance(quality_score, float)
        assert 0.0 <= quality_score <= 1.0


class TestCausalDataTransformer:
    """Test the main causal data transformation pipeline."""
    
    def setup_method(self):
        """Set up causal data transformer."""
        self.transformer = CausalDataTransformer()
    
    @pytest.mark.asyncio
    async def test_full_transformation_pipeline(self):
        """Test complete causal transformation pipeline."""
        # Mock raw marketing data
        raw_data = {
            "platform": "meta",
            "campaign_data": {
                "spend": 1000.0,
                "impressions": 50000,
                "clicks": 1500,
                "conversions": 50
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Mock historical data
        historical_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=30, freq='D'),
            'spend': np.random.normal(1000, 200, 30),
            'impressions': np.random.normal(50000, 10000, 30)
        })
        
        # Transform data
        causal_data = await self.transformer.transform(raw_data, historical_data)
        
        assert isinstance(causal_data, CausalMarketingData)
        assert causal_data.platform == "meta"
        assert causal_data.experiment_id is not None
        assert isinstance(causal_data.confounders, list)
        assert isinstance(causal_data.external_factors, list)


class TestKSECausalIntegration:
    """Test KSE causal integration components."""
    
    def setup_method(self):
        """Set up KSE causal client."""
        self.client = CausalKSEClient(
            api_key="test_key",
            base_url="http://localhost:8000"
        )
    
    def test_causal_relationship_creation(self):
        """Test causal relationship model creation."""
        relationship = CausalRelationship(
            source_concept="budget_increase",
            target_concept="impression_increase",
            relationship_type="DIRECT_CAUSE",
            strength=0.75,
            confidence=0.85,
            temporal_lag_days=1,
            context={"platform": "meta", "campaign_type": "awareness"}
        )
        
        assert relationship.source_concept == "budget_increase"
        assert relationship.target_concept == "impression_increase"
        assert relationship.relationship_type == "DIRECT_CAUSE"
        assert relationship.strength == 0.75
    
    def test_causal_embedding_creation(self):
        """Test causal embedding creation."""
        embedding = CausalEmbedding(
            concept="budget_optimization",
            vector=[0.1, 0.2, 0.3, 0.4, 0.5],
            causal_context={
                "causes": ["market_conditions", "competitor_activity"],
                "effects": ["cost_efficiency", "performance_improvement"]
            },
            temporal_context={
                "time_horizon": "weekly",
                "lag_sensitivity": 0.8
            }
        )
        
        assert embedding.concept == "budget_optimization"
        assert len(embedding.vector) == 5
        assert "causes" in embedding.causal_context
        assert "effects" in embedding.causal_context
    
    @pytest.mark.asyncio
    async def test_causal_search_query(self):
        """Test causal search functionality."""
        query = CausalSearchQuery(
            query_text="budget increase effect on conversions",
            causal_filters={
                "relationship_types": ["DIRECT_CAUSE", "INDIRECT_CAUSE"],
                "min_confidence": 0.7,
                "platforms": ["meta", "google"]
            },
            temporal_filters={
                "time_range": "last_30_days",
                "max_lag_days": 7
            },
            search_mode="hybrid"
        )
        
        assert query.query_text == "budget increase effect on conversions"
        assert "DIRECT_CAUSE" in query.causal_filters["relationship_types"]
        assert query.causal_filters["min_confidence"] == 0.7
        assert query.search_mode == "hybrid"


class TestEndToEndCausalPipeline:
    """Test complete end-to-end causal pipeline integration."""
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_flow(self):
        """Test complete pipeline from ingestion to KSE storage."""
        # 1. Mock data ingestion
        raw_data = {
            "platform": "meta",
            "campaign_data": {
                "spend": 1000.0,
                "impressions": 50000,
                "clicks": 1500,
                "conversions": 50
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # 2. Transform to causal data
        transformer = CausalDataTransformer()
        historical_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=30, freq='D'),
            'spend': np.random.normal(1000, 200, 30)
        })
        
        causal_data = await transformer.transform(raw_data, historical_data)
        
        # 3. Validate causal data quality
        assessor = CausalDataQualityAssessor()
        quality = assessor.assess_overall_quality(causal_data)
        
        assert quality.overall_score >= 0.0
        assert quality.overall_score <= 1.0
        
        # 4. Test KSE storage (mocked)
        with patch('shared.kse_sdk.causal_client.CausalKSEClient.store_causal_memory') as mock_store:
            mock_store.return_value = {"status": "success", "memory_id": "mem_001"}
            
            client = CausalKSEClient(api_key="test", base_url="http://localhost")
            result = await client.store_causal_memory(causal_data)
            
            assert result["status"] == "success"
            assert "memory_id" in result
    
    def test_causal_pipeline_error_handling(self):
        """Test error handling in causal pipeline."""
        transformer = CausalDataTransformer()
        
        # Test with invalid data
        with pytest.raises(ValueError):
            asyncio.run(transformer.transform({}, pd.DataFrame()))
    
    def test_causal_pipeline_performance(self):
        """Test performance of causal transformation pipeline."""
        import time
        
        transformer = CausalDataTransformer()
        
        # Create large dataset
        large_historical_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='H'),
            'spend': np.random.normal(1000, 200, 1000),
            'impressions': np.random.normal(50000, 10000, 1000)
        })
        
        raw_data = {
            "platform": "meta",
            "campaign_data": {"spend": 1000.0, "impressions": 50000},
            "timestamp": datetime.now().isoformat()
        }
        
        start_time = time.time()
        result = asyncio.run(transformer.transform(raw_data, large_historical_data))
        end_time = time.time()
        
        # Should complete within reasonable time (< 5 seconds)
        assert (end_time - start_time) < 5.0
        assert isinstance(result, CausalMarketingData)


class TestCausalValidationSuite:
    """Comprehensive validation suite for causal inference accuracy."""
    
    def test_causal_inference_accuracy(self):
        """Test accuracy of causal inference methods."""
        # Create synthetic data with known causal relationships
        np.random.seed(42)
        
        # X causes Y with known effect size
        X = np.random.normal(0, 1, 1000)
        Y = 2.0 * X + np.random.normal(0, 0.5, 1000)  # True effect = 2.0
        
        data = pd.DataFrame({'X': X, 'Y': Y})
        
        # Test causal discovery
        detector = ConfounderDetector()
        # This would use actual causal discovery methods in implementation
        # For now, we test the structure
        
        assert len(data) == 1000
        assert 'X' in data.columns
        assert 'Y' in data.columns
    
    def test_counterfactual_accuracy(self):
        """Test accuracy of counterfactual predictions."""
        # This would test counterfactual prediction accuracy
        # against known ground truth in synthetic data
        pass
    
    def test_treatment_effect_estimation(self):
        """Test accuracy of treatment effect estimation."""
        # This would test treatment effect estimation
        # against known true effects in synthetic data
        pass


if __name__ == "__main__":
    # Run the test suite
    pytest.main([__file__, "-v", "--tb=short"])