"""
Test suite for data quality engine integration with the Agentic microservice.

This module tests the complete data quality evaluation framework including:
- API endpoint functionality
- Data quality assessment accuracy
- Agent testing validation
- Error handling and edge cases
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

# Import the FastAPI app and data quality components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from core.data_quality_engine import DataQualityEngine, QualityLevel, QualityDimension


class TestDataQualityIntegration:
    """Test suite for data quality engine integration."""
    
    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_high_quality_data(self):
        """Sample data with high quality for testing."""
        return {
            "customer_id": [1, 2, 3, 4, 5],
            "name": ["Alice Johnson", "Bob Smith", "Carol Davis", "David Wilson", "Eva Brown"],
            "email": ["alice@email.com", "bob@email.com", "carol@email.com", "david@email.com", "eva@email.com"],
            "age": [25, 30, 35, 28, 32],
            "purchase_amount": [100.50, 250.75, 175.25, 320.00, 89.99],
            "purchase_date": ["2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18", "2024-01-19"]
        }
    
    @pytest.fixture
    def sample_poor_quality_data(self):
        """Sample data with poor quality for testing."""
        return {
            "customer_id": [1, 2, None, 4, 2],  # Missing value and duplicate
            "name": ["Alice", "", "Carol Davis", "David Wilson", None],  # Empty and missing values
            "email": ["alice@email", "invalid-email", "carol@email.com", "david@email.com", "eva@email.com"],  # Invalid formats
            "age": [25, -5, 150, 28, "thirty"],  # Invalid values
            "purchase_amount": [100.50, None, 175.25, -50.00, 89.99],  # Missing and negative values
            "purchase_date": ["2024-01-15", "invalid-date", "2024-01-17", "2020-01-01", "2024-01-19"]  # Invalid and old dates
        }
    
    def test_data_quality_evaluate_endpoint_success(self, client, sample_high_quality_data):
        """Test successful data quality evaluation via API endpoint."""
        request_data = {
            "data": sample_high_quality_data,
            "dataset_id": "test_dataset_001",
            "test_type": "general",
            "include_profiling": True
        }
        
        response = client.post("/data-quality/evaluate", json=request_data)
        
        assert response.status_code == 200
        result = response.json()
        
        # Verify response structure
        assert "dataset_id" in result
        assert "evaluation_timestamp" in result
        assert "overall_score" in result
        assert "overall_level" in result
        assert "dimension_scores" in result
        assert "critical_issues" in result
        assert "recommendations" in result
        assert "data_profile" in result
        
        # Verify high quality data gets good scores
        assert result["overall_score"] >= 0.85
        assert result["overall_level"] in ["EXCELLENT", "GOOD"]
        assert len(result["critical_issues"]) == 0
        
        # Verify all 8 dimensions are evaluated
        expected_dimensions = [
            "COMPLETENESS", "ACCURACY", "CONSISTENCY", "VALIDITY",
            "UNIQUENESS", "TIMELINESS", "RELEVANCE", "INTEGRITY"
        ]
        for dimension in expected_dimensions:
            assert dimension in result["dimension_scores"]
            assert "score" in result["dimension_scores"][dimension]
            assert "level" in result["dimension_scores"][dimension]
    
    def test_data_quality_evaluate_endpoint_poor_data(self, client, sample_poor_quality_data):
        """Test data quality evaluation with poor quality data."""
        request_data = {
            "data": sample_poor_quality_data,
            "dataset_id": "test_dataset_poor",
            "test_type": "general",
            "include_profiling": True
        }
        
        response = client.post("/data-quality/evaluate", json=request_data)
        
        assert response.status_code == 200
        result = response.json()
        
        # Verify poor quality data gets low scores
        assert result["overall_score"] < 0.70
        assert result["overall_level"] in ["POOR", "CRITICAL"]
        assert len(result["critical_issues"]) > 0
        assert len(result["recommendations"]) > 0
    
    def test_validate_for_agent_testing_endpoint_success(self, client, sample_high_quality_data):
        """Test agent testing validation with high quality data."""
        request_data = {
            "data": sample_high_quality_data,
            "test_type": "marketing_campaign"
        }
        
        response = client.post("/data-quality/validate-for-testing", json=request_data)
        
        assert response.status_code == 200
        result = response.json()
        
        # Verify response structure
        assert "is_valid" in result
        assert "validation_result" in result
        assert "overall_score" in result
        assert "overall_level" in result
        assert "critical_issues" in result
        assert "recommendations" in result
        assert "quality_summary" in result
        
        # High quality data should pass validation
        assert result["is_valid"] is True
        assert result["validation_result"] == "PASS"
        assert result["overall_score"] >= 0.85
    
    def test_validate_for_agent_testing_endpoint_failure(self, client, sample_poor_quality_data):
        """Test agent testing validation with poor quality data."""
        request_data = {
            "data": sample_poor_quality_data,
            "test_type": "marketing_campaign"
        }
        
        response = client.post("/data-quality/validate-for-testing", json=request_data)
        
        assert response.status_code == 200
        result = response.json()
        
        # Poor quality data should fail validation
        assert result["is_valid"] is False
        assert result["validation_result"] == "FAIL"
        assert result["overall_score"] < 0.85
        assert len(result["critical_issues"]) > 0
    
    def test_get_quality_dimensions_endpoint(self, client):
        """Test quality dimensions information endpoint."""
        response = client.get("/data-quality/dimensions")
        
        assert response.status_code == 200
        result = response.json()
        
        # Verify response structure
        assert "dimensions" in result
        assert "quality_levels" in result
        assert "validation_thresholds" in result
        
        # Verify all 8 dimensions are documented
        expected_dimensions = [
            "completeness", "accuracy", "consistency", "validity",
            "uniqueness", "timeliness", "relevance", "integrity"
        ]
        for dimension in expected_dimensions:
            assert dimension in result["dimensions"]
            assert "weight" in result["dimensions"][dimension]
            assert "description" in result["dimensions"][dimension]
            assert "importance" in result["dimensions"][dimension]
        
        # Verify quality levels
        expected_levels = ["excellent", "good", "acceptable", "poor", "critical"]
        for level in expected_levels:
            assert level in result["quality_levels"]
            assert "range" in result["quality_levels"][level]
            assert "description" in result["quality_levels"][level]
    
    def test_data_quality_health_check_endpoint(self, client):
        """Test data quality engine health check."""
        response = client.get("/data-quality/health")
        
        assert response.status_code == 200
        result = response.json()
        
        # Verify response structure
        assert "status" in result
        assert "engine_initialized" in result
        assert "timestamp" in result
        
        # Health check should pass
        assert result["status"] == "healthy"
        assert result["engine_initialized"] is True
        assert "test_evaluation_successful" in result
        assert result["test_evaluation_successful"] is True
    
    def test_data_quality_endpoint_error_handling(self, client):
        """Test error handling in data quality endpoints."""
        # Test with invalid data structure
        invalid_request = {
            "data": "invalid_data_format",
            "dataset_id": "test_invalid"
        }
        
        response = client.post("/data-quality/evaluate", json=invalid_request)
        assert response.status_code == 500
        
        # Test with missing required fields
        incomplete_request = {
            "data": {"test": [1, 2, 3]}
            # Missing dataset_id - this should cause validation error
        }
        
        response = client.post("/data-quality/evaluate", json=incomplete_request)
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_data_quality_engine_direct_integration(self, sample_high_quality_data):
        """Test direct integration with data quality engine."""
        engine = DataQualityEngine()
        
        # Test data quality evaluation
        report = await engine.evaluate_data_quality(
            data=sample_high_quality_data,
            dataset_id="direct_test_001",
            include_profiling=True
        )
        
        # Verify report structure
        assert report.dataset_id == "direct_test_001"
        assert isinstance(report.evaluation_timestamp, datetime)
        assert 0.0 <= report.overall_score <= 1.0
        assert isinstance(report.overall_level, QualityLevel)
        assert len(report.dimension_scores) == 8
        assert isinstance(report.critical_issues, list)
        assert isinstance(report.recommendations, list)
        
        # Test agent testing validation
        is_valid, validation_report = await engine.validate_data_for_agent_testing(
            data=sample_high_quality_data,
            test_type="marketing_campaign"
        )
        
        assert isinstance(is_valid, bool)
        assert validation_report.dataset_id is not None
    
    @pytest.mark.asyncio
    async def test_data_quality_with_pandas_dataframe(self):
        """Test data quality evaluation with pandas DataFrame input."""
        engine = DataQualityEngine()
        
        # Create DataFrame with mixed quality data
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "value": [10.5, 20.0, None, 40.5, 50.0],  # Missing value
            "category": ["A", "B", "A", "C", "B"],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
        })
        
        report = await engine.evaluate_data_quality(
            data=df,
            dataset_id="pandas_test_001",
            include_profiling=True
        )
        
        # Verify DataFrame handling
        assert report.dataset_id == "pandas_test_001"
        assert report.overall_score > 0.0
        assert QualityDimension.COMPLETENESS in report.dimension_scores
        
        # Should detect missing value in completeness dimension
        completeness_score = report.dimension_scores[QualityDimension.COMPLETENESS].score
        assert completeness_score < 1.0  # Should be less than perfect due to missing value
    
    def test_data_quality_performance_benchmark(self, client):
        """Test data quality evaluation performance with larger datasets."""
        # Generate larger dataset for performance testing
        large_data = {
            "id": list(range(1000)),
            "value": np.random.normal(100, 15, 1000).tolist(),
            "category": np.random.choice(["A", "B", "C", "D"], 1000).tolist(),
            "timestamp": pd.date_range("2024-01-01", periods=1000, freq="H").strftime("%Y-%m-%d %H:%M:%S").tolist()
        }
        
        request_data = {
            "data": large_data,
            "dataset_id": "performance_test_001",
            "include_profiling": True
        }
        
        import time
        start_time = time.time()
        
        response = client.post("/data-quality/evaluate", json=request_data)
        
        end_time = time.time()
        evaluation_time = end_time - start_time
        
        assert response.status_code == 200
        assert evaluation_time < 10.0  # Should complete within 10 seconds
        
        result = response.json()
        assert result["overall_score"] >= 0.0
        assert "data_profile" in result
    
    def test_data_quality_edge_cases(self, client):
        """Test data quality evaluation with edge cases."""
        # Test with empty dataset
        empty_data = {"column1": [], "column2": []}
        request_data = {
            "data": empty_data,
            "dataset_id": "empty_test_001"
        }
        
        response = client.post("/data-quality/evaluate", json=request_data)
        assert response.status_code == 200
        result = response.json()
        assert result["overall_score"] >= 0.0
        
        # Test with single row
        single_row_data = {"column1": [1], "column2": ["value"]}
        request_data = {
            "data": single_row_data,
            "dataset_id": "single_row_test_001"
        }
        
        response = client.post("/data-quality/evaluate", json=request_data)
        assert response.status_code == 200
        result = response.json()
        assert result["overall_score"] >= 0.0
        
        # Test with all null values
        null_data = {"column1": [None, None, None], "column2": [None, None, None]}
        request_data = {
            "data": null_data,
            "dataset_id": "null_test_001"
        }
        
        response = client.post("/data-quality/evaluate", json=request_data)
        assert response.status_code == 200
        result = response.json()
        assert result["overall_level"] in ["POOR", "CRITICAL"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])