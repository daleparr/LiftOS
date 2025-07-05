"""
Comprehensive test suite for Enhanced Observability Service
Tests time-series storage, metrics collection, health monitoring, and alerting
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

# Import test dependencies
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.database.observability_models import (
    MetricEntry, LogEntry, HealthCheckEntry, AlertEntry,
    ServiceRegistry, MetricAggregation, SystemSnapshot
)
from shared.utils.time_series_storage import (
    TimeSeriesStorage, MetricPoint, TimeSeriesQuery, AggregatedMetric
)
from shared.database.connection import DatabaseManager


class TestTimeSeriesStorage:
    """Test time-series storage functionality"""
    
    @pytest.fixture
    def storage(self):
        """Create test time-series storage"""
        db_manager = MagicMock()
        storage = TimeSeriesStorage(db_manager)
        storage.db_manager.get_session = AsyncMock()
        return storage
    
    @pytest.mark.asyncio
    async def test_store_metric(self, storage):
        """Test storing a single metric"""
        metric = MetricPoint(
            name="test_metric",
            value=42.5,
            timestamp=datetime.now(timezone.utc),
            labels={"service": "test", "environment": "dev"},
            service_name="test_service"
        )
        
        result = await storage.store_metric(metric)
        assert result is True
        assert len(storage.metric_buffer) == 1
    
    @pytest.mark.asyncio
    async def test_store_metrics_batch(self, storage):
        """Test batch metric storage"""
        metrics = [
            MetricPoint(
                name=f"test_metric_{i}",
                value=float(i),
                timestamp=datetime.now(timezone.utc),
                labels={"batch": "test"},
                service_name="test_service"
            )
            for i in range(10)
        ]
        
        # Mock database session
        mock_session = AsyncMock()
        storage.db_manager.get_session.return_value.__aenter__.return_value = mock_session
        
        result = await storage.store_metrics_batch(metrics)
        assert result is True
        mock_session.add_all.assert_called_once()
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query_metrics(self, storage):
        """Test metric querying with filters"""
        # Mock database response
        mock_session = AsyncMock()
        mock_metrics = [
            MagicMock(
                name="test_metric",
                value=42.5,
                timestamp=datetime.now(timezone.utc),
                labels={"service": "test"},
                organization_id=None,
                service_name="test_service"
            )
        ]
        mock_session.execute.return_value.scalars.return_value.all.return_value = mock_metrics
        storage.db_manager.get_session.return_value.__aenter__.return_value = mock_session
        
        query = TimeSeriesQuery(
            metric_name="test_metric",
            service_name="test_service",
            limit=100
        )
        
        results = await storage.query_metrics(query)
        assert len(results) == 1
        assert results[0]["name"] == "test_metric"
        assert results[0]["value"] == 42.5
    
    @pytest.mark.asyncio
    async def test_health_check_storage(self, storage):
        """Test health check result storage"""
        mock_session = AsyncMock()
        storage.db_manager.get_session.return_value.__aenter__.return_value = mock_session
        
        result = await storage.store_health_check(
            service_name="test_service",
            status="healthy",
            response_time_ms=150.5,
            details={"version": "1.0.0"},
            endpoint_url="http://test:8000/health"
        )
        
        assert result is True
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
    
    def test_hash_labels(self, storage):
        """Test label hashing for grouping"""
        labels1 = {"service": "test", "env": "prod"}
        labels2 = {"env": "prod", "service": "test"}  # Different order
        labels3 = {"service": "test", "env": "dev"}   # Different values
        
        hash1 = storage._hash_labels(labels1)
        hash2 = storage._hash_labels(labels2)
        hash3 = storage._hash_labels(labels3)
        
        assert hash1 == hash2  # Same labels, different order
        assert hash1 != hash3  # Different labels


class TestObservabilityAPI:
    """Test observability service API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from services.observability.app import app
        return httpx.AsyncClient(app=app, base_url="http://test")
    
    @pytest.fixture
    def auth_headers(self):
        """Mock authentication headers"""
        return {"Authorization": "Bearer test_token"}
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = await client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "observability"
        assert "timestamp" in data
    
    @pytest.mark.asyncio
    @patch('services.observability.app.verify_token')
    @patch('services.observability.app.time_series_storage')
    async def test_record_metric(self, mock_storage, mock_auth, client, auth_headers):
        """Test metric recording endpoint"""
        mock_auth.return_value = {"user_id": "test_user"}
        mock_storage.store_metric.return_value = True
        
        metric_data = {
            "name": "test_metric",
            "value": 42.5,
            "labels": {"service": "test"},
            "service_name": "test_service"
        }
        
        response = await client.post(
            "/api/v1/metrics",
            json=metric_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "recorded"
        assert "timestamp" in data
    
    @pytest.mark.asyncio
    @patch('services.observability.app.verify_token')
    @patch('services.observability.app.time_series_storage')
    async def test_record_metrics_batch(self, mock_storage, mock_auth, client, auth_headers):
        """Test batch metric recording"""
        mock_auth.return_value = {"user_id": "test_user"}
        mock_storage.store_metrics_batch.return_value = True
        
        batch_data = {
            "metrics": [
                {
                    "name": f"test_metric_{i}",
                    "value": float(i),
                    "labels": {"batch": "test"},
                    "service_name": "test_service"
                }
                for i in range(5)
            ]
        }
        
        response = await client.post(
            "/api/v1/metrics/batch",
            json=batch_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "recorded"
        assert data["count"] == 5
    
    @pytest.mark.asyncio
    @patch('services.observability.app.verify_token')
    @patch('services.observability.app.time_series_storage')
    async def test_query_metrics(self, mock_storage, mock_auth, client, auth_headers):
        """Test metric querying endpoint"""
        mock_auth.return_value = {"user_id": "test_user"}
        mock_storage.query_metrics.return_value = [
            {
                "name": "test_metric",
                "value": 42.5,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "labels": {"service": "test"},
                "service_name": "test_service"
            }
        ]
        
        query_data = {
            "metric_name": "test_metric",
            "service_name": "test_service",
            "limit": 100
        }
        
        response = await client.post(
            "/api/v1/metrics/query",
            json=query_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        assert data["count"] == 1
        assert len(data["metrics"]) == 1
    
    @pytest.mark.asyncio
    @patch('services.observability.app.verify_token')
    @patch('services.observability.app.db_manager')
    async def test_register_service(self, mock_db, mock_auth, client, auth_headers):
        """Test service registration endpoint"""
        mock_auth.return_value = {"user_id": "test_user"}
        
        # Mock database session
        mock_session = AsyncMock()
        mock_session.execute.return_value.scalar.return_value = None  # No existing service
        mock_db.get_session.return_value.__aenter__.return_value = mock_session
        
        service_data = {
            "service_name": "new_service",
            "health_endpoint": "http://new_service:8000/health",
            "description": "Test service",
            "check_interval_seconds": 30,
            "timeout_seconds": 10
        }
        
        response = await client.post(
            "/api/v1/services/register",
            json=service_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "registered"
        assert data["service_name"] == "new_service"
    
    @pytest.mark.asyncio
    @patch('services.observability.app.verify_token')
    @patch('services.observability.app.db_manager')
    async def test_create_alert(self, mock_db, mock_auth, client, auth_headers):
        """Test alert creation endpoint"""
        mock_auth.return_value = {"user_id": "test_user"}
        
        # Mock database session
        mock_session = AsyncMock()
        mock_db.get_session.return_value.__aenter__.return_value = mock_session
        
        alert_data = {
            "name": "Test Alert",
            "description": "This is a test alert",
            "severity": "warning",
            "service_name": "test_service",
            "metadata": {"test": True}
        }
        
        response = await client.post(
            "/api/v1/alerts",
            json=alert_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "created"
        assert "alert_id" in data
        assert "timestamp" in data


class TestHealthMonitoring:
    """Test health monitoring functionality"""
    
    @pytest.fixture
    def health_monitor(self):
        """Create test health monitor"""
        from services.observability.app import HealthMonitor
        return HealthMonitor()
    
    @pytest.mark.asyncio
    @patch('services.observability.app.db_manager')
    @patch('services.observability.app.time_series_storage')
    async def test_check_service_health_success(self, mock_storage, mock_db, health_monitor):
        """Test successful health check"""
        # Mock service registry
        mock_service = MagicMock()
        mock_service.service_name = "test_service"
        mock_service.health_endpoint = "http://test:8000/health"
        mock_service.timeout_seconds = 10
        mock_service.alert_on_failure = True
        mock_service.failure_threshold = 3
        
        # Mock HTTP response
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "healthy"}
            mock_response.content = b'{"status": "healthy"}'
            
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            # Mock storage
            mock_storage.store_health_check.return_value = True
            mock_storage.store_metric.return_value = True
            
            await health_monitor._check_service_health(mock_service)
            
            # Verify health check was stored
            mock_storage.store_health_check.assert_called_once()
            mock_storage.store_metric.assert_called_once()
            
            # Verify failure count was reset
            assert health_monitor.failure_counts[mock_service.service_name] == 0
    
    @pytest.mark.asyncio
    @patch('services.observability.app.db_manager')
    @patch('services.observability.app.time_series_storage')
    async def test_check_service_health_failure(self, mock_storage, mock_db, health_monitor):
        """Test failed health check"""
        # Mock service registry
        mock_service = MagicMock()
        mock_service.service_name = "test_service"
        mock_service.health_endpoint = "http://test:8000/health"
        mock_service.timeout_seconds = 10
        mock_service.alert_on_failure = True
        mock_service.failure_threshold = 3
        
        # Mock HTTP response (failure)
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 500
            
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            # Mock storage
            mock_storage.store_health_check.return_value = True
            mock_storage.store_metric.return_value = True
            
            await health_monitor._check_service_health(mock_service)
            
            # Verify failure count was incremented
            assert health_monitor.failure_counts[mock_service.service_name] == 1
    
    @pytest.mark.asyncio
    @patch('services.observability.app.db_manager')
    async def test_create_health_alert(self, mock_db, health_monitor):
        """Test health alert creation"""
        # Mock service registry
        mock_service = MagicMock()
        mock_service.service_name = "test_service"
        mock_service.failure_threshold = 3
        
        # Set failure count to threshold
        health_monitor.failure_counts[mock_service.service_name] = 3
        
        # Mock database session
        mock_session = AsyncMock()
        mock_session.execute.return_value.scalar.return_value = None  # No existing alert
        mock_db.get_session.return_value.__aenter__.return_value = mock_session
        
        await health_monitor._create_health_alert(mock_service, {"error": "Connection failed"})
        
        # Verify alert was created
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()


class TestMetricsAggregation:
    """Test metrics aggregation functionality"""
    
    @pytest.mark.asyncio
    async def test_compute_real_time_aggregation(self):
        """Test real-time metric aggregation"""
        db_manager = MagicMock()
        storage = TimeSeriesStorage(db_manager)
        
        # Mock database session and query result
        mock_session = AsyncMock()
        mock_session.execute.return_value.scalar.return_value = 42.5
        storage.db_manager.get_session.return_value.__aenter__.return_value = mock_session
        
        result = await storage.compute_real_time_aggregation(
            metric_name="test_metric",
            aggregation_type="avg",
            start_time=datetime.now(timezone.utc) - timedelta(hours=1),
            end_time=datetime.now(timezone.utc),
            service_name="test_service"
        )
        
        assert result == 42.5
        mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_aggregated_metrics(self):
        """Test retrieving pre-computed aggregations"""
        db_manager = MagicMock()
        storage = TimeSeriesStorage(db_manager)
        
        # Mock aggregation data
        mock_agg = MagicMock()
        mock_agg.metric_name = "test_metric"
        mock_agg.aggregation_type = "avg"
        mock_agg.time_window = "1h"
        mock_agg.timestamp = datetime.now(timezone.utc)
        mock_agg.value = 42.5
        mock_agg.sample_count = 100
        mock_agg.labels = {"service": "test"}
        
        mock_session = AsyncMock()
        mock_session.execute.return_value.scalars.return_value.all.return_value = [mock_agg]
        storage.db_manager.get_session.return_value.__aenter__.return_value = mock_session
        
        results = await storage.get_aggregated_metrics(
            metric_name="test_metric",
            aggregation_type="avg",
            time_window="1h",
            start_time=datetime.now(timezone.utc) - timedelta(hours=24),
            end_time=datetime.now(timezone.utc)
        )
        
        assert len(results) == 1
        assert results[0].metric_name == "test_metric"
        assert results[0].value == 42.5


class TestSystemSnapshots:
    """Test system snapshot functionality"""
    
    @pytest.mark.asyncio
    async def test_create_system_snapshot(self):
        """Test system snapshot creation"""
        db_manager = MagicMock()
        storage = TimeSeriesStorage(db_manager)
        
        # Mock database session and query results
        mock_session = AsyncMock()
        
        # Mock health status counts
        mock_session.execute.side_effect = [
            # Health status counts
            MagicMock(fetchall=MagicMock(return_value=[("healthy", 5), ("unhealthy", 1)])),
            # Alert counts
            MagicMock(fetchall=MagicMock(return_value=[("critical", 2), ("warning", 3)])),
            # Average response time
            MagicMock(scalar=MagicMock(return_value=150.5))
        ]
        
        storage.db_manager.get_session.return_value.__aenter__.return_value = mock_session
        
        result = await storage.create_system_snapshot()
        
        assert result is True
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()


class TestIntegrationScenarios:
    """Test end-to-end integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_full_monitoring_cycle(self):
        """Test complete monitoring cycle: register service -> health check -> metrics -> alerts"""
        # This would be a comprehensive integration test
        # that exercises the full monitoring workflow
        pass
    
    @pytest.mark.asyncio
    async def test_high_volume_metrics(self):
        """Test handling high volume of metrics"""
        db_manager = MagicMock()
        storage = TimeSeriesStorage(db_manager)
        
        # Mock successful batch storage
        storage.db_manager.get_session = AsyncMock()
        mock_session = AsyncMock()
        storage.db_manager.get_session.return_value.__aenter__.return_value = mock_session
        
        # Generate large batch of metrics
        metrics = [
            MetricPoint(
                name=f"high_volume_metric_{i % 10}",
                value=float(i),
                timestamp=datetime.now(timezone.utc),
                labels={"batch": str(i // 100)},
                service_name="load_test_service"
            )
            for i in range(1000)
        ]
        
        result = await storage.store_metrics_batch(metrics)
        assert result is True
        mock_session.add_all.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_alert_escalation_scenario(self):
        """Test alert escalation when service fails repeatedly"""
        from services.observability.app import HealthMonitor
        
        health_monitor = HealthMonitor()
        
        # Mock service that will fail
        mock_service = MagicMock()
        mock_service.service_name = "failing_service"
        mock_service.health_endpoint = "http://failing:8000/health"
        mock_service.timeout_seconds = 10
        mock_service.alert_on_failure = True
        mock_service.failure_threshold = 3
        
        # Simulate multiple failures
        with patch('httpx.AsyncClient') as mock_client, \
             patch('services.observability.app.time_series_storage') as mock_storage, \
             patch('services.observability.app.db_manager') as mock_db:
            
            # Mock failed HTTP response
            mock_client.return_value.__aenter__.return_value.get.side_effect = Exception("Connection failed")
            mock_storage.store_health_check.return_value = True
            
            # Mock database for alert creation
            mock_session = AsyncMock()
            mock_session.execute.return_value.scalar.return_value = None
            mock_db.get_session.return_value.__aenter__.return_value = mock_session
            
            # Simulate 3 consecutive failures
            for _ in range(3):
                await health_monitor._check_service_health(mock_service)
            
            # Verify failure count reached threshold
            assert health_monitor.failure_counts[mock_service.service_name] == 3
            
            # Verify alert was created on the third failure
            mock_session.add.assert_called()
            mock_session.commit.assert_called()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])