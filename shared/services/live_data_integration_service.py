"""
Live Data Integration Service
Manages the transition from mock to live data and handles hybrid data scenarios
"""

import asyncio
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import json

from shared.models.marketing import DataSource
from shared.services.platform_connection_service import get_platform_connection_service
from shared.services.data_source_validator import get_data_source_validator
from shared.connectors.connector_factory import get_connector_manager
from shared.utils.mock_data_generator import MockDataGenerator
from shared.utils.logging import setup_logging

logger = setup_logging("live_data_integration")

class DataMode(Enum):
    MOCK_ONLY = "mock_only"
    LIVE_ONLY = "live_only"
    HYBRID = "hybrid"
    AUTO = "auto"

class DataSourceStatus(Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"
    TESTING = "testing"

@dataclass
class DataSourceHealth:
    """Health status of a data source"""
    source: str
    status: DataSourceStatus
    quality_score: Optional[float] = None
    last_successful_sync: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[str] = None
    response_time_ms: Optional[int] = None

@dataclass
class IntegrationTestResult:
    """Result of live data integration test"""
    connection_id: str
    platform: str
    test_type: str
    success: bool
    data_retrieved: bool
    record_count: int
    quality_score: float
    response_time_ms: int
    error_message: Optional[str] = None
    recommendations: List[str] = None

class LiveDataIntegrationService:
    """Service for managing live data integration and testing"""
    
    def __init__(self):
        self.platform_service = None
        self.validator_service = None
        self.connector_manager = get_connector_manager()
        self.mock_generator = MockDataGenerator()
        
    async def initialize(self):
        """Initialize the service"""
        self.platform_service = get_platform_connection_service()
        self.validator_service = await get_data_source_validator()
    
    async def test_live_connection(self, user_id: str, org_id: str, 
                                 connection_id: str) -> IntegrationTestResult:
        """Test a live platform connection"""
        start_time = datetime.utcnow()
        
        try:
            # Get connection details
            connections = await self.platform_service.get_user_connections(user_id, org_id)
            connection = next((c for c in connections if c.id == connection_id), None)
            
            if not connection:
                return IntegrationTestResult(
                    connection_id=connection_id,
                    platform="unknown",
                    test_type="connection_test",
                    success=False,
                    data_retrieved=False,
                    record_count=0,
                    quality_score=0.0,
                    response_time_ms=0,
                    error_message="Connection not found"
                )
            
            # Test connection
            connector = await self.connector_manager.get_connector(
                connection_id,
                DataSource(connection.platform),
                connection.credentials
            )
            
            # Test data extraction
            end_date = datetime.utcnow().date()
            start_date = end_date - timedelta(days=3)  # Small test range
            
            test_data = await connector.extract_data(start_date, end_date)
            
            # Calculate response time
            response_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            # Validate data quality
            quality_report = await self.validator_service.validate_connection(
                user_id, org_id, connection_id
            )
            
            # Generate recommendations
            recommendations = []
            if quality_report.overall_score < 80:
                recommendations.append("Consider reviewing data quality issues")
            if response_time > 5000:
                recommendations.append("API response time is slow, monitor performance")
            if len(test_data) == 0:
                recommendations.append("No data retrieved, check date range and permissions")
            
            return IntegrationTestResult(
                connection_id=connection_id,
                platform=connection.platform,
                test_type="live_data_test",
                success=True,
                data_retrieved=len(test_data) > 0,
                record_count=len(test_data),
                quality_score=quality_report.overall_score,
                response_time_ms=response_time,
                recommendations=recommendations
            )
            
        except Exception as e:
            response_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            logger.error(f"Live connection test failed for {connection_id}: {str(e)}")
            
            return IntegrationTestResult(
                connection_id=connection_id,
                platform=connection.platform if 'connection' in locals() else "unknown",
                test_type="live_data_test",
                success=False,
                data_retrieved=False,
                record_count=0,
                quality_score=0.0,
                response_time_ms=response_time,
                error_message=str(e),
                recommendations=["Check connection credentials and API permissions"]
            )
    
    async def test_all_connections(self, user_id: str, org_id: str) -> List[IntegrationTestResult]:
        """Test all user connections"""
        try:
            connections = await self.platform_service.get_user_connections(user_id, org_id)
            
            test_tasks = [
                self.test_live_connection(user_id, org_id, conn.id)
                for conn in connections if conn.status == "active"
            ]
            
            results = await asyncio.gather(*test_tasks, return_exceptions=True)
            
            # Filter out exceptions and log them
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Test failed for connection {connections[i].id}: {str(result)}")
                else:
                    valid_results.append(result)
            
            return valid_results
            
        except Exception as e:
            logger.error(f"Failed to test all connections: {str(e)}")
            return []
    
    async def get_data_source_health(self, user_id: str, org_id: str) -> List[DataSourceHealth]:
        """Get health status of all data sources"""
        try:
            connections = await self.platform_service.get_user_connections(user_id, org_id)
            health_statuses = []
            
            for connection in connections:
                try:
                    # Get recent sync information
                    sync_logs = await self.platform_service.get_sync_logs(
                        user_id, org_id, connection.id, limit=5
                    )
                    
                    # Calculate health metrics
                    recent_errors = [log for log in sync_logs if log.sync_status == "failed"]
                    last_successful = next(
                        (log for log in sync_logs if log.sync_status == "success"),
                        None
                    )
                    
                    # Determine status
                    if connection.status != "active":
                        status = DataSourceStatus.UNAVAILABLE
                    elif len(recent_errors) >= 3:
                        status = DataSourceStatus.DEGRADED
                    elif last_successful and last_successful.completed_at:
                        hours_since_success = (datetime.utcnow() - last_successful.completed_at).total_seconds() / 3600
                        if hours_since_success > 48:
                            status = DataSourceStatus.DEGRADED
                        else:
                            status = DataSourceStatus.AVAILABLE
                    else:
                        status = DataSourceStatus.UNAVAILABLE
                    
                    # Get quality score
                    quality_score = None
                    try:
                        quality_report = await self.validator_service.validate_connection(
                            user_id, org_id, connection.id
                        )
                        quality_score = quality_report.overall_score
                    except:
                        pass
                    
                    health_statuses.append(DataSourceHealth(
                        source=f"{connection.platform}_{connection.id}",
                        status=status,
                        quality_score=quality_score,
                        last_successful_sync=last_successful.completed_at if last_successful else None,
                        error_count=len(recent_errors),
                        last_error=recent_errors[0].error_message if recent_errors else None,
                        response_time_ms=None  # Would be calculated from recent tests
                    ))
                    
                except Exception as e:
                    logger.error(f"Failed to get health for connection {connection.id}: {str(e)}")
                    health_statuses.append(DataSourceHealth(
                        source=f"{connection.platform}_{connection.id}",
                        status=DataSourceStatus.UNAVAILABLE,
                        error_count=1,
                        last_error=str(e)
                    ))
            
            return health_statuses
            
        except Exception as e:
            logger.error(f"Failed to get data source health: {str(e)}")
            return []
    
    async def get_hybrid_data(self, user_id: str, org_id: str, platform: str,
                            start_date: date, end_date: date,
                            data_mode: DataMode = DataMode.AUTO) -> Tuple[List[Dict[str, Any]], str]:
        """Get data using hybrid approach (live + mock fallback)"""
        try:
            # Get platform connections
            connections = await self.platform_service.get_user_connections(user_id, org_id)
            platform_connections = [c for c in connections if c.platform == platform and c.status == "active"]
            
            if not platform_connections and data_mode in [DataMode.LIVE_ONLY, DataMode.AUTO]:
                # No live connections available
                if data_mode == DataMode.LIVE_ONLY:
                    raise ValueError(f"No active connections found for platform {platform}")
                else:
                    # Fall back to mock data
                    mock_data = self.mock_generator.generate_platform_data(
                        DataSource(platform), start_date, end_date
                    )
                    return mock_data, "mock"
            
            if data_mode == DataMode.MOCK_ONLY:
                # Force mock data
                mock_data = self.mock_generator.generate_platform_data(
                    DataSource(platform), start_date, end_date
                )
                return mock_data, "mock"
            
            # Try to get live data
            for connection in platform_connections:
                try:
                    connector = await self.connector_manager.get_connector(
                        connection.id,
                        DataSource(platform),
                        connection.credentials
                    )
                    
                    live_data = await connector.extract_data(start_date, end_date)
                    
                    if live_data:
                        # Validate data quality
                        quality_report = await self.validator_service.validate_connection(
                            user_id, org_id, connection.id
                        )
                        
                        # If quality is acceptable, return live data
                        if quality_report.overall_score >= 60:  # Minimum acceptable quality
                            return live_data, "live"
                        else:
                            logger.warning(f"Live data quality too low ({quality_report.overall_score}%) for {connection.id}")
                    
                except Exception as e:
                    logger.error(f"Failed to get live data from {connection.id}: {str(e)}")
                    continue
            
            # If we reach here, live data failed or quality was poor
            if data_mode == DataMode.AUTO or data_mode == DataMode.HYBRID:
                # Fall back to mock data
                mock_data = self.mock_generator.generate_platform_data(
                    DataSource(platform), start_date, end_date
                )
                return mock_data, "mock_fallback"
            else:
                raise ValueError(f"Failed to retrieve live data for platform {platform}")
                
        except Exception as e:
            logger.error(f"Failed to get hybrid data for {platform}: {str(e)}")
            
            # Last resort: return mock data if possible
            if data_mode != DataMode.LIVE_ONLY:
                mock_data = self.mock_generator.generate_platform_data(
                    DataSource(platform), start_date, end_date
                )
                return mock_data, "mock_error_fallback"
            else:
                raise
    
    async def run_integration_test_suite(self, user_id: str, org_id: str) -> Dict[str, Any]:
        """Run comprehensive integration test suite"""
        test_start = datetime.utcnow()
        
        try:
            # Test all connections
            connection_tests = await self.test_all_connections(user_id, org_id)
            
            # Get data source health
            health_statuses = await self.get_data_source_health(user_id, org_id)
            
            # Test hybrid data retrieval for each platform
            platforms = list(set([test.platform for test in connection_tests]))
            hybrid_tests = []
            
            for platform in platforms:
                try:
                    end_date = datetime.utcnow().date()
                    start_date = end_date - timedelta(days=7)
                    
                    data, source = await self.get_hybrid_data(
                        user_id, org_id, platform, start_date, end_date, DataMode.AUTO
                    )
                    
                    hybrid_tests.append({
                        "platform": platform,
                        "success": True,
                        "data_source": source,
                        "record_count": len(data),
                        "test_date_range": f"{start_date} to {end_date}"
                    })
                    
                except Exception as e:
                    hybrid_tests.append({
                        "platform": platform,
                        "success": False,
                        "error": str(e),
                        "test_date_range": f"{start_date} to {end_date}"
                    })
            
            # Calculate summary metrics
            total_connections = len(connection_tests)
            successful_connections = len([t for t in connection_tests if t.success])
            healthy_sources = len([h for h in health_statuses if h.status == DataSourceStatus.AVAILABLE])
            
            avg_quality_score = 0
            if connection_tests:
                avg_quality_score = sum([t.quality_score for t in connection_tests]) / len(connection_tests)
            
            avg_response_time = 0
            if connection_tests:
                avg_response_time = sum([t.response_time_ms for t in connection_tests]) / len(connection_tests)
            
            test_duration = (datetime.utcnow() - test_start).total_seconds()
            
            return {
                "test_summary": {
                    "total_connections_tested": total_connections,
                    "successful_connections": successful_connections,
                    "success_rate": (successful_connections / total_connections * 100) if total_connections > 0 else 0,
                    "healthy_data_sources": healthy_sources,
                    "average_quality_score": round(avg_quality_score, 1),
                    "average_response_time_ms": round(avg_response_time, 1),
                    "test_duration_seconds": round(test_duration, 2),
                    "test_timestamp": test_start.isoformat()
                },
                "connection_tests": [
                    {
                        "connection_id": test.connection_id,
                        "platform": test.platform,
                        "success": test.success,
                        "data_retrieved": test.data_retrieved,
                        "record_count": test.record_count,
                        "quality_score": test.quality_score,
                        "response_time_ms": test.response_time_ms,
                        "error_message": test.error_message,
                        "recommendations": test.recommendations or []
                    }
                    for test in connection_tests
                ],
                "health_status": [
                    {
                        "source": health.source,
                        "status": health.status.value,
                        "quality_score": health.quality_score,
                        "last_successful_sync": health.last_successful_sync.isoformat() if health.last_successful_sync else None,
                        "error_count": health.error_count,
                        "last_error": health.last_error
                    }
                    for health in health_statuses
                ],
                "hybrid_data_tests": hybrid_tests,
                "recommendations": self._generate_integration_recommendations(
                    connection_tests, health_statuses, hybrid_tests
                )
            }
            
        except Exception as e:
            logger.error(f"Integration test suite failed: {str(e)}")
            return {
                "test_summary": {
                    "success": False,
                    "error": str(e),
                    "test_timestamp": test_start.isoformat()
                }
            }
    
    def _generate_integration_recommendations(self, connection_tests: List[IntegrationTestResult],
                                            health_statuses: List[DataSourceHealth],
                                            hybrid_tests: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on integration test results"""
        recommendations = []
        
        # Connection-based recommendations
        failed_connections = [t for t in connection_tests if not t.success]
        if failed_connections:
            recommendations.append(f"Review {len(failed_connections)} failed connections for credential or permission issues")
        
        low_quality_connections = [t for t in connection_tests if t.quality_score < 70]
        if low_quality_connections:
            recommendations.append(f"Improve data quality for {len(low_quality_connections)} connections with scores below 70%")
        
        slow_connections = [t for t in connection_tests if t.response_time_ms > 10000]
        if slow_connections:
            recommendations.append(f"Monitor {len(slow_connections)} connections with slow response times (>10s)")
        
        # Health-based recommendations
        degraded_sources = [h for h in health_statuses if h.status == DataSourceStatus.DEGRADED]
        if degraded_sources:
            recommendations.append(f"Address {len(degraded_sources)} degraded data sources")
        
        # Hybrid test recommendations
        failed_hybrid_tests = [t for t in hybrid_tests if not t.get("success", False)]
        if failed_hybrid_tests:
            recommendations.append(f"Review {len(failed_hybrid_tests)} platforms with hybrid data retrieval issues")
        
        # General recommendations
        if len(connection_tests) == 0:
            recommendations.append("Set up platform connections to enable live data integration")
        
        success_rate = len([t for t in connection_tests if t.success]) / len(connection_tests) * 100 if connection_tests else 0
        if success_rate < 80:
            recommendations.append("Overall connection success rate is below 80% - review system health")
        
        return recommendations

# Global service instance
_integration_service = None

async def get_live_data_integration_service() -> LiveDataIntegrationService:
    """Get the global live data integration service instance"""
    global _integration_service
    if _integration_service is None:
        _integration_service = LiveDataIntegrationService()
        await _integration_service.initialize()
    return _integration_service