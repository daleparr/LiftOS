"""
Live Data Integration API Endpoints
Provides REST API for testing and managing live data integration
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Optional, Dict, Any
from datetime import date, datetime, timedelta
import logging

from shared.models.platform_connections import (
    DataQualityReportResponse,
    ValidationResultResponse
)
from shared.services.live_data_integration_service import (
    get_live_data_integration_service,
    DataMode,
    IntegrationTestResult,
    DataSourceHealth
)
from shared.auth.jwt_auth import get_current_user, require_permissions
from shared.utils.logging import setup_logging
from pydantic import BaseModel, Field

logger = setup_logging("live_data_integration_endpoints")

router = APIRouter(prefix="/api/v1/live-integration", tags=["live-integration"])

# Request/Response Models
class IntegrationTestResponse(BaseModel):
    """Response model for integration test"""
    connection_id: str
    platform: str
    test_type: str
    success: bool
    data_retrieved: bool
    record_count: int
    quality_score: float
    response_time_ms: int
    error_message: Optional[str] = None
    recommendations: List[str] = []

class DataSourceHealthResponse(BaseModel):
    """Response model for data source health"""
    source: str
    status: str
    quality_score: Optional[float] = None
    last_successful_sync: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[str] = None
    response_time_ms: Optional[int] = None

class HybridDataRequest(BaseModel):
    """Request model for hybrid data retrieval"""
    platform: str
    start_date: date
    end_date: date
    data_mode: str = Field("auto", pattern="^(mock_only|live_only|hybrid|auto)$")

class HybridDataResponse(BaseModel):
    """Response model for hybrid data"""
    platform: str
    data_source: str  # live, mock, mock_fallback, mock_error_fallback
    record_count: int
    date_range: Dict[str, str]
    data: List[Dict[str, Any]]
    metadata: Dict[str, Any] = {}

class IntegrationTestSuiteResponse(BaseModel):
    """Response model for integration test suite"""
    test_summary: Dict[str, Any]
    connection_tests: List[IntegrationTestResponse]
    health_status: List[DataSourceHealthResponse]
    hybrid_data_tests: List[Dict[str, Any]]
    recommendations: List[str]

@router.post("/test-connection/{connection_id}", response_model=IntegrationTestResponse)
async def test_live_connection(
    connection_id: str,
    current_user: dict = Depends(get_current_user),
    integration_service = Depends(get_live_data_integration_service)
):
    """Test a specific live platform connection"""
    try:
        user_id = current_user["user_id"]
        org_id = current_user["org_id"]
        
        # Validate permissions
        await require_permissions(current_user, ["data:read"])
        
        # Run integration test
        result = await integration_service.test_live_connection(user_id, org_id, connection_id)
        
        return IntegrationTestResponse(
            connection_id=result.connection_id,
            platform=result.platform,
            test_type=result.test_type,
            success=result.success,
            data_retrieved=result.data_retrieved,
            record_count=result.record_count,
            quality_score=result.quality_score,
            response_time_ms=result.response_time_ms,
            error_message=result.error_message,
            recommendations=result.recommendations or []
        )
        
    except Exception as e:
        logger.error(f"Failed to test connection {connection_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Connection test failed")

@router.post("/test-all-connections", response_model=List[IntegrationTestResponse])
async def test_all_connections(
    current_user: dict = Depends(get_current_user),
    integration_service = Depends(get_live_data_integration_service)
):
    """Test all user platform connections"""
    try:
        user_id = current_user["user_id"]
        org_id = current_user["org_id"]
        
        # Validate permissions
        await require_permissions(current_user, ["data:read"])
        
        # Run tests for all connections
        results = await integration_service.test_all_connections(user_id, org_id)
        
        return [
            IntegrationTestResponse(
                connection_id=result.connection_id,
                platform=result.platform,
                test_type=result.test_type,
                success=result.success,
                data_retrieved=result.data_retrieved,
                record_count=result.record_count,
                quality_score=result.quality_score,
                response_time_ms=result.response_time_ms,
                error_message=result.error_message,
                recommendations=result.recommendations or []
            )
            for result in results
        ]
        
    except Exception as e:
        logger.error(f"Failed to test all connections: {str(e)}")
        raise HTTPException(status_code=500, detail="Connection tests failed")

@router.get("/health-status", response_model=List[DataSourceHealthResponse])
async def get_data_source_health(
    current_user: dict = Depends(get_current_user),
    integration_service = Depends(get_live_data_integration_service)
):
    """Get health status of all data sources"""
    try:
        user_id = current_user["user_id"]
        org_id = current_user["org_id"]
        
        # Validate permissions
        await require_permissions(current_user, ["data:read"])
        
        # Get health statuses
        health_statuses = await integration_service.get_data_source_health(user_id, org_id)
        
        return [
            DataSourceHealthResponse(
                source=health.source,
                status=health.status.value,
                quality_score=health.quality_score,
                last_successful_sync=health.last_successful_sync,
                error_count=health.error_count,
                last_error=health.last_error,
                response_time_ms=health.response_time_ms
            )
            for health in health_statuses
        ]
        
    except Exception as e:
        logger.error(f"Failed to get health status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get health status")

@router.post("/hybrid-data", response_model=HybridDataResponse)
async def get_hybrid_data(
    request: HybridDataRequest,
    current_user: dict = Depends(get_current_user),
    integration_service = Depends(get_live_data_integration_service)
):
    """Get data using hybrid approach (live + mock fallback)"""
    try:
        user_id = current_user["user_id"]
        org_id = current_user["org_id"]
        
        # Validate permissions
        await require_permissions(current_user, ["data:read"])
        
        # Validate date range
        if request.start_date > request.end_date:
            raise HTTPException(status_code=400, detail="Start date must be before end date")
        
        # Convert data mode
        data_mode = DataMode(request.data_mode)
        
        # Get hybrid data
        data, source = await integration_service.get_hybrid_data(
            user_id, org_id, request.platform,
            request.start_date, request.end_date, data_mode
        )
        
        return HybridDataResponse(
            platform=request.platform,
            data_source=source,
            record_count=len(data),
            date_range={
                "start_date": request.start_date.isoformat(),
                "end_date": request.end_date.isoformat()
            },
            data=data,
            metadata={
                "data_mode_requested": request.data_mode,
                "data_mode_used": data_mode.value,
                "retrieved_at": datetime.utcnow().isoformat()
            }
        )
        
    except ValueError as e:
        logger.error(f"Invalid request for hybrid data: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get hybrid data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve data")

@router.post("/test-suite", response_model=IntegrationTestSuiteResponse)
async def run_integration_test_suite(
    current_user: dict = Depends(get_current_user),
    integration_service = Depends(get_live_data_integration_service)
):
    """Run comprehensive integration test suite"""
    try:
        user_id = current_user["user_id"]
        org_id = current_user["org_id"]
        
        # Validate permissions
        await require_permissions(current_user, ["data:read"])
        
        # Run comprehensive test suite
        test_results = await integration_service.run_integration_test_suite(user_id, org_id)
        
        # Convert connection tests
        connection_tests = [
            IntegrationTestResponse(**test)
            for test in test_results.get("connection_tests", [])
        ]
        
        # Convert health status
        health_status = [
            DataSourceHealthResponse(**health)
            for health in test_results.get("health_status", [])
        ]
        
        return IntegrationTestSuiteResponse(
            test_summary=test_results.get("test_summary", {}),
            connection_tests=connection_tests,
            health_status=health_status,
            hybrid_data_tests=test_results.get("hybrid_data_tests", []),
            recommendations=test_results.get("recommendations", [])
        )
        
    except Exception as e:
        logger.error(f"Failed to run integration test suite: {str(e)}")
        raise HTTPException(status_code=500, detail="Integration test suite failed")

@router.get("/data-modes")
async def get_available_data_modes():
    """Get available data modes for hybrid data retrieval"""
    return {
        "data_modes": [
            {
                "value": "auto",
                "name": "Auto",
                "description": "Automatically choose best available data source"
            },
            {
                "value": "live_only",
                "name": "Live Only",
                "description": "Use only live data from platform APIs"
            },
            {
                "value": "mock_only",
                "name": "Mock Only",
                "description": "Use only mock/demo data"
            },
            {
                "value": "hybrid",
                "name": "Hybrid",
                "description": "Prefer live data with mock fallback"
            }
        ]
    }

@router.get("/integration-status")
async def get_integration_status(
    current_user: dict = Depends(get_current_user),
    integration_service = Depends(get_live_data_integration_service)
):
    """Get overall integration status summary"""
    try:
        user_id = current_user["user_id"]
        org_id = current_user["org_id"]
        
        # Validate permissions
        await require_permissions(current_user, ["data:read"])
        
        # Get health statuses
        health_statuses = await integration_service.get_data_source_health(user_id, org_id)
        
        # Calculate summary metrics
        total_sources = len(health_statuses)
        healthy_sources = len([h for h in health_statuses if h.status.value == "available"])
        degraded_sources = len([h for h in health_statuses if h.status.value == "degraded"])
        unavailable_sources = len([h for h in health_statuses if h.status.value == "unavailable"])
        
        # Calculate average quality score
        quality_scores = [h.quality_score for h in health_statuses if h.quality_score is not None]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Determine overall status
        if total_sources == 0:
            overall_status = "no_connections"
        elif healthy_sources / total_sources >= 0.8:
            overall_status = "healthy"
        elif healthy_sources / total_sources >= 0.5:
            overall_status = "degraded"
        else:
            overall_status = "critical"
        
        return {
            "overall_status": overall_status,
            "total_data_sources": total_sources,
            "healthy_sources": healthy_sources,
            "degraded_sources": degraded_sources,
            "unavailable_sources": unavailable_sources,
            "health_percentage": round((healthy_sources / total_sources * 100) if total_sources > 0 else 0, 1),
            "average_quality_score": round(avg_quality, 1),
            "last_updated": datetime.utcnow().isoformat(),
            "recommendations": [
                rec for rec in [
                    "Set up platform connections to enable live data" if total_sources == 0 else None,
                    f"Address {degraded_sources} degraded data sources" if degraded_sources > 0 else None,
                    f"Fix {unavailable_sources} unavailable data sources" if unavailable_sources > 0 else None,
                    "Overall system health is good" if overall_status == "healthy" else None
                ] if rec is not None
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to get integration status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get integration status")

async def _run_background_test_suite(integration_service, user_id: str, org_id: str):
    """Run integration test suite in background"""
    try:
        results = await integration_service.run_integration_test_suite(user_id, org_id)
        logger.info(f"Background integration test suite completed for user {user_id}")
        
        # Here you could store results in cache/database and/or send notifications
        
    except Exception as e:
        logger.error(f"Background integration test suite failed for user {user_id}: {str(e)}")