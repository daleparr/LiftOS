"""
Analytics and Optimization API Endpoints
Provides advanced analytics, performance optimization, and intelligent recommendations
"""

# Add path for shared modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from shared.models.platform_connections import (
    AnalyticsTimeframe, OptimizationType, PerformanceAnalytics,
    CostAnalytics, QualityTrends, PredictiveAnalytics,
    OptimizationRecommendation, PlatformOptimization
)
from shared.services.analytics_optimization_service import AnalyticsOptimizationService
from shared.auth.jwt_auth import get_current_user
from shared.models.base import User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analytics", tags=["analytics"])

# Service will be initialized via dependency injection
async def get_analytics_service() -> AnalyticsOptimizationService:
    """Dependency to get analytics optimization service"""
    from shared.database.connection import get_session
    from shared.services.platform_connection_service import get_platform_connection_service
    from shared.services.data_source_validator import get_data_source_validator
    from shared.services.live_data_integration_service import LiveDataIntegrationService
    from shared.services.monitoring_service import MonitoringService
    
    db_session = next(get_session())
    connection_service = get_platform_connection_service()
    validator = await get_data_source_validator()
    integration_service = LiveDataIntegrationService()
    await integration_service.initialize()
    monitoring_service = MonitoringService(db_session)
    
    return AnalyticsOptimizationService(
        db_session=db_session,
        connection_service=connection_service,
        validator=validator,
        integration_service=integration_service,
        monitoring_service=monitoring_service
    )

@router.get("/performance/{platform}", response_model=PerformanceAnalytics)
async def get_performance_analytics(
    platform: str,
    timeframe: AnalyticsTimeframe = Query(AnalyticsTimeframe.LAST_7_DAYS),
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsOptimizationService = Depends(get_analytics_service)
):
    """Get performance analytics for a platform"""
    try:
        analytics = await analytics_service.get_performance_analytics(
            platform=platform,
            timeframe=timeframe,
            user_id=current_user.id
        )
        return analytics
    except Exception as e:
        logger.error(f"Error getting performance analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cost/{platform}", response_model=CostAnalytics)
async def get_cost_analytics(
    platform: str,
    timeframe: AnalyticsTimeframe = Query(AnalyticsTimeframe.LAST_30_DAYS),
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsOptimizationService = Depends(get_analytics_service)
):
    """Get cost analytics for a platform"""
    try:
        analytics = await analytics_service.get_cost_analytics(
            platform=platform,
            timeframe=timeframe,
            user_id=current_user.id
        )
        return analytics
    except Exception as e:
        logger.error(f"Error getting cost analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/quality/{platform}", response_model=QualityTrends)
async def get_quality_trends(
    platform: str,
    timeframe: AnalyticsTimeframe = Query(AnalyticsTimeframe.LAST_30_DAYS),
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsOptimizationService = Depends(get_analytics_service)
):
    """Get data quality trends for a platform"""
    try:
        trends = await analytics_service.get_quality_trends(
            platform=platform,
            timeframe=timeframe,
            user_id=current_user.id
        )
        return trends
    except Exception as e:
        logger.error(f"Error getting quality trends: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predictive/{platform}", response_model=PredictiveAnalytics)
async def get_predictive_analytics(
    platform: str,
    horizon_days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsOptimizationService = Depends(get_analytics_service)
):
    """Get predictive analytics for a platform"""
    try:
        predictions = await analytics_service.get_predictive_analytics(
            platform=platform,
            horizon_days=horizon_days,
            user_id=current_user.id
        )
        return predictions
    except Exception as e:
        logger.error(f"Error getting predictive analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/optimization/{platform}", response_model=PlatformOptimization)
async def get_platform_optimization(
    platform: str,
    optimization_goals: List[OptimizationType] = Query([OptimizationType.PERFORMANCE]),
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsOptimizationService = Depends(get_analytics_service)
):
    """Get platform optimization recommendations"""
    try:
        optimization = await analytics_service.get_platform_optimization(
            platform=platform,
            optimization_goals=optimization_goals,
            user_id=current_user.id
        )
        return optimization
    except Exception as e:
        logger.error(f"Error getting platform optimization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommendations", response_model=List[OptimizationRecommendation])
async def get_optimization_recommendations(
    platform: Optional[str] = Query(None),
    optimization_type: Optional[OptimizationType] = Query(None),
    priority_threshold: float = Query(0.5, ge=0.0, le=1.0),
    limit: int = Query(10, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsOptimizationService = Depends(get_analytics_service)
):
    """Get optimization recommendations across platforms"""
    try:
        recommendations = await analytics_service.get_optimization_recommendations(
            user_id=current_user.id,
            platform=platform,
            optimization_type=optimization_type,
            priority_threshold=priority_threshold,
            limit=limit
        )
        return recommendations
    except Exception as e:
        logger.error(f"Error getting optimization recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recommendations/{recommendation_id}/apply")
async def apply_optimization_recommendation(
    recommendation_id: str,
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsOptimizationService = Depends(get_analytics_service)
):
    """Apply an optimization recommendation"""
    try:
        result = await analytics_service.apply_optimization_recommendation(
            recommendation_id=recommendation_id,
            user_id=current_user.id
        )
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Error applying optimization recommendation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recommendations/{recommendation_id}/dismiss")
async def dismiss_optimization_recommendation(
    recommendation_id: str,
    reason: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsOptimizationService = Depends(get_analytics_service)
):
    """Dismiss an optimization recommendation"""
    try:
        await analytics_service.dismiss_optimization_recommendation(
            recommendation_id=recommendation_id,
            user_id=current_user.id,
            reason=reason
        )
        return {"success": True, "message": "Recommendation dismissed"}
    except Exception as e:
        logger.error(f"Error dismissing optimization recommendation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard/overview")
async def get_analytics_overview(
    timeframe: AnalyticsTimeframe = Query(AnalyticsTimeframe.LAST_7_DAYS),
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsOptimizationService = Depends(get_analytics_service)
):
    """Get analytics overview for dashboard"""
    try:
        overview = await analytics_service.get_analytics_overview(
            user_id=current_user.id,
            timeframe=timeframe
        )
        return overview
    except Exception as e:
        logger.error(f"Error getting analytics overview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard/insights")
async def get_intelligent_insights(
    limit: int = Query(5, ge=1, le=20),
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsOptimizationService = Depends(get_analytics_service)
):
    """Get intelligent insights for dashboard"""
    try:
        insights = await analytics_service.get_intelligent_insights(
            user_id=current_user.id,
            limit=limit
        )
        return insights
    except Exception as e:
        logger.error(f"Error getting intelligent insights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/anomalies")
async def get_anomaly_detection(
    platform: Optional[str] = Query(None),
    timeframe: AnalyticsTimeframe = Query(AnalyticsTimeframe.LAST_7_DAYS),
    severity_threshold: float = Query(0.7, ge=0.0, le=1.0),
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsOptimizationService = Depends(get_analytics_service)
):
    """Get anomaly detection results"""
    try:
        anomalies = await analytics_service.detect_anomalies(
            user_id=current_user.id,
            platform=platform,
            timeframe=timeframe,
            severity_threshold=severity_threshold
        )
        return anomalies
    except Exception as e:
        logger.error(f"Error detecting anomalies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/benchmarks/{platform}")
async def get_platform_benchmarks(
    platform: str,
    metric_type: str = Query("performance"),
    timeframe: AnalyticsTimeframe = Query(AnalyticsTimeframe.LAST_30_DAYS),
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsOptimizationService = Depends(get_analytics_service)
):
    """Get platform benchmarks and comparisons"""
    try:
        benchmarks = await analytics_service.get_platform_benchmarks(
            platform=platform,
            metric_type=metric_type,
            timeframe=timeframe,
            user_id=current_user.id
        )
        return benchmarks
    except Exception as e:
        logger.error(f"Error getting platform benchmarks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/export")
async def export_analytics_data(
    platforms: List[str],
    timeframe: AnalyticsTimeframe,
    export_format: str = Query("csv", regex="^(csv|json|excel)$"),
    include_predictions: bool = Query(False),
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsOptimizationService = Depends(get_analytics_service)
):
    """Export analytics data in various formats"""
    try:
        export_result = await analytics_service.export_analytics_data(
            platforms=platforms,
            timeframe=timeframe,
            export_format=export_format,
            include_predictions=include_predictions,
            user_id=current_user.id
        )
        return export_result
    except Exception as e:
        logger.error(f"Error exporting analytics data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def analytics_health_check():
    """Health check for analytics service"""
    try:
        health = await analytics_service.health_check()
        return health
    except Exception as e:
        logger.error(f"Analytics service health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Analytics service unavailable")

# Export router with expected name
analytics_router = router