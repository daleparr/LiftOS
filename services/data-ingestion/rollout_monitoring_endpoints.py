"""
Rollout and Monitoring API Endpoints

This module provides REST API endpoints for gradual rollout management
and production monitoring capabilities.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Body
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.database.connection import get_session
from shared.auth.jwt_auth import get_current_user
from shared.services.rollout_manager import (
    RolloutManager,
    RolloutConfig,
    RolloutType,
    RolloutStatus
)
from shared.services.monitoring_service import (
    MonitoringService,
    AlertType,
    AlertSeverity,
    MetricType
)
from shared.services.platform_connection_service import PlatformConnectionService
from shared.services.data_source_validator import DataSourceValidator
from shared.services.live_data_integration_service import LiveDataIntegrationService
from shared.models.platform_connections import (
    RolloutConfigRequest,
    RolloutStatusResponse,
    MonitoringDashboardResponse,
    AlertRequest,
    MetricRequest
)

logger = logging.getLogger(__name__)
security = HTTPBearer()

# Create router
router = APIRouter(prefix="/api/v1/rollout-monitoring", tags=["Rollout & Monitoring"])

def get_rollout_manager(
    db: Session = Depends(get_session),
    current_user: dict = Depends(get_current_user)
) -> RolloutManager:
    """Get rollout manager instance"""
    connection_service = PlatformConnectionService(db)
    validator = DataSourceValidator(db)
    integration_service = LiveDataIntegrationService(db, connection_service, validator)
    
    return RolloutManager(db, connection_service, validator, integration_service)

def get_monitoring_service(
    db: Session = Depends(get_session),
    current_user: dict = Depends(get_current_user)
) -> MonitoringService:
    """Get monitoring service instance"""
    connection_service = PlatformConnectionService(db)
    validator = DataSourceValidator(db)
    integration_service = LiveDataIntegrationService(db, connection_service, validator)
    
    return MonitoringService(db, connection_service, validator, integration_service)

# Rollout Management Endpoints

@router.post("/rollouts", response_model=Dict[str, Any])
async def create_rollout(
    config_request: RolloutConfigRequest,
    rollout_manager: RolloutManager = Depends(get_rollout_manager),
    current_user: dict = Depends(get_current_user)
):
    """Create a new rollout configuration"""
    try:
        # Convert request to RolloutConfig
        config = RolloutConfig(
            rollout_id=config_request.rollout_id,
            name=config_request.name,
            description=config_request.description,
            rollout_type=RolloutType(config_request.rollout_type),
            target_percentage=config_request.target_percentage,
            target_users=config_request.target_users,
            target_platforms=config_request.target_platforms,
            feature_flags=config_request.feature_flags,
            start_date=config_request.start_date,
            end_date=config_request.end_date,
            success_criteria=config_request.success_criteria,
            rollback_criteria=config_request.rollback_criteria,
            monitoring_config=config_request.monitoring_config
        )
        
        result = await rollout_manager.create_rollout(config, current_user["user_id"])
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"Error creating rollout: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create rollout: {str(e)}")

@router.post("/rollouts/{rollout_id}/start", response_model=Dict[str, Any])
async def start_rollout(
    rollout_id: str,
    rollout_manager: RolloutManager = Depends(get_rollout_manager),
    current_user: dict = Depends(get_current_user)
):
    """Start a rollout"""
    try:
        result = await rollout_manager.start_rollout(rollout_id, current_user["user_id"])
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"Error starting rollout {rollout_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start rollout: {str(e)}")

@router.post("/rollouts/{rollout_id}/pause", response_model=Dict[str, Any])
async def pause_rollout(
    rollout_id: str,
    reason: Optional[str] = Body(None),
    rollout_manager: RolloutManager = Depends(get_rollout_manager),
    current_user: dict = Depends(get_current_user)
):
    """Pause a rollout"""
    try:
        result = await rollout_manager.pause_rollout(rollout_id, current_user["user_id"], reason)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"Error pausing rollout {rollout_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to pause rollout: {str(e)}")

@router.post("/rollouts/{rollout_id}/rollback", response_model=Dict[str, Any])
async def rollback_rollout(
    rollout_id: str,
    reason: Optional[str] = Body(None),
    rollout_manager: RolloutManager = Depends(get_rollout_manager),
    current_user: dict = Depends(get_current_user)
):
    """Rollback a rollout"""
    try:
        result = await rollout_manager.rollback_rollout(rollout_id, current_user["user_id"], reason)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"Error rolling back rollout {rollout_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to rollback rollout: {str(e)}")

@router.get("/rollouts/{rollout_id}/status", response_model=RolloutStatusResponse)
async def get_rollout_status(
    rollout_id: str,
    rollout_manager: RolloutManager = Depends(get_rollout_manager),
    current_user: dict = Depends(get_current_user)
):
    """Get rollout status and metrics"""
    try:
        result = await rollout_manager.get_rollout_status(rollout_id)
        
        if not result["success"]:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return RolloutStatusResponse(**result)
        
    except Exception as e:
        logger.error(f"Error getting rollout status {rollout_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get rollout status: {str(e)}")

@router.get("/rollouts", response_model=Dict[str, Any])
async def list_rollouts(
    rollout_manager: RolloutManager = Depends(get_rollout_manager),
    current_user: dict = Depends(get_current_user)
):
    """List all active rollouts"""
    try:
        result = await rollout_manager.list_active_rollouts()
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"Error listing rollouts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list rollouts: {str(e)}")

# Monitoring Endpoints

@router.post("/metrics", response_model=Dict[str, Any])
async def record_metric(
    metric_request: MetricRequest,
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
    current_user: dict = Depends(get_current_user)
):
    """Record a custom metric"""
    try:
        await monitoring_service.record_metric(
            name=metric_request.name,
            value=metric_request.value,
            metric_type=MetricType(metric_request.metric_type),
            tags=metric_request.tags,
            metadata=metric_request.metadata
        )
        
        return {"success": True, "message": "Metric recorded successfully"}
        
    except Exception as e:
        logger.error(f"Error recording metric: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to record metric: {str(e)}")

@router.get("/metrics", response_model=List[Dict[str, Any]])
async def get_metrics(
    metric_name: Optional[str] = Query(None),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
    current_user: dict = Depends(get_current_user)
):
    """Get metrics with optional filtering"""
    try:
        result = await monitoring_service.get_metrics(
            metric_name=metric_name,
            start_time=start_time,
            end_time=end_time
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@router.post("/alerts", response_model=Dict[str, Any])
async def create_alert(
    alert_request: AlertRequest,
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
    current_user: dict = Depends(get_current_user)
):
    """Create a custom alert"""
    try:
        alert_id = await monitoring_service.create_alert(
            alert_type=AlertType(alert_request.alert_type),
            severity=AlertSeverity(alert_request.severity),
            title=alert_request.title,
            description=alert_request.description,
            source=alert_request.source,
            metadata=alert_request.metadata
        )
        
        return {
            "success": True,
            "alert_id": alert_id,
            "message": "Alert created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating alert: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create alert: {str(e)}")

@router.get("/alerts", response_model=List[Dict[str, Any]])
async def get_alerts(
    alert_type: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    resolved: Optional[bool] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
    current_user: dict = Depends(get_current_user)
):
    """Get alerts with optional filtering"""
    try:
        alert_type_enum = AlertType(alert_type) if alert_type else None
        severity_enum = AlertSeverity(severity) if severity else None
        
        result = await monitoring_service.get_alerts(
            alert_type=alert_type_enum,
            severity=severity_enum,
            resolved=resolved,
            limit=limit
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")

@router.post("/alerts/{alert_id}/resolve", response_model=Dict[str, Any])
async def resolve_alert(
    alert_id: str,
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
    current_user: dict = Depends(get_current_user)
):
    """Resolve an alert"""
    try:
        success = await monitoring_service.resolve_alert(alert_id, current_user["user_id"])
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return {
            "success": True,
            "message": "Alert resolved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error resolving alert {alert_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to resolve alert: {str(e)}")

@router.get("/health", response_model=Dict[str, Any])
async def get_health_status(
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
    current_user: dict = Depends(get_current_user)
):
    """Get overall system health status"""
    try:
        result = await monitoring_service.get_health_status()
        return result
        
    except Exception as e:
        logger.error(f"Error getting health status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get health status: {str(e)}")

@router.get("/dashboard", response_model=MonitoringDashboardResponse)
async def get_dashboard_data(
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
    current_user: dict = Depends(get_current_user)
):
    """Get comprehensive monitoring dashboard data"""
    try:
        result = await monitoring_service.get_dashboard_data()
        return MonitoringDashboardResponse(**result)
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")

# Advanced Monitoring Endpoints

@router.get("/metrics/summary", response_model=Dict[str, Any])
async def get_metrics_summary(
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
    current_user: dict = Depends(get_current_user)
):
    """Get metrics summary for dashboard"""
    try:
        result = await monitoring_service._get_metrics_summary()
        return {"success": True, "metrics": result}
        
    except Exception as e:
        logger.error(f"Error getting metrics summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics summary: {str(e)}")

@router.get("/performance", response_model=Dict[str, Any])
async def get_performance_metrics(
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
    current_user: dict = Depends(get_current_user)
):
    """Get performance metrics"""
    try:
        result = await monitoring_service._get_performance_metrics()
        return {"success": True, "performance": result}
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")

@router.get("/errors/analysis", response_model=Dict[str, Any])
async def get_error_analysis(
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
    current_user: dict = Depends(get_current_user)
):
    """Get error analysis data"""
    try:
        result = await monitoring_service._get_error_analysis()
        return {"success": True, "error_analysis": result}
        
    except Exception as e:
        logger.error(f"Error getting error analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get error analysis: {str(e)}")

@router.get("/platforms/statistics", response_model=Dict[str, Any])
async def get_platform_statistics(
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
    current_user: dict = Depends(get_current_user)
):
    """Get platform-specific statistics"""
    try:
        result = await monitoring_service._get_platform_statistics()
        return {"success": True, "platform_stats": result}
        
    except Exception as e:
        logger.error(f"Error getting platform statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get platform statistics: {str(e)}")

# Rollout Analytics Endpoints

@router.get("/rollouts/{rollout_id}/metrics", response_model=Dict[str, Any])
async def get_rollout_metrics(
    rollout_id: str,
    rollout_manager: RolloutManager = Depends(get_rollout_manager),
    current_user: dict = Depends(get_current_user)
):
    """Get detailed metrics for a specific rollout"""
    try:
        status_result = await rollout_manager.get_rollout_status(rollout_id)
        
        if not status_result["success"]:
            raise HTTPException(status_code=404, detail=status_result["error"])
        
        return {
            "success": True,
            "rollout_id": rollout_id,
            "metrics": status_result.get("metrics_history", []),
            "latest_metrics": status_result.get("latest_metrics"),
            "recommendations": status_result.get("recommendations", [])
        }
        
    except Exception as e:
        logger.error(f"Error getting rollout metrics {rollout_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get rollout metrics: {str(e)}")

@router.get("/rollouts/analytics", response_model=Dict[str, Any])
async def get_rollout_analytics(
    rollout_manager: RolloutManager = Depends(get_rollout_manager),
    current_user: dict = Depends(get_current_user)
):
    """Get rollout analytics and trends"""
    try:
        rollouts_result = await rollout_manager.list_active_rollouts()
        
        if not rollouts_result["success"]:
            raise HTTPException(status_code=500, detail=rollouts_result["error"])
        
        # Calculate analytics
        rollouts = rollouts_result["rollouts"]
        total_rollouts = len(rollouts)
        
        # Group by rollout type
        rollout_types = {}
        for rollout in rollouts:
            rollout_type = rollout["rollout_type"]
            if rollout_type not in rollout_types:
                rollout_types[rollout_type] = 0
            rollout_types[rollout_type] += 1
        
        # Calculate average progress
        total_progress = sum(rollout.get("progress", {}).get("percentage", 0) for rollout in rollouts)
        avg_progress = total_progress / total_rollouts if total_rollouts > 0 else 0
        
        return {
            "success": True,
            "analytics": {
                "total_rollouts": total_rollouts,
                "rollout_types": rollout_types,
                "average_progress": avg_progress,
                "rollouts": rollouts
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting rollout analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get rollout analytics: {str(e)}")