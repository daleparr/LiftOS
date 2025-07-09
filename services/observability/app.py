"""
Enhanced Observability Service for Lift OS Core

Comprehensive metrics collection, logging aggregation, health monitoring, and alerting
with time-series database storage and real-time analytics capabilities.
"""

import os
import asyncio
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, Union
from collections import defaultdict, deque
import json
import logging
from dataclasses import dataclass, asdict

import httpx
from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc
import pandas as pd
import numpy as np

# Import shared modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# KSE-SDK Integration
from shared.kse_sdk.client import LiftKSEClient
from shared.kse_sdk.core import Entity, SearchQuery, SearchResult
from shared.kse_sdk.models import EntityType, Domain

from shared.models.base import APIResponse, HealthCheck
from shared.database.observability_models import (
    MetricEntry, LogEntry, HealthCheckEntry, AlertEntry,
    ServiceRegistry, MetricAggregation, SystemSnapshot
)
from shared.database.connection import DatabaseManager, get_database
from shared.utils.time_series_storage import (
    TimeSeriesStorage, MetricPoint, TimeSeriesQuery, AggregatedMetric,
    get_time_series_storage
)
from shared.utils.config import get_service_config
from shared.utils.logging import setup_logging
from shared.health.health_checks import HealthChecker
from shared.auth.jwt_utils import verify_token, require_permissions

# Service configuration
config = get_service_config("observability", 8004)
logger = setup_logging("observability")

# Health checker
health_checker = HealthChecker("observability")

# Database manager
db_manager = DatabaseManager()

# FastAPI app

# KSE Client for intelligence integration
kse_client = None

async def initialize_kse_client():
    """Initialize KSE client for intelligence integration"""
    global kse_client
    try:
        kse_client = LiftKSEClient()
        print("KSE Client initialized successfully")
        return True
    except Exception as e:
        print(f"KSE Client initialization failed: {e}")
        kse_client = None
        return False

app = FastAPI(
    title="LiftOS Enhanced Observability Service",
    description="Comprehensive metrics, monitoring, and observability with time-series storage",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if config.DEBUG else ["http://localhost:3000", "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
time_series_storage: Optional[TimeSeriesStorage] = None
service_registry_cache: Dict[str, Dict[str, Any]] = {}
alert_rules: Dict[str, Dict[str, Any]] = {}

# Request/Response Models
class MetricRequest(BaseModel):
    """Request model for recording metrics"""
    name: str = Field(..., description="Metric name")
    value: Union[int, float] = Field(..., description="Metric value")
    labels: Optional[Dict[str, str]] = Field(default={}, description="Metric labels")
    organization_id: Optional[str] = Field(None, description="Organization ID")
    service_name: Optional[str] = Field(None, description="Service name")
    timestamp: Optional[datetime] = Field(None, description="Custom timestamp")
    
    @validator('name')
    def validate_metric_name(cls, v):
        if not v or len(v) > 255:
            raise ValueError('Metric name must be 1-255 characters')
        return v


class MetricsBatchRequest(BaseModel):
    """Request model for batch metric recording"""
    metrics: List[MetricRequest] = Field(..., description="List of metrics")
    
    @validator('metrics')
    def validate_metrics_count(cls, v):
        if len(v) > 1000:
            raise ValueError('Maximum 1000 metrics per batch')
        return v


class LogRequest(BaseModel):
    """Request model for recording logs"""
    level: str = Field(..., regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    message: str = Field(..., max_length=10000)
    service_name: str = Field(..., max_length=100)
    organization_id: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None


class AlertRequest(BaseModel):
    """Request model for creating alerts"""
    name: str = Field(..., max_length=255)
    description: str = Field(..., max_length=1000)
    severity: str = Field(..., regex="^(critical|warning|info)$")
    organization_id: Optional[str] = None
    service_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    rule_name: Optional[str] = None
    rule_expression: Optional[str] = None


class MetricsQueryRequest(BaseModel):
    """Request model for querying metrics"""
    metric_name: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    labels: Optional[Dict[str, str]] = None
    organization_id: Optional[str] = None
    service_name: Optional[str] = None
    aggregation: Optional[str] = Field(None, regex="^(avg|sum|min|max|count)$")
    time_window: Optional[str] = Field(None, regex="^(1m|5m|15m|1h|6h|1d)$")
    limit: Optional[int] = Field(1000, le=10000)


class ServiceRegistryRequest(BaseModel):
    """Request model for service registration"""
    service_name: str = Field(..., max_length=100)
    health_endpoint: str = Field(..., max_length=500)
    description: Optional[str] = None
    check_interval_seconds: Optional[int] = Field(30, ge=10, le=3600)
    timeout_seconds: Optional[int] = Field(10, ge=1, le=60)
    alert_on_failure: Optional[bool] = True
    failure_threshold: Optional[int] = Field(3, ge=1, le=10)
    tags: Optional[Dict[str, str]] = None


class SystemOverviewResponse(BaseModel):
    """Response model for system overview"""
    system_health: str
    total_services: int
    healthy_services: int
    unhealthy_services: int
    degraded_services: int
    total_alerts: int
    critical_alerts: int
    warning_alerts: int
    avg_response_time_ms: Optional[float]
    uptime_percentage: float
    last_updated: datetime
    trends: Dict[str, Any]


class MetricsAnalyticsResponse(BaseModel):
    """Response model for metrics analytics"""
    metric_name: str
    time_range: Dict[str, datetime]
    aggregation_type: str
    data_points: List[Dict[str, Any]]
    statistics: Dict[str, float]
    trends: Dict[str, Any]


# Background Tasks and Monitoring
class HealthMonitor:
    """Advanced health monitoring with configurable checks"""
    
    def __init__(self):
        self.check_interval = 30
        self.timeout = 10
        self.failure_counts: Dict[str, int] = defaultdict(int)
        self.last_check_times: Dict[str, datetime] = {}
    
    async def start_monitoring(self):
        """Start health monitoring background task"""
        while True:
            try:
                await self._check_all_services()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _check_all_services(self):
        """Check health of all registered services"""
        async with db_manager.get_session() as session:
            # Get active services from registry
            result = await session.execute(
                select(ServiceRegistry).where(ServiceRegistry.is_active == True)
            )
            services = result.scalars().all()
            
            # Check each service
            tasks = []
            for service in services:
                task = asyncio.create_task(
                    self._check_service_health(service)
                )
                tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_service_health(self, service: ServiceRegistry):
        """Check health of a single service"""
        try:
            start_time = datetime.now(timezone.utc)
            
            async with httpx.AsyncClient(timeout=service.timeout_seconds) as client:
                response = await client.get(service.health_endpoint)
                
            end_time = datetime.now(timezone.utc)
            response_time = (end_time - start_time).total_seconds() * 1000
            
            # Determine status
            if response.status_code == 200:
                status = "healthy"
                self.failure_counts[service.service_name] = 0
                details = response.json() if response.content else {}
            else:
                status = "unhealthy"
                self.failure_counts[service.service_name] += 1
                details = {"status_code": response.status_code}
            
            # Store health check result
            await time_series_storage.store_health_check(
                service_name=service.service_name,
                status=status,
                response_time_ms=response_time,
                details=details,
                endpoint_url=service.health_endpoint
            )
            
            # Store response time metric
            metric = MetricPoint(
                name="service_response_time_ms",
                value=response_time,
                timestamp=end_time,
                labels={
                    "service": service.service_name,
                    "status": status,
                    "endpoint": service.health_endpoint
                },
                service_name="observability"
            )
            await time_series_storage.store_metric(metric)
            
            # Check for alert conditions
            if (status == "unhealthy" and 
                service.alert_on_failure and 
                self.failure_counts[service.service_name] >= service.failure_threshold):
                
                await self._create_health_alert(service, details)
            
            self.last_check_times[service.service_name] = end_time
            
        except Exception as e:
            logger.error(f"Health check failed for {service.service_name}: {e}")
            
            # Record failure
            self.failure_counts[service.service_name] += 1
            
            await time_series_storage.store_health_check(
                service_name=service.service_name,
                status="unhealthy",
                response_time_ms=0,
                details={"error": str(e)},
                endpoint_url=service.health_endpoint
            )
            
            # Create alert if threshold reached
            if (service.alert_on_failure and 
                self.failure_counts[service.service_name] >= service.failure_threshold):
                
                await self._create_health_alert(service, {"error": str(e)})
    
    async def _create_health_alert(self, service: ServiceRegistry, details: Dict[str, Any]):
        """Create health-based alert"""
        try:
            alert_id = f"health_{service.service_name}_{int(time.time())}"
            
            async with db_manager.get_session() as session:
                # Check if similar alert already exists
                existing_alert = await session.execute(
                    select(AlertEntry).where(
                        and_(
                            AlertEntry.service_name == service.service_name,
                            AlertEntry.status == 'firing',
                            AlertEntry.name.like(f"Service {service.service_name} health%")
                        )
                    )
                )
                
                if existing_alert.scalar():
                    return  # Don't create duplicate alerts
                
                alert = AlertEntry(
                    alert_id=alert_id,
                    name=f"Service {service.service_name} health check failing",
                    description=f"Service {service.service_name} has failed {self.failure_counts[service.service_name]} consecutive health checks",
                    severity="critical",
                    status="firing",
                    timestamp=datetime.now(timezone.utc),
                    service_name=service.service_name,
                    metadata={
                        "health_details": details,
                        "failure_count": self.failure_counts[service.service_name],
                        "threshold": service.failure_threshold
                    },
                    rule_name="health_check_failure",
                    rule_expression=f"consecutive_failures >= {service.failure_threshold}"
                )
                
                session.add(alert)
                await session.commit()
                
                logger.warning(f"Health alert created: {alert.name}")
                
        except Exception as e:
            logger.error(f"Failed to create health alert: {e}")


# Global health monitor
health_monitor = HealthMonitor()


# Startup and Shutdown Events
@app.on_event("startup")
async def startup_event():
    """Initialize the observability service"""
    global time_series_storage
    
    logger.info("Starting Enhanced Observability Service")
    
    try:
        # Initialize database
        await db_manager.initialize()
        
        # Initialize time-series storage
        time_series_storage = await get_time_series_storage()
        
        # Start health monitoring
        asyncio.create_task(health_monitor.start_monitoring())
        
        # Start system snapshot task
        asyncio.create_task(periodic_system_snapshots())
        
        # Register default services
        await register_default_services()
        
        logger.info("Enhanced Observability Service initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize observability service: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown the observability service"""
    logger.info("Shutting down Enhanced Observability Service")
    
    try:
        if time_series_storage:
            await time_series_storage.shutdown()
        
        logger.info("Enhanced Observability Service shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


async def register_default_services():
    """Register default LiftOS services for monitoring"""
    default_services = [
        {
            "service_name": "gateway",
            "health_endpoint": "http://gateway:8000/health",
            "description": "API Gateway Service"
        },
        {
            "service_name": "auth",
            "health_endpoint": "http://auth:8001/health",
            "description": "Authentication Service"
        },
        {
            "service_name": "billing",
            "health_endpoint": "http://billing:8002/health",
            "description": "Billing Service"
        },
        {
            "service_name": "memory",
            "health_endpoint": "http://memory:8003/health",
            "description": "Memory Service"
        },
        {
            "service_name": "registry",
            "health_endpoint": "http://registry:8005/health",
            "description": "Registry Service"
        },
        {
            "service_name": "data-ingestion",
            "health_endpoint": "http://data-ingestion:8006/health",
            "description": "Data Ingestion Service"
        }
    ]
    
    try:
        async with db_manager.get_session() as session:
            for service_config in default_services:
                # Check if service already exists
                existing = await session.execute(
                    select(ServiceRegistry).where(
                        ServiceRegistry.service_name == service_config["service_name"]
                    )
                )
                
                if not existing.scalar():
                    service = ServiceRegistry(**service_config)
                    session.add(service)
            
            await session.commit()
            logger.info("Default services registered for monitoring")
            
    except Exception as e:
        logger.error(f"Failed to register default services: {e}")


async def periodic_system_snapshots():
    """Create periodic system health snapshots"""
    while True:
        try:
            await asyncio.sleep(300)  # Every 5 minutes
            await time_series_storage.create_system_snapshot()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error creating system snapshot: {e}")


# Health Check Endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "observability",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "2.0.0",
        "database_connected": db_manager._initialized,
        "time_series_storage": time_series_storage is not None
    }


# Metrics Endpoints
@app.post("/api/v1/metrics")
async def record_metric(
    request: MetricRequest,
    current_user: dict = Depends(verify_token)
):
    """Record a single metric"""
    try:
        # Create metric point
        metric = MetricPoint(
            name=request.name,
            value=request.value,
            timestamp=request.timestamp or datetime.now(timezone.utc),
            labels=request.labels or {},
            organization_id=request.organization_id,
            service_name=request.service_name
        )
        
        # Store metric
        success = await time_series_storage.store_metric(metric)
        
        if success:
            return {"status": "recorded", "timestamp": metric.timestamp.isoformat()}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to record metric"
            )
            
    except Exception as e:
        logger.error(f"Failed to record metric: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/api/v1/metrics/batch")
async def record_metrics_batch(
    request: MetricsBatchRequest,
    current_user: dict = Depends(verify_token)
):
    """Record multiple metrics in batch"""
    try:
        # Convert to metric points
        metrics = []
        for metric_req in request.metrics:
            metric = MetricPoint(
                name=metric_req.name,
                value=metric_req.value,
                timestamp=metric_req.timestamp or datetime.now(timezone.utc),
                labels=metric_req.labels or {},
                organization_id=metric_req.organization_id,
                service_name=metric_req.service_name
            )
            metrics.append(metric)
        
        # Store metrics
        success = await time_series_storage.store_metrics_batch(metrics)
        
        if success:
            return {
                "status": "recorded",
                "count": len(metrics),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to record metrics batch"
            )
            
    except Exception as e:
        logger.error(f"Failed to record metrics batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/api/v1/metrics/query")
async def query_metrics(
    request: MetricsQueryRequest,
    current_user: dict = Depends(verify_token)
):
    """Query metrics with advanced filtering and aggregation"""
    try:
        # Create time-series query
        query = TimeSeriesQuery(
            metric_name=request.metric_name,
            start_time=request.start_time,
            end_time=request.end_time,
            labels=request.labels,
            organization_id=request.organization_id,
            service_name=request.service_name,
            aggregation=request.aggregation,
            time_window=request.time_window,
            limit=request.limit
        )
        
        # Query metrics
        if request.aggregation and request.time_window:
            # Use pre-computed aggregations if available
            aggregated_metrics = await time_series_storage.get_aggregated_metrics(
                metric_name=request.metric_name,
                aggregation_type=request.aggregation,
                time_window=request.time_window,
                start_time=request.start_time or datetime.now(timezone.utc) - timedelta(hours=24),
                end_time=request.end_time or datetime.now(timezone.utc),
                labels=request.labels,
                service_name=request.service_name
            )
            
            results = [asdict(agg) for agg in aggregated_metrics]
        else:
            # Query raw metrics
            results = await time_series_storage.query_metrics(query)
        
        return {
            "metrics": results,
            "count": len(results),
            "query": request.dict(exclude_none=True)
        }
        
    except Exception as e:
        logger.error(f"Failed to query metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/v1/metrics/{metric_name}/analytics", response_model=MetricsAnalyticsResponse)
async def get_metrics_analytics(
    metric_name: str,
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    aggregation: str = Query("avg", regex="^(avg|sum|min|max|count)$"),
    time_window: str = Query("1h", regex="^(1m|5m|15m|1h|6h|1d)$"),
    service_name: Optional[str] = Query(None),
    current_user: dict = Depends(verify_token)
):
    """Get advanced analytics for a specific metric"""
    try:
        # Set default time range
        if not end_time:
            end_time = datetime.now(timezone.utc)
        if not start_time:
            start_time = end_time - timedelta(hours=24)
        
        # Get aggregated data
        aggregated_metrics = await time_series_storage.get_aggregated_metrics(
            metric_name=metric_name,
            aggregation_type=aggregation,
            time_window=time_window,
            start_time=start_time,
            end_time=end_time,
            service_name=service_name
        )
        
        # Convert to data points
        data_points = [
            {
                "timestamp": agg.timestamp.isoformat(),
                "value": agg.value,
                "sample_count": agg.sample_count
            }
            for agg in aggregated_metrics
        ]
        
        # Calculate statistics
        values = [agg.value for agg in aggregated_metrics]
        statistics = {}
        
        if values:
            statistics = {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "median": sorted(values)[len(values) // 2] if values else 0,
                "std_dev": np.std(values) if len(values) > 1 else 0,
                "total_samples": sum(agg.sample_count for agg in aggregated_metrics)
            }
        
        # Calculate trends (simple linear regression)
        trends = {}
        if len(values) > 1:
            x = np.arange(len(values))
            y = np.array(values)
            slope, intercept = np.polyfit(x, y, 1)
            trends = {
                "slope": float(slope),
                "direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
                "change_rate": float(slope / np.mean(y) * 100) if np.mean(y) != 0 else 0
            }
        
        return MetricsAnalyticsResponse(
            metric_name=metric_name,
            time_range={"start": start_time, "end": end_time},
            aggregation_type=aggregation,
            data_points=data_points,
            statistics=statistics,
            trends=trends
        )
        
    except Exception as e:
        logger.error(f"Failed to get metrics analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Service Registry Endpoints
@app.post("/api/v1/services/register")
async def register_service(
    request: ServiceRegistryRequest,
    current_user: dict = Depends(verify_token)
):
    """Register a service for health monitoring"""
    try:
        async with db_manager.get_session() as session:
            # Check if service already exists
            existing = await session.execute(
                select(ServiceRegistry).where(
                    ServiceRegistry.service_name == request.service_name
                )
            )
            
            if existing.scalar():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Service {request.service_name} already registered"
                )
            
            # Create new service registration
            service = ServiceRegistry(
                service_name=request.service_name,
                health_endpoint=request.health_endpoint,
                description=request.description,
                check_interval_seconds=request.check_interval_seconds,
                timeout_seconds=request.timeout_seconds,
                alert_on_failure=request.alert_on_failure,
                failure_threshold=request.failure_threshold,
                tags=request.tags
            )
            
            session.add(service)
            await session.commit()
            
            logger.info(f"Service registered: {request.service_name}")
            
            return {
                "status": "registered",
                "service_name": request.service_name,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to register service: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/v1/services")
async def list_services(
    current_user: dict = Depends(verify_token)
):
    """List all registered services"""
    try:
        async with db_manager.get_session() as session:
            result = await session.execute(select(ServiceRegistry))
            services = result.scalars().all()
            
            return {
                "services": [
                    {
                        "service_name": s.service_name,
                        "health_endpoint": s.health_endpoint,
                        "description": s.description,
                        "is_active": s.is_active,
                        "check_interval_seconds": s.check_interval_seconds,
                        "timeout_seconds": s.timeout_seconds,
                        "alert_on_failure": s.alert_on_failure,
                        "failure_threshold": s.failure_threshold,
                        "tags": s.tags,
                        "created_at": s.created_at.isoformat(),
                        "last_check_at": s.last_check_at.isoformat() if s.last_check_at else None
                    }
                    for s in services
                ],
                "count": len(services)
            }
            
    except Exception as e:
        logger.error(f"Failed to list services: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Health Monitoring Endpoints
@app.get("/api/v1/health/services")
async def get_services_health(
    hours: int = Query(24, ge=1, le=168),
    current_user: dict = Depends(verify_token)
):
    """Get health status and history of all services"""
    try:
        async with db_manager.get_session() as session:
            # Get latest health status for each service
            latest_health_subquery = (
                select(
                    HealthCheckEntry.service_name,
                    func.max(HealthCheckEntry.timestamp).label('latest_timestamp')
                )
                .group_by(HealthCheckEntry.service_name)
                .subquery()
            )
            
            latest_health_query = (
                select(HealthCheckEntry)
                .join(
                    latest_health_subquery,
                    and_(
                        HealthCheckEntry.service_name == latest_health_subquery.c.service_name,
                        HealthCheckEntry.timestamp == latest_health_subquery.c.latest_timestamp
                    )
                )
            )
            
            result = await session.execute(latest_health_query)
            latest_health_checks = result.scalars().all()
            
            # Build response
            services_health = {}
            for hc in latest_health_checks:
                # Get health history
                history = await time_series_storage.get_service_health_history(
                    service_name=hc.service_name,
                    hours=hours
                )
                
                services_health[hc.service_name] = {
                    "current_status": hc.status,
                    "last_check": hc.timestamp.isoformat(),
                    "response_time_ms": float(hc.response_time_ms),
                    "details": hc.details,
                    "endpoint_url": hc.endpoint_url,
                    "history": history
                }
            
            # Calculate summary statistics
            total_services = len(services_health)
            healthy_services = len([s for s in services_health.values() if s["current_status"] == "healthy"])
            unhealthy_services = len([s for s in services_health.values() if s["current_status"] == "unhealthy"])
            degraded_services = len([s for s in services_health.values() if s["current_status"] == "degraded"])
            
            avg_response_time = np.mean([
                s["response_time_ms"] for s in services_health.values()
                if s["response_time_ms"] > 0
            ]) if services_health else 0
            
            return {
                "services": services_health,
                "summary": {
                    "total_services": total_services,
                    "healthy_services": healthy_services,
                    "unhealthy_services": unhealthy_services,
                    "degraded_services": degraded_services,
                    "avg_response_time_ms": float(avg_response_time),
                    "uptime_percentage": (healthy_services / total_services * 100) if total_services > 0 else 0,
                    "last_updated": datetime.now(timezone.utc).isoformat()
                }
            }
            
    except Exception as e:
        logger.error(f"Failed to get services health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# System Overview Endpoint
@app.get("/api/v1/overview", response_model=SystemOverviewResponse)
async def get_system_overview(
    current_user: dict = Depends(verify_token)
):
    """Get comprehensive system overview with trends"""
    try:
        async with db_manager.get_session() as session:
            # Get latest system snapshot
            latest_snapshot = await session.execute(
                select(SystemSnapshot)
                .order_by(SystemSnapshot.timestamp.desc())
                .limit(1)
            )
            snapshot = latest_snapshot.scalar()
            
            if not snapshot:
                # Create initial snapshot if none exists
                await time_series_storage.create_system_snapshot()
                return SystemOverviewResponse(
                    system_health="unknown",
                    total_services=0,
                    healthy_services=0,
                    unhealthy_services=0,
                    degraded_services=0,
                    total_alerts=0,
                    critical_alerts=0,
                    warning_alerts=0,
                    avg_response_time_ms=None,
                    uptime_percentage=0.0,
                    last_updated=datetime.now(timezone.utc),
                    trends={}
                )
            
            # Get trends from previous snapshots
            previous_snapshots = await session.execute(
                select(SystemSnapshot)
                .where(SystemSnapshot.timestamp >= datetime.now(timezone.utc) - timedelta(hours=24))
                .order_by(SystemSnapshot.timestamp.desc())
                .limit(24)
            )
            snapshots = previous_snapshots.scalars().all()
            
            # Calculate trends
            trends = {}
            if len(snapshots) > 1:
                # Health trend
                health_scores = [(s.healthy_services / s.total_services * 100) if s.total_services > 0 else 0 for s in snapshots]
                health_trend = health_scores[0] - health_scores[-1] if len(health_scores) > 1 else 0
                
                # Response time trend
                response_times = [float(s.avg_response_time_ms) for s in snapshots if s.avg_response_time_ms]
                response_trend = response_times[0] - response_times[-1] if len(response_times) > 1 else 0
                
                # Alert trend
                alert_counts = [s.total_alerts for s in snapshots]
                alert_trend = alert_counts[0] - alert_counts[-1] if len(alert_counts) > 1 else 0
                
                trends = {
                    "health_trend_percentage": round(health_trend, 2),
                    "response_time_trend_ms": round(response_trend, 2),
                    "alert_trend_count": alert_trend,
                    "trend_period_hours": 24
                }
            
            # Determine overall system health
            uptime_percentage = (snapshot.healthy_services / snapshot.total_services * 100) if snapshot.total_services > 0 else 0
            
            if uptime_percentage >= 95:
                system_health = "healthy"
            elif uptime_percentage >= 80:
                system_health = "degraded"
            else:
                system_health = "unhealthy"
            
            return SystemOverviewResponse(
                system_health=system_health,
                total_services=snapshot.total_services,
                healthy_services=snapshot.healthy_services,
                unhealthy_services=snapshot.unhealthy_services,
                degraded_services=snapshot.degraded_services,
                total_alerts=snapshot.total_alerts,
                critical_alerts=snapshot.critical_alerts,
                warning_alerts=snapshot.warning_alerts,
                avg_response_time_ms=float(snapshot.avg_response_time_ms) if snapshot.avg_response_time_ms else None,
                uptime_percentage=uptime_percentage,
                last_updated=snapshot.timestamp,
                trends=trends
            )
            
    except Exception as e:
        logger.error(f"Failed to get system overview: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Logging Endpoints
@app.post("/api/v1/logs")
async def record_log(
    request: LogRequest,
    current_user: dict = Depends(verify_token)
):
    """Record a log entry"""
    try:
        async with db_manager.get_session() as session:
            log_entry = LogEntry(
                level=request.level,
                message=request.message,
                timestamp=request.timestamp or datetime.now(timezone.utc),
                service_name=request.service_name,
                organization_id=request.organization_id,
                correlation_id=request.correlation_id,
                meta_data=request.metadata
            )
            
            session.add(log_entry)
            await session.commit()
            
            return {
                "status": "recorded",
                "timestamp": log_entry.timestamp.isoformat(),
                "log_id": str(log_entry.id)
            }
            
    except Exception as e:
        logger.error(f"Failed to record log: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/v1/logs/{service_name}")
async def get_logs(
    service_name: str,
    level: Optional[str] = Query(None, regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    correlation_id: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
    current_user: dict = Depends(verify_token)
):
    """Get logs for a service with filtering"""
    try:
        async with db_manager.get_session() as session:
            # Build query
            query = select(LogEntry).where(LogEntry.service_name == service_name)
            
            # Apply filters
            if level:
                query = query.where(LogEntry.level == level)
            
            if start_time:
                query = query.where(LogEntry.timestamp >= start_time)
            
            if end_time:
                query = query.where(LogEntry.timestamp <= end_time)
            
            if correlation_id:
                query = query.where(LogEntry.correlation_id == correlation_id)
            
            # Order and limit
            query = query.order_by(LogEntry.timestamp.desc()).limit(limit)
            
            result = await session.execute(query)
            logs = result.scalars().all()
            
            return {
                "logs": [
                    {
                        "id": str(log.id),
                        "level": log.level,
                        "message": log.message,
                        "timestamp": log.timestamp.isoformat(),
                        "service_name": log.service_name,
                        "organization_id": str(log.organization_id) if log.organization_id else None,
                        "correlation_id": log.correlation_id,
                        "metadata": log.meta_data
                    }
                    for log in logs
                ],
                "service_name": service_name,
                "count": len(logs),
                "filters": {
                    "level": level,
                    "start_time": start_time.isoformat() if start_time else None,
                    "end_time": end_time.isoformat() if end_time else None,
                    "correlation_id": correlation_id,
                    "limit": limit
                }
            }
            
    except Exception as e:
        logger.error(f"Failed to get logs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Alert Endpoints
@app.post("/api/v1/alerts")
async def create_alert(
    request: AlertRequest,
    current_user: dict = Depends(verify_token)
):
    """Create a new alert"""
    try:
        alert_id = f"manual_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        async with db_manager.get_session() as session:
            alert = AlertEntry(
                alert_id=alert_id,
                name=request.name,
                description=request.description,
                severity=request.severity,
                status="firing",
                timestamp=datetime.now(timezone.utc),
                organization_id=request.organization_id,
                service_name=request.service_name,
                meta_data=request.metadata,
                rule_name=request.rule_name,
                rule_expression=request.rule_expression
            )
            
            session.add(alert)
            await session.commit()
            
            logger.info(f"Alert created: {alert.name}")
            
            return {
                "alert_id": alert_id,
                "status": "created",
                "timestamp": alert.timestamp.isoformat()
            }
            
    except Exception as e:
        logger.error(f"Failed to create alert: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/v1/alerts")
async def get_alerts(
    status: Optional[str] = Query(None, regex="^(firing|resolved|acknowledged)$"),
    severity: Optional[str] = Query(None, regex="^(critical|warning|info)$"),
    service_name: Optional[str] = Query(None),
    organization_id: Optional[str] = Query(None),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    limit: int = Query(100, le=1000),
    current_user: dict = Depends(verify_token)
):
    """Get alerts with filtering"""
    try:
        async with db_manager.get_session() as session:
            # Build query
            query = select(AlertEntry)
            
            conditions = []
            
            if status:
                conditions.append(AlertEntry.status == status)
            
            if severity:
                conditions.append(AlertEntry.severity == severity)
            
            if service_name:
                conditions.append(AlertEntry.service_name == service_name)
            
            if organization_id:
                conditions.append(AlertEntry.organization_id == organization_id)
            
            if start_time:
                conditions.append(AlertEntry.timestamp >= start_time)
            
            if end_time:
                conditions.append(AlertEntry.timestamp <= end_time)
            
            if conditions:
                query = query.where(and_(*conditions))
            
            # Order and limit
            query = query.order_by(AlertEntry.timestamp.desc()).limit(limit)
            
            result = await session.execute(query)
            alerts = result.scalars().all()
            
            # Calculate summary
            summary = {
                "critical": len([a for a in alerts if a.severity == "critical"]),
                "warning": len([a for a in alerts if a.severity == "warning"]),
                "info": len([a for a in alerts if a.severity == "info"]),
                "firing": len([a for a in alerts if a.status == "firing"]),
                "resolved": len([a for a in alerts if a.status == "resolved"]),
                "acknowledged": len([a for a in alerts if a.status == "acknowledged"])
            }
            
            return {
                "alerts": [
                    {
                        "alert_id": alert.alert_id,
                        "name": alert.name,
                        "description": alert.description,
                        "severity": alert.severity,
                        "status": alert.status,
                        "timestamp": alert.timestamp.isoformat(),
                        "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
                        "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                        "organization_id": str(alert.organization_id) if alert.organization_id else None,
                        "service_name": alert.service_name,
                        "metadata": alert.meta_data,
                        "rule_name": alert.rule_name,
                        "rule_expression": alert.rule_expression,
                        "duration_seconds": alert.duration_seconds
                    }
                    for alert in alerts
                ],
                "count": len(alerts),
                "summary": summary,
                "filters": {
                    "status": status,
                    "severity": severity,
                    "service_name": service_name,
                    "organization_id": organization_id,
                    "start_time": start_time.isoformat() if start_time else None,
                    "end_time": end_time.isoformat() if end_time else None,
                    "limit": limit
                }
            }
            
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.put("/api/v1/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    current_user: dict = Depends(verify_token)
):
    """Resolve an alert"""
    try:
        async with db_manager.get_session() as session:
            # Find alert
            result = await session.execute(
                select(AlertEntry).where(AlertEntry.alert_id == alert_id)
            )
            alert = result.scalar()
            
            if not alert:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Alert {alert_id} not found"
                )
            
            # Update alert status
            alert.status = "resolved"
            alert.resolved_at = datetime.now(timezone.utc)
            alert.updated_at = datetime.now(timezone.utc)
            
            await session.commit()
            
            logger.info(f"Alert resolved: {alert.name}")
            
            return {
                "alert_id": alert_id,
                "status": "resolved",
                "resolved_at": alert.resolved_at.isoformat()
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resolve alert: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.put("/api/v1/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    current_user: dict = Depends(verify_token)
):
    """Acknowledge an alert"""
    try:
        async with db_manager.get_session() as session:
            # Find alert
            result = await session.execute(
                select(AlertEntry).where(AlertEntry.alert_id == alert_id)
            )
            alert = result.scalar()
            
            if not alert:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Alert {alert_id} not found"
                )
            
            # Update alert status
            alert.status = "acknowledged"
            alert.acknowledged_at = datetime.now(timezone.utc)
            alert.updated_at = datetime.now(timezone.utc)
            
            await session.commit()
            
            logger.info(f"Alert acknowledged: {alert.name}")
            
            return {
                "alert_id": alert_id,
                "status": "acknowledged",
                "acknowledged_at": alert.acknowledged_at.isoformat()
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to acknowledge alert: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Dashboard and Analytics Endpoints
@app.get("/api/v1/dashboard/metrics")
async def get_dashboard_metrics(
    time_range: str = Query("24h", regex="^(1h|6h|24h|7d|30d)$"),
    current_user: dict = Depends(verify_token)
):
    """Get key metrics for dashboard display"""
    try:
        # Parse time range
        time_ranges = {
            "1h": timedelta(hours=1),
            "6h": timedelta(hours=6),
            "24h": timedelta(hours=24),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30)
        }
        
        end_time = datetime.now(timezone.utc)
        start_time = end_time - time_ranges[time_range]
        
        # Get key metrics
        dashboard_data = {}
        
        # Service health metrics
        health_history = await time_series_storage.query_metrics(
            TimeSeriesQuery(
                metric_name="service_response_time_ms",
                start_time=start_time,
                end_time=end_time,
                limit=1000
            )
        )
        
        dashboard_data["service_health"] = {
            "response_times": health_history,
            "avg_response_time": np.mean([m["value"] for m in health_history]) if health_history else 0
        }
        
        # System resource metrics (if available)
        cpu_metrics = await time_series_storage.query_metrics(
            TimeSeriesQuery(
                metric_name="cpu_usage_percent",
                start_time=start_time,
                end_time=end_time,
                limit=100
            )
        )
        
        memory_metrics = await time_series_storage.query_metrics(
            TimeSeriesQuery(
                metric_name="memory_usage_percent",
                start_time=start_time,
                end_time=end_time,
                limit=100
            )
        )
        
        dashboard_data["system_resources"] = {
            "cpu_usage": cpu_metrics,
            "memory_usage": memory_metrics
        }
        
        # Alert statistics
        async with db_manager.get_session() as session:
            alert_stats = await session.execute(
                select(
                    AlertEntry.severity,
                    AlertEntry.status,
                    func.count(AlertEntry.id).label('count')
                ).where(
                    AlertEntry.timestamp >= start_time
                ).group_by(AlertEntry.severity, AlertEntry.status)
            )
            
            alert_data = {}
            for severity, status, count in alert_stats.fetchall():
                if severity not in alert_data:
                    alert_data[severity] = {}
                alert_data[severity][status] = count
            
            dashboard_data["alerts"] = alert_data
        
        return {
            "time_range": time_range,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "data": dashboard_data,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get dashboard metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=config.PORT,
        reload=config.DEBUG,
        log_level="info"
    )