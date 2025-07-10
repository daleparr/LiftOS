"""
Production Monitoring and Alerting Service

This service provides comprehensive monitoring, alerting, and observability
for platform connections in production environments.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, asdict
import json
import statistics
from collections import defaultdict, deque

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc

from ..database.user_platform_models import (
    UserPlatformConnection,
    ConnectionAuditLog
)
from ..models.platform_connections import PlatformCredentials
from ..models.platform_connections import ConnectionStatus
from ..models.marketing import DataSource as PlatformType
from .platform_connection_service import PlatformConnectionService
from .data_source_validator import DataSourceValidator
from .live_data_integration_service import LiveDataIntegrationService

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    """Alert types"""
    PERFORMANCE = "performance"
    ERROR_RATE = "error_rate"
    DATA_QUALITY = "data_quality"
    CONNECTION_FAILURE = "connection_failure"
    SECURITY = "security"
    CAPACITY = "capacity"
    ANOMALY = "anomaly"

class MetricType(Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    description: str
    source: str
    timestamp: datetime
    metadata: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None

@dataclass
class Metric:
    """Metric data structure"""
    name: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    tags: Dict[str, str]
    metadata: Dict[str, Any]

@dataclass
class HealthCheck:
    """Health check result"""
    service: str
    status: str
    response_time: float
    timestamp: datetime
    details: Dict[str, Any]

class MonitoringService:
    """
    Comprehensive monitoring service for platform connections
    with alerting, metrics collection, and health monitoring.
    """
    
    def __init__(
        self,
        db_session: Session,
        connection_service: PlatformConnectionService,
        validator: DataSourceValidator,
        integration_service: LiveDataIntegrationService
    ):
        self.db = db_session
        self.connection_service = connection_service
        self.validator = validator
        self.integration_service = integration_service
        
        # Monitoring state
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts: Dict[str, Alert] = {}
        self.health_checks: Dict[str, HealthCheck] = {}
        self.alert_handlers: Dict[AlertType, List[Callable]] = defaultdict(list)
        
        # Monitoring configuration
        self.monitoring_config = {
            "collection_interval": 30,  # seconds
            "alert_thresholds": {
                "error_rate": 0.05,
                "response_time": 5000,  # ms
                "data_quality": 0.8,
                "connection_failure_rate": 0.1
            },
            "retention_period": timedelta(days=30)
        }
        
        # Start monitoring tasks
        self._start_monitoring_tasks()
    
    def _start_monitoring_tasks(self):
        """Start background monitoring tasks"""
        asyncio.create_task(self._collect_metrics_loop())
        asyncio.create_task(self._health_check_loop())
        asyncio.create_task(self._alert_processing_loop())
        asyncio.create_task(self._cleanup_old_data_loop())
    
    async def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        tags: Dict[str, str] = None,
        metadata: Dict[str, Any] = None
    ):
        """Record a metric"""
        try:
            metric = Metric(
                name=name,
                metric_type=metric_type,
                value=value,
                timestamp=datetime.utcnow(),
                tags=tags or {},
                metadata=metadata or {}
            )
            
            self.metrics[name].append(metric)
            
            # Check for alert conditions
            await self._check_metric_alerts(metric)
            
        except Exception as e:
            logger.error(f"Error recording metric {name}: {str(e)}")
    
    async def create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        title: str,
        description: str,
        source: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Create a new alert"""
        try:
            alert_id = f"{alert_type.value}_{int(datetime.utcnow().timestamp())}"
            
            alert = Alert(
                alert_id=alert_id,
                alert_type=alert_type,
                severity=severity,
                title=title,
                description=description,
                source=source,
                timestamp=datetime.utcnow(),
                metadata=metadata or {}
            )
            
            self.alerts[alert_id] = alert
            
            # Trigger alert handlers
            await self._trigger_alert_handlers(alert)
            
            logger.warning(f"Alert created: {title} ({severity.value})")
            
            return alert_id
            
        except Exception as e:
            logger.error(f"Error creating alert: {str(e)}")
            return ""
    
    async def resolve_alert(
        self,
        alert_id: str,
        resolved_by: str = "system"
    ) -> bool:
        """Resolve an alert"""
        try:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = datetime.utcnow()
                alert.resolved_by = resolved_by
                
                logger.info(f"Alert resolved: {alert.title} by {resolved_by}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {str(e)}")
            return False
    
    async def get_metrics(
        self,
        metric_name: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        tags: Dict[str, str] = None
    ) -> List[Dict[str, Any]]:
        """Get metrics with optional filtering"""
        try:
            results = []
            
            if metric_name:
                metric_names = [metric_name] if metric_name in self.metrics else []
            else:
                metric_names = list(self.metrics.keys())
            
            for name in metric_names:
                for metric in self.metrics[name]:
                    # Time filtering
                    if start_time and metric.timestamp < start_time:
                        continue
                    if end_time and metric.timestamp > end_time:
                        continue
                    
                    # Tag filtering
                    if tags:
                        if not all(metric.tags.get(k) == v for k, v in tags.items()):
                            continue
                    
                    results.append(asdict(metric))
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            return []
    
    async def get_alerts(
        self,
        alert_type: AlertType = None,
        severity: AlertSeverity = None,
        resolved: bool = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get alerts with optional filtering"""
        try:
            alerts = list(self.alerts.values())
            
            # Apply filters
            if alert_type:
                alerts = [a for a in alerts if a.alert_type == alert_type]
            
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            
            if resolved is not None:
                alerts = [a for a in alerts if a.resolved == resolved]
            
            # Sort by timestamp (newest first)
            alerts.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Apply limit
            alerts = alerts[:limit]
            
            return [asdict(alert) for alert in alerts]
            
        except Exception as e:
            logger.error(f"Error getting alerts: {str(e)}")
            return []
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        try:
            # Get recent health checks
            recent_checks = {
                service: check for service, check in self.health_checks.items()
                if check.timestamp > datetime.utcnow() - timedelta(minutes=5)
            }
            
            # Calculate overall health
            healthy_services = sum(1 for check in recent_checks.values() if check.status == "healthy")
            total_services = len(recent_checks)
            health_percentage = (healthy_services / total_services * 100) if total_services > 0 else 0
            
            # Get active alerts by severity
            active_alerts = [a for a in self.alerts.values() if not a.resolved]
            alerts_by_severity = defaultdict(int)
            for alert in active_alerts:
                alerts_by_severity[alert.severity.value] += 1
            
            # Get recent metrics summary
            metrics_summary = await self._get_metrics_summary()
            
            return {
                "overall_health": "healthy" if health_percentage >= 90 else "degraded" if health_percentage >= 70 else "unhealthy",
                "health_percentage": health_percentage,
                "services": {
                    service: {
                        "status": check.status,
                        "response_time": check.response_time,
                        "last_check": check.timestamp.isoformat()
                    }
                    for service, check in recent_checks.items()
                },
                "alerts": {
                    "total_active": len(active_alerts),
                    "by_severity": dict(alerts_by_severity)
                },
                "metrics": metrics_summary,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting health status: {str(e)}")
            return {
                "overall_health": "unknown",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        try:
            # Get health status
            health_status = await self.get_health_status()
            
            # Get recent alerts
            recent_alerts = await self.get_alerts(resolved=False, limit=10)
            
            # Get performance metrics
            performance_metrics = await self._get_performance_metrics()
            
            # Get platform statistics
            platform_stats = await self._get_platform_statistics()
            
            # Get error analysis
            error_analysis = await self._get_error_analysis()
            
            return {
                "health_status": health_status,
                "recent_alerts": recent_alerts,
                "performance_metrics": performance_metrics,
                "platform_statistics": platform_stats,
                "error_analysis": error_analysis,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def register_alert_handler(
        self,
        alert_type: AlertType,
        handler: Callable[[Alert], None]
    ):
        """Register an alert handler"""
        self.alert_handlers[alert_type].append(handler)
    
    async def _collect_metrics_loop(self):
        """Background task to collect system metrics"""
        while True:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.monitoring_config["collection_interval"])
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {str(e)}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _health_check_loop(self):
        """Background task to perform health checks"""
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(60)  # Health checks every minute
            except Exception as e:
                logger.error(f"Error in health check loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def _alert_processing_loop(self):
        """Background task to process and manage alerts"""
        while True:
            try:
                await self._process_alerts()
                await asyncio.sleep(30)  # Process alerts every 30 seconds
            except Exception as e:
                logger.error(f"Error in alert processing loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def _cleanup_old_data_loop(self):
        """Background task to clean up old monitoring data"""
        while True:
            try:
                await self._cleanup_old_data()
                await asyncio.sleep(3600)  # Cleanup every hour
            except Exception as e:
                logger.error(f"Error in cleanup loop: {str(e)}")
                await asyncio.sleep(3600)
    
    async def _collect_system_metrics(self):
        """Collect system-wide metrics"""
        try:
            # Connection metrics
            total_connections = self.db.query(UserPlatformConnection).count()
            active_connections = self.db.query(UserPlatformConnection).filter(
                UserPlatformConnection.status == ConnectionStatus.ACTIVE
            ).count()
            
            await self.record_metric("connections.total", total_connections, MetricType.GAUGE)
            await self.record_metric("connections.active", active_connections, MetricType.GAUGE)
            
            # Error rate metrics
            recent_logs = self.db.query(ConnectionAuditLog).filter(
                ConnectionAuditLog.timestamp > datetime.utcnow() - timedelta(minutes=5)
            ).all()
            
            total_operations = len(recent_logs)
            error_operations = len([log for log in recent_logs if "error" in log.action.lower()])
            error_rate = error_operations / total_operations if total_operations > 0 else 0
            
            await self.record_metric("operations.total", total_operations, MetricType.COUNTER)
            await self.record_metric("operations.error_rate", error_rate, MetricType.GAUGE)
            
            # Platform-specific metrics
            platform_counts = self.db.query(
                UserPlatformConnection.platform_type,
                func.count(UserPlatformConnection.id)
            ).filter(
                UserPlatformConnection.status == ConnectionStatus.ACTIVE
            ).group_by(UserPlatformConnection.platform_type).all()
            
            for platform, count in platform_counts:
                await self.record_metric(
                    "connections.by_platform",
                    count,
                    MetricType.GAUGE,
                    tags={"platform": platform.value}
                )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
    
    async def _perform_health_checks(self):
        """Perform health checks on system components"""
        try:
            # Database health check
            start_time = datetime.utcnow()
            try:
                self.db.execute("SELECT 1")
                db_response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                db_status = "healthy"
                db_details = {"response_time_ms": db_response_time}
            except Exception as e:
                db_response_time = -1
                db_status = "unhealthy"
                db_details = {"error": str(e)}
            
            self.health_checks["database"] = HealthCheck(
                service="database",
                status=db_status,
                response_time=db_response_time,
                timestamp=datetime.utcnow(),
                details=db_details
            )
            
            # Integration service health check
            start_time = datetime.utcnow()
            try:
                health_result = await self.integration_service.get_health_status()
                integration_response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                integration_status = "healthy" if health_result.get("success") else "unhealthy"
                integration_details = health_result
            except Exception as e:
                integration_response_time = -1
                integration_status = "unhealthy"
                integration_details = {"error": str(e)}
            
            self.health_checks["integration_service"] = HealthCheck(
                service="integration_service",
                status=integration_status,
                response_time=integration_response_time,
                timestamp=datetime.utcnow(),
                details=integration_details
            )
            
            # Validation service health check
            start_time = datetime.utcnow()
            try:
                # Simple validation check
                validation_result = await self.validator.get_validation_summary()
                validation_response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                validation_status = "healthy" if validation_result.get("success") else "unhealthy"
                validation_details = validation_result
            except Exception as e:
                validation_response_time = -1
                validation_status = "unhealthy"
                validation_details = {"error": str(e)}
            
            self.health_checks["validation_service"] = HealthCheck(
                service="validation_service",
                status=validation_status,
                response_time=validation_response_time,
                timestamp=datetime.utcnow(),
                details=validation_details
            )
            
        except Exception as e:
            logger.error(f"Error performing health checks: {str(e)}")
    
    async def _process_alerts(self):
        """Process and manage alerts"""
        try:
            # Auto-resolve alerts that are no longer relevant
            for alert_id, alert in list(self.alerts.items()):
                if not alert.resolved:
                    should_resolve = await self._should_auto_resolve_alert(alert)
                    if should_resolve:
                        await self.resolve_alert(alert_id, "auto-resolved")
            
            # Check for new alert conditions
            await self._check_system_alerts()
            
        except Exception as e:
            logger.error(f"Error processing alerts: {str(e)}")
    
    async def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        try:
            cutoff_time = datetime.utcnow() - self.monitoring_config["retention_period"]
            
            # Clean up old metrics
            for metric_name in list(self.metrics.keys()):
                metric_queue = self.metrics[metric_name]
                # Remove old metrics
                while metric_queue and metric_queue[0].timestamp < cutoff_time:
                    metric_queue.popleft()
            
            # Clean up old resolved alerts
            old_alerts = [
                alert_id for alert_id, alert in self.alerts.items()
                if alert.resolved and alert.resolved_at and alert.resolved_at < cutoff_time
            ]
            
            for alert_id in old_alerts:
                del self.alerts[alert_id]
            
            logger.info(f"Cleaned up {len(old_alerts)} old alerts")
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")
    
    async def _check_metric_alerts(self, metric: Metric):
        """Check if a metric triggers any alerts"""
        try:
            thresholds = self.monitoring_config["alert_thresholds"]
            
            # Error rate alert
            if metric.name == "operations.error_rate" and metric.value > thresholds["error_rate"]:
                await self.create_alert(
                    AlertType.ERROR_RATE,
                    AlertSeverity.HIGH,
                    "High Error Rate Detected",
                    f"Error rate is {metric.value:.2%}, exceeding threshold of {thresholds['error_rate']:.2%}",
                    "monitoring_service",
                    {"metric_value": metric.value, "threshold": thresholds["error_rate"]}
                )
            
            # Response time alert
            if "response_time" in metric.name and metric.value > thresholds["response_time"]:
                await self.create_alert(
                    AlertType.PERFORMANCE,
                    AlertSeverity.MEDIUM,
                    "High Response Time",
                    f"Response time is {metric.value}ms, exceeding threshold of {thresholds['response_time']}ms",
                    "monitoring_service",
                    {"metric_value": metric.value, "threshold": thresholds["response_time"]}
                )
            
        except Exception as e:
            logger.error(f"Error checking metric alerts: {str(e)}")
    
    async def _check_system_alerts(self):
        """Check for system-wide alert conditions"""
        try:
            # Check for unhealthy services
            unhealthy_services = [
                service for service, check in self.health_checks.items()
                if check.status != "healthy"
            ]
            
            if unhealthy_services:
                await self.create_alert(
                    AlertType.CONNECTION_FAILURE,
                    AlertSeverity.HIGH,
                    "Unhealthy Services Detected",
                    f"Services in unhealthy state: {', '.join(unhealthy_services)}",
                    "monitoring_service",
                    {"unhealthy_services": unhealthy_services}
                )
            
        except Exception as e:
            logger.error(f"Error checking system alerts: {str(e)}")
    
    async def _should_auto_resolve_alert(self, alert: Alert) -> bool:
        """Check if an alert should be auto-resolved"""
        try:
            # Auto-resolve error rate alerts if error rate is back to normal
            if alert.alert_type == AlertType.ERROR_RATE:
                recent_error_rate = await self._get_recent_metric_value("operations.error_rate")
                if recent_error_rate is not None and recent_error_rate <= self.monitoring_config["alert_thresholds"]["error_rate"]:
                    return True
            
            # Auto-resolve performance alerts if response time is back to normal
            if alert.alert_type == AlertType.PERFORMANCE:
                # Check if the specific metric that triggered the alert is back to normal
                # This would require more sophisticated tracking
                pass
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking auto-resolve for alert {alert.alert_id}: {str(e)}")
            return False
    
    async def _get_recent_metric_value(self, metric_name: str) -> Optional[float]:
        """Get the most recent value for a metric"""
        try:
            if metric_name in self.metrics and self.metrics[metric_name]:
                return self.metrics[metric_name][-1].value
            return None
        except Exception as e:
            logger.error(f"Error getting recent metric value for {metric_name}: {str(e)}")
            return None
    
    async def _get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of recent metrics"""
        try:
            summary = {}
            
            for metric_name, metric_queue in self.metrics.items():
                if metric_queue:
                    recent_values = [m.value for m in list(metric_queue)[-10:]]  # Last 10 values
                    summary[metric_name] = {
                        "current": recent_values[-1],
                        "average": statistics.mean(recent_values),
                        "min": min(recent_values),
                        "max": max(recent_values),
                        "count": len(recent_values)
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting metrics summary: {str(e)}")
            return {}
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for dashboard"""
        try:
            # Get response time metrics
            response_time_metrics = []
            for service, check in self.health_checks.items():
                if check.response_time > 0:
                    response_time_metrics.append({
                        "service": service,
                        "response_time": check.response_time,
                        "timestamp": check.timestamp.isoformat()
                    })
            
            # Get error rate trend
            error_rate_history = []
            if "operations.error_rate" in self.metrics:
                for metric in list(self.metrics["operations.error_rate"])[-20:]:  # Last 20 values
                    error_rate_history.append({
                        "value": metric.value,
                        "timestamp": metric.timestamp.isoformat()
                    })
            
            return {
                "response_times": response_time_metrics,
                "error_rate_history": error_rate_history
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return {}
    
    async def _get_platform_statistics(self) -> Dict[str, Any]:
        """Get platform-specific statistics"""
        try:
            # Get connection counts by platform
            platform_stats = {}
            
            if "connections.by_platform" in self.metrics:
                for metric in self.metrics["connections.by_platform"]:
                    platform = metric.tags.get("platform", "unknown")
                    if platform not in platform_stats:
                        platform_stats[platform] = []
                    platform_stats[platform].append({
                        "count": metric.value,
                        "timestamp": metric.timestamp.isoformat()
                    })
            
            return platform_stats
            
        except Exception as e:
            logger.error(f"Error getting platform statistics: {str(e)}")
            return {}
    
    async def _get_error_analysis(self) -> Dict[str, Any]:
        """Get error analysis data"""
        try:
            # Get recent error logs
            recent_errors = self.db.query(ConnectionAuditLog).filter(
                and_(
                    ConnectionAuditLog.timestamp > datetime.utcnow() - timedelta(hours=24),
                    ConnectionAuditLog.action.contains("error")
                )
            ).order_by(desc(ConnectionAuditLog.timestamp)).limit(50).all()
            
            # Analyze error patterns
            error_types = defaultdict(int)
            error_platforms = defaultdict(int)
            
            for error in recent_errors:
                # Extract error type from action
                error_types[error.action] += 1
                
                # Extract platform from details if available
                try:
                    details = json.loads(error.details) if error.details else {}
                    platform = details.get("platform_type", "unknown")
                    error_platforms[platform] += 1
                except:
                    error_platforms["unknown"] += 1
            
            return {
                "total_errors": len(recent_errors),
                "error_types": dict(error_types),
                "error_by_platform": dict(error_platforms),
                "recent_errors": [
                    {
                        "timestamp": error.timestamp.isoformat(),
                        "action": error.action,
                        "user_id": error.user_id
                    }
                    for error in recent_errors[:10]  # Last 10 errors
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting error analysis: {str(e)}")
            return {}
    
    async def _trigger_alert_handlers(self, alert: Alert):
        """Trigger registered alert handlers"""
        try:
            handlers = self.alert_handlers.get(alert.alert_type, [])
            for handler in handlers:
                try:
                    await handler(alert) if asyncio.iscoroutinefunction(handler) else handler(alert)
                except Exception as e:
                    logger.error(f"Error in alert handler: {str(e)}")
        except Exception as e:
            logger.error(f"Error triggering alert handlers: {str(e)}")