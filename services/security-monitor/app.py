"""
LiftOS Security Monitoring Service
Real-time security monitoring, alerting, and threat detection
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Set
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import json
import sys
import os
from collections import defaultdict, deque
import logging

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.security.audit_logger import SecurityAuditLogger, SecurityEventType
from shared.security.enhanced_jwt import get_enhanced_jwt_manager
from shared.database.database import get_async_session
from shared.database.security_models import SecurityAuditLog, EnhancedUserSession
from shared.utils.config import get_service_config
from shared.utils.logging import setup_logging
from shared.models.base import APIResponse
from sqlalchemy import select, func, and_, or_

# Service configuration
config = get_service_config("security_monitor", 8007)
logger = setup_logging("security_monitor")

# FastAPI app
app = FastAPI(
    title="LiftOS Security Monitoring Service",
    description="Real-time security monitoring, alerting, and threat detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if config.DEBUG else ["http://localhost:3000", "http://localhost:8501", "http://localhost:8502"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class SecurityAlert(BaseModel):
    """Security alert model"""
    id: str
    severity: str  # low, medium, high, critical
    alert_type: str
    title: str
    description: str
    timestamp: datetime
    org_id: Optional[str] = None
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    details: Dict[str, Any] = {}
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None

class ThreatIndicator(BaseModel):
    """Threat indicator model"""
    indicator_type: str  # ip, user_agent, pattern
    value: str
    severity: str
    description: str
    first_seen: datetime
    last_seen: datetime
    count: int = 1

class SecurityMetrics(BaseModel):
    """Security metrics model"""
    timestamp: datetime
    total_events: int
    failed_logins: int
    successful_logins: int
    api_key_accesses: int
    security_violations: int
    unique_ips: int
    unique_users: int
    risk_score_avg: float
    alerts_generated: int

# Global state
class SecurityMonitor:
    """Security monitoring and alerting system"""
    
    def __init__(self):
        self.audit_logger = SecurityAuditLogger()
        self.jwt_manager = get_enhanced_jwt_manager()
        
        # Alert storage (in production, use Redis or database)
        self.alerts: Dict[str, SecurityAlert] = {}
        self.threat_indicators: Dict[str, ThreatIndicator] = {}
        
        # Real-time monitoring
        self.active_connections: Set[WebSocket] = set()
        self.event_buffer: deque = deque(maxlen=1000)
        self.metrics_history: deque = deque(maxlen=100)
        
        # Threat detection patterns
        self.failed_login_threshold = 5  # per 5 minutes
        self.api_access_threshold = 1000  # per hour
        self.risk_score_threshold = 0.8
        
        # Tracking windows
        self.failed_logins: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.api_accesses: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Start background monitoring
        asyncio.create_task(self.start_monitoring())
    
    async def start_monitoring(self):
        """Start background monitoring tasks"""
        while True:
            try:
                await self.collect_security_metrics()
                await self.analyze_threats()
                await self.cleanup_old_data()
                await asyncio.sleep(60)  # Run every minute
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def collect_security_metrics(self):
        """Collect security metrics from audit logs"""
        try:
            async with get_async_session() as session:
                now = datetime.now(timezone.utc)
                hour_ago = now - timedelta(hours=1)
                
                # Get recent events
                result = await session.execute(
                    select(SecurityAuditLog)
                    .where(SecurityAuditLog.timestamp > hour_ago)
                    .order_by(SecurityAuditLog.timestamp.desc())
                )
                recent_events = result.scalars().all()
                
                # Calculate metrics
                total_events = len(recent_events)
                failed_logins = len([e for e in recent_events if e.event_type == "LOGIN_FAILED"])
                successful_logins = len([e for e in recent_events if e.event_type == "LOGIN_SUCCESS"])
                api_key_accesses = len([e for e in recent_events if e.event_type == "API_KEY_ACCESS"])
                security_violations = len([e for e in recent_events if e.event_type == "SECURITY_VIOLATION"])
                
                unique_ips = len(set(e.ip_address for e in recent_events if e.ip_address))
                unique_users = len(set(e.user_id for e in recent_events if e.user_id))
                
                risk_scores = [e.risk_score for e in recent_events if e.risk_score is not None]
                risk_score_avg = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0
                
                metrics = SecurityMetrics(
                    timestamp=now,
                    total_events=total_events,
                    failed_logins=failed_logins,
                    successful_logins=successful_logins,
                    api_key_accesses=api_key_accesses,
                    security_violations=security_violations,
                    unique_ips=unique_ips,
                    unique_users=unique_users,
                    risk_score_avg=risk_score_avg,
                    alerts_generated=len([a for a in self.alerts.values() if a.timestamp > hour_ago])
                )
                
                self.metrics_history.append(metrics)
                
                # Broadcast metrics to connected clients
                await self.broadcast_metrics(metrics)
                
        except Exception as e:
            logger.error(f"Failed to collect security metrics: {e}")
    
    async def analyze_threats(self):
        """Analyze recent events for threat patterns"""
        try:
            async with get_async_session() as session:
                now = datetime.now(timezone.utc)
                
                # Analyze failed login patterns
                await self.analyze_failed_logins(session, now)
                
                # Analyze API access patterns
                await self.analyze_api_access_patterns(session, now)
                
                # Analyze high-risk events
                await self.analyze_high_risk_events(session, now)
                
                # Analyze suspicious IP patterns
                await self.analyze_suspicious_ips(session, now)
                
        except Exception as e:
            logger.error(f"Failed to analyze threats: {e}")
    
    async def analyze_failed_logins(self, session, now: datetime):
        """Analyze failed login patterns"""
        try:
            five_min_ago = now - timedelta(minutes=5)
            
            # Get recent failed logins grouped by IP
            result = await session.execute(
                select(SecurityAuditLog.ip_address, func.count(SecurityAuditLog.id))
                .where(
                    and_(
                        SecurityAuditLog.event_type == "LOGIN_FAILED",
                        SecurityAuditLog.timestamp > five_min_ago
                    )
                )
                .group_by(SecurityAuditLog.ip_address)
                .having(func.count(SecurityAuditLog.id) >= self.failed_login_threshold)
            )
            
            suspicious_ips = result.fetchall()
            
            for ip, count in suspicious_ips:
                if ip:
                    await self.create_alert(
                        alert_type="brute_force_attempt",
                        severity="high",
                        title=f"Brute Force Attack Detected",
                        description=f"IP {ip} has {count} failed login attempts in 5 minutes",
                        details={"ip_address": ip, "failed_attempts": count, "timeframe": "5_minutes"}
                    )
                    
                    # Add to threat indicators
                    self.threat_indicators[f"ip_{ip}"] = ThreatIndicator(
                        indicator_type="ip",
                        value=ip,
                        severity="high",
                        description=f"Brute force attempts: {count}",
                        first_seen=now,
                        last_seen=now,
                        count=count
                    )
                    
        except Exception as e:
            logger.error(f"Failed to analyze failed logins: {e}")
    
    async def analyze_api_access_patterns(self, session, now: datetime):
        """Analyze API access patterns for anomalies"""
        try:
            hour_ago = now - timedelta(hours=1)
            
            # Get API access counts by user
            result = await session.execute(
                select(SecurityAuditLog.user_id, func.count(SecurityAuditLog.id))
                .where(
                    and_(
                        SecurityAuditLog.event_type == "API_ACCESS",
                        SecurityAuditLog.timestamp > hour_ago
                    )
                )
                .group_by(SecurityAuditLog.user_id)
                .having(func.count(SecurityAuditLog.id) >= self.api_access_threshold)
            )
            
            high_usage_users = result.fetchall()
            
            for user_id, count in high_usage_users:
                if user_id:
                    await self.create_alert(
                        alert_type="unusual_api_usage",
                        severity="medium",
                        title=f"Unusual API Usage Pattern",
                        description=f"User {user_id} has made {count} API calls in 1 hour",
                        user_id=user_id,
                        details={"user_id": user_id, "api_calls": count, "timeframe": "1_hour"}
                    )
                    
        except Exception as e:
            logger.error(f"Failed to analyze API access patterns: {e}")
    
    async def analyze_high_risk_events(self, session, now: datetime):
        """Analyze high-risk security events"""
        try:
            hour_ago = now - timedelta(hours=1)
            
            # Get high-risk events
            result = await session.execute(
                select(SecurityAuditLog)
                .where(
                    and_(
                        SecurityAuditLog.risk_score >= self.risk_score_threshold,
                        SecurityAuditLog.timestamp > hour_ago
                    )
                )
                .order_by(SecurityAuditLog.risk_score.desc())
            )
            
            high_risk_events = result.scalars().all()
            
            for event in high_risk_events:
                await self.create_alert(
                    alert_type="high_risk_event",
                    severity="critical" if event.risk_score >= 0.9 else "high",
                    title=f"High Risk Security Event",
                    description=f"Event with risk score {event.risk_score:.2f}: {event.action}",
                    user_id=event.user_id,
                    org_id=event.org_id,
                    ip_address=event.ip_address,
                    details={
                        "event_type": event.event_type,
                        "action": event.action,
                        "risk_score": event.risk_score,
                        "details": event.details
                    }
                )
                
        except Exception as e:
            logger.error(f"Failed to analyze high-risk events: {e}")
    
    async def analyze_suspicious_ips(self, session, now: datetime):
        """Analyze suspicious IP patterns"""
        try:
            day_ago = now - timedelta(days=1)
            
            # Get IPs with multiple event types (potential reconnaissance)
            result = await session.execute(
                select(
                    SecurityAuditLog.ip_address,
                    func.count(func.distinct(SecurityAuditLog.event_type)).label('event_types'),
                    func.count(SecurityAuditLog.id).label('total_events')
                )
                .where(SecurityAuditLog.timestamp > day_ago)
                .group_by(SecurityAuditLog.ip_address)
                .having(func.count(func.distinct(SecurityAuditLog.event_type)) >= 3)
            )
            
            suspicious_ips = result.fetchall()
            
            for ip, event_types, total_events in suspicious_ips:
                if ip and total_events > 10:  # Minimum threshold
                    await self.create_alert(
                        alert_type="reconnaissance_activity",
                        severity="medium",
                        title=f"Potential Reconnaissance Activity",
                        description=f"IP {ip} has {event_types} different event types with {total_events} total events",
                        ip_address=ip,
                        details={
                            "ip_address": ip,
                            "event_types": event_types,
                            "total_events": total_events,
                            "timeframe": "24_hours"
                        }
                    )
                    
        except Exception as e:
            logger.error(f"Failed to analyze suspicious IPs: {e}")
    
    async def create_alert(
        self,
        alert_type: str,
        severity: str,
        title: str,
        description: str,
        org_id: Optional[str] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Dict[str, Any] = None
    ):
        """Create a new security alert"""
        try:
            alert_id = f"{alert_type}_{int(time.time())}_{hash(description) % 10000}"
            
            alert = SecurityAlert(
                id=alert_id,
                severity=severity,
                alert_type=alert_type,
                title=title,
                description=description,
                timestamp=datetime.now(timezone.utc),
                org_id=org_id,
                user_id=user_id,
                ip_address=ip_address,
                details=details or {}
            )
            
            self.alerts[alert_id] = alert
            
            # Broadcast alert to connected clients
            await self.broadcast_alert(alert)
            
            logger.warning(f"Security alert created: {title} - {description}")
            
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
    
    async def broadcast_alert(self, alert: SecurityAlert):
        """Broadcast alert to all connected WebSocket clients"""
        if self.active_connections:
            message = {
                "type": "alert",
                "data": alert.dict()
            }
            
            disconnected = set()
            for connection in self.active_connections:
                try:
                    await connection.send_text(json.dumps(message, default=str))
                except:
                    disconnected.add(connection)
            
            # Remove disconnected clients
            self.active_connections -= disconnected
    
    async def broadcast_metrics(self, metrics: SecurityMetrics):
        """Broadcast metrics to all connected WebSocket clients"""
        if self.active_connections:
            message = {
                "type": "metrics",
                "data": metrics.dict()
            }
            
            disconnected = set()
            for connection in self.active_connections:
                try:
                    await connection.send_text(json.dumps(message, default=str))
                except:
                    disconnected.add(connection)
            
            # Remove disconnected clients
            self.active_connections -= disconnected
    
    async def cleanup_old_data(self):
        """Clean up old alerts and metrics"""
        try:
            now = datetime.now(timezone.utc)
            week_ago = now - timedelta(days=7)
            
            # Remove old resolved alerts
            old_alerts = [
                alert_id for alert_id, alert in self.alerts.items()
                if alert.resolved and alert.resolved_at and alert.resolved_at < week_ago
            ]
            
            for alert_id in old_alerts:
                del self.alerts[alert_id]
            
            logger.info(f"Cleaned up {len(old_alerts)} old alerts")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")

# Global security monitor instance
security_monitor = SecurityMonitor()

# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "security-monitor", "timestamp": datetime.now(timezone.utc)}

@app.get("/alerts", response_model=List[SecurityAlert])
async def get_alerts(
    severity: Optional[str] = None,
    alert_type: Optional[str] = None,
    resolved: Optional[bool] = None,
    limit: int = 100
):
    """Get security alerts with optional filtering"""
    alerts = list(security_monitor.alerts.values())
    
    # Apply filters
    if severity:
        alerts = [a for a in alerts if a.severity == severity]
    if alert_type:
        alerts = [a for a in alerts if a.alert_type == alert_type]
    if resolved is not None:
        alerts = [a for a in alerts if a.resolved == resolved]
    
    # Sort by timestamp (newest first) and limit
    alerts.sort(key=lambda x: x.timestamp, reverse=True)
    return alerts[:limit]

@app.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str, resolved_by: str):
    """Resolve a security alert"""
    if alert_id not in security_monitor.alerts:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    alert = security_monitor.alerts[alert_id]
    alert.resolved = True
    alert.resolved_at = datetime.now(timezone.utc)
    alert.resolved_by = resolved_by
    
    return APIResponse(
        success=True,
        message=f"Alert {alert_id} resolved",
        data=alert.dict()
    )

@app.get("/metrics", response_model=List[SecurityMetrics])
async def get_metrics(hours: int = 24):
    """Get security metrics history"""
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
    
    filtered_metrics = [
        m for m in security_monitor.metrics_history
        if m.timestamp > cutoff_time
    ]
    
    return sorted(filtered_metrics, key=lambda x: x.timestamp)

@app.get("/threats", response_model=List[ThreatIndicator])
async def get_threat_indicators():
    """Get current threat indicators"""
    return list(security_monitor.threat_indicators.values())

@app.get("/dashboard/summary")
async def get_dashboard_summary():
    """Get security dashboard summary"""
    now = datetime.now(timezone.utc)
    
    # Get recent metrics
    recent_metrics = security_monitor.metrics_history[-1] if security_monitor.metrics_history else None
    
    # Count alerts by severity
    alert_counts = defaultdict(int)
    unresolved_alerts = [a for a in security_monitor.alerts.values() if not a.resolved]
    
    for alert in unresolved_alerts:
        alert_counts[alert.severity] += 1
    
    # Get threat indicator counts
    threat_counts = defaultdict(int)
    for indicator in security_monitor.threat_indicators.values():
        threat_counts[indicator.severity] += 1
    
    return {
        "timestamp": now,
        "alerts": {
            "total_unresolved": len(unresolved_alerts),
            "critical": alert_counts["critical"],
            "high": alert_counts["high"],
            "medium": alert_counts["medium"],
            "low": alert_counts["low"]
        },
        "threats": {
            "total_indicators": len(security_monitor.threat_indicators),
            "critical": threat_counts["critical"],
            "high": threat_counts["high"],
            "medium": threat_counts["medium"],
            "low": threat_counts["low"]
        },
        "metrics": recent_metrics.dict() if recent_metrics else None,
        "system_status": "operational"
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time security monitoring"""
    await websocket.accept()
    security_monitor.active_connections.add(websocket)
    
    try:
        # Send initial data
        summary = await get_dashboard_summary()
        await websocket.send_text(json.dumps({
            "type": "summary",
            "data": summary
        }, default=str))
        
        # Keep connection alive
        while True:
            try:
                # Wait for client messages (ping/pong)
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_text(json.dumps({"type": "ping"}))
            
    except WebSocketDisconnect:
        security_monitor.active_connections.discard(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        security_monitor.active_connections.discard(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8007,
        reload=config.DEBUG,
        log_level="info"
    )