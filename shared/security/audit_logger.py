"""
Security Audit Logger for LiftOS
Comprehensive audit trail for all security events following SOC 2 compliance standards
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from enum import Enum

from ..database.models import SecurityAuditLog

logger = logging.getLogger(__name__)

class SecurityEventType(Enum):
    """Security event types for audit logging"""
    
    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    TOKEN_REFRESH = "token_refresh"
    SESSION_EXPIRED = "session_expired"
    
    # API Key events
    API_KEY_CREATED = "api_key_created"
    API_KEY_ACCESSED = "api_key_accessed"
    API_KEY_ROTATED = "api_key_rotated"
    API_KEY_REVOKED = "api_key_revoked"
    API_KEY_FAILED_ACCESS = "api_key_failed_access"
    
    # Authorization events
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_DENIED = "permission_denied"
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REVOKED = "role_revoked"
    
    # Security violations
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    IP_BLOCKED = "ip_blocked"
    BRUTE_FORCE_ATTEMPT = "brute_force_attempt"
    
    # Data access events
    SENSITIVE_DATA_ACCESS = "sensitive_data_access"
    DATA_EXPORT = "data_export"
    CONFIGURATION_CHANGE = "configuration_change"
    
    # System events
    SECURITY_ALERT = "security_alert"
    COMPLIANCE_VIOLATION = "compliance_violation"

class SecurityAuditLogger:
    """
    Enterprise security audit logger with SOC 2 compliance features
    """
    
    def __init__(self):
        self.logger = logging.getLogger("security_audit")
    
    async def log_security_event(
        self,
        session: AsyncSession,
        event_type: SecurityEventType,
        user_id: Optional[str] = None,
        org_id: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None,
        risk_level: str = "low"
    ) -> str:
        """
        Log a security event with comprehensive details
        Returns the audit log ID
        """
        try:
            audit_entry = SecurityAuditLog(
                event_type=event_type.value,
                user_id=user_id,
                org_id=org_id,
                resource=resource,
                action=action,
                ip_address=ip_address,
                user_agent=user_agent,
                success=success,
                details=details or {},
                risk_level=risk_level,
                timestamp=datetime.now(timezone.utc)
            )
            
            session.add(audit_entry)
            await session.flush()
            
            audit_id = str(audit_entry.id)
            
            # Also log to application logger for immediate visibility
            log_level = logging.WARNING if not success or risk_level == "high" else logging.INFO
            self.logger.log(
                log_level,
                f"Security Event: {event_type.value} | User: {user_id} | Success: {success} | Details: {details}"
            )
            
            return audit_id
            
        except Exception as e:
            logger.error(f"Failed to log security event: {e}")
            raise
    
    # Authentication event loggers
    async def log_login_success(
        self,
        session: AsyncSession,
        user_id: str,
        org_id: str,
        ip_address: str,
        user_agent: str,
        method: str = "password"
    ) -> str:
        """Log successful login"""
        return await self.log_security_event(
            session=session,
            event_type=SecurityEventType.LOGIN_SUCCESS,
            user_id=user_id,
            org_id=org_id,
            action="login",
            ip_address=ip_address,
            user_agent=user_agent,
            success=True,
            details={"method": method},
            risk_level="low"
        )
    
    async def log_login_failure(
        self,
        session: AsyncSession,
        email: str,
        ip_address: str,
        user_agent: str,
        reason: str,
        attempt_count: int = 1
    ) -> str:
        """Log failed login attempt"""
        risk_level = "high" if attempt_count >= 3 else "medium"
        return await self.log_security_event(
            session=session,
            event_type=SecurityEventType.LOGIN_FAILURE,
            action="login",
            ip_address=ip_address,
            user_agent=user_agent,
            success=False,
            details={
                "email": email,
                "reason": reason,
                "attempt_count": attempt_count
            },
            risk_level=risk_level
        )
    
    async def log_logout(
        self,
        session: AsyncSession,
        user_id: str,
        org_id: str,
        ip_address: str,
        session_duration_minutes: int
    ) -> str:
        """Log user logout"""
        return await self.log_security_event(
            session=session,
            event_type=SecurityEventType.LOGOUT,
            user_id=user_id,
            org_id=org_id,
            action="logout",
            ip_address=ip_address,
            success=True,
            details={"session_duration_minutes": session_duration_minutes},
            risk_level="low"
        )
    
    # API Key event loggers
    async def log_api_key_created(
        self,
        session: AsyncSession,
        user_id: str,
        org_id: str,
        provider: str,
        key_id: str
    ) -> str:
        """Log API key creation"""
        return await self.log_security_event(
            session=session,
            event_type=SecurityEventType.API_KEY_CREATED,
            user_id=user_id,
            org_id=org_id,
            resource=f"api_key:{provider}",
            action="create",
            success=True,
            details={
                "provider": provider,
                "key_id": key_id
            },
            risk_level="medium"
        )
    
    async def log_api_key_accessed(
        self,
        session: AsyncSession,
        user_id: str,
        org_id: str,
        provider: str,
        key_id: str
    ) -> str:
        """Log API key access"""
        return await self.log_security_event(
            session=session,
            event_type=SecurityEventType.API_KEY_ACCESSED,
            user_id=user_id,
            org_id=org_id,
            resource=f"api_key:{provider}",
            action="access",
            success=True,
            details={
                "provider": provider,
                "key_id": key_id
            },
            risk_level="low"
        )
    
    async def log_api_key_rotated(
        self,
        session: AsyncSession,
        user_id: str,
        org_id: str,
        provider: str,
        old_key_id: str,
        new_key_id: str
    ) -> str:
        """Log API key rotation"""
        return await self.log_security_event(
            session=session,
            event_type=SecurityEventType.API_KEY_ROTATED,
            user_id=user_id,
            org_id=org_id,
            resource=f"api_key:{provider}",
            action="rotate",
            success=True,
            details={
                "provider": provider,
                "old_key_id": old_key_id,
                "new_key_id": new_key_id
            },
            risk_level="medium"
        )
    
    async def log_api_key_revoked(
        self,
        session: AsyncSession,
        user_id: str,
        org_id: str,
        provider: str,
        reason: str
    ) -> str:
        """Log API key revocation"""
        return await self.log_security_event(
            session=session,
            event_type=SecurityEventType.API_KEY_REVOKED,
            user_id=user_id,
            org_id=org_id,
            resource=f"api_key:{provider}",
            action="revoke",
            success=True,
            details={
                "provider": provider,
                "reason": reason
            },
            risk_level="high"
        )
    
    # Security violation loggers
    async def log_rate_limit_exceeded(
        self,
        session: AsyncSession,
        ip_address: str,
        endpoint: str,
        request_count: int,
        time_window: int
    ) -> str:
        """Log rate limit violation"""
        return await self.log_security_event(
            session=session,
            event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
            action="rate_limit_violation",
            ip_address=ip_address,
            success=False,
            details={
                "endpoint": endpoint,
                "request_count": request_count,
                "time_window_seconds": time_window
            },
            risk_level="medium"
        )
    
    async def log_suspicious_activity(
        self,
        session: AsyncSession,
        user_id: Optional[str],
        ip_address: str,
        activity_type: str,
        details: Dict[str, Any]
    ) -> str:
        """Log suspicious activity"""
        return await self.log_security_event(
            session=session,
            event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
            user_id=user_id,
            action=activity_type,
            ip_address=ip_address,
            success=False,
            details=details,
            risk_level="high"
        )
    
    # Authorization event loggers
    async def log_permission_denied(
        self,
        session: AsyncSession,
        user_id: str,
        org_id: str,
        resource: str,
        required_permission: str,
        ip_address: str
    ) -> str:
        """Log permission denied event"""
        return await self.log_security_event(
            session=session,
            event_type=SecurityEventType.PERMISSION_DENIED,
            user_id=user_id,
            org_id=org_id,
            resource=resource,
            action="access_denied",
            ip_address=ip_address,
            success=False,
            details={"required_permission": required_permission},
            risk_level="medium"
        )
    
    # Query methods for audit analysis
    async def get_security_events(
        self,
        session: AsyncSession,
        user_id: Optional[str] = None,
        org_id: Optional[str] = None,
        event_types: Optional[List[SecurityEventType]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        risk_levels: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Query security events with filters"""
        try:
            query = select(SecurityAuditLog)
            
            # Apply filters
            if user_id:
                query = query.where(SecurityAuditLog.user_id == user_id)
            if org_id:
                query = query.where(SecurityAuditLog.org_id == org_id)
            if event_types:
                event_type_values = [et.value for et in event_types]
                query = query.where(SecurityAuditLog.event_type.in_(event_type_values))
            if start_date:
                query = query.where(SecurityAuditLog.timestamp >= start_date)
            if end_date:
                query = query.where(SecurityAuditLog.timestamp <= end_date)
            if risk_levels:
                query = query.where(SecurityAuditLog.risk_level.in_(risk_levels))
            
            query = query.order_by(SecurityAuditLog.timestamp.desc()).limit(limit)
            
            result = await session.execute(query)
            events = result.scalars().all()
            
            return [
                {
                    'id': str(event.id),
                    'event_type': event.event_type,
                    'user_id': event.user_id,
                    'org_id': event.org_id,
                    'resource': event.resource,
                    'action': event.action,
                    'ip_address': event.ip_address,
                    'user_agent': event.user_agent,
                    'success': event.success,
                    'details': event.details,
                    'risk_level': event.risk_level,
                    'timestamp': event.timestamp.isoformat()
                }
                for event in events
            ]
            
        except Exception as e:
            logger.error(f"Failed to query security events: {e}")
            raise
    
    async def get_security_summary(
        self,
        session: AsyncSession,
        org_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get security summary for an organization"""
        try:
            start_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            # Get all events for the period
            events = await self.get_security_events(
                session=session,
                org_id=org_id,
                start_date=start_date,
                limit=10000  # High limit for summary
            )
            
            # Calculate summary statistics
            total_events = len(events)
            failed_events = len([e for e in events if not e['success']])
            high_risk_events = len([e for e in events if e['risk_level'] == 'high'])
            
            # Count by event type
            event_type_counts = {}
            for event in events:
                event_type = event['event_type']
                event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
            
            # Recent high-risk events
            recent_high_risk = [
                e for e in events[:10] 
                if e['risk_level'] == 'high'
            ]
            
            return {
                'period_days': days,
                'total_events': total_events,
                'failed_events': failed_events,
                'high_risk_events': high_risk_events,
                'success_rate': (total_events - failed_events) / total_events if total_events > 0 else 1.0,
                'event_type_counts': event_type_counts,
                'recent_high_risk_events': recent_high_risk
            }
            
        except Exception as e:
            logger.error(f"Failed to generate security summary: {e}")
            raise

# Global audit logger instance
_security_audit_logger: Optional[SecurityAuditLogger] = None

def get_security_audit_logger() -> SecurityAuditLogger:
    """Get the global security audit logger instance"""
    global _security_audit_logger
    if _security_audit_logger is None:
        _security_audit_logger = SecurityAuditLogger()
    return _security_audit_logger