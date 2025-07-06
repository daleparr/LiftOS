"""
Observability Integration for Agentic Microservice

This module integrates the Agentic microservice with LiftOS's observability
framework to provide comprehensive monitoring and tracing of agent operations.
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from contextlib import asynccontextmanager
from functools import wraps

# Import LiftOS shared components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from shared.mmm_spine_integration.observability import (
    ObservabilityManager, TraceLevel, TraceSpan
)
from shared.utils.logging import get_logger

logger = get_logger(__name__)


class AgenticObservabilityManager:
    """
    Specialized observability manager for agent operations with
    agent-specific metrics and tracing capabilities.
    """
    
    def __init__(self):
        self.observability_manager = ObservabilityManager()
        self.agent_metrics: Dict[str, Dict[str, Any]] = {}
        self.active_agent_spans: Dict[str, str] = {}  # agent_id -> span_id
        
    async def initialize(self) -> None:
        """Initialize the observability manager."""
        await self.observability_manager.initialize()
        logger.info("Agentic observability manager initialized")
    
    @asynccontextmanager
    async def trace_agent_operation(
        self,
        agent_id: str,
        operation_name: str,
        level: TraceLevel = TraceLevel.INFO,
        tags: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for tracing agent operations with automatic span management.
        
        Args:
            agent_id: ID of the agent performing the operation
            operation_name: Name of the operation being traced
            level: Trace level for filtering
            tags: Additional tags for the span
        """
        span_id = str(uuid.uuid4())
        trace_id = self.active_agent_spans.get(agent_id, str(uuid.uuid4()))
        
        if tags is None:
            tags = {}
        
        tags.update({
            "agent_id": agent_id,
            "service": "agentic",
            "operation_type": "agent_operation"
        })
        
        span = TraceSpan(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=self.active_agent_spans.get(agent_id),
            operation_name=operation_name,
            service_name="agentic",
            start_time_ns=time.time_ns(),
            tags=tags,
            level=level
        )
        
        self.active_agent_spans[agent_id] = span_id
        
        try:
            await self.observability_manager.start_span(span)
            yield span
            
            # Mark span as successful
            span.status = "success"
            
        except Exception as e:
            # Mark span as failed and capture error
            span.status = "error"
            span.error = str(e)
            span.tags["error_type"] = type(e).__name__
            
            # Log error details
            span.logs.append({
                "timestamp": time.time_ns(),
                "level": "error",
                "message": f"Agent operation failed: {str(e)}",
                "error_details": {
                    "exception_type": type(e).__name__,
                    "exception_message": str(e)
                }
            })
            
            raise
            
        finally:
            # Finalize span
            span.end_time_ns = time.time_ns()
            span.duration_ns = span.end_time_ns - span.start_time_ns
            
            await self.observability_manager.finish_span(span)
            
            # Update agent metrics
            await self._update_agent_metrics(agent_id, span)
            
            # Remove from active spans if this was the root span
            if self.active_agent_spans.get(agent_id) == span_id:
                del self.active_agent_spans[agent_id]
    
    async def _update_agent_metrics(self, agent_id: str, span: TraceSpan) -> None:
        """Update metrics for an agent based on completed span."""
        if agent_id not in self.agent_metrics:
            self.agent_metrics[agent_id] = {
                "total_operations": 0,
                "successful_operations": 0,
                "failed_operations": 0,
                "total_duration_ns": 0,
                "average_duration_ns": 0,
                "last_operation_time": None,
                "operation_counts": {},
                "error_counts": {}
            }
        
        metrics = self.agent_metrics[agent_id]
        
        # Update basic counters
        metrics["total_operations"] += 1
        metrics["total_duration_ns"] += span.duration_ns or 0
        metrics["last_operation_time"] = datetime.now().isoformat()
        
        # Update success/failure counts
        if span.status == "success":
            metrics["successful_operations"] += 1
        else:
            metrics["failed_operations"] += 1
            
            # Track error types
            error_type = span.tags.get("error_type", "unknown")
            metrics["error_counts"][error_type] = metrics["error_counts"].get(error_type, 0) + 1
        
        # Update operation type counts
        operation_name = span.operation_name
        metrics["operation_counts"][operation_name] = metrics["operation_counts"].get(operation_name, 0) + 1
        
        # Update average duration
        metrics["average_duration_ns"] = metrics["total_duration_ns"] / metrics["total_operations"]
        
        # Calculate success rate
        metrics["success_rate"] = metrics["successful_operations"] / metrics["total_operations"]
    
    async def log_agent_event(
        self,
        agent_id: str,
        event_type: str,
        message: str,
        level: TraceLevel = TraceLevel.INFO,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an agent-specific event.
        
        Args:
            agent_id: ID of the agent
            event_type: Type of event (e.g., "decision", "error", "performance")
            message: Event message
            level: Log level
            metadata: Additional metadata
        """
        if metadata is None:
            metadata = {}
        
        log_entry = {
            "timestamp": time.time_ns(),
            "agent_id": agent_id,
            "event_type": event_type,
            "message": message,
            "level": level.name,
            "metadata": metadata
        }
        
        # Add to current span if one exists
        current_span_id = self.active_agent_spans.get(agent_id)
        if current_span_id:
            # This would add to the current span's logs
            pass
        
        logger.info(f"Agent event [{agent_id}] {event_type}: {message}")
    
    async def get_agent_metrics(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific agent."""
        return self.agent_metrics.get(agent_id)
    
    async def get_all_agent_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all agents."""
        return self.agent_metrics.copy()
    
    async def generate_performance_report(self, agent_id: str) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Performance report
        """
        metrics = self.agent_metrics.get(agent_id)
        if not metrics:
            return {"error": "No metrics found for agent"}
        
        # Calculate performance indicators
        avg_duration_ms = (metrics["average_duration_ns"] / 1_000_000) if metrics["average_duration_ns"] else 0
        
        report = {
            "agent_id": agent_id,
            "report_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_operations": metrics["total_operations"],
                "success_rate": metrics["success_rate"],
                "average_duration_ms": avg_duration_ms,
                "last_operation": metrics["last_operation_time"]
            },
            "performance_metrics": {
                "operations_per_minute": self._calculate_operations_per_minute(agent_id),
                "error_rate": metrics["failed_operations"] / metrics["total_operations"] if metrics["total_operations"] > 0 else 0,
                "most_common_operation": max(metrics["operation_counts"], key=metrics["operation_counts"].get) if metrics["operation_counts"] else None,
                "most_common_error": max(metrics["error_counts"], key=metrics["error_counts"].get) if metrics["error_counts"] else None
            },
            "detailed_metrics": metrics
        }
        
        return report
    
    def _calculate_operations_per_minute(self, agent_id: str) -> float:
        """Calculate operations per minute for an agent (simplified)."""
        metrics = self.agent_metrics.get(agent_id, {})
        total_ops = metrics.get("total_operations", 0)
        
        # Simplified calculation - in production would use actual time windows
        return total_ops / 60.0 if total_ops > 0 else 0.0


def trace_agent_method(
    operation_name: Optional[str] = None,
    level: TraceLevel = TraceLevel.INFO,
    capture_args: bool = False,
    capture_result: bool = False
):
    """
    Decorator for automatically tracing agent methods.
    
    Args:
        operation_name: Name of the operation (defaults to method name)
        level: Trace level
        capture_args: Whether to capture method arguments
        capture_result: Whether to capture method result
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            # Extract agent_id from self if available
            agent_id = getattr(self, 'agent_id', 'unknown')
            op_name = operation_name or f"{self.__class__.__name__}.{func.__name__}"
            
            # Get observability manager
            obs_manager = getattr(self, '_observability_manager', None)
            if not obs_manager:
                # Fallback to direct execution if no observability manager
                return await func(self, *args, **kwargs)
            
            tags = {}
            if capture_args:
                tags["args"] = str(args)
                tags["kwargs"] = str(kwargs)
            
            async with obs_manager.trace_agent_operation(
                agent_id=agent_id,
                operation_name=op_name,
                level=level,
                tags=tags
            ) as span:
                result = await func(self, *args, **kwargs)
                
                if capture_result:
                    span.tags["result"] = str(result)[:1000]  # Limit size
                
                return result
        
        @wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            # For synchronous methods, just execute directly
            # In production, could convert to async or use different tracing
            return func(self, *args, **kwargs)
        
        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class AgentPerformanceMonitor:
    """
    Monitor for tracking agent performance metrics and alerting on issues.
    """
    
    def __init__(self, observability_manager: AgenticObservabilityManager):
        self.observability_manager = observability_manager
        self.performance_thresholds = {
            "max_avg_duration_ms": 5000,  # 5 seconds
            "min_success_rate": 0.95,     # 95%
            "max_error_rate": 0.05        # 5%
        }
        self.alerts: List[Dict[str, Any]] = []
    
    async def check_agent_health(self, agent_id: str) -> Dict[str, Any]:
        """
        Check the health of a specific agent.
        
        Args:
            agent_id: ID of the agent to check
            
        Returns:
            Health check results
        """
        metrics = await self.observability_manager.get_agent_metrics(agent_id)
        if not metrics:
            return {
                "agent_id": agent_id,
                "status": "unknown",
                "message": "No metrics available"
            }
        
        health_status = "healthy"
        issues = []
        
        # Check average duration
        avg_duration_ms = (metrics["average_duration_ns"] / 1_000_000) if metrics["average_duration_ns"] else 0
        if avg_duration_ms > self.performance_thresholds["max_avg_duration_ms"]:
            health_status = "degraded"
            issues.append(f"Average duration ({avg_duration_ms:.2f}ms) exceeds threshold")
        
        # Check success rate
        success_rate = metrics.get("success_rate", 0)
        if success_rate < self.performance_thresholds["min_success_rate"]:
            health_status = "unhealthy"
            issues.append(f"Success rate ({success_rate:.2%}) below threshold")
        
        # Check error rate
        error_rate = metrics["failed_operations"] / metrics["total_operations"] if metrics["total_operations"] > 0 else 0
        if error_rate > self.performance_thresholds["max_error_rate"]:
            health_status = "unhealthy"
            issues.append(f"Error rate ({error_rate:.2%}) above threshold")
        
        return {
            "agent_id": agent_id,
            "status": health_status,
            "issues": issues,
            "metrics_summary": {
                "total_operations": metrics["total_operations"],
                "success_rate": success_rate,
                "average_duration_ms": avg_duration_ms,
                "error_rate": error_rate
            }
        }
    
    async def generate_alert(
        self,
        agent_id: str,
        alert_type: str,
        message: str,
        severity: str = "warning"
    ) -> None:
        """
        Generate an alert for agent performance issues.
        
        Args:
            agent_id: ID of the agent
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity level
        """
        alert = {
            "alert_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "alert_type": alert_type,
            "message": message,
            "severity": severity
        }
        
        self.alerts.append(alert)
        
        # Log the alert
        await self.observability_manager.log_agent_event(
            agent_id=agent_id,
            event_type="alert",
            message=f"{alert_type}: {message}",
            level=TraceLevel.WARNING if severity == "warning" else TraceLevel.ERROR,
            metadata={"alert_id": alert["alert_id"], "severity": severity}
        )
        
        logger.warning(f"Agent alert [{agent_id}] {alert_type}: {message}")


# Global observability manager instance
_global_observability_manager: Optional[AgenticObservabilityManager] = None


async def get_observability_manager() -> AgenticObservabilityManager:
    """Get the global observability manager instance."""
    global _global_observability_manager
    
    if _global_observability_manager is None:
        _global_observability_manager = AgenticObservabilityManager()
        await _global_observability_manager.initialize()
    
    return _global_observability_manager


async def initialize_observability() -> None:
    """Initialize the global observability manager."""
    await get_observability_manager()
    logger.info("Agentic observability integration initialized")