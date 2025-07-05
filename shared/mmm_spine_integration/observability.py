"""
Lightweight Observability Framework - Adapted from MMM Spine
Provides zero-impact tracing with nanosecond precision and minimal overhead
"""

import asyncio
import time
import threading
import json
import uuid
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from enum import Enum
import traceback
import sys

from ..utils.logging import get_logger

logger = get_logger(__name__)


class TraceLevel(Enum):
    """Trace levels for filtering and performance optimization."""
    CRITICAL = 1
    ERROR = 2
    WARNING = 3
    INFO = 4
    DEBUG = 5
    TRACE = 6


@dataclass
class TraceSpan:
    """Represents a trace span with nanosecond precision."""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    operation_name: str
    service_name: str
    start_time_ns: int
    end_time_ns: Optional[int] = None
    duration_ns: Optional[int] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "active"
    error: Optional[str] = None
    level: TraceLevel = TraceLevel.INFO


@dataclass
class MetricPoint:
    """Represents a metric data point."""
    metric_name: str
    value: Union[int, float]
    timestamp_ns: int
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: str = "gauge"  # gauge, counter, histogram, timer


@dataclass
class AccountabilityEvent:
    """Represents an accountability event for decision tracking."""
    event_id: str
    org_id: str
    operation: str
    decision_point: str
    reasoning: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    confidence: float
    timestamp_ns: int
    trace_id: Optional[str] = None
    span_id: Optional[str] = None


class LightweightTracer:
    """
    Ultra-lightweight tracer with zero-impact design.
    Adapted from MMM Spine lightweight_tracer.py
    """
    
    def __init__(self, service_name: str, max_spans: int = 10000, 
                 sampling_rate: float = 1.0, min_trace_level: TraceLevel = TraceLevel.INFO):
        """
        Initialize lightweight tracer.
        
        Args:
            service_name: Name of the service
            max_spans: Maximum number of spans to keep in memory
            sampling_rate: Sampling rate for traces (0.0 to 1.0)
            min_trace_level: Minimum trace level to record
        """
        self.service_name = service_name
        self.max_spans = max_spans
        self.sampling_rate = sampling_rate
        self.min_trace_level = min_trace_level
        
        # Span storage with circular buffer
        self.active_spans: Dict[str, TraceSpan] = {}
        self.completed_spans: deque = deque(maxlen=max_spans)
        
        # Performance tracking
        self.total_spans = 0
        self.sampled_spans = 0
        self.overhead_ns = 0
        
        # Thread-local storage for current span
        self.local = threading.local()
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(f"Initialized LightweightTracer for service={service_name}, "
                   f"sampling_rate={sampling_rate}")
    
    def _should_sample(self) -> bool:
        """Determine if this trace should be sampled."""
        import random
        return random.random() < self.sampling_rate
    
    def _get_current_span(self) -> Optional[TraceSpan]:
        """Get current span from thread-local storage."""
        return getattr(self.local, 'current_span', None)
    
    def _set_current_span(self, span: Optional[TraceSpan]):
        """Set current span in thread-local storage."""
        self.local.current_span = span
    
    def start_span(self, operation_name: str, parent_span_id: Optional[str] = None,
                   tags: Optional[Dict[str, Any]] = None, 
                   level: TraceLevel = TraceLevel.INFO) -> Optional[TraceSpan]:
        """
        Start a new trace span with minimal overhead.
        
        Args:
            operation_name: Name of the operation
            parent_span_id: Optional parent span ID
            tags: Optional tags for the span
            level: Trace level
            
        Returns:
            TraceSpan if sampled, None otherwise
        """
        start_overhead = time.time_ns()
        
        # Check if we should trace this operation
        if level.value > self.min_trace_level.value:
            return None
        
        if not self._should_sample():
            return None
        
        # Get parent span if not specified
        if parent_span_id is None:
            current_span = self._get_current_span()
            if current_span:
                parent_span_id = current_span.span_id
        
        # Create span
        span_id = uuid.uuid4().hex[:16]
        trace_id = parent_span_id or uuid.uuid4().hex[:16]
        
        span = TraceSpan(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            service_name=self.service_name,
            start_time_ns=time.time_ns(),
            tags=tags or {},
            level=level
        )
        
        with self.lock:
            self.active_spans[span_id] = span
            self.total_spans += 1
            self.sampled_spans += 1
        
        # Set as current span
        self._set_current_span(span)
        
        # Track overhead
        overhead = time.time_ns() - start_overhead
        self.overhead_ns += overhead
        
        return span
    
    def finish_span(self, span: TraceSpan, error: Optional[str] = None):
        """
        Finish a trace span with minimal overhead.
        
        Args:
            span: Span to finish
            error: Optional error message
        """
        if not span:
            return
        
        start_overhead = time.time_ns()
        
        # Update span
        span.end_time_ns = time.time_ns()
        span.duration_ns = span.end_time_ns - span.start_time_ns
        span.status = "error" if error else "completed"
        span.error = error
        
        with self.lock:
            # Move from active to completed
            if span.span_id in self.active_spans:
                del self.active_spans[span.span_id]
            
            self.completed_spans.append(span)
        
        # Clear current span if this was it
        current_span = self._get_current_span()
        if current_span and current_span.span_id == span.span_id:
            # Set parent as current span
            if span.parent_span_id and span.parent_span_id in self.active_spans:
                self._set_current_span(self.active_spans[span.parent_span_id])
            else:
                self._set_current_span(None)
        
        # Track overhead
        overhead = time.time_ns() - start_overhead
        self.overhead_ns += overhead
    
    def add_span_tag(self, span: TraceSpan, key: str, value: Any):
        """Add tag to span with minimal overhead."""
        if span:
            span.tags[key] = value
    
    def add_span_log(self, span: TraceSpan, message: str, level: str = "info"):
        """Add log entry to span with minimal overhead."""
        if span:
            span.logs.append({
                "timestamp_ns": time.time_ns(),
                "level": level,
                "message": message
            })
    
    @contextmanager
    def trace_operation(self, operation_name: str, tags: Optional[Dict[str, Any]] = None,
                       level: TraceLevel = TraceLevel.INFO):
        """
        Context manager for tracing operations.
        
        Args:
            operation_name: Name of the operation
            tags: Optional tags
            level: Trace level
        """
        span = self.start_span(operation_name, tags=tags, level=level)
        error = None
        
        try:
            yield span
        except Exception as e:
            error = str(e)
            if span:
                self.add_span_tag(span, "error", True)
                self.add_span_tag(span, "error.message", str(e))
                self.add_span_tag(span, "error.type", type(e).__name__)
            raise
        finally:
            if span:
                self.finish_span(span, error)
    
    @asynccontextmanager
    async def trace_async_operation(self, operation_name: str, 
                                   tags: Optional[Dict[str, Any]] = None,
                                   level: TraceLevel = TraceLevel.INFO):
        """
        Async context manager for tracing operations.
        
        Args:
            operation_name: Name of the operation
            tags: Optional tags
            level: Trace level
        """
        span = self.start_span(operation_name, tags=tags, level=level)
        error = None
        
        try:
            yield span
        except Exception as e:
            error = str(e)
            if span:
                self.add_span_tag(span, "error", True)
                self.add_span_tag(span, "error.message", str(e))
                self.add_span_tag(span, "error.type", type(e).__name__)
            raise
        finally:
            if span:
                self.finish_span(span, error)
    
    def get_trace_stats(self) -> Dict[str, Any]:
        """Get tracing statistics."""
        with self.lock:
            avg_overhead = self.overhead_ns / self.total_spans if self.total_spans > 0 else 0
            
            return {
                "service_name": self.service_name,
                "total_spans": self.total_spans,
                "sampled_spans": self.sampled_spans,
                "sampling_rate": self.sampling_rate,
                "active_spans": len(self.active_spans),
                "completed_spans": len(self.completed_spans),
                "average_overhead_ns": avg_overhead,
                "total_overhead_ms": self.overhead_ns / 1e6
            }
    
    def get_recent_spans(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent completed spans."""
        with self.lock:
            recent_spans = list(self.completed_spans)[-limit:]
            
            return [
                {
                    "span_id": span.span_id,
                    "trace_id": span.trace_id,
                    "operation_name": span.operation_name,
                    "duration_ms": span.duration_ns / 1e6 if span.duration_ns else None,
                    "status": span.status,
                    "tags": span.tags,
                    "error": span.error,
                    "level": span.level.name
                }
                for span in recent_spans
            ]


class MetricsCollector:
    """
    High-performance metrics collector with minimal overhead.
    Adapted from MMM Spine metrics_collector.py
    """
    
    def __init__(self, service_name: str, max_metrics: int = 50000,
                 aggregation_interval: int = 60):
        """
        Initialize metrics collector.
        
        Args:
            service_name: Name of the service
            max_metrics: Maximum number of metric points to keep
            aggregation_interval: Interval for metric aggregation in seconds
        """
        self.service_name = service_name
        self.max_metrics = max_metrics
        self.aggregation_interval = aggregation_interval
        
        # Metric storage
        self.raw_metrics: deque = deque(maxlen=max_metrics)
        self.aggregated_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Performance counters
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background aggregation
        self.last_aggregation = time.time()
        
        logger.info(f"Initialized MetricsCollector for service={service_name}")
    
    def record_metric(self, metric_name: str, value: Union[int, float],
                     tags: Optional[Dict[str, str]] = None,
                     metric_type: str = "gauge"):
        """
        Record a metric with minimal overhead.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            tags: Optional tags
            metric_type: Type of metric (gauge, counter, histogram, timer)
        """
        timestamp_ns = time.time_ns()
        
        metric_point = MetricPoint(
            metric_name=metric_name,
            value=value,
            timestamp_ns=timestamp_ns,
            tags=tags or {},
            metric_type=metric_type
        )
        
        with self.lock:
            self.raw_metrics.append(metric_point)
            
            # Update real-time aggregations
            if metric_type == "counter":
                self.counters[metric_name] += value
            elif metric_type == "gauge":
                self.gauges[metric_name] = value
            elif metric_type == "histogram":
                self.histograms[metric_name].append(value)
                # Keep only recent values for histograms
                if len(self.histograms[metric_name]) > 1000:
                    self.histograms[metric_name] = self.histograms[metric_name][-1000:]
            elif metric_type == "timer":
                self.timers[metric_name].append(value)
                # Keep only recent values for timers
                if len(self.timers[metric_name]) > 1000:
                    self.timers[metric_name] = self.timers[metric_name][-1000:]
    
    def increment_counter(self, metric_name: str, value: int = 1,
                         tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        self.record_metric(metric_name, value, tags, "counter")
    
    def set_gauge(self, metric_name: str, value: Union[int, float],
                  tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric."""
        self.record_metric(metric_name, value, tags, "gauge")
    
    def record_histogram(self, metric_name: str, value: Union[int, float],
                        tags: Optional[Dict[str, str]] = None):
        """Record a histogram value."""
        self.record_metric(metric_name, value, tags, "histogram")
    
    def record_timer(self, metric_name: str, duration_ms: float,
                    tags: Optional[Dict[str, str]] = None):
        """Record a timer value."""
        self.record_metric(metric_name, duration_ms, tags, "timer")
    
    @contextmanager
    def time_operation(self, metric_name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        start_time = time.time_ns()
        try:
            yield
        finally:
            duration_ms = (time.time_ns() - start_time) / 1e6
            self.record_timer(metric_name, duration_ms, tags)
    
    def get_metric_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        with self.lock:
            # Calculate histogram statistics
            histogram_stats = {}
            for name, values in self.histograms.items():
                if values:
                    sorted_values = sorted(values)
                    n = len(sorted_values)
                    histogram_stats[name] = {
                        "count": n,
                        "min": min(sorted_values),
                        "max": max(sorted_values),
                        "mean": sum(sorted_values) / n,
                        "p50": sorted_values[n // 2],
                        "p95": sorted_values[int(n * 0.95)],
                        "p99": sorted_values[int(n * 0.99)]
                    }
            
            # Calculate timer statistics
            timer_stats = {}
            for name, values in self.timers.items():
                if values:
                    sorted_values = sorted(values)
                    n = len(sorted_values)
                    timer_stats[name] = {
                        "count": n,
                        "min_ms": min(sorted_values),
                        "max_ms": max(sorted_values),
                        "mean_ms": sum(sorted_values) / n,
                        "p50_ms": sorted_values[n // 2],
                        "p95_ms": sorted_values[int(n * 0.95)],
                        "p99_ms": sorted_values[int(n * 0.99)]
                    }
            
            return {
                "service_name": self.service_name,
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": histogram_stats,
                "timers": timer_stats,
                "total_metrics": len(self.raw_metrics),
                "last_aggregation": self.last_aggregation
            }


class AccountabilityOrchestrator:
    """
    Accountability orchestrator for decision tracking and reasoning transparency.
    Adapted from MMM Spine accountability_orchestrator.py
    """
    
    def __init__(self, org_id: str, max_events: int = 100000):
        """
        Initialize accountability orchestrator.
        
        Args:
            org_id: Organization ID
            max_events: Maximum number of events to keep
        """
        self.org_id = org_id
        self.max_events = max_events
        
        # Event storage
        self.events: deque = deque(maxlen=max_events)
        self.decision_chains: Dict[str, List[str]] = defaultdict(list)
        
        # Performance tracking
        self.total_events = 0
        self.decision_points = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(f"Initialized AccountabilityOrchestrator for org={org_id}")
    
    def record_decision(self, operation: str, decision_point: str, reasoning: str,
                       inputs: Dict[str, Any], outputs: Dict[str, Any],
                       confidence: float, trace_id: Optional[str] = None,
                       span_id: Optional[str] = None) -> str:
        """
        Record a decision event for accountability.
        
        Args:
            operation: Operation name
            decision_point: Decision point identifier
            reasoning: Reasoning for the decision
            inputs: Input data for the decision
            outputs: Output data from the decision
            confidence: Confidence level (0.0 to 1.0)
            trace_id: Optional trace ID
            span_id: Optional span ID
            
        Returns:
            Event ID
        """
        event_id = str(uuid.uuid4())
        
        event = AccountabilityEvent(
            event_id=event_id,
            org_id=self.org_id,
            operation=operation,
            decision_point=decision_point,
            reasoning=reasoning,
            inputs=inputs,
            outputs=outputs,
            confidence=confidence,
            timestamp_ns=time.time_ns(),
            trace_id=trace_id,
            span_id=span_id
        )
        
        with self.lock:
            self.events.append(event)
            self.decision_chains[operation].append(event_id)
            self.total_events += 1
            self.decision_points += 1
        
        logger.debug(f"Recorded decision: {decision_point} for operation {operation}")
        return event_id
    
    def get_decision_chain(self, operation: str) -> List[Dict[str, Any]]:
        """Get decision chain for an operation."""
        with self.lock:
            event_ids = self.decision_chains.get(operation, [])
            
            # Find events by ID
            events_by_id = {event.event_id: event for event in self.events}
            
            chain = []
            for event_id in event_ids:
                if event_id in events_by_id:
                    event = events_by_id[event_id]
                    chain.append({
                        "event_id": event.event_id,
                        "decision_point": event.decision_point,
                        "reasoning": event.reasoning,
                        "confidence": event.confidence,
                        "timestamp": event.timestamp_ns / 1e9,
                        "trace_id": event.trace_id,
                        "span_id": event.span_id
                    })
            
            return chain
    
    def get_accountability_stats(self) -> Dict[str, Any]:
        """Get accountability statistics."""
        with self.lock:
            recent_events = list(self.events)[-1000:]  # Last 1000 events
            
            # Calculate confidence statistics
            confidences = [event.confidence for event in recent_events]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Count operations
            operations = defaultdict(int)
            for event in recent_events:
                operations[event.operation] += 1
            
            return {
                "org_id": self.org_id,
                "total_events": self.total_events,
                "decision_points": self.decision_points,
                "recent_events": len(recent_events),
                "average_confidence": avg_confidence,
                "operations_count": dict(operations),
                "decision_chains": len(self.decision_chains)
            }


class ObservabilityManager:
    """
    Unified observability manager combining tracing, metrics, and accountability.
    """
    
    def __init__(self, service_name: str, org_id: str):
        """
        Initialize observability manager.
        
        Args:
            service_name: Name of the service
            org_id: Organization ID
        """
        self.service_name = service_name
        self.org_id = org_id
        
        # Initialize components
        self.tracer = LightweightTracer(service_name)
        self.metrics = MetricsCollector(service_name)
        self.accountability = AccountabilityOrchestrator(org_id)
        
        logger.info(f"Initialized ObservabilityManager for service={service_name}, org={org_id}")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive observability statistics."""
        return {
            "service_name": self.service_name,
            "org_id": self.org_id,
            "tracing": self.tracer.get_trace_stats(),
            "metrics": self.metrics.get_metric_summary(),
            "accountability": self.accountability.get_accountability_stats(),
            "timestamp": time.time()
        }
    
    @contextmanager
    def observe_operation(self, operation_name: str, 
                         tags: Optional[Dict[str, Any]] = None,
                         level: TraceLevel = TraceLevel.INFO):
        """
        Context manager for comprehensive operation observability.
        
        Args:
            operation_name: Name of the operation
            tags: Optional tags
            level: Trace level
        """
        # Start tracing
        with self.tracer.trace_operation(operation_name, tags, level) as span:
            # Start timing
            with self.metrics.time_operation(f"{operation_name}.duration", 
                                           tags.get("metric_tags") if tags else None):
                try:
                    yield span
                    # Record success metric
                    self.metrics.increment_counter(f"{operation_name}.success")
                except Exception as e:
                    # Record error metric
                    self.metrics.increment_counter(f"{operation_name}.error")
                    raise
    
    @asynccontextmanager
    async def observe_async_operation(self, operation_name: str,
                                     tags: Optional[Dict[str, Any]] = None,
                                     level: TraceLevel = TraceLevel.INFO):
        """
        Async context manager for comprehensive operation observability.
        
        Args:
            operation_name: Name of the operation
            tags: Optional tags
            level: Trace level
        """
        # Start tracing
        async with self.tracer.trace_async_operation(operation_name, tags, level) as span:
            # Start timing
            start_time = time.time_ns()
            try:
                yield span
                # Record success metric
                self.metrics.increment_counter(f"{operation_name}.success")
            except Exception as e:
                # Record error metric
                self.metrics.increment_counter(f"{operation_name}.error")
                raise
            finally:
                # Record timing
                duration_ms = (time.time_ns() - start_time) / 1e6
                self.metrics.record_timer(f"{operation_name}.duration", duration_ms,
                                        tags.get("metric_tags") if tags else None)