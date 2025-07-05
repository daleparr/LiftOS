"""
Time-Series Storage Utilities for Observability
Efficient storage and querying of metrics and time-series data
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import hashlib
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, asc
from sqlalchemy.dialects.postgresql import insert
import pandas as pd
import numpy as np

from ..database.observability_models import (
    MetricEntry, LogEntry, HealthCheckEntry, AlertEntry,
    MetricAggregation, SystemSnapshot, ServiceRegistry
)
from ..database.connection import DatabaseManager
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point"""
    name: str
    value: Union[int, float]
    timestamp: datetime
    labels: Dict[str, str]
    organization_id: Optional[str] = None
    service_name: Optional[str] = None


@dataclass
class TimeSeriesQuery:
    """Time-series query parameters"""
    metric_name: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    labels: Optional[Dict[str, str]] = None
    organization_id: Optional[str] = None
    service_name: Optional[str] = None
    aggregation: Optional[str] = None  # avg, sum, min, max, count
    time_window: Optional[str] = None  # 1m, 5m, 1h, 1d
    limit: Optional[int] = None


@dataclass
class AggregatedMetric:
    """Aggregated metric result"""
    metric_name: str
    aggregation_type: str
    time_window: str
    timestamp: datetime
    value: float
    sample_count: int
    labels: Dict[str, str]


class TimeSeriesStorage:
    """High-performance time-series storage for observability data"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = get_logger(f"{__name__}.TimeSeriesStorage")
        
        # In-memory buffers for batch writes
        self.metric_buffer: deque = deque(maxlen=10000)
        self.log_buffer: deque = deque(maxlen=10000)
        self.health_buffer: deque = deque(maxlen=1000)
        
        # Buffer flush settings
        self.buffer_flush_size = 100
        self.buffer_flush_interval = 30  # seconds
        
        # Aggregation cache
        self.aggregation_cache: Dict[str, Any] = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Background tasks
        self._flush_task = None
        self._aggregation_task = None
        self._cleanup_task = None
    
    async def initialize(self):
        """Initialize time-series storage"""
        try:
            await self.db_manager.initialize()
            
            # Start background tasks
            self._flush_task = asyncio.create_task(self._periodic_flush())
            self._aggregation_task = asyncio.create_task(self._periodic_aggregation())
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            
            self.logger.info("Time-series storage initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize time-series storage: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown time-series storage"""
        try:
            # Cancel background tasks
            if self._flush_task:
                self._flush_task.cancel()
            if self._aggregation_task:
                self._aggregation_task.cancel()
            if self._cleanup_task:
                self._cleanup_task.cancel()
            
            # Flush remaining buffers
            await self._flush_buffers()
            
            self.logger.info("Time-series storage shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during time-series storage shutdown: {e}")
    
    async def store_metric(self, metric: MetricPoint) -> bool:
        """Store a single metric point"""
        try:
            # Add to buffer
            self.metric_buffer.append(metric)
            
            # Flush if buffer is full
            if len(self.metric_buffer) >= self.buffer_flush_size:
                await self._flush_metrics()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store metric: {e}")
            return False
    
    async def store_metrics_batch(self, metrics: List[MetricPoint]) -> bool:
        """Store multiple metrics efficiently"""
        try:
            async with self.db_manager.get_session() as session:
                metric_entries = []
                
                for metric in metrics:
                    entry = MetricEntry(
                        name=metric.name,
                        value=metric.value,
                        timestamp=metric.timestamp,
                        labels=metric.labels,
                        organization_id=metric.organization_id,
                        service_name=metric.service_name
                    )
                    metric_entries.append(entry)
                
                session.add_all(metric_entries)
                await session.commit()
                
                self.logger.debug(f"Stored {len(metrics)} metrics")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to store metrics batch: {e}")
            return False
    
    async def query_metrics(self, query: TimeSeriesQuery) -> List[Dict[str, Any]]:
        """Query metrics with time-series optimizations"""
        try:
            async with self.db_manager.get_session() as session:
                # Build query
                stmt = select(MetricEntry)
                
                # Apply filters
                conditions = []
                
                if query.metric_name:
                    conditions.append(MetricEntry.name == query.metric_name)
                
                if query.start_time:
                    conditions.append(MetricEntry.timestamp >= query.start_time)
                
                if query.end_time:
                    conditions.append(MetricEntry.timestamp <= query.end_time)
                
                if query.organization_id:
                    conditions.append(MetricEntry.organization_id == query.organization_id)
                
                if query.service_name:
                    conditions.append(MetricEntry.service_name == query.service_name)
                
                if query.labels:
                    for key, value in query.labels.items():
                        conditions.append(MetricEntry.labels[key].astext == value)
                
                if conditions:
                    stmt = stmt.where(and_(*conditions))
                
                # Order by timestamp
                stmt = stmt.order_by(MetricEntry.timestamp.desc())
                
                # Apply limit
                if query.limit:
                    stmt = stmt.limit(query.limit)
                
                # Execute query
                result = await session.execute(stmt)
                metrics = result.scalars().all()
                
                # Convert to dict format
                return [
                    {
                        'name': m.name,
                        'value': float(m.value),
                        'timestamp': m.timestamp.isoformat(),
                        'labels': m.labels or {},
                        'organization_id': str(m.organization_id) if m.organization_id else None,
                        'service_name': m.service_name
                    }
                    for m in metrics
                ]
                
        except Exception as e:
            self.logger.error(f"Failed to query metrics: {e}")
            return []
    
    async def get_aggregated_metrics(
        self, 
        metric_name: str,
        aggregation_type: str,
        time_window: str,
        start_time: datetime,
        end_time: datetime,
        labels: Optional[Dict[str, str]] = None,
        service_name: Optional[str] = None
    ) -> List[AggregatedMetric]:
        """Get pre-computed aggregated metrics"""
        try:
            async with self.db_manager.get_session() as session:
                # Build query for aggregations
                stmt = select(MetricAggregation).where(
                    and_(
                        MetricAggregation.metric_name == metric_name,
                        MetricAggregation.aggregation_type == aggregation_type,
                        MetricAggregation.time_window == time_window,
                        MetricAggregation.timestamp >= start_time,
                        MetricAggregation.timestamp <= end_time
                    )
                )
                
                if service_name:
                    stmt = stmt.where(MetricAggregation.service_name == service_name)
                
                if labels:
                    labels_hash = self._hash_labels(labels)
                    stmt = stmt.where(MetricAggregation.labels_hash == labels_hash)
                
                stmt = stmt.order_by(MetricAggregation.timestamp.asc())
                
                result = await session.execute(stmt)
                aggregations = result.scalars().all()
                
                return [
                    AggregatedMetric(
                        metric_name=agg.metric_name,
                        aggregation_type=agg.aggregation_type,
                        time_window=agg.time_window,
                        timestamp=agg.timestamp,
                        value=float(agg.value),
                        sample_count=agg.sample_count,
                        labels=agg.labels or {}
                    )
                    for agg in aggregations
                ]
                
        except Exception as e:
            self.logger.error(f"Failed to get aggregated metrics: {e}")
            return []
    
    async def compute_real_time_aggregation(
        self,
        metric_name: str,
        aggregation_type: str,
        start_time: datetime,
        end_time: datetime,
        labels: Optional[Dict[str, str]] = None,
        service_name: Optional[str] = None
    ) -> Optional[float]:
        """Compute aggregation in real-time for recent data"""
        try:
            async with self.db_manager.get_session() as session:
                # Build aggregation query
                if aggregation_type == 'avg':
                    agg_func = func.avg(MetricEntry.value)
                elif aggregation_type == 'sum':
                    agg_func = func.sum(MetricEntry.value)
                elif aggregation_type == 'min':
                    agg_func = func.min(MetricEntry.value)
                elif aggregation_type == 'max':
                    agg_func = func.max(MetricEntry.value)
                elif aggregation_type == 'count':
                    agg_func = func.count(MetricEntry.value)
                else:
                    raise ValueError(f"Unsupported aggregation type: {aggregation_type}")
                
                stmt = select(agg_func).where(
                    and_(
                        MetricEntry.name == metric_name,
                        MetricEntry.timestamp >= start_time,
                        MetricEntry.timestamp <= end_time
                    )
                )
                
                if service_name:
                    stmt = stmt.where(MetricEntry.service_name == service_name)
                
                if labels:
                    for key, value in labels.items():
                        stmt = stmt.where(MetricEntry.labels[key].astext == value)
                
                result = await session.execute(stmt)
                value = result.scalar()
                
                return float(value) if value is not None else None
                
        except Exception as e:
            self.logger.error(f"Failed to compute real-time aggregation: {e}")
            return None
    
    async def store_health_check(
        self,
        service_name: str,
        status: str,
        response_time_ms: float,
        details: Optional[Dict[str, Any]] = None,
        endpoint_url: Optional[str] = None
    ) -> bool:
        """Store health check result"""
        try:
            async with self.db_manager.get_session() as session:
                health_entry = HealthCheckEntry(
                    service_name=service_name,
                    status=status,
                    timestamp=datetime.now(timezone.utc),
                    response_time_ms=response_time_ms,
                    details=details,
                    endpoint_url=endpoint_url
                )
                
                session.add(health_entry)
                await session.commit()
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to store health check: {e}")
            return False
    
    async def get_service_health_history(
        self,
        service_name: str,
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get health check history for a service"""
        try:
            start_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            async with self.db_manager.get_session() as session:
                stmt = select(HealthCheckEntry).where(
                    and_(
                        HealthCheckEntry.service_name == service_name,
                        HealthCheckEntry.timestamp >= start_time
                    )
                ).order_by(HealthCheckEntry.timestamp.desc())
                
                result = await session.execute(stmt)
                health_checks = result.scalars().all()
                
                return [
                    {
                        'service_name': hc.service_name,
                        'status': hc.status,
                        'timestamp': hc.timestamp.isoformat(),
                        'response_time_ms': float(hc.response_time_ms),
                        'details': hc.details,
                        'endpoint_url': hc.endpoint_url
                    }
                    for hc in health_checks
                ]
                
        except Exception as e:
            self.logger.error(f"Failed to get service health history: {e}")
            return []
    
    async def create_system_snapshot(self) -> bool:
        """Create a system health snapshot"""
        try:
            async with self.db_manager.get_session() as session:
                # Get current system metrics
                now = datetime.now(timezone.utc)
                one_hour_ago = now - timedelta(hours=1)
                
                # Count services by status
                health_stats = await session.execute(
                    select(
                        HealthCheckEntry.status,
                        func.count(HealthCheckEntry.service_name.distinct())
                    ).where(
                        HealthCheckEntry.timestamp >= one_hour_ago
                    ).group_by(HealthCheckEntry.status)
                )
                
                status_counts = dict(health_stats.fetchall())
                
                # Count alerts by severity
                alert_stats = await session.execute(
                    select(
                        AlertEntry.severity,
                        func.count(AlertEntry.id)
                    ).where(
                        and_(
                            AlertEntry.status == 'firing',
                            AlertEntry.timestamp >= one_hour_ago
                        )
                    ).group_by(AlertEntry.severity)
                )
                
                alert_counts = dict(alert_stats.fetchall())
                
                # Calculate average response time
                avg_response_time = await session.execute(
                    select(func.avg(HealthCheckEntry.response_time_ms)).where(
                        HealthCheckEntry.timestamp >= one_hour_ago
                    )
                )
                avg_response = avg_response_time.scalar()
                
                # Create snapshot
                snapshot = SystemSnapshot(
                    timestamp=now,
                    total_services=sum(status_counts.values()),
                    healthy_services=status_counts.get('healthy', 0),
                    unhealthy_services=status_counts.get('unhealthy', 0),
                    degraded_services=status_counts.get('degraded', 0),
                    total_alerts=sum(alert_counts.values()),
                    critical_alerts=alert_counts.get('critical', 0),
                    warning_alerts=alert_counts.get('warning', 0),
                    avg_response_time_ms=float(avg_response) if avg_response else None
                )
                
                session.add(snapshot)
                await session.commit()
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to create system snapshot: {e}")
            return False
    
    def _hash_labels(self, labels: Dict[str, str]) -> str:
        """Create a hash of labels for efficient grouping"""
        sorted_labels = sorted(labels.items())
        labels_str = json.dumps(sorted_labels, sort_keys=True)
        return hashlib.sha256(labels_str.encode()).hexdigest()[:16]
    
    async def _flush_buffers(self):
        """Flush all buffers to database"""
        await self._flush_metrics()
        await self._flush_logs()
        await self._flush_health_checks()
    
    async def _flush_metrics(self):
        """Flush metric buffer to database"""
        if not self.metric_buffer:
            return
        
        try:
            metrics = list(self.metric_buffer)
            self.metric_buffer.clear()
            
            await self.store_metrics_batch(metrics)
            
        except Exception as e:
            self.logger.error(f"Failed to flush metrics: {e}")
    
    async def _flush_logs(self):
        """Flush log buffer to database"""
        # Implementation for log flushing
        pass
    
    async def _flush_health_checks(self):
        """Flush health check buffer to database"""
        # Implementation for health check flushing
        pass
    
    async def _periodic_flush(self):
        """Periodic buffer flushing"""
        while True:
            try:
                await asyncio.sleep(self.buffer_flush_interval)
                await self._flush_buffers()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in periodic flush: {e}")
    
    async def _periodic_aggregation(self):
        """Periodic metric aggregation"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._compute_aggregations()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in periodic aggregation: {e}")
    
    async def _periodic_cleanup(self):
        """Periodic data cleanup"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_old_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in periodic cleanup: {e}")
    
    async def _compute_aggregations(self):
        """Compute metric aggregations"""
        # Implementation for computing aggregations
        pass
    
    async def _cleanup_old_data(self):
        """Clean up old time-series data"""
        try:
            async with self.db_manager.get_session() as session:
                # Clean up old metrics (keep 30 days)
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
                
                await session.execute(
                    MetricEntry.__table__.delete().where(
                        MetricEntry.timestamp < cutoff_date
                    )
                )
                
                # Clean up old logs (keep 7 days)
                log_cutoff = datetime.now(timezone.utc) - timedelta(days=7)
                
                await session.execute(
                    LogEntry.__table__.delete().where(
                        LogEntry.timestamp < log_cutoff
                    )
                )
                
                await session.commit()
                
                self.logger.info("Completed data cleanup")
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")


# Global time-series storage instance
_time_series_storage: Optional[TimeSeriesStorage] = None


async def get_time_series_storage() -> TimeSeriesStorage:
    """Get global time-series storage instance"""
    global _time_series_storage
    
    if _time_series_storage is None:
        from ..database.connection import DatabaseManager
        db_manager = DatabaseManager()
        _time_series_storage = TimeSeriesStorage(db_manager)
        await _time_series_storage.initialize()
    
    return _time_series_storage