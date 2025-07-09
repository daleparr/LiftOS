"""
KSE Memory SDK Analytics Service
"""

from typing import Dict, Any, List, Optional
import asyncio
import time
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
from ..core.interfaces import AnalyticsInterface

logger = logging.getLogger(__name__)


class AnalyticsService(AnalyticsInterface):
    """
    Analytics service for tracking KSE Memory operations and performance.
    """
    
    def __init__(self, config):
        """
        Initialize analytics service.
        
        Args:
            config: Configuration dictionary or KSEConfig object
        """
        self.config = config
        self.metrics: Dict[str, Any] = defaultdict(lambda: defaultdict(int))
        self.performance_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.operation_history: deque = deque(maxlen=10000)
        self.start_time = time.time()
        
        # Configuration - handle both dict and dataclass
        if hasattr(config, 'get'):
            # Dictionary config
            self.retention_days = config.get('retention_days', 30)
            self.aggregation_interval = config.get('aggregation_interval', 300)  # 5 minutes
        else:
            # Dataclass config
            self.retention_days = getattr(config, 'retention_days', 30)
            self.aggregation_interval = getattr(config, 'aggregation_interval', 300)  # 5 minutes
        
    async def initialize(self) -> bool:
        """Initialize analytics service."""
        try:
            logger.info("Analytics service initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize analytics service: {e}")
            return False
    
    async def track_operation(self, operation: str, duration: float, 
                            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Track an operation with its duration and metadata.
        
        Args:
            operation: Operation name
            duration: Operation duration in seconds
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        try:
            timestamp = time.time()
            
            # Update metrics
            self.metrics[operation]['count'] += 1
            self.metrics[operation]['total_duration'] += duration
            
            # Track performance data
            self.performance_data[operation].append({
                'timestamp': timestamp,
                'duration': duration,
                'metadata': metadata or {}
            })
            
            # Add to operation history
            self.operation_history.append({
                'operation': operation,
                'timestamp': timestamp,
                'duration': duration,
                'metadata': metadata or {}
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to track operation: {e}")
            return False
    
    async def track_error(self, operation: str, error: str, 
                         metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Track an error for an operation.
        
        Args:
            operation: Operation name
            error: Error message
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        try:
            timestamp = time.time()
            
            # Update error metrics
            self.metrics[operation]['errors'] += 1
            self.metrics[f"{operation}_errors"][error] += 1
            
            # Add to operation history
            self.operation_history.append({
                'operation': operation,
                'timestamp': timestamp,
                'error': error,
                'metadata': metadata or {}
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to track error: {e}")
            return False
    
    async def get_metrics(self, operation: Optional[str] = None, 
                         time_range: Optional[int] = None) -> Dict[str, Any]:
        """
        Get metrics for operations.
        
        Args:
            operation: Specific operation to get metrics for
            time_range: Time range in seconds (from now backwards)
            
        Returns:
            Metrics data
        """
        try:
            if operation:
                # Get metrics for specific operation
                op_metrics = self.metrics.get(operation, {})
                
                # Calculate averages
                count = op_metrics.get('count', 0)
                total_duration = op_metrics.get('total_duration', 0)
                avg_duration = total_duration / count if count > 0 else 0
                
                # Get recent performance data
                recent_data = []
                if time_range:
                    cutoff_time = time.time() - time_range
                    recent_data = [
                        entry for entry in self.performance_data[operation]
                        if entry['timestamp'] >= cutoff_time
                    ]
                else:
                    recent_data = list(self.performance_data[operation])
                
                return {
                    'operation': operation,
                    'total_count': count,
                    'total_duration': total_duration,
                    'average_duration': avg_duration,
                    'error_count': op_metrics.get('errors', 0),
                    'recent_operations': len(recent_data),
                    'recent_data': recent_data[-100:]  # Last 100 operations
                }
            else:
                # Get metrics for all operations
                all_metrics = {}
                for op_name, op_data in self.metrics.items():
                    if not op_name.endswith('_errors'):
                        count = op_data.get('count', 0)
                        total_duration = op_data.get('total_duration', 0)
                        avg_duration = total_duration / count if count > 0 else 0
                        
                        all_metrics[op_name] = {
                            'count': count,
                            'total_duration': total_duration,
                            'average_duration': avg_duration,
                            'error_count': op_data.get('errors', 0)
                        }
                
                return {
                    'operations': all_metrics,
                    'uptime': time.time() - self.start_time,
                    'total_operations': sum(op['count'] for op in all_metrics.values())
                }
                
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {}
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary across all operations.
        
        Returns:
            Performance summary
        """
        try:
            summary = {
                'uptime': time.time() - self.start_time,
                'total_operations': 0,
                'total_errors': 0,
                'operations_per_second': 0,
                'average_response_time': 0,
                'top_operations': [],
                'error_rate': 0
            }
            
            total_duration = 0
            operation_counts = []
            
            for operation, data in self.metrics.items():
                if not operation.endswith('_errors'):
                    count = data.get('count', 0)
                    duration = data.get('total_duration', 0)
                    errors = data.get('errors', 0)
                    
                    summary['total_operations'] += count
                    summary['total_errors'] += errors
                    total_duration += duration
                    
                    if count > 0:
                        operation_counts.append({
                            'operation': operation,
                            'count': count,
                            'average_duration': duration / count,
                            'error_rate': errors / count if count > 0 else 0
                        })
            
            # Calculate derived metrics
            uptime = summary['uptime']
            if uptime > 0:
                summary['operations_per_second'] = summary['total_operations'] / uptime
            
            if summary['total_operations'] > 0:
                summary['average_response_time'] = total_duration / summary['total_operations']
                summary['error_rate'] = summary['total_errors'] / summary['total_operations']
            
            # Top operations by count
            summary['top_operations'] = sorted(
                operation_counts, 
                key=lambda x: x['count'], 
                reverse=True
            )[:10]
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {}
    
    async def get_recent_operations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent operations.
        
        Args:
            limit: Maximum number of operations to return
            
        Returns:
            List of recent operations
        """
        try:
            recent_ops = list(self.operation_history)[-limit:]
            return recent_ops
        except Exception as e:
            logger.error(f"Failed to get recent operations: {e}")
            return []
    
    async def clear_metrics(self, operation: Optional[str] = None) -> bool:
        """
        Clear metrics data.
        
        Args:
            operation: Specific operation to clear, or None for all
            
        Returns:
            True if successful
        """
        try:
            if operation:
                if operation in self.metrics:
                    del self.metrics[operation]
                if operation in self.performance_data:
                    self.performance_data[operation].clear()
            else:
                self.metrics.clear()
                self.performance_data.clear()
                self.operation_history.clear()
                self.start_time = time.time()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear metrics: {e}")
            return False
    
    async def export_metrics(self, format_type: str = 'json') -> Optional[str]:
        """
        Export metrics data.
        
        Args:
            format_type: Export format ('json', 'csv')
            
        Returns:
            Exported data as string
        """
        try:
            if format_type == 'json':
                import json
                export_data = {
                    'metrics': dict(self.metrics),
                    'performance_data': {
                        op: list(data) for op, data in self.performance_data.items()
                    },
                    'operation_history': list(self.operation_history),
                    'export_timestamp': time.time()
                }
                return json.dumps(export_data, indent=2)
            
            elif format_type == 'csv':
                # Simple CSV export of operation metrics
                lines = ['operation,count,total_duration,average_duration,errors']
                for operation, data in self.metrics.items():
                    if not operation.endswith('_errors'):
                        count = data.get('count', 0)
                        total_duration = data.get('total_duration', 0)
                        avg_duration = total_duration / count if count > 0 else 0
                        errors = data.get('errors', 0)
                        
                        lines.append(f"{operation},{count},{total_duration},{avg_duration},{errors}")
                
                return '\n'.join(lines)
            
            else:
                logger.error(f"Unsupported export format: {format_type}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return None

    # Abstract method implementations required by AnalyticsInterface
    
    async def track_search(self, query, results, duration_ms: float) -> bool:
        """Track search operation."""
        try:
            # Import here to avoid circular imports
            from ..core.models import SearchQuery, SearchResult
            
            query_text = query.query if hasattr(query, 'query') else str(query)
            result_count = len(results) if results else 0
            
            metadata = {
                'query': query_text,
                'result_count': result_count,
                'search_type': getattr(query, 'search_type', 'unknown'),
                'domain': getattr(query, 'domain', None)
            }
            
            return await self.track_operation('search', duration_ms / 1000.0, metadata)
            
        except Exception as e:
            logger.error(f"Failed to track search: {e}")
            return False

    async def track_entity_operation(self, operation: str, entity_id: str,
                                   duration_ms: float, success: bool) -> bool:
        """Track entity operation."""
        try:
            metadata = {
                'entity_id': entity_id,
                'success': success,
                'operation_type': 'entity_operation'
            }
            
            # Track as error if not successful
            if not success:
                await self.track_error(operation, f"Entity operation failed for {entity_id}", metadata)
            
            return await self.track_operation(operation, duration_ms / 1000.0, metadata)
            
        except Exception as e:
            logger.error(f"Failed to track entity operation: {e}")
            return False

    async def get_search_analytics(self, time_range: str = "24h") -> Dict[str, Any]:
        """Get search analytics."""
        try:
            # Parse time range
            time_range_seconds = self._parse_time_range(time_range)
            cutoff_time = time.time() - time_range_seconds
            
            # Filter search operations
            search_ops = [
                op for op in self.operation_history
                if op.get('operation') == 'search' and op.get('timestamp', 0) >= cutoff_time
            ]
            
            if not search_ops:
                return {
                    'total_searches': 0,
                    'average_duration': 0,
                    'average_results': 0,
                    'popular_queries': [],
                    'search_types': {},
                    'domains': {}
                }
            
            # Calculate analytics
            total_searches = len(search_ops)
            total_duration = sum(op.get('duration', 0) for op in search_ops)
            total_results = sum(op.get('metadata', {}).get('result_count', 0) for op in search_ops)
            
            # Count queries, search types, and domains
            query_counts = defaultdict(int)
            search_type_counts = defaultdict(int)
            domain_counts = defaultdict(int)
            
            for op in search_ops:
                metadata = op.get('metadata', {})
                query = metadata.get('query', 'unknown')
                search_type = metadata.get('search_type', 'unknown')
                domain = metadata.get('domain', 'unknown')
                
                query_counts[query] += 1
                search_type_counts[search_type] += 1
                domain_counts[domain] += 1
            
            return {
                'total_searches': total_searches,
                'average_duration': total_duration / total_searches if total_searches > 0 else 0,
                'average_results': total_results / total_searches if total_searches > 0 else 0,
                'popular_queries': dict(sorted(query_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
                'search_types': dict(search_type_counts),
                'domains': dict(domain_counts),
                'time_range': time_range
            }
            
        except Exception as e:
            logger.error(f"Failed to get search analytics: {e}")
            return {}

    async def get_performance_metrics(self, time_range: str = "24h") -> Dict[str, Any]:
        """Get performance metrics."""
        try:
            # Parse time range
            time_range_seconds = self._parse_time_range(time_range)
            cutoff_time = time.time() - time_range_seconds
            
            # Filter operations by time range
            recent_ops = [
                op for op in self.operation_history
                if op.get('timestamp', 0) >= cutoff_time
            ]
            
            if not recent_ops:
                return {
                    'total_operations': 0,
                    'average_response_time': 0,
                    'operations_per_second': 0,
                    'error_rate': 0,
                    'slowest_operations': [],
                    'fastest_operations': []
                }
            
            # Calculate metrics
            total_operations = len(recent_ops)
            total_duration = sum(op.get('duration', 0) for op in recent_ops)
            error_count = sum(1 for op in recent_ops if 'error' in op)
            
            # Sort by duration for slowest/fastest
            ops_by_duration = sorted(recent_ops, key=lambda x: x.get('duration', 0), reverse=True)
            
            return {
                'total_operations': total_operations,
                'average_response_time': total_duration / total_operations if total_operations > 0 else 0,
                'operations_per_second': total_operations / time_range_seconds if time_range_seconds > 0 else 0,
                'error_rate': error_count / total_operations if total_operations > 0 else 0,
                'slowest_operations': [
                    {
                        'operation': op.get('operation'),
                        'duration': op.get('duration'),
                        'timestamp': op.get('timestamp')
                    }
                    for op in ops_by_duration[:5]
                ],
                'fastest_operations': [
                    {
                        'operation': op.get('operation'),
                        'duration': op.get('duration'),
                        'timestamp': op.get('timestamp')
                    }
                    for op in ops_by_duration[-5:]
                ],
                'time_range': time_range
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {}

    async def get_usage_statistics(self, time_range: str = "24h") -> Dict[str, Any]:
        """Get usage statistics."""
        try:
            # Parse time range
            time_range_seconds = self._parse_time_range(time_range)
            cutoff_time = time.time() - time_range_seconds
            
            # Filter operations by time range
            recent_ops = [
                op for op in self.operation_history
                if op.get('timestamp', 0) >= cutoff_time
            ]
            
            # Count operations by type
            operation_counts = defaultdict(int)
            hourly_counts = defaultdict(int)
            
            for op in recent_ops:
                operation_type = op.get('operation', 'unknown')
                operation_counts[operation_type] += 1
                
                # Group by hour
                timestamp = op.get('timestamp', 0)
                hour_key = int(timestamp // 3600) * 3600  # Round to hour
                hourly_counts[hour_key] += 1
            
            return {
                'total_operations': len(recent_ops),
                'operation_breakdown': dict(operation_counts),
                'hourly_distribution': dict(hourly_counts),
                'unique_operations': len(operation_counts),
                'peak_hour': max(hourly_counts.items(), key=lambda x: x[1])[0] if hourly_counts else None,
                'time_range': time_range
            }
            
        except Exception as e:
            logger.error(f"Failed to get usage statistics: {e}")
            return {}

    def _parse_time_range(self, time_range: str) -> int:
        """Parse time range string to seconds."""
        try:
            if time_range.endswith('h'):
                return int(time_range[:-1]) * 3600
            elif time_range.endswith('d'):
                return int(time_range[:-1]) * 86400
            elif time_range.endswith('m'):
                return int(time_range[:-1]) * 60
            elif time_range.endswith('s'):
                return int(time_range[:-1])
            else:
                # Default to hours
                return int(time_range) * 3600
        except (ValueError, IndexError):
            # Default to 24 hours
            return 86400