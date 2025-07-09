"""
KSE Intelligence Orchestrator - Phase 2 Advanced Intelligence Flow
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class IntelligenceEventType(Enum):
    """Types of intelligence events."""
    PATTERN_DISCOVERED = "pattern_discovered"
    INSIGHT_GENERATED = "insight_generated"
    ANOMALY_DETECTED = "anomaly_detected"
    CORRELATION_FOUND = "correlation_found"
    PREDICTION_MADE = "prediction_made"
    OPTIMIZATION_SUGGESTED = "optimization_suggested"
    CAUSAL_RELATIONSHIP = "causal_relationship"
    BEHAVIORAL_PATTERN = "behavioral_pattern"


class IntelligencePriority(Enum):
    """Priority levels for intelligence events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class IntelligenceEvent:
    """Intelligence event for cross-service sharing."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: IntelligenceEventType = IntelligenceEventType.INSIGHT_GENERATED
    source_service: str = ""
    target_services: List[str] = field(default_factory=list)
    priority: IntelligencePriority = IntelligencePriority.MEDIUM
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    organization_id: str = ""
    domain: str = "general"
    confidence_score: float = 0.0
    impact_score: float = 0.0
    processed_by: Set[str] = field(default_factory=set)
    expires_at: Optional[float] = None


@dataclass
class IntelligencePattern:
    """Discovered intelligence pattern."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pattern_type: str = ""
    description: str = ""
    services_involved: List[str] = field(default_factory=list)
    frequency: int = 0
    confidence: float = 0.0
    impact: float = 0.0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    data_points: List[Dict[str, Any]] = field(default_factory=list)
    correlations: Dict[str, float] = field(default_factory=dict)


class IntelligenceOrchestrator:
    """
    Advanced intelligence orchestrator for cross-service learning and pattern recognition.
    """
    
    def __init__(self, kse_client):
        """Initialize intelligence orchestrator."""
        self.kse_client = kse_client
        self.event_queue = asyncio.Queue()
        self.subscribers: Dict[str, List[Callable]] = {}
        self.patterns: Dict[str, IntelligencePattern] = {}
        self.active_correlations: Dict[str, Dict[str, Any]] = {}
        self.service_metrics: Dict[str, Dict[str, Any]] = {}
        self.intelligence_cache: Dict[str, Any] = {}
        self.processing_task: Optional[asyncio.Task] = None
        self.pattern_detection_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.max_events_per_minute = 1000
        self.pattern_detection_interval = 30  # seconds
        self.correlation_threshold = 0.7
        self.pattern_confidence_threshold = 0.8
        
    async def initialize(self) -> bool:
        """Initialize the intelligence orchestrator."""
        try:
            # Start background processing tasks
            self.processing_task = asyncio.create_task(self._process_intelligence_events())
            self.pattern_detection_task = asyncio.create_task(self._detect_patterns())
            
            logger.info("Intelligence orchestrator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize intelligence orchestrator: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown the intelligence orchestrator."""
        try:
            if self.processing_task:
                self.processing_task.cancel()
            if self.pattern_detection_task:
                self.pattern_detection_task.cancel()
                
            logger.info("Intelligence orchestrator shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during orchestrator shutdown: {e}")
    
    async def publish_intelligence(self, event: IntelligenceEvent) -> bool:
        """
        Publish intelligence event for cross-service sharing.
        
        Args:
            event: Intelligence event to publish
            
        Returns:
            True if published successfully
        """
        try:
            # Validate event
            if not event.source_service:
                logger.warning("Intelligence event missing source service")
                return False
            
            # Set expiration if not set
            if not event.expires_at:
                event.expires_at = time.time() + 3600  # 1 hour default
            
            # Add to queue for processing
            await self.event_queue.put(event)
            
            # Update service metrics
            await self._update_service_metrics(event.source_service, "events_published")
            
            logger.debug(f"Published intelligence event {event.id} from {event.source_service}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish intelligence event: {e}")
            return False
    
    async def subscribe_to_intelligence(self, service_name: str, 
                                      event_types: List[IntelligenceEventType],
                                      callback: Callable) -> bool:
        """
        Subscribe service to intelligence events.
        
        Args:
            service_name: Name of subscribing service
            event_types: Types of events to subscribe to
            callback: Callback function for events
            
        Returns:
            True if subscribed successfully
        """
        try:
            for event_type in event_types:
                subscription_key = f"{service_name}:{event_type.value}"
                
                if subscription_key not in self.subscribers:
                    self.subscribers[subscription_key] = []
                
                self.subscribers[subscription_key].append(callback)
            
            logger.info(f"Service {service_name} subscribed to {len(event_types)} intelligence event types")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe service to intelligence: {e}")
            return False
    
    async def subscribe_to_event(self, event_type: IntelligenceEventType,
                               handler: Callable[[IntelligenceEvent], None]) -> bool:
        """
        Subscribe to intelligence events (alias for subscribe_to_intelligence).
        
        Args:
            event_type: Type of events to subscribe to
            handler: Event handler function
            
        Returns:
            True if subscription successful
        """
        # Use wildcard service name for global subscription
        return await self.subscribe_to_intelligence("*", [event_type], handler)
    
    async def publish_event(self, event: IntelligenceEvent) -> bool:
        """
        Publish intelligence event (alias for publish_intelligence).
        
        Args:
            event: Intelligence event to publish
            
        Returns:
            True if published successfully
        """
        return await self.publish_intelligence(event)
    
    async def get_intelligence_insights(self, service_name: str,
                                      domain: str = "general",
                                      limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get relevant intelligence insights for a service.
        
        Args:
            service_name: Name of requesting service
            domain: Domain to filter insights
            limit: Maximum number of insights
            
        Returns:
            List of intelligence insights
        """
        try:
            insights = []
            
            # Get recent patterns relevant to the service
            relevant_patterns = [
                pattern for pattern in self.patterns.values()
                if (service_name in pattern.services_involved or 
                    pattern.confidence >= self.pattern_confidence_threshold)
            ]
            
            # Sort by impact and recency
            relevant_patterns.sort(
                key=lambda p: (p.impact, p.last_seen), 
                reverse=True
            )
            
            for pattern in relevant_patterns[:limit]:
                insights.append({
                    'type': 'pattern',
                    'id': pattern.id,
                    'pattern_type': pattern.pattern_type,
                    'description': pattern.description,
                    'confidence': pattern.confidence,
                    'impact': pattern.impact,
                    'services_involved': pattern.services_involved,
                    'correlations': pattern.correlations,
                    'last_seen': pattern.last_seen
                })
            
            # Get active correlations
            for correlation_id, correlation in self.active_correlations.items():
                if service_name in correlation.get('services', []):
                    insights.append({
                        'type': 'correlation',
                        'id': correlation_id,
                        'description': correlation.get('description', ''),
                        'strength': correlation.get('strength', 0.0),
                        'services': correlation.get('services', []),
                        'discovered_at': correlation.get('discovered_at', time.time())
                    })
            
            return insights[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get intelligence insights: {e}")
            return []
    
    async def discover_cross_service_patterns(self, 
                                            time_window_hours: int = 24) -> List[IntelligencePattern]:
        """
        Discover patterns across multiple services.
        
        Args:
            time_window_hours: Time window for pattern analysis
            
        Returns:
            List of discovered patterns
        """
        try:
            discovered_patterns = []
            cutoff_time = time.time() - (time_window_hours * 3600)
            
            # Analyze service interaction patterns
            service_interactions = await self._analyze_service_interactions(cutoff_time)
            
            # Detect behavioral patterns
            behavioral_patterns = await self._detect_behavioral_patterns(cutoff_time)
            
            # Detect performance patterns
            performance_patterns = await self._detect_performance_patterns(cutoff_time)
            
            # Combine all patterns
            discovered_patterns.extend(service_interactions)
            discovered_patterns.extend(behavioral_patterns)
            discovered_patterns.extend(performance_patterns)
            
            # Store significant patterns
            for pattern in discovered_patterns:
                if pattern.confidence >= self.pattern_confidence_threshold:
                    self.patterns[pattern.id] = pattern
            
            logger.info(f"Discovered {len(discovered_patterns)} cross-service patterns")
            return discovered_patterns
            
        except Exception as e:
            logger.error(f"Failed to discover cross-service patterns: {e}")
            return []
    
    async def _process_intelligence_events(self):
        """Background task to process intelligence events."""
        while True:
            try:
                # Get event from queue
                event = await self.event_queue.get()
                
                # Check if event has expired
                if event.expires_at and time.time() > event.expires_at:
                    continue
                
                # Process event
                await self._handle_intelligence_event(event)
                
                # Mark task as done
                self.event_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing intelligence event: {e}")
    
    async def _handle_intelligence_event(self, event: IntelligenceEvent):
        """Handle individual intelligence event."""
        try:
            # Store event in KSE memory
            await self._store_intelligence_event(event)
            
            # Notify subscribers
            await self._notify_subscribers(event)
            
            # Update correlations
            await self._update_correlations(event)
            
            # Check for immediate patterns
            await self._check_immediate_patterns(event)
            
            logger.debug(f"Processed intelligence event {event.id}")
            
        except Exception as e:
            logger.error(f"Failed to handle intelligence event: {e}")
    
    async def _store_intelligence_event(self, event: IntelligenceEvent):
        """Store intelligence event in KSE memory."""
        try:
            # Create entity for the intelligence event
            entity_data = {
                'id': event.id,
                'type': 'intelligence_event',
                'event_type': event.event_type.value,
                'source_service': event.source_service,
                'target_services': event.target_services,
                'priority': event.priority.value,
                'data': event.data,
                'metadata': event.metadata,
                'timestamp': event.timestamp,
                'organization_id': event.organization_id,
                'domain': event.domain,
                'confidence_score': event.confidence_score,
                'impact_score': event.impact_score
            }
            
            # Store in KSE memory (would use actual KSE client methods)
            # await self.kse_client.store_entity(entity_data)
            
        except Exception as e:
            logger.error(f"Failed to store intelligence event: {e}")
    
    async def _notify_subscribers(self, event: IntelligenceEvent):
        """Notify subscribers of intelligence event."""
        try:
            subscription_key = f"*:{event.event_type.value}"
            
            # Notify global subscribers
            if subscription_key in self.subscribers:
                for callback in self.subscribers[subscription_key]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(event)
                        else:
                            callback(event)
                    except Exception as e:
                        logger.error(f"Error in subscriber callback: {e}")
            
            # Notify target service subscribers
            for target_service in event.target_services:
                target_key = f"{target_service}:{event.event_type.value}"
                if target_key in self.subscribers:
                    for callback in self.subscribers[target_key]:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(event)
                            else:
                                callback(event)
                        except Exception as e:
                            logger.error(f"Error in target subscriber callback: {e}")
            
        except Exception as e:
            logger.error(f"Failed to notify subscribers: {e}")
    
    async def _detect_patterns(self):
        """Background task for pattern detection."""
        while True:
            try:
                await asyncio.sleep(self.pattern_detection_interval)
                
                # Discover new patterns
                await self.discover_cross_service_patterns()
                
                # Clean up old patterns
                await self._cleanup_old_patterns()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in pattern detection: {e}")
    
    async def _analyze_service_interactions(self, cutoff_time: float) -> List[IntelligencePattern]:
        """Analyze service interaction patterns."""
        patterns = []
        
        # This would analyze actual service interaction data
        # For now, return empty list as placeholder
        
        return patterns
    
    async def _detect_behavioral_patterns(self, cutoff_time: float) -> List[IntelligencePattern]:
        """Detect behavioral patterns."""
        patterns = []
        
        # This would analyze behavioral data
        # For now, return empty list as placeholder
        
        return patterns
    
    async def _detect_performance_patterns(self, cutoff_time: float) -> List[IntelligencePattern]:
        """Detect performance patterns."""
        patterns = []
        
        # This would analyze performance metrics
        # For now, return empty list as placeholder
        
        return patterns
    
    async def _update_correlations(self, event: IntelligenceEvent):
        """Update correlation tracking."""
        try:
            # This would implement correlation analysis
            pass
            
        except Exception as e:
            logger.error(f"Failed to update correlations: {e}")
    
    async def _check_immediate_patterns(self, event: IntelligenceEvent):
        """Check for immediate patterns from new event."""
        try:
            # This would implement immediate pattern detection
            pass
            
        except Exception as e:
            logger.error(f"Failed to check immediate patterns: {e}")
    
    async def _update_service_metrics(self, service_name: str, metric: str):
        """Update service metrics."""
        try:
            if service_name not in self.service_metrics:
                self.service_metrics[service_name] = {}
            
            if metric not in self.service_metrics[service_name]:
                self.service_metrics[service_name][metric] = 0
            
            self.service_metrics[service_name][metric] += 1
            
        except Exception as e:
            logger.error(f"Failed to update service metrics: {e}")
    
    async def _cleanup_old_patterns(self):
        """Clean up old patterns."""
        try:
            current_time = time.time()
            cutoff_time = current_time - (7 * 24 * 3600)  # 7 days
            
            patterns_to_remove = [
                pattern_id for pattern_id, pattern in self.patterns.items()
                if pattern.last_seen < cutoff_time
            ]
            
            for pattern_id in patterns_to_remove:
                del self.patterns[pattern_id]
            
            if patterns_to_remove:
                logger.info(f"Cleaned up {len(patterns_to_remove)} old patterns")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old patterns: {e}")