"""
Advanced Intelligence Flow Manager - Phase 2
Real-time cross-service intelligence sharing and pattern recognition
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid

from .orchestrator import (
    IntelligenceOrchestrator, 
    IntelligenceEvent, 
    IntelligenceEventType,
    IntelligencePriority
)

logger = logging.getLogger(__name__)


class FlowType(Enum):
    """Types of intelligence flows."""
    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAMING = "streaming"
    EVENT_DRIVEN = "event_driven"


class FlowDirection(Enum):
    """Direction of intelligence flow."""
    UPSTREAM = "upstream"
    DOWNSTREAM = "downstream"
    BIDIRECTIONAL = "bidirectional"
    BROADCAST = "broadcast"


@dataclass
class IntelligenceFlow:
    """Definition of an intelligence flow between services."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    source_service: str = ""
    target_services: List[str] = field(default_factory=list)
    flow_type: FlowType = FlowType.REAL_TIME
    direction: FlowDirection = FlowDirection.BIDIRECTIONAL
    event_types: List[IntelligenceEventType] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    transformations: List[str] = field(default_factory=list)
    priority: IntelligencePriority = IntelligencePriority.MEDIUM
    enabled: bool = True
    created_at: float = field(default_factory=time.time)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceIntelligenceProfile:
    """Intelligence profile for a service."""
    service_name: str = ""
    capabilities: List[str] = field(default_factory=list)
    data_types: List[str] = field(default_factory=list)
    intelligence_patterns: List[str] = field(default_factory=list)
    learning_preferences: Dict[str, Any] = field(default_factory=dict)
    contribution_score: float = 0.0
    consumption_score: float = 0.0
    last_activity: float = field(default_factory=time.time)


class AdvancedIntelligenceFlowManager:
    """
    Advanced intelligence flow manager for sophisticated cross-service learning.
    """
    
    def __init__(self, kse_client, orchestrator: IntelligenceOrchestrator):
        """Initialize advanced intelligence flow manager."""
        self.kse_client = kse_client
        self.orchestrator = orchestrator
        self.flows: Dict[str, IntelligenceFlow] = {}
        self.service_profiles: Dict[str, ServiceIntelligenceProfile] = {}
        self.active_streams: Dict[str, asyncio.Task] = {}
        self.flow_metrics: Dict[str, Dict[str, Any]] = {}
        self.learning_models: Dict[str, Any] = {}
        
        # Configuration
        self.max_concurrent_flows = 50
        self.flow_timeout_seconds = 300
        self.pattern_learning_enabled = True
        self.adaptive_routing_enabled = True
        
    async def initialize(self) -> bool:
        """Initialize the flow manager."""
        try:
            # Initialize service profiles for the four microservices
            await self._initialize_service_profiles()
            
            # Create default intelligence flows
            await self._create_default_flows()
            
            # Start flow monitoring
            asyncio.create_task(self._monitor_flows())
            
            logger.info("Advanced intelligence flow manager initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize flow manager: {e}")
            return False
    
    async def create_intelligence_flow(self, flow: IntelligenceFlow) -> bool:
        """
        Create a new intelligence flow.
        
        Args:
            flow: Intelligence flow definition
            
        Returns:
            True if created successfully
        """
        try:
            # Validate flow
            if not flow.source_service or not flow.target_services:
                logger.error("Flow must have source and target services")
                return False
            
            # Check if services exist
            if flow.source_service not in self.service_profiles:
                await self._create_service_profile(flow.source_service)
            
            for target in flow.target_services:
                if target not in self.service_profiles:
                    await self._create_service_profile(target)
            
            # Store flow
            self.flows[flow.id] = flow
            
            # Start flow if enabled
            if flow.enabled:
                await self._start_flow(flow)
            
            logger.info(f"Created intelligence flow: {flow.name} ({flow.id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create intelligence flow: {e}")
            return False
    
    async def enable_real_time_learning(self, service_name: str, 
                                       learning_config: Dict[str, Any]) -> bool:
        """
        Enable real-time learning for a service.
        
        Args:
            service_name: Name of the service
            learning_config: Learning configuration
            
        Returns:
            True if enabled successfully
        """
        try:
            if service_name not in self.service_profiles:
                await self._create_service_profile(service_name)
            
            profile = self.service_profiles[service_name]
            profile.learning_preferences.update(learning_config)
            
            # Create learning flow
            learning_flow = IntelligenceFlow(
                name=f"{service_name}_learning_flow",
                source_service=service_name,
                target_services=["*"],  # Broadcast to all services
                flow_type=FlowType.REAL_TIME,
                direction=FlowDirection.BROADCAST,
                event_types=[
                    IntelligenceEventType.PATTERN_DISCOVERED,
                    IntelligenceEventType.INSIGHT_GENERATED,
                    IntelligenceEventType.OPTIMIZATION_SUGGESTED
                ],
                priority=IntelligencePriority.HIGH
            )
            
            await self.create_intelligence_flow(learning_flow)
            
            logger.info(f"Enabled real-time learning for {service_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enable real-time learning: {e}")
            return False
    
    async def register_service_capabilities(self, service_name: str,
                                          capabilities: Dict[str, float],
                                          input_types: List[str],
                                          output_types: List[str]) -> bool:
        """
        Register service capabilities for intelligence routing.
        
        Args:
            service_name: Name of the service
            capabilities: Dictionary of capability names to confidence scores
            input_types: List of input data types the service can handle
            output_types: List of output data types the service produces
            
        Returns:
            True if registration successful
        """
        try:
            if service_name not in self.service_profiles:
                await self._create_service_profile(service_name)
            
            profile = self.service_profiles[service_name]
            profile.capabilities = list(capabilities.keys())
            profile.data_types = input_types + output_types
            profile.contribution_score = sum(capabilities.values()) / len(capabilities)
            profile.last_activity = time.time()
            
            logger.info(f"Registered capabilities for service: {service_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register service capabilities: {e}")
            return False
    
    async def optimize_flow(self, flow_type: str, source_service: str,
                          target_services: List[str], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Optimize intelligence flow configuration.
        
        Args:
            flow_type: Type of flow to optimize
            source_service: Source service name
            target_services: List of target service names
            context: Context for optimization
            
        Returns:
            Optimized flow configuration or None if failed
        """
        try:
            # Create a test flow
            flow = IntelligenceFlow(
                name=f"{source_service}_to_{'_'.join(target_services)}_optimized",
                source_service=source_service,
                target_services=target_services,
                flow_type=FlowType.REAL_TIME if flow_type == "real_time" else FlowType.BATCH,
                direction=FlowDirection.BIDIRECTIONAL,
                event_types=[IntelligenceEventType.INSIGHT_GENERATED],
                priority=IntelligencePriority.MEDIUM
            )
            
            # Analyze and optimize
            performance = await self._analyze_flow_performance(flow)
            await self._optimize_flow_parameters(flow, performance)
            
            return {
                "flow_id": flow.id,
                "flow_name": flow.name,
                "optimized_parameters": {
                    "flow_type": flow.flow_type.value,
                    "direction": flow.direction.value,
                    "priority": flow.priority.value
                },
                "expected_performance": performance
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize flow: {e}")
            return None
    
    async def discover_intelligence_opportunities(self,
                                                service_name: str) -> List[Dict[str, Any]]:
        """
        Discover intelligence sharing opportunities for a service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            List of intelligence opportunities
        """
        try:
            opportunities = []
            
            if service_name not in self.service_profiles:
                return opportunities
            
            profile = self.service_profiles[service_name]
            
            # Analyze potential collaborations
            for other_service, other_profile in self.service_profiles.items():
                if other_service == service_name:
                    continue
                
                try:
                    # Check for complementary capabilities
                    complementary_score = await self._calculate_complementary_score(
                        profile, other_profile
                    )
                    
                    if complementary_score > 0.7:
                        benefits = await self._identify_collaboration_benefits(
                            profile, other_profile
                        )
                        
                        opportunities.append({
                            'type': 'collaboration',
                            'target_service': other_service,
                            'score': float(complementary_score),
                            'suggested_flow_type': FlowDirection.BIDIRECTIONAL.value,
                            'potential_benefits': benefits
                        })
                except Exception as e:
                    logger.error(f"Error analyzing collaboration for {other_service}: {e}")
                    continue
            
            # Check for learning opportunities
            learning_opportunities = await self._identify_learning_opportunities(profile)
            opportunities.extend(learning_opportunities)
            
            # Sort by potential impact (ensure all scores are numeric)
            for opp in opportunities:
                if 'score' not in opp or not isinstance(opp['score'], (int, float)):
                    opp['score'] = 0.0
            
            opportunities.sort(key=lambda x: float(x.get('score', 0)), reverse=True)
            
            return opportunities[:10]  # Top 10 opportunities
            
        except Exception as e:
            logger.error(f"Failed to discover intelligence opportunities: {e}")
            return []
    
    async def optimize_intelligence_flows(self) -> Dict[str, Any]:
        """
        Optimize existing intelligence flows based on performance metrics.
        
        Returns:
            Optimization results
        """
        try:
            optimization_results = {
                'flows_optimized': 0,
                'flows_disabled': 0,
                'new_flows_suggested': 0,
                'performance_improvement': 0.0
            }
            
            for flow_id, flow in self.flows.items():
                if not flow.enabled:
                    continue
                
                # Analyze flow performance
                performance = await self._analyze_flow_performance(flow)
                
                if performance['efficiency'] < 0.5:
                    # Consider disabling low-performing flow
                    flow.enabled = False
                    optimization_results['flows_disabled'] += 1
                    logger.info(f"Disabled low-performing flow: {flow.name}")
                
                elif performance['efficiency'] > 0.8:
                    # Optimize high-performing flow
                    await self._optimize_flow_parameters(flow, performance)
                    optimization_results['flows_optimized'] += 1
            
            # Suggest new flows based on patterns
            new_flow_suggestions = await self._suggest_new_flows()
            optimization_results['new_flows_suggested'] = len(new_flow_suggestions)
            
            logger.info(f"Flow optimization completed: {optimization_results}")
            return optimization_results
            
        except Exception as e:
            logger.error(f"Failed to optimize intelligence flows: {e}")
            return {}
    
    async def get_cross_service_insights(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Get cross-service intelligence insights.
        
        Args:
            time_window_hours: Time window for analysis
            
        Returns:
            Cross-service insights
        """
        try:
            insights = {
                'service_interactions': {},
                'intelligence_patterns': [],
                'collaboration_effectiveness': {},
                'learning_progress': {},
                'optimization_opportunities': []
            }
            
            cutoff_time = time.time() - (time_window_hours * 3600)
            
            # Analyze service interactions
            for flow_id, flow in self.flows.items():
                if flow.enabled and flow.created_at > cutoff_time:
                    interaction_key = f"{flow.source_service}->{','.join(flow.target_services)}"
                    
                    if interaction_key not in insights['service_interactions']:
                        insights['service_interactions'][interaction_key] = {
                            'frequency': 0,
                            'effectiveness': 0.0,
                            'data_volume': 0
                        }
                    
                    metrics = self.flow_metrics.get(flow_id, {})
                    insights['service_interactions'][interaction_key]['frequency'] += 1
                    insights['service_interactions'][interaction_key]['effectiveness'] += metrics.get('effectiveness', 0.0)
                    insights['service_interactions'][interaction_key]['data_volume'] += metrics.get('data_volume', 0)
            
            # Get intelligence patterns from orchestrator
            patterns = await self.orchestrator.discover_cross_service_patterns(time_window_hours)
            insights['intelligence_patterns'] = [
                {
                    'pattern_type': pattern.pattern_type,
                    'confidence': pattern.confidence,
                    'services_involved': pattern.services_involved,
                    'impact': pattern.impact
                }
                for pattern in patterns
            ]
            
            # Calculate collaboration effectiveness
            for service_name, profile in self.service_profiles.items():
                insights['collaboration_effectiveness'][service_name] = {
                    'contribution_score': profile.contribution_score,
                    'consumption_score': profile.consumption_score,
                    'balance_ratio': profile.contribution_score / max(profile.consumption_score, 0.1)
                }
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to get cross-service insights: {e}")
            return {}
    
    async def _initialize_service_profiles(self):
        """Initialize service profiles for the four microservices."""
        services = {
            'surfacing': {
                'capabilities': ['data_discovery', 'pattern_recognition', 'anomaly_detection'],
                'data_types': ['structured', 'unstructured', 'time_series'],
                'intelligence_patterns': ['trend_analysis', 'outlier_detection', 'correlation_discovery']
            },
            'causal': {
                'capabilities': ['causal_inference', 'relationship_analysis', 'impact_assessment'],
                'data_types': ['causal_graphs', 'intervention_data', 'observational_data'],
                'intelligence_patterns': ['causal_discovery', 'effect_estimation', 'confounding_analysis']
            },
            'llm': {
                'capabilities': ['natural_language_processing', 'text_generation', 'semantic_analysis'],
                'data_types': ['text', 'conversations', 'documents'],
                'intelligence_patterns': ['language_understanding', 'context_awareness', 'knowledge_extraction']
            },
            'agentic': {
                'capabilities': ['decision_making', 'planning', 'goal_optimization'],
                'data_types': ['behavioral_data', 'decision_trees', 'optimization_results'],
                'intelligence_patterns': ['behavior_modeling', 'strategy_optimization', 'adaptive_learning']
            }
        }
        
        for service_name, config in services.items():
            profile = ServiceIntelligenceProfile(
                service_name=service_name,
                capabilities=config['capabilities'],
                data_types=config['data_types'],
                intelligence_patterns=config['intelligence_patterns']
            )
            self.service_profiles[service_name] = profile
    
    async def _create_default_flows(self):
        """Create default intelligence flows between services."""
        # Surfacing -> Causal flow
        surfacing_causal_flow = IntelligenceFlow(
            name="surfacing_to_causal_patterns",
            source_service="surfacing",
            target_services=["causal"],
            flow_type=FlowType.REAL_TIME,
            direction=FlowDirection.DOWNSTREAM,
            event_types=[IntelligenceEventType.PATTERN_DISCOVERED, IntelligenceEventType.ANOMALY_DETECTED],
            priority=IntelligencePriority.HIGH
        )
        await self.create_intelligence_flow(surfacing_causal_flow)
        
        # Causal -> LLM flow
        causal_llm_flow = IntelligenceFlow(
            name="causal_to_llm_insights",
            source_service="causal",
            target_services=["llm"],
            flow_type=FlowType.EVENT_DRIVEN,
            direction=FlowDirection.DOWNSTREAM,
            event_types=[IntelligenceEventType.CAUSAL_RELATIONSHIP, IntelligenceEventType.INSIGHT_GENERATED],
            priority=IntelligencePriority.MEDIUM
        )
        await self.create_intelligence_flow(causal_llm_flow)
        
        # LLM -> Agentic flow
        llm_agentic_flow = IntelligenceFlow(
            name="llm_to_agentic_decisions",
            source_service="llm",
            target_services=["agentic"],
            flow_type=FlowType.REAL_TIME,
            direction=FlowDirection.DOWNSTREAM,
            event_types=[IntelligenceEventType.INSIGHT_GENERATED, IntelligenceEventType.OPTIMIZATION_SUGGESTED],
            priority=IntelligencePriority.HIGH
        )
        await self.create_intelligence_flow(llm_agentic_flow)
        
        # Agentic -> All services feedback flow
        agentic_feedback_flow = IntelligenceFlow(
            name="agentic_feedback_loop",
            source_service="agentic",
            target_services=["surfacing", "causal", "llm"],
            flow_type=FlowType.BATCH,
            direction=FlowDirection.BROADCAST,
            event_types=[IntelligenceEventType.OPTIMIZATION_SUGGESTED, IntelligenceEventType.BEHAVIORAL_PATTERN],
            priority=IntelligencePriority.MEDIUM
        )
        await self.create_intelligence_flow(agentic_feedback_flow)
    
    async def _create_service_profile(self, service_name: str):
        """Create a basic service profile."""
        profile = ServiceIntelligenceProfile(service_name=service_name)
        self.service_profiles[service_name] = profile
    
    async def _start_flow(self, flow: IntelligenceFlow):
        """Start an intelligence flow."""
        try:
            if flow.flow_type == FlowType.REAL_TIME:
                task = asyncio.create_task(self._handle_real_time_flow(flow))
                self.active_streams[flow.id] = task
            elif flow.flow_type == FlowType.STREAMING:
                task = asyncio.create_task(self._handle_streaming_flow(flow))
                self.active_streams[flow.id] = task
            
            logger.debug(f"Started flow: {flow.name}")
            
        except Exception as e:
            logger.error(f"Failed to start flow: {e}")
    
    async def _handle_real_time_flow(self, flow: IntelligenceFlow):
        """Handle real-time intelligence flow."""
        # Implementation would handle real-time event processing
        pass
    
    async def _handle_streaming_flow(self, flow: IntelligenceFlow):
        """Handle streaming intelligence flow."""
        # Implementation would handle streaming data processing
        pass
    
    async def _monitor_flows(self):
        """Monitor active flows."""
        while True:
            try:
                await asyncio.sleep(60)  # Monitor every minute
                
                # Update flow metrics
                for flow_id, flow in self.flows.items():
                    if flow.enabled:
                        await self._update_flow_metrics(flow)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in flow monitoring: {e}")
    
    async def _calculate_complementary_score(self, profile1: ServiceIntelligenceProfile, 
                                           profile2: ServiceIntelligenceProfile) -> float:
        """Calculate complementary score between two service profiles."""
        # Simple implementation - would be more sophisticated in practice
        common_capabilities = set(profile1.capabilities) & set(profile2.capabilities)
        total_capabilities = set(profile1.capabilities) | set(profile2.capabilities)
        
        if not total_capabilities:
            return 0.0
        
        # Higher score for complementary (non-overlapping) capabilities
        complementary_ratio = 1.0 - (len(common_capabilities) / len(total_capabilities))
        return complementary_ratio
    
    async def _identify_collaboration_benefits(self, profile1: ServiceIntelligenceProfile,
                                             profile2: ServiceIntelligenceProfile) -> List[str]:
        """Identify potential collaboration benefits."""
        benefits = []
        
        # Check for complementary data types
        if set(profile1.data_types) != set(profile2.data_types):
            benefits.append("Cross-domain data enrichment")
        
        # Check for complementary capabilities
        unique_capabilities = set(profile1.capabilities) ^ set(profile2.capabilities)
        if unique_capabilities:
            benefits.append("Capability enhancement")
        
        return benefits
    
    async def _identify_learning_opportunities(self, profile: ServiceIntelligenceProfile) -> List[Dict[str, Any]]:
        """Identify learning opportunities for a service."""
        opportunities = []
        
        # Check for underutilized patterns
        if profile.consumption_score < profile.contribution_score:
            opportunities.append({
                'type': 'learning',
                'description': 'Increase intelligence consumption',
                'score': 0.8,
                'suggested_action': 'Subscribe to more intelligence events'
            })
        
        return opportunities
    
    async def _analyze_flow_performance(self, flow: IntelligenceFlow) -> Dict[str, Any]:
        """Analyze performance of an intelligence flow."""
        metrics = self.flow_metrics.get(flow.id, {})
        
        return {
            'efficiency': metrics.get('efficiency', 0.5),
            'throughput': metrics.get('throughput', 0),
            'error_rate': metrics.get('error_rate', 0.0),
            'latency': metrics.get('latency', 0.0)
        }
    
    async def _optimize_flow_parameters(self, flow: IntelligenceFlow, performance: Dict[str, Any]):
        """Optimize flow parameters based on performance."""
        # Implementation would adjust flow parameters
        pass
    
    async def _suggest_new_flows(self) -> List[IntelligenceFlow]:
        """Suggest new intelligence flows based on patterns."""
        suggestions = []
        # Implementation would analyze patterns and suggest new flows
        return suggestions
    
    async def _update_flow_metrics(self, flow: IntelligenceFlow):
        """Update metrics for a flow."""
        if flow.id not in self.flow_metrics:
            self.flow_metrics[flow.id] = {}
        
        # Update basic metrics
        self.flow_metrics[flow.id]['last_updated'] = time.time()
        self.flow_metrics[flow.id]['efficiency'] = 0.8  # Placeholder
    
    async def trigger_real_time_learning(self, context: Dict[str, Any]) -> bool:
        """
        Trigger real-time learning based on context.
        
        Args:
            context: Learning context with service, operation, and data points
            
        Returns:
            True if learning was triggered successfully
        """
        try:
            service_name = context.get("service", "unknown")
            operation = context.get("operation", "unknown")
            data_points = context.get("data_points", 0)
            
            # Create a learning event
            learning_event = IntelligenceEvent(
                event_type=IntelligenceEventType.INSIGHT_GENERATED,
                source_service=service_name,
                target_services=["*"],  # Broadcast to all services
                data={
                    "learning_context": context,
                    "operation": operation,
                    "data_points": data_points,
                    "learning_type": "real_time",
                    "timestamp": time.time()
                },
                priority=IntelligencePriority.MEDIUM
            )
            
            # Publish the learning event through the orchestrator
            await self.orchestrator.publish_intelligence_event(learning_event)
            
            # Update service profile with learning activity
            if service_name in self.service_profiles:
                profile = self.service_profiles[service_name]
                profile.last_activity = time.time()
                profile.consumption_score += 0.1  # Increase learning score
            
            logger.info(f"Triggered real-time learning for {service_name}: {operation}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to trigger real-time learning: {e}")
            return False