"""
Agent Manager for Agentic Module

Manages the lifecycle of marketing agents including creation, configuration,
deployment, and performance tracking.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import uuid

from models.agent_models import (
    MarketingAgent, MarketingAgentType, AgentCapability,
    ModelConfig, MarketingContext
)
from services.memory_service import MemoryService
from utils.config import AgenticConfig
from agents.marketing_agent_library import MarketingAgentLibrary

logger = logging.getLogger(__name__)


class AgentManager:
    """
    Manages marketing agents within the LiftOS ecosystem.
    
    This class handles agent registration, configuration, deployment,
    and performance monitoring for marketing analytics use cases.
    """
    
    def __init__(self, memory_service: MemoryService, config: AgenticConfig):
        """Initialize the agent manager."""
        self.memory_service = memory_service
        self.config = config
        self.agent_library = MarketingAgentLibrary()
        
        # In-memory cache for active agents
        self._agent_cache: Dict[str, MarketingAgent] = {}
        self._agent_tasks: Dict[str, asyncio.Task] = {}
        
        # Performance tracking
        self._performance_metrics: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Agent Manager initialized")
    
    async def load_default_agents(self) -> None:
        """Load default marketing agents from the library."""
        try:
            default_agents = await self.agent_library.get_default_agents()
            
            for agent in default_agents:
                # Check if agent already exists
                existing = await self.get_agent(agent.agent_id)
                if not existing:
                    await self.register_agent(agent)
                    logger.info(f"Loaded default agent: {agent.name}")
                else:
                    logger.debug(f"Agent {agent.name} already exists, skipping")
            
            logger.info(f"Loaded {len(default_agents)} default agents")
            
        except Exception as e:
            logger.error(f"Failed to load default agents: {e}")
            raise
    
    async def register_agent(self, agent: MarketingAgent) -> MarketingAgent:
        """Register a new agent in the system."""
        try:
            # Validate agent configuration
            await self._validate_agent_config(agent)
            
            # Generate ID if not provided
            if not agent.agent_id:
                agent.agent_id = f"agent_{uuid.uuid4().hex[:8]}"
            
            # Set timestamps
            agent.created_at = datetime.utcnow()
            agent.updated_at = datetime.utcnow()
            
            # Store in memory service
            await self.memory_service.store_agent(agent)
            
            # Add to cache
            self._agent_cache[agent.agent_id] = agent
            
            # Initialize performance metrics
            self._performance_metrics[agent.agent_id] = {
                "total_tasks": 0,
                "successful_tasks": 0,
                "failed_tasks": 0,
                "average_duration": 0.0,
                "last_activity": None
            }
            
            logger.info(f"Registered agent: {agent.name} ({agent.agent_id})")
            return agent
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent.name}: {e}")
            raise
    
    async def create_agent(self, agent: MarketingAgent) -> MarketingAgent:
        """Create and register a new agent."""
        return await self.register_agent(agent)
    
    async def get_agent(self, agent_id: str) -> Optional[MarketingAgent]:
        """Get an agent by ID."""
        try:
            # Check cache first
            if agent_id in self._agent_cache:
                return self._agent_cache[agent_id]
            
            # Load from memory service
            agent = await self.memory_service.get_agent(agent_id)
            if agent:
                self._agent_cache[agent_id] = agent
            
            return agent
            
        except Exception as e:
            logger.error(f"Failed to get agent {agent_id}: {e}")
            return None
    
    async def list_agents(
        self, 
        agent_type: Optional[MarketingAgentType] = None,
        active_only: bool = True,
        capabilities: Optional[List[AgentCapability]] = None
    ) -> List[MarketingAgent]:
        """List agents with optional filtering."""
        try:
            agents = await self.memory_service.list_agents()
            
            # Apply filters
            filtered_agents = []
            for agent in agents:
                # Filter by type
                if agent_type and agent.agent_type != agent_type:
                    continue
                
                # Filter by active status
                if active_only and not agent.is_active:
                    continue
                
                # Filter by capabilities
                if capabilities:
                    if not all(cap in agent.capabilities for cap in capabilities):
                        continue
                
                filtered_agents.append(agent)
            
            # Update cache
            for agent in filtered_agents:
                self._agent_cache[agent.agent_id] = agent
            
            return filtered_agents
            
        except Exception as e:
            logger.error(f"Failed to list agents: {e}")
            return []
    
    async def update_agent(
        self, 
        agent_id: str, 
        updates: Dict[str, Any]
    ) -> Optional[MarketingAgent]:
        """Update an existing agent."""
        try:
            agent = await self.get_agent(agent_id)
            if not agent:
                return None
            
            # Apply updates
            for field, value in updates.items():
                if hasattr(agent, field):
                    setattr(agent, field, value)
            
            # Update timestamp
            agent.updated_at = datetime.utcnow()
            
            # Validate updated configuration
            await self._validate_agent_config(agent)
            
            # Store updated agent
            await self.memory_service.store_agent(agent)
            
            # Update cache
            self._agent_cache[agent_id] = agent
            
            logger.info(f"Updated agent: {agent.name} ({agent_id})")
            return agent
            
        except Exception as e:
            logger.error(f"Failed to update agent {agent_id}: {e}")
            raise
    
    async def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent."""
        try:
            # Stop any running tasks
            if agent_id in self._agent_tasks:
                task = self._agent_tasks[agent_id]
                if not task.done():
                    task.cancel()
                del self._agent_tasks[agent_id]
            
            # Remove from memory service
            success = await self.memory_service.delete_agent(agent_id)
            
            if success:
                # Remove from cache
                self._agent_cache.pop(agent_id, None)
                self._performance_metrics.pop(agent_id, None)
                
                logger.info(f"Deleted agent: {agent_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete agent {agent_id}: {e}")
            return False
    
    async def find_suitable_agents(
        self, 
        required_capabilities: List[AgentCapability],
        context: Optional[MarketingContext] = None,
        max_agents: int = 5
    ) -> List[MarketingAgent]:
        """Find agents suitable for specific requirements."""
        try:
            all_agents = await self.list_agents(active_only=True)
            suitable_agents = []
            
            for agent in all_agents:
                # Check capabilities
                if not all(cap in agent.capabilities for cap in required_capabilities):
                    continue
                
                # Check context compatibility
                if context and not agent.is_suitable_for_context(context):
                    continue
                
                # Check availability (not at max concurrent tasks)
                current_tasks = len([
                    task for task in self._agent_tasks.values() 
                    if not task.done()
                ])
                if current_tasks >= agent.max_concurrent_tasks:
                    continue
                
                suitable_agents.append(agent)
            
            # Sort by performance metrics
            suitable_agents.sort(
                key=lambda a: (
                    a.success_rate or 0.0,
                    -(a.average_task_duration or float('inf'))
                ),
                reverse=True
            )
            
            return suitable_agents[:max_agents]
            
        except Exception as e:
            logger.error(f"Failed to find suitable agents: {e}")
            return []
    
    async def execute_agent_task(
        self,
        agent_id: str,
        task_data: Dict[str, Any],
        timeout_seconds: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute a task using a specific agent."""
        try:
            agent = await self.get_agent(agent_id)
            if not agent:
                raise ValueError(f"Agent {agent_id} not found")
            
            if not agent.is_active:
                raise ValueError(f"Agent {agent_id} is not active")
            
            # Use agent timeout if not specified
            timeout = timeout_seconds or agent.timeout_seconds
            
            # Record task start
            start_time = datetime.utcnow()
            
            try:
                # Execute the task (this would integrate with AI services)
                result = await self._execute_task_with_timeout(
                    agent, task_data, timeout
                )
                
                # Record success
                duration = (datetime.utcnow() - start_time).total_seconds()
                await self._update_agent_performance(agent_id, duration, True)
                
                return result
                
            except asyncio.TimeoutError:
                logger.warning(f"Task timeout for agent {agent_id}")
                await self._update_agent_performance(agent_id, timeout, False)
                raise
            except Exception as e:
                duration = (datetime.utcnow() - start_time).total_seconds()
                await self._update_agent_performance(agent_id, duration, False)
                raise
            
        except Exception as e:
            logger.error(f"Failed to execute task for agent {agent_id}: {e}")
            raise
    
    async def get_agent_performance(self, agent_id: str) -> Dict[str, Any]:
        """Get performance metrics for an agent."""
        try:
            agent = await self.get_agent(agent_id)
            if not agent:
                return {}
            
            metrics = self._performance_metrics.get(agent_id, {})
            
            return {
                "agent_id": agent_id,
                "agent_name": agent.name,
                "total_tasks_completed": agent.total_tasks_completed,
                "success_rate": agent.success_rate,
                "average_task_duration": agent.average_task_duration,
                "last_used": agent.last_used,
                "current_metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance for agent {agent_id}: {e}")
            return {}
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            # Cancel all running tasks
            for task in self._agent_tasks.values():
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self._agent_tasks:
                await asyncio.gather(
                    *self._agent_tasks.values(), 
                    return_exceptions=True
                )
            
            # Clear caches
            self._agent_cache.clear()
            self._agent_tasks.clear()
            self._performance_metrics.clear()
            
            logger.info("Agent Manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during Agent Manager cleanup: {e}")
    
    async def _validate_agent_config(self, agent: MarketingAgent) -> None:
        """Validate agent configuration."""
        # Check required fields
        if not agent.name or not agent.description:
            raise ValueError("Agent name and description are required")
        
        if not agent.capabilities:
            raise ValueError("Agent must have at least one capability")
        
        if not agent.system_prompt:
            raise ValueError("Agent must have a system prompt")
        
        # Validate model configuration
        if not agent.model_config.provider or not agent.model_config.model_name:
            raise ValueError("Agent must have valid model configuration")
        
        # Validate timeout settings
        if agent.timeout_seconds <= 0:
            raise ValueError("Agent timeout must be positive")
        
        if agent.max_concurrent_tasks <= 0:
            raise ValueError("Max concurrent tasks must be positive")
    
    async def _execute_task_with_timeout(
        self,
        agent: MarketingAgent,
        task_data: Dict[str, Any],
        timeout_seconds: int
    ) -> Dict[str, Any]:
        """Execute a task with timeout."""
        # This is a placeholder for actual AI service integration
        # In a real implementation, this would:
        # 1. Format the task data according to the agent's requirements
        # 2. Call the appropriate AI service (OpenAI, Anthropic, etc.)
        # 3. Process the response and return structured results
        
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return {
            "status": "completed",
            "agent_id": agent.agent_id,
            "task_type": task_data.get("type", "unknown"),
            "result": "Task completed successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _update_agent_performance(
        self,
        agent_id: str,
        duration: float,
        success: bool
    ) -> None:
        """Update agent performance metrics."""
        try:
            # Update agent model
            agent = await self.get_agent(agent_id)
            if agent:
                agent.update_performance_metrics(duration, success)
                await self.memory_service.store_agent(agent)
            
            # Update local metrics
            if agent_id in self._performance_metrics:
                metrics = self._performance_metrics[agent_id]
                metrics["total_tasks"] += 1
                if success:
                    metrics["successful_tasks"] += 1
                else:
                    metrics["failed_tasks"] += 1
                
                # Update average duration (exponential moving average)
                if metrics["average_duration"] == 0:
                    metrics["average_duration"] = duration
                else:
                    alpha = 0.1
                    metrics["average_duration"] = (
                        alpha * duration + 
                        (1 - alpha) * metrics["average_duration"]
                    )
                
                metrics["last_activity"] = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Failed to update performance for agent {agent_id}: {e}")