"""
Agent Models for Agentic Module

Defines data structures for marketing agents, their configurations,
capabilities, and operational contexts.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


class MarketingAgentType(str, Enum):
    """Types of marketing agents available in the system."""
    CAMPAIGN_OPTIMIZER = "campaign_optimizer"
    ATTRIBUTION_ANALYST = "attribution_analyst"
    BUDGET_ALLOCATOR = "budget_allocator"
    AUDIENCE_SEGMENTER = "audience_segmenter"
    CREATIVE_TESTER = "creative_tester"
    CHANNEL_MIXER = "channel_mixer"
    PERFORMANCE_MONITOR = "performance_monitor"
    FORECASTER = "forecaster"


class AgentCapability(str, Enum):
    """Capabilities that marketing agents can possess."""
    DATA_ANALYSIS = "data_analysis"
    STATISTICAL_MODELING = "statistical_modeling"
    CAUSAL_INFERENCE = "causal_inference"
    OPTIMIZATION = "optimization"
    FORECASTING = "forecasting"
    SEGMENTATION = "segmentation"
    ATTRIBUTION = "attribution"
    BUDGET_ALLOCATION = "budget_allocation"
    CREATIVE_ANALYSIS = "creative_analysis"
    CHANNEL_ANALYSIS = "channel_analysis"
    PERFORMANCE_TRACKING = "performance_tracking"
    RECOMMENDATION = "recommendation"


class ModelConfig(BaseModel):
    """Configuration for AI model used by the agent."""
    provider: str = Field(..., description="AI provider (openai, anthropic, etc.)")
    model_name: str = Field(..., description="Specific model name")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    custom_parameters: Dict[str, Any] = Field(default_factory=dict)


class MarketingContext(BaseModel):
    """Context information for marketing operations."""
    industry: Optional[str] = Field(None, description="Industry vertical")
    business_model: Optional[str] = Field(None, description="B2B, B2C, marketplace, etc.")
    target_audience: Optional[str] = Field(None, description="Primary target audience")
    budget_range: Optional[str] = Field(None, description="Budget tier (small, medium, large)")
    channels: List[str] = Field(default_factory=list, description="Marketing channels in use")
    kpis: List[str] = Field(default_factory=list, description="Key performance indicators")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Business constraints")
    objectives: List[str] = Field(default_factory=list, description="Marketing objectives")


class MarketingAgent(BaseModel):
    """
    Represents a marketing agent with its configuration and capabilities.
    
    This is the core model for agents that perform marketing analytics tasks
    within the LiftOS ecosystem.
    """
    
    # Identity
    agent_id: str = Field(..., description="Unique identifier for the agent")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Description of agent's purpose")
    agent_type: MarketingAgentType = Field(..., description="Type of marketing agent")
    version: str = Field(default="1.0.0", description="Agent version")
    
    # Capabilities
    capabilities: List[AgentCapability] = Field(..., description="Agent's capabilities")
    specializations: List[str] = Field(default_factory=list, description="Specific specializations")
    
    # Configuration
    ai_model_config: ModelConfig = Field(..., description="AI model configuration")
    system_prompt: str = Field(..., description="System prompt for the agent")
    tools: List[str] = Field(default_factory=list, description="Available tools/functions")
    
    # Context
    marketing_context: Optional[MarketingContext] = Field(None, description="Marketing context")
    
    # Operational
    is_active: bool = Field(default=True, description="Whether agent is active")
    max_concurrent_tasks: int = Field(default=5, gt=0, description="Max concurrent tasks")
    timeout_seconds: int = Field(default=300, gt=0, description="Task timeout in seconds")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = Field(None, description="User who created the agent")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    
    # Performance tracking
    total_tasks_completed: int = Field(default=0, ge=0)
    average_task_duration: Optional[float] = Field(None, ge=0.0)
    success_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    last_used: Optional[datetime] = Field(None)
    
    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True
    )
        
    def update_performance_metrics(
        self, 
        task_duration: float, 
        success: bool
    ) -> None:
        """Update agent performance metrics after task completion."""
        self.total_tasks_completed += 1
        self.last_used = datetime.utcnow()
        
        # Update average duration
        if self.average_task_duration is None:
            self.average_task_duration = task_duration
        else:
            # Exponential moving average
            alpha = 0.1
            self.average_task_duration = (
                alpha * task_duration + 
                (1 - alpha) * self.average_task_duration
            )
        
        # Update success rate
        if self.success_rate is None:
            self.success_rate = 1.0 if success else 0.0
        else:
            # Exponential moving average
            alpha = 0.1
            current_success = 1.0 if success else 0.0
            self.success_rate = (
                alpha * current_success + 
                (1 - alpha) * self.success_rate
            )
        
        self.updated_at = datetime.utcnow()
    
    def can_handle_capability(self, capability: AgentCapability) -> bool:
        """Check if agent can handle a specific capability."""
        return capability in self.capabilities
    
    def is_suitable_for_context(self, context: MarketingContext) -> bool:
        """Check if agent is suitable for a given marketing context."""
        if not self.marketing_context:
            return True  # Generic agent
        
        # Check industry match
        if (self.marketing_context.industry and 
            context.industry and 
            self.marketing_context.industry != context.industry):
            return False
        
        # Check business model match
        if (self.marketing_context.business_model and 
            context.business_model and 
            self.marketing_context.business_model != context.business_model):
            return False
        
        # Check channel overlap
        if (self.marketing_context.channels and 
            context.channels and 
            not set(self.marketing_context.channels).intersection(set(context.channels))):
            return False
        
        return True