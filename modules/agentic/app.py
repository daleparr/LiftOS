"""
Agentic Microservice - Main FastAPI Application

This module provides agent testing and evaluation capabilities for LiftOS,
adapted from the AgentSIM framework for marketing analytics use cases.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import KSE SDK for universal intelligence substrate
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.kse_sdk.client import LiftKSEClient
from shared.kse_sdk.core import Entity, SearchQuery, SearchResult, ConceptualSpace, KSEConfig
from shared.kse_sdk.core.models import SearchType

# Import Phase 2 Advanced Intelligence Flow
from shared.kse_sdk.intelligence.orchestrator import (
    IntelligenceOrchestrator,
    IntelligenceEvent,
    IntelligenceEventType,
    IntelligencePriority
)
from shared.kse_sdk.intelligence.flow_manager import AdvancedIntelligenceFlowManager

from models import (
    MarketingAgent, AgentEvaluationResult, MarketingTestCase,
    TestResult, TestScenario, MarketingAgentType, EvaluationCategory
)
from core.agent_manager import AgentManager
from core.evaluation_engine import EvaluationEngine
from core.test_orchestrator import TestOrchestrator
from core.data_quality_engine import DataQualityEngine
from .services.memory_service import MemoryService
from .services.auth_service import AuthService
from .utils.config import AgenticConfig
from .utils.logging_config import setup_logging

# Import integration components
from .integrations.unified_integration import get_integration_manager, UnifiedAgentIntegrationManager
from .integrations.causal_integration import initialize_causal_integration
from .integrations.observability_integration import initialize_observability_integration
from .integrations.memory_integration import initialize_memory_integration
from .integrations.mmm_integration import initialize_mmm_integration

# Setup logging
logger = setup_logging(__name__)

# Initialize KSE client for universal intelligence substrate
kse_client = LiftKSEClient()

# Initialize KSE integration for agentic intelligence
agentic_kse = None

# Global services
agent_manager: Optional[AgentManager] = None
evaluation_engine: Optional[EvaluationEngine] = None
test_orchestrator: Optional[TestOrchestrator] = None
data_quality_engine: Optional[DataQualityEngine] = None
memory_service: Optional[MemoryService] = None
auth_service: Optional[AuthService] = None
integration_manager: Optional[UnifiedAgentIntegrationManager] = None


# KSE Integration for Agentic Intelligence
class AgenticKSEIntegration:
    """KSE integration for agent testing and evaluation intelligence with Phase 2 Advanced Intelligence Flow"""
    
    def __init__(self, kse_client: LiftKSEClient):
        self.kse_client = kse_client
        self.logger = logger
        
        # Phase 2: Advanced Intelligence Flow Components
        self.intelligence_orchestrator = None
        self.flow_manager = None
    
    async def initialize(self):
        """Initialize KSE client connection and Phase 2 advanced intelligence flow"""
        try:
            await self.kse_client.initialize()
            
            # Initialize Phase 2 Advanced Intelligence Flow
            self.intelligence_orchestrator = IntelligenceOrchestrator(self.kse_client)
            self.flow_manager = AdvancedIntelligenceFlowManager(self.kse_client)
            
            await self.intelligence_orchestrator.initialize()
            await self.flow_manager.initialize()
            
            # Register for cross-service intelligence events
            await self._setup_intelligence_flows()
            
            self.logger.info("Agentic KSE integration with Phase 2 Advanced Intelligence Flow initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Agentic KSE integration: {str(e)}")
            raise
    
    async def _setup_intelligence_flows(self):
        """Setup Phase 2 cross-service intelligence flows for agentic capabilities"""
        try:
            # Subscribe to intelligence events from other services
            await self.intelligence_orchestrator.subscribe_to_event(
                IntelligenceEventType.PATTERN_DISCOVERY,
                self._handle_pattern_discovery
            )
            
            await self.intelligence_orchestrator.subscribe_to_event(
                IntelligenceEventType.INSIGHT_GENERATION,
                self._handle_insight_generation
            )
            
            # Register service capabilities for intelligent routing
            await self.flow_manager.register_service_capabilities(
                service_name="agentic",
                capabilities={
                    "agent_orchestration": 0.95,
                    "decision_making": 0.9,
                    "workflow_automation": 0.85,
                    "intelligent_routing": 0.8,
                    "adaptive_behavior": 0.9
                },
                input_types=["agent_request", "workflow_definition", "decision_context"],
                output_types=["agent_actions", "workflow_results", "decisions"]
            )
            
            self.logger.info("Phase 2 intelligence flows setup completed for Agentic service")
            
        except Exception as e:
            self.logger.error(f"Failed to setup intelligence flows: {str(e)}")
    
    async def _handle_pattern_discovery(self, event: IntelligenceEvent):
        """Handle pattern discovery events from other services"""
        try:
            if event.data.get("service") != "agentic":
                # Process patterns from other services for agent optimization
                pattern_type = event.data.get("pattern_type")
                if pattern_type in ["optimization", "causal", "content", "behavioral"]:
                    # Use patterns to enhance agent decision-making
                    await self._enhance_agent_intelligence(event.data)
                    
        except Exception as e:
            self.logger.error(f"Failed to handle pattern discovery event: {str(e)}")
    
    async def _handle_insight_generation(self, event: IntelligenceEvent):
        """Handle insight generation events from other services"""
        try:
            insight_type = event.data.get("insight_type")
            if insight_type in ["causal_insights", "treatment_recommendations", "content_relevance_improvement"]:
                # Use insights to improve agent workflows and decision-making
                await self._improve_agent_workflows(event.data)
                
        except Exception as e:
            self.logger.error(f"Failed to handle insight generation event: {str(e)}")
    
    async def _enhance_agent_intelligence(self, pattern_data: Dict[str, Any]):
        """Enhance agent intelligence with patterns from other services"""
        try:
            # Create agent enhancement strategy based on external patterns
            enhancement_strategy = {
                "pattern_source": pattern_data.get("service"),
                "pattern_type": pattern_data.get("pattern_type"),
                "enhancement_focus": "agent_intelligence",
                "confidence": pattern_data.get("confidence", 0.0),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Store enhancement strategy for agent optimization
            entity = Entity(
                id=f"agent_enhancement_{uuid.uuid4()}",
                type="agent_enhancement_strategy",
                content=enhancement_strategy,
                metadata={
                    "source_service": pattern_data.get("service"),
                    "pattern_type": pattern_data.get("pattern_type"),
                    "entity_type": "agent_enhancement_strategy"
                }
            )
            
            await self.kse_client.store_entity("global", entity)
            self.logger.info(f"Enhanced agent intelligence from {pattern_data.get('service')} patterns")
            
        except Exception as e:
            self.logger.error(f"Failed to enhance agent intelligence: {str(e)}")
    
    async def _improve_agent_workflows(self, insight_data: Dict[str, Any]):
        """Improve agent workflows using insights from other services"""
        try:
            # Generate workflow improvement strategy
            workflow_improvement = {
                "insight_source": insight_data.get("service"),
                "insight_type": insight_data.get("insight_type"),
                "improvement_focus": "agent_workflows",
                "confidence": insight_data.get("confidence", 0.0),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Publish improved workflow strategy
            improvement_event = IntelligenceEvent(
                event_type=IntelligenceEventType.INSIGHT_GENERATION,
                source_service="agentic",
                target_service="all",
                priority=IntelligencePriority.HIGH,
                data={
                    "insight_type": "workflow_optimization",
                    "improvement_strategy": workflow_improvement,
                    "original_insight": insight_data.get("insight_type"),
                    "confidence": min(insight_data.get("confidence", 0.0) * 0.95, 1.0),  # High confidence for agentic improvements
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            await self.intelligence_orchestrator.publish_event(improvement_event)
            self.logger.info(f"Improved agent workflows with {insight_data.get('insight_type')} insight")
            
        except Exception as e:
            self.logger.error(f"Failed to improve agent workflows: {str(e)}")
    
    async def retrieve_agent_patterns(self, agent_type: str, org_id: str = "system") -> Dict[str, Any]:
        """Retrieve successful agent patterns and configurations from KSE"""
        try:
            # Search for successful agent patterns
            search_results = await self.kse_client.hybrid_search(
                org_id=org_id,
                query=f"agent patterns {agent_type}",
                search_type="hybrid",
                limit=10,
                filters={
                    "entity_type": "agent_pattern",
                    "agent_type": agent_type,
                    "success_rate": {"$gte": 0.8}
                }
            )
            
            # Process agent patterns
            agent_patterns = {
                "successful_configurations": [],
                "performance_benchmarks": {},
                "optimization_strategies": [],
                "best_practices": []
            }
            
            for result in search_results:
                if result.metadata.get("entity_type") == "agent_pattern":
                    agent_patterns["successful_configurations"].append({
                        "pattern_id": result.id,
                        "configuration": result.content.get("configuration"),
                        "success_rate": result.metadata.get("success_rate"),
                        "performance_metrics": result.content.get("performance_metrics"),
                        "confidence_score": result.score
                    })
            
            self.logger.info(f"Retrieved {len(search_results)} agent patterns for {agent_type}")
            return agent_patterns
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve agent patterns: {str(e)}")
            return {"error": str(e)}
    
    async def enrich_evaluation_results(self, evaluation_results: List[Dict], org_id: str) -> None:
        """Enrich KSE with agent evaluation outcomes and learnings"""
        try:
            for evaluation in evaluation_results:
                entity = Entity(
                    id=f"agent_eval_{evaluation.get('agent_id', 'unknown')}_{evaluation.get('test_id', uuid.uuid4())}",
                    type="agent_evaluation",
                    content=evaluation,
                    metadata={
                        "org_id": org_id,
                        "agent_type": evaluation.get("agent_type", "unknown"),
                        "success_rate": evaluation.get("overall_score", 0.0),
                        "evaluation_categories": evaluation.get("category_scores", {}),
                        "test_scenario": evaluation.get("test_scenario", "unknown"),
                        "evaluation_timestamp": datetime.utcnow().isoformat(),
                        "entity_type": "agent_evaluation"
                    }
                )
                
                await self.kse_client.store_entity(org_id, entity)
            
            self.logger.info(f"Enriched KSE with {len(evaluation_results)} agent evaluation results")
            
        except Exception as e:
            self.logger.error(f"Failed to enrich evaluation results: {str(e)}")
    
    async def retrieve_test_insights(self, test_type: str, org_id: str) -> Dict[str, Any]:
        """Retrieve test insights and patterns from other services via KSE"""
        try:
            search_results = await self.kse_client.hybrid_search(
                org_id=org_id,
                query=f"test insights {test_type}",
                search_type="hybrid",
                limit=8,
                filters={
                    "entity_type": "test_insight",
                    "test_type": test_type
                }
            )
            
            test_insights = {
                "test_patterns": [],
                "success_factors": [],
                "failure_modes": [],
                "optimization_opportunities": []
            }
            
            for result in search_results:
                content = result.content
                if content.get("test_patterns"):
                    test_insights["test_patterns"].extend(content["test_patterns"])
                if content.get("success_factors"):
                    test_insights["success_factors"].extend(content["success_factors"])
            
            return test_insights
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve test insights: {str(e)}")
            return {}
    
    async def enrich_agent_performance(self, performance_data: Dict[str, Any], org_id: str) -> None:
        """Enrich KSE with agent performance data and optimization insights"""
        try:
            entity = Entity(
                id=f"agent_performance_{performance_data.get('agent_id', uuid.uuid4())}",
                type="agent_performance",
                content=performance_data,
                metadata={
                    "org_id": org_id,
                    "agent_type": performance_data.get("agent_type", "unknown"),
                    "performance_score": performance_data.get("performance_score", 0.0),
                    "optimization_level": performance_data.get("optimization_level", "baseline"),
                    "performance_timestamp": datetime.utcnow().isoformat(),
                    "entity_type": "agent_performance"
                }
            )
            
            await self.kse_client.store_entity(org_id, entity)
            self.logger.info(f"Enriched KSE with agent performance data for org {org_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to enrich agent performance: {str(e)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global agent_manager, evaluation_engine, test_orchestrator, data_quality_engine, memory_service, auth_service, integration_manager, agentic_kse
    
    logger.info("Starting Agentic microservice...")
    
    try:
        # Initialize KSE integration
        agentic_kse = AgenticKSEIntegration(kse_client)
        await agentic_kse.initialize()
        logger.info("Agentic KSE integration initialized successfully")
        
        # Load configuration
        config = AgenticConfig()
        
        # Initialize services
        memory_service = MemoryService(config.memory_service_url)
        auth_service = AuthService(config.auth_service_url)
        
        # Initialize core components
        agent_manager = AgentManager(memory_service, config)
        evaluation_engine = EvaluationEngine(memory_service, config)
        data_quality_engine = DataQualityEngine(config)
        test_orchestrator = TestOrchestrator(
            agent_manager, evaluation_engine, memory_service, config
        )
        
        # Load default agents and test cases
        await agent_manager.load_default_agents()
        await test_orchestrator.load_default_test_cases()
        
        # Initialize LiftOS Core integrations
        logger.info("Initializing LiftOS Core integrations...")
        await initialize_causal_integration()
        await initialize_observability_integration()
        await initialize_memory_integration()
        await initialize_mmm_integration()
        
        # Initialize unified integration manager
        integration_manager = await get_integration_manager()
        
        logger.info("Agentic microservice started successfully with all integrations")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start Agentic microservice: {e}")
        raise
    finally:
        logger.info("Shutting down Agentic microservice...")
        
        # Cleanup resources
        if test_orchestrator:
            await test_orchestrator.cleanup()
        if evaluation_engine:
            await evaluation_engine.cleanup()
        if agent_manager:
            await agent_manager.cleanup()


# Create FastAPI app
app = FastAPI(
    title="LiftOS Agentic Microservice",
    description="Agent testing and evaluation service for marketing analytics",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency injection
async def get_agent_manager() -> AgentManager:
    """Get agent manager instance."""
    if agent_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent manager not initialized"
        )
    return agent_manager


async def get_evaluation_engine() -> EvaluationEngine:
    """Get evaluation engine instance."""
    if evaluation_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Evaluation engine not initialized"
        )
    return evaluation_engine


async def get_integration_manager() -> UnifiedAgentIntegrationManager:
    """Get unified integration manager instance."""
    if integration_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Integration manager not initialized"
        )
    return integration_manager


async def get_test_orchestrator() -> TestOrchestrator:
    """Get test orchestrator instance."""
    if test_orchestrator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Test orchestrator not initialized"
        )
    return test_orchestrator


async def get_auth_service() -> AuthService:
    """Get auth service instance."""
    if auth_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Auth service not initialized"
        )
    return auth_service


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "agentic",
        "version": "1.0.0",
        "components": {
            "agent_manager": agent_manager is not None,
            "evaluation_engine": evaluation_engine is not None,
            "test_orchestrator": test_orchestrator is not None,
            "memory_service": memory_service is not None,
            "auth_service": auth_service is not None,
        }
    }


# Agent Management Endpoints
@app.get("/agents", response_model=List[MarketingAgent])
async def list_agents(
    agent_type: Optional[MarketingAgentType] = None,
    active_only: bool = True,
    manager: AgentManager = Depends(get_agent_manager)
):
    """List available marketing agents."""
    try:
        agents = await manager.list_agents(agent_type=agent_type, active_only=active_only)
        return agents
    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list agents: {str(e)}"
        )


@app.get("/agents/{agent_id}", response_model=MarketingAgent)
async def get_agent(
    agent_id: str,
    manager: AgentManager = Depends(get_agent_manager)
):
    """Get specific agent by ID."""
    try:
        agent = await manager.get_agent(agent_id)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found"
            )
        return agent
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent {agent_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agent: {str(e)}"
        )


@app.post("/agents", response_model=MarketingAgent)
async def create_agent(
    agent: MarketingAgent,
    manager: AgentManager = Depends(get_agent_manager),
    auth: AuthService = Depends(get_auth_service)
):
    """Create a new marketing agent."""
    try:
        # TODO: Add authentication check
        created_agent = await manager.create_agent(agent)
        return created_agent
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create agent: {str(e)}"
        )


@app.put("/agents/{agent_id}", response_model=MarketingAgent)
async def update_agent(
    agent_id: str,
    agent_updates: Dict[str, Any],
    manager: AgentManager = Depends(get_agent_manager),
    auth: AuthService = Depends(get_auth_service)
):
    """Update an existing agent."""
    try:
        # TODO: Add authentication check
        updated_agent = await manager.update_agent(agent_id, agent_updates)
        if not updated_agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found"
            )
        return updated_agent
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating agent {agent_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update agent: {str(e)}"
        )


@app.delete("/agents/{agent_id}")
async def delete_agent(
    agent_id: str,
    manager: AgentManager = Depends(get_agent_manager),
    auth: AuthService = Depends(get_auth_service)
):
    """Delete an agent."""
    try:
        # TODO: Add authentication check
        success = await manager.delete_agent(agent_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found"
            )
        return {"message": f"Agent {agent_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting agent {agent_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete agent: {str(e)}"
        )


# Evaluation Endpoints
@app.post("/evaluate/{agent_id}", response_model=AgentEvaluationResult)
async def evaluate_agent(
    agent_id: str,
    test_case_id: Optional[str] = None,
    evaluation_type: str = "comprehensive",
    background_tasks: BackgroundTasks = BackgroundTasks(),
    engine: EvaluationEngine = Depends(get_evaluation_engine),
    manager: AgentManager = Depends(get_agent_manager)
):
    """Evaluate an agent's performance."""
    try:
        # Get agent
        agent = await manager.get_agent(agent_id)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found"
            )
        
        # Start evaluation
        evaluation_result = await engine.evaluate_agent(
            agent, test_case_id, evaluation_type
        )
        
        return evaluation_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error evaluating agent {agent_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to evaluate agent: {str(e)}"
        )


@app.get("/evaluations/{evaluation_id}", response_model=AgentEvaluationResult)
async def get_evaluation_result(
    evaluation_id: str,
    engine: EvaluationEngine = Depends(get_evaluation_engine)
):
    """Get evaluation result by ID."""
    try:
        result = await engine.get_evaluation_result(evaluation_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evaluation {evaluation_id} not found"
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting evaluation {evaluation_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get evaluation: {str(e)}"
        )


# Test Management Endpoints
@app.get("/test-cases", response_model=List[MarketingTestCase])
async def list_test_cases(
    category: Optional[str] = None,
    priority: Optional[str] = None,
    orchestrator: TestOrchestrator = Depends(get_test_orchestrator)
):
    """List available test cases."""
    try:
        test_cases = await orchestrator.list_test_cases(
            category=category, priority=priority
        )
        return test_cases
    except Exception as e:
        logger.error(f"Error listing test cases: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list test cases: {str(e)}"
        )


@app.post("/test-cases/{test_case_id}/run", response_model=TestResult)
async def run_test_case(
    test_case_id: str,
    agent_id: str,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    orchestrator: TestOrchestrator = Depends(get_test_orchestrator)
):
    """Run a specific test case against an agent."""
    try:
        result = await orchestrator.run_test_case(test_case_id, agent_id)
        return result
    except Exception as e:
        logger.error(f"Error running test case {test_case_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run test case: {str(e)}"
        )


# Scenario Management Endpoints
@app.get("/scenarios", response_model=List[TestScenario])
async def list_scenarios(
    scenario_type: Optional[str] = None,
    orchestrator: TestOrchestrator = Depends(get_test_orchestrator)
):
    """List available test scenarios."""
    try:
        scenarios = await orchestrator.list_scenarios(scenario_type=scenario_type)
        return scenarios
    except Exception as e:
        logger.error(f"Error listing scenarios: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list scenarios: {str(e)}"
        )


# Analytics Endpoints
@app.get("/analytics/agent-performance")
async def get_agent_performance_analytics(
    agent_id: Optional[str] = None,
    time_range: str = "7d",
    engine: EvaluationEngine = Depends(get_evaluation_engine)
):
    """Get agent performance analytics."""
    try:
        analytics = await engine.get_performance_analytics(agent_id, time_range)
        return analytics
    except Exception as e:
        logger.error(f"Error getting performance analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get analytics: {str(e)}"
        )


@app.get("/analytics/evaluation-trends")
async def get_evaluation_trends(
    category: Optional[EvaluationCategory] = None,
    time_range: str = "30d",
    engine: EvaluationEngine = Depends(get_evaluation_engine)
):
    """Get evaluation trends over time."""
    try:
        trends = await engine.get_evaluation_trends(category, time_range)
        return trends
    except Exception as e:
        logger.error(f"Error getting evaluation trends: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get trends: {str(e)}"
        )


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


# Data Quality Management Endpoints

class DataQualityRequest(BaseModel):
    """Request model for data quality evaluation."""
    data: Any = Field(..., description="Data to evaluate (dict, list of dicts, or DataFrame-compatible)")
    dataset_id: str = Field(..., description="Unique identifier for the dataset")
    test_type: Optional[str] = Field("general", description="Type of test (affects validation rules)")
    include_profiling: bool = Field(True, description="Whether to include data profiling")


class DataQualityValidationRequest(BaseModel):
    """Request model for data quality validation for agent testing."""
    data: Any = Field(..., description="Data to validate")
    test_type: str = Field("marketing_campaign", description="Type of agent test")


def get_data_quality_engine() -> DataQualityEngine:
    """Dependency to get data quality engine instance."""
    if data_quality_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Data quality engine not initialized"
        )
    return data_quality_engine


@app.post("/data-quality/evaluate", response_model=Dict[str, Any])
async def evaluate_data_quality(
    request: DataQualityRequest,
    engine: DataQualityEngine = Depends(get_data_quality_engine)
):
    """
    Evaluate data quality across multiple dimensions.
    
    This endpoint performs comprehensive data quality assessment including:
    - Completeness analysis
    - Accuracy validation  
    - Consistency checks
    - Validity verification
    - Uniqueness assessment
    - Timeliness evaluation
    - Relevance analysis
    - Integrity validation
    """
    try:
        logger.info(f"Starting data quality evaluation for dataset: {request.dataset_id}")
        
        report = await engine.evaluate_data_quality(
            data=request.data,
            dataset_id=request.dataset_id,
            include_profiling=request.include_profiling
        )
        
        # Convert report to dict for JSON response
        result = {
            "dataset_id": report.dataset_id,
            "evaluation_timestamp": report.evaluation_timestamp.isoformat(),
            "overall_score": report.overall_score,
            "overall_level": report.overall_level.value,
            "dimension_scores": {
                dimension.value: {
                    "score": metric.score,
                    "level": metric.level.value,
                    "description": metric.description,
                    "issues_found": metric.issues_found,
                    "recommendations": metric.recommendations,
                    "metadata": metric.metadata
                }
                for dimension, metric in report.dimension_scores.items()
            },
            "critical_issues": report.critical_issues,
            "recommendations": report.recommendations,
            "data_profile": report.data_profile
        }
        
        logger.info(f"Data quality evaluation completed. Score: {report.overall_score:.3f}")
        return result
        
    except Exception as e:
        logger.error(f"Error evaluating data quality: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to evaluate data quality: {str(e)}"
        )


@app.post("/data-quality/validate-for-testing", response_model=Dict[str, Any])
async def validate_data_for_agent_testing(
    request: DataQualityValidationRequest,
    engine: DataQualityEngine = Depends(get_data_quality_engine)
):
    """
    Validate data quality specifically for agent testing scenarios.
    
    This endpoint determines if data meets the quality standards required
    for reliable agent testing and evaluation.
    """
    try:
        logger.info(f"Validating data quality for agent testing: {request.test_type}")
        
        is_valid, report = await engine.validate_data_for_agent_testing(
            data=request.data,
            test_type=request.test_type
        )
        
        result = {
            "is_valid": is_valid,
            "validation_result": "PASS" if is_valid else "FAIL",
            "overall_score": report.overall_score,
            "overall_level": report.overall_level.value,
            "critical_issues": report.critical_issues,
            "recommendations": report.recommendations,
            "quality_summary": {
                dimension.value: {
                    "score": metric.score,
                    "level": metric.level.value
                }
                for dimension, metric in report.dimension_scores.items()
            }
        }
        
        if is_valid:
            logger.info("Data quality validation PASSED - suitable for agent testing")
        else:
            logger.warning("Data quality validation FAILED - not suitable for agent testing")
        
        return result
        
    except Exception as e:
        logger.error(f"Error validating data quality: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate data quality: {str(e)}"
        )


@app.get("/data-quality/dimensions", response_model=Dict[str, Any])
async def get_quality_dimensions():
    """
    Get information about data quality dimensions and their weights.
    
    Returns details about the 8 quality dimensions used in evaluation.
    """
    try:
        dimensions_info = {
            "dimensions": {
                "completeness": {
                    "weight": 0.20,
                    "description": "Measures the extent to which data is present and not missing",
                    "importance": "Critical for avoiding biased evaluations"
                },
                "accuracy": {
                    "weight": 0.25,
                    "description": "Measures how well data represents real-world values",
                    "importance": "Highest priority - directly impacts agent testing results"
                },
                "consistency": {
                    "weight": 0.15,
                    "description": "Measures uniformity of data format and representation",
                    "importance": "Prevents agent processing errors"
                },
                "validity": {
                    "weight": 0.15,
                    "description": "Measures conformance to defined formats and business rules",
                    "importance": "Ensures data meets expected patterns"
                },
                "uniqueness": {
                    "weight": 0.10,
                    "description": "Measures absence of duplicate records",
                    "importance": "Prevents skewed performance metrics"
                },
                "timeliness": {
                    "weight": 0.05,
                    "description": "Measures how current and up-to-date the data is",
                    "importance": "Ensures relevance to current conditions"
                },
                "relevance": {
                    "weight": 0.05,
                    "description": "Measures applicability of data to the intended use case",
                    "importance": "Reduces noise in agent evaluations"
                },
                "integrity": {
                    "weight": 0.05,
                    "description": "Measures referential and structural data integrity",
                    "importance": "Prevents processing failures"
                }
            },
            "quality_levels": {
                "excellent": {"range": "95-100%", "description": "Production-ready data"},
                "good": {"range": "85-94%", "description": "High-quality with minor issues"},
                "acceptable": {"range": "70-84%", "description": "Moderate quality, use with caution"},
                "poor": {"range": "50-69%", "description": "Significant issues, remediation required"},
                "critical": {"range": "<50%", "description": "Severe problems, immediate action required"}
            },
            "validation_thresholds": {
                "agent_testing_minimum": 0.85,
                "critical_issues_allowed": 0,
                "key_dimensions_minimum": 0.80
            }
        }
        
        return dimensions_info
        
    except Exception as e:
        logger.error(f"Error getting quality dimensions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get quality dimensions: {str(e)}"
        )


@app.get("/data-quality/health", response_model=Dict[str, Any])
async def data_quality_health_check():
    """Health check for data quality engine."""
    try:
        engine = get_data_quality_engine()
        
        # Test with sample data
        sample_data = {"test_column": [1, 2, 3, 4, 5]}
        test_report = await engine.evaluate_data_quality(
            data=sample_data,
            dataset_id="health_check_test",
            include_profiling=False
        )
        
        return {
            "status": "healthy",
            "engine_initialized": True,
            "test_evaluation_successful": True,
            "sample_score": test_report.overall_score,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Data quality health check failed: {e}")
        return {
            "status": "unhealthy",
            "engine_initialized": data_quality_engine is not None,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


# ============================================================================
# LiftOS Core Integration Endpoints
# ============================================================================

@app.post("/integrations/comprehensive-test", response_model=Dict[str, Any])
async def run_comprehensive_agent_test(
    request: Dict[str, Any],
    integration_mgr: UnifiedAgentIntegrationManager = Depends(get_integration_manager)
):
    """
    Run a comprehensive agent test using all LiftOS Core integrations.
    
    This endpoint orchestrates testing across:
    - Data quality assessment
    - Causal relationship validation
    - Performance monitoring
    - Memory insights
    - MMM analysis (for marketing scenarios)
    """
    try:
        agent_id = request.get("agent_id")
        test_scenario = request.get("test_scenario", {})
        test_data = request.get("test_data", {})
        include_mmm = request.get("include_mmm", False)
        
        if not agent_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="agent_id is required"
            )
        
        logger.info(f"Starting comprehensive test for agent {agent_id}")
        
        result = await integration_mgr.run_comprehensive_agent_test(
            agent_id=agent_id,
            test_scenario=test_scenario,
            test_data=test_data,
            include_mmm=include_mmm
        )
        
        # Convert result to dict for JSON response
        response = {
            "test_id": result.test_id,
            "agent_id": result.agent_id,
            "test_type": result.test_type,
            "timestamp": result.timestamp.isoformat(),
            "success": result.success,
            "overall_confidence": result.overall_confidence,
            "data_quality": {
                "score": result.data_quality_score,
                "issues": result.data_quality_issues
            },
            "causal_validation": {
                "score": result.causal_validity_score,
                "relationships": result.causal_relationships,
                "confounders": result.confounders_identified
            },
            "performance_metrics": result.performance_metrics,
            "memory_insights": {
                "relevant_experiences": len(result.relevant_experiences),
                "learning_patterns": len(result.learning_patterns)
            },
            "recommendations": result.recommendations
        }
        
        if result.mmm_predictions:
            response["mmm_analysis"] = {
                "predictions": result.mmm_predictions,
                "marketing_performance": result.marketing_performance
            }
        
        logger.info(f"Comprehensive test completed for agent {agent_id}: {result.overall_confidence:.2f} confidence")
        return response
        
    except Exception as e:
        logger.error(f"Error in comprehensive agent test: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run comprehensive test: {str(e)}"
        )


@app.get("/integrations/agent-profile/{agent_id}", response_model=Dict[str, Any])
async def get_agent_profile(
    agent_id: str,
    integration_mgr: UnifiedAgentIntegrationManager = Depends(get_integration_manager)
):
    """Get comprehensive profile for an agent including all integration insights."""
    try:
        profile = await integration_mgr.get_agent_profile(agent_id)
        return profile
        
    except Exception as e:
        logger.error(f"Error getting agent profile for {agent_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agent profile: {str(e)}"
        )


@app.get("/integrations/health", response_model=Dict[str, Any])
async def integration_health_check():
    """Health check for all LiftOS Core integrations."""
    try:
        integration_mgr = await get_integration_manager()
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "integrations": {
                "causal_engine": integration_mgr.causal_engine is not None,
                "observability_manager": integration_mgr.observability_manager is not None,
                "memory_manager": integration_mgr.memory_manager is not None,
                "mmm_engine": integration_mgr.mmm_engine is not None
            }
        }
        
        # Check if all integrations are initialized
        all_healthy = all(health_status["integrations"].values())
        if not all_healthy:
            health_status["status"] = "degraded"
            health_status["message"] = "Some integrations are not initialized"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Integration health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@app.post("/integrations/memory/store-experience", response_model=Dict[str, Any])
async def store_agent_experience(
    request: Dict[str, Any],
    integration_mgr: UnifiedAgentIntegrationManager = Depends(get_integration_manager)
):
    """Store an agent experience in memory for future learning."""
    try:
        agent_id = request.get("agent_id")
        experience_type = request.get("experience_type")
        experience_data = request.get("experience_data")
        context = request.get("context")
        importance_score = request.get("importance_score", 0.5)
        
        if not all([agent_id, experience_type, experience_data]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="agent_id, experience_type, and experience_data are required"
            )
        
        memory_id = await integration_mgr.memory_manager.store_agent_experience(
            agent_id=agent_id,
            experience_type=experience_type,
            experience_data=experience_data,
            context=context,
            importance_score=importance_score
        )
        
        return {
            "memory_id": memory_id,
            "agent_id": agent_id,
            "stored_at": datetime.utcnow().isoformat(),
            "message": "Experience stored successfully"
        }
        
    except Exception as e:
        logger.error(f"Error storing agent experience: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store experience: {str(e)}"
        )


@app.get("/integrations/memory/experiences/{agent_id}", response_model=Dict[str, Any])
async def get_agent_experiences(
    agent_id: str,
    query: str = "",
    limit: int = 10,
    integration_mgr: UnifiedAgentIntegrationManager = Depends(get_integration_manager)
):
    """Retrieve relevant experiences for an agent."""
    try:
        experiences = await integration_mgr.memory_manager.retrieve_relevant_experiences(
            agent_id=agent_id,
            query=query,
            limit=limit
        )
        
        return {
            "agent_id": agent_id,
            "query": query,
            "experiences": experiences,
            "count": len(experiences),
            "retrieved_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error retrieving agent experiences: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve experiences: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )